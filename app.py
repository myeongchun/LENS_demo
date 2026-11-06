from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MAX_BYTES = 50 * 1024 * 1024
MAX_EXPANDED = 100 * 1024 * 1024
MAX_ROWS = 20000
MAX_FEATURES = 512


def read_csv(raw: bytes, separator: str = ",") -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw), sep=separator, encoding="utf-8-sig",
                        nrows=MAX_ROWS + 1)
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Datasets may contain up to {MAX_ROWS:,} rows.")
    return frame


def read_npz(raw: bytes) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_EXPANDED:
            raise ValueError("The total uncompressed archive size must not exceed 100 MiB.")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        # Never deserialize Python objects from an uploaded archive.
        return {key: archive[key] for key in archive.files}


def validate_features(values) -> np.ndarray:
    X = np.asarray(values)
    if X.ndim != 2 or not 3 <= X.shape[0] <= MAX_ROWS:
        raise ValueError(f"X must be a two-dimensional array with 3 to {MAX_ROWS:,} rows.")
    if not 2 <= X.shape[1] <= MAX_FEATURES:
        raise ValueError(f"Select between 2 and {MAX_FEATURES} numerical features.")
    if X.dtype.kind not in "biuf":
        raise ValueError("Features must contain real-valued numerical data.")
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Features contain missing or infinite values. Please clean the data before running LENS.")
    if not np.any(np.ptp(X, axis=0) > 0):
        raise ValueError("All features are constant. Please provide data with at least one varying feature.")
    return X


def validate_labels(values, count: int) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or len(y) != count:
        raise ValueError("Labels must be a one-dimensional array with one label per row of X.")
    if pd.isna(y).any():
        raise ValueError("Labels contain missing values.")
    if y.dtype.kind in "fc" and not np.isfinite(y).all():
        raise ValueError("Labels contain infinite values.")
    return y.astype(str)


@st.cache_resource
def execution_lock():
    # main.run_lens changes global random seeds; serialize concurrent sessions.
    return threading.Lock()


# ---------------------------------------------------------------------------
# Lossless ensemble compression (Sec. 5.1)
#
# Every member scores x by its reconstruction error
#     e_t(x) = || (I - V_t V_t^T)(x - mu_t) ||^2 / d,
# and LENS combines them as  sum_t alpha_t e_t(x).  A weighted sum of quadratic
# forms is one quadratic form, so the ensemble equals
#     score(x) = (x - x*)^T C (x - x*) / d + gamma,   C = sum_t alpha_t (I - V_t V_t^T).
# The original repository returns only the errors of each member, so the app
# records each member's mean and principal directions while run_lens runs.
# ---------------------------------------------------------------------------

def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def principal_space(X, weights, r: float, eps: float = 1e-6):
    """Mean and principal directions of one latent space, following backbone/PCA.py."""
    X = _as_numpy(X)
    w = _as_numpy(weights)
    w = w / max(w.sum(), eps)
    mean = w @ X
    centered = (X - mean) * np.sqrt(w)[:, None]
    covariance = centered.T @ centered
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values[::-1], 0.0, None)
    vectors = vectors[:, ::-1]
    cumulative = np.cumsum(values) / max(values.sum(), eps)
    k = int(np.searchsorted(cumulative, r, side="left")) + 1
    k = max(1, min(k, X.shape[1]))
    if k == X.shape[1] and X.shape[1] > 1:
        k -= 1
    return mean, np.ascontiguousarray(vectors[:, :k])


def run_and_record(tensor, config, seed):
    """Run the original run_lens and record each kept member's space and beta."""
    import main as lens

    original_pca, original_combine = lens.weighted_pca, lens.combine_scores
    spaces, combined = [], {}

    def recording_pca(X, weights, r, *args, **kwargs):
        result = original_pca(X, weights, r, *args, **kwargs)
        spaces.append(principal_space(X, weights, float(r)))
        return result

    def recording_combine(member_scores, betas):
        combined["count"] = len(member_scores)
        combined["betas"] = [float(b) for b in betas]
        return original_combine(member_scores, betas)

    lens.weighted_pca, lens.combine_scores = recording_pca, recording_combine
    try:
        scores = np.asarray(lens.run_lens(tensor, config, seed)).reshape(-1)
    finally:
        lens.weighted_pca, lens.combine_scores = original_pca, original_combine
    # run_lens keeps the members of rounds 0..T-1; a final rejected round may follow.
    count = combined["count"]
    return scores, spaces[:count], combined["betas"]


def compress_ensemble(spaces, betas, scaled: np.ndarray, scores: np.ndarray) -> dict:
    n, d = scaled.shape
    beta = np.clip(np.asarray(betas, dtype=np.float64), 1e-10, 1.0 - 1e-10)
    alpha = np.log(1.0 / beta)
    alpha /= alpha.sum() + 1e-10                          # utils/consensus.py
    C, b, c = np.zeros((d, d)), np.zeros(d), 0.0
    for weight, (mean, V) in zip(alpha, spaces):
        P = np.eye(d) - V @ V.T
        Pm = P @ mean
        C += weight * P
        b += weight * Pm
        c += weight * float(mean @ Pm)
    C = 0.5 * (C + C.T)
    center = np.linalg.lstsq(C, b, rcond=None)[0]         # most normal point x*
    gamma = (c - float(b @ center)) / d
    weights, directions = np.linalg.eigh(C)
    weights = np.clip(weights[::-1], 0.0, None)
    directions = directions[:, ::-1].copy()
    flip = directions[np.abs(directions).argmax(axis=0), np.arange(d)] < 0
    directions[:, flip] *= -1                              # largest loading positive
    totals, compressed = np.zeros(d), np.empty(n)
    for start in range(0, n, 4096):
        terms = ((scaled[start:start + 4096] - center) @ directions) ** 2 * weights / d
        totals += terms.sum(axis=0)
        compressed[start:start + 4096] = terms.sum(axis=1) + gamma
    gap = float(np.max(np.abs(compressed - scores)) / max(np.max(np.abs(scores)), 1e-12))
    return {"C": C, "center": center, "gamma": gamma, "weights": weights,
            "directions": directions, "share": totals / max(totals.sum(), 1e-300),
            "gap": gap, "count": len(spaces), "dims": [V.shape[1] for _, V in spaces],
            "alpha": alpha.tolist()}


def execute(X, config, seed):
    import torch
    from sklearn.preprocessing import MinMaxScaler

    # Match utils/data_loader.py in the original repository.
    scaler = MinMaxScaler().fit(X)
    scaled = scaler.transform(X)
    started = time.perf_counter()
    with execution_lock():
        tensor = torch.as_tensor(scaled, dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            scores, spaces, betas = run_and_record(tensor, config, seed)
    elapsed = time.perf_counter() - started
    if len(scores) != len(X) or not np.isfinite(scores).all():
        raise ValueError("LENS returned invalid anomaly scores.")
    model = compress_ensemble(spaces, betas, scaled, scores)
    model["scaler"] = scaler
    return scores, model, elapsed


# ---------------------------------------------------------------------------
# Interpretation view
# ---------------------------------------------------------------------------

def dominant_attributes(vector: np.ndarray, names: list[str], limit: int = 3) -> str:
    order = np.argsort(-np.abs(vector))[:limit]
    keep = [i for i in order if abs(vector[i]) >= 0.25] or [order[0]]
    return ", ".join(f"{names[i]} ({vector[i]:+.2f})" for i in keep)


def ranked_bars(labels, values, value_name: str):
    width = len(str(len(labels)))
    frame = pd.DataFrame({"item": [f"{i + 1:0{width}d} · {name}" for i, name in enumerate(labels)],
                          value_name: values})
    st.bar_chart(frame, x="item", y=value_name, horizontal=True, x_label="", y_label=value_name)


def show_interpretation(model: dict, X: np.ndarray, scores: np.ndarray, names: list[str]):
    n, d = X.shape
    weights, directions, share = model["weights"], model["directions"], model["share"]
    st.markdown(
        f"LENS merges its **{model['count']} latent spaces** into one scoring function without loss:\n\n"
        r"$$\mathrm{score}(x) = \frac{1}{d}\sum_j \lambda_j\,\big[u_j^{\top}(x - x^*)\big]^2 + \gamma$$"
        "\n\nEach eigen-direction $u_j$ of the merged matrix $C$ gets a weight $\\lambda_j$ between 0 and 1. "
        "A single PCA can only keep (1) or drop (0) a direction; a weight in between means some latent "
        "spaces kept the direction and others did not. $x^*$ is the point LENS regards as the most normal. "
        "Directions and the center are expressed in the min-max scaled feature space."
    )
    a, b, c = st.columns(3)
    a.metric("Latent spaces merged", model["count"],
             help="Dimensions kept by each space: " + ", ".join(map(str, model["dims"])))
    b.metric("Directions with weight > 0.01", int((weights > 0.01).sum()), help=f"Out of {d} directions.")
    c.metric("Compression check", "Lossless" if model["gap"] < 1e-4 else f"{model['gap']:.1e}",
             help=("Largest difference between the merged score and the ensemble score, relative to the "
                   f"largest score: {model['gap']:.1e}. Small differences come from float32 vs. float64 arithmetic."))

    st.subheader("Where the score comes from")
    shown = st.slider("Directions to show", 1, d, min(10, d), key="interp-top")
    order = np.argsort(-share)[:shown]
    table = pd.DataFrame({
        "Direction": [f"u{j + 1}" for j in order],
        "Weight λ": weights[order],
        "Dominant attributes (loading)": [dominant_attributes(directions[:, j], names) for j in order],
        "Share of score (%)": share[order] * 100,
    })
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        "Weight λ": st.column_config.NumberColumn(format="%.3f"),
        "Share of score (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=100.0),
    })
    st.caption("Directions are numbered by weight (u1 has the largest λ) and listed by their share of the "
               "anomaly score summed over all samples (the constant γ excluded).")
    st.download_button("Download directions (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                       "lens-directions.csv", "text/csv")

    with st.expander("Most normal point x* (original units)"):
        center = model["scaler"].inverse_transform(model["center"][None, :])[0]
        st.dataframe(pd.DataFrame({"Feature": names, "x*": center}), hide_index=True, use_container_width=True)

    st.subheader("Explain one sample")
    ranking = np.argsort(-scores, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[ranking] = np.arange(1, n + 1)
    index = int(st.number_input("Sample index", 0, n - 1, int(ranking[0]), key="interp-sample",
                                help="Row position in your file, starting at 0. Defaults to the highest-scoring sample."))
    delta = model["scaler"].transform(X[index:index + 1])[0] - model["center"]
    terms = weights * (directions.T @ delta) ** 2 / d
    features = delta * (model["C"] @ delta) / d            # sums to score - gamma
    st.write(f"Sample **{index}** · anomaly score **{scores[index]:.4g}** · rank **{rank[index]:,}** of {n:,}")
    left, right = st.columns(2)
    with left:
        st.markdown("**By direction**")
        top = np.argsort(-terms)[:8]
        ranked_bars([f"u{j + 1}" for j in top], terms[top] / max(terms.sum(), 1e-300) * 100, "share of this score (%)")
    with right:
        st.markdown("**By feature**")
        top = np.argsort(-np.abs(features))[:8]
        ranked_bars([names[i] for i in top], features[top] / max(features.sum(), 1e-300) * 100, "share of this score (%)")
        st.caption("Per-feature parts add up to the score (minus γ); a negative part means the feature "
                   "pulls the sample toward x* along correlated directions.")


def main():
    st.set_page_config(page_title="LENS Demo", layout="wide")
    st.title("LENS: Learning an Ensemble of Normality Subspaces for Unsupervised Anomaly Detection")
    st.write("Upload a CSV or NPZ dataset, optionally specify labels, and run LENS.")
    st.caption("CPU execution · Labels are used for evaluation only · Higher scores indicate greater abnormality")
    root = Path(__file__).resolve().parent
    if not (root / "main.py").exists() or not (root / "config.json").exists():
        st.error("Place app.py in the LENS repository root, alongside main.py and config.json.")
        return
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        for key in ("N", "r", "filtering_ratio", "rho"):
            if key not in config:
                raise ValueError(f"Missing configuration key in config.json: {key}.")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    with st.sidebar:
        st.header("Run Settings")
        config["N"] = st.number_input("Ensemble size (N)", 1, 50, int(config["N"]))
        config["r"] = st.slider("Explained variance ratio (r)", 0.01, 0.99, float(config["r"]))
        config["filtering_ratio"] = st.slider("Filtering ratio", 0.0, 0.9,
                                               float(config["filtering_ratio"]))
        config["rho"] = st.slider("Reliability coefficient (rho)", 0.0, 1.0, float(config["rho"]))
        seed = st.number_input("Random seed", 0, 2147483647, 0)
        st.caption("Limits: 50 MiB per file, 20,000 samples, and 512 features. Larger datasets may take longer to process.")
    uploaded = st.file_uploader("Upload Dataset", type=["csv", "npz"])
    if uploaded is None:
        return
    if uploaded.size > MAX_BYTES:
        st.error("The uploaded file must not exceed 50 MiB.")
        return
    raw = uploaded.getvalue()
    file_key = hashlib.sha256(raw).hexdigest()[:16]
    y = None
    feature_names = []
    try:
        if uploaded.name.lower().endswith(".csv"):
            separator = st.selectbox("CSV delimiter", [",", ";", "\t"], key=f"sep-{file_key}")
            frame = read_csv(raw, separator)
            st.dataframe(frame.head(10), hide_index=True)
            columns = list(frame.columns)
            default_label = next((c for c in ("label", "y", "target") if c in columns), None)
            label = st.selectbox("Label column (optional)", [None] + columns,
                                 index=columns.index(default_label) + 1 if default_label else 0,
                                 format_func=lambda v: "No labels" if v is None else str(v),
                                 key=f"label-{file_key}")
            options = [c for c in columns if c != label]
            defaults = [c for c in options if pd.api.types.is_numeric_dtype(frame[c])]
            feature_names = st.multiselect("Numerical features (exclude identifier columns)", options,
                                           default=defaults, key=f"features-{file_key}-{label}")
            X = validate_features(frame[feature_names].to_numpy())
            if label is not None:
                y = validate_labels(frame[label].to_numpy(), len(X))
        else:
            arrays = read_npz(raw)
            keys = list(arrays)
            candidates = [k for k in keys if arrays[k].ndim == 2]
            if not candidates:
                raise ValueError("The NPZ archive does not contain a two-dimensional feature array.")
            xkey = st.selectbox("Feature array", candidates,
                                index=candidates.index("X") if "X" in candidates else 0,
                                key=f"x-{file_key}")
            X = validate_features(arrays[xkey])
            ykeys = [None] + [k for k in keys if k != xkey]
            ykey = st.selectbox("Label array (optional)", ykeys,
                                index=ykeys.index("y") if "y" in ykeys else 0,
                                format_func=lambda v: "No labels" if v is None else v,
                                key=f"y-{file_key}")
            if ykey is not None:
                y = validate_labels(arrays[ykey], len(X))
            feature_names = [f"x{i}" for i in range(X.shape[1])]
            st.dataframe(pd.DataFrame(X[:10], columns=feature_names), hide_index=True)
        anomaly_values = []
        if y is not None:
            unique = list(pd.unique(y))
            st.write("Label Counts", pd.Series(y).value_counts().rename("count"))
            anomaly_values = st.multiselect("Anomaly label values", unique,
                                            default=["1"] if "1" in unique else [],
                                            key=f"anomaly-{file_key}-{hashlib.sha256(y.tobytes()).hexdigest()[:8]}")
            st.caption("Selected values are treated as anomalies (1). All remaining values are treated as normal (0).")
            if not anomaly_values or len(anomaly_values) == len(unique):
                raise ValueError("For evaluation, select at least one anomaly label and leave at least one normal label. To compute scores only, select 'No labels'.")
        st.write(f"{len(X):,} samples · {X.shape[1]} features")
    except Exception as exc:
        st.error(f"Please check your input: {exc}")
        return

    signature = hashlib.sha256(raw + X.tobytes() + json.dumps(
        [config, int(seed), feature_names, anomaly_values,
         y.tolist() if y is not None else None], ensure_ascii=False).encode()).hexdigest()
    if st.button("Run LENS", type="primary"):
        st.session_state.pop("lens_result", None)
        try:
            with st.spinner("Running LENS…"):
                scores, model, elapsed = execute(X, config, int(seed))
            metrics = {}
            result = pd.DataFrame({"sample_index": np.arange(len(X)), "anomaly_score": scores})
            if y is not None:
                from sklearn.metrics import average_precision_score, roc_auc_score
                binary = np.isin(y, anomaly_values).astype(int)
                metrics = {"AUROC": float(roc_auc_score(binary, scores)),
                           "AUPRC (Average Precision)": float(average_precision_score(binary, scores))}
                result["original_label"] = y
                result["is_anomaly"] = binary
            report = {"source_file": uploaded.name, "samples": len(X), "features": feature_names,
                      "config": config, "seed": int(seed), "device": "cpu",
                      "preprocessing": "MinMaxScaler [0, 1]", "anomaly_values": anomaly_values,
                      "metrics": metrics, "elapsed_seconds": elapsed,
                      "compression": {"latent_spaces": model["count"], "dims": model["dims"],
                                      "max_relative_difference": model["gap"]}}
            st.session_state["lens_result"] = (signature, result, report, model)
        except Exception as exc:
            st.error(f"Execution failed: {exc}")
            st.info("Check that the required dependencies and the original LENS source files are available.")
    saved = st.session_state.get("lens_result")
    if saved and saved[0] == signature:
        _, result, report, model = saved
        st.success(f"Completed in {report['elapsed_seconds']:.2f} seconds")
        scores_tab, interpretation_tab = st.tabs(["Scores", "Interpretation"])
        with scores_tab:
            for name, value in report["metrics"].items():
                st.metric(name, f"{value:.4f}")
            if not report["metrics"]:
                st.info("Anomaly scores were computed without labels. Evaluation metrics are unavailable.")
            st.dataframe(result.head(100), hide_index=True)
            st.download_button("Download Anomaly Scores (CSV)", result.to_csv(index=False).encode("utf-8-sig"),
                               "lens-results.csv", "text/csv")
            st.download_button("Download Evaluation Report (JSON)",
                               json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                               "lens-report.json", "application/json")
        with interpretation_tab:
            show_interpretation(model, X, result["anomaly_score"].to_numpy(), list(map(str, feature_names)))


if __name__ == "__main__":
    main()
