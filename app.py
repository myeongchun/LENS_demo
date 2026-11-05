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


def execute(X, config, seed):
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from main import run_lens

    # Match utils/data_loader.py in the original repository.
    scaled = MinMaxScaler().fit_transform(X)
    with execution_lock():
        tensor = torch.as_tensor(scaled, dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            scores = np.asarray(run_lens(tensor, config, seed)).reshape(-1)
    if len(scores) != len(X) or not np.isfinite(scores).all():
        raise ValueError("LENS returned invalid anomaly scores.")
    return scores


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
            started = time.perf_counter()
            with st.spinner("Running LENS…"):
                scores = execute(X, config, int(seed))
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
                      "metrics": metrics, "elapsed_seconds": time.perf_counter() - started}
            st.session_state["lens_result"] = (signature, result, report)
        except Exception as exc:
            st.error(f"Execution failed: {exc}")
            st.info("Check that the required dependencies and the original LENS source files are available.")
    saved = st.session_state.get("lens_result")
    if saved and saved[0] == signature:
        _, result, report = saved
        st.success(f"Completed in {report['elapsed_seconds']:.2f} seconds")
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


if __name__ == "__main__":
    main()
