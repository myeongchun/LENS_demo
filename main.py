import argparse
import json
import os
import random
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

torch.set_float32_matmul_precision("high")
torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)

from backends.device import resolve_device
from backbone.PCA import weighted_pca
from regulation.filtering import leverage_aware_filtering
from regulation.reliability import reliability_aware_weighting
from regulation.scoring import compute_latent_leverage_score
from regulation.weighting import update_sample_weights
from utils.consensus import combine_scores
from utils.data_loader import load_npz_data
from utils.metrics import calculate_metrics


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
DATASET_DIR = ROOT / "datasets"


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_lens(X: torch.Tensor, config: dict, seed: int) -> np.ndarray:
    set_seed(seed)
    sample_count = X.shape[0]
    weights = torch.full(
        (sample_count,), 1.0 / sample_count, dtype=X.dtype, device=X.device
    )
    previous_latent_leverage_score = torch.zeros_like(weights)
    permanently_filtered = torch.zeros(
        sample_count, dtype=torch.bool, device=X.device
    )

    member_scores: list[np.ndarray] = []
    betas: list[float] = []
    N = int(config["N"])
    for round_index in range(N):
        errors, mahalanobis_distance = weighted_pca(
            X,
            weights,
            r=float(config["r"]),
        )
        leverage, latent_leverage_score = compute_latent_leverage_score(
            mahalanobis_distance
        )
        keep_mask, permanently_filtered = leverage_aware_filtering(
            latent_leverage_score,
            float(config["filtering_ratio"]),
            permanently_filtered,
        )
        reliability = reliability_aware_weighting(
            latent_leverage_score=latent_leverage_score,
            previous_latent_leverage_score=previous_latent_leverage_score,
            rho=float(config["rho"]),
            first_round=(round_index == 0),
        )
        new_weights, beta, collapsed = update_sample_weights(
            weights=weights,
            errors=errors,
            reliability=reliability,
            keep_mask=keep_mask,
        )

        if beta >= 1.0 and round_index > 0:
            break

        member_scores.append(errors.detach().cpu().numpy())
        betas.append(beta)
        weights = new_weights
        previous_latent_leverage_score = latent_leverage_score

        if beta >= 1.0 or collapsed:
            break


    return combine_scores(member_scores, betas)


def resolve_datasets(dataset: str) -> list[Path]:
    if dataset.lower() == "all":
        return sorted(DATASET_DIR.glob("*.npz"))

    name = dataset if dataset.endswith(".npz") else f"{dataset}.npz"
    return [DATASET_DIR / name]


def main() -> None:
    parser = argparse.ArgumentParser(description="LENS2027 anomaly detector")
    parser.add_argument("--dataset", required=True, help="NPZ name or 'all'")
    parser.add_argument("--device", default="auto", help="auto, cpu, gpu, or GPU index")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    dataset_paths = resolve_datasets(args.dataset)
    for path in dataset_paths:
        X, y = load_npz_data(path, device)
        scores = run_lens(X, config, args.seed)
        metrics = calculate_metrics(y, scores)
        if len(dataset_paths) > 1:
            print(f"Dataset: {path.name} ", end="")
        print(
            f"AUROC: {metrics['auroc'] * 100:.3f}% "
            f"AUPRC: {metrics['auprc'] * 100:.3f}%"
        )


if __name__ == "__main__":
    main()
