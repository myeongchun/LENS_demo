import torch


def compute_latent_leverage_score(
    mahalanobis_distance: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    leverage = mahalanobis_distance.clamp_min(0.0)
    latent_leverage_score = leverage / leverage.sum().clamp_min(eps)
    return leverage, latent_leverage_score
