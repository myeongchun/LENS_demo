import torch


def reliability_aware_weighting(
    latent_leverage_score: torch.Tensor,
    previous_latent_leverage_score: torch.Tensor,
    rho: float,
    first_round: bool,
    eps: float = 1e-10,
) -> torch.Tensor:
    if first_round:
        pi = torch.zeros_like(latent_leverage_score)
    else:
        denominator_eps = 0.1 / max(latent_leverage_score.numel(), 1)
        reliability_change = (
            (latent_leverage_score - previous_latent_leverage_score)
            / (previous_latent_leverage_score + denominator_eps)
        ).abs().clamp(max=5.0)
        reliability_scale = torch.median(reliability_change).clamp_min(eps)
        pi = reliability_change / reliability_scale

    reliability = torch.exp(-pi / rho)
    return reliability
