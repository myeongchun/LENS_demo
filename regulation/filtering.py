import torch

def leverage_aware_filtering(
    latent_leverage_score: torch.Tensor,
    ratio: float,
    permanently_filtered: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_filtered = torch.zeros_like(permanently_filtered)
    if ratio > 0 and latent_leverage_score.sum() > 1e-12:
        indices = torch.argsort(latent_leverage_score, descending=True)
        cumulative_score = torch.cumsum(latent_leverage_score[indices], dim=0)
        count = int(torch.searchsorted(cumulative_score, ratio).item()) + 1
        current_filtered[indices[:count]] = True
    updated = permanently_filtered | current_filtered
    return ~updated, updated
