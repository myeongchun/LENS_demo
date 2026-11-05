import torch

def update_sample_weights(
    weights: torch.Tensor,
    errors: torch.Tensor,
    reliability: torch.Tensor,
    keep_mask: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, float, bool]:
    kept_errors = errors[keep_mask]
    maximum = (kept_errors.max() if kept_errors.numel() else errors.max()).clamp_min(eps)
    normalized_error = (errors / maximum).clamp(0.0, 1.0)
    weighted_error = (weights * normalized_error).sum().clamp(eps, 1.0 - eps)
    beta = (weighted_error / (1.0 - weighted_error)).sqrt()
    error_factor = (beta + eps).pow(1.0 - normalized_error)

    updated = weights * error_factor * reliability
    updated = updated.masked_fill(~keep_mask, 0.0)
    total = updated.sum()
    collapsed = bool(total <= eps)
    if collapsed:
        uniform = torch.full_like(weights, 1.0 / weights.numel())
        return uniform, float(beta.item()), True
    return updated / total, float(beta.item()), False
