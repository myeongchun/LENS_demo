import torch

def weighted_pca(
    X: torch.Tensor,
    weights: torch.Tensor,
    r: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_weights = weights / weights.sum().clamp_min(eps)
    mean = (normalized_weights[:, None] * X).sum(dim=0, keepdim=True)
    centered = X - mean
    weighted_centered = centered * normalized_weights.sqrt().unsqueeze(1)
    covariance = weighted_centered.T @ weighted_centered
    covariance = 0.5 * (covariance + covariance.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
    eigenvectors = eigenvectors.flip(1)
    cumulative = torch.cumsum(eigenvalues, dim=0) / eigenvalues.sum().clamp_min(eps)
    k = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(r, dtype=X.dtype, device=X.device),
        ).item()
    ) + 1
    k = max(1, min(k, X.shape[1]))
    if k == X.shape[1] and X.shape[1] > 1:
        k -= 1

    components = eigenvectors[:, :k]
    latent = centered @ components
    reconstruction = latent @ components.T + mean
    reconstruction_error = torch.mean((X - reconstruction).square(), dim=1)

    latent_variance = eigenvalues[:k].clamp_min(eps)
    mahalanobis_distance = (
        latent.square() / latent_variance
    ).sum(dim=1).clamp_min(0.0)
    return reconstruction_error, mahalanobis_distance
