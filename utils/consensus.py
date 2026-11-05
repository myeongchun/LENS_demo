import numpy as np


def combine_scores(member_scores: list[np.ndarray], betas: list[float]) -> np.ndarray:
    beta_values = np.clip(np.asarray(betas), 1e-10, 1.0 - 1e-10)
    alpha = np.log(1.0 / beta_values)
    alpha /= alpha.sum() + 1e-10
    return np.stack(member_scores, axis=1) @ alpha
