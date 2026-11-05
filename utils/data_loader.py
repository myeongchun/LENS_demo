from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def load_npz_data(path: str | Path, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        X = MinMaxScaler().fit_transform(data["X"])
        y = np.asarray(data["y"])
    return torch.as_tensor(X, dtype=torch.float32, device=device), y
