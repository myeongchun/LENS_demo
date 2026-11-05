import torch

def cpu_device() -> torch.device:
    return torch.device("cpu")
