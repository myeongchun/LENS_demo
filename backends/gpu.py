import torch

def gpu_device(index: str = "0") -> torch.device:
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    return device
