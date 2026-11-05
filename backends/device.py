from .cpu import cpu_device
from .gpu import gpu_device

def resolve_device(requested: str):
    value = str(requested).strip().lower()
    if value == "cpu":
        return cpu_device()
    if value == "auto":
        try:
            return gpu_device("0")
        except RuntimeError:
            return cpu_device()
    if value in {"gpu", "cuda"}:
        value = "0"
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    return gpu_device(value)
