import os

import torch
from torchvision import transforms

def get_transform(img_size=32):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),      # 32x32
        transforms.ToTensor(),                        # [0,1]
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # [-1,1]
    ])


def get_best_device():
    forced = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if forced == "cpu":
        return torch.device("cpu")
    if forced == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] FORCE_DEVICE=cuda verildi ama CUDA mevcut degil. Auto secime donuluyor.")
    if forced == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[WARN] FORCE_DEVICE=mps verildi ama MPS mevcut degil. Auto secime donuluyor.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_num_workers(max_workers=6):
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return min(max_workers, max(2, cpu_count // 2))


def should_pin_memory(device: torch.device):
    return device.type == "cuda"
