from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Config:
    # Paths
    data_root: Path = Path("data_final")  # expects: data_final/train, data_final/val, data_final/test
    out_root: Path = Path("runs")

    # Image
    img_size: int = 64
    channels: int = 3

    # Repro
    seed: int = 42

    # GAN (CGAN)
    z_dim: int = 128
    gan_batch: int = 64
    gan_epochs: int = 100
    gan_lr_g: float = 1e-4       # generator lr
    gan_lr_d: float = 2e-4       # discriminator lr
    sample_every: int = 5        # save sample grid every N epochs
    ckpt_every: int = 20         # save checkpoint every N epochs
    fid_every: int = 5           # compute FID every N epochs
    fid_n_samples: int = 512     # real/fake samples used for FID
    fid_eval_split: str = "val"  # use val split for stable FID

    # Classifier
    clf_batch: int = 64
    clf_epochs: int = 30
    clf_lr: float = 1e-3
    # M4 has 10 cores; cap raised to 8 so data loading saturates the performance cores
    num_workers: int = min(8, max(1, (os.cpu_count() or 4) // 2))
    persistent_workers: bool = True   # keep workers alive across epochs (no respawn cost)
    prefetch_factor: int = 2          # batches queued ahead per worker

    # Apple Silicon (M4/MPS) defaults
    device: str = "mps"  # fallback to cpu in training code if MPS is unavailable
    pin_memory: bool = False  # mainly useful for CUDA, not MPS
