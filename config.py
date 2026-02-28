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
    gan_batch: int = 96
    gan_epochs: int = 200
    gan_lr_g: float = 2e-4       # give G more room in the vanilla CGAN loop
    gan_lr_d: float = 1e-4       # avoid an over-dominant discriminator
    sample_every: int = 10       # save sample grid every N epochs
    ckpt_every: int = 25         # save checkpoint every N epochs
    fid_every: int = 10          # compute FID every N epochs
    fid_n_samples: int = 2048    # more stable FID for checkpoint selection
    fid_eval_split: str = "val"  # use val split for stable FID

    # Classifier
    clf_batch: int = 64
    clf_epochs: int = 50
    clf_lr: float = 3e-4
    # On this 10-core M4 Air, 5 workers is usually the sweet spot without wasting CPU on I/O.
    num_workers: int = min(5, max(1, (os.cpu_count() or 4) // 2))
    persistent_workers: bool = True   # keep workers alive across epochs (no respawn cost)
    prefetch_factor: int = 2          # batches queued ahead per worker

    # Apple Silicon (M4/MPS) defaults
    device: str = "mps"  # fallback to cpu in training code if MPS is unavailable
    pin_memory: bool = False  # mainly useful for CUDA, not MPS
