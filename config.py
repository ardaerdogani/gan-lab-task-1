from dataclasses import dataclass, replace
from pathlib import Path
import os
from typing import Any, Dict, Optional


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 2
    if cpu_count <= 2:
        return 1
    return min(4, cpu_count // 2)


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
    # Conservative default that behaves reasonably on laptops and shared hub nodes.
    num_workers: int = _default_num_workers()
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Runtime defaults
    device: str = "auto"  # prefer CUDA, then MPS, then CPU
    pin_memory: Optional[bool] = None  # auto-enable for CUDA unless explicitly overridden

    def resolve_device(self) -> str:
        requested = self.device.lower()
        try:
            import torch
        except ImportError:
            return "cpu" if requested == "auto" else self.device

        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())

        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if mps_available:
                return "mps"
            return "cpu"

        if requested.startswith("cuda"):
            return self.device if torch.cuda.is_available() else "cpu"
        if requested == "mps":
            return "mps" if mps_available else "cpu"
        return self.device

    def resolve_pin_memory(self) -> bool:
        if self.pin_memory is not None:
            return self.pin_memory
        return self.resolve_device().startswith("cuda")

    def loader_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.resolve_pin_memory(),
        }
        if self.num_workers > 0:
            options["persistent_workers"] = self.persistent_workers
            options["prefetch_factor"] = self.prefetch_factor
        return options

    def with_overrides(self, **kwargs: Any) -> "Config":
        return replace(self, **kwargs)
