import os
import platform
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Literal


def _default_runtime_profile() -> Literal["default", "apple_silicon_local"]:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "apple_silicon_local"
    return "default"


def _default_num_workers() -> int:
    if sys.platform == "darwin":
        # macOS notebook kernels are still more reliable with worker-free loaders.
        return 0
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
    gan_batch: int = 80
    gan_epochs: int = 200
    gan_lr_g: float = 2e-4
    gan_lr_d: float = 1e-4
    sample_every: int = 25
    ckpt_every: int = 25
    fid_every: int = 25
    fid_n_samples: int = 2048
    fid_enabled: bool = True
    fid_reference_split: str = "train"

    # Classifier
    clf_batch: int = 64
    clf_epochs: int = 50
    clf_lr: float = 3e-4
    num_workers: int = _default_num_workers()
    persistent_workers: bool = False
    prefetch_factor: int = 2
    classifier_compile: bool = False

    # Runtime defaults
    runtime_profile: Literal["default", "apple_silicon_local"] = _default_runtime_profile()
    device: str = "cpu"

    @property
    def fid_eval_split(self) -> str:
        return self.fid_reference_split

    def loader_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            options["persistent_workers"] = self.persistent_workers
            options["prefetch_factor"] = self.prefetch_factor
        return options

    def with_overrides(self, **kwargs: Any) -> "Config":
        return replace(self, **kwargs)

    def with_runtime_profile_defaults(self) -> "Config":
        if self.runtime_profile != "apple_silicon_local":
            return self

        return replace(
            self,
            device="cpu",
            num_workers=0,
            persistent_workers=False,
            classifier_compile=False,
        )
