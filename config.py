import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional


def _default_runtime_profile() -> Literal["default", "m4_balanced"]:
    if sys.platform == "darwin" and os.uname().machine == "arm64":
        return "m4_balanced"
    return "default"


def _default_num_workers() -> int:
    if sys.platform == "darwin":
        # macOS notebook kernels and Python 3.13 are still prone to
        # DataLoader worker spawn stalls, especially on Apple Silicon.
        return 0
    cpu_count = os.cpu_count() or 2
    if cpu_count <= 2:
        return 1
    return min(4, cpu_count // 2)


def _cuda_mem_get_info(torch_module: Any, index: int) -> tuple[int, int]:
    try:
        with torch_module.cuda.device(index):
            return torch_module.cuda.mem_get_info()
    except Exception:
        return torch_module.cuda.mem_get_info(index)


def _preferred_cuda_device(torch_module: Any, preference: Literal["first", "most_free"]) -> str:
    if not torch_module.cuda.is_available():
        return "cpu"

    device_count = torch_module.cuda.device_count()
    if device_count <= 1 or preference == "first":
        return "cuda:0"

    best_index = 0
    best_free_bytes = -1
    for index in range(device_count):
        try:
            free_bytes, _ = _cuda_mem_get_info(torch_module, index)
        except Exception:
            continue

        if int(free_bytes) > best_free_bytes:
            best_index = index
            best_free_bytes = int(free_bytes)

    return f"cuda:{best_index}"


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
    gan_lr_g: float = 2e-4       # give G more room in the vanilla CGAN loop
    gan_lr_d: float = 1e-4       # avoid an over-dominant discriminator
    sample_every: int = 25       # save sample grid every N epochs
    ckpt_every: int = 25         # save checkpoint every N epochs
    fid_every: int = 25          # compute FID every N epochs
    fid_n_samples: int = 2048    # more stable FID for checkpoint selection
    fid_enabled: bool = True
    fid_reference_split: str = "train"

    # Classifier
    clf_batch: int = 64
    clf_epochs: int = 50
    clf_lr: float = 3e-4
    # Conservative default that behaves reasonably on laptops and shared hub nodes.
    num_workers: int = _default_num_workers()
    persistent_workers: bool = False
    prefetch_factor: int = 2
    classifier_compile: bool = False

    # Runtime defaults
    runtime_profile: Literal["default", "m4_balanced"] = _default_runtime_profile()
    device: str = "auto"  # prefer CUDA, then MPS, then CPU
    cuda_auto_select: Literal["first", "most_free"] = "most_free"
    cuda_min_free_gib: float = 4.0  # fail early on obviously saturated shared GPUs
    cpu_fallback_when_cuda_busy: bool = False
    pin_memory: Optional[bool] = None  # auto-enable for CUDA unless explicitly overridden
    allow_tf32: bool = True

    @property
    def fid_eval_split(self) -> str:
        return self.fid_reference_split

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
                return _preferred_cuda_device(torch, self.cuda_auto_select)
            if mps_available:
                return "mps"
            return "cpu"

        if requested == "cuda":
            return _preferred_cuda_device(torch, self.cuda_auto_select)
        if requested.startswith("cuda:"):
            if not torch.cuda.is_available():
                return "cpu"
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError:
                return _preferred_cuda_device(torch, self.cuda_auto_select)
            if index < 0 or index >= torch.cuda.device_count():
                return _preferred_cuda_device(torch, self.cuda_auto_select)
            return requested
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

    def with_runtime_profile_defaults(self) -> "Config":
        if self.runtime_profile != "m4_balanced":
            return self

        resolved_device = self.resolve_device()
        overrides: Dict[str, Any] = {
            "gan_batch": min(self.gan_batch, 80),
            "clf_batch": min(self.clf_batch, 64),
            "sample_every": max(self.sample_every, 25),
            "fid_every": max(self.fid_every, 25),
            "persistent_workers": False,
            "classifier_compile": False,
        }
        if resolved_device == "mps":
            overrides.update(
                {
                    "device": "mps",
                    "num_workers": 0,
                    "pin_memory": False,
                }
            )
        return replace(self, **overrides)
