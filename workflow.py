"""Shared runtime helpers for the notebook-first GAN workflow."""

from __future__ import annotations

import gc
import json
import os
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
from scipy import linalg
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms, utils as vutils
from torchvision.models import inception_v3
from torchvision.utils import save_image

from config import Config
from models.classifier import FruitCNN
from models.gan import Generator, ProjectionDiscriminator

FID_FEATURE_DIM = 2048
DEFAULT_TASK1_SIZES = [100, 200, 400, 800, 1300]
DEFAULT_TASK1_SCENARIOS = ["real", "synth", "both", "real_aug"]


def summarize_dataset(data_root: Path | str, splits: Sequence[str] = ("train", "val", "test")) -> Dict[str, Dict[str, int]]:
    data_root = Path(data_root)
    summary: Dict[str, Dict[str, int]] = {}
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        counts: Dict[str, int] = {}
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            counts[class_dir.name] = sum(1 for path in class_dir.iterdir() if path.is_file())
        summary[split] = counts
    return summary


def sample_image_paths(data_root: Path | str, split: str = "train") -> list[tuple[str, Path]]:
    split_root = Path(data_root) / split
    samples: list[tuple[str, Path]] = []
    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        image_paths = sorted(path for path in class_dir.glob("*") if path.is_file())
        if image_paths:
            samples.append((class_dir.name, image_paths[0]))
    return samples


def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def round_seconds(value: float) -> float:
    rounded = round(float(value), 1)
    if value > 0 and rounded == 0.0:
        return 0.1
    return rounded


def _cuda_mem_get_info(index: int) -> tuple[int, int]:
    try:
        with torch.cuda.device(index):
            return torch.cuda.mem_get_info()
    except Exception:
        return torch.cuda.mem_get_info(index)


def get_visible_cuda_devices() -> list[Dict[str, Any]]:
    devices: list[Dict[str, Any]] = []
    if not torch.cuda.is_available():
        return devices

    for index in range(torch.cuda.device_count()):
        record: Dict[str, Any] = {
            "cuda_index": index,
            "cuda_name": torch.cuda.get_device_name(index),
        }
        try:
            free_bytes, total_bytes = _cuda_mem_get_info(index)
        except Exception:
            devices.append(record)
            continue

        record["cuda_free_gib"] = round(int(free_bytes) / (1024 ** 3), 1)
        record["cuda_total_gib"] = round(int(total_bytes) / (1024 ** 3), 1)
        devices.append(record)
    return devices


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _is_runtime_headroom_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "does not have enough free memory to start this stage" in str(exc)


def get_execution_device_candidates(cfg: Config) -> list[str]:
    requested = cfg.device.lower()
    visible_cuda = sorted(
        get_visible_cuda_devices(),
        key=lambda item: float(item.get("cuda_free_gib", -1.0)),
        reverse=True,
    )
    cuda_candidates = [f"cuda:{item['cuda_index']}" for item in visible_cuda]

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested == "auto":
        if cuda_candidates:
            return cuda_candidates
        if mps_available:
            return ["mps"]
        return ["cpu"]

    if requested == "cuda":
        if cuda_candidates:
            return cuda_candidates
        return ["cpu"]

    if requested.startswith("cuda:"):
        return [cfg.resolve_device()]
    if requested == "mps":
        return ["mps" if mps_available else "cpu"]
    return [cfg.resolve_device()]


def run_stage_with_device_fallback(
    cfg: Config,
    stage_label: str,
    stage_fn: Any,
) -> Dict[str, Any]:
    candidates = get_execution_device_candidates(cfg)
    if cfg.cpu_fallback_when_cuda_busy and "cpu" not in candidates:
        candidates.append("cpu")
    last_retryable_error: RuntimeError | None = None

    for attempt_index, candidate in enumerate(candidates, start=1):
        stage_cfg = cfg if candidate == cfg.device else cfg.with_overrides(device=candidate)
        try:
            device, runtime_summary = resolve_execution_device(stage_cfg)
            return stage_fn(stage_cfg, device, runtime_summary)
        except RuntimeError as exc:
            clear_torch_memory()
            next_candidates = candidates[attempt_index:]
            can_retry = candidate.startswith("cuda") and cfg.device.lower() in {"auto", "cuda"} and (
                (_is_cuda_oom(exc) or _is_runtime_headroom_error(exc)) and bool(next_candidates)
            )
            if not can_retry:
                raise

            last_retryable_error = exc
            next_target = next_candidates[0]
            if next_target == "cpu":
                print(
                    f"[{stage_label}] {candidate} unavailable ({exc}). "
                    "Falling back to CPU because all visible CUDA devices are busy."
                )
            else:
                print(
                    f"[{stage_label}] {candidate} unavailable ({exc}). "
                    "Retrying on the next visible CUDA device..."
                )

    if last_retryable_error is not None:
        raise RuntimeError(
            f"[{stage_label}] Exhausted visible CUDA devices after retrying {candidates}. "
            f"Last error: {last_retryable_error}"
        ) from last_retryable_error
    raise RuntimeError(f"[{stage_label}] No execution devices were available.")


def clear_torch_memory(*objects: Any) -> None:
    for obj in objects:
        if isinstance(obj, nn.Module):
            try:
                obj.to("cpu")
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_runtime_headroom(cfg: Config, runtime_summary: Mapping[str, Any]) -> None:
    resolved_device = str(runtime_summary.get("device", "cpu"))
    if not resolved_device.startswith("cuda"):
        return

    free_gib = runtime_summary.get("cuda_free_gib")
    total_gib = runtime_summary.get("cuda_total_gib")
    if free_gib is None or total_gib is None:
        return

    min_free_gib = float(cfg.cuda_min_free_gib)
    if float(free_gib) >= min_free_gib:
        return

    visible_devices = get_visible_cuda_devices()
    visible_summary = ", ".join(
        (
            f"cuda:{item['cuda_index']}="
            f"{item.get('cuda_free_gib', '?')}/{item.get('cuda_total_gib', '?')}GiB"
        )
        for item in visible_devices
    )
    advice = [
        "free memory on the selected GPU",
        "lower `gan_batch` / `clf_batch`",
        "restart the notebook kernel after switching devices",
    ]
    requested = cfg.device.lower()
    if requested.startswith("cuda:"):
        advice.insert(0, 'set `DEVICE_OVERRIDE="auto"` or choose a roomier GPU such as `cuda:0`')
    elif requested == "auto":
        advice.insert(0, "wait for one of the visible GPUs to free memory")
        advice.insert(1, "lower `CUDA_MIN_FREE_GIB_OVERRIDE` in the notebook if you want a riskier start threshold")
    elif torch.cuda.device_count() > 1:
        advice.insert(0, 'leave `device="auto"` so the runtime can pick the roomiest visible GPU')

    raise RuntimeError(
        "Selected CUDA device does not have enough free memory to start this stage. "
        f"requested={cfg.device!r}, resolved={resolved_device}, "
        f"free={float(free_gib):.1f}GiB/{float(total_gib):.1f}GiB, "
        f"required>={min_free_gib:.1f}GiB. "
        f"Visible GPUs: {visible_summary or 'unavailable'}. "
        f"Next steps: {'; '.join(advice)}."
    )


def resolve_execution_device(cfg: Config) -> tuple[torch.device, Dict[str, Any]]:
    clear_torch_memory()
    resolved_device = cfg.resolve_device()
    runtime_summary = resolve_runtime_summary(cfg, resolved_device=resolved_device)
    validate_runtime_headroom(cfg, runtime_summary)
    device = torch.device(runtime_summary["device"])
    configure_device_runtime(cfg, device)
    return device, runtime_summary


def resolve_runtime_summary(cfg: Config, resolved_device: str | None = None) -> Dict[str, Any]:
    resolved_device = resolved_device or cfg.resolve_device()
    summary: Dict[str, Any] = {
        "device": resolved_device,
        "loader_options": cfg.loader_options(),
        "allow_tf32": cfg.allow_tf32,
    }
    visible_devices = get_visible_cuda_devices()
    if visible_devices:
        summary["visible_cuda_devices"] = visible_devices

    if resolved_device.startswith("cuda"):
        try:
            device = torch.device(resolved_device)
            index = 0 if device.index is None else device.index
            free_bytes, total_bytes = _cuda_mem_get_info(index)

            summary.update(
                {
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_index": index,
                    "cuda_name": torch.cuda.get_device_name(index),
                }
            )
            if free_bytes is not None and total_bytes is not None:
                summary["cuda_free_gib"] = round(int(free_bytes) / (1024 ** 3), 1)
                summary["cuda_total_gib"] = round(int(total_bytes) / (1024 ** 3), 1)
        except Exception:
            pass

    return summary


def configure_device_runtime(cfg: Config, device: torch.device) -> None:
    if device.type != "cuda":
        return

    if device.index is not None:
        torch.cuda.set_device(device.index)

    if cfg.allow_tf32:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        cuda_backend = getattr(torch.backends, "cuda", None)
        if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
            cuda_backend.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _import_classification_report():
    try:
        from sklearn.metrics import classification_report
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for classifier evaluation. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return classification_report


def _load_checkpoint(path: Path | str, device: torch.device) -> Mapping[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _validate_split_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} directory: {path}")
    if not any(child.is_dir() for child in path.iterdir()):
        raise ValueError(f"{label} directory has no class folders: {path}")


def _validate_checkpoint(path: Path | str) -> None:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


def _validate_class_alignment(expected: Sequence[str], actual: Sequence[str], source_label: str) -> None:
    if list(expected) != list(actual):
        raise RuntimeError(
            f"Class mismatch for {source_label}: expected {list(expected)}, got {list(actual)}"
        )


def validate_fid_sample_count(cfg: Config, available_samples: int, strict: bool = True) -> int:
    if not cfg.fid_enabled:
        return 0
    if available_samples <= 0:
        raise ValueError("FID reference split has no images.")

    target_samples = min(int(cfg.fid_n_samples), int(available_samples))
    if target_samples <= 1:
        raise ValueError(
            "FID requires at least two reference images; "
            f"got target={target_samples} from available={available_samples}."
        )
    if target_samples < FID_FEATURE_DIM:
        message = (
            "Strict FID requires at least "
            f"{FID_FEATURE_DIM} reference images for 2048-d Inception features. "
            f"Got target={target_samples} from available={available_samples}."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return target_samples


def get_transform(img_size: int, train: bool = True, augmentation_policy: str = "none") -> transforms.Compose:
    tf_list: list[Any] = [transforms.Resize((img_size, img_size))]
    if train and augmentation_policy == "classical":
        tf_list.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    tf_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transforms.Compose(tf_list)


def get_gan_transform(img_size: int, train: bool = True) -> transforms.Compose:
    tf_list: list[Any] = [transforms.Resize((img_size, img_size))]
    if train:
        tf_list.append(transforms.RandomHorizontalFlip())
    tf_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transforms.Compose(tf_list)


def scenario_augmentation_policy(scenario: str) -> str:
    return "classical" if scenario == "real_aug" else "none"


def subsample_imagefolder(dataset: datasets.ImageFolder, n_per_class: int, seed: int = 42) -> Subset:
    if n_per_class <= 0:
        raise ValueError(f"n_per_class must be positive, got {n_per_class}")

    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    selected: list[int] = []
    shortages: list[str] = []
    for label, indices in sorted(class_indices.items()):
        if len(indices) < n_per_class:
            shortages.append(f"{dataset.classes[label]}={len(indices)}")
            continue
        rng.shuffle(indices)
        selected.extend(indices[:n_per_class])

    if shortages:
        raise ValueError(
            f"Requested n_per_class={n_per_class}, but some classes are too small: {', '.join(shortages)}"
        )

    return Subset(dataset, selected)


def _make_imagefolder(root: Path | str, transform: transforms.Compose, label: str) -> datasets.ImageFolder:
    path = Path(root)
    _validate_split_dir(path, label)
    ds = datasets.ImageFolder(str(path), transform=transform)
    if not ds.classes:
        raise ValueError(f"{label} contains no classes: {path}")
    return ds


def make_gan_loader(
    cfg: Config,
    split_root: Path | str,
    train: bool,
    batch_size: int | None = None,
    n_per_class: int | None = None,
) -> tuple[DataLoader, list[str]]:
    tf = get_gan_transform(cfg.img_size, train=train)
    ds = _make_imagefolder(split_root, transform=tf, label="GAN split")
    if train and n_per_class:
        ds = subsample_imagefolder(ds, n_per_class=n_per_class, seed=cfg.seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size or cfg.gan_batch,
        shuffle=train,
        drop_last=train,
        **cfg.loader_options(),
    )
    class_names = ds.dataset.classes if isinstance(ds, Subset) else ds.classes
    return loader, list(class_names)


def build_classifier_datasets(
    cfg: Config,
    scenario: str,
    n_per_class: int | None,
    synth_dir: Path | str,
    real_train_root: Path | str | None = None,
    test_root: Path | str | None = None,
) -> tuple[Dataset[Any], Dataset[Any], list[str], str]:
    augmentation_policy = scenario_augmentation_policy(scenario)
    tf_train = get_transform(cfg.img_size, train=True, augmentation_policy=augmentation_policy)
    tf_test = get_transform(cfg.img_size, train=False, augmentation_policy="none")

    real_train_root = Path(real_train_root) if real_train_root else cfg.data_root / "train"
    test_root = Path(test_root) if test_root else cfg.data_root / "test"
    synth_dir = Path(synth_dir)

    real_train = _make_imagefolder(real_train_root, tf_train, "real training split")
    test_ds = _make_imagefolder(test_root, tf_test, "test split")
    _validate_class_alignment(real_train.classes, test_ds.classes, "test split")

    if scenario in {"real", "real_aug"}:
        train_ds: Dataset[Any]
        train_ds = subsample_imagefolder(real_train, n_per_class, cfg.seed) if n_per_class else real_train
    elif scenario == "synth":
        synth_train = _make_imagefolder(synth_dir, tf_train, "synthetic training split")
        _validate_class_alignment(real_train.classes, synth_train.classes, "synthetic training split")
        train_ds = subsample_imagefolder(synth_train, n_per_class, cfg.seed) if n_per_class else synth_train
    elif scenario == "both":
        synth_train = _make_imagefolder(synth_dir, tf_train, "synthetic training split")
        _validate_class_alignment(real_train.classes, synth_train.classes, "synthetic training split")
        if n_per_class:
            real_sub = subsample_imagefolder(real_train, n_per_class, cfg.seed)
            synth_sub = subsample_imagefolder(synth_train, n_per_class, cfg.seed)
            train_ds = ConcatDataset([real_sub, synth_sub])
        else:
            train_ds = ConcatDataset([real_train, synth_train])
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return train_ds, test_ds, list(real_train.classes), augmentation_policy


def scenario_time_breakdown(
    scenario: str,
    gan_summary: Mapping[str, Any] | None = None,
    generation_summary: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    if scenario not in {"synth", "both"}:
        return {}

    return {
        "gan_train_time_sec": round_seconds(float((gan_summary or {}).get("train_time_sec", 0.0))),
        "synth_generation_time_sec": round_seconds(float((generation_summary or {}).get("generate_time_sec", 0.0))),
    }


class FIDEvaluator:
    def __init__(self, cfg: Config, real_loader: DataLoader, device: torch.device, strict: bool = True):
        self.cfg = cfg
        self.real_loader = real_loader
        self.device = device
        self.strict = strict
        self._model: nn.Module | None = None
        self._real_stats: tuple[np.ndarray, np.ndarray] | None = None

    def _load_inception(self) -> nn.Module:
        if self._model is None:
            print("Loading Inception-v3 for FID...", flush=True)
            model = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
            model.fc = torch.nn.Identity()
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    @torch.no_grad()
    def _get_inception_features(self, images: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        model = self._load_inception()
        up = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(self.device)
            batch = (batch + 1) / 2
            batch = (batch - mean) / std
            batch = up(batch)
            feats.append(model(batch).cpu())
        return torch.cat(feats, dim=0).numpy()

    def _compute_stats(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(0)
        sigma = np.cov(feats, rowvar=False)
        sigma = sigma + np.eye(sigma.shape[0], dtype=sigma.dtype) * 1e-6
        return mu, sigma

    def _collect_real_stats(self) -> tuple[np.ndarray, np.ndarray]:
        target = validate_fid_sample_count(self.cfg, len(self.real_loader.dataset), strict=self.strict)
        real_imgs = []
        count = 0
        for imgs, _ in self.real_loader:
            real_imgs.append(imgs)
            count += imgs.size(0)
            if count >= target:
                break
        real_batch = torch.cat(real_imgs)[:target]
        return self._compute_stats(self._get_inception_features(real_batch, batch_size=min(self.cfg.gan_batch, 64)))

    def _ensure_real_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self._real_stats is None:
            self._real_stats = self._collect_real_stats()
        return self._real_stats

    def compute(self, generator: nn.Module, num_classes: int) -> float:
        mu_real, sig_real = self._ensure_real_stats()
        target = min(int(self.cfg.fid_n_samples), len(self.real_loader.dataset))
        fake_imgs = []

        generator.eval()
        remaining = target
        while remaining > 0:
            batch_size = min(self.cfg.gan_batch, remaining)
            z = torch.randn(batch_size, self.cfg.z_dim, device=self.device)
            y = torch.randint(0, num_classes, (batch_size,), device=self.device)
            with torch.no_grad():
                fake_imgs.append(generator(z, y).cpu())
            remaining -= batch_size
        generator.train()

        fake_batch = torch.cat(fake_imgs)[:target]
        mu_fake, sig_fake = self._compute_stats(
            self._get_inception_features(fake_batch, batch_size=min(self.cfg.gan_batch, 64))
        )
        return calc_fid(mu_real, sig_real, mu_fake, sig_fake)


def calc_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def save_samples(generator: nn.Module, fixed_z: torch.Tensor, fixed_y: torch.Tensor, epoch: int, out_dir: Path) -> None:
    generator.eval()
    with torch.no_grad():
        imgs = generator(fixed_z, fixed_y)
    vutils.save_image(
        imgs,
        out_dir / f"samples_epoch{epoch:04d}.png",
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )
    generator.train()


def maybe_compile_classifier(model: nn.Module, cfg: Config) -> nn.Module:
    if not cfg.classifier_compile:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("`torch.compile` is not available in this PyTorch build.")
    return torch.compile(model)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def preflight_check(
    cfg: Config,
    *,
    data_root: Path | str | None = None,
    real_train_root: Path | str | None = None,
    test_root: Path | str | None = None,
    synth_dir: Path | str | None = None,
    checkpoint: Path | str | None = None,
    require_sklearn: bool = False,
    require_fid: bool = False,
    strict_fid: bool = True,
    validate_runtime: bool = True,
) -> Dict[str, Any]:
    summary = resolve_runtime_summary(cfg)
    if validate_runtime:
        validate_runtime_headroom(cfg, summary)

    if require_sklearn:
        _import_classification_report()
        summary["sklearn"] = True

    if data_root is not None:
        root = Path(data_root)
        _validate_split_dir(root / "train", "train split")
        if (root / "test").exists():
            _validate_split_dir(root / "test", "test split")
        summary["data_root"] = str(root)

    if real_train_root is not None:
        _validate_split_dir(Path(real_train_root), "real training split")
        summary["real_train_root"] = str(real_train_root)

    if test_root is not None:
        _validate_split_dir(Path(test_root), "test split")
        summary["test_root"] = str(test_root)

    if synth_dir is not None and Path(synth_dir).exists():
        _validate_split_dir(Path(synth_dir), "synthetic training split")
        summary["synth_dir"] = str(synth_dir)

    if checkpoint is not None:
        _validate_checkpoint(checkpoint)
        summary["checkpoint"] = str(checkpoint)

    if require_fid:
        fid_root = Path(data_root or cfg.data_root)
        split_root = fid_root / cfg.fid_reference_split
        _validate_split_dir(split_root, f"FID reference split '{cfg.fid_reference_split}'")
        available = sum(1 for path in split_root.rglob("*") if path.is_file())
        summary["fid_reference_split"] = cfg.fid_reference_split
        summary["fid_reference_available"] = available
        validate_fid_sample_count(cfg, available, strict=strict_fid)

    return summary


def train_gan(
    cfg: Config,
    *,
    data_root: Path | str | None = None,
    fid_root: Path | str | None = None,
    out_dir: Path | str | None = None,
    train_n_per_class: int | None = None,
    strict_fid: bool = True,
    return_models: bool = True,
) -> Dict[str, Any]:
    data_root = Path(data_root) if data_root is not None else cfg.data_root
    fid_root = Path(fid_root) if fid_root is not None else data_root
    out_dir = Path(out_dir) if out_dir is not None else Path(cfg.out_root) / "gan"
    def _run(stage_cfg: Config, device: torch.device, runtime_summary: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Device: {runtime_summary['device']}")
        if runtime_summary.get("cuda_name"):
            free_gib = runtime_summary.get("cuda_free_gib")
            total_gib = runtime_summary.get("cuda_total_gib")
            if free_gib is not None and total_gib is not None:
                print(
                    "CUDA runtime: "
                    f"index={runtime_summary['cuda_index']} "
                    f"name={runtime_summary['cuda_name']} "
                    f"free={free_gib:.1f}GiB/{total_gib:.1f}GiB"
                )
            else:
                print(
                    "CUDA runtime: "
                    f"index={runtime_summary['cuda_index']} "
                    f"name={runtime_summary['cuda_name']}"
                )
        print(f"Loader options: {runtime_summary['loader_options']}")

        preflight_check(stage_cfg, data_root=data_root)
        if stage_cfg.fid_enabled:
            preflight_check(
                stage_cfg,
                data_root=fid_root,
                require_fid=True,
                strict_fid=strict_fid,
            )

        set_random_seeds(stage_cfg.seed)

        train_loader, class_names = make_gan_loader(
            stage_cfg,
            split_root=data_root / "train",
            train=True,
            n_per_class=train_n_per_class,
        )
        fid_loader, fid_classes = make_gan_loader(
            stage_cfg,
            split_root=fid_root / stage_cfg.fid_reference_split,
            train=False,
            batch_size=min(stage_cfg.gan_batch, 64),
        )
        _validate_class_alignment(class_names, fid_classes, "FID reference split")

        num_classes = len(class_names)
        print(f"Train loader ready: {len(train_loader.dataset)} images across {num_classes} classes")
        print(f"FID loader ready: {len(fid_loader.dataset)} images from '{stage_cfg.fid_reference_split}' split")

        generator = Generator(z_dim=stage_cfg.z_dim, num_classes=num_classes).to(device)
        discriminator = ProjectionDiscriminator(num_classes=num_classes).to(device)
        criterion = nn.BCEWithLogitsLoss()
        opt_g = torch.optim.Adam(generator.parameters(), lr=stage_cfg.gan_lr_g, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=stage_cfg.gan_lr_d, betas=(0.5, 0.999))
        print(
            "GAN initialized: "
            f"batch={stage_cfg.gan_batch}, z_dim={stage_cfg.z_dim}, fid_enabled={stage_cfg.fid_enabled}, "
            f"fid_n_samples={stage_cfg.fid_n_samples}"
        )

        fixed_z = torch.randn(24, stage_cfg.z_dim, device=device)
        fixed_y = torch.arange(num_classes, device=device).repeat_interleave(8)

        stage_out_dir = _ensure_dir(out_dir)
        ckpt_dir = _ensure_dir(stage_out_dir / "checkpoints")

        fid_evaluator = (
            FIDEvaluator(stage_cfg, fid_loader, device, strict=strict_fid) if stage_cfg.fid_enabled else None
        )
        train_log: list[dict[str, Any]] = []
        best_fid = float("inf")
        best_epoch: int | None = None
        latest_checkpoint: Path | None = None
        start_time = time.time()
        print("Starting GAN training...")

        for epoch in range(1, stage_cfg.gan_epochs + 1):
            epoch_start = time.time()
            d_loss_acc = 0.0
            g_loss_acc = 0.0
            d_x_acc = 0.0
            d_gz_acc = 0.0
            n_batches = 0

            for real_imgs, real_labels in train_loader:
                real_imgs = real_imgs.to(device)
                real_labels = real_labels.to(device)
                batch_size = real_imgs.size(0)

                real_targets = torch.ones(batch_size, 1, device=device)
                fake_targets = torch.zeros(batch_size, 1, device=device)

                with torch.no_grad():
                    z = torch.randn(batch_size, stage_cfg.z_dim, device=device)
                    fake_imgs = generator(z, real_labels)

                d_real_out = discriminator(real_imgs, real_labels)
                d_fake_out = discriminator(fake_imgs, real_labels)
                d_loss = criterion(d_real_out, real_targets) + criterion(d_fake_out, fake_targets)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                opt_d.step()

                z = torch.randn(batch_size, stage_cfg.z_dim, device=device)
                gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
                fake_imgs = generator(z, gen_labels)
                g_out = discriminator(fake_imgs, gen_labels)
                g_loss = criterion(g_out, real_targets)
                opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                opt_g.step()

                d_loss_acc += d_loss.item()
                g_loss_acc += g_loss.item()
                d_x_acc += torch.sigmoid(d_real_out).mean().item()
                d_gz_acc += torch.sigmoid(g_out).mean().item()
                n_batches += 1

            sync_device(device)
            d_avg = d_loss_acc / max(1, n_batches)
            g_avg = g_loss_acc / max(1, n_batches)
            d_x_avg = d_x_acc / max(1, n_batches)
            d_gz_avg = d_gz_acc / max(1, n_batches)

            log_row: dict[str, Any] = {
                "epoch": epoch,
                "d_loss": d_avg,
                "g_loss": g_avg,
                "d_x": d_x_avg,
                "d_gz": d_gz_avg,
            }
            fid_str = ""

            if stage_cfg.fid_enabled and fid_evaluator is not None and (
                epoch % stage_cfg.fid_every == 0 or epoch == stage_cfg.gan_epochs
            ):
                fid = fid_evaluator.compute(generator, num_classes)
                sync_device(device)
                log_row["fid"] = fid
                fid_str = f"  FID: {fid:.2f}"
                if fid < best_fid:
                    best_fid = fid
                    best_epoch = epoch
                    best_path = ckpt_dir / "best_fid.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "fid": fid,
                            "class_names": class_names,
                            "G": generator.state_dict(),
                            "D": discriminator.state_dict(),
                            "opt_G": opt_g.state_dict(),
                            "opt_D": opt_d.state_dict(),
                        },
                        best_path,
                    )
                    latest_checkpoint = best_path

            train_log.append(log_row)

            epoch_time = time.time() - epoch_start
            print(
                f"[Epoch {epoch:03d}/{stage_cfg.gan_epochs}]  D_loss: {d_avg:.4f}  "
                f"G_loss: {g_avg:.4f}  D(x): {d_x_avg:.3f}  D(G(z)): {d_gz_avg:.3f}"
                f"  Time: {epoch_time:.1f}s{fid_str}"
            )

            if epoch % stage_cfg.sample_every == 0 or epoch == 1:
                save_samples(generator, fixed_z, fixed_y, epoch, stage_out_dir)
            if epoch % stage_cfg.ckpt_every == 0 or epoch == stage_cfg.gan_epochs:
                latest_checkpoint = ckpt_dir / f"ckpt_epoch{epoch:04d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "class_names": class_names,
                        "G": generator.state_dict(),
                        "D": discriminator.state_dict(),
                        "opt_G": opt_g.state_dict(),
                        "opt_D": opt_d.state_dict(),
                    },
                    latest_checkpoint,
                )

        with (stage_out_dir / "train_log.json").open("w") as f:
            json.dump(train_log, f, indent=2)

        generator_checkpoint = ckpt_dir / "best_fid.pt"
        if not generator_checkpoint.exists():
            generator_checkpoint = latest_checkpoint
        if generator_checkpoint is None:
            raise RuntimeError("GAN training finished without producing a checkpoint.")

        summary = {
            "data_root": str(data_root),
            "fid_root": str(fid_root),
            "fid_reference_split": stage_cfg.fid_reference_split,
            "device": str(device),
            "loader_options": runtime_summary["loader_options"],
            "runtime_summary": runtime_summary,
            "num_classes": num_classes,
            "train_samples": len(train_loader.dataset),
            "fid_samples": len(fid_loader.dataset),
            "train_n_per_class": train_n_per_class,
            "train_time_sec": round_seconds(time.time() - start_time),
            "best_fid": None if best_fid == float("inf") else round(best_fid, 4),
            "best_epoch": best_epoch,
            "fid_enabled": stage_cfg.fid_enabled,
            "generator_checkpoint": str(generator_checkpoint),
            "out_dir": str(stage_out_dir),
        }
        with (stage_out_dir / "run_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        if not return_models:
            clear_torch_memory(
                generator,
                discriminator,
                fid_evaluator._model if fid_evaluator is not None else None,
            )
            return {"summary": summary}

        return {"G": generator, "D": discriminator, "summary": summary}

    return run_stage_with_device_fallback(cfg, "train_gan", _run)


def clear_synth_dir(out_root: Path | str, class_names: Sequence[str]) -> None:
    out_root = Path(out_root)
    for class_name in class_names:
        class_dir = out_root / class_name
        if not class_dir.exists():
            continue
        for image_path in class_dir.glob("*.png"):
            image_path.unlink()


def generate_synthetic_pool(
    cfg: Config,
    *,
    checkpoint: Path | str,
    n_per_class: int,
    out_dir: Path | str,
    batch_size: int = 64,
    seed: int | None = None,
    class_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint)
    out_root = Path(out_dir)
    seed = cfg.seed if seed is None else seed

    def _run(stage_cfg: Config, device: torch.device, runtime_summary: Dict[str, Any]) -> Dict[str, Any]:
        preflight_check(stage_cfg, checkpoint=ckpt_path)
        set_random_seeds(seed)

        state = _load_checkpoint(ckpt_path, torch.device("cpu"))
        resolved_class_names = class_names
        if resolved_class_names is None:
            resolved_class_names = state.get("class_names")
        resolved_class_names = list(resolved_class_names or ["apple", "banana", "orange"])

        generator = Generator(z_dim=stage_cfg.z_dim, num_classes=len(resolved_class_names)).to(device)
        generator.load_state_dict(state["G"])
        generator.eval()

        start_time = time.time()
        counts: Dict[str, int] = {}
        clear_synth_dir(out_root, resolved_class_names)

        for class_index, class_name in enumerate(resolved_class_names):
            class_dir = _ensure_dir(out_root / class_name)
            generated = 0
            while generated < n_per_class:
                current_batch = min(batch_size, n_per_class - generated)
                z = torch.randn(current_batch, stage_cfg.z_dim, device=device)
                y = torch.full((current_batch,), class_index, dtype=torch.long, device=device)
                with torch.no_grad():
                    imgs = generator(z, y)

                for i in range(current_batch):
                    save_image(
                        imgs[i],
                        class_dir / f"{class_name}_synth_{generated + i:05d}.png",
                        normalize=True,
                        value_range=(-1, 1),
                    )
                generated += current_batch
            counts[class_name] = generated

        summary = {
            "checkpoint": str(ckpt_path),
            "out_dir": str(out_root),
            "n_per_class": n_per_class,
            "counts": counts,
            "seed": seed,
            "runtime_summary": runtime_summary,
            "generate_time_sec": round_seconds(time.time() - start_time),
        }
        _ensure_dir(out_root)
        with (out_root / "generation_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        del state
        clear_torch_memory(generator)
        return summary

    return run_stage_with_device_fallback(cfg, "generate_synthetic_pool", _run)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def run_classifier_experiment(
    cfg: Config,
    scenario: str,
    n_per_class: int | None,
    *,
    synth_dir: Path | str,
    out_dir: Path | str,
    real_train_root: Path | str | None = None,
    test_root: Path | str | None = None,
    time_breakdown: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    classification_report = _import_classification_report()

    def _run(stage_cfg: Config, device: torch.device, runtime_summary: Dict[str, Any]) -> Dict[str, Any]:
        preflight_check(
            stage_cfg,
            real_train_root=real_train_root or (stage_cfg.data_root / "train"),
            test_root=test_root or (stage_cfg.data_root / "test"),
            synth_dir=synth_dir if scenario in {"synth", "both"} else None,
            require_sklearn=True,
        )

        set_random_seeds(stage_cfg.seed)
        train_ds, test_ds, class_names, augmentation_policy = build_classifier_datasets(
            stage_cfg,
            scenario,
            n_per_class,
            synth_dir=synth_dir,
            real_train_root=real_train_root,
            test_root=test_root,
        )

        loader_kwargs = stage_cfg.loader_options()
        train_loader = DataLoader(
            train_ds,
            batch_size=stage_cfg.clf_batch,
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )
        test_loader = DataLoader(test_ds, batch_size=stage_cfg.clf_batch, shuffle=False, **loader_kwargs)

        model = maybe_compile_classifier(FruitCNN(num_classes=len(class_names)).to(device), stage_cfg)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=stage_cfg.clf_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg.clf_epochs)

        print(
            f"[{scenario}] n_per_class={n_per_class or 'all'} "
            f"train_size={len(train_ds)} device={runtime_summary['device']}"
        )
        start_time = time.time()
        for epoch in range(1, stage_cfg.clf_epochs + 1):
            loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            if epoch % 10 == 0 or epoch == stage_cfg.clf_epochs:
                print(f"  Epoch {epoch:03d}  loss={loss:.4f}  train_acc={acc:.4f}")
        sync_device(device)
        train_time = time.time() - start_time

        preds, labels = evaluate(model, test_loader, device)
        report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
        test_acc = float(report["accuracy"])

        scenario_time_parts = dict(time_breakdown or {})
        gan_train_time = round_seconds(float(scenario_time_parts.get("gan_train_time_sec", 0.0)))
        synth_generation_time = round_seconds(float(scenario_time_parts.get("synth_generation_time_sec", 0.0)))
        pipeline_time = round_seconds(train_time + gan_train_time + synth_generation_time)

        result = {
            "scenario": scenario,
            "augmentation_policy": augmentation_policy,
            "n_per_class": n_per_class or "all",
            "train_size": len(train_ds),
            "test_accuracy": round(test_acc, 4),
            "train_time_sec": round_seconds(train_time),
            "classifier_train_time_sec": round_seconds(train_time),
            "gan_train_time_sec": gan_train_time,
            "synth_generation_time_sec": synth_generation_time,
            "pipeline_time_sec": pipeline_time,
            "real_train_root": str(real_train_root or (stage_cfg.data_root / "train")),
            "test_root": str(test_root or (stage_cfg.data_root / "test")),
            "synth_dir": str(Path(synth_dir)),
            "classifier_compile": stage_cfg.classifier_compile,
            "runtime_summary": runtime_summary,
            "per_class": {
                name: {
                    "precision": round(float(report[name]["precision"]), 4),
                    "recall": round(float(report[name]["recall"]), 4),
                    "f1": round(float(report[name]["f1-score"]), 4),
                }
                for name in class_names
            },
        }
        if extra_metadata:
            result.update(extra_metadata)

        out_path = _ensure_dir(Path(out_dir))
        tag = f"{scenario}_n{n_per_class}" if n_per_class else f"{scenario}_all"
        with (out_path / f"result_{tag}.json").open("w") as f:
            json.dump(result, f, indent=2)
        clear_torch_memory(model)
        return result

    return run_stage_with_device_fallback(cfg, "run_classifier_experiment", _run)


def run_task1_pipeline(
    cfg: Config,
    *,
    sizes: Sequence[int] | None = None,
    scenarios: Sequence[str] | None = None,
    data_root: Path | str | None = None,
    out_root: Path | str | None = None,
    synth_batch_size: int = 64,
    seed: int | None = None,
    strict_fid: bool = True,
) -> Dict[str, Any]:
    sizes = list(sizes or DEFAULT_TASK1_SIZES)
    scenarios = list(scenarios or DEFAULT_TASK1_SCENARIOS)
    data_root = Path(data_root) if data_root is not None else cfg.data_root
    out_root = Path(out_root) if out_root is not None else Path(cfg.out_root) / "task1"
    seed = cfg.seed if seed is None else seed

    preflight_check(
        cfg,
        data_root=data_root,
        require_sklearn=True,
        require_fid=cfg.fid_enabled,
        strict_fid=strict_fid,
        validate_runtime=False,
    )
    _ensure_dir(out_root)

    all_results: list[dict[str, Any]] = []
    pipeline_summary: list[dict[str, Any]] = []
    clf_out_dir = _ensure_dir(out_root / "clf")

    for n_per_class in sizes:
        print(f"\n=== Task 1 budget: n={n_per_class} per class ===")
        gan_out_dir = out_root / "gan" / f"n{n_per_class}"
        synth_out_dir = out_root / "synth" / f"n{n_per_class}"
        gan_run = train_gan(
            cfg,
            data_root=data_root,
            fid_root=data_root,
            out_dir=gan_out_dir,
            train_n_per_class=n_per_class,
            strict_fid=strict_fid,
            return_models=False,
        )
        gan_summary = gan_run["summary"]
        generation_summary = generate_synthetic_pool(
            cfg,
            checkpoint=gan_summary["generator_checkpoint"],
            n_per_class=n_per_class,
            out_dir=synth_out_dir,
            batch_size=synth_batch_size,
            seed=seed,
        )
        pipeline_summary.append(
            {
                "n_per_class": n_per_class,
                "gan": gan_summary,
                "synth": generation_summary,
            }
        )

        for scenario in scenarios:
            result = run_classifier_experiment(
                cfg,
                scenario,
                n_per_class,
                synth_dir=synth_out_dir,
                out_dir=clf_out_dir,
                real_train_root=data_root / "train",
                test_root=data_root / "test",
                time_breakdown=scenario_time_breakdown(
                    scenario,
                    gan_summary=gan_summary,
                    generation_summary=generation_summary,
                ),
                extra_metadata={
                    "budget_n_per_class": n_per_class,
                    "gan_out_dir": str(gan_out_dir),
                    "synth_out_dir": str(synth_out_dir),
                },
            )
            all_results.append(result)
        clear_torch_memory()

    with (clf_out_dir / "all_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    with (out_root / "pipeline_summary.json").open("w") as f:
        json.dump(pipeline_summary, f, indent=2)

    return {
        "all_results": all_results,
        "pipeline_summary": pipeline_summary,
        "out_root": str(out_root),
    }
