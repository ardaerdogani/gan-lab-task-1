from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from fid_core import (
    InceptionFeatureExtractor,
    compute_fid_from_features,
    extract_features_from_paths,
    extract_features_from_tensor_batches,
    list_images_by_class,
)
from utils import get_best_device


@dataclass
class FIDEvalState:
    enabled: bool
    model: nn.Module | None = None
    device: torch.device | None = None
    num_workers: int = 0
    img_size: int = 299
    batch_size: int = 64
    class_idx_to_name: Dict[int, str] | None = None
    real_features_overall: np.ndarray | None = None
    real_features_by_class: Dict[int, np.ndarray] | None = None
    fixed_noise_by_class: Dict[int, torch.Tensor] | None = None


def resolve_fid_device(name: str, train_device: torch.device) -> torch.device:
    normalized = name.strip().lower()
    if normalized in {"same", ""}:
        return train_device
    if normalized == "auto":
        return get_best_device()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise ValueError("fid_device=cuda secildi ama CUDA mevcut degil.")
    if normalized == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError("fid_device=mps secildi ama MPS mevcut degil.")
    raise ValueError(f"Unsupported fid_device: {name}")


def _sample_paths(paths, count, seed):
    if len(paths) < count:
        count = len(paths)
    if count < 2:
        raise ValueError("FID subseti icin class basina en az 2 goruntu gerekli.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=count, replace=False)
    idx.sort()
    return [paths[i] for i in idx.tolist()]


def build_fid_eval_state(
    args,
    class_to_idx: Dict[str, int],
    z_dim: int,
    train_device: torch.device,
    num_workers: int,
) -> FIDEvalState:
    if args.fid_every <= 0:
        return FIDEvalState(enabled=False)
    if args.fid_eval_per_class < 2:
        raise ValueError("--fid-eval-per-class en az 2 olmali.")

    fid_device = resolve_fid_device(args.fid_device, train_device)
    weights_path = Path(args.fid_weights_path) if args.fid_weights_path else None
    fid_model = InceptionFeatureExtractor(weights_path=weights_path).to(fid_device)

    class_to_paths = list_images_by_class(Path(args.data_train))
    class_idx_to_name = {idx: name for name, idx in class_to_idx.items()}

    real_features_by_class = {}
    fixed_noise_by_class = {}
    overall_real_features = []

    for class_idx in sorted(class_idx_to_name.keys()):
        class_name = class_idx_to_name[class_idx]
        if class_name not in class_to_paths:
            raise FileNotFoundError(f"FID class path missing: {class_name}")
        sampled_paths = _sample_paths(
            class_to_paths[class_name],
            count=args.fid_eval_per_class,
            seed=args.seed + 3000 + class_idx,
        )

        real_features = extract_features_from_paths(
            image_paths=sampled_paths,
            model=fid_model,
            device=fid_device,
            batch_size=args.batch_size,
            num_workers=num_workers,
            img_size=299,
        )
        real_features_by_class[class_idx] = real_features
        overall_real_features.append(real_features)

        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(args.seed + 5000 + class_idx)
        fixed_noise = torch.randn(len(sampled_paths), z_dim, generator=cpu_gen, dtype=torch.float32)
        fixed_noise_by_class[class_idx] = fixed_noise.to(train_device)

    return FIDEvalState(
        enabled=True,
        model=fid_model,
        device=fid_device,
        num_workers=num_workers,
        img_size=299,
        batch_size=args.batch_size,
        class_idx_to_name=class_idx_to_name,
        real_features_overall=np.concatenate(overall_real_features, axis=0).astype(np.float64),
        real_features_by_class=real_features_by_class,
        fixed_noise_by_class=fixed_noise_by_class,
    )


@torch.no_grad()
def evaluate_fid(generator, fid_state: FIDEvalState):
    if not fid_state.enabled:
        return None, {}

    was_training = generator.training
    generator.eval()

    fake_features_by_class = {}
    ordered_idx = sorted(fid_state.class_idx_to_name.keys())

    for class_idx in ordered_idx:
        noise = fid_state.fixed_noise_by_class[class_idx]
        labels = torch.full((noise.size(0),), class_idx, device=noise.device, dtype=torch.long)

        batches = []
        for start in range(0, noise.size(0), fid_state.batch_size):
            z = noise[start : start + fid_state.batch_size]
            y = labels[start : start + fid_state.batch_size]
            fake = generator(z, y).detach()
            batches.append(fake.to(fid_state.device))

        fake_features = extract_features_from_tensor_batches(
            image_batches=batches,
            model=fid_state.model,
            device=fid_state.device,
            img_size=fid_state.img_size,
            input_range="-1_1",
        )
        fake_features_by_class[class_idx] = fake_features

    overall_fake = np.concatenate([fake_features_by_class[idx] for idx in ordered_idx], axis=0)
    overall_fid = compute_fid_from_features(fid_state.real_features_overall, overall_fake)

    per_class_fid = {}
    for class_idx in ordered_idx:
        class_name = fid_state.class_idx_to_name[class_idx]
        per_class_fid[class_name] = compute_fid_from_features(
            fid_state.real_features_by_class[class_idx],
            fake_features_by_class[class_idx],
        )

    if was_training:
        generator.train()
    return float(overall_fid), per_class_fid
