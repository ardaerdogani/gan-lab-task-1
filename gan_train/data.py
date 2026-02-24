from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from utils import get_transform, should_pin_memory


def _allocate_per_class(total_count, class_counts):
    if total_count < len(class_counts):
        raise ValueError("Toplam sample sayisi class sayisindan kucuk olamaz.")

    ratios = class_counts / class_counts.sum()
    raw = ratios * total_count
    alloc = np.floor(raw).astype(np.int64)
    alloc = np.maximum(alloc, 1)

    while alloc.sum() > total_count:
        candidates = np.where(alloc > 1)[0]
        if len(candidates) == 0:
            break
        idx = candidates[np.argmax(alloc[candidates])]
        alloc[idx] -= 1

    remainder = int(total_count - alloc.sum())
    if remainder > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(frac)[::-1]
        for i in range(remainder):
            alloc[order[i % len(order)]] += 1

    return alloc


def make_stratified_subset_by_count(dataset, total_count, seed):
    n_total = len(dataset)
    if total_count is None or total_count >= n_total:
        return dataset
    if total_count <= 0:
        raise ValueError("subset_count pozitif olmali.")

    targets = np.array(dataset.targets)
    class_ids = np.unique(targets)
    class_counts = np.array([int((targets == c).sum()) for c in class_ids], dtype=np.int64)
    alloc = _allocate_per_class(total_count, class_counts)

    rng = np.random.default_rng(seed)
    selected = []
    for class_id, take_count in zip(class_ids, alloc.tolist()):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        k = min(int(take_count), len(class_indices))
        selected.extend(class_indices[:k].tolist())

    if len(selected) < total_count:
        remaining = np.setdiff1d(np.arange(n_total), np.array(selected, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)
        needed = total_count - len(selected)
        selected.extend(remaining[:needed].tolist())

    rng.shuffle(selected)
    return Subset(dataset, selected)


def extract_targets(dataset):
    if isinstance(dataset, Subset):
        base_targets = np.array(extract_targets(dataset.dataset))
        return base_targets[np.array(dataset.indices)]
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)}")


def extract_class_to_idx(dataset: ImageFolder | Subset) -> Dict[str, int]:
    if isinstance(dataset, Subset):
        if not isinstance(dataset.dataset, ImageFolder):
            raise TypeError(f"Subset base dataset does not expose class_to_idx: {type(dataset.dataset)}")
        return dataset.dataset.class_to_idx
    if isinstance(dataset, ImageFolder):
        return dataset.class_to_idx
    raise TypeError(f"Unsupported dataset type for class_to_idx extraction: {type(dataset)}")


def build_balanced_sampler(dataset, num_classes):
    targets = extract_targets(dataset).astype(np.int64)
    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    class_weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = class_counts > 0
    class_weights[nonzero] = class_counts.sum() / (class_counts[nonzero] * nonzero.sum())
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return class_counts.astype(np.int64), class_weights.astype(np.float32), sampler


def build_train_loader(
    data_train,
    img_size,
    batch_size,
    num_workers,
    subset_count,
    seed,
    device,
    balanced_sampler,
) -> Tuple[ImageFolder | Subset, DataLoader]:
    train_ds = ImageFolder(data_train, transform=get_transform(img_size=img_size))
    train_ds = make_stratified_subset_by_count(train_ds, subset_count, seed)
    class_to_idx = extract_class_to_idx(train_ds)
    num_classes = len(class_to_idx)

    sampler = None
    if balanced_sampler:
        _, _, sampler = build_balanced_sampler(train_ds, num_classes=num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=should_pin_memory(device),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    return train_ds, train_loader
