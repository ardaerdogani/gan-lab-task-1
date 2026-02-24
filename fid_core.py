from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import InterpolationMode

from utils import should_pin_memory

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self, weights_path: Path | None = None):
        super().__init__()
        self.model = self._build_model(weights_path)
        self.model.eval()

    @staticmethod
    def _build_model(weights_path: Path | None):
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            has_aux_logits = any(k.startswith("AuxLogits.") for k in state.keys())
            model = inception_v3(
                weights=None,
                aux_logits=has_aux_logits,
                transform_input=False,
                init_weights=False,
            )
            model.load_state_dict(state, strict=True)
        else:
            try:
                model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
            except Exception as exc:
                raise RuntimeError(
                    "Inception agirliklari yuklenemedi. Internete erisim yoksa --weights-path ile "
                    "yerel inception_v3 agirlik dosyasini ver."
                ) from exc
        model.fc = nn.Identity()
        return model

    def forward(self, x):
        features = self.model(x)
        if isinstance(features, tuple):
            features = features[0]
        return features


def get_fid_path_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Klasor bulunamadi: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def list_images_by_class(root: Path) -> Dict[str, List[Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Klasor bulunamadi: {root}")
    class_to_paths: Dict[str, List[Path]] = {}
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        class_to_paths[class_dir.name] = list_images(class_dir)
    return class_to_paths


def choose_paths(paths: Sequence[Path], max_images: int | None, seed: int) -> List[Path]:
    paths = list(paths)
    if max_images is None or len(paths) <= max_images:
        return paths
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))[:max_images]
    idx.sort()
    return [paths[i] for i in idx.tolist()]


def sample_equal_pair(
    real_paths: Sequence[Path],
    fake_paths: Sequence[Path],
    seed: int,
    max_count: int | None = None,
) -> tuple[List[Path], List[Path]]:
    real_paths = list(real_paths)
    fake_paths = list(fake_paths)
    if len(real_paths) < 2 or len(fake_paths) < 2:
        raise ValueError("FID icin her tarafta en az 2 goruntu gerekiyor.")

    target = min(len(real_paths), len(fake_paths))
    if max_count is not None:
        target = min(target, int(max_count))
    if target < 2:
        raise ValueError("Esitlenmis sample sayisi 2'den kucuk olamaz.")

    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real_paths), size=target, replace=False)
    fake_idx = rng.choice(len(fake_paths), size=target, replace=False)
    real_idx.sort()
    fake_idx.sort()
    return [real_paths[i] for i in real_idx.tolist()], [fake_paths[i] for i in fake_idx.tolist()]


@torch.no_grad()
def extract_features_from_paths(
    image_paths: Sequence[Path],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    img_size: int,
) -> np.ndarray:
    ds = ImagePathDataset(image_paths=image_paths, transform=get_fid_path_transform(img_size=img_size))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=should_pin_memory(device),
        persistent_workers=num_workers > 0,
    )

    features = []
    for images in loader:
        images = images.to(device, non_blocking=True)
        feats = model(images).flatten(start_dim=1)
        features.append(feats.cpu().numpy())

    if not features:
        raise ValueError("Ozellik cikarmak icin en az bir goruntu gerekiyor.")
    return np.concatenate(features, axis=0).astype(np.float64)


def preprocess_tensor_for_fid(images: torch.Tensor, img_size: int, input_range: str) -> torch.Tensor:
    if input_range == "-1_1":
        images = (images + 1.0) / 2.0
    elif input_range == "0_1":
        pass
    else:
        raise ValueError(f"Unsupported input_range: {input_range}")

    images = images.clamp(0.0, 1.0)
    images = torch.nn.functional.interpolate(
        images,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def extract_features_from_tensor_batches(
    image_batches: Iterable[torch.Tensor],
    model: nn.Module,
    device: torch.device,
    img_size: int,
    input_range: str = "-1_1",
) -> np.ndarray:
    features = []
    for batch in image_batches:
        if batch.numel() == 0:
            continue
        batch = batch.to(device, non_blocking=True)
        batch = preprocess_tensor_for_fid(batch, img_size=img_size, input_range=input_range)
        feats = model(batch).flatten(start_dim=1)
        features.append(feats.cpu().numpy())

    if not features:
        raise ValueError("Tensor batch listesinden ozellik cikartilamadi.")
    return np.concatenate(features, axis=0).astype(np.float64)


def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_fid_from_features(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu_real, sigma_real = compute_stats(real_features)
    mu_fake, sigma_fake = compute_stats(fake_features)
    return frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


def compute_fid_for_path_pair(
    real_paths: Sequence[Path],
    fake_paths: Sequence[Path],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    img_size: int,
) -> float:
    if len(real_paths) < 2 or len(fake_paths) < 2:
        raise ValueError("FID icin her tarafta en az 2 goruntu gerekiyor.")
    real_feat = extract_features_from_paths(
        image_paths=real_paths,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
    )
    fake_feat = extract_features_from_paths(
        image_paths=fake_paths,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
    )
    return compute_fid_from_features(real_feat, fake_feat)
