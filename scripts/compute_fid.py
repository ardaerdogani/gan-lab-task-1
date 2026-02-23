from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import default_num_workers, get_best_device, should_pin_memory

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
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


def list_images(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Klasor bulunamadi: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def choose_paths(paths, max_images, seed):
    if max_images is None or len(paths) <= max_images:
        return paths
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))[:max_images]
    idx.sort()
    return [paths[i] for i in idx.tolist()]


def get_transform(img_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


@torch.no_grad()
def extract_features(image_paths, model, device, batch_size, num_workers, img_size):
    ds = ImagePathDataset(image_paths=image_paths, transform=get_transform(img_size=img_size))
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
    return np.concatenate(features, axis=0).astype(np.float64)


def compute_stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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


def compute_fid_for_pair(
    real_paths,
    fake_paths,
    model,
    device,
    batch_size,
    num_workers,
    img_size,
):
    if len(real_paths) < 2 or len(fake_paths) < 2:
        raise ValueError("FID icin her tarafta en az 2 goruntu gerekiyor.")

    real_feat = extract_features(real_paths, model, device, batch_size, num_workers, img_size)
    fake_feat = extract_features(fake_paths, model, device, batch_size, num_workers, img_size)
    mu_real, sigma_real = compute_stats(real_feat)
    mu_fake, sigma_fake = compute_stats(fake_feat)
    return frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID between real and synthetic image folders.")
    parser.add_argument("--real-dir", type=str, default="data/split/train")
    parser.add_argument("--fake-dir", type=str, default="data/synthetic")
    parser.add_argument("--weights-path", type=str, default=None, help="Local inception_v3 weights path (.pth)")
    parser.add_argument("--img-size", type=int, default=299)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 => auto")
    parser.add_argument("--max-images", type=int, default=None, help="Sample up to N images from each domain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", action="store_true", help="Also compute class-wise FID")
    parser.add_argument("--out-json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_best_device()
    num_workers = default_num_workers() if args.num_workers < 0 else args.num_workers

    real_root = Path(args.real_dir)
    fake_root = Path(args.fake_dir)
    weights_path = Path(args.weights_path) if args.weights_path else None

    try:
        model = InceptionFeatureExtractor(weights_path=weights_path).to(device)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    real_paths = choose_paths(list_images(real_root), args.max_images, args.seed)
    fake_paths = choose_paths(list_images(fake_root), args.max_images, args.seed)
    overall_fid = compute_fid_for_pair(
        real_paths=real_paths,
        fake_paths=fake_paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=num_workers,
        img_size=args.img_size,
    )

    output = {
        "real_dir": str(real_root),
        "fake_dir": str(fake_root),
        "real_count": len(real_paths),
        "fake_count": len(fake_paths),
        "overall_fid": overall_fid,
    }

    if args.per_class:
        class_fids = {}
        real_classes = sorted([p.name for p in real_root.iterdir() if p.is_dir()])
        fake_classes = sorted([p.name for p in fake_root.iterdir() if p.is_dir()])
        common_classes = sorted(set(real_classes) & set(fake_classes))
        for class_name in common_classes:
            rc = choose_paths(list_images(real_root / class_name), args.max_images, args.seed)
            fc = choose_paths(list_images(fake_root / class_name), args.max_images, args.seed)
            class_fids[class_name] = compute_fid_for_pair(
                real_paths=rc,
                fake_paths=fc,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=num_workers,
                img_size=args.img_size,
            )
        output["per_class_fid"] = class_fids

    print(json.dumps(output, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()
