"""
Generate synthetic image pool from a trained CGAN generator.
Usage:
    python scripts/generate_synth.py --ckpt runs/gan/checkpoints/ckpt_epoch0100.pt \
                                      --n_per_class 500 \
                                      --out_dir data_synth
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

# allow running from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from models.gan import Generator

CLASS_NAMES = ["apple", "banana", "orange"]


def generate_synth_pool(
    ckpt: str | Path,
    n_per_class: int,
    out_dir: str | Path,
    batch_size: int = 64,
    seed: int = 42,
    class_names: list[str] | None = None,
):
    ckpt = Path(ckpt)
    out_root = Path(out_dir)
    class_names = class_names or CLASS_NAMES
    cfg = Config()
    device = torch.device(cfg.device if torch.backends.mps.is_available() else "cpu")
    num_classes = len(class_names)

    G = Generator(z_dim=cfg.z_dim, num_classes=num_classes).to(device)
    ckpt_state = torch.load(ckpt, map_location=device, weights_only=True)
    G.load_state_dict(ckpt_state["G"])
    G.eval()

    torch.manual_seed(seed)
    start_time = time.time()
    counts = {}

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = out_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        while generated < n_per_class:
            bs = min(batch_size, n_per_class - generated)
            z = torch.randn(bs, cfg.z_dim, device=device)
            y = torch.full((bs,), cls_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                imgs = G(z, y)

            for i in range(bs):
                save_image(
                    imgs[i],
                    cls_dir / f"{cls_name}_synth_{generated + i:05d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )
            generated += bs

        print(f"{cls_name}: {generated} images saved to {cls_dir}")
        counts[cls_name] = generated

    summary = {
        "checkpoint": str(ckpt),
        "out_dir": str(out_root),
        "n_per_class": n_per_class,
        "counts": counts,
        "seed": seed,
        "generate_time_sec": round(time.time() - start_time, 1),
    }
    with open(out_root / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Summary saved to {out_root / 'generation_summary.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--n_per_class", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="data_synth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_synth_pool(
        ckpt=args.ckpt,
        n_per_class=args.n_per_class,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
