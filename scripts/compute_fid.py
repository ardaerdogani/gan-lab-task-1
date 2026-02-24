from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fid_core import (
    InceptionFeatureExtractor,
    choose_paths,
    compute_fid_for_path_pair,
    list_images,
)
from utils import default_num_workers, get_best_device


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID between real and synthetic image folders.")
    parser.add_argument("--real-dir", type=str, default="data/split/train")
    parser.add_argument("--fake-dir", type=str, default="data/synthetic/task1_main")
    parser.add_argument("--weights-path", type=str, default=None, help="Local inception_v3 weights path (.pth)")
    parser.add_argument("--img-size", type=int, default=299)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 => auto")
    parser.add_argument("--max-images", type=int, default=None, help="Sample up to N images from each domain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", action="store_true", help="Also compute class-wise FID")
    parser.add_argument("--out-json", type=str, default="reports/fid_task1.json")
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
    overall_fid = compute_fid_for_path_pair(
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
            class_fids[class_name] = compute_fid_for_path_pair(
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
