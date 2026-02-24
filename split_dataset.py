from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_CLASSES = ["apple", "orange", "banana"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test split from raw fruit folders.")
    parser.add_argument("--raw-root", type=str, default="data/raw")
    parser.add_argument("--out-root", type=str, default="data/split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--recursive", action="store_true", help="Scan source class directories recursively.")
    parser.add_argument("--clear-out", action="store_true", help="Delete output root before writing new split.")
    parser.add_argument(
        "--write-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to materialize split files. hardlink/symlink save disk space.",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def resolve_source_dir(raw_root: Path, class_name: str) -> Path:
    # Keep backward compatibility with singular/plural class folder names.
    candidates = [class_name, f"{class_name}s"]
    for name in candidates:
        d = raw_root / name
        if d.is_dir():
            return d
    return raw_root / class_name


def list_images(source_dir: Path, recursive: bool) -> list[Path]:
    if not source_dir.is_dir():
        return []
    iterator = source_dir.rglob("*") if recursive else source_dir.glob("*")
    return [p for p in iterator if p.is_file() and is_image(p)]


def materialize_file(src: Path, dst: Path, write_mode: str) -> None:
    if write_mode == "copy":
        shutil.copy2(src, dst)
        return

    if write_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # Cross-device or unsupported FS fallback.
            shutil.copy2(src, dst)
            return

    if write_mode == "symlink":
        dst.symlink_to(src.resolve())
        return

    raise ValueError(f"Unsupported write mode: {write_mode}")


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise SystemExit(
            f"ERROR: train/val/test ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"({args.train_ratio}, {args.val_ratio}, {args.test_ratio})"
        )

    if not raw_root.is_dir():
        raise SystemExit(f"ERROR: raw root not found: {raw_root}")

    if args.clear_out and out_root.exists():
        shutil.rmtree(out_root)

    rng = random.Random(args.seed)
    splits = [
        ("train", args.train_ratio),
        ("val", args.val_ratio),
        ("test", args.test_ratio),
    ]

    for split_name, _ in splits:
        for class_name in DEFAULT_CLASSES:
            (out_root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    total_written = 0
    for class_name in DEFAULT_CLASSES:
        source_dir = resolve_source_dir(raw_root, class_name)
        images = list_images(source_dir, recursive=args.recursive)
        images.sort()
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val :]

        for p in train_imgs:
            materialize_file(p, out_root / "train" / class_name / p.name, args.write_mode)
        for p in val_imgs:
            materialize_file(p, out_root / "val" / class_name / p.name, args.write_mode)
        for p in test_imgs:
            materialize_file(p, out_root / "test" / class_name / p.name, args.write_mode)

        class_written = len(train_imgs) + len(val_imgs) + len(test_imgs)
        total_written += class_written
        print(
            class_name,
            "source:",
            source_dir,
            "total:",
            n,
            "train:",
            len(train_imgs),
            "val:",
            len(val_imgs),
            "test:",
            len(test_imgs),
            "write_mode:",
            args.write_mode,
        )

    print("Done:", out_root)
    print("Total written:", total_written)


if __name__ == "__main__":
    main()
