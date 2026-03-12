"""
Create deterministic class-balanced training subsets for the corrected Task 1 pipeline.

Each output split contains only a train/ directory with exactly N images per class:
  data_splits/task1/n100/train/apple/...
  data_splits/task1/n100/train/banana/...
  data_splits/task1/n100/train/orange/...

Usage:
    python scripts/create_task1_splits.py
    python scripts/create_task1_splits.py --n_per_class 100 200 400 --mode symlink --overwrite
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


DEFAULT_SIZES = [100, 200, 400, 800, 1300]


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def select_files(class_dir: Path, n_per_class: int, seed: int) -> list[Path]:
    files = sorted([p for p in class_dir.iterdir() if p.is_file()])
    if len(files) < n_per_class:
        raise ValueError(f"{class_dir} has only {len(files)} files, need {n_per_class}.")
    rng = random.Random(f"{class_dir.name}:{seed}")
    rng.shuffle(files)
    return sorted(files[:n_per_class], key=lambda p: p.name)


def build_split(
    train_root: Path,
    out_root: Path,
    n_per_class: int,
    seed: int,
    mode: str,
    overwrite: bool,
) -> dict:
    split_root = out_root / f"n{n_per_class}" / "train"
    expected_total = 0

    if overwrite and split_root.parent.exists():
        shutil.rmtree(split_root.parent)

    if split_root.parent.exists():
        existing = sum(1 for p in split_root.rglob("*") if p.is_file())
        class_count = len([p for p in train_root.iterdir() if p.is_dir()])
        if existing == n_per_class * class_count:
            return {
                "split_root": str(split_root.parent),
                "n_per_class": n_per_class,
                "total_files": existing,
                "mode": mode,
                "reused": True,
            }

    class_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for class_dir in class_dirs:
        selected = select_files(class_dir, n_per_class, seed)
        dst_dir = split_root / class_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in selected:
            materialize_file(src, dst_dir / src.name, mode)
        expected_total += len(selected)

    return {
        "split_root": str(split_root.parent),
        "n_per_class": n_per_class,
        "total_files": expected_total,
        "mode": mode,
        "reused": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=Path, default=Path("data_final/train"))
    parser.add_argument("--out_root", type=Path, default=Path("data_splits/task1"))
    parser.add_argument("--n_per_class", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summaries = []
    for n in args.n_per_class:
        summary = build_split(
            train_root=args.train_root,
            out_root=args.out_root,
            n_per_class=n,
            seed=args.seed,
            mode=args.mode,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        state = "reused" if summary["reused"] else "created"
        print(f"[{state}] n={n} -> {summary['split_root']}")

    print("\nSummary:")
    for summary in summaries:
        print(
            f"  n={summary['n_per_class']:>4}  total_files={summary['total_files']:>5}  "
            f"mode={summary['mode']:<8} root={summary['split_root']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
