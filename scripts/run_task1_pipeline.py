"""
Run the corrected Task 1 pipeline end-to-end.

For each data size N (images per class), this script:
1. Builds a deterministic real training subset with N images per class.
2. Trains a GAN on that subset.
3. Generates an N-per-class synthetic pool from the best-FID checkpoint.
4. Trains classifier baselines on the matching real/synthetic budget.
5. Records pipeline costs including GAN training and synthesis time.

Usage:
    python scripts/run_task1_pipeline.py
    python scripts/run_task1_pipeline.py --sizes 100 200 --include_real_aug
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from train_classifier import run as run_classifier
from train_gan import train as run_gan

from create_task1_splits import DEFAULT_SIZES, build_split
from generate_synth import generate_synth_pool


def scenario_time_breakdown(scenario: str, gan_summary: dict, synth_summary: dict) -> dict:
    if scenario not in {"synth", "both"}:
        return {}
    return {
        "gan_train_time_sec": gan_summary.get("train_time_sec", 0.0),
        "synth_generation_time_sec": synth_summary.get("generate_time_sec", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--train_root", type=Path, default=Path("data_final/train"))
    parser.add_argument("--fid_root", type=Path, default=Path("data_final"))
    parser.add_argument("--test_root", type=Path, default=Path("data_final/test"))
    parser.add_argument("--split_root", type=Path, default=Path("data_splits/task1"))
    parser.add_argument("--out_root", type=Path, default=Path("runs/task1"))
    parser.add_argument("--split_mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--overwrite_splits", action="store_true")
    parser.add_argument("--include_real_aug", action="store_true",
                        help="Include the optional real-only classical augmentation baseline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synth_batch_size", type=int, default=64)
    args = parser.parse_args()

    cfg = Config()
    clf_out_dir = args.out_root / "clf"
    scenarios = ["real", "synth", "both"] + (["real_aug"] if args.include_real_aug else [])

    all_results = []
    pipeline_summary = []

    for n in args.sizes:
        print(f"\n{'=' * 72}")
        print(f"Task 1 pipeline for n_per_class={n}")
        print(f"{'=' * 72}")

        split_summary = build_split(
            train_root=args.train_root,
            out_root=args.split_root,
            n_per_class=n,
            seed=args.seed,
            mode=args.split_mode,
            overwrite=args.overwrite_splits,
        )
        split_root = Path(split_summary["split_root"])
        real_train_root = split_root / "train"

        gan_out_dir = args.out_root / "gan" / f"n{n}"
        gan_run = run_gan(cfg, data_root=split_root, fid_root=args.fid_root, out_dir=gan_out_dir)
        gan_summary = gan_run["summary"]
        del gan_run
        gc.collect()

        best_ckpt = Path(gan_summary["out_dir"]) / "checkpoints" / "best_fid.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Missing best GAN checkpoint: {best_ckpt}")

        synth_out_dir = args.out_root / "synth" / f"n{n}"
        synth_summary = generate_synth_pool(
            ckpt=best_ckpt,
            n_per_class=n,
            out_dir=synth_out_dir,
            batch_size=args.synth_batch_size,
            seed=args.seed,
        )

        size_summary = {
            "n_per_class": n,
            "split_root": str(split_root),
            "gan": gan_summary,
            "synth": synth_summary,
            "classifier_results": [],
        }

        for scenario in scenarios:
            result = run_classifier(
                cfg,
                scenario,
                n,
                str(synth_out_dir),
                str(clf_out_dir),
                real_train_root=str(real_train_root),
                test_root=str(args.test_root),
                time_breakdown=scenario_time_breakdown(scenario, gan_summary, synth_summary),
                extra_metadata={
                    "generator_train_n_per_class": n,
                    "synthetic_pool_n_per_class": n,
                    "task1_fair_pipeline": True,
                },
            )
            all_results.append(result)
            size_summary["classifier_results"].append(result)

        pipeline_summary.append(size_summary)

    clf_out_dir.mkdir(parents=True, exist_ok=True)
    with open(clf_out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    args.out_root.mkdir(parents=True, exist_ok=True)
    with open(args.out_root / "pipeline_summary.json", "w") as f:
        json.dump(pipeline_summary, f, indent=2)

    print(f"\nSaved classifier results to {clf_out_dir / 'all_results.json'}")
    print(f"Saved pipeline summary to {args.out_root / 'pipeline_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
