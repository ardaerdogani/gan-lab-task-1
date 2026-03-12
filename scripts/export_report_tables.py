"""
Export CSV tables for the thesis-style report.

This script is intentionally stdlib-only so it can run in minimal Python
environments without requiring ML dependencies.

Inputs (defaults):
  - Real dataset folder:          data_final/{train,val,test}/{class}/*
  - GAN training log (FID):       runs/gan/train_log.json
  - Classifier experiment grid:   runs/clf/all_results.json

Outputs (defaults):
  - reports/tables/dataset_summary.csv
  - reports/tables/gan_fid_by_epoch.csv
  - reports/tables/clf_results.csv

Usage:
  python scripts/export_report_tables.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


SCENARIO_ORDER = {"real": 0, "real_aug": 1, "synth": 2, "both": 3}
DEFAULT_SPLITS = ("train", "val", "test")


def iter_files_count(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for p in dir_path.rglob("*") if p.is_file())


def export_dataset_summary(data_root: Path, out_csv: Path, splits: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        for class_dir in class_dirs:
            rows.append(
                {"split": split, "class": class_dir.name, "n_images": iter_files_count(class_dir)}
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "class", "n_images"])
        w.writeheader()
        w.writerows(rows)

    return rows


def export_gan_fid(train_log_json: Path, out_csv: Path) -> list[dict]:
    with train_log_json.open() as f:
        log = json.load(f)

    rows: list[dict] = []
    for item in log:
        if "fid" not in item:
            continue
        rows.append({"epoch": int(item["epoch"]), "fid": float(item["fid"])})
    rows.sort(key=lambda r: r["epoch"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "fid"])
        w.writeheader()
        w.writerows(rows)

    return rows


def _n_per_class_key(v):
    if isinstance(v, int):
        return (0, v)
    try:
        return (0, int(v))
    except Exception:
        return (1, str(v))


def export_clf_results(all_results_json: Path, out_csv: Path) -> list[dict]:
    with all_results_json.open() as f:
        results = json.load(f)

    rows: list[dict] = []
    for r in results:
        per_class = r.get("per_class", {}) or {}
        rows.append(
            {
                "n_per_class": r.get("n_per_class"),
                "scenario": r.get("scenario"),
                "augmentation_policy": r.get("augmentation_policy"),
                "train_size": r.get("train_size"),
                "test_accuracy": r.get("test_accuracy"),
                "train_time_sec": r.get("train_time_sec"),
                "classifier_train_time_sec": r.get("classifier_train_time_sec", r.get("train_time_sec")),
                "gan_train_time_sec": r.get("gan_train_time_sec", 0.0),
                "synth_generation_time_sec": r.get("synth_generation_time_sec", 0.0),
                "pipeline_time_sec": r.get("pipeline_time_sec", r.get("train_time_sec")),
                "apple_f1": (per_class.get("apple", {}) or {}).get("f1"),
                "banana_f1": (per_class.get("banana", {}) or {}).get("f1"),
                "orange_f1": (per_class.get("orange", {}) or {}).get("f1"),
            }
        )

    rows.sort(
        key=lambda r: (
            _n_per_class_key(r["n_per_class"]),
            SCENARIO_ORDER.get(r["scenario"], 999),
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n_per_class",
                "scenario",
                "augmentation_policy",
                "train_size",
                "test_accuracy",
                "train_time_sec",
                "classifier_train_time_sec",
                "gan_train_time_sec",
                "synth_generation_time_sec",
                "pipeline_time_sec",
                "apple_f1",
                "banana_f1",
                "orange_f1",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    return rows


def print_quick_summary(dataset_rows: list[dict], fid_rows: list[dict], clf_rows: list[dict]) -> None:
    if dataset_rows:
        print("\nDataset summary (n_images):")
        by_split = {}
        for r in dataset_rows:
            by_split.setdefault(r["split"], []).append(r)
        for split, items in by_split.items():
            total = sum(int(x["n_images"]) for x in items)
            parts = ", ".join(f"{x['class']}={x['n_images']}" for x in items)
            print(f"  {split:>5}: total={total}  ({parts})")

    if fid_rows:
        best = min(fid_rows, key=lambda r: r["fid"])
        last = fid_rows[-1]
        print("\nGAN FID:")
        print(f"  best:  epoch={best['epoch']}  fid={best['fid']:.4f}")
        print(f"  last:  epoch={last['epoch']}  fid={last['fid']:.4f}")

    if clf_rows:
        # Print a compact accuracy table by size (rows) x scenario (cols)
        print("\nClassifier test accuracy (fraction):")
        sizes = sorted({r["n_per_class"] for r in clf_rows}, key=_n_per_class_key)
        lookup = {(r["scenario"], r["n_per_class"]): r for r in clf_rows}
        scenarios = [s for s in ["real", "real_aug", "synth", "both"] if any(r["scenario"] == s for r in clf_rows)]
        header = f"{'Size':>6} | " + " | ".join(f"{s:>7}" for s in scenarios)
        print("  " + header)
        print("  " + "-" * len(header))
        for n in sizes:
            vals = []
            for s in scenarios:
                v = lookup.get((s, n), {}).get("test_accuracy")
                vals.append(f"{float(v):7.4f}" if v is not None else "   N/A ")
            print(f"  {str(n):>6} | " + " | ".join(vals))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data_final"))
    parser.add_argument("--gan_log", type=Path, default=Path("runs/gan/train_log.json"))
    parser.add_argument("--clf_results", type=Path, default=Path("runs/clf/all_results.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()

    dataset_csv = args.out_dir / "dataset_summary.csv"
    gan_fid_csv = args.out_dir / "gan_fid_by_epoch.csv"
    clf_csv = args.out_dir / "clf_results.csv"

    dataset_rows = export_dataset_summary(args.data_root, dataset_csv, DEFAULT_SPLITS)
    fid_rows = export_gan_fid(args.gan_log, gan_fid_csv)
    clf_rows = export_clf_results(args.clf_results, clf_csv)

    print(f"\nWrote: {dataset_csv}")
    print(f"Wrote: {gan_fid_csv}")
    print(f"Wrote: {clf_csv}")
    print_quick_summary(dataset_rows, fid_rows, clf_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
