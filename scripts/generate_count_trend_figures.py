from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_rows(rows, scenario):
    return [
        r
        for r in rows
        if r["scenario"] == scenario and r["augmentation"] == "no" and r["ratio"] != "fixed"
    ]


def sort_by_real_count(rows):
    return sorted(rows, key=lambda r: int(float(r["real_train_samples"])))


def extract_xy(rows, metric):
    x = [int(float(r["real_train_samples"])) for r in rows]
    y = [float(r[metric]) for r in rows]
    return x, y


def save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_plots(rows, out_dir: Path):
    real = sort_by_real_count(filter_rows(rows, "real_only"))
    mix = sort_by_real_count(filter_rows(rows, "real_plus_synth"))
    synth_only = [r for r in rows if r["scenario"] == "synth_only" and r["augmentation"] == "no"]

    if len(real) == 0 or len(mix) == 0:
        raise ValueError("CSV icinde real_only ve real_plus_synth satirlari bulunamadi.")

    synth_acc = float(synth_only[0]["accuracy"]) if synth_only else None
    synth_f1 = float(synth_only[0]["macro_f1"]) if synth_only else None
    synth_t = float(synth_only[0]["train_time_s"]) if synth_only else None

    x_real, y_real_acc = extract_xy(real, "accuracy")
    x_mix, y_mix_acc = extract_xy(mix, "accuracy")
    _, y_real_f1 = extract_xy(real, "macro_f1")
    _, y_mix_f1 = extract_xy(mix, "macro_f1")
    _, y_real_t = extract_xy(real, "train_time_s")
    _, y_mix_t = extract_xy(mix, "train_time_s")

    fig, (ax_acc, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    ax_acc.plot(x_real, y_real_acc, marker="o", label="Only Real")
    ax_acc.plot(x_mix, y_mix_acc, marker="o", label="Real + Synthetic")
    if synth_acc is not None:
        ax_acc.axhline(synth_acc, linestyle="--", linewidth=1.2, label=f"Only Synthetic ({synth_acc:.4f})")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Training Size (real count)")
    ax_acc.set_ylabel("Score")
    ax_acc.set_xticks(x_real)
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.legend(loc="lower right")

    ax_f1.plot(x_real, y_real_f1, marker="o", label="Only Real")
    ax_f1.plot(x_mix, y_mix_f1, marker="o", label="Real + Synthetic")
    if synth_f1 is not None:
        ax_f1.axhline(synth_f1, linestyle="--", linewidth=1.2, label=f"Only Synthetic ({synth_f1:.4f})")
    ax_f1.set_title("Macro F1")
    ax_f1.set_xlabel("Training Size (real count)")
    ax_f1.set_ylabel("Score")
    ax_f1.set_xticks(x_real)
    ax_f1.set_ylim(0.0, 1.0)
    ax_f1.legend(loc="lower right")

    save(fig, out_dir / "performance_vs_training_size.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(x_real, y_real_t, marker="o", label="Only Real")
    ax.plot(x_mix, y_mix_t, marker="o", label="Real + Synthetic")
    if synth_t is not None:
        ax.axhline(synth_t, linestyle="--", linewidth=1.2, label=f"Only Synthetic ({synth_t:.2f}s)")
    ax.set_title("Training Time vs Training Size")
    ax.set_xlabel("Training Size (real count)")
    ax.set_ylabel("Train Time (s)")
    ax.set_xticks(x_real)
    ax.legend(loc="upper left")
    save(fig, out_dir / "training_time_vs_training_size.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate count-based trend plots for professor's Task 1 request.")
    parser.add_argument("--csv-path", type=str, default="runs_classifier/task1_counts_three_case_cpu.csv")
    parser.add_argument("--out-dir", type=str, default="reports/figures")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(Path(args.csv_path))
    make_plots(rows=rows, out_dir=Path(args.out_dir))
    print("Saved plots to:", Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
