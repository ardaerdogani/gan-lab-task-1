from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_ratio_numeric(value: str):
    if value is None:
        return None
    value = value.strip()
    if not value or value == "fixed":
        return None
    if value.endswith("%"):
        value = value[:-1]
    try:
        return float(value)
    except ValueError:
        return None


def pick_row(rows, ratio, scenario, augmentation="no"):
    for r in rows:
        if r.get("ratio") == ratio and r.get("scenario") == scenario and r.get("augmentation") == augmentation:
            return r

    # Backward-compatible fallback:
    # when classifier CSV was produced with --counts, "100%" may not exist.
    if ratio == "100%":
        candidates = [
            r
            for r in rows
            if r.get("scenario") == scenario and r.get("augmentation") == augmentation and r.get("ratio") != "fixed"
        ]
        scored = []
        for r in candidates:
            score = _parse_ratio_numeric(r.get("ratio", ""))
            if score is not None:
                scored.append((score, r))
        if scored:
            return max(scored, key=lambda x: x[0])[1]

    raise ValueError(f"Row not found: ratio={ratio}, scenario={scenario}, augmentation={augmentation}")


def build_three_case(rows, ratio):
    synth = pick_row(rows, ratio="fixed", scenario="synth_only", augmentation="no")
    real = pick_row(rows, ratio=ratio, scenario="real_only", augmentation="no")
    mixed = pick_row(rows, ratio=ratio, scenario="real_plus_synth", augmentation="no")
    selected = [synth, real, mixed]
    scenarios = [row["scenario"] for row in selected]
    if len(selected) != 3 or sorted(scenarios) != ["real_only", "real_plus_synth", "synth_only"]:
        raise ValueError("Three-case table must contain exactly: synth_only, real_only, real_plus_synth.")
    return selected


def write_markdown(out_md: Path, ratio: str, selected_rows):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Task 2 - Three-Case Comparison")
    lines.append("")
    lines.append(f"Selected real-data setting: `{ratio}`")
    lines.append("")
    lines.append("| Case | Accuracy | Macro F1 | Train Time (s) |")
    lines.append("|---|---:|---:|---:|")
    labels = {
        "synth_only": "Only Synthetic",
        "real_only": "Only Real",
        "real_plus_synth": "Real + Synthetic",
    }
    for row in selected_rows:
        lines.append(
            f"| {labels[row['scenario']]} | {float(row['accuracy']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(row['train_time_s']):.2f} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_plot(out_png: Path, selected_rows, ratio: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Only Synthetic", "Only Real", "Real + Synthetic"]
    acc = [float(selected_rows[0]["accuracy"]), float(selected_rows[1]["accuracy"]), float(selected_rows[2]["accuracy"])]
    f1 = [float(selected_rows[0]["macro_f1"]), float(selected_rows[1]["macro_f1"]), float(selected_rows[2]["macro_f1"])]

    x = [0, 1, 2]
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width / 2 for i in x], acc, width=width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], f1, width=width, label="Macro F1")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Task 2 Three-Case Classification Comparison ({ratio})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Create Task 2 three-case report from classifier CSV.")
    parser.add_argument("--csv-path", type=str, default="runs_classifier/task1_counts_three_case_cpu.csv")
    parser.add_argument("--ratio", type=str, default="1600", help="Real-data setting to use (e.g. 1600)")
    parser.add_argument("--out-md", type=str, default="reports/task2_three_case_comparison.md")
    parser.add_argument("--out-png", type=str, default="reports/figures/task2_three_case_comparison.png")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(Path(args.csv_path))
    selected = build_three_case(rows=rows, ratio=args.ratio)
    write_markdown(out_md=Path(args.out_md), ratio=args.ratio, selected_rows=selected)
    write_plot(out_png=Path(args.out_png), selected_rows=selected, ratio=args.ratio)
    print("Saved markdown:", Path(args.out_md).resolve())
    print("Saved figure:", Path(args.out_png).resolve())


if __name__ == "__main__":
    main()
