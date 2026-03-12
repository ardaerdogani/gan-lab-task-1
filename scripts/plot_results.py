"""
Generate plots and tables from experiment results.

Reads runs/clf/all_results.json and produces:
  - Accuracy vs Data Size line plot
  - Training Time vs Data Size plot
  - Summary table printed to console

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results runs/clf/all_results.json --out_dir runs/clf/plots
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENARIO_STYLE = {
    "real":  {"color": "#2196F3", "marker": "o", "label": "Real only"},
    "real_aug": {"color": "#8E24AA", "marker": "D", "label": "Real + classical aug"},
    "synth": {"color": "#FF9800", "marker": "s", "label": "Synth only"},
    "both":  {"color": "#4CAF50", "marker": "^", "label": "Real + Synth"},
}
SCENARIO_ORDER = ["real", "real_aug", "synth", "both"]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def group_by_scenario(results):
    grouped = {}
    for r in results:
        grouped.setdefault(r["scenario"], []).append(r)
    for v in grouped.values():
        v.sort(key=lambda x: x["n_per_class"] if isinstance(x["n_per_class"], int) else 99999)
    return {k: grouped[k] for k in SCENARIO_ORDER if k in grouped}


def resolve_time_field(results, requested):
    if requested != "auto":
        return requested
    if any(r.get("pipeline_time_sec", r.get("train_time_sec", 0.0)) != r.get("train_time_sec", 0.0) for r in results):
        return "pipeline_time_sec"
    return "train_time_sec"


def time_value(result, time_field):
    if time_field == "pipeline_time_sec":
        return result.get("pipeline_time_sec", result.get("train_time_sec", 0.0))
    if time_field == "classifier_train_time_sec":
        return result.get("classifier_train_time_sec", result.get("train_time_sec", 0.0))
    return result.get(time_field, result.get("train_time_sec", 0.0))


def plot_accuracy(grouped, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario, runs in grouped.items():
        style = SCENARIO_STYLE[scenario]
        sizes = [r["n_per_class"] for r in runs]
        accs = [r["test_accuracy"] * 100 for r in runs]
        ax.plot(sizes, accs, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8)

    ax.set_xlabel("Training images per class", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy vs Training Data Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([r["n_per_class"] for r in list(grouped.values())[0]])
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_size.png", dpi=150)
    print(f"Saved: {out_dir / 'accuracy_vs_size.png'}")
    plt.close(fig)


def plot_time(grouped, out_dir, time_field):
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario, runs in grouped.items():
        style = SCENARIO_STYLE[scenario]
        sizes = [r["n_per_class"] for r in runs]
        times = [time_value(r, time_field) for r in runs]
        ax.plot(sizes, times, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8)

    ax.set_xlabel("Training images per class", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    title = "Pipeline Cost vs Training Data Size" if time_field == "pipeline_time_sec" else "Training Time vs Data Size"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([r["n_per_class"] for r in list(grouped.values())[0]])
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_size.png", dpi=150)
    print(f"Saved: {out_dir / 'time_vs_size.png'}")
    plt.close(fig)


def plot_per_class_f1(grouped, out_dir):
    """Bar chart: per-class F1 at the largest data size for each scenario."""
    n_cols = len(grouped)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]
    for ax, (scenario, runs) in zip(axes, grouped.items()):
        r = runs[-1]  # largest size
        classes = list(r["per_class"].keys())
        f1s = [r["per_class"][c]["f1"] * 100 for c in classes]
        colors = ["#EF5350", "#FFEE58", "#FFA726"]
        ax.bar(classes, f1s, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{SCENARIO_STYLE[scenario]['label']}\n(n={r['n_per_class']}/class)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_ylabel("F1 Score (%)" if ax == axes[0] else "")
        for i, v in enumerate(f1s):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    fig.suptitle("Per-Class F1 Score (Largest Data Size)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_f1.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'per_class_f1.png'}")
    plt.close(fig)


def print_table(grouped, time_field):
    scenario_names = list(grouped.keys())
    sizes = sorted(set(r["n_per_class"] for runs in grouped.values() for r in runs))
    lookup = {(r["scenario"], r["n_per_class"]): r for runs in grouped.values() for r in runs}

    acc_header = f"{'Size':>6} | " + " | ".join(f"{SCENARIO_STYLE[s]['label'][:16]:>16}" for s in scenario_names)
    print("\n" + "=" * len(acc_header))
    print(acc_header)
    print("-" * len(acc_header))
    for n in sizes:
        vals = []
        for s in scenario_names:
            r = lookup.get((s, n))
            if r:
                vals.append(f"{r['test_accuracy']*100:>8.2f}%")
            else:
                vals.append("     N/A")
        row = [f"{n:>6}"] + [f"{v:>16}" for v in vals]
        print(" | ".join(row))
    print("=" * len(acc_header))

    time_header = f"{'Size':>6} | " + " | ".join(f"{(s + ' time')[:16]:>16}" for s in scenario_names)
    print("\n" + "=" * len(time_header))
    print(time_header)
    print("-" * len(time_header))
    for n in sizes:
        times = []
        for s in scenario_names:
            r = lookup.get((s, n))
            if r:
                times.append(f"{time_value(r, time_field):>7.1f}s")
            else:
                times.append("    N/A")
        row = [f"{n:>6}"] + [f"{t:>16}" for t in times]
        print(" | ".join(row))
    print("=" * len(time_header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="runs/clf/all_results.json")
    parser.add_argument("--out_dir", type=str, default="runs/clf/plots")
    parser.add_argument("--time_field", choices=["auto", "train_time_sec", "classifier_train_time_sec", "pipeline_time_sec"],
                        default="auto")
    args = parser.parse_args()

    results = load_results(args.results)
    grouped = group_by_scenario(results)
    time_field = resolve_time_field(results, args.time_field)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy(grouped, out_dir)
    plot_time(grouped, out_dir, time_field)
    plot_per_class_f1(grouped, out_dir)
    print_table(grouped, time_field)


if __name__ == "__main__":
    main()
