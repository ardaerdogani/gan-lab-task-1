from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "runs_classifier" / "amount_vs_accuracy_time_balanced_cpu.csv"
OUT_DIR = ROOT / "reports" / "figures"

# Historical baseline values (v1: earlier smaller dataset run from repo history)
V1_MAIN = {
    "synth_only": {"accuracy": 0.9039, "macro_f1": 0.9045, "train_time_s": 7.77},
    "real_only_100": {"accuracy": 0.9476, "macro_f1": 0.9462, "train_time_s": 13.24},
    "real_plus_synth_100": {"accuracy": 0.8690, "macro_f1": 0.8650, "train_time_s": 17.61},
    "real_only_aug_100": {"accuracy": 0.9301, "macro_f1": 0.9303, "train_time_s": 12.71},
}

V1_RATIO = {
    "10%": {"real_only": 0.7904, "real_plus_synth": 0.8297},
    "25%": {"real_only": 0.8515, "real_plus_synth": 0.9083},
    "50%": {"real_only": 0.8821, "real_plus_synth": 0.8996},
    "100%": {"real_only": 0.9476, "real_plus_synth": 0.8690},
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in rows:
        keyed[(r["ratio"], r["scenario"], r["augmentation"])] = r
    return keyed


def style():
    plt.style.use("seaborn-v0_8-whitegrid")


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def scenario_line_plots(rows: list[dict[str, str]]) -> None:
    ratio_labels = ["10%", "25%", "50%", "100%"]
    x = [10, 25, 50, 100]
    key = rows_by_key(rows)

    real_acc = [float(key[(r, "real_only", "no")]["accuracy"]) for r in ratio_labels]
    mix_acc = [float(key[(r, "real_plus_synth", "no")]["accuracy"]) for r in ratio_labels]
    aug_acc = [float(key[(r, "real_only", "yes")]["accuracy"]) for r in ratio_labels]
    synth_acc = float(key[("fixed", "synth_only", "no")]["accuracy"])

    real_f1 = [float(key[(r, "real_only", "no")]["macro_f1"]) for r in ratio_labels]
    mix_f1 = [float(key[(r, "real_plus_synth", "no")]["macro_f1"]) for r in ratio_labels]
    aug_f1 = [float(key[(r, "real_only", "yes")]["macro_f1"]) for r in ratio_labels]
    synth_f1 = float(key[("fixed", "synth_only", "no")]["macro_f1"])

    real_t = [float(key[(r, "real_only", "no")]["train_time_s"]) for r in ratio_labels]
    mix_t = [float(key[(r, "real_plus_synth", "no")]["train_time_s"]) for r in ratio_labels]
    aug_t = [float(key[(r, "real_only", "yes")]["train_time_s"]) for r in ratio_labels]
    synth_t = float(key[("fixed", "synth_only", "no")]["train_time_s"])

    # Accuracy
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, real_acc, marker="o", label="Real-only")
    ax.plot(x, mix_acc, marker="o", label="Real+Synth")
    ax.plot(x, aug_acc, marker="o", label="Real-only + Classic Aug")
    ax.axhline(synth_acc, linestyle="--", linewidth=1.5, label=f"Synth-only ({synth_acc:.4f})")
    ax.set_title("Accuracy vs Real Data Ratio (CPU Balanced)")
    ax.set_xlabel("Real Data Ratio (%)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, ratio_labels)
    ax.set_ylim(0.82, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "accuracy_vs_ratio_cpu_balanced.png")

    # Macro F1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, real_f1, marker="o", label="Real-only")
    ax.plot(x, mix_f1, marker="o", label="Real+Synth")
    ax.plot(x, aug_f1, marker="o", label="Real-only + Classic Aug")
    ax.axhline(synth_f1, linestyle="--", linewidth=1.5, label=f"Synth-only ({synth_f1:.4f})")
    ax.set_title("Macro F1 vs Real Data Ratio (CPU Balanced)")
    ax.set_xlabel("Real Data Ratio (%)")
    ax.set_ylabel("Macro F1")
    ax.set_xticks(x, ratio_labels)
    ax.set_ylim(0.70, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "macrof1_vs_ratio_cpu_balanced.png")

    # Train time
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, real_t, marker="o", label="Real-only")
    ax.plot(x, mix_t, marker="o", label="Real+Synth")
    ax.plot(x, aug_t, marker="o", label="Real-only + Classic Aug")
    ax.axhline(synth_t, linestyle="--", linewidth=1.5, label=f"Synth-only ({synth_t:.2f}s)")
    ax.set_title("Training Time vs Real Data Ratio (CPU Balanced)")
    ax.set_xlabel("Real Data Ratio (%)")
    ax.set_ylabel("Train Time (s)")
    ax.set_xticks(x, ratio_labels)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "time_vs_ratio_cpu_balanced.png")


def historical_comparison_plots(rows: list[dict[str, str]]) -> None:
    key = rows_by_key(rows)

    # Main 100% scenario grouped bars
    categories = ["Synth-only", "Real-only 100%", "Real+Synth 100%", "Real+Aug 100%"]
    v1_acc = [
        V1_MAIN["synth_only"]["accuracy"],
        V1_MAIN["real_only_100"]["accuracy"],
        V1_MAIN["real_plus_synth_100"]["accuracy"],
        V1_MAIN["real_only_aug_100"]["accuracy"],
    ]
    v2_acc = [
        float(key[("fixed", "synth_only", "no")]["accuracy"]),
        float(key[("100%", "real_only", "no")]["accuracy"]),
        float(key[("100%", "real_plus_synth", "no")]["accuracy"]),
        float(key[("100%", "real_only", "yes")]["accuracy"]),
    ]

    v1_f1 = [
        V1_MAIN["synth_only"]["macro_f1"],
        V1_MAIN["real_only_100"]["macro_f1"],
        V1_MAIN["real_plus_synth_100"]["macro_f1"],
        V1_MAIN["real_only_aug_100"]["macro_f1"],
    ]
    v2_f1 = [
        float(key[("fixed", "synth_only", "no")]["macro_f1"]),
        float(key[("100%", "real_only", "no")]["macro_f1"]),
        float(key[("100%", "real_plus_synth", "no")]["macro_f1"]),
        float(key[("100%", "real_only", "yes")]["macro_f1"]),
    ]

    x = list(range(len(categories)))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    axes[0].bar([i - width / 2 for i in x], v1_acc, width=width, label="v1 (smaller data)")
    axes[0].bar([i + width / 2 for i in x], v2_acc, width=width, label="v2 (expanded data)")
    axes[0].set_title("Accuracy: v1 vs v2 (Main 100% Scenarios)")
    axes[0].set_xticks(x, categories, rotation=20, ha="right")
    axes[0].set_ylim(0.84, 1.0)

    axes[1].bar([i - width / 2 for i in x], v1_f1, width=width, label="v1 (smaller data)")
    axes[1].bar([i + width / 2 for i in x], v2_f1, width=width, label="v2 (expanded data)")
    axes[1].set_title("Macro F1: v1 vs v2 (Main 100% Scenarios)")
    axes[1].set_xticks(x, categories, rotation=20, ha="right")
    axes[1].set_ylim(0.84, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    save(fig, "historical_v1_v2_main_100_comparison.png")

    # Ratio-level real vs mix accuracy (v1 and v2)
    ratio_labels = ["10%", "25%", "50%", "100%"]
    xr = [10, 25, 50, 100]
    v1_real = [V1_RATIO[r]["real_only"] for r in ratio_labels]
    v1_mix = [V1_RATIO[r]["real_plus_synth"] for r in ratio_labels]
    v2_real = [float(key[(r, "real_only", "no")]["accuracy"]) for r in ratio_labels]
    v2_mix = [float(key[(r, "real_plus_synth", "no")]["accuracy"]) for r in ratio_labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xr, v1_real, marker="o", linestyle="--", label="v1 Real-only")
    ax.plot(xr, v1_mix, marker="o", linestyle="--", label="v1 Real+Synth")
    ax.plot(xr, v2_real, marker="o", linestyle="-", label="v2 Real-only")
    ax.plot(xr, v2_mix, marker="o", linestyle="-", label="v2 Real+Synth")
    ax.set_title("Historical Accuracy Comparison by Ratio (v1 vs v2)")
    ax.set_xlabel("Real Data Ratio (%)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(xr, ratio_labels)
    ax.set_ylim(0.75, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "historical_v1_v2_ratio_accuracy.png")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    style()
    rows = load_rows(CSV_PATH)
    scenario_line_plots(rows)
    historical_comparison_plots(rows)

    print("Saved figures to:", OUT_DIR)
    for p in sorted(OUT_DIR.glob("*.png")):
        print("-", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
