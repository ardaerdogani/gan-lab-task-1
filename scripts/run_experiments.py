"""
Run the full data-size experiment grid.

Grid:
  sizes     = [100, 200, 400, 800, 1300]  (per class)
  scenarios = [real, synth, both] (+ real_aug if requested)

Outputs all results to runs/clf/ as JSON files, plus a combined summary.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --sizes 200 400 800
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from train_classifier import run


DEFAULT_SIZES = [100, 200, 400, 800, 1300]
DEFAULT_SCENARIOS = ["real", "synth", "both"]
OPTIONAL_SCENARIOS = ["real_aug"]


def time_value(result):
    return result.get("pipeline_time_sec", result.get("train_time_sec", 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--synth_dir", type=str, default="data_synth")
    parser.add_argument("--out_dir", type=str, default="runs/clf")
    parser.add_argument("--real_train_root", type=str, default=None)
    parser.add_argument("--test_root", type=str, default=None)
    parser.add_argument("--include_real_aug", action="store_true",
                        help="Also run the optional classical augmentation baseline.")
    parser.add_argument("--gan_train_time_sec", type=float, default=0.0)
    parser.add_argument("--synth_generation_time_sec", type=float, default=0.0)
    args = parser.parse_args()

    cfg = Config()
    all_results = []
    scenarios = DEFAULT_SCENARIOS + (OPTIONAL_SCENARIOS if args.include_real_aug else [])

    for n in args.sizes:
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"  Experiment: scenario={scenario}  n_per_class={n}")
            print(f"{'='*60}")
            time_breakdown = {}
            if scenario in {"synth", "both"}:
                time_breakdown = {
                    "gan_train_time_sec": args.gan_train_time_sec,
                    "synth_generation_time_sec": args.synth_generation_time_sec,
                }
            result = run(
                cfg,
                scenario,
                n,
                args.synth_dir,
                args.out_dir,
                real_train_root=args.real_train_root,
                test_root=args.test_root,
                time_breakdown=time_breakdown,
            )
            all_results.append(result)

    # save combined summary
    out_path = Path(args.out_dir)
    with open(out_path / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # print summary table
    headers = ["real", "synth", "both"] + (["real_aug"] if args.include_real_aug else [])
    acc_header = " | ".join(f"{h:>8}" for h in headers)
    time_header = " | ".join(f"{(h + '(s)')[:8]:>8}" for h in headers)
    table_header = f"{'Size':>6} | {acc_header} | {time_header}"
    print(f"\n{'='*len(table_header)}")
    print(table_header)
    print(f"{'-'*len(table_header)}")

    by_size = {}
    for r in all_results:
        key = r["n_per_class"]
        by_size.setdefault(key, {})[r["scenario"]] = r

    for n in args.sizes:
        row = by_size.get(n, {})
        accs = [f"{row.get(s, {}).get('test_accuracy', 0):.4f}" if s in row else "  N/A " for s in headers]
        times = [f"{time_value(row.get(s, {})):>7.1f}" if s in row else "  N/A " for s in headers]
        print(f"{n:>6} | " + " | ".join(f"{x:>8}" for x in accs + times))

    print(f"{'='*len(table_header)}")
    print(f"Results saved to {out_path / 'all_results.json'}")


if __name__ == "__main__":
    main()
