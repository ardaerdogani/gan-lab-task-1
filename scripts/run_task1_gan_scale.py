from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Task 1 GAN scaling experiments and collect FID by data amount.")
    parser.add_argument("--counts", type=int, nargs="+", default=[200, 400, 800, 1600])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--num-per-class", type=int, default=400)
    parser.add_argument("--weights-path", type=str, required=True, help="Inception v3 weights (.pth) for FID")
    parser.add_argument("--python-bin", type=str, default=".venv/bin/python")
    parser.add_argument("--max-images", type=int, default=3000)
    parser.add_argument("--out-csv", type=str, default="runs_gan/task1_fid_by_count.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    python_bin = str((ROOT / args.python_bin).resolve()) if not args.python_bin.startswith("/") else args.python_bin
    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for count in args.counts:
        run_tag = f"count_{count}"
        gan_out_dir = ROOT / "runs_gan" / f"task1_{run_tag}"
        synth_out_dir = ROOT / "data" / f"synthetic_task1_{run_tag}"
        gan_out_dir.mkdir(parents=True, exist_ok=True)
        synth_out_dir.mkdir(parents=True, exist_ok=True)

        run_cmd(
            [
                python_bin,
                "train_gan.py",
                "--subset-count",
                str(count),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--sample-every",
                str(args.sample_every),
                "--checkpoint-every",
                str(args.checkpoint_every),
                "--num-workers",
                "0",
                "--out-dir",
                str(gan_out_dir),
            ]
        )

        ckpt_epoch = args.epochs - (args.epochs % args.checkpoint_every)
        if ckpt_epoch == 0:
            ckpt_epoch = args.epochs
        ckpt_path = gan_out_dir / f"ckpt_epoch_{ckpt_epoch:03d}.pt"
        if not ckpt_path.exists():
            ckpts = sorted(gan_out_dir.glob("ckpt_epoch_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"Checkpoint not found in {gan_out_dir}")
            ckpt_path = ckpts[-1]

        run_cmd(
            [
                python_bin,
                "generate_synthetic.py",
                "--ckpt-path",
                str(ckpt_path),
                "--out-root",
                str(synth_out_dir),
                "--num-per-class",
                str(args.num_per_class),
                "--batch-size",
                str(args.batch_size),
            ]
        )

        fid_json = ROOT / "runs_gan" / f"task1_{run_tag}_fid.json"
        run_cmd(
            [
                python_bin,
                "scripts/compute_fid.py",
                "--real-dir",
                "data/split/train",
                "--fake-dir",
                str(synth_out_dir),
                "--weights-path",
                str(args.weights_path),
                "--max-images",
                str(args.max_images),
                "--num-workers",
                "0",
                "--out-json",
                str(fid_json),
            ]
        )

        fid_data = json.loads(fid_json.read_text(encoding="utf-8"))
        rows.append(
            {
                "real_train_count": count,
                "gan_ckpt": str(ckpt_path.relative_to(ROOT)),
                "synthetic_dir": str(synth_out_dir.relative_to(ROOT)),
                "overall_fid": float(fid_data["overall_fid"]),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["real_train_count", "gan_ckpt", "synthetic_dir", "overall_fid"])
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out_csv.resolve())


if __name__ == "__main__":
    main()
