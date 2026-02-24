from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gan_train.artifacts import prepare_fixed_grid, save_checkpoint, save_class_grids
from gan_train.augment import diff_augment, parse_diffaugment_policy
from gan_train.data import build_train_loader, extract_class_to_idx, extract_targets
from gan_train.defaults import (
    BATCH_SIZE,
    BETAS,
    CHANNELS,
    DATA_TRAIN,
    EPOCHS,
    FAKE_LABEL,
    IMG_SIZE,
    LR,
    OUT_DIR,
    REAL_LABEL,
    SEED,
    Z_DIM,
)
from gan_train.fid_monitor import build_fid_eval_state, evaluate_fid
from gan_train.optimization import grad_norm, gradient_penalty, sample_fake_labels
from gan_train.runtime import set_seed
from models_gan import Discriminator, Generator, weights_init
from utils import default_num_workers, get_best_device


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional GAN training")
    parser.add_argument("--data-train", type=str, default=DATA_TRAIN)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--z-dim", type=int, default=Z_DIM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lr-g", type=float, default=None)
    parser.add_argument("--lr-d", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=BETAS[0])
    parser.add_argument("--beta2", type=float, default=BETAS[1])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 => auto")
    parser.add_argument(
        "--subset-count",
        type=int,
        default=None,
        help="Task 1 trend icin egitimde kullanilacak real image sayisi (or: 200, 400, 800, 1600).",
    )
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)

    parser.add_argument("--gan-loss", choices=["bce", "hinge", "wgangp"], default="bce")
    parser.add_argument("--use-spectral-norm", action="store_true")
    parser.add_argument("--use-diffaugment", action="store_true")
    parser.add_argument("--diffaugment-policy", type=str, default="color,translation,cutout")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--uniform-fake-labels", action="store_true")
    parser.add_argument("--n-critic", type=int, default=1)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--real-label", type=float, default=REAL_LABEL)
    parser.add_argument("--fake-label", type=float, default=FAKE_LABEL)

    parser.add_argument("--fid-every", type=int, default=0, help="Evaluate FID every N epochs (0 disables).")
    parser.add_argument("--fid-eval-per-class", type=int, default=128)
    parser.add_argument("--fid-weights-path", type=str, default=None)
    parser.add_argument("--fid-device", type=str, default="same")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_critic <= 0:
        raise ValueError("--n-critic must be >= 1")

    set_seed(args.seed)

    device = get_best_device()
    num_workers = default_num_workers() if args.num_workers < 0 else args.num_workers
    if device.type in {"cuda", "mps"}:
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, train_loader = build_train_loader(
        data_train=args.data_train,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=num_workers,
        subset_count=args.subset_count,
        seed=args.seed,
        device=device,
        balanced_sampler=args.balanced_sampler,
    )

    class_to_idx = extract_class_to_idx(train_ds)
    num_classes = len(class_to_idx)
    class_targets = extract_targets(train_ds)
    class_counts = np.bincount(class_targets, minlength=num_classes).astype(np.int64)

    lr_g = args.lr_g if args.lr_g is not None else args.lr
    lr_d = args.lr_d if args.lr_d is not None else args.lr
    if args.gan_loss == "hinge" and args.lr_g is None and args.lr_d is None:
        lr_g = args.lr * 0.5
        lr_d = args.lr * 2.0

    print("Device:", device)
    print("num_workers:", num_workers)
    print("class_to_idx:", class_to_idx)
    print("effective_train_samples:", len(train_ds))
    print("effective_class_counts:", class_counts.tolist())
    print("gan_loss:", args.gan_loss)
    print("use_spectral_norm:", args.use_spectral_norm)
    print("use_diffaugment:", args.use_diffaugment, args.diffaugment_policy if args.use_diffaugment else "")
    print("balanced_sampler:", args.balanced_sampler)
    print("uniform_fake_labels:", args.uniform_fake_labels)
    print("n_critic:", args.n_critic)
    print("lr_g:", lr_g, "lr_d:", lr_d)

    generator = Generator(
        z_dim=args.z_dim,
        num_classes=num_classes,
        img_channels=CHANNELS,
        img_size=args.img_size,
    ).to(device)
    discriminator = Discriminator(
        num_classes=num_classes,
        img_channels=CHANNELS,
        img_size=args.img_size,
        use_spectral_norm=args.use_spectral_norm,
    ).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    bce_criterion = nn.BCEWithLogitsLoss()
    opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(args.beta1, args.beta2))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(args.beta1, args.beta2))
    diffaugment_ops = parse_diffaugment_policy(args.diffaugment_policy) if args.use_diffaugment else []

    fixed_z, fixed_labels = prepare_fixed_grid(
        num_classes=num_classes,
        z_dim=args.z_dim,
        device=device,
        per_class=36,
        seed=args.seed,
    )

    fid_state = build_fid_eval_state(
        args=args,
        class_to_idx=class_to_idx,
        z_dim=args.z_dim,
        train_device=device,
        num_workers=num_workers,
    )
    if fid_state.enabled:
        print(
            "FID monitor enabled:",
            f"every={args.fid_every}",
            f"per_class={args.fid_eval_per_class}",
            f"device={fid_state.device}",
        )

    metrics_path = out_dir / "metrics.csv"
    class_names_ordered = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    metric_fieldnames = [
        "epoch",
        "loss_d",
        "loss_g",
        "grad_norm_d",
        "grad_norm_g",
        "d_real_mean",
        "d_fake_mean",
        "gp",
        "fid_overall",
    ] + [f"fid_{name}" for name in class_names_ordered]

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
        writer.writeheader()

    global_step = 0
    last_g_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        loss_g_sum = 0.0
        loss_d_sum = 0.0
        d_real_sum = 0.0
        d_fake_sum = 0.0
        grad_norm_d_sum = 0.0
        grad_norm_g_sum = 0.0
        gp_sum = 0.0
        g_update_count = 0
        d_update_count = 0

        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            n = real_imgs.size(0)

            fake_labels_d = sample_fake_labels(labels, num_classes, device, args.uniform_fake_labels)
            z = torch.randn(n, args.z_dim, device=device)
            fake_imgs = generator(z, fake_labels_d).detach()

            real_for_d = diff_augment(real_imgs, diffaugment_ops) if args.use_diffaugment else real_imgs
            fake_for_d = diff_augment(fake_imgs, diffaugment_ops) if args.use_diffaugment else fake_imgs

            opt_d.zero_grad(set_to_none=True)
            d_real = discriminator(real_for_d, labels)
            d_fake = discriminator(fake_for_d, fake_labels_d)

            gp_value = torch.tensor(0.0, device=device)
            if args.gan_loss == "bce":
                real_targets = torch.full((n,), args.real_label, device=device)
                fake_targets = torch.full((n,), args.fake_label, device=device)
                loss_d = bce_criterion(d_real, real_targets) + bce_criterion(d_fake, fake_targets)
            elif args.gan_loss == "hinge":
                loss_d = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()
            elif args.gan_loss == "wgangp":
                gp_value = gradient_penalty(discriminator, real_imgs, fake_imgs, labels)
                loss_d = -(d_real.mean() - d_fake.mean()) + args.gp_lambda * gp_value
            else:
                raise ValueError(f"Unsupported gan_loss: {args.gan_loss}")

            loss_d.backward()
            grad_d = grad_norm(discriminator)
            opt_d.step()

            loss_d_sum += float(loss_d.item())
            d_real_sum += float(d_real.mean().item())
            d_fake_sum += float(d_fake.mean().item())
            gp_sum += float(gp_value.item())
            grad_norm_d_sum += grad_d
            d_update_count += 1

            do_g_update = ((global_step + 1) % args.n_critic) == 0
            if do_g_update:
                fake_labels_g = sample_fake_labels(labels, num_classes, device, args.uniform_fake_labels)
                z_g = torch.randn(n, args.z_dim, device=device)
                fake_for_g = generator(z_g, fake_labels_g)
                fake_for_g_in = diff_augment(fake_for_g, diffaugment_ops) if args.use_diffaugment else fake_for_g

                opt_g.zero_grad(set_to_none=True)
                d_fake_for_g = discriminator(fake_for_g_in, fake_labels_g)
                if args.gan_loss == "bce":
                    g_targets = torch.full((n,), args.real_label, device=device)
                    loss_g = bce_criterion(d_fake_for_g, g_targets)
                elif args.gan_loss in {"hinge", "wgangp"}:
                    loss_g = -d_fake_for_g.mean()
                else:
                    raise ValueError(f"Unsupported gan_loss: {args.gan_loss}")

                loss_g.backward()
                grad_g = grad_norm(generator)
                opt_g.step()

                last_g_loss = float(loss_g.item())
                loss_g_sum += last_g_loss
                grad_norm_g_sum += grad_g
                g_update_count += 1

            pbar.set_postfix({"loss_d": f"{loss_d.item():.3f}", "loss_g": f"{last_g_loss:.3f}"})
            global_step += 1

        loss_d_avg = loss_d_sum / max(1, d_update_count)
        loss_g_avg = loss_g_sum / max(1, g_update_count)
        grad_norm_d_avg = grad_norm_d_sum / max(1, d_update_count)
        grad_norm_g_avg = grad_norm_g_sum / max(1, g_update_count)
        d_real_avg = d_real_sum / max(1, d_update_count)
        d_fake_avg = d_fake_sum / max(1, d_update_count)
        gp_avg = gp_sum / max(1, d_update_count)

        fid_overall = ""
        per_class_fid = {}
        should_eval_fid = fid_state.enabled and (epoch == 1 or (args.fid_every > 0 and epoch % args.fid_every == 0))
        if should_eval_fid:
            fid_overall_value, per_class_fid = evaluate_fid(generator, fid_state)
            fid_overall = f"{fid_overall_value:.6f}"
            print(
                f"Epoch {epoch}: loss_d={loss_d_avg:.4f} loss_g={loss_g_avg:.4f} "
                f"grad_d={grad_norm_d_avg:.4f} grad_g={grad_norm_g_avg:.4f} fid={fid_overall}"
            )
        else:
            print(
                f"Epoch {epoch}: loss_d={loss_d_avg:.4f} loss_g={loss_g_avg:.4f} "
                f"grad_d={grad_norm_d_avg:.4f} grad_g={grad_norm_g_avg:.4f}"
            )

        should_save_sample = epoch == 1 or (args.sample_every > 0 and epoch % args.sample_every == 0)
        should_save_ckpt = args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0
        if should_save_sample:
            save_class_grids(
                epoch=epoch,
                generator=generator,
                out_dir=out_dir,
                fixed_z=fixed_z,
                fixed_labels=fixed_labels,
                num_classes=num_classes,
            )
        if should_save_ckpt:
            save_checkpoint(
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                opt_g=opt_g,
                opt_d=opt_d,
                class_to_idx=class_to_idx,
                out_dir=out_dir,
                config={
                    "img_size": args.img_size,
                    "z_dim": args.z_dim,
                    "num_classes": num_classes,
                    "gan_loss": args.gan_loss,
                    "use_spectral_norm": args.use_spectral_norm,
                    "use_diffaugment": args.use_diffaugment,
                    "diffaugment_policy": args.diffaugment_policy,
                    "n_critic": args.n_critic,
                    "balanced_sampler": args.balanced_sampler,
                    "uniform_fake_labels": args.uniform_fake_labels,
                    "gp_lambda": args.gp_lambda,
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "beta1": args.beta1,
                    "beta2": args.beta2,
                },
            )

        row = {
            "epoch": epoch,
            "loss_d": f"{loss_d_avg:.6f}",
            "loss_g": f"{loss_g_avg:.6f}",
            "grad_norm_d": f"{grad_norm_d_avg:.6f}",
            "grad_norm_g": f"{grad_norm_g_avg:.6f}",
            "d_real_mean": f"{d_real_avg:.6f}",
            "d_fake_mean": f"{d_fake_avg:.6f}",
            "gp": f"{gp_avg:.6f}",
            "fid_overall": fid_overall,
        }
        for class_name in class_names_ordered:
            value = per_class_fid.get(class_name)
            row[f"fid_{class_name}"] = "" if value is None else f"{float(value):.6f}"

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
            writer.writerow(row)

    print("Training finished.")
    print("Outputs in:", out_dir.resolve())
    print("Metrics CSV:", metrics_path.resolve())


if __name__ == "__main__":
    main()
