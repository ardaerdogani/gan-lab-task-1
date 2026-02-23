import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from models_gan import Discriminator, Generator, weights_init
from utils import default_num_workers, get_best_device, get_transform, should_pin_memory

DATA_TRAIN = "data/split/train"
OUT_DIR = Path("runs_gan")
IMG_SIZE = 32
CHANNELS = 3
Z_DIM = 128
BATCH_SIZE = 64
EPOCHS = 100
LR = 2e-4
BETAS = (0.5, 0.999)
SEED = 42
REAL_LABEL = 1.0
FAKE_LABEL = 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _allocate_per_class(total_count, class_counts):
    if total_count < len(class_counts):
        raise ValueError("Toplam sample sayisi class sayisindan kucuk olamaz.")

    ratios = class_counts / class_counts.sum()
    raw = ratios * total_count
    alloc = np.floor(raw).astype(np.int64)
    alloc = np.maximum(alloc, 1)

    while alloc.sum() > total_count:
        candidates = np.where(alloc > 1)[0]
        if len(candidates) == 0:
            break
        idx = candidates[np.argmax(alloc[candidates])]
        alloc[idx] -= 1

    remainder = int(total_count - alloc.sum())
    if remainder > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(frac)[::-1]
        for i in range(remainder):
            alloc[order[i % len(order)]] += 1

    return alloc


def make_stratified_subset_by_count(dataset, total_count, seed):
    n_total = len(dataset)
    if total_count is None or total_count >= n_total:
        return dataset
    if total_count <= 0:
        raise ValueError("subset_count pozitif olmali.")

    targets = np.array(dataset.targets)
    class_ids = np.unique(targets)
    class_counts = np.array([int((targets == c).sum()) for c in class_ids], dtype=np.int64)
    alloc = _allocate_per_class(total_count, class_counts)

    rng = np.random.default_rng(seed)
    selected = []
    for class_id, take_count in zip(class_ids, alloc.tolist()):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        k = min(int(take_count), len(class_indices))
        selected.extend(class_indices[:k].tolist())

    if len(selected) < total_count:
        remaining = np.setdiff1d(np.arange(n_total), np.array(selected, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)
        needed = total_count - len(selected)
        selected.extend(remaining[:needed].tolist())

    rng.shuffle(selected)
    return Subset(dataset, selected)


def extract_targets(dataset):
    if isinstance(dataset, Subset):
        base_targets = np.array(extract_targets(dataset.dataset))
        return base_targets[np.array(dataset.indices)]
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)}")


def build_train_loader(data_train, img_size, batch_size, num_workers, subset_count, seed, device):
    train_ds = ImageFolder(data_train, transform=get_transform(img_size=img_size))
    train_ds = make_stratified_subset_by_count(train_ds, subset_count, seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=should_pin_memory(device),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    return train_ds, train_loader


@torch.no_grad()
def save_class_grids(epoch, generator, device, out_dir, num_classes, z_dim, per_class=36, nrow=6):
    generator.eval()
    grids = []
    for class_idx in range(num_classes):
        z = torch.randn(per_class, z_dim, device=device)
        y = torch.full((per_class,), class_idx, device=device, dtype=torch.long)
        fake = generator(z, y)
        grid = make_grid(fake, nrow=nrow, normalize=True, value_range=(-1, 1))
        grids.append(grid)

    full = torch.cat(grids, dim=1)
    save_image(full, out_dir / f"samples_epoch_{epoch:03d}.png")
    generator.train()


def save_checkpoint(epoch, generator, discriminator, opt_g, opt_d, class_to_idx, out_dir, img_size, z_dim, num_classes):
    torch.save(
        {
            "G": generator.state_dict(),
            "D": discriminator.state_dict(),
            "opt_G": opt_g.state_dict(),
            "opt_D": opt_d.state_dict(),
            "epoch": epoch,
            "class_to_idx": class_to_idx,
            "img_size": img_size,
            "z_dim": z_dim,
            "num_classes": num_classes,
        },
        out_dir / f"ckpt_epoch_{epoch:03d}.pt",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional GAN training")
    parser.add_argument("--data-train", type=str, default=DATA_TRAIN)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--z-dim", type=int, default=Z_DIM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
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
    return parser.parse_args()


def main():
    args = parse_args()
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
    )

    class_to_idx = train_ds.dataset.class_to_idx if isinstance(train_ds, Subset) else train_ds.class_to_idx
    num_classes = len(class_to_idx)
    class_targets = extract_targets(train_ds)
    class_counts = np.bincount(class_targets, minlength=num_classes).astype(np.int64)

    print("Device:", device)
    print("num_workers:", num_workers)
    print("class_to_idx:", class_to_idx)
    print("effective_train_samples:", len(train_ds))
    print("effective_class_counts:", class_counts.tolist())

    generator = Generator(z_dim=args.z_dim, num_classes=num_classes, img_channels=CHANNELS).to(device)
    discriminator = Discriminator(num_classes=num_classes, img_channels=CHANNELS).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        loss_g_sum = 0.0
        loss_d_sum = 0.0

        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            n = real_imgs.size(0)

            discriminator.zero_grad(set_to_none=True)
            real_targets = torch.full((n,), REAL_LABEL, device=device)
            fake_targets = torch.full((n,), FAKE_LABEL, device=device)

            d_real = discriminator(real_imgs, labels)
            loss_d_real = criterion(d_real, real_targets)

            z = torch.randn(n, args.z_dim, device=device)
            fake_imgs = generator(z, labels)
            d_fake = discriminator(fake_imgs.detach(), labels)
            loss_d_fake = criterion(d_fake, fake_targets)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            generator.zero_grad(set_to_none=True)
            g_targets = torch.full((n,), REAL_LABEL, device=device)
            d_fake_for_g = discriminator(fake_imgs, labels)
            loss_g = criterion(d_fake_for_g, g_targets)
            loss_g.backward()
            opt_g.step()

            loss_g_sum += loss_g.item()
            loss_d_sum += loss_d.item()
            pbar.set_postfix({"loss_d": f"{loss_d.item():.3f}", "loss_g": f"{loss_g.item():.3f}"})

        loss_g_avg = loss_g_sum / len(train_loader)
        loss_d_avg = loss_d_sum / len(train_loader)
        print(f"Epoch {epoch}: loss_d={loss_d_avg:.4f} loss_g={loss_g_avg:.4f}")

        should_save_sample = epoch == 1 or (args.sample_every > 0 and epoch % args.sample_every == 0)
        should_save_ckpt = args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0
        if should_save_sample:
            save_class_grids(
                epoch=epoch,
                generator=generator,
                device=device,
                out_dir=out_dir,
                num_classes=num_classes,
                z_dim=args.z_dim,
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
                img_size=args.img_size,
                z_dim=args.z_dim,
                num_classes=num_classes,
            )

    print("Training finished.")
    print("Outputs in:", out_dir.resolve())


if __name__ == "__main__":
    main()
