from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data import get_train_loader
from models_gan import Discriminator, Generator, weights_init
from utils import default_num_workers, get_best_device

DATA_TRAIN = "data/split/train"
OUT_DIR = Path("runs_gan")
IMG_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 3
Z_DIM = 128
BATCH_SIZE = 64
EPOCHS = 100
LR = 2e-4
BETAS = (0.5, 0.999)
SEED = 42
REAL_LABEL = 1.0
FAKE_LABEL = 0.0


@torch.no_grad()
def save_class_grids(epoch: int, generator, device, out_dir: Path, per_class: int = 36, nrow: int = 6):
    generator.eval()
    grids = []
    for class_idx in range(NUM_CLASSES):
        z = torch.randn(per_class, Z_DIM, device=device)
        y = torch.full((per_class,), class_idx, device=device, dtype=torch.long)
        fake = generator(z, y)
        grid = make_grid(fake, nrow=nrow, normalize=True, value_range=(-1, 1))
        grids.append(grid)

    full = torch.cat(grids, dim=1)
    save_image(full, out_dir / f"samples_epoch_{epoch:03d}.png")
    generator.train()


def save_checkpoint(epoch: int, generator, discriminator, opt_g, opt_d, class_to_idx, out_dir: Path):
    torch.save(
        {
            "G": generator.state_dict(),
            "D": discriminator.state_dict(),
            "opt_G": opt_g.state_dict(),
            "opt_D": opt_d.state_dict(),
            "epoch": epoch,
            "class_to_idx": class_to_idx,
            "img_size": IMG_SIZE,
            "z_dim": Z_DIM,
            "num_classes": NUM_CLASSES,
        },
        out_dir / f"ckpt_epoch_{epoch:03d}.pt",
    )


def main():
    torch.manual_seed(SEED)
    device = get_best_device()
    num_workers = default_num_workers()
    if device.type in {"cuda", "mps"}:
        torch.set_float32_matmul_precision("high")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, train_loader = get_train_loader(
        data_train=DATA_TRAIN,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_workers=num_workers,
        drop_last=True,
    )
    print("Device:", device)
    print("num_workers:", num_workers)
    print("class_to_idx:", train_ds.class_to_idx)

    generator = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES, img_channels=CHANNELS).to(device)
    discriminator = Discriminator(num_classes=NUM_CLASSES, img_channels=CHANNELS).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        loss_g_sum = 0.0
        loss_d_sum = 0.0

        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device, dtype=torch.long)
            n = real_imgs.size(0)

            discriminator.zero_grad(set_to_none=True)
            real_targets = torch.full((n,), REAL_LABEL, device=device)
            fake_targets = torch.full((n,), FAKE_LABEL, device=device)

            d_real = discriminator(real_imgs, labels)
            loss_d_real = criterion(d_real, real_targets)

            z = torch.randn(n, Z_DIM, device=device)
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

        if epoch == 1 or epoch % 5 == 0:
            save_class_grids(epoch, generator, device, OUT_DIR)
        if epoch % 10 == 0:
            save_checkpoint(epoch, generator, discriminator, opt_g, opt_d, train_ds.class_to_idx, OUT_DIR)

    print("Training finished.")
    print("Outputs in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
