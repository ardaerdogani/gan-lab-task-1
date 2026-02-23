import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models_gan import Generator
from utils import get_best_device


CKPT_PATH = Path("runs_gan/ckpt_epoch_090.pt")
OUT_ROOT = Path("data/synthetic")
NUM_PER_CLASS = 400
BATCH_SIZE = 64
SEED = 42


def load_generator(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt.get("class_to_idx")
    if not class_to_idx:
        raise ValueError("Checkpoint icinde 'class_to_idx' bulunamadi.")

    z_dim = ckpt.get("z_dim", 128)
    num_classes = ckpt.get("num_classes", len(class_to_idx))

    generator = Generator(z_dim=z_dim, num_classes=num_classes, img_channels=3).to(device)
    generator.load_state_dict(ckpt["G"])
    generator.eval()
    return generator, class_to_idx, z_dim


@torch.no_grad()
def generate_for_class(generator, class_name, class_idx, z_dim, device, out_root, num_per_class, batch_size):
    out_dir = out_root / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    while saved < num_per_class:
        current_batch = min(batch_size, num_per_class - saved)
        z = torch.randn(current_batch, z_dim, device=device)
        y = torch.full((current_batch,), class_idx, device=device, dtype=torch.long)
        fake = generator(z, y)

        # [-1,1] -> [0,1] ve PNG olarak kaydet
        fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
        for i in range(current_batch):
            img_path = out_dir / f"{class_name}_{saved + i:04d}.png"
            save_image(fake[i], img_path)

        saved += current_batch
        print(f"{class_name}: {saved}/{num_per_class}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic images from GAN checkpoint")
    parser.add_argument("--ckpt-path", type=str, default=str(CKPT_PATH))
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    parser.add_argument("--num-per-class", type=int, default=NUM_PER_CLASS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_best_device()
    if device.type in {"cuda", "mps"}:
        torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt_path)
    out_root = Path(args.out_root)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadi: {ckpt_path}")

    generator, class_to_idx, z_dim = load_generator(ckpt_path, device)

    # idx sirasina gore isleyelim ki class_to_idx ile birebir uyumlu olsun.
    idx_to_class = sorted(class_to_idx.items(), key=lambda item: item[1])
    print("Using checkpoint:", ckpt_path)
    print("Device:", device)
    print("class_to_idx:", class_to_idx)
    print("num_per_class:", args.num_per_class)
    print("batch_size:", args.batch_size)

    for class_name, class_idx in idx_to_class:
        generate_for_class(
            generator=generator,
            class_name=class_name,
            class_idx=class_idx,
            z_dim=z_dim,
            device=device,
            out_root=out_root,
            num_per_class=args.num_per_class,
            batch_size=args.batch_size,
        )

    print("Done. Saved synthetic images under:", out_root.resolve())


if __name__ == "__main__":
    main()
