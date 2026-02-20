from pathlib import Path

import torch
from torchvision.utils import save_image

from models_gan import Generator
from utils import get_best_device


# Kendi checkpoint dosyana gore degistirebilirsin.
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
def generate_for_class(generator, class_name, class_idx, z_dim, device):
    out_dir = OUT_ROOT / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    while saved < NUM_PER_CLASS:
        current_batch = min(BATCH_SIZE, NUM_PER_CLASS - saved)
        z = torch.randn(current_batch, z_dim, device=device)
        y = torch.full((current_batch,), class_idx, device=device, dtype=torch.long)
        fake = generator(z, y)

        # [-1,1] -> [0,1] ve PNG olarak kaydet
        fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
        for i in range(current_batch):
            img_path = out_dir / f"{class_name}_{saved + i:04d}.png"
            save_image(fake[i], img_path)

        saved += current_batch
        print(f"{class_name}: {saved}/{NUM_PER_CLASS}")


def main():
    torch.manual_seed(SEED)
    device = get_best_device()
    if device.type in {"cuda", "mps"}:
        torch.set_float32_matmul_precision("high")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadi: {CKPT_PATH}")

    generator, class_to_idx, z_dim = load_generator(CKPT_PATH, device)

    # idx sirasina gore isleyelim ki class_to_idx ile birebir uyumlu olsun.
    idx_to_class = sorted(class_to_idx.items(), key=lambda item: item[1])
    print("Using checkpoint:", CKPT_PATH)
    print("Device:", device)
    print("class_to_idx:", class_to_idx)

    for class_name, class_idx in idx_to_class:
        generate_for_class(generator, class_name, class_idx, z_dim, device)

    print("Done. Saved synthetic images under:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
