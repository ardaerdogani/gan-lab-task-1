from __future__ import annotations

import torch
from torchvision.utils import make_grid, save_image


def prepare_fixed_grid(
    num_classes: int,
    z_dim: int,
    device: torch.device,
    per_class: int = 36,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed + 913)
    z = torch.randn(num_classes * per_class, z_dim, generator=cpu_gen, dtype=torch.float32)
    labels = torch.cat([torch.full((per_class,), c, dtype=torch.long) for c in range(num_classes)], dim=0)
    return z.to(device), labels.to(device)


@torch.no_grad()
def save_class_grids(
    epoch,
    generator,
    out_dir,
    fixed_z: torch.Tensor,
    fixed_labels: torch.Tensor,
    num_classes: int,
    per_class: int = 36,
    nrow: int = 6,
):
    generator.eval()
    fake = generator(fixed_z, fixed_labels)
    grids = []
    for class_idx in range(num_classes):
        start = class_idx * per_class
        end = start + per_class
        grid = make_grid(fake[start:end], nrow=nrow, normalize=True, value_range=(-1, 1))
        grids.append(grid)
    full = torch.cat(grids, dim=1)
    save_image(full, out_dir / f"samples_epoch_{epoch:03d}.png")
    generator.train()


def save_checkpoint(epoch, generator, discriminator, opt_g, opt_d, class_to_idx, out_dir, config):
    torch.save(
        {
            "G": generator.state_dict(),
            "D": discriminator.state_dict(),
            "opt_G": opt_g.state_dict(),
            "opt_D": opt_d.state_dict(),
            "epoch": epoch,
            "class_to_idx": class_to_idx,
            "img_size": config["img_size"],
            "z_dim": config["z_dim"],
            "num_classes": config["num_classes"],
            "gan_loss": config["gan_loss"],
            "use_spectral_norm": config["use_spectral_norm"],
            "use_diffaugment": config["use_diffaugment"],
            "diffaugment_policy": config["diffaugment_policy"],
            "n_critic": config["n_critic"],
            "balanced_sampler": config["balanced_sampler"],
            "uniform_fake_labels": config["uniform_fake_labels"],
            "gp_lambda": config["gp_lambda"],
            "lr_g": config["lr_g"],
            "lr_d": config["lr_d"],
            "beta1": config["beta1"],
            "beta2": config["beta2"],
        },
        out_dir / f"ckpt_epoch_{epoch:03d}.pt",
    )
