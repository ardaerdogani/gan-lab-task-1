from __future__ import annotations

import torch


def parse_diffaugment_policy(policy: str):
    return [part.strip() for part in policy.split(",") if part.strip()]


def rand_brightness(x):
    noise = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5
    return x + noise


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) * 2.0
    return (x - x_mean) * scale + x_mean


def rand_contrast(x):
    x_mean = x.mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) + 0.5
    return (x - x_mean) * scale + x_mean


def rand_translation(x, ratio=0.125):
    if ratio <= 0:
        return x
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    if shift_x == 0 and shift_y == 0:
        return x

    translation_x = torch.randint(-shift_x, shift_x + 1, size=(x.size(0), 1, 1), device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=(x.size(0), 1, 1), device=x.device)

    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)

    x_pad = torch.nn.functional.pad(x, [1, 1, 1, 1])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    if ratio <= 0:
        return x
    cutout_h = max(1, int(x.size(2) * ratio + 0.5))
    cutout_w = max(1, int(x.size(3) * ratio + 0.5))
    offset_x = torch.randint(0, x.size(2), size=(x.size(0), 1, 1), device=x.device)
    offset_y = torch.randint(0, x.size(3), size=(x.size(0), 1, 1), device=x.device)

    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(cutout_h, device=x.device),
        torch.arange(cutout_w, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_h // 2, 0, x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_w // 2, 0, x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), device=x.device, dtype=x.dtype)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


def diff_augment(x, policy_ops):
    if not policy_ops:
        return x
    for op in policy_ops:
        if op == "color":
            x = rand_brightness(x)
            x = rand_saturation(x)
            x = rand_contrast(x)
        elif op == "translation":
            x = rand_translation(x)
        elif op == "cutout":
            x = rand_cutout(x)
        else:
            raise ValueError(f"Unsupported DiffAugment op: {op}")
    return x
