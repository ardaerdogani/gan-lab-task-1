from __future__ import annotations

import torch


def sample_fake_labels(real_labels, num_classes, device, uniform_fake_labels):
    if uniform_fake_labels:
        return torch.randint(0, num_classes, size=real_labels.shape, device=device, dtype=torch.long)
    return real_labels


def gradient_penalty(discriminator, real_imgs, fake_imgs, labels):
    n = real_imgs.size(0)
    alpha = torch.rand(n, 1, 1, 1, device=real_imgs.device)
    interpolated = (alpha * real_imgs + (1.0 - alpha) * fake_imgs).requires_grad_(True)
    d_inter = discriminator(interpolated, labels)
    ones = torch.ones_like(d_inter)
    gradients = torch.autograd.grad(
        outputs=d_inter,
        inputs=interpolated,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(n, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def grad_norm(model):
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return total ** 0.5
