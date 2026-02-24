from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def labels_to_map(y: torch.Tensor, num_classes: int, h: int, w: int) -> torch.Tensor:
    n = y.shape[0]
    onehot = torch.zeros(n, num_classes, device=y.device)
    onehot.scatter_(1, y.view(-1, 1), 1.0)
    return onehot.view(n, num_classes, 1, 1).expand(n, num_classes, h, w)


def maybe_sn(module: nn.Module, use_spectral_norm: bool) -> nn.Module:
    return spectral_norm(module) if use_spectral_norm else module


class Generator(nn.Module):
    def __init__(self, z_dim: int, num_classes: int, img_channels: int = 3, img_size: int = 32):
        super().__init__()
        if img_size not in {32, 64}:
            raise ValueError(f"Generator supports img_size 32 or 64, got {img_size}")

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_size = img_size
        in_dim = z_dim + num_classes

        base = 64
        blocks = [
            nn.ConvTranspose2d(in_dim, base * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
        ]

        if img_size == 64:
            blocks.extend(
                [
                    nn.ConvTranspose2d(base, base // 2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base // 2),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(base // 2, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.Tanh(),
                ]
            )
        else:
            blocks.extend(
                [
                    nn.ConvTranspose2d(base, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.Tanh(),
                ]
            )

        self.net = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        y_onehot = torch.zeros(n, self.num_classes, device=z.device)
        y_onehot.scatter_(1, y.view(-1, 1), 1.0)
        x = torch.cat([z, y_onehot], dim=1)
        x = x.view(n, self.z_dim + self.num_classes, 1, 1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_channels: int = 3,
        img_size: int = 32,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        if img_size not in {32, 64}:
            raise ValueError(f"Discriminator supports img_size 32 or 64, got {img_size}")

        self.num_classes = num_classes
        self.img_size = img_size
        in_ch = img_channels + num_classes

        def conv(in_channels, out_channels, kernel=4, stride=2, pad=1, bias=False):
            return maybe_sn(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad, bias=bias),
                use_spectral_norm,
            )

        blocks = [conv(in_ch, 64), nn.LeakyReLU(0.2, inplace=True)]

        if use_spectral_norm:
            blocks.extend([conv(64, 128), nn.LeakyReLU(0.2, inplace=True)])
            blocks.extend([conv(128, 256), nn.LeakyReLU(0.2, inplace=True)])
            if img_size == 64:
                blocks.extend([conv(256, 512), nn.LeakyReLU(0.2, inplace=True)])
                blocks.append(conv(512, 1, kernel=4, stride=1, pad=0))
            else:
                blocks.append(conv(256, 1, kernel=4, stride=1, pad=0))
        else:
            blocks.extend([conv(64, 128), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)])
            blocks.extend([conv(128, 256), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True)])
            if img_size == 64:
                blocks.extend([conv(256, 512), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True)])
                blocks.append(conv(512, 1, kernel=4, stride=1, pad=0))
            else:
                blocks.append(conv(256, 1, kernel=4, stride=1, pad=0))

        self.net = nn.Sequential(*blocks)

    def forward(self, img: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_map = labels_to_map(y, self.num_classes, img.size(2), img.size(3))
        x = torch.cat([img, y_map], dim=1)
        out = self.net(x)
        return out.view(-1)


def weights_init(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
