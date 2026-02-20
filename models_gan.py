import torch
import torch.nn as nn


def labels_to_map(y: torch.Tensor, num_classes: int, h: int, w: int) -> torch.Tensor:
    n = y.shape[0]
    onehot = torch.zeros(n, num_classes, device=y.device)
    onehot.scatter_(1, y.view(-1, 1), 1.0)
    return onehot.view(n, num_classes, 1, 1).expand(n, num_classes, h, w)


class Generator(nn.Module):
    def __init__(self, z_dim: int, num_classes: int, img_channels: int = 3):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        in_dim = z_dim + num_classes

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        y_onehot = torch.zeros(n, self.num_classes, device=z.device)
        y_onehot.scatter_(1, y.view(-1, 1), 1.0)

        x = torch.cat([z, y_onehot], dim=1)
        x = x.view(n, self.z_dim + self.num_classes, 1, 1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int, img_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        in_ch = img_channels + num_classes

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_map = labels_to_map(y, self.num_classes, img.size(2), img.size(3))
        x = torch.cat([img, y_map], dim=1)
        out = self.net(x)
        return out.view(-1)


def weights_init(module):
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
