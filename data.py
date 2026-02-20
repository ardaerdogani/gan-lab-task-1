import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import default_num_workers


def get_transform(img_size=32):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def get_train_loader(
    data_train="data/split/train",
    batch_size=64,
    img_size=32,
    num_workers=None,
    drop_last=True,
):
    if num_workers is None:
        num_workers = default_num_workers()

    transform = get_transform(img_size=img_size)
    train_ds = ImageFolder(data_train, transform=transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )
    return train_ds, train_loader
