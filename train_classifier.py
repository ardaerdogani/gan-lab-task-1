import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import default_num_workers, get_best_device, get_transform, should_pin_memory


REAL_TRAIN = "data/split/train"
REAL_VAL = "data/split/val"
REAL_TEST = "data/split/test"
SYNTH_DIR = "data/synthetic"
OUT_DIR = Path("runs_classifier")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_train_transform(img_size=32, use_augmentation=False):
    if not use_augmentation:
        return get_transform(img_size)
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stratified_subset(dataset, ratio, seed):
    if ratio >= 1.0:
        return dataset

    targets = np.array(dataset.targets)
    selected = []
    rng = np.random.default_rng(seed)
    for class_id in np.unique(targets):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        k = max(1, int(len(class_indices) * ratio))
        selected.extend(class_indices[:k].tolist())

    rng.shuffle(selected)
    return Subset(dataset, selected)


def extract_targets(dataset):
    if isinstance(dataset, Subset):
        base_targets = np.array(extract_targets(dataset.dataset))
        return base_targets[np.array(dataset.indices)]

    if isinstance(dataset, ConcatDataset):
        parts = [np.array(extract_targets(ds)) for ds in dataset.datasets]
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)

    raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)}")


def build_balancing_weights(dataset, num_classes):
    targets = extract_targets(dataset).astype(np.int64)
    if len(targets) == 0:
        raise ValueError("Balancing icin train dataset bos olamaz.")

    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    class_weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = class_counts > 0
    class_weights[nonzero] = class_counts.sum() / (class_counts[nonzero] * nonzero.sum())

    sample_weights = class_weights[targets].astype(np.float64)
    return class_counts.astype(np.int64), class_weights, sample_weights


def make_loader(dataset, batch_size, shuffle, num_workers, device, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=should_pin_memory(device),
        persistent_workers=num_workers > 0,
    )


def train_one_model(train_loader, epochs, lr, device, num_classes, class_weights=None):
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for _ in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time
    return model, train_time


def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro"))
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm


def run_experiment(
    name,
    train_dataset,
    test_loader,
    epochs,
    lr,
    batch_size,
    num_workers,
    device,
    num_classes,
    use_balancing,
):
    print(f"\n===== {name} =====")
    sampler = None
    class_weights_tensor = None
    if use_balancing:
        class_counts, class_weights_np, sample_weights_np = build_balancing_weights(train_dataset, num_classes)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights_np, dtype=torch.double),
            num_samples=len(sample_weights_np),
            replacement=True,
        )
        class_weights_tensor = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)
        print("Balancing: on")
        print("Train class counts:", class_counts.tolist())
        print("Class weights:", [round(float(w), 4) for w in class_weights_np.tolist()])
    else:
        print("Balancing: off")

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
        sampler=sampler,
    )

    model, train_time = train_one_model(
        train_loader=train_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        num_classes=num_classes,
        class_weights=class_weights_tensor,
    )
    acc, f1, cm = evaluate(model, test_loader, device)
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(f1, 4))
    print("Train Time (s):", round(train_time, 2))
    print("Confusion Matrix:\n", cm)
    return acc, f1, train_time


def write_results(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ratio", "scenario", "augmentation", "accuracy", "macro_f1", "train_time_s"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Classifier ablation: amount vs accuracy vs time")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 => auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--skip-aug", action="store_true", help="Skip optional classic augmentation scenario")
    parser.add_argument("--disable-balancing", action="store_true", help="Disable weighted sampler + class-weighted loss")
    parser.add_argument("--out-csv", type=str, default="runs_classifier/amount_vs_accuracy_time.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_best_device()
    num_workers = default_num_workers() if args.num_workers < 0 else args.num_workers

    if device.type in {"cuda", "mps"}:
        torch.set_float32_matmul_precision("high")

    eval_transform = get_transform(args.img_size)
    train_transform_plain = get_train_transform(args.img_size, use_augmentation=False)
    train_transform_aug = get_train_transform(args.img_size, use_augmentation=True)

    real_train_plain = ImageFolder(REAL_TRAIN, transform=train_transform_plain)
    real_train_aug = ImageFolder(REAL_TRAIN, transform=train_transform_aug)
    real_test = ImageFolder(REAL_TEST, transform=eval_transform)
    synth_train_plain = ImageFolder(SYNTH_DIR, transform=train_transform_plain)

    if real_train_plain.class_to_idx != synth_train_plain.class_to_idx:
        raise ValueError("real ve synthetic class_to_idx uyumsuz.")

    num_classes = len(real_train_plain.classes)
    test_loader = make_loader(real_test, args.batch_size, shuffle=False, num_workers=num_workers, device=device)

    print("Device:", device)
    print("class_to_idx:", real_train_plain.class_to_idx)
    print("Ratios:", args.ratios)
    print("num_workers:", num_workers)
    print("balancing:", "off" if args.disable_balancing else "on")

    rows = []

    synth_acc, synth_f1, synth_time = run_experiment(
        name="Synth-only (fixed baseline)",
        train_dataset=synth_train_plain,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=num_workers,
        device=device,
        num_classes=num_classes,
        use_balancing=not args.disable_balancing,
    )
    rows.append(
        {
            "ratio": "fixed",
            "scenario": "synth_only",
            "augmentation": "no",
            "accuracy": round(synth_acc, 4),
            "macro_f1": round(synth_f1, 4),
            "train_time_s": round(synth_time, 2),
        }
    )

    for ratio in args.ratios:
        ratio_tag = f"{int(ratio * 100)}%"
        real_subset_plain = make_stratified_subset(real_train_plain, ratio, args.seed)
        real_subset_aug = make_stratified_subset(real_train_aug, ratio, args.seed)
        mixed_dataset = ConcatDataset([real_subset_plain, synth_train_plain])

        real_acc, real_f1, real_time = run_experiment(
            name=f"Real-only ({ratio_tag})",
            train_dataset=real_subset_plain,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=num_workers,
            device=device,
            num_classes=num_classes,
            use_balancing=not args.disable_balancing,
        )
        rows.append(
            {
                "ratio": ratio_tag,
                "scenario": "real_only",
                "augmentation": "no",
                "accuracy": round(real_acc, 4),
                "macro_f1": round(real_f1, 4),
                "train_time_s": round(real_time, 2),
            }
        )

        mix_acc, mix_f1, mix_time = run_experiment(
            name=f"Real+Synth ({ratio_tag})",
            train_dataset=mixed_dataset,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=num_workers,
            device=device,
            num_classes=num_classes,
            use_balancing=not args.disable_balancing,
        )
        rows.append(
            {
                "ratio": ratio_tag,
                "scenario": "real_plus_synth",
                "augmentation": "no",
                "accuracy": round(mix_acc, 4),
                "macro_f1": round(mix_f1, 4),
                "train_time_s": round(mix_time, 2),
            }
        )

        if not args.skip_aug:
            aug_acc, aug_f1, aug_time = run_experiment(
                name=f"Real-only + Classic Aug ({ratio_tag})",
                train_dataset=real_subset_aug,
                test_loader=test_loader,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                num_workers=num_workers,
                device=device,
                num_classes=num_classes,
                use_balancing=not args.disable_balancing,
            )
            rows.append(
                {
                    "ratio": ratio_tag,
                    "scenario": "real_only",
                    "augmentation": "yes",
                    "accuracy": round(aug_acc, 4),
                    "macro_f1": round(aug_f1, 4),
                    "train_time_s": round(aug_time, 2),
                }
            )

    out_csv = Path(args.out_csv)
    write_results(rows, out_csv)
    print("\nSaved table:", out_csv.resolve())


if __name__ == "__main__":
    main()
