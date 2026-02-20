import random, shutil
from pathlib import Path

random.seed(42)

RAW = Path("data/raw")
OUT = Path("data/split")
classes = ["apple", "orange", "banana"]

splits = [("train", 0.70), ("val", 0.15), ("test", 0.15)]

# klasörleri oluştur
for split, _ in splits:
    for c in classes:
        (OUT / split / c).mkdir(parents=True, exist_ok=True)

def is_image(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]

def resolve_source_dir(class_name: str) -> Path:
    # raw klasorleri cogul olabilir: apples/oranges/bananas
    candidates = [class_name, f"{class_name}s"]
    for name in candidates:
        d = RAW / name
        if d.is_dir():
            return d
    return RAW / class_name

for c in classes:
    source_dir = resolve_source_dir(c)
    if not source_dir.is_dir():
        print(f"[WARN] source klasoru yok: {source_dir}")
        print(c, "total:", 0, "train:", 0, "val:", 0, "test:", 0)
        continue

    imgs = [p for p in source_dir.glob("*") if p.is_file() and is_image(p)]
    imgs.sort()
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    # test kalan

    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train+n_val]
    test_imgs = imgs[n_train+n_val:]

    for p in train_imgs:
        shutil.copy2(p, OUT / "train" / c / p.name)
    for p in val_imgs:
        shutil.copy2(p, OUT / "val" / c / p.name)
    for p in test_imgs:
        shutil.copy2(p, OUT / "test" / c / p.name)

    print(c, "source:", source_dir.name, "total:", n, "train:", len(train_imgs), "val:", len(val_imgs), "test:", len(test_imgs))

print("Done:", OUT)
