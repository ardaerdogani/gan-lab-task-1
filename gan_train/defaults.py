from pathlib import Path

DATA_TRAIN = "data/split/train"
OUT_DIR = Path("runs_gan")
IMG_SIZE = 32
CHANNELS = 3
Z_DIM = 128
BATCH_SIZE = 64
EPOCHS = 100
LR = 2e-4
BETAS = (0.5, 0.999)
SEED = 42
REAL_LABEL = 1.0
FAKE_LABEL = 0.0
