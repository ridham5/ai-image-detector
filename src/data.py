"""Data loading.

Every dataset is just a folder of images where some parent folder name says
whether the image is real or fake. This matches both datasets we use:

    CIFAKE:    cifake/train/REAL/*.jpg, cifake/train/FAKE/*.jpg, cifake/test/...
    GenImage:  genimage/<generator>/val/nature/*.JPEG, .../val/ai/*.png

Label convention used everywhere in this repo:  0 = REAL, 1 = FAKE (AI-generated).
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

REAL_NAMES = {"real", "nature", "0_real"}
FAKE_NAMES = {"fake", "ai", "1_fake"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _label_from_path(rel: Path) -> int | None:
    """Look at the folder names in the path (closest first) for a real/fake name."""
    for part in reversed(rel.parts[:-1]):
        p = part.lower()
        if p in REAL_NAMES:
            return 0
        if p in FAKE_NAMES:
            return 1
    return None


def list_images(root: str | Path, max_per_class: int | None = None, seed: int = 0):
    """Return a list of (path, label) for every image under `root`.

    max_per_class: randomly keep at most this many images per class (useful on
    free Colab, where training on all 100k CIFAKE images can be slow).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    real, fake = [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS or not p.is_file():
            continue
        lab = _label_from_path(p.relative_to(root))
        if lab == 0:
            real.append((str(p), 0))
        elif lab == 1:
            fake.append((str(p), 1))
    if not real or not fake:
        raise RuntimeError(
            f"Found {len(real)} real and {len(fake)} fake images under {root}. "
            f"Folders must be named one of {sorted(REAL_NAMES)} / {sorted(FAKE_NAMES)} (any case)."
        )
    rng = random.Random(seed)
    if max_per_class:
        if len(real) > max_per_class:
            real = rng.sample(real, max_per_class)
        if len(fake) > max_per_class:
            fake = rng.sample(fake, max_per_class)
    items = real + fake
    rng.shuffle(items)
    return items


def train_val_split(items, val_frac: float = 0.1, seed: int = 0):
    """Split a list of (path, label) into train / validation, keeping class balance."""
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for it in items:
        by_class[it[1]].append(it)
    train, val = [], []
    for lab in (0, 1):
        group = by_class[lab][:]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_frac))
        val += group[:n_val]
        train += group[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


try:  # torch is only needed for the CNN / CLIP methods
    from torch.utils.data import Dataset

    class ImageListDataset(Dataset):
        """Torch dataset over (path, label) items with an optional perturbation."""

        def __init__(self, items, transform, perturb=None):
            self.items = items
            self.transform = transform
            self.perturb = perturb

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            img = load_rgb(path)
            if self.perturb is not None:
                img = self.perturb(img)
            return self.transform(img), label

except ImportError:  # pragma: no cover
    pass
