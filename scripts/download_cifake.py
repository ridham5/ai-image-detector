"""Download CIFAKE (Bird & Lotfi, 2024) from Kaggle and link it to data/cifake.

    python scripts/download_cifake.py

Needs `pip install kagglehub`. Public dataset; if Kaggle asks you to log in, create a free
account, go to Settings -> API -> Create New Token, and upload kaggle.json (see README).
Resulting layout:  data/cifake/train/{REAL,FAKE}  and  data/cifake/test/{REAL,FAKE}
"""
import os
import sys
from pathlib import Path

import kagglehub

path = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))
print("downloaded to", path)

# find the folder that contains train/ and test/
root = next((p for p in [path, *path.rglob("*")] if (p / "train").is_dir() and (p / "test").is_dir()), None)
if root is None:
    sys.exit(f"Could not find train/ and test/ under {path}; look inside and pass that folder to the scripts.")

dst = Path("data/cifake")
dst.parent.mkdir(parents=True, exist_ok=True)
if dst.is_symlink() or dst.exists():
    print(f"{dst} already exists - leaving it")
else:
    try:
        os.symlink(root.resolve(), dst)
        print(f"linked {dst} -> {root}")
    except OSError:  # some filesystems (e.g. Google Drive) do not allow links
        import shutil
        shutil.copytree(root, dst)
        print(f"copied {root} -> {dst}")
for split in ("train", "test"):
    for c in ("REAL", "FAKE"):
        d = dst / split / c
        print(f"  {d}: {len(list(d.glob('*'))) if d.exists() else 'MISSING'} images")
