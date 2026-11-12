"""Train one detector.

Examples
    python scripts/train.py --method fft_svm    --train-dir data/cifake/train --max-per-class 10000
    python scripts/train.py --method resnet50   --train-dir data/cifake/train --max-per-class 20000 --epochs 3
    python scripts/train.py --method clip_probe --train-dir data/cifake/train --max-per-class 20000
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import list_images, train_val_split  # noqa: E402
from src.detectors import DEFAULT_FILES, METHODS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", default=None, help="model file (default: models/<method>.*)")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1, help="part of train set held out for validation (resnet50)")
    ap.add_argument("--seed", type=int, default=42)
    # resnet50
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (only for testing)")
    ap.add_argument("--num-workers", type=int, default=2)
    # fft_svm
    ap.add_argument("--fft-size", type=int, default=128)
    ap.add_argument("--svm-C", type=float, default=10.0)
    # clip_probe
    ap.add_argument("--clip-arch", default="ViT-B-32")
    ap.add_argument("--clip-pretrained", default="openai")
    ap.add_argument("--probe-C", type=float, default=1.0)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device (on a Mac, cpu avoids MPS training problems)")
    a = ap.parse_args()
    if a.device != "auto":
        import os
        os.environ["DETECTOR_DEVICE"] = a.device

    random.seed(a.seed)
    np.random.seed(a.seed)
    out = a.out or f"models/{DEFAULT_FILES[a.method]}"
    items = list_images(a.train_dir, a.max_per_class, a.seed)
    print(f"{len(items)} training images ({sum(l for _, l in items)} fake) from {a.train_dir}")
    t = time.time()

    if a.method == "fft_svm":
        det = METHODS[a.method](size=a.fft_size, C=a.svm_C).fit(items)
        det.save(out)
    elif a.method == "resnet50":
        import torch
        torch.manual_seed(a.seed)
        train, val = train_val_split(items, a.val_frac, a.seed)
        print(f"train {len(train)} / val {len(val)}")
        det = METHODS[a.method](img_size=a.img_size, pretrained=not a.no_pretrained)
        print(f"device: {det.device}")
        det.fit(train, val, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                num_workers=a.num_workers, save_path=out)
    else:
        pre = None if a.clip_pretrained.lower() == "none" else a.clip_pretrained
        det = METHODS[a.method](arch=a.clip_arch, pretrained=pre, C=a.probe_C)
        print(f"device: {det.device}")
        det.fit(items, batch_size=a.batch_size, num_workers=a.num_workers)
        det.save(out)

    meta = {"method": a.method, "train_dir": a.train_dir, "n_train": len(items),
            "seconds": round(time.time() - t, 1), "args": vars(a)}
    Path(out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}  ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
