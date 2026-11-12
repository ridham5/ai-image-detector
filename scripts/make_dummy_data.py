"""Make a tiny fake dataset to check the code runs end to end (NOT for real results).

    python scripts/make_dummy_data.py --out data/dummy

"Real" images = smooth random textures. "Fake" images = the same, but up-sampled with
nearest-neighbour from a smaller size, which leaves a periodic grid pattern - a toy
version of the up-sampling artifacts real generators leave behind.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def texture(rng, size):
    a = rng.normal(0, 1, (size, size, 3))
    img = Image.fromarray(np.uint8(np.clip((a * 40 + 128), 0, 255)))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 3.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--n", type=int, default=120, help="images per class per split")
    ap.add_argument("--size", type=int, default=32)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for split, n in (("train", a.n), ("test", max(20, a.n // 3))):
        for cls in ("REAL", "FAKE"):
            d = Path(a.out) / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                if cls == "REAL":
                    img = texture(rng, a.size)
                else:
                    img = texture(rng, a.size // 2).resize((a.size, a.size), Image.NEAREST)
                img.save(d / f"{i:04d}.png")
    print(f"wrote dummy dataset to {a.out}")


if __name__ == "__main__":
    main()
