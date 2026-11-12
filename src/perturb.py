"""Image perturbations for the robustness experiment.

Real-world images get compressed, resized and blurred when shared online.
A good detector should still work after that. Each perturbation is written
as a short spec string so it can be passed on the command line:

    none          no change
    jpeg:75       re-save as JPEG with quality 75
    blur:1.5      Gaussian blur with sigma 1.5 pixels
    resize:0.5    shrink to 50% then scale back up to the original size
    noise:8       add Gaussian noise with std 8 (on the 0-255 scale)
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_SUITE = ["none", "jpeg:95", "jpeg:75", "jpeg:50", "blur:1", "blur:2", "resize:0.5", "noise:8"]


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


def blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def noise(img: Image.Image, std: float, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.asarray(img).astype(np.float32)
    a = a + rng.normal(0, float(std), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def get_perturbation(spec: str | None):
    """Turn a spec string like 'jpeg:75' into a function PIL -> PIL (or None)."""
    if spec is None or spec == "none":
        return None
    name, _, val = spec.partition(":")
    fns = {"jpeg": jpeg, "blur": blur, "resize": resize, "noise": noise}
    if name not in fns or not val:
        raise ValueError(f"Unknown perturbation '{spec}'. Use e.g. none, jpeg:75, blur:1, resize:0.5, noise:8")
    v = float(val)
    return lambda img: fns[name](img, v)
