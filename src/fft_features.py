"""Method 1: frequency-domain features (the classic computer vision baseline).

Idea: image generators build images by repeatedly up-sampling small feature
maps. Up-sampling leaves periodic, grid-like traces that are hard to see in
pixels but show up clearly as peaks / unusual energy in the 2D Fourier
spectrum, especially at high frequencies (Durall et al. 2020, Frank et al. 2020).

For each image we compute:
  1. grayscale, resized to S x S
  2. the 2D DFT  F(u,v), shifted so frequency (0,0) is in the centre
  3. the log power spectrum  L(u,v) = log(|F(u,v)|^2 + eps)
  4. feature A: the azimuthal (radial) average of L -> 1D curve of energy vs. frequency
  5. feature B: L average-pooled to a coarse P x P grid -> keeps off-axis peaks
     that the radial average would wash out
The final feature vector is [A, B]. An SVM is trained on these vectors.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

EPS = 1e-8


def log_power_spectrum(img: Image.Image, size: int = 128) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((size, size), Image.BICUBIC), dtype=np.float32)
    g = g - g.mean()  # remove the DC term so it doesn't dominate
    F = np.fft.fftshift(np.fft.fft2(g))
    return np.log(np.abs(F) ** 2 + EPS)


def radial_profile(spec: np.ndarray) -> np.ndarray:
    """Average the spectrum over rings of equal distance from the centre."""
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(spec.shape)
    r = np.hypot(y - cy, x - cx).astype(int)
    sums = np.bincount(r.ravel(), weights=spec.ravel())
    counts = np.bincount(r.ravel())
    prof = sums / np.maximum(counts, 1)
    return prof[: min(cy, cx)]  # keep radii that fit inside the image


def pooled_spectrum(spec: np.ndarray, grid: int = 8) -> np.ndarray:
    h, w = spec.shape
    s = spec[: h - h % grid, : w - w % grid]
    return s.reshape(grid, s.shape[0] // grid, grid, s.shape[1] // grid).mean(axis=(1, 3)).ravel()


def fft_feature(img: Image.Image, size: int = 128, grid: int = 8) -> np.ndarray:
    spec = log_power_spectrum(img, size)
    rad = radial_profile(spec)
    # normalise the radial curve per image (shape matters, not overall brightness)
    rad = (rad - rad.min()) / (rad.max() - rad.min() + EPS)
    pooled = pooled_spectrum(spec, grid)
    pooled = (pooled - pooled.mean()) / (pooled.std() + EPS)
    return np.concatenate([rad, pooled]).astype(np.float32)
