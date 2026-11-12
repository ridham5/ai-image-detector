"""Dataset figures for the report (no model needed).

    python scripts/figures.py --data-dir data/cifake/test

Writes to results/:
    fig_samples.png   a grid of real vs AI-generated example images
    fig_spectrum.png  average log power spectrum of real images, fake images, and their difference
                      -> visual evidence for WHY the FFT method can work
    fig_radial.png    average radial spectrum (energy vs. frequency), real vs fake
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import list_images, load_rgb  # noqa: E402
from src.fft_features import log_power_spectrum, radial_profile  # noqa: E402
from src.plotstyle import INK2, setup  # noqa: E402

REAL_C, FAKE_C = "#2a78d6", "#eb6834"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-spectrum", type=int, default=2000, help="images per class to average")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = setup()
    plt.rcParams["axes.grid"] = False

    items = list_images(a.data_dir, a.n_spectrum, seed=1)
    real = [p for p, l in items if l == 0]
    fake = [p for p, l in items if l == 1]

    # samples
    k = 6
    fig, axes = plt.subplots(2, k, figsize=(7.0, 2.6))
    for row, (paths, name) in enumerate([(real, "Real"), (fake, "AI-generated")]):
        for j in range(k):
            ax = axes[row, j]
            ax.imshow(load_rgb(paths[j]))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.savefig(out / "fig_samples.png")
    plt.close(fig)

    # mean spectra
    def mean_spec(paths):
        return np.mean([log_power_spectrum(load_rgb(p), a.size) for p in paths], axis=0)
    sr, sf = mean_spec(real), mean_spec(fake)
    diff = sf - sr
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), layout="constrained")
    vmin, vmax = np.percentile(np.stack([sr, sf]), [1, 99.5])
    for ax, s, title in [(axes[0], sr, "Real (mean log power)"), (axes[1], sf, "AI-generated (mean log power)")]:
        im = ax.imshow(s, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02, label="log power")
    lim = np.percentile(np.abs(diff), 99)
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)  # diverging, grey-white midpoint = no difference
    axes[2].set_title("Fake − Real", fontsize=8)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    fig.savefig(out / "fig_spectrum.png")
    plt.close(fig)

    # radial profile
    rr = np.array([radial_profile(log_power_spectrum(load_rgb(p), a.size)) for p in real])
    rf = np.array([radial_profile(log_power_spectrum(load_rgb(p), a.size)) for p in fake])
    plt.rcParams["axes.grid"] = True
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    f = np.arange(rr.shape[1]) / (a.size)  # cycles per pixel
    for arr, c, lab in [(rr, REAL_C, "Real"), (rf, FAKE_C, "AI-generated")]:
        m, s = arr.mean(0), arr.std(0)
        ax.plot(f, m, color=c, lw=1.6, label=lab)
        ax.fill_between(f, m - s, m + s, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("Spatial frequency (cycles / pixel)")
    ax.set_ylabel("Mean log power")
    ax.legend(fontsize=8)
    ax.tick_params(colors=INK2)
    fig.savefig(out / "fig_radial.png")
    plt.close(fig)
    print(f"wrote fig_samples.png, fig_spectrum.png, fig_radial.png to {out}/  ({len(real)} real, {len(fake)} fake)")


if __name__ == "__main__":
    main()
