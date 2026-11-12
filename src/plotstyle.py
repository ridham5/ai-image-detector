"""One shared look for every report figure (crisp, print-friendly, colour-blind-checked palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fixed colour per method, always in this order (never re-coloured between figures)
METHOD_COLORS = {"fft_svm": "#2a78d6", "resnet50": "#eb6834", "clip_probe": "#1baf7a"}
METHOD_LABELS = {"fft_svm": "FFT + SVM", "resnet50": "ResNet-50", "clip_probe": "CLIP + linear probe"}
METHOD_ORDER = ["fft_svm", "resnet50", "clip_probe"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
    })
    return plt
