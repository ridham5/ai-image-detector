"""Grad-CAM figure for the report: which regions made ResNet-50 decide "fake"?

    python scripts/gradcam_figure.py --test-dir data/cifake/test --n 4

Picks n correctly-classified real, n correctly-classified fake and (if any) n mistakes,
and writes results/fig_gradcam.png (image row + heatmap row).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import list_images, load_rgb  # noqa: E402
from src.detectors import CNNDetector  # noqa: E402
from src.gradcam import GradCAM, overlay  # noqa: E402
from src.plotstyle import setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/resnet50.pt")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--n", type=int, default=4, help="examples per group")
    ap.add_argument("--pool", type=int, default=300, help="images to search through")
    ap.add_argument("--out", default="results/fig_gradcam.png")
    a = ap.parse_args()

    det = CNNDetector.load(a.model)
    cam = GradCAM(det)
    groups = {"Real, correct": [], "Fake, correct": [], "Mistake": []}
    for path, label in list_images(a.test_dir, a.pool, seed=3):
        img = load_rgb(path)
        p, heat = cam(img)
        pred = int(p >= 0.5)
        key = "Mistake" if pred != label else ("Fake, correct" if label else "Real, correct")
        if len(groups[key]) < a.n:
            groups[key].append((img, heat, p, label))
        if all(len(v) >= a.n for v in groups.values()):
            break
    groups = {k: v for k, v in groups.items() if v}
    cells = [(k, *x) for k, v in groups.items() for x in v]

    plt = setup()
    plt.rcParams["axes.grid"] = False
    fig, axes = plt.subplots(2, len(cells), figsize=(7.0, 7.0 / len(cells) * 2.3), squeeze=False)
    for j, (group, img, heat, p, label) in enumerate(cells):
        size = det.img_size
        from torchvision import transforms as T
        axes[0, j].imshow(T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size)])(img))
        axes[1, j].imshow(overlay(img, heat, size))
        axes[0, j].set_title(f"{group}\nP(fake)={p:.2f}", fontsize=7)
        for i in (0, 1):
            axes[i, j].axis("off")
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
