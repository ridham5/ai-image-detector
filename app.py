"""Live demo: upload any image, see what each trained detector thinks + a Grad-CAM heatmap.

    python app.py            (in Colab: add --share to get a public link)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.detectors import DEFAULT_FILES, load_detector  # noqa: E402
from src.plotstyle import METHOD_LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()

    import gradio as gr

    dets = {}
    for m, f in DEFAULT_FILES.items():
        p = Path(a.models_dir) / f
        if p.exists():
            print(f"loading {m} from {p}")
            dets[m] = load_detector(m, str(p))
    if not dets:
        sys.exit(f"No trained models found in {a.models_dir}/ - run scripts/train.py first.")

    cam = None
    if "resnet50" in dets:
        from src.gradcam import GradCAM, overlay
        cam = GradCAM(dets["resnet50"])

    def predict(img):
        if img is None:
            return {}, None
        img = img.convert("RGB")
        scores = {}
        for m, d in dets.items():
            p = float(d.predict_proba([img])[0])
            scores[f"{METHOD_LABELS[m]}: AI-generated"] = p
        heat = None
        if cam is not None:
            _, h = cam(img)
            heat = overlay(img, h, dets["resnet50"].img_size)
        return scores, heat

    with gr.Blocks(title="Real or AI-generated?") as demo:
        gr.Markdown("## Real or AI-generated?\nUpload an image. Each bar is one detector's probability that the "
                    "image was made by AI. The heatmap shows where ResNet-50 looked.")
        with gr.Row():
            inp = gr.Image(type="pil", label="Image")
            with gr.Column():
                out = gr.Label(label="P(AI-generated) per detector", num_top_classes=len(dets))
                heat = gr.Image(label="Grad-CAM (ResNet-50)")
        inp.change(predict, inp, [out, heat])
    demo.launch(share=a.share)


if __name__ == "__main__":
    main()
