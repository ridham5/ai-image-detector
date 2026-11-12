# Real or AI-generated? Detecting synthetic images with frequency, CNN and CLIP features

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ridham5/ai-image-detector/blob/main/notebooks/colab_run_all.ipynb)

CPS843 Intro to Computer Vision, Fall 2026, Toronto Metropolitan University — term project code.

We compare three ways of telling camera photos from AI-generated images:

| # | Method | Idea | Library |
|---|---|---|---|
| 1 | **FFT + SVM** | AI generators leave grid-like up-sampling traces that show up in the Fourier spectrum | NumPy, scikit-learn |
| 2 | **ResNet-50** | Fine-tune an ImageNet CNN end to end | PyTorch, torchvision |
| 3 | **CLIP + linear probe** | Keep a CLIP image encoder frozen, train only a linear classifier on its features | open_clip, scikit-learn |

and run three experiments: **(1)** main comparison on CIFAKE, **(2)** robustness to JPEG / blur / resize / noise,
**(3)** generalisation to generators not seen in training (GenImage). Grad-CAM shows where ResNet-50 looks,
and a Gradio app gives a live demo.

---

## Quick start (Google Colab — recommended)

1. Click **Open in Colab** above (or open `notebooks/colab_run_all.ipynb` from this repo in Colab).
2. `Runtime → Change runtime type → T4 GPU → Save`.
3. Run the cells top to bottom. Models and results are backed up to your Google Drive (`MyDrive/cps843-ai-image-detector/outputs`).
4. Tables and figures for the report end up in `results/`.

The notebook starts with small settings (cell 5) for a quick check; raise them for the final run.

## Running on your own machine

1. Install Python 3.10+.
2. Install PyTorch from <https://pytorch.org> (pick your OS and CUDA version).
3. `pip install -r requirements.txt`
4. `python scripts/download_cifake.py`
5. `bash scripts/run_all.sh`

## Check the code works without any real data (1 minute, CPU)

```bash
python scripts/make_dummy_data.py --out data/dummy
python scripts/train.py    --method fft_svm --train-dir data/dummy/train
python scripts/evaluate.py --method fft_svm --test-dir data/dummy/test --test-name dummy --robustness
python scripts/make_tables.py --main-test dummy
```
(The dummy "fake" images have an obvious grid pattern, so near-perfect scores here mean nothing.)

---

## Repository layout

```
src/
  data.py           finds images and labels them from folder names (REAL/FAKE, nature/ai)
  fft_features.py   Method 1 features: log power spectrum, radial profile, pooled spectrum
  detectors.py      the 3 detectors behind one interface: fit / predict_proba / save / load
  perturb.py        JPEG, blur, resize, noise for the robustness experiment
  metrics.py        accuracy, precision, recall, F1, ROC-AUC, per-class accuracy -> results.csv
  gradcam.py        Grad-CAM heatmaps for ResNet-50
  plotstyle.py      one consistent look for every report figure
scripts/
  download_cifake.py   get the dataset from Kaggle
  train.py             train one method
  evaluate.py          test one method (optionally under all perturbations)
  figures.py           dataset figures: samples, mean Fourier spectra (no model needed)
  make_tables.py       results.csv -> Markdown + LaTeX tables + bar charts
  gradcam_figure.py    Grad-CAM figure
  make_dummy_data.py   tiny synthetic dataset for testing the code
  run_all.sh           the whole pipeline in one command
notebooks/colab_run_all.ipynb   the whole pipeline on Colab
app.py                           Gradio demo
docs/EXPLAINER.md                plain-English guide to every part + the equations, for writing the report
```

## Data

**CIFAKE** (Bird & Lotfi, 2024) — 60,000 real CIFAR-10 images + 60,000 Stable Diffusion v1.4 images, 32×32 px,
split 100k train / 20k test. Kaggle: `birdy654/cifake-real-and-ai-generated-synthetic-images`.
`scripts/download_cifake.py` fetches it and links it to `data/cifake/`.

**GenImage** (Zhu et al., NeurIPS 2023) — real ImageNet photos + images from 8 generators (Midjourney, SD v1.4/v1.5,
ADM, GLIDE, Wukong, VQDM, BigGAN). It is very large, so download only one or two generators' **validation** parts
from the GenImage GitHub page and arrange them as:
```
data/genimage/midjourney/val/ai/*.png
data/genimage/midjourney/val/nature/*.JPEG
data/genimage/biggan/val/ai/...    (etc.)
```
Any folder of images works as long as real images sit under a folder named `real`/`nature` and fakes under `fake`/`ai`
(any capitals). So you can also test on images you make yourself with a current generator.

## Outputs (all in `results/`)

| File | Use in the report |
|---|---|
| `table_main.tex` | Experiments: main comparison table |
| `table_robustness.tex`, `fig_robustness.png` | Experiments: robustness |
| `table_generalization.tex`, `fig_generalization.png` | Experiments: generalisation |
| `fig_samples.png` | Data section: example images |
| `fig_spectrum.png`, `fig_radial.png` | Method section: why the FFT method works |
| `fig_gradcam.png` | Experiments: qualitative analysis |
| `results.csv` | every number, one row per run |

The `.tex` tables drop straight into the IEEE LaTeX template (`\input{table_main.tex}`; add `\usepackage{booktabs,graphicx}`).

## Working on the code as a team

```bash
git clone https://github.com/ridham5/ai-image-detector.git
cd ai-image-detector
git checkout -b <your-name>/<feature>     # work on a branch
git add -A && git commit -m "What you changed"
git push -u origin <your-name>/<feature>  # then open a Pull Request on GitHub
```
`data/` and `models/` are in `.gitignore`, so large files stay out of the repo.

## Third-party code and models (cite these in the report)

- PyTorch and torchvision (ResNet-50 ImageNet weights) — Paszke et al., NeurIPS 2019
- OpenCLIP (`open_clip_torch`) — Ilharco et al., 2021; CLIP — Radford et al., ICML 2021
- scikit-learn (SVM, logistic regression, metrics) — Pedregosa et al., JMLR 2011
- Grad-CAM method — Selvaraju et al., ICCV 2017
- Gradio (demo)
- CIFAKE dataset — Bird & Lotfi, IEEE Access 2024; GenImage — Zhu et al., NeurIPS 2023
