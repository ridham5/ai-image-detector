# How the project works — plain-English guide for writing the report

Use this to **understand** each part, then write the report **in your own words**. The report is checked by
Turnitin, so don't copy sentences from here. The equations are standard; write them in LaTeX and explain them yourself.

**Hook for the whole project:** *3 methods, 3 experiments, 1 demo.*

---

## The problem

Binary classification. Input: an image. Output: `p` = probability it is AI-generated.
Label 0 = real, 1 = fake. We say "fake" when `p ≥ 0.5`.

**Why it matters:** AI images are now realistic enough to fool people, which feeds misinformation and fraud.
Detectors trained on one generator often fail on newer ones. That is why experiment 3 is the most interesting.

---

## Pipeline

```
image ──► preprocess ──┬──► FFT spectrum ──► radial + pooled features ──► SVM ──────────► p(fake)
                        ├──► ResNet-50 (fine-tuned) ──────────────────────► sigmoid ──────► p(fake) ──► Grad-CAM
                        └──► CLIP encoder (frozen) ──► normalised features ──► logistic reg. ► p(fake)
```
Draw this as the pipeline figure for the Method section (draw.io or PowerPoint, exported as PDF/PNG).

---

## Method 1 — FFT + SVM (the "classic CV" baseline) · Person A

**Plain English:** generators build images by up-sampling small grids again and again. That leaves a faint
repeating pattern, which you can't see in the pixels but can see clearly in the Fourier spectrum.

**Hook:** *Fakes have a fingerprint in frequency.*

**Steps (in `src/fft_features.py`):**
1. Convert to grayscale, resize to 128×128, subtract the mean.
2. 2D DFT, shifted so frequency (0,0) sits in the centre:

   $$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y)\,e^{-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)}$$

3. Log power spectrum: $L(u,v)=\log\left(|F(u,v)|^2+\varepsilon\right)$
4. **Radial profile** (energy vs. frequency) — average $L$ over rings of radius $r$:

   $$R(r)=\frac{1}{|S_r|}\sum_{(u,v)\in S_r} L(u,v),\qquad S_r=\{(u,v):\lfloor\sqrt{u^2+v^2}\rfloor=r\}$$

   (64 values, rescaled to 0–1 per image)
5. **Pooled spectrum:** average $L$ in an 8×8 grid of blocks (64 values). This keeps grid peaks that sit off the
   axes, which the ring average would blur away.
6. Feature vector = 128 numbers → standardise → **SVM with RBF kernel**:

   $$f(\mathbf{x})=\sum_i \alpha_i y_i \exp\left(-\gamma\lVert\mathbf{x}_i-\mathbf{x}\rVert^2\right)+b$$

7. Platt scaling turns the SVM score into a probability: $p=\frac{1}{1+e^{Af(\mathbf{x})+B}}$

**Settings:** image size 128, C = 10, γ = "scale", 10,000 training images per class.
**Figures to use:** `fig_spectrum.png` and `fig_radial.png` show *why* it works. They need no model, so make them early.
**Course link:** Fourier transform and filtering (weeks 3–4).

---

## Method 2 — ResNet-50 fine-tuning · Person B

**Plain English:** take a CNN that already learned general visual features on 1.2M ImageNet photos, swap its last
layer for one output ("fake score"), and retrain the whole network on our data.

**Hook:** *Borrow ImageNet's eyes, teach them one new question.*

- Output: $p=\sigma(\mathbf{w}^\top\phi(\mathbf{x})+b)$, where $\phi$ = the 2048-number ResNet feature and $\sigma$ = sigmoid.
- Loss (binary cross-entropy):

  $$\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]$$

- Optimiser AdamW, learning rate 1e-4 with cosine decay, weight decay 1e-4, batch 64, 3 epochs, mixed precision.
- Input 224×224 (CIFAKE's 32×32 images are up-scaled). Train augmentation: random crop + horizontal flip.
- 10% of the training set is held out for validation; the epoch with the best validation accuracy is kept.
  The learning curve is saved in `models/resnet50.history.json`, so you can plot loss per epoch.

**Grad-CAM** (`src/gradcam.py`) — which pixels pushed the score up? For feature maps $A^k$ of the last block:

$$\alpha_k=\frac{1}{Z}\sum_{i,j}\frac{\partial y}{\partial A^k_{ij}},\qquad L_{\text{Grad-CAM}}=\mathrm{ReLU}\Big(\sum_k\alpha_k A^k\Big)$$

**Course link:** deep learning and CNNs (week 11).

---

## Method 3 — CLIP + linear probe · Person B

**Plain English:** CLIP learned to match 400M images with their captions, so its features describe images very
generally. We **freeze** CLIP and train only a straight-line (linear) classifier on its features. Because CLIP was
never trained to spot one generator's artifacts, the classifier can't memorise them. This is why it tends to
generalise better to new generators (the idea behind UnivFD, Ojha et al., CVPR 2023).

**Hook:** *Frozen giant + tiny classifier = less overfitting.*

- Feature: $\mathbf{z}=E(\mathbf{x})/\lVert E(\mathbf{x})\rVert$ (CLIP ViT-B/32 image encoder, L2-normalised, 512 numbers)
- Classifier: logistic regression $p=\sigma(\mathbf{w}^\top\mathbf{z}+b)$ with L2 regularisation (C = 1).
- Only ~513 numbers are learned, versus ~23.5M for ResNet-50. That is a nice point for the discussion.
- Option to try: a bigger CLIP (`--clip-arch ViT-L-14`), which UnivFD used. It is slower but usually better.

---

## The experiments · Person C

| # | Question | How | Output |
|---|---|---|---|
| 1 | Which method is best on images like its training data? | Train on CIFAKE train, test on CIFAKE test | `table_main` |
| 2 | Does it survive real-world processing? | Re-test after JPEG 95/75/50, blur σ=1/2, resize ½, noise σ=8 | `table_robustness`, `fig_robustness` |
| 3 | Does it work on generators it never saw? | Train on CIFAKE (SD v1.4), test on GenImage subsets | `table_generalization`, `fig_generalization` |
| + | What does the CNN look at? | Grad-CAM on right and wrong examples | `fig_gradcam` |

**Metrics** (TP = fake called fake, FP = real called fake, and so on):
- Accuracy $=\frac{TP+TN}{N}$; Precision $=\frac{TP}{TP+FP}$; Recall $=\frac{TP}{TP+FN}$; $F_1=\frac{2PR}{P+R}$
- **ROC-AUC:** the chance that a random fake scores higher than a random real image. 0.5 = guessing, 1 = perfect.
  It doesn't depend on the 0.5 threshold, which matters in experiment 3.
- **Real acc / Fake acc** (accuracy on each class): shows when a model just calls everything "fake".
  On new generators, a common failure is high real accuracy with very low fake accuracy, even when AUC is decent.
  That means the threshold is off rather than the features being useless, which is a good discussion point.

---

## What to look for when analysing results (questions, not answers — test them)

1. Is ResNet-50 the best on clean CIFAKE? What did fine-tuning 23.5M parameters buy over a 513-parameter probe?
2. Does FFT + SVM collapse under JPEG and blur? Why would it? (JPEG and blur remove high frequencies, which is
   where the fingerprint lives.)
3. Does CLIP lose less accuracy on unseen generators than ResNet-50?
4. Are GAN generators (BigGAN) easier or harder to catch than diffusion models after training on Stable Diffusion?
5. Do the Grad-CAM maps highlight the object, or texture and background? What does that suggest?
6. **Dataset bias check:** do real and fake images in a dataset differ in file format, size or compression?
   If so, a model might learn that shortcut instead of real generation artifacts. Mention this as a limitation.

---

## Mapping to the report and marks

| Report section | Marks | Source in this repo | Who |
|---|---|---|---|
| Abstract + Intro + Related work | 4 | papers + this guide + GitHub link | all (C leads) |
| Technical approach | 6 | Methods 1–3 above, pipeline figure, `fig_spectrum` | A + B |
| Experiments + analysis | 6 | all tables/figures in `results/`, questions above | C |
| Format, figures, conclusion, roles | 4 | IEEE template, `.tex` tables | C, all review |

## Key references (check each before citing)

- Wang et al., "CNN-generated images are surprisingly easy to spot… for now," CVPR 2020
- Frank et al., "Leveraging frequency analysis for deep fake image recognition," ICML 2020
- Durall et al., "Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions," CVPR 2020
- Ojha et al., "Towards universal fake image detectors that generalize across generative models," CVPR 2023
- Corvi et al., "On the detection of synthetic images generated by diffusion models," ICASSP 2023
- Bird & Lotfi, "CIFAKE: Image classification and explainable identification of AI-generated synthetic images," IEEE Access 2024
- Zhu et al., "GenImage: A million-scale benchmark for detecting AI-generated image," NeurIPS 2023
- He et al., "Deep residual learning for image recognition," CVPR 2016
- Radford et al., "Learning transferable visual models from natural language supervision," ICML 2021
- Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," ICCV 2017
