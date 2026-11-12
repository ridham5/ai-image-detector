"""The three detectors behind one common interface.

Every detector has:
    fit(train_items, ...)            train on a list of (path, label)
    predict_proba(images) -> array   P(fake) for a list of PIL images
    save(path) / load(path)

so evaluate.py, the robustness experiment and the demo treat all three the same.

  FFTSVMDetector     Method 1  hand-crafted Fourier features + RBF SVM     (scikit-learn)
  CNNDetector        Method 2  ImageNet ResNet-50, fine-tuned end to end    (PyTorch / torchvision)
  CLIPProbeDetector  Method 3  frozen CLIP image encoder + logistic regr.   (open_clip + scikit-learn)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed

from .data import load_rgb
from .fft_features import fft_feature


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------------------------------------------------------- Method 1
class FFTSVMDetector:
    name = "fft_svm"

    def __init__(self, size: int = 128, grid: int = 8, C: float = 10.0, gamma="scale", n_jobs: int = -1):
        self.size, self.grid, self.C, self.gamma, self.n_jobs = size, grid, C, gamma, n_jobs
        self.clf = None

    def features(self, images) -> np.ndarray:
        return np.stack(Parallel(n_jobs=self.n_jobs)(
            delayed(fft_feature)(im, self.size, self.grid) for im in images))

    def features_from_paths(self, paths, perturb=None) -> np.ndarray:
        def one(p):
            im = load_rgb(p)
            if perturb is not None:
                im = perturb(im)
            return fft_feature(im, self.size, self.grid)
        return np.stack(Parallel(n_jobs=self.n_jobs)(delayed(one)(p) for p in paths))

    def fit(self, train_items, **_):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        t = time.time()
        X = self.features_from_paths([p for p, _ in train_items])
        y = np.array([l for _, l in train_items])
        print(f"[fft_svm] features {X.shape} in {time.time() - t:.1f}s; training SVM ...")
        # SVM outputs a score, not a probability; Platt calibration (sigmoid) turns it into P(fake)
        svm = SVC(kernel="rbf", C=self.C, gamma=self.gamma, random_state=0)
        self.clf = make_pipeline(StandardScaler(),
                                 CalibratedClassifierCV(svm, method="sigmoid", cv=5, ensemble=False))
        self.clf.fit(X, y)
        return self

    def predict_proba_paths(self, paths, perturb=None):
        return self.clf.predict_proba(self.features_from_paths(paths, perturb))[:, 1]

    def predict_proba(self, images):
        return self.clf.predict_proba(self.features(images))[:, 1]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"size": self.size, "grid": self.grid, "C": self.C, "gamma": self.gamma, "clf": self.clf}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        det = cls(size=d["size"], grid=d["grid"], C=d["C"], gamma=d["gamma"])
        det.clf = d["clf"]
        return det


# --------------------------------------------------------------------------- Method 2
def _device():
    """Pick the device: env var DETECTOR_DEVICE (cpu / mps / cuda) wins, else the fastest available."""
    import os
    import torch
    forced = os.environ.get("DETECTOR_DEVICE", "").strip().lower()
    if forced in {"cpu", "mps", "cuda"}:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CNNDetector:
    """ResNet-50 pretrained on ImageNet; last layer replaced by one logit (fake vs real)."""
    name = "resnet50"

    def __init__(self, img_size: int = 224, pretrained: bool = True):
        self.img_size, self.pretrained = img_size, pretrained
        self.device = _device()
        self.model = self._build(pretrained).to(self.device)

    def _build(self, pretrained):
        import torch.nn as nn
        from torchvision.models import ResNet50_Weights, resnet50
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m

    def transforms(self, train: bool):
        from torchvision import transforms as T
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        s = self.img_size
        if train:
            # Resize keeps small CIFAKE images usable; crop + flip are light augmentations.
            return T.Compose([T.Resize(int(s * 1.14)), T.RandomCrop(s), T.RandomHorizontalFlip(), T.ToTensor(), norm])
        return T.Compose([T.Resize(int(s * 1.14)), T.CenterCrop(s), T.ToTensor(), norm])

    def fit(self, train_items, val_items=None, epochs: int = 3, batch_size: int = 64, lr: float = 1e-4,
            weight_decay: float = 1e-4, num_workers: int = 2, save_path: str | None = None, **_):
        import torch
        from torch.utils.data import DataLoader

        from .data import ImageListDataset

        dl = DataLoader(ImageListDataset(train_items, self.transforms(True)), batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=self.device == "cuda", drop_last=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(dl)))
        loss_fn = torch.nn.BCEWithLogitsLoss()
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best, history = -1.0, []
        print(f"[resnet50] {len(dl)} steps per epoch (batch {batch_size}); first step can take a while ...", flush=True)
        for ep in range(1, epochs + 1):
            self.model.train()
            t, run, seen = time.time(), 0.0, 0
            for i, (x, y) in enumerate(dl, 1):
                x, y = x.to(self.device, non_blocking=True), y.float().to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = loss_fn(self.model(x).squeeze(1), y)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Training loss became {loss.item()} at epoch {ep} step {i} - training blew up. "
                        f"On a Mac, re-run with --device cpu (or use Google Colab).")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # keep updates small
                scaler.step(opt)
                scaler.update()
                sched.step()
                if loss.item() > 50:
                    raise RuntimeError(
                        f"Training loss jumped to {loss.item():.3g} at epoch {ep} step {i} - training blew up. "
                        f"On a Mac, re-run with --device cpu (or use Google Colab).")
                run += loss.item() * len(y)
                seen += len(y)
                if i == 1 or i % 10 == 0 or i == len(dl):
                    per = (time.time() - t) / i
                    print(f"  epoch {ep} step {i}/{len(dl)} loss {run / seen:.4f}  "
                          f"{per:.1f}s/step, ~{per * (len(dl) - i) / 60:.1f} min left in epoch", flush=True)
            msg = f"[resnet50] epoch {ep}/{epochs} train_loss {run / max(seen, 1):.4f} ({time.time() - t:.0f}s)"
            row = {"epoch": ep, "train_loss": run / max(seen, 1)}
            if val_items:
                from .metrics import compute_metrics
                p = self.predict_proba_paths([q for q, _ in val_items], batch_size=batch_size, num_workers=num_workers)
                m = compute_metrics([l for _, l in val_items], p)
                msg += f" val_acc {m['accuracy']:.4f} val_auc {m['auc']:.4f}"
                row.update(val_acc=m["accuracy"], val_auc=m["auc"])
                if save_path and m["accuracy"] > best:  # keep the best epoch on validation data
                    best = m["accuracy"]
                    self.save(save_path)
                    msg += "  <- saved (best so far)"
            history.append(row)
            print(msg, flush=True)
        if save_path and not val_items:
            self.save(save_path)
        if save_path:
            Path(save_path).with_suffix(".history.json").write_text(json.dumps(history, indent=2))
        if save_path and val_items:  # reload the best epoch
            self.model.load_state_dict(torch.load(save_path, map_location=self.device)["state_dict"])
        return self

    def _predict_loader(self, loader):
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for x, _ in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    logits = self.model(x.to(self.device)).squeeze(1)
                out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict_proba_paths(self, paths, perturb=None, batch_size: int = 128, num_workers: int = 2):
        from torch.utils.data import DataLoader

        from .data import ImageListDataset
        ds = ImageListDataset([(p, 0) for p in paths], self.transforms(False), perturb)
        return self._predict_loader(DataLoader(ds, batch_size=batch_size, num_workers=num_workers))

    def predict_proba(self, images):
        import torch
        tf = self.transforms(False)
        x = torch.stack([tf(im) for im in images])
        return self._predict_loader([(x, None)])

    def save(self, path):
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"img_size": self.img_size, "state_dict": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path):
        import torch
        d = torch.load(path, map_location="cpu")
        det = cls(img_size=d["img_size"], pretrained=False)
        det.model.load_state_dict(d["state_dict"])
        det.model.to(det.device).eval()
        return det


# --------------------------------------------------------------------------- Method 3
class CLIPProbeDetector:
    """Frozen CLIP image encoder + a linear (logistic regression) classifier on its features.

    This follows the idea of UnivFD (Ojha et al., CVPR 2023): CLIP's features were
    never trained to spot fakes, so the classifier can't overfit to one generator's
    artifacts, which tends to help on generators not seen during training.
    """
    name = "clip_probe"

    def __init__(self, arch: str = "ViT-B-32", pretrained: str | None = "openai", C: float = 1.0):
        self.arch, self.pretrained, self.C = arch, pretrained, C
        self.device = _device()
        self.clf = None
        self._load_clip()

    def _load_clip(self):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.arch, pretrained=self.pretrained)
        self.model = self.model.to(self.device).eval()

    def _encode_loader(self, loader):
        import torch
        feats = []
        with torch.no_grad():
            for x, _ in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    f = self.model.encode_image(x.to(self.device))
                f = f.float()
                feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())  # L2-normalise
        return np.concatenate(feats) if feats else np.zeros((0, 1))

    def features_from_paths(self, paths, perturb=None, batch_size: int = 128, num_workers: int = 2):
        from torch.utils.data import DataLoader

        from .data import ImageListDataset
        ds = ImageListDataset([(p, 0) for p in paths], self.preprocess, perturb)
        return self._encode_loader(DataLoader(ds, batch_size=batch_size, num_workers=num_workers))

    def fit(self, train_items, batch_size: int = 128, num_workers: int = 2, **_):
        from sklearn.linear_model import LogisticRegression
        t = time.time()
        X = self.features_from_paths([p for p, _ in train_items], batch_size=batch_size, num_workers=num_workers)
        y = np.array([l for _, l in train_items])
        print(f"[clip_probe] CLIP features {X.shape} in {time.time() - t:.1f}s; training logistic regression ...")
        self.clf = LogisticRegression(C=self.C, max_iter=5000)
        self.clf.fit(X, y)
        return self

    def predict_proba_paths(self, paths, perturb=None, batch_size: int = 128, num_workers: int = 2):
        return self.clf.predict_proba(self.features_from_paths(paths, perturb, batch_size, num_workers))[:, 1]

    def predict_proba(self, images):
        import torch
        x = torch.stack([self.preprocess(im) for im in images])
        return self.clf.predict_proba(self._encode_loader([(x, None)]))[:, 1]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"arch": self.arch, "pretrained": self.pretrained, "C": self.C, "clf": self.clf}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        det = cls(arch=d["arch"], pretrained=d["pretrained"], C=d["C"])
        det.clf = d["clf"]
        return det


# --------------------------------------------------------------------------- helpers
METHODS = {"fft_svm": FFTSVMDetector, "resnet50": CNNDetector, "clip_probe": CLIPProbeDetector}
DEFAULT_FILES = {"fft_svm": "fft_svm.joblib", "resnet50": "resnet50.pt", "clip_probe": "clip_probe.joblib"}


def load_detector(method: str, path: str):
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}")
    return METHODS[method].load(path)
