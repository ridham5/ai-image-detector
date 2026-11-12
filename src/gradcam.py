"""Grad-CAM (Selvaraju et al., ICCV 2017) for the ResNet-50 detector.

Grad-CAM answers "which parts of the image made the model say FAKE?".
For the last convolution block (layer4) with feature maps A^k:
    weight  a_k = mean over all pixels of  d(score) / d(A^k)
    heatmap L   = ReLU( sum_k a_k * A^k )
then L is up-sampled to the image size and drawn on top of the image.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


class GradCAM:
    def __init__(self, cnn_detector):
        self.det = cnn_detector
        self.model = cnn_detector.model
        self._acts = self._grads = None
        # forward hook stores the feature maps and attaches a hook to their gradient
        self.model.layer4[-1].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self._acts = out.detach()
        if out.requires_grad:
            out.register_hook(self._save_grads)

    def _save_grads(self, grad):
        self._grads = grad.detach()

    def __call__(self, img: Image.Image):
        """Return (p_fake, heatmap in [0,1] at the model's input resolution)."""
        import torch
        self.model.eval()
        x = self.det.transforms(False)(img).unsqueeze(0).to(self.det.device)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit = self.model(x).squeeze()
        logit.backward()  # gradient of the "fake" score
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self._acts).sum(dim=1)).squeeze(0)
        cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                                              align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return float(torch.sigmoid(logit).item()), cam.cpu().numpy()


def overlay(img: Image.Image, cam: np.ndarray, size: int, alpha: float = 0.45) -> Image.Image:
    """Draw the heatmap on the (center-cropped) image the model actually saw."""
    import matplotlib
    from torchvision import transforms as T
    base = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size)])(img).convert("RGB")
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    return Image.blend(base, Image.fromarray(heat).resize(base.size), alpha)
