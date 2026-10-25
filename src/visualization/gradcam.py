"""Grad-CAM: which image regions pushed the model towards a class."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.classifier import Classifier


class GradCAM:
    """Hooks one conv layer; call close() when finished so the hooks do not pile up."""

    def __init__(self, model: Classifier, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [target_layer.register_forward_hook(self._on_forward)]

    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:  # the gradient arrives later, during backward()
            output.register_hook(lambda g: setattr(self, "gradients", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """x: [1,3,H,W]. Returns an [H,W] map in [0,1] for class_idx (default: predicted class)."""
        with torch.enable_grad():  # callers may be inside no_grad
            self.model.zero_grad()
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # one weight per channel
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        span = cam.max() - cam.min()
        if span < 1e-12:  # constant map (e.g. nothing active): avoid dividing by zero
            return np.zeros_like(cam)
        return (cam - cam.min()) / span

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Blend a jet-coloured heatmap over the image (cam is resized to the image size)."""
    from matplotlib import colormaps
    image = image.convert("RGB")
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    base = np.asarray(image) / 255.0
    return Image.fromarray((((1 - alpha) * base + alpha * colored) * 255).astype(np.uint8))


def denormalize(x: torch.Tensor) -> Image.Image:
    """Undo the ImageNet normalisation of a [3,H,W] tensor so it can be shown."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    arr = (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def save_example_grid(items: list[dict], title: str, path: str) -> None:
    """One figure: a column per example, top row image, bottom row Grad-CAM overlay.

    Each item: {"image": PIL, "cam": ndarray, "true": str, "pred": str, "conf": float}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = max(len(items), 1)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for j, it in enumerate(items):
        axes[0, j].imshow(it["image"])
        axes[1, j].imshow(overlay(it["image"], it["cam"]))
        axes[0, j].set_title(f"true: {it['true']}\npred: {it['pred']} ({it['conf']:.2f})", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
