"""Test-set evaluation, Grad-CAM example selection and single-image prediction."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.transforms import build_transforms
from src.training.trainer import load_checkpoint


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    """Run the model over a loader; return numpy y_true, y_pred and raw logits."""
    model.eval().to(device)
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    logits = torch.cat(logits).numpy()
    return {"y_true": torch.cat(labels).numpy(), "y_pred": logits.argmax(1), "logits": logits}


def select_gradcam_examples(y_true, y_pred, probs, k: int = 4) -> dict[str, list[int]]:
    """Pick image indices for the four Grad-CAM categories.

    Confidence = probability of the predicted class. Correct categories (confident/barely)
    and wrong ones never share an index. confused_pair holds wrong predictions of the single
    most frequent (true, pred) mistake, so it may overlap with confident_wrong.
    """
    y_true, y_pred, probs = np.asarray(y_true), np.asarray(y_pred), np.asarray(probs)
    conf = probs[np.arange(len(y_pred)), y_pred]
    correct = np.flatnonzero(y_true == y_pred)
    wrong = np.flatnonzero(y_true != y_pred)
    correct = correct[np.argsort(-conf[correct], kind="stable")]  # most confident first
    wrong = wrong[np.argsort(-conf[wrong], kind="stable")]
    confident_correct = correct[:k]
    barely_correct = correct[k:][::-1][:k]  # least confident, skipping the ones already taken
    confused = []
    if len(wrong):
        pair = Counter(zip(y_true[wrong].tolist(), y_pred[wrong].tolist())).most_common(1)[0][0]
        confused = [i for i in wrong if (y_true[i], y_pred[i]) == pair][:k]
    return {"confident_correct": [int(i) for i in confident_correct],
            "barely_correct": [int(i) for i in barely_correct],
            "confident_wrong": [int(i) for i in wrong[:k]],
            "confused_pair": [int(i) for i in confused]}


def predict_image(path: str, checkpoint: str, topk: int = 5,
                  device: str = "cpu") -> list[tuple[str, float]]:
    """Top-k (class name, probability) for one image file."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    ckpt = torch.load(checkpoint, map_location="cpu")  # only for config and class names
    names = ckpt["class_names"]
    size = ckpt["config"].get("data", {}).get("image_size", 224)
    model = load_checkpoint(checkpoint, device=device)
    with Image.open(path) as im:
        x = build_transforms(False, size)(im.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    p, idx = probs.topk(min(topk, len(names)))
    return [(names[i], float(v)) for v, i in zip(p.tolist(), idx.tolist())]
