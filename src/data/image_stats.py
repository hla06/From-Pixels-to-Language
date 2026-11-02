"""Per-image size, colour mode, perceptual hash (near-duplicates) and quality measures."""
from collections import defaultdict
from itertools import combinations

import imagehash
import numpy as np
import pandas as pd

QUALITY_SIZE = 128  # quality is measured on a fixed-size thumbnail so scores compare across images
COLUMNS = ["width", "height", "mode", "phash", "brightness", "contrast", "sharpness",
           "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]


def _quality(im) -> dict:
    """brightness = mean grey level (0-255); contrast = std of grey level; sharpness = variance
    of the Laplacian (low = blurry); plus per-channel mean/std on the 0-1 scale."""
    rgb = np.asarray(im.convert("RGB").resize((QUALITY_SIZE, QUALITY_SIZE)), dtype=np.float64)
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
           - 4 * gray[1:-1, 1:-1])
    out = {"brightness": gray.mean(), "contrast": gray.std(), "sharpness": lap.var()}
    for i, c in enumerate("rgb"):
        out[f"{c}_mean"] = rgb[..., i].mean() / 255
        out[f"{c}_std"] = rgb[..., i].std() / 255
    return out


def image_stats(split) -> pd.DataFrame:
    """One row per image, in split order. Each image is decoded a single time."""
    rows = []
    for row in split:
        im = row["image"]
        rows.append({"width": im.width, "height": im.height, "mode": im.mode,
                     "phash": str(imagehash.phash(im)), **_quality(im)})
    return pd.DataFrame(rows, columns=COLUMNS)


def channel_stats(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Dataset-wide per-channel mean and std (pixel level, from the per-image values)."""
    means = [float(df[f"{c}_mean"].mean()) for c in "rgb"]
    stds = [float(np.sqrt((df[f"{c}_std"] ** 2 + (df[f"{c}_mean"] - m) ** 2).mean()))
            for c, m in zip("rgb", means)]
    return means, stds


def find_duplicates(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Index pairs (i < j) whose perceptual hashes are identical."""
    groups = defaultdict(list)
    for i, h in enumerate(df["phash"]):
        groups[h].append(i)
    return [p for idx in groups.values() for p in combinations(idx, 2)]
