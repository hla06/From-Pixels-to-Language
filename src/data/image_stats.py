"""Per-image size, colour mode and perceptual hash (for near-duplicate detection)."""
from collections import defaultdict
from itertools import combinations

import imagehash
import pandas as pd


def image_stats(split) -> pd.DataFrame:
    """One row per image, in split order. Each image is decoded a single time."""
    rows = []
    for row in split:
        im = row["image"]
        rows.append({"width": im.width, "height": im.height, "mode": im.mode,
                     "phash": str(imagehash.phash(im))})
    return pd.DataFrame(rows, columns=["width", "height", "mode", "phash"])


def find_duplicates(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Index pairs (i < j) whose perceptual hashes are identical."""
    groups = defaultdict(list)
    for i, h in enumerate(df["phash"]):
        groups[h].append(i)
    return [p for idx in groups.values() for p in combinations(idx, 2)]
