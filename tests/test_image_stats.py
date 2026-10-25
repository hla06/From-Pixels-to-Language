import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image

from src.data.image_stats import find_duplicates, image_stats


def _noise(seed, mode="RGB"):
    arr = np.random.RandomState(seed).randint(0, 255, (64, 80, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert(mode)


def _split():
    a = _noise(0)
    return [{"image": a}, {"image": a.copy()}, {"image": _noise(1, "L")}]


def test_image_stats_reports_mode():
    df = image_stats(_split())
    assert list(df.columns) == ["width", "height", "mode", "phash"]
    assert list(df["mode"]) == ["RGB", "RGB", "L"]
    assert df.loc[0, "width"] == 80 and df.loc[0, "height"] == 64


def test_find_duplicates_same_hash():
    assert find_duplicates(image_stats(_split())) == [(0, 1)]
