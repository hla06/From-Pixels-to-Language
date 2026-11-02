import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image

from src.data.image_stats import channel_stats, find_duplicates, image_stats


def _noise(seed, mode="RGB"):
    arr = np.random.RandomState(seed).randint(0, 255, (64, 80, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert(mode)


def _split():
    a = _noise(0)
    return [{"image": a}, {"image": a.copy()}, {"image": _noise(1, "L")}]


def test_image_stats_reports_mode():
    df = image_stats(_split())
    assert list(df.columns)[:4] == ["width", "height", "mode", "phash"]
    assert list(df["mode"]) == ["RGB", "RGB", "L"]
    assert df.loc[0, "width"] == 80 and df.loc[0, "height"] == 64


def test_find_duplicates_same_hash():
    assert find_duplicates(image_stats(_split())) == [(0, 1)]


def test_quality_measures_separate_flat_from_noisy():
    flat = Image.new("RGB", (64, 64), (200, 200, 200))
    df = image_stats([{"image": flat}, {"image": _noise(2)}])
    assert abs(df.loc[0, "brightness"] - 200) < 1e-6
    assert df.loc[0, "sharpness"] == 0 and df.loc[1, "sharpness"] > 100
    assert df.loc[0, "contrast"] == 0 < df.loc[1, "contrast"]


def test_channel_stats_of_constant_images():
    df = image_stats([{"image": Image.new("RGB", (32, 32), (255, 0, 51))}])
    mean, std = channel_stats(df)
    assert [round(m, 2) for m in mean] == [1.0, 0.0, 0.2] and std == [0.0, 0.0, 0.0]
