import torch
from PIL import Image

from src.data.cub_dataset import CubDataset, build_loaders
from src.data.transforms import build_transforms


def _fake_split(n=12):
    return [{"image": Image.new("RGB", (64, 48), (i * 10, 0, 0)), "label": i % 3}
            for i in range(n)]


def test_cub_dataset_len_and_item():
    ds = CubDataset(_fake_split(), [1, 4, 7], build_transforms(train=False))
    assert len(ds) == 3
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 224, 224)
    assert isinstance(y, int) and y == 1


def test_build_loaders_keys_and_batch(monkeypatch):
    import src.data.cub_dataset as m
    monkeypatch.setattr(m, "_load_hf", lambda: (_fake_split(30), _fake_split(6), ["a", "b", "c"]))
    cfg = {"seed": 0, "data": {"batch_size": 4, "num_workers": 0, "val_fraction": 0.2}}
    loaders = build_loaders(cfg)
    assert set(loaders) == {"train", "val", "test"}
    x, y = next(iter(loaders["train"]))
    assert x.shape == (4, 3, 224, 224) and y.shape == (4,)
    assert len(loaders["val"].dataset) == 6
