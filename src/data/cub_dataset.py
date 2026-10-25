"""CUB-200-2011 dataset wrapper, loading and DataLoader construction."""
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.splits import random_split, stratified_split
from src.data.transforms import build_transforms
from src.training.seed import seed_worker


class CubDataset(Dataset):
    """View of an HF split restricted to `indices`; yields (image tensor, int label)."""

    def __init__(self, hf_split, indices: Sequence[int], transform):
        self.split = hf_split
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.split[self.indices[i]]
        return self.transform(row["image"]), int(row["label"])


def _load_hf():
    """Download/read both HF repos; return (train_split, test_split, class_names)."""
    from datasets import ClassLabel, load_dataset
    train = load_dataset("Multimodal-Fatima/CUB_train", split="train")
    test = load_dataset("Multimodal-Fatima/CUB_test", split="test")
    feat = train.features["label"]
    if isinstance(feat, ClassLabel):
        names = list(feat.names)
    else:
        n = max(train["label"]) + 1
        names = [str(i) for i in range(n)]
    return train, test, names


def _labels(split) -> list[int]:
    # read the label column without decoding images when it is a real HF dataset
    col = split["label"] if hasattr(split, "column_names") else [r["label"] for r in split]
    return [int(y) for y in col]


def load_cub(cfg: dict):
    """Return (train_ds, val_ds, test_ds, class_names). Test is the provided CUB_test split."""
    d = cfg.get("data", {})
    size = d.get("image_size", 224)
    train_hf, test_hf, names = _load_hf()
    labels = _labels(train_hf)
    pool = list(range(len(labels)))
    if d.get("subset"):  # smoke runs: seeded random subset of train images
        rng = np.random.RandomState(cfg["seed"])
        pool = sorted(rng.choice(len(labels), d["subset"], replace=False).tolist())
    frac = d.get("val_fraction", 0.2)
    if d.get("subset"):  # a tiny subset has fewer val images than classes, so stratifying would fail
        tr, va = random_split(len(pool), frac, cfg["seed"])
    else:
        tr, va = stratified_split([labels[i] for i in pool], frac, cfg["seed"])
    tr_idx, va_idx = [pool[i] for i in tr], [pool[i] for i in va]
    train_ds = CubDataset(train_hf, tr_idx, build_transforms(True, size, d.get("aug")))
    val_ds = CubDataset(train_hf, va_idx, build_transforms(False, size))
    test_ds = CubDataset(test_hf, range(len(test_hf)), build_transforms(False, size))
    return train_ds, val_ds, test_ds, names


def build_loaders(cfg: dict) -> dict[str, DataLoader]:
    train_ds, val_ds, test_ds, _ = load_cub(cfg)
    d = cfg.get("data", {})
    gen = torch.Generator().manual_seed(cfg["seed"])  # fixes shuffle order
    kw = dict(batch_size=d.get("batch_size", 32), num_workers=d.get("num_workers", 0),
              worker_init_fn=seed_worker, generator=gen)
    return {
        "train": DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
        "val": DataLoader(val_ds, shuffle=False, **kw),
        "test": DataLoader(test_ds, shuffle=False, **kw),
    }
