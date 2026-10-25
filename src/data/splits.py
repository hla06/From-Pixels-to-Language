"""Stratified train/validation split of the CUB train set."""
from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(labels: Sequence[int], val_fraction: float = 0.2,
                     seed: int = 42) -> tuple[list[int], list[int]]:
    """Return (train_idx, val_idx) with every multi-image class in both parts.

    A class with a single image cannot be stratified, so it goes to train only.
    """
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)
    splittable = [i for idx in by_class.values() if len(idx) > 1 for i in idx]
    singles = [idx[0] for idx in by_class.values() if len(idx) == 1]
    train_idx, val_idx = train_test_split(
        splittable, test_size=val_fraction, stratify=[labels[i] for i in splittable],
        random_state=seed)
    return sorted(train_idx + singles), sorted(val_idx)


def random_split(n: int, val_fraction: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]:
    """Plain seeded shuffle split of range(n); used for tiny smoke subsets where stratifying fails."""
    order = np.random.RandomState(seed).permutation(n).tolist()
    n_val = int(round(n * val_fraction))
    return sorted(order[n_val:]), sorted(order[:n_val])
