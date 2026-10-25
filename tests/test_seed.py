import random

import numpy as np
import torch

from src.training.seed import set_seed


def test_set_seed_reproducible():
    set_seed(1)
    a = (torch.rand(3), random.random(), np.random.rand())
    set_seed(1)
    b = (torch.rand(3), random.random(), np.random.rand())
    assert torch.equal(a[0], b[0])
    assert a[1] == b[1]
    assert a[2] == b[2]
