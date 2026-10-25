"""Seeding helpers so every run is reproducible."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU+CUDA) and force deterministic cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive numpy/random seeds from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
