"""Hyperparameter samplers. Strategy pattern: Tuner depends on Sampler.trials() only,
so grid and random search are interchangeable."""
import itertools
import math
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class Sampler(ABC):
    @abstractmethod
    def trials(self) -> Iterator[dict]:
        """Yield one {param: value} dict per trial."""


class GridSampler(Sampler):
    """Every combination of the listed values (Cartesian product)."""

    def __init__(self, space: dict[str, list]):
        self.space = space

    def trials(self) -> Iterator[dict]:
        names = list(self.space)
        for values in itertools.product(*self.space.values()):
            yield dict(zip(names, values))


class RandomSampler(Sampler):
    """n_trials independent draws; reproducible because the generator is seeded."""

    def __init__(self, space: dict[str, dict], n_trials: int, seed: int):
        self.space, self.n_trials, self.seed = space, n_trials, seed

    def _draw(self, rng: np.random.Generator, spec: dict) -> float:
        low, high = spec["low"], spec["high"]
        if spec["type"] == "log_uniform":  # uniform in log space: each decade is equally likely
            return float(math.exp(rng.uniform(math.log(low), math.log(high))))
        if spec["type"] == "uniform":
            return float(rng.uniform(low, high))
        raise ValueError(f"Unknown distribution '{spec['type']}'; use 'log_uniform' or 'uniform'")

    def trials(self) -> Iterator[dict]:
        rng = np.random.default_rng(self.seed)  # fresh generator per call, so repeat calls match
        for _ in range(self.n_trials):
            yield {name: self._draw(rng, spec) for name, spec in self.space.items()}
