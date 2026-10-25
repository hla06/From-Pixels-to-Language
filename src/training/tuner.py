"""Runs a Sampler's trials through a training function and keeps the best one."""
import copy
import time
from dataclasses import dataclass, field
from typing import Callable

from src.training.samplers import Sampler


@dataclass
class TuneResult:
    method: str
    best_score: float
    best_params: dict
    n_trials: int
    seconds: float
    scores: list[float]
    trials: list[dict] = field(default_factory=list)  # params per trial, for plotting


class Tuner:
    """train_fn(cfg) -> score (higher is better); it is injected so tests use a fake."""

    def __init__(self, base_cfg: dict, sampler: Sampler, method: str,
                 train_fn: Callable[[dict], float], epochs_per_trial: int = 8):
        self.base_cfg, self.sampler, self.method = base_cfg, sampler, method
        self.train_fn, self.epochs_per_trial = train_fn, epochs_per_trial

    def _trial_cfg(self, i: int, params: dict) -> dict:
        cfg = copy.deepcopy(self.base_cfg)  # the base config stays untouched
        t = cfg["train"]
        t["lr_backbone"] = params["lr"]
        t["lr_head"] = 10 * params["lr"]  # head learns faster than the pretrained backbone
        t["weight_decay"] = params["weight_decay"]
        t["dropout"] = params["dropout"]
        t["epochs"] = self.epochs_per_trial
        cfg["tracker"]["run_name"] = f"{self.method}-trial-{i}"
        cfg["tracker"]["group"] = f"tune-{self.method}"
        cfg["history_dir"] = "outputs/tuning/history"  # keep trials out of the final curves
        return cfg

    def run(self) -> TuneResult:
        trials = list(self.sampler.trials())
        scores = []
        start = time.perf_counter()
        for i, params in enumerate(trials):
            scores.append(float(self.train_fn(self._trial_cfg(i, params))))
        seconds = time.perf_counter() - start
        best = max(range(len(scores)), key=scores.__getitem__)
        return TuneResult(self.method, scores[best], trials[best], len(scores), seconds,
                          scores, trials)
