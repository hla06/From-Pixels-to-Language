"""Comparison figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.tuner import TuneResult


def plot_tuning_comparison(results: dict[str, TuneResult], out: Path) -> None:
    """Left: best-so-far score per trial. Right: every trial's score against its learning rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for method, r in results.items():
        best_so_far = [max(r.scores[: i + 1]) for i in range(len(r.scores))]
        ax1.plot(range(1, len(best_so_far) + 1), best_so_far, marker="o", label=method)
        ax2.scatter([t["lr"] for t in r.trials], r.scores, label=method)
    ax1.set(xlabel="trial", ylabel="best val accuracy so far", title="Search progress")
    ax2.set(xlabel="learning rate", ylabel="val accuracy", xscale="log", title="Score vs learning rate")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
