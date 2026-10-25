"""Comparison figures."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def _save(fig, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict[str, dict], out: Path) -> None:
    """Grouped bars of test accuracy and macro-F1 per run; results[run] holds metric dicts."""
    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 4))
    for k, (key, label) in enumerate([("accuracy", "accuracy"), ("f1_macro", "macro-F1")]):
        ax.bar(x + (k - 0.5) * 0.38, [results[n].get(key, 0) for n in names], 0.38, label=label)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set(ylabel="test score", title="Model comparison", ylim=(0, 1))
    ax.legend()
    _save(fig, out)


def plot_training_curves(histories: dict[str, list[dict]], out: Path) -> None:
    """Loss (left) and validation accuracy (right) per epoch; solid = val loss, dashed = train loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        line, = ax1.plot(ep, [r["val_loss"] for r in h], label=name)
        ax1.plot(ep, [r["train_loss"] for r in h], "--", color=line.get_color())
        ax2.plot(ep, [r["val_acc"] for r in h], label=name)
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss (dashed = train)")
    ax2.set(xlabel="epoch", ylabel="val accuracy", title="Validation accuracy")
    ax2.legend(fontsize=7)
    _save(fig, out)


def plot_confused_pairs(cm, class_names, out: Path, k: int = 10) -> None:
    """Horizontal bars of the k most confused (true -> predicted) class pairs."""
    from src.evaluation.metrics import most_confused_pairs
    pairs = most_confused_pairs(np.asarray(cm), k)
    fig, ax = plt.subplots(figsize=(8, 0.4 * max(len(pairs), 1) + 1.5))
    labels = [f"{class_names[t]} -> {class_names[p]}" for t, p, _ in pairs]
    ax.barh(range(len(pairs)), [c for _, _, c in pairs])
    ax.set_yticks(range(len(pairs)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set(xlabel="images", title="Most confused class pairs")
    _save(fig, out)


def plot_per_class_f1(y_true, y_pred, class_names, out: Path) -> None:
    """Per-class F1 sorted ascending, so the hardest classes sit on the left."""
    from sklearn.metrics import f1_score
    labels = list(range(len(class_names)))
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(f1)), np.sort(f1))
    ax.set(xlabel="classes (sorted by F1)", ylabel="F1", title="Per-class F1")
    _save(fig, out)
