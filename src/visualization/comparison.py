"""Comparison figures."""
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

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
    """Loss (left) and accuracy (right) per epoch; solid = validation, dashed = training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        line, = ax1.plot(ep, [r["val_loss"] for r in h], label=name)
        ax1.plot(ep, [r["train_loss"] for r in h], "--", color=line.get_color())
        ax2.plot(ep, [r["val_acc"] for r in h], color=line.get_color(), label=name)
        ax2.plot(ep, [r["train_acc"] for r in h], "--", color=line.get_color())
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss (dashed = train)")
    ax2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy (dashed = train)")
    ax2.legend(fontsize=7)
    _save(fig, out)


def plot_curves_grid(histories: dict[str, list[dict]], out: Path, cols: int = 3) -> None:
    """One panel per run: training vs validation accuracy (left axis) and loss (right axis)."""
    rows = -(-len(histories) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, (name, h) in zip(axes.ravel(), histories.items()):
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_acc"] for r in h], "--", color="tab:blue", label="train acc")
        ax.plot(ep, [r["val_acc"] for r in h], color="tab:blue", label="val acc")
        ax.set(title=name, xlabel="epoch", ylim=(0, 1))
        ax2 = ax.twinx()
        ax2.plot(ep, [r["train_loss"] for r in h], "--", color="tab:red", label="train loss")
        ax2.plot(ep, [r["val_loss"] for r in h], color="tab:red", label="val loss")
        ax2.set_ylabel("loss", color="tab:red")
        ax.set_ylabel("accuracy", color="tab:blue")
    for ax in axes.ravel()[len(histories):]:
        ax.axis("off")
    proxies = [Line2D([], [], color=c, ls=ls) for c in ("tab:blue", "tab:red") for ls in ("--", "-")]
    fig.legend(proxies, ["train acc", "val acc", "train loss", "val loss"],
               loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_selection_table(histories: dict[str, list[dict]], evals: dict[str, dict],
                          out_stem: Path) -> None:
    """Best validation accuracy next to test accuracy per run, sorted by validation accuracy.

    The final model is chosen on the validation column; the test column is only reported.
    Writes <out_stem>.csv and <out_stem>.md.
    """
    rows = [{"run": n, "best_val_acc": max(r["val_acc"] for r in h),
             "epochs": len(h), "test_acc": evals[n]["metrics"]["accuracy"]}
            for n, h in histories.items() if n in evals]
    rows.sort(key=lambda r: r["best_val_acc"], reverse=True)
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    lines = ["run,best_val_acc,epochs,test_acc"] + [
        f"{r['run']},{r['best_val_acc']:.4f},{r['epochs']},{r['test_acc']:.4f}" for r in rows]
    out_stem.with_suffix(".csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    md = ["| run | best val acc | epochs | test acc |", "|---|---|---|---|"] + [
        f"| {r['run']} | {r['best_val_acc']:.4f} | {r['epochs']} | {r['test_acc']:.4f} |" for r in rows]
    out_stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


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
