"""Classification metrics and confusion-matrix helpers."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, logits=None, num_classes: int = 200) -> dict:
    """Return classification metrics as a dict of floats.

    - accuracy: fraction of images whose predicted class is exactly right.
    - top5_accuracy (only if logits given): fraction whose true class is among the
      5 highest-scoring classes; forgiving for look-alike birds.
    - precision_macro: of images predicted as a class, the share truly in it,
      averaged equally over classes (low = many false alarms).
    - recall_macro: of images truly in a class, the share found, averaged equally
      over classes (low = many misses).
    - f1_macro: harmonic mean of precision and recall per class, then averaged.

    Macro averages use sklearn's default label set (classes present in y_true or
    y_pred) with zero_division=0, so a class that is never predicted scores 0
    rather than raising or being skipped.
    """
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if logits is not None:
        k = min(5, num_classes)
        topk = np.argsort(logits, axis=1)[:, -k:]
        out["top5_accuracy"] = float((topk == np.asarray(y_true)[:, None]).any(axis=1).mean())
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    out.update(precision_macro=float(p), recall_macro=float(r), f1_macro=float(f))
    return out


def confusion(y_true, y_pred, num_classes: int = 200) -> np.ndarray:
    """num_classes x num_classes matrix; rows = true class, columns = predicted."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def most_confused_pairs(cm: np.ndarray, k: int = 10) -> list[tuple[int, int, int]]:
    """Top-k off-diagonal cells as (true, pred, count), largest first."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:k]
    return [(int(i), int(j), int(off[i, j]))
            for i, j in zip(*np.unravel_index(order, off.shape)) if off[i, j] > 0]
