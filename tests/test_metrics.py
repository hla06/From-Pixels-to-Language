import numpy as np

from src.evaluation.metrics import compute_metrics, confusion, most_confused_pairs


def test_accuracy_known():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1])
    m = compute_metrics(y_true, y_pred, num_classes=3)
    assert m["accuracy"] == 0.8
    assert "top5_accuracy" not in m


def test_macro_f1_handles_absent_class():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 2, 1, 1])  # class 2 predicted but never true
    m = compute_metrics(y_true, y_pred, num_classes=3)
    assert all(np.isfinite(m[k]) for k in ("precision_macro", "recall_macro", "f1_macro"))
    assert m["recall_macro"] == (0.5 + 1.0 + 0.0) / 3


def test_top5_with_logits():
    logits = np.array([[0.1, 0.9, 0.2, 0.3, 0.4, -1.0],
                       [0.9, 0.0, 0.0, 0.0, 0.0, 0.1]])
    y_true = np.array([5, 5])  # row 0: 5 is lowest (miss); row 1: 5 is 2nd best (hit)
    y_pred = logits.argmax(1)
    m = compute_metrics(y_true, y_pred, logits=logits, num_classes=6)
    assert m["top5_accuracy"] == 0.5
    assert m["accuracy"] == 0.0


def test_confusion_diagonal_sum_equals_correct():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1])
    cm = confusion(y_true, y_pred, num_classes=4)
    assert cm.shape == (4, 4)
    assert np.trace(cm) == (y_true == y_pred).sum()


def test_most_confused_excludes_diagonal():
    cm = np.array([[9, 3, 0], [1, 8, 5], [0, 0, 7]])
    assert most_confused_pairs(cm, k=2) == [(1, 2, 5), (0, 1, 3)]
