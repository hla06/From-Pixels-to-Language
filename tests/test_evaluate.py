import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluate import evaluate, predict_image, select_gradcam_examples
from src.models.classifier import Classifier
from src.models.registry import build_encoder


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    model = Classifier(build_encoder("scratch_cnn", pretrained=False), num_classes=6)
    cfg = {"encoder": {"name": "scratch_cnn", "pretrained": False, "freeze": False},
           "train": {"dropout": 0.1}, "data": {"image_size": 64}}
    path = tmp_path / "m.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "class_names": [f"bird_{i}" for i in range(6)], "val_acc": 0.0}, path)
    return str(path)


def test_predict_image_handles_rgba_and_gray(tmp_path, ckpt):
    rgba, gray = tmp_path / "a.png", tmp_path / "b.jpg"
    Image.new("RGBA", (80, 70), (10, 200, 30, 128)).save(rgba)
    Image.new("L", (90, 60), 128).save(gray)
    for p in (rgba, gray):
        out = predict_image(str(p), ckpt, topk=3)
        assert len(out) == 3
        assert all(isinstance(n, str) and isinstance(s, float) for n, s in out)
        assert sum(s for _, s in out) <= 1 + 1e-6
        assert out[0][1] >= out[1][1] >= out[2][1]


def test_predict_missing_file_clear_error(ckpt):
    with pytest.raises(FileNotFoundError, match="nope.jpg"):
        predict_image("nope.jpg", ckpt)


def test_select_examples_categories():
    rng = np.random.RandomState(0)
    n, c = 60, 4
    y_true = rng.randint(0, c, n)
    probs = rng.dirichlet(np.ones(c), n)
    y_pred = probs.argmax(1)
    ex = select_gradcam_examples(y_true, y_pred, probs, k=3)
    assert set(ex) == {"confident_correct", "barely_correct", "confident_wrong", "confused_pair"}
    correct = set(ex["confident_correct"]) | set(ex["barely_correct"])
    assert not set(ex["confident_correct"]) & set(ex["barely_correct"])
    assert all(y_true[i] == y_pred[i] for i in correct)
    assert all(y_true[i] != y_pred[i] for i in ex["confident_wrong"] + ex["confused_pair"])
    assert correct.isdisjoint(ex["confident_wrong"] + ex["confused_pair"])
    c_conf = [probs[i, y_pred[i]] for i in ex["confident_correct"]]
    assert c_conf == sorted(c_conf, reverse=True)
    assert min(c_conf) >= max(probs[i, y_pred[i]] for i in ex["barely_correct"])


def test_select_examples_all_correct_has_no_wrong():
    y = np.array([0, 1, 0, 1])
    probs = np.eye(2)[y] * 0.8 + 0.1
    ex = select_gradcam_examples(y, y, probs, k=2)
    assert ex["confident_wrong"] == [] and ex["confused_pair"] == []


def test_evaluate_returns_arrays():
    torch.manual_seed(0)
    model = Classifier(build_encoder("scratch_cnn", pretrained=False), num_classes=3).eval()
    ds = TensorDataset(torch.randn(10, 3, 32, 32), torch.arange(10) % 3)
    out = evaluate(model, DataLoader(ds, batch_size=4), "cpu")
    assert out["y_true"].shape == (10,) and out["y_pred"].shape == (10,)
    assert out["logits"].shape == (10, 3)
