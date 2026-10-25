import pytest

from src.training.tracker import NullTracker, build_tracker


def test_null_tracker_records():
    t = NullTracker()
    t.log_metrics({"loss": 1.0}, step=2)
    assert ("log_metrics", {"loss": 1.0}, 2) in t.calls


def test_build_tracker_null():
    assert isinstance(build_tracker({"tracker": {"name": "null"}}), NullTracker)


def test_build_tracker_unknown():
    with pytest.raises(ValueError, match="wandb"):
        build_tracker({"tracker": {"name": "foo"}})


def test_finish_is_idempotent_for_wandb(monkeypatch):
    import sys
    import types
    from src.training.tracker import WandbTracker
    finished = []
    run = types.SimpleNamespace(finish=lambda: finished.append(1))
    fake = types.SimpleNamespace(init=lambda **kw: run)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    t = WandbTracker("p", "r")
    t.finish()
    t.finish()
    assert finished == [1]


def test_run_train_finishes_tracker_when_fit_fails(monkeypatch):
    import src.cli as cli
    tracker = NullTracker()
    monkeypatch.setattr(cli, "build_tracker", lambda cfg: tracker)

    class Boom:
        def __init__(self, *a, **k): ...
        def fit(self): raise RuntimeError("boom")

    monkeypatch.setattr(cli, "Trainer", Boom)
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4) % 3), batch_size=2)
    cfg = {"seed": 0, "encoder": {"name": "scratch_cnn", "pretrained": False, "freeze": False},
           "data": {"num_classes": 3},
           "train": {"epochs": 1, "patience": 1, "lr_backbone": 1e-3, "lr_head": 1e-3,
                     "weight_decay": 0, "dropout": 0.1, "batch_size": 2},
           "tracker": {"name": "null"}}
    import pytest
    with pytest.raises(RuntimeError):
        cli.run_train(cfg, loaders={"train": dl, "val": dl, "test": dl})
    assert ("finish",) in tracker.calls
