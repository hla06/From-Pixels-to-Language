import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.classifier import Classifier
from src.models.registry import build_encoder
from src.training.seed import set_seed
from src.training.tracker import NullTracker
from src.training.trainer import Trainer, build_optimizer, build_scheduler, load_checkpoint

NUM_CLASSES = 3


def make_cfg(tmp_path, epochs=3, patience=5, freeze=False):
    return {
        "checkpoint_dir": str(tmp_path),
        "encoder": {"name": "scratch_cnn", "pretrained": False, "freeze": freeze},
        "train": {
            "epochs": epochs, "patience": patience, "lr_backbone": 1e-3, "lr_head": 1e-2,
            "weight_decay": 1e-4, "dropout": 0.5, "batch_size": 8,
            "label_smoothing": 0.1, "warmup_epochs": 1,
        },
    }


def make_loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(24, 3, 64, 64, generator=g)
    y = torch.arange(24) % NUM_CLASSES
    ds = TensorDataset(x, y)
    # num_workers=0 keeps CPU runs bit-identical
    return {"train": DataLoader(ds, batch_size=8, shuffle=True), "val": DataLoader(ds, batch_size=8)}


def make_trainer(cfg, trainer_cls=Trainer, freeze=False):
    set_seed(0)
    enc = build_encoder("scratch_cnn", pretrained=False, freeze=freeze)
    model = Classifier(enc, num_classes=NUM_CLASSES, dropout=cfg["train"]["dropout"])
    opt = build_optimizer(model, cfg)
    loaders = make_loaders()
    sched = build_scheduler(opt, cfg, steps_per_epoch=len(loaders["train"]))
    tracker = NullTracker()
    t = trainer_cls(model, loaders, opt, sched, tracker, "cpu", cfg, class_names=["a", "b", "c"])
    return t, tracker


def test_fit_logs_each_epoch(tmp_path):
    t, tracker = make_trainer(make_cfg(tmp_path, epochs=3))
    t.fit()
    logs = [c for c in tracker.calls if c[0] == "log_metrics"]
    assert len(logs) == 3
    for _, metrics, _ in logs:
        assert {"train_loss", "val_loss", "train_acc", "val_acc", "lr"} <= set(metrics)
    assert tracker.calls[0][0] == "log_params"
    assert tracker.calls[-1] == ("finish",)
    assert any(c[0] == "log_artifact" for c in tracker.calls)


def test_best_checkpoint_saved_and_loadable(tmp_path):
    t, _ = make_trainer(make_cfg(tmp_path))
    out = t.fit()
    ckpt = torch.load(out["checkpoint"], map_location="cpu")
    assert {"model_state", "config", "class_names", "val_acc"} <= set(ckpt)
    # restore best weights into the trained model, then compare logits
    t.model.load_state_dict(ckpt["model_state"])
    t.model.eval()
    loaded = load_checkpoint(out["checkpoint"])
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(t.model(x), loaded(x), atol=1e-6)
    assert not loaded.training


def test_checkpoint_loads_with_map_location_cpu(tmp_path):
    t, _ = make_trainer(make_cfg(tmp_path, epochs=1))
    out = t.fit()
    model = load_checkpoint(out["checkpoint"], device="cpu")
    assert next(model.parameters()).device.type == "cpu"


def test_same_seed_same_history(tmp_path):
    h1 = make_trainer(make_cfg(tmp_path / "a", epochs=2))[0].fit()["history"]
    h2 = make_trainer(make_cfg(tmp_path / "b", epochs=2))[0].fit()["history"]
    assert h1 == h2


def test_early_stopping_stops(tmp_path):
    class ConstantVal(Trainer):
        def validate(self):
            return 1.0, 0.5

    t, _ = make_trainer(make_cfg(tmp_path, epochs=6, patience=1), trainer_cls=ConstantVal)
    out = t.fit()
    assert len(out["history"]) < 6
    assert out["best_val_acc"] == 0.5


def test_label_smoothing_applied(tmp_path):
    t, _ = make_trainer(make_cfg(tmp_path))
    assert t.criterion.label_smoothing == 0.1


def test_frozen_encoder_has_only_head_group(tmp_path):
    cfg = make_cfg(tmp_path, freeze=True)
    t, _ = make_trainer(cfg, freeze=True)
    assert len(t.optimizer.param_groups) == 1


def test_scheduler_warmup_then_cosine(tmp_path):
    cfg = make_cfg(tmp_path, epochs=3)
    t, _ = make_trainer(cfg)
    opt, sched = t.optimizer, t.scheduler
    lrs = []
    for _ in range(9):
        lrs.append(opt.param_groups[-1]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[2]          # warming up
    assert lrs[2] == pytest.approx(1e-2)  # peak at end of warmup
    assert lrs[-1] < lrs[3]         # decaying


def test_artifact_named_after_run(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["tracker"] = {"name": "null", "run_name": "my_run"}
    t, tracker = make_trainer(cfg)
    t.fit()
    assert [c[2] for c in tracker.calls if c[0] == "log_artifact"] == ["my_run-best"]


def test_resume_continues_after_interrupted_epoch(tmp_path):
    class Crash(Trainer):
        def train_epoch(self):
            if len(self.tracker.calls) > 2:  # params + 2 epochs logged, then die in epoch 3
                raise RuntimeError("runtime reset")
            return super().train_epoch()

    cfg = make_cfg(tmp_path, epochs=4)
    t, _ = make_trainer(cfg, trainer_cls=Crash)
    with pytest.raises(RuntimeError):
        t.fit()
    assert (tmp_path / "best.last.pt").exists()
    cfg["resume"] = True
    t2, tracker2 = make_trainer(cfg)
    out = t2.fit()
    assert [h["epoch"] for h in out["history"]] == [1, 2, 3, 4]
    assert len([c for c in tracker2.calls if c[0] == "log_metrics"]) == 2  # only epochs 3, 4 ran
    assert not (tmp_path / "best.last.pt").exists()
