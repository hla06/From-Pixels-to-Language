import glob

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.cli import main, run_train
from src.config import load_config
from src.models.registry import build_encoder


def test_all_configs_resolve_encoder():
    paths = [p for p in glob.glob("configs/*.yaml") if "encoder" in load_config(p)]
    assert len(paths) == 10
    for p in paths:
        cfg = load_config(p)
        assert cfg["tracker"]["run_name"] == p.replace("\\", "/").split("/")[-1][:-5]
        build_encoder(cfg["encoder"]["name"], pretrained=False)


@pytest.mark.parametrize("task", ["colorization", "caption"])
def test_milestone2_tasks_exit_nonzero_with_message(task, capsys):
    assert main(["train", "--task", task, "--config", "x"]) == 2
    out = capsys.readouterr()
    assert "Milestone 2" in out.out + out.err


def test_unknown_task_exits_nonzero(capsys):
    assert main(["train", "--task", "bogus", "--config", "x"]) != 0
    assert "bogus" in capsys.readouterr().err


@pytest.mark.parametrize("cmd", ["evaluate", "gradcam"])
def test_checkpoint_flag_is_required(cmd, capsys):
    assert main([cmd]) != 0


def test_predict_milestone2_returns_2(capsys):
    assert main(["predict", "--task", "caption", "--image", "x", "--checkpoint", "y"]) == 2
    assert "Milestone 2" in capsys.readouterr().err


def test_predict_milestone2_without_checkpoint_and_missing_checkpoint(capsys):
    assert main(["predict", "--task", "caption", "--image", "p"]) == 2
    assert "Milestone 2" in capsys.readouterr().err


def test_predict_missing_checkpoint_or_image_returns_2(tiny_ckpt, tmp_path, capsys):
    from PIL import Image
    img = tmp_path / "i.png"
    Image.new("RGB", (8, 8)).save(img)
    assert main(["predict", "--image", str(img), "--checkpoint", str(tmp_path / "nope.pt")]) == 2
    assert "nope.pt" in capsys.readouterr().err
    assert main(["predict", "--image", str(tmp_path / "missing.png"), "--checkpoint", tiny_ckpt]) == 2
    assert "missing.png" in capsys.readouterr().err


def test_predict_checkpoint_defaults_to_final_model():
    from src.cli import DEFAULT_CHECKPOINT, build_parser
    args = build_parser().parse_args(["predict", "--image", "p"])
    assert args.checkpoint == DEFAULT_CHECKPOINT == "outputs/checkpoints/resnet18_finetune.pt"


@pytest.fixture
def tiny_ckpt(tmp_path):
    from src.models.classifier import Classifier
    model = Classifier(build_encoder("scratch_cnn", pretrained=False), num_classes=3)
    cfg = {"seed": 0, "encoder": {"name": "scratch_cnn", "pretrained": False, "freeze": False},
           "train": {"dropout": 0.1, "batch_size": 4}, "data": {"image_size": 64}}
    path = tmp_path / "tiny.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "class_names": ["a", "b", "c"], "val_acc": 0.0}, path)
    return str(path)


def test_evaluate_and_gradcam_commands(tiny_ckpt, tmp_path, monkeypatch, capsys):
    import json
    import src.cli as cli
    ds = TensorDataset(torch.randn(12, 3, 64, 64), torch.arange(12) % 3)
    monkeypatch.setattr(cli, "load_cub", lambda cfg: (None, None, ds, ["a", "b", "c"]))
    monkeypatch.setattr(cli, "EVAL_DIR", tmp_path / "eval")
    assert main(["evaluate", "--checkpoint", tiny_ckpt]) == 0
    out = json.loads((tmp_path / "eval" / "tiny.json").read_text())
    assert {"metrics", "y_true", "y_pred", "most_confused_pairs", "name", "class_names"} <= set(out)
    assert len(out["y_true"]) == 12 and "top5_accuracy" in out["metrics"]
    assert main(["gradcam", "--checkpoint", tiny_ckpt, "--out", str(tmp_path / "fig")]) == 0
    assert list((tmp_path / "fig").glob("gradcam_tiny_*.png"))


def test_predict_command_prints_topk_and_saves_overlay(tiny_ckpt, tmp_path, capsys):
    from PIL import Image
    img = tmp_path / "i.png"
    Image.new("RGB", (80, 80), "green").save(img)
    out = tmp_path / "cam.png"
    assert main(["predict", "--image", str(img), "--checkpoint", tiny_ckpt, "--topk", "2",
                 "--gradcam-out", str(out)]) == 0
    assert out.exists()
    assert len([l for l in capsys.readouterr().out.splitlines() if l[:1].isdigit()]) == 2


def test_run_train_with_injected_loaders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def loader(n):
        ds = TensorDataset(torch.randn(n, 3, 32, 32), torch.arange(n) % 3)
        return DataLoader(ds, batch_size=4)
    cfg = {"seed": 0, "checkpoint_dir": str(tmp_path),
           "encoder": {"name": "scratch_cnn", "pretrained": False, "freeze": False},
           "data": {"num_classes": 3},
           "train": {"epochs": 1, "patience": 1, "lr_backbone": 1e-3, "lr_head": 1e-3,
                     "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 4,
                     "label_smoothing": 0.1},
           "tracker": {"name": "null", "run_name": "hist_run"}}
    out = run_train(cfg, loaders={"train": loader(8), "val": loader(6), "test": loader(6)})
    assert "best_val_acc" in out
    import json
    hist = json.loads((tmp_path / "outputs" / "history" / "hist_run.json").read_text())
    assert hist and hist[0]["epoch"] == 1 and "val_acc" in hist[0]


def test_run_train_writes_history_where_cfg_says(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    ds = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 3)
    loaders = {k: DataLoader(ds, batch_size=4) for k in ("train", "val", "test")}
    cfg = {"seed": 0, "checkpoint_dir": str(tmp_path), "history_dir": str(tmp_path / "elsewhere"),
           "encoder": {"name": "scratch_cnn", "pretrained": False, "freeze": False},
           "data": {"num_classes": 3},
           "train": {"epochs": 1, "patience": 1, "lr_backbone": 1e-3, "lr_head": 1e-3,
                     "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 4,
                     "label_smoothing": 0.1},
           "tracker": {"name": "null", "run_name": "r"}}
    run_train(cfg, loaders=loaders)
    assert json.loads((tmp_path / "elsewhere" / "r.json").read_text())
    assert not (tmp_path / "outputs" / "history").exists()


def test_apply_smoke_sets_history_dir():
    from src.cli import apply_smoke
    cfg = apply_smoke({"data": {}, "train": {}})
    assert cfg["history_dir"] == "outputs/smoke"
