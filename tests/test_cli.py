import glob

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.cli import main, run_train
from src.config import load_config
from src.models.registry import build_encoder


def test_all_configs_resolve_encoder():
    paths = [p for p in glob.glob("configs/*.yaml") if "encoder" in load_config(p)]
    assert len(paths) == 9
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
def test_stub_subcommands_return_2(cmd, capsys):
    assert main([cmd, "--config", "x"]) == 2
    assert "not implemented yet" in capsys.readouterr().out


def test_run_train_with_injected_loaders(tmp_path):
    def loader(n):
        ds = TensorDataset(torch.randn(n, 3, 32, 32), torch.arange(n) % 3)
        return DataLoader(ds, batch_size=4)
    cfg = {"seed": 0, "checkpoint_dir": str(tmp_path),
           "encoder": {"name": "scratch_cnn", "pretrained": False, "freeze": False},
           "data": {"num_classes": 3},
           "train": {"epochs": 1, "patience": 1, "lr_backbone": 1e-3, "lr_head": 1e-3,
                     "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 4,
                     "label_smoothing": 0.1},
           "tracker": {"name": "null"}}
    out = run_train(cfg, loaders={"train": loader(8), "val": loader(6), "test": loader(6)})
    assert "best_val_acc" in out
