import copy
import json
import statistics

import pytest

from src.cli import main
from src.training.samplers import GridSampler, RandomSampler
from src.training.tuner import Tuner, TuneResult
from src.visualization.comparison import plot_tuning_comparison

GRID = {"lr": [1e-4, 3e-4, 1e-3], "weight_decay": [1e-4, 1e-2], "dropout": [0.2, 0.5]}
RANDOM = {"lr": {"type": "log_uniform", "low": 1e-4, "high": 1e-3},
          "weight_decay": {"type": "log_uniform", "low": 1e-4, "high": 1e-2},
          "dropout": {"type": "uniform", "low": 0.2, "high": 0.5}}


def test_grid_has_twelve_unique_trials():
    trials = list(GridSampler(GRID).trials())
    assert len(trials) == 12
    assert len({tuple(sorted(t.items())) for t in trials}) == 12


def test_random_twelve_trials_within_ranges():
    trials = list(RandomSampler(RANDOM, n_trials=12, seed=0).trials())
    assert len(trials) == 12
    for t in trials:
        assert 1e-4 <= t["lr"] <= 1e-3
        assert 1e-4 <= t["weight_decay"] <= 1e-2
        assert 0.2 <= t["dropout"] <= 0.5


def test_random_reproducible_by_seed():
    a = list(RandomSampler(RANDOM, 12, seed=7).trials())
    b = list(RandomSampler(RANDOM, 12, seed=7).trials())
    c = list(RandomSampler(RANDOM, 12, seed=8).trials())
    assert a == b and a != c


def test_log_uniform_not_linear():
    lrs = [t["lr"] for t in RandomSampler({"lr": RANDOM["lr"]}, 2000, seed=1).trials()]
    assert statistics.median(lrs) < (1e-4 + 1e-3) / 2


def _base_cfg():
    return {"seed": 1, "train": {"epochs": 30, "lr_backbone": 0, "lr_head": 0,
                                 "weight_decay": 0, "dropout": 0},
            "tracker": {"name": "null"}}


def test_tuner_picks_max_score_and_applies_params():
    seen = []

    def fake_train(cfg):
        seen.append(copy.deepcopy(cfg))
        return 1.0 - abs(cfg["train"]["lr_backbone"] - 3e-4)  # best at lr=3e-4

    base = _base_cfg()
    result = Tuner(base, GridSampler(GRID), "grid", fake_train, epochs_per_trial=8).run()
    assert result.best_params["lr"] == 3e-4
    assert result.n_trials == 12 and len(result.scores) == 12
    assert result.best_score == max(result.scores)
    assert base == _base_cfg()  # base config never mutated
    c = seen[3]
    assert c["train"]["epochs"] == 8
    assert c["train"]["lr_head"] == pytest.approx(10 * c["train"]["lr_backbone"])
    assert c["tracker"]["run_name"] == "grid-trial-3"
    assert c["tracker"]["group"] == "tune-grid"
    assert all(s["seed"] == 1 for s in seen)


def test_plot_comparison_writes_png(tmp_path):
    def res(m):
        return TuneResult(m, 0.7, {"lr": 1e-3}, 3, 1.5, [0.5, 0.7, 0.6],
                          [{"lr": 1e-4}, {"lr": 1e-3}, {"lr": 3e-4}])
    out = tmp_path / "cmp.png"
    plot_tuning_comparison({"grid": res("grid"), "random": res("random")}, out)
    assert out.stat().st_size > 0


def test_tune_unknown_method_exits_nonzero(capsys):
    assert main(["tune", "--method", "bayes"]) != 0
    assert "bayes" in capsys.readouterr().err


def test_tune_cli_writes_json(tmp_path, monkeypatch):
    import src.cli as cli
    monkeypatch.setattr(cli, "run_train", lambda cfg: {"best_val_acc": cfg["train"]["lr_backbone"]})
    monkeypatch.setattr(cli, "TUNING_DIR", tmp_path)
    assert main(["tune", "--method", "random"]) == 0
    data = json.loads((tmp_path / "random.json").read_text())
    assert data["method"] == "random" and data["n_trials"] == 6
    assert len(data["scores"]) == 6


def test_trial_cfg_redirects_history():
    seen = []
    Tuner(_base_cfg(), GridSampler({"lr": [1e-4], "weight_decay": [1e-4], "dropout": [0.2]}),
          "grid", lambda c: seen.append(c) or 0.5).run()
    assert seen[0]["history_dir"] == "outputs/tuning/history"


def test_tuner_on_trial_reports_each_score():
    got = []
    Tuner(_base_cfg(), GridSampler(GRID), "grid", lambda c: 0.5,
          on_trial=lambda params, score: got.append((params, score))).run()
    assert len(got) == 12 and got[0][1] == 0.5


def test_tune_cli_writes_partial_on_failure(tmp_path, monkeypatch):
    import src.cli as cli
    calls = []

    def flaky(cfg):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("boom")
        return 0.1 * len(calls)

    monkeypatch.setattr(cli, "run_train", lambda cfg: {"best_val_acc": flaky(cfg)})
    monkeypatch.setattr(cli, "TUNING_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        main(["tune", "--method", "grid"])
    data = json.loads((tmp_path / "grid.partial.json").read_text())
    assert data["scores"] == pytest.approx([0.1, 0.2]) and len(data["trials"]) == 2
    assert not (tmp_path / "grid.json").exists()


def test_figures_ignore_partial_tuning_files(tmp_path):
    from src.cli import _load_jsons
    (tmp_path / "grid.json").write_text("{}")
    (tmp_path / "grid.partial.json").write_text("{}")
    assert list(_load_jsons(tmp_path)) == ["grid"]
