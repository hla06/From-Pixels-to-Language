import json

import numpy as np
import pytest

from src.cli import main
from src.evaluation.benchmark import benchmark_table
from src.visualization.comparison import (plot_confused_pairs, plot_model_comparison,
                                          plot_per_class_f1, plot_training_curves,
                                          plot_curves_grid, write_selection_table)

PUB = [{"name": "A", "accuracy": 80.0, "source_url": "https://x.org/a", "setup": "r50"},
       {"name": "B", "accuracy": 85.0, "source_url": "https://x.org/b", "setup": "vit"}]


def test_benchmark_table_rows():
    df = benchmark_table({"ours": 70.0}, PUB)
    assert list(df.columns) == ["name", "accuracy", "source_url", "setup"]
    assert len(df) == 3 and df.iloc[0]["source_url"] == "this work"


@pytest.mark.parametrize("url", [None, "", "  "])
def test_benchmark_rejects_missing_source(url):
    bad = [{"name": "A", "accuracy": 1.0, "source_url": url, "setup": "s"}]
    with pytest.raises(ValueError):
        benchmark_table({}, bad)


def test_plot_functions_write_pngs(tmp_path):
    cm = np.array([[3, 1, 0], [2, 1, 0], [0, 0, 4]])
    names = ["a", "b", "c"]
    plot_model_comparison({"r1": {"accuracy": 0.5, "f1_macro": 0.4}}, tmp_path / "m.png")
    plot_training_curves({"r1": [{"epoch": 1, "train_loss": 1, "val_loss": 1, "train_acc": .1, "val_acc": .2}]},
                         tmp_path / "c.png")
    h = {f"r{i}": [{"epoch": 1, "train_loss": 1, "val_loss": 1, "train_acc": .1, "val_acc": .2}]
         for i in range(4)}
    plot_curves_grid(h, tmp_path / "g.png", cols=3)
    plot_confused_pairs(cm, names, tmp_path / "p.png", k=2)
    plot_per_class_f1([0, 1, 2, 2], [0, 2, 2, 2], names, tmp_path / "f.png")
    assert len(list(tmp_path.glob("*.png"))) == 5


def test_selection_table_sorts_by_validation_not_test(tmp_path):
    def hist(*vals):
        return [{"epoch": i + 1, "val_acc": v} for i, v in enumerate(vals)]
    histories = {"a": hist(.5, .7), "b": hist(.6, .8, .75), "c": hist(.9)}
    evals = {"a": {"metrics": {"accuracy": .95}}, "b": {"metrics": {"accuracy": .60}}}  # c: no eval
    write_selection_table(histories, evals, tmp_path / "sel")
    rows = (tmp_path / "sel.csv").read_text().splitlines()
    assert rows[0] == "run,best_val_acc,epochs,test_acc"
    assert rows[1].startswith("b,0.8000,3,") and rows[2].startswith("a,0.7000,2,")
    assert len(rows) == 3 and (tmp_path / "sel.md").exists()


def test_figures_command_on_synthetic_json(tmp_path, monkeypatch, capsys):
    import src.cli as cli
    h, e, t, f = (tmp_path / d for d in ("history", "eval", "tuning", "figures"))
    for d in (h, e, t):
        d.mkdir()
    (h / "r1.json").write_text(json.dumps([{"epoch": 1, "train_loss": 1, "val_loss": 1,
                                            "train_acc": .1, "val_acc": .2}]))
    (e / "r1.json").write_text(json.dumps({
        "name": "r1", "metrics": {"accuracy": .5, "f1_macro": .4}, "class_names": ["a", "b", "c"],
        "y_true": [0, 1, 2, 2], "y_pred": [0, 2, 2, 2], "most_confused_pairs": []}))
    (t / "grid.json").write_text(json.dumps({
        "method": "grid", "best_score": .5, "best_params": {}, "n_trials": 2, "seconds": 1.0,
        "scores": [.3, .5], "trials": [{"lr": 1e-3}, {"lr": 1e-4}]}))
    bench = tmp_path / "benchmark.yaml"
    bench.write_text("published:\n  - {name: A, accuracy: 80.0, source_url: 'https://x.org/a', setup: s}\n")
    for attr, val in [("HISTORY_DIR", h), ("EVAL_DIR", e), ("TUNING_DIR", t), ("FIGURES_DIR", f),
                      ("BENCHMARK_CONFIG", bench)]:
        monkeypatch.setattr(cli, attr, val)
    assert main(["figures"]) == 0
    names = {p.name for p in f.iterdir()}
    assert {"model_comparison.png", "training_curves.png", "tuning_comparison.png",
            "confused_pairs_r1.png", "per_class_f1_r1.png", "benchmark.csv", "benchmark.md",
            "training_curves_grid.png", "selection.csv", "selection.md"} <= names


def test_figures_command_skips_missing_inputs(tmp_path, monkeypatch, capsys):
    import src.cli as cli
    for attr in ("HISTORY_DIR", "EVAL_DIR", "TUNING_DIR"):
        monkeypatch.setattr(cli, attr, tmp_path / "none")
    monkeypatch.setattr(cli, "FIGURES_DIR", tmp_path / "f")
    monkeypatch.setattr(cli, "BENCHMARK_CONFIG", tmp_path / "missing.yaml")
    assert main(["figures"]) == 0
    assert "skip" in capsys.readouterr().out
