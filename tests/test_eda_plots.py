import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image

import src.data.cub_dataset as m
from src.visualization.eda_plots import run_eda, update_notes


def _split(n, off=0):
    caps = "\n".join(f"A  Bird {k}" for k in range(10))
    return [{"image": Image.fromarray(np.random.RandomState(i + off).randint(
                0, 255, (60, 70, 3), dtype=np.uint8)),
             "label": i % 2, "description": caps, "file_name": f"f{i + off}.jpg"}
            for i in range(n)]


def test_update_notes_is_idempotent(tmp_path):
    p = tmp_path / "NOTES.md"
    p.write_text("# Notes\n\n## Cleaning steps\n\n## Failed experiments\n", encoding="utf-8")
    update_notes(p, "first")
    update_notes(p, "second")
    text = p.read_text(encoding="utf-8")
    assert "second" in text and "first" not in text
    assert text.count("## Failed experiments") == 1


def test_run_eda_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_load_hf", lambda: (_split(6), _split(4, 100), ["a", "b"]))
    notes = tmp_path / "NOTES.md"
    notes.write_text("# Notes\n\n## Cleaning steps\n", encoding="utf-8")
    cfg = {"seed": 0, "eda": {"out_dir": str(tmp_path / "out"), "notes_path": str(notes)}}
    run_eda(cfg)
    run_eda(cfg)
    out = tmp_path / "out"
    for f in ["eda/class_distribution.png", "eda/image_sizes.png",
              "eda/caption_lengths.png", "eda/augmentations.png"]:
        assert (out / f).exists()
    clean = json.loads((out / "data/captions_clean.json").read_text(encoding="utf-8"))
    assert len(clean) == 10 and clean["f0.jpg"][0] == "a bird 0"
    assert notes.read_text(encoding="utf-8").count("eda:start") == 1
