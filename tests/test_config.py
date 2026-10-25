import yaml

from src.config import load_config


def test_load_config_roundtrip(tmp_path):
    cfg = {"seed": 42, "model": {"name": "resnet18", "lr": 0.001}, "list": [1, 2]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert load_config(p) == cfg
    assert load_config(str(p)) == cfg
