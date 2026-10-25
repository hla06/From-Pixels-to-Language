"""Command line entry point: `python -m src.cli <command> ...`.

One handler per subcommand, looked up in HANDLERS, so a later task only replaces one function.
"""
import argparse
import sys

import torch

from src.config import load_config
from src.data.cub_dataset import build_loaders, load_cub
from src.models.classifier import Classifier
from src.models.registry import build_encoder
from src.training.seed import set_seed
from src.training.tracker import build_tracker
from src.training.trainer import Trainer, build_optimizer, build_scheduler

TASKS = ["classification", "colorization", "caption"]
DEFAULT_EDA_CONFIG = "configs/scratch_cnn.yaml"  # eda only needs seed and data settings
SMOKE_SUBSET, SMOKE_BATCH = 32, 8


def run_train(cfg: dict, loaders: dict | None = None) -> dict:
    """Build every piece from the config and run Trainer.fit(); `loaders` can be injected (tests)."""
    set_seed(cfg["seed"])
    tracker = build_tracker(cfg)
    class_names = None
    if loaders is None:
        cfg["data"].setdefault("batch_size", cfg["train"]["batch_size"])
        _, _, _, class_names = load_cub(cfg)  # only for names; loaders are built next
        loaders = build_loaders(cfg)
    num_classes = cfg["data"].get("num_classes", 200)
    enc = cfg["encoder"]
    encoder = build_encoder(enc["name"], pretrained=enc["pretrained"], freeze=enc["freeze"])
    model = Classifier(encoder, num_classes=num_classes, dropout=cfg["train"]["dropout"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(loaders["train"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.get("smoke"):
        device = "cpu"
    trainer = Trainer(model, loaders, optimizer, scheduler, tracker, device, cfg,
                      class_names=class_names)
    return trainer.fit()


def apply_smoke(cfg: dict) -> dict:
    """Shrink a config to a CPU-friendly pipeline check: 32 images, 1 epoch, no tracking."""
    cfg["smoke"] = True
    cfg["data"].update(subset=SMOKE_SUBSET, batch_size=SMOKE_BATCH, num_workers=0)
    cfg["train"].update(epochs=1, patience=1, batch_size=SMOKE_BATCH)
    cfg["tracker"] = {"name": "null", "run_name": "smoke"}
    cfg["checkpoint_dir"] = "outputs/smoke"
    return cfg


def _milestone2_guard(args) -> bool:
    if args.task in ("colorization", "caption"):
        print(f"Task '{args.task}' belongs to Milestone 2 and is not implemented yet.",
              file=sys.stderr)
        return True
    return False


def cmd_eda(args) -> int:
    from src.visualization.eda_plots import run_eda
    run_eda(load_config(args.config))
    return 0


def cmd_train(args) -> int:
    if _milestone2_guard(args):
        return 2
    cfg = load_config(args.config)
    if args.smoke:
        apply_smoke(cfg)
    result = run_train(cfg)
    print(f"best_val_acc={result['best_val_acc']:.4f} checkpoint={result['checkpoint']}")
    return 0


def _not_implemented(args) -> int:
    print("not implemented yet (Task 9/10)")
    return 2


def cmd_predict(args) -> int:
    if _milestone2_guard(args):
        return 2
    return _not_implemented(args)


HANDLERS = {
    "eda": cmd_eda,
    "train": cmd_train,
    "tune": _not_implemented,
    "predict": cmd_predict,
    "evaluate": _not_implemented,
    "gradcam": _not_implemented,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in HANDLERS:
        p = sub.add_parser(name)
        p.add_argument("--config", default=DEFAULT_EDA_CONFIG if name == "eda" else None)
        if name in ("train", "predict", "evaluate", "gradcam"):
            p.add_argument("--task", choices=TASKS, default="classification")
        if name == "train":
            p.add_argument("--smoke", action="store_true")
        if name in ("predict", "gradcam"):
            p.add_argument("--image")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:  # argparse exits on bad input; return its code instead
        return e.code if isinstance(e.code, int) else 2
    return HANDLERS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
