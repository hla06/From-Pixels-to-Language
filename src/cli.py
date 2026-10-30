"""Command line entry point: `python -m src.cli <command> ...`.

One handler per subcommand, looked up in HANDLERS, so a later task only replaces one function.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

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
DEFAULT_TUNE_CONFIG = "configs/tune_resnet18.yaml"
TUNING_DIR = Path("outputs/tuning")
HISTORY_DIR = Path("outputs/history")
EVAL_DIR = Path("outputs/eval")
FIGURES_DIR = Path("outputs/figures")
DEFAULT_CHECKPOINT = "outputs/checkpoints/resnet18_finetune.pt"  # the final model
BENCHMARK_CONFIG = Path("configs/benchmark.yaml")


def _history_path(cfg: dict) -> Path:
    run_name = cfg.get("tracker", {}).get("run_name") or "run"
    return Path(cfg.get("history_dir", str(HISTORY_DIR))) / f"{run_name}.json"


def _finished_result(cfg: dict) -> dict | None:
    """With cfg['resume'], a run whose history file and best checkpoint exist counts as done."""
    if not cfg.get("resume"):
        return None
    run_name = cfg.get("tracker", {}).get("run_name")
    ckpt = Path(cfg.get("checkpoint_dir", "outputs/checkpoints")) / (
        f"{run_name}.pt" if run_name else "best.pt")
    hist = _history_path(cfg)
    if not (ckpt.exists() and hist.exists()):
        return None
    history = json.loads(hist.read_text(encoding="utf-8"))
    return {"best_val_acc": max(h["val_acc"] for h in history), "history": history,
            "checkpoint": str(ckpt)}


def run_train(cfg: dict, loaders: dict | None = None) -> dict:
    """Build every piece from the config and run Trainer.fit(); `loaders` can be injected (tests)."""
    done = _finished_result(cfg)
    if done is not None:
        print(f"resume: {done['checkpoint']} already trained, skipping")
        return done
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
    try:
        result = trainer.fit()
    finally:
        tracker.finish()  # no-op if fit() already finished; closes the W&B run after a crash
    hist = _history_path(cfg)
    hist.parent.mkdir(parents=True, exist_ok=True)
    hist.write_text(json.dumps(result["history"], indent=2),
                                                  encoding="utf-8")
    return result


def apply_smoke(cfg: dict) -> dict:
    """Shrink a config to a CPU-friendly pipeline check: 32 images, 1 epoch, no tracking."""
    cfg["smoke"] = True
    cfg["data"].update(subset=SMOKE_SUBSET, batch_size=SMOKE_BATCH, num_workers=0)
    cfg["train"].update(epochs=1, patience=1, batch_size=SMOKE_BATCH)
    cfg["tracker"] = {"name": "null", "run_name": "smoke"}
    cfg["checkpoint_dir"] = "outputs/smoke"
    cfg["history_dir"] = "outputs/smoke"  # smoke runs must not show up in the training curves
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
    cfg["resume"] = args.resume
    if args.smoke:
        apply_smoke(cfg)
    result = run_train(cfg)
    print(f"best_val_acc={result['best_val_acc']:.4f} checkpoint={result['checkpoint']}")
    return 0


def cmd_tune(args) -> int:
    from src.training.samplers import GridSampler, RandomSampler
    from src.training.tuner import Tuner
    cfg = load_config(args.config)
    cfg["resume"] = args.resume
    tuning = cfg.pop("tuning")  # search spaces are not part of a training run's config
    if args.method == "grid":
        sampler = GridSampler(tuning["grid"])
    else:
        sampler = RandomSampler(tuning["random"], tuning["n_trials"], cfg["seed"])
    done_trials, done_scores = [], []

    def remember(params, score):
        done_trials.append(params)
        done_scores.append(score)

    tuner = Tuner(cfg, sampler, args.method, lambda c: run_train(c)["best_val_acc"],
                  epochs_per_trial=tuning["epochs_per_trial"], on_trial=remember)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = tuner.run()
    except Exception:
        # keep the finished trials: a crash on trial N must not lose trials 1..N-1
        partial = TUNING_DIR / f"{args.method}.partial.json"
        partial.write_text(json.dumps({"method": args.method, "scores": done_scores,
                                       "trials": done_trials}, indent=2), encoding="utf-8")
        print(f"tuning failed; {len(done_scores)} finished trials saved to {partial}",
              file=sys.stderr)
        raise
    path = TUNING_DIR / f"{args.method}.json"
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")
    print(f"{args.method}: best_score={result.best_score:.4f} params={result.best_params} "
          f"trials={result.n_trials} seconds={result.seconds:.0f} -> {path}")
    return 0


def cmd_predict(args) -> int:
    if _milestone2_guard(args):
        return 2
    try:
        return _predict(args)
    except FileNotFoundError as e:  # missing image or checkpoint: a message, not a traceback
        print(f"error: {e.filename or e}. Check --image / --checkpoint "
              f"(default checkpoint: {DEFAULT_CHECKPOINT}, the final model; train it first "
              f"or pass --checkpoint).", file=sys.stderr)
        return 2


def _predict(args) -> int:
    from src.evaluation.evaluate import predict_image
    for name, prob in predict_image(args.image, args.checkpoint, topk=args.topk):
        print(f"{prob:.4f}  {name}")
    if args.gradcam_out:
        from PIL import Image
        from src.data.transforms import build_transforms
        from src.training.trainer import load_checkpoint
        from src.visualization.gradcam import GradCAM, denormalize, overlay
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        size = ckpt["config"].get("data", {}).get("image_size", 224)
        model = load_checkpoint(args.checkpoint)
        with Image.open(args.image) as im:
            x = build_transforms(False, size)(im.convert("RGB")).unsqueeze(0)
        cam = GradCAM(model, model.encoder.gradcam_layer())
        try:
            heat = cam(x)
        finally:
            cam.close()
        Path(args.gradcam_out).parent.mkdir(parents=True, exist_ok=True)
        overlay(denormalize(x[0]), heat).save(args.gradcam_out)
        print(f"Grad-CAM overlay -> {args.gradcam_out}")
    return 0


def _test_run(args):
    """Run a checkpoint over the test split. Returns (model, test_ds, class_names, result)."""
    from src.evaluation.evaluate import evaluate
    from src.training.trainer import load_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_config(args.config) if args.config else ckpt["config"]  # config only for data settings
    batch_size = cfg["data"].get("batch_size", cfg["train"]["batch_size"])
    _, _, test_ds, _ = load_cub(cfg)
    model = load_checkpoint(args.checkpoint, device=device)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                         num_workers=cfg["data"].get("num_workers", 0))
    return model, test_ds, ckpt["class_names"], evaluate(model, loader, device)


def cmd_evaluate(args) -> int:
    from src.evaluation.metrics import compute_metrics, confusion, most_confused_pairs
    _, _, names, res = _test_run(args)
    metrics = compute_metrics(res["y_true"], res["y_pred"], res["logits"], len(names))
    pairs = most_confused_pairs(confusion(res["y_true"], res["y_pred"], len(names)))
    named = [{"true": names[t], "pred": names[p], "count": c} for t, p, c in pairs]
    name = args.name or Path(args.checkpoint).stem
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")
    print("most confused pairs (true -> predicted, count):")
    for p in named:
        print(f"  {p['true']} -> {p['pred']}: {p['count']}")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"{name}.json"
    out.write_text(json.dumps({"name": name, "metrics": metrics, "class_names": list(names),
                               "y_true": res["y_true"].tolist(), "y_pred": res["y_pred"].tolist(),
                               "most_confused_pairs": named}, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


def cmd_gradcam(args) -> int:
    from src.evaluation.evaluate import select_gradcam_examples
    from src.visualization.gradcam import GradCAM, denormalize, save_example_grid
    model, test_ds, names, res = _test_run(args)
    probs = torch.softmax(torch.from_numpy(res["logits"]), dim=1).numpy()
    picks = select_gradcam_examples(res["y_true"], res["y_pred"], probs)
    stem = Path(args.checkpoint).stem
    device = next(model.parameters()).device
    cam = GradCAM(model, model.encoder.gradcam_layer())
    try:
        for category, idxs in picks.items():
            items = []
            for i in idxs:
                x, _ = test_ds[i]
                pred = int(res["y_pred"][i])
                items.append({"image": denormalize(x), "cam": cam(x.unsqueeze(0).to(device)),
                              "true": names[res["y_true"][i]], "pred": names[pred],
                              "conf": float(probs[i, pred])})
            path = Path(args.out) / f"gradcam_{stem}_{category}.png"
            save_example_grid(items, f"{stem}: {category}", str(path))
            print(f"{category}: {len(items)} examples -> {path}")
    finally:
        cam.close()
    return 0


def _load_jsons(directory: Path) -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(directory).glob("*.json"))
            if not p.stem.endswith(".partial")}  # partial tuning dumps are not full results


def cmd_figures(args) -> int:
    """Build every comparison figure from whatever result files exist; skip the rest."""
    import yaml
    from src.evaluation.benchmark import benchmark_table
    from src.evaluation.metrics import confusion
    from src.training.tuner import TuneResult
    from src.visualization import comparison as cmp
    out = Path(FIGURES_DIR)
    out.mkdir(parents=True, exist_ok=True)
    histories, evals, tuning = (_load_jsons(d) for d in (HISTORY_DIR, EVAL_DIR, TUNING_DIR))

    def skip(what):
        print(f"skip {what}: no input files")

    if evals:
        cmp.plot_model_comparison({n: e["metrics"] for n, e in evals.items()},
                                  out / "model_comparison.png")
    else:
        skip("model_comparison")
    if histories:
        cmp.plot_training_curves(histories, out / "training_curves.png")
    else:
        skip("training_curves")
    if tuning:
        cmp.plot_tuning_comparison({m: TuneResult(**r) for m, r in tuning.items()},
                                   out / "tuning_comparison.png")
    else:
        skip("tuning_comparison")
    for name, e in evals.items():
        names = e.get("class_names")
        if not names:
            print(f"skip confused_pairs/per_class_f1 for {name}: eval json has no class_names")
            continue
        cm = confusion(e["y_true"], e["y_pred"], len(names))
        cmp.plot_confused_pairs(cm, names, out / f"confused_pairs_{name}.png")
        cmp.plot_per_class_f1(e["y_true"], e["y_pred"], names, out / f"per_class_f1_{name}.png")
    published = []
    if Path(BENCHMARK_CONFIG).exists():
        cfg = yaml.safe_load(Path(BENCHMARK_CONFIG).read_text(encoding="utf-8")) or {}
        published = cfg.get("published") or []
    ours = {n: 100 * e["metrics"]["accuracy"] for n, e in evals.items()}  # percent, like published
    if ours or published:
        df = benchmark_table(ours, published)
        df.to_csv(out / "benchmark.csv", index=False)
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
        (out / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        skip("benchmark")
    print(f"figures -> {out}")
    return 0


HANDLERS = {
    "eda": cmd_eda,
    "train": cmd_train,
    "tune": cmd_tune,
    "predict": cmd_predict,
    "evaluate": cmd_evaluate,
    "gradcam": cmd_gradcam,
    "figures": cmd_figures,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in HANDLERS:
        p = sub.add_parser(name)
        p.add_argument("--config", default=DEFAULT_EDA_CONFIG if name == "eda" else None)
        if name in ("train", "predict"):
            p.add_argument("--task", choices=TASKS, default="classification")
        if name in ("predict", "evaluate", "gradcam"):
            p.add_argument("--checkpoint", required=name != "predict",
                           default=DEFAULT_CHECKPOINT if name == "predict" else None)
        if name == "tune":
            p.set_defaults(config=DEFAULT_TUNE_CONFIG)
            p.add_argument("--method", choices=["grid", "random"], default="grid")
        if name == "train":
            p.add_argument("--smoke", action="store_true")
        if name in ("train", "tune"):
            p.add_argument("--resume", action="store_true",
                           help="skip finished runs and continue an interrupted one")
        if name == "predict":
            p.add_argument("--image", required=True)
            p.add_argument("--topk", type=int, default=5)
            p.add_argument("--gradcam-out")
        if name == "evaluate":
            p.add_argument("--name")
        if name == "gradcam":
            p.add_argument("--out", default=str(FIGURES_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:  # argparse exits on bad input; return its code instead
        return e.code if isinstance(e.code, int) else 2
    return HANDLERS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
