"""Download the best checkpoints that training uploaded to W&B (one artifact per run)."""
import json
from pathlib import Path

FINAL_RUNS = ["scratch_cnn", "resnet18_random", "resnet18_frozen", "resnet18_finetune",
              "resnet18_tuned", "resnet50_frozen", "resnet50_finetune",
              "efficientnet_b0_frozen", "efficientnet_b0_finetune"]


def fetch_checkpoints(run_names: list[str], dest: str, project: str = "pixels-to-language",
                      entity: str | None = None, api=None) -> list[str]:
    """Fetch `<run>-best:latest` into dest/<run>.pt; returns the runs that could not be fetched.

    Runs whose checkpoint already exists are skipped, so it is safe to call after every reset.
    `api` can be injected (tests); by default it is wandb.Api().
    """
    if api is None:
        import wandb
        api = wandb.Api()
    entity = entity or api.default_entity
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for run in run_names:
        if (dest_dir / f"{run}.pt").exists():
            print(f"{run}: already in {dest_dir}")
            continue
        try:
            api.artifact(f"{entity}/{project}/{run}-best:latest").download(root=str(dest_dir))
        except Exception as e:  # missing artifact or a reference-only upload; keep going
            print(f"{run}: could not fetch ({e})")
            failed.append(run)
            continue
        ok = (dest_dir / f"{run}.pt").exists()
        print(f"{run}: {'downloaded' if ok else 'artifact has no .pt file (reference only?)'}")
        if not ok:
            failed.append(run)
    return failed


HISTORY_KEYS = ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]


def fetch_histories(run_names: list[str], dest: str, project: str = "pixels-to-language",
                    entity: str | None = None, api=None) -> list[str]:
    """Rebuild `dest/<run>.json` (the per-epoch history the trainer writes) from W&B.

    Several W&B runs can share a name (interrupted sessions were re-run), so the run is the one
    that logged the `<run>-best` checkpoint artifact: the same run whose checkpoint was evaluated.
    Existing files are kept. Returns the runs that could not be fetched.
    """
    if api is None:
        import wandb
        api = wandb.Api()
    entity = entity or api.default_entity
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in run_names:
        out = dest_dir / f"{name}.json"
        if out.exists():
            print(f"{name}: already in {dest_dir}")
            continue
        try:
            run = api.artifact(f"{entity}/{project}/{name}-best:latest").logged_by()
            rows = [r for r in run.scan_history() if "val_acc" in r]
            history = [{"epoch": int(r["_step"]), **{k: r[k] for k in HISTORY_KEYS}} for r in rows]
            if not history:
                raise ValueError("run has no epoch metrics")
        except Exception as e:  # missing artifact, run deleted, or no metrics logged
            print(f"{name}: could not fetch ({e})")
            failed.append(name)
            continue
        out.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"{name}: {len(history)} epochs from run {run.id}")
    return failed
