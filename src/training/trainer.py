"""Training loop (Template Method), optimizer/scheduler builders and checkpoint I/O."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.classifier import Classifier
from src.models.registry import build_encoder
from src.training.tracker import Tracker


def build_optimizer(model: Classifier, cfg: dict) -> torch.optim.Optimizer:
    """AdamW with a small lr for the pretrained encoder and a larger one for the new head."""
    t = cfg["train"]
    groups = []
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc_params:  # a frozen encoder contributes no group
        groups.append({"params": enc_params, "lr": t["lr_backbone"]})
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    groups.append({"params": head_params, "lr": t["lr_head"]})  # head is always the last group
    return torch.optim.AdamW(groups, weight_decay=t["weight_decay"])


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Per-step linear warmup, then cosine decay to ~0."""
    t = cfg["train"]
    warmup = max(1, int(t.get("warmup_epochs", 1) * steps_per_epoch))
    total = max(warmup + 1, t["epochs"] * steps_per_epoch)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / (total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    """fit() fixes the order of steps; train_epoch/validate/save_checkpoint can be overridden."""

    def __init__(self, model, loaders: dict[str, DataLoader], optimizer, scheduler,
                 tracker: Tracker, device: str, cfg: dict, class_names=None):
        self.model = model.to(device)
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tracker = tracker
        self.device = device
        self.cfg = cfg
        self.class_names = list(class_names) if class_names is not None else None
        t = cfg["train"]
        self.criterion = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.1))
        # Mixed precision only on GPU; on CPU both objects are disabled no-ops.
        self.use_amp = device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val_acc = -1.0
        self.checkpoint_path = self._checkpoint_path()

    def _checkpoint_path(self) -> str:
        run_name = self.cfg.get("tracker", {}).get("run_name")
        folder = Path(self.cfg.get("checkpoint_dir", "outputs/checkpoints"))
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder / (f"{run_name}.pt" if run_name else "best.pt"))

    def fit(self) -> dict:
        t = self.cfg["train"]
        self.tracker.log_params({"train": t, "encoder": self.cfg.get("encoder", {})})
        history, bad_epochs = [], 0
        for epoch in range(1, t["epochs"] + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            lr = self.optimizer.param_groups[-1]["lr"]
            metrics = {"train_loss": train_loss, "val_loss": val_loss,
                       "train_acc": train_acc, "val_acc": val_acc, "lr": lr}
            self.tracker.log_metrics(metrics, step=epoch)
            history.append({"epoch": epoch, **metrics})
            if val_acc > self.best_val_acc:
                self.best_val_acc, bad_epochs = val_acc, 0
                self.save_checkpoint()
            else:
                bad_epochs += 1
                if bad_epochs >= t["patience"]:
                    break
        run_name = self.cfg.get("tracker", {}).get("run_name") or "best-model"
        # unique per run so W&B artifacts of different runs do not collapse into one lineage
        self.tracker.log_artifact(self.checkpoint_path, f"{run_name}-best")
        self.tracker.finish()
        return {"best_val_acc": self.best_val_acc, "history": history,
                "checkpoint": self.checkpoint_path}

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in self.loaders["train"]:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()  # per step, matching build_scheduler
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        return total_loss / n, correct / n

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in self.loaders["val"]:
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device.split(":")[0], enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        return total_loss / n, correct / n

    def save_checkpoint(self) -> None:
        names = self.class_names or [str(i) for i in range(self.model.head[1].out_features)]
        torch.save({"model_state": self.model.state_dict(), "config": self.cfg,
                    "class_names": names, "val_acc": self.best_val_acc}, self.checkpoint_path)


def load_checkpoint(path: str, device: str = "cpu") -> Classifier:
    """Rebuild the Classifier from the stored config and load its weights (eval mode)."""
    # map_location lets a GPU-saved checkpoint load on a CPU-only machine.
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    encoder = build_encoder(cfg["encoder"]["name"], pretrained=False, freeze=False)
    model = Classifier(encoder, num_classes=len(ckpt["class_names"]),
                       dropout=cfg["train"]["dropout"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()
