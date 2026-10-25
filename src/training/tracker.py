"""Experiment tracking behind one interface (Strategy pattern).

The training loop only talks to `Tracker`; W&B or a no-op backend plugs in.
"""
import os
from abc import ABC, abstractmethod

BIG_FILE_BYTES = 100 * 1024 * 1024


class Tracker(ABC):
    @abstractmethod
    def log_params(self, params: dict) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str, name: str) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


class NullTracker(Tracker):
    """Records calls in memory; used by tests and offline runs."""

    def __init__(self):
        self.calls: list[tuple] = []

    def log_params(self, params):
        self.calls.append(("log_params", params))

    def log_metrics(self, metrics, step):
        self.calls.append(("log_metrics", metrics, step))

    def log_artifact(self, path, name):
        self.calls.append(("log_artifact", path, name))

    def finish(self):
        self.calls.append(("finish",))


class WandbTracker(Tracker):
    def __init__(self, project, run_name, group=None, config=None):
        import wandb  # lazy: tests never need wandb or a login

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, group=group, config=config)

    def log_params(self, params):
        self._run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics, step):
        self._run.log(metrics, step=step)

    def log_artifact(self, path, name):
        artifact = self._wandb.Artifact(name, type="model")
        if os.path.getsize(path) > BIG_FILE_BYTES:
            # too big to upload: store only a reference to the file
            artifact.add_reference("file://" + os.path.abspath(path))
        else:
            artifact.add_file(path)
        self._run.log_artifact(artifact)

    def finish(self):
        self._run.finish()


def build_tracker(cfg: dict) -> Tracker:
    t = cfg["tracker"]
    name = t["name"]
    if name == "null":
        return NullTracker()
    if name == "wandb":
        return WandbTracker(
            project=t.get("project", "pixels-to-language"),
            run_name=t.get("run_name"),
            group=t.get("group"),
            config=cfg,
        )
    raise ValueError(f"Unknown tracker '{name}'; valid names: 'wandb', 'null'")
