from src.cli import main
from src.training.fetch import fetch_checkpoints


class FakeArtifact:
    def __init__(self, name):
        self.name = name

    def download(self, root):
        if "missing" in self.name:
            raise RuntimeError("not found")
        run = self.name.split("/")[-1].split("-best")[0]
        (open(f"{root}/{run}.pt", "w")).write("x")


class FakeApi:
    default_entity = "me"

    def __init__(self):
        self.asked = []

    def artifact(self, name):
        self.asked.append(name)
        return FakeArtifact(name)


def test_fetch_downloads_skips_existing_and_reports_failures(tmp_path):
    (tmp_path / "have.pt").write_text("x")
    api = FakeApi()
    failed = fetch_checkpoints(["have", "new", "missing"], str(tmp_path), api=api)
    assert failed == ["missing"]
    assert (tmp_path / "new.pt").exists()
    assert api.asked == ["me/pixels-to-language/new-best:latest",
                         "me/pixels-to-language/missing-best:latest"]


def test_cli_command_registered(monkeypatch, tmp_path):
    import src.training.fetch as f
    monkeypatch.setattr(f, "fetch_checkpoints", lambda names, dest, project: [])
    assert main(["fetch-checkpoints", "--dest", str(tmp_path)]) == 0


class FakeRun:
    id = "abc"

    def __init__(self, rows):
        self._rows = rows

    def scan_history(self):
        return iter(self._rows)


def _rows(n):
    return [{"_step": i, "train_loss": 2.0 - i / 10, "val_loss": 2.1 - i / 10,
             "train_acc": i / 10, "val_acc": i / 12, "lr": 1e-3, "_runtime": 5} for i in range(1, n + 1)]


class HistoryApi:
    default_entity = "me"

    def artifact(self, name):
        run_name = name.split("/")[-1].split("-best")[0]
        rows = {"good": _rows(3), "empty": []}.get(run_name)
        if rows is None:
            raise RuntimeError("not found")
        return type("A", (), {"logged_by": lambda self: FakeRun(rows)})()


def test_fetch_histories_writes_trainer_format_and_reports_failures(tmp_path):
    import json
    from src.training.fetch import fetch_histories
    (tmp_path / "have.json").write_text("[]")
    failed = fetch_histories(["have", "good", "empty", "missing"], str(tmp_path), api=HistoryApi())
    assert failed == ["empty", "missing"]
    h = json.loads((tmp_path / "good.json").read_text())
    assert [r["epoch"] for r in h] == [1, 2, 3]
    assert set(h[0]) == {"epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"}
    assert (tmp_path / "have.json").read_text() == "[]"


def test_fetch_history_cli_registered(monkeypatch, tmp_path):
    import src.training.fetch as f
    monkeypatch.setattr(f, "fetch_histories", lambda names, dest, project: [])
    assert main(["fetch-history", "--dest", str(tmp_path)]) == 0
