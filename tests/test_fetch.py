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
