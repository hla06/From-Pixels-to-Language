import pytest

from src.training.tracker import NullTracker, build_tracker


def test_null_tracker_records():
    t = NullTracker()
    t.log_metrics({"loss": 1.0}, step=2)
    assert ("log_metrics", {"loss": 1.0}, 2) in t.calls


def test_build_tracker_null():
    assert isinstance(build_tracker({"tracker": {"name": "null"}}), NullTracker)


def test_build_tracker_unknown():
    with pytest.raises(ValueError, match="wandb"):
        build_tracker({"tracker": {"name": "foo"}})
