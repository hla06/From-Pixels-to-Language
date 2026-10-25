"""Benchmark table: our results next to published ones."""
import pandas as pd

COLUMNS = ["name", "accuracy", "source_url", "setup"]


def benchmark_table(ours: dict[str, float], published: list[dict]) -> pd.DataFrame:
    """Accuracies are percentages. Every published entry must cite a source_url."""
    for entry in published:
        if not str(entry.get("source_url") or "").strip():
            raise ValueError(f"published entry {entry.get('name')!r} has no source_url")
    rows = [{"name": n, "accuracy": a, "source_url": "this work", "setup": ""}
            for n, a in ours.items()]
    rows += [{c: e.get(c, "") for c in COLUMNS} for e in published]
    return pd.DataFrame(rows, columns=COLUMNS)
