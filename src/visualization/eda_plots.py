"""EDA figures and the `run_eda` entry point (wired to the CLI later)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt
import numpy as np

from src.data import cub_dataset
from src.data.captions import (caption_stats, clean_caption, probable_typos,
                               split_captions, write_clean_captions)
from src.data.image_stats import find_duplicates, image_stats
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms

START, END = "<!-- eda:start -->", "<!-- eda:end -->"


def _save(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_class_distribution(labels, out_path: Path) -> None:
    counts = np.bincount(np.asarray(labels))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(len(counts)), counts)
    ax.set(xlabel="class id", ylabel="images", title="Images per class (train)")
    _save(fig, out_path)


def plot_image_sizes(df, out_path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.scatter(df["width"], df["height"], s=4, alpha=0.4)
    a.set(xlabel="width", ylabel="height", title="Image sizes")
    b.hist(df["width"] / df["height"], bins=40)
    b.set(xlabel="aspect ratio (w/h)", title="Aspect ratios")
    _save(fig, out_path)


def plot_caption_lengths(lengths, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=range(0, int(max(lengths)) + 2))
    ax.set(xlabel="words per caption", ylabel="captions", title="Caption length")
    _save(fig, out_path)


def plot_augmentations(images, out_path: Path, n_aug: int = 3) -> None:
    """One row per image: the original, then n_aug random training augmentations."""
    tf = build_transforms(train=True)
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    fig, axes = plt.subplots(len(images), n_aug + 1,
                             figsize=(2.5 * (n_aug + 1), 2.5 * len(images)), squeeze=False)
    for r, im in enumerate(images):
        axes[r, 0].imshow(im)
        axes[r, 0].set_title("original" if r == 0 else "")
        for c in range(1, n_aug + 1):
            x = tf(im).permute(1, 2, 0).numpy() * std + mean  # undo normalisation for display
            axes[r, c].imshow(np.clip(x, 0, 1))
            axes[r, c].set_title("augmented" if r == 0 else "")
    for ax in axes.ravel():
        ax.axis("off")
    _save(fig, out_path)


def update_notes(path, body: str) -> None:
    """Keep `body` in one delimited block under '## Cleaning steps'; re-runs replace it."""
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{START}\n{body.strip()}\n{END}"
    if START in text:
        head, rest = text.split(START, 1)
        text = head + block + rest.split(END, 1)[1]
    elif "## Cleaning steps" in text:
        text = text.replace("## Cleaning steps", f"## Cleaning steps\n\n{block}", 1)
    else:
        text += f"\n## Cleaning steps\n\n{block}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _column(split, name: str):
    """Column values; reads the HF column directly (no image decoding), or walks a list of dicts."""
    if hasattr(split, "column_names"):
        return split[name] if name in split.column_names else None
    return [r[name] for r in split] if name in split[0] else None


def _clean_records(split, tag: str) -> dict[str, list[str]]:
    """{file_name: cleaned captions}; falls back to '<tag>_<i>' if there is no file_name column."""
    names = _column(split, "file_name") or [f"{tag}_{i}" for i in range(len(split))]
    return {n: [clean_caption(c) for c in split_captions(d)]
            for n, d in zip(names, _column(split, "description"))}


def run_eda(cfg: dict) -> None:
    e = cfg.get("eda", {})
    out = Path(e.get("out_dir", "outputs"))
    notes = e.get("notes_path", out / "NOTES.md")
    train, test, _ = cub_dataset._load_hf()

    recs_tr, recs_te = _clean_records(train, "train"), _clean_records(test, "test")
    write_clean_captions({**recs_tr, **recs_te}, out / "data" / "captions_clean.json")
    s_tr, s_te = caption_stats(list(recs_tr.values())), caption_stats(list(recs_te.values()))
    typos = probable_typos(list(recs_tr.values()) + list(recs_te.values()))

    df = image_stats(train)
    dups = find_duplicates(df)
    n_gray = int((df["mode"] != "RGB").sum())

    lengths = [len(c.split()) for caps in recs_tr.values() for c in caps]
    plot_class_distribution(cub_dataset._labels(train), out / "eda" / "class_distribution.png")
    plot_image_sizes(df, out / "eda" / "image_sizes.png")
    plot_caption_lengths(lengths, out / "eda" / "caption_lengths.png")
    plot_augmentations([train[i]["image"].convert("RGB") for i in range(min(3, len(train)))],
                       out / "eda" / "augmentations.png")

    body = f"""### Cleaning steps applied (auto-generated by the eda command)
- Split `description` on newline and dropped blank lines.
- Lowercased, stripped and collapsed whitespace in every caption. Punctuation and typos were not changed automatically.
- Images are converted to 3-channel RGB in the transforms (grayscale/RGBA files).

### Findings
- Train captions: {s_tr}
- Test captions: {s_te}
- Non-RGB train images: {n_gray} of {len(df)}.
- Train image pairs with identical perceptual hash: {len(dups)}.
- Probable typos (rare word, frequent look-alike): {len(typos)} found, first 20: {typos[:20]}
"""
    update_notes(notes, body)
