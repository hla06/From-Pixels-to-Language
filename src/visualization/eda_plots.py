"""EDA figures and the `run_eda` entry point (wired to the CLI later)."""
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt
import numpy as np

from src.data import cub_dataset
from src.data.captions import (build_typo_map, caption_stats, clean_caption, correct_typos,
                               lowercase_caption, split_captions, write_clean_captions)
from src.data.image_stats import channel_stats, find_duplicates, image_stats
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


def plot_samples(images, titles, captions, out_path: Path, cols: int = 4) -> None:
    """A grid of photos, each titled with its species and one cleaned caption underneath."""
    rows = -(-len(images) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.9 * rows), squeeze=False)
    for ax, im, title, cap in zip(axes.ravel(), images, titles, captions):
        ax.imshow(im)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.text(0.5, -0.04, "\n".join(textwrap.wrap(f'"{cap}"', 38)), transform=ax.transAxes,
                ha="center", va="top", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    _save(fig, out_path)


def plot_image_quality(df, out_path: Path) -> None:
    """Brightness, contrast and sharpness (variance of the Laplacian, log scale) of every image."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(df["brightness"], bins=40)
    axes[0].set(xlabel="mean grey level (0-255)", ylabel="images", title="Brightness")
    axes[1].hist(df["contrast"], bins=40)
    axes[1].set(xlabel="std of grey level", title="Contrast")
    axes[2].hist(np.log10(df["sharpness"] + 1), bins=40)
    axes[2].set(xlabel="log10(1 + variance of Laplacian)", title="Sharpness (low = blurry)")
    _save(fig, out_path)


def plot_top_words(captions, out_path: Path, k: int = 25) -> None:
    """The k most frequent words in the cleaned captions."""
    counts = Counter(w for caps in captions for c in caps for w in c.split()).most_common(k)
    words, n = zip(*counts)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(n)), n)
    ax.set_xticks(range(len(n)), words, rotation=70, ha="right")
    ax.set(ylabel="occurrences", title=f"Top {k} caption words (train, after cleaning)")
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


def _raw_records(split, tag: str) -> dict[str, list[str]]:
    """{file_name: the raw captions}; falls back to '<tag>_<i>' if there is no file_name column."""
    names = _column(split, "file_name") or [f"{tag}_{i}" for i in range(len(split))]
    return {n: split_captions(d) for n, d in zip(names, _column(split, "description"))}


def _sample_indices(labels, k: int = 8) -> list[int]:
    """First image of k classes spread evenly over the label range."""
    first = {}
    for i, lab in enumerate(labels):
        first.setdefault(int(lab), i)
    picks = sorted(first)[::max(1, len(first) // k)][:k]
    return [first[c] for c in picks]


def _fmt(stats: dict) -> str:
    return ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items())


def run_eda(cfg: dict) -> None:
    e = cfg.get("eda", {})
    out = Path(e.get("out_dir", "outputs"))
    notes = e.get("notes_path", out / "NOTES.md")
    train, test, class_names = cub_dataset._load_hf()

    raw_tr, raw_te = _raw_records(train, "train"), _raw_records(test, "test")
    # stage 1: lowercase only; stage 2: punctuation removed; stage 3: typos corrected
    low_tr = [[lowercase_caption(c) for c in caps] for caps in raw_tr.values()]
    norm_tr = {n: [clean_caption(c) for c in caps] for n, caps in raw_tr.items()}
    norm_te = {n: [clean_caption(c) for c in caps] for n, caps in raw_te.items()}
    typo_map = build_typo_map(list(norm_tr.values()))  # from train only: no test words leak in
    recs_tr = {n: [correct_typos(c, typo_map) for c in caps] for n, caps in norm_tr.items()}
    recs_te = {n: [correct_typos(c, typo_map) for c in caps] for n, caps in norm_te.items()}
    write_clean_captions({**recs_tr, **recs_te}, out / "data" / "captions_clean.json")
    (out / "data" / "typo_map.json").write_text(
        json.dumps(typo_map, indent=1, sort_keys=True), encoding="utf-8")
    s_low = caption_stats(low_tr)
    s_norm = caption_stats(list(norm_tr.values()))
    s_tr, s_te = caption_stats(list(recs_tr.values())), caption_stats(list(recs_te.values()))
    n_words = sum(len(c.split()) for caps in norm_tr.values() for c in caps)
    n_changed = sum(a != b for n in norm_tr for a, b in zip(norm_tr[n], recs_tr[n]))

    df = image_stats(train)
    dups = find_duplicates(df)
    n_gray = int((df["mode"] != "RGB").sum())
    small = int((df[["width", "height"]].min(axis=1) < 200).sum())
    dark, bright = int((df["brightness"] < 40).sum()), int((df["brightness"] > 215).sum())
    mean, std = channel_stats(df)

    labels = cub_dataset._labels(train)
    lengths = [len(c.split()) for caps in recs_tr.values() for c in caps]
    plot_class_distribution(labels, out / "eda" / "class_distribution.png")
    plot_image_sizes(df, out / "eda" / "image_sizes.png")
    plot_image_quality(df, out / "eda" / "image_quality.png")
    plot_caption_lengths(lengths, out / "eda" / "caption_lengths.png")
    plot_top_words(list(recs_tr.values()), out / "eda" / "top_words.png")
    plot_augmentations([train[i]["image"].convert("RGB") for i in range(min(3, len(train)))],
                       out / "eda" / "augmentations.png")
    idx = _sample_indices(labels)
    file_names = list(recs_tr)
    plot_samples([train[i]["image"].convert("RGB") for i in idx],
                 [str(class_names[int(labels[i])]) if class_names else str(labels[i]) for i in idx],
                 [recs_tr[file_names[i]][0] for i in idx], out / "eda" / "samples.png")

    shown = list(typo_map.items())[:20]
    body = f"""### Cleaning steps applied (auto-generated by the eda command)
- Split `description` on newline and dropped blank lines.
- Unicode NFKC normalisation and lowercasing.
- Deleted apostrophes ("bird's" -> "birds"); turned every other non-alphanumeric character
  (punctuation, hyphens, quotes) into a space; collapsed whitespace.
- Typo correction: a word seen at most 2 times in the train captions, at least 6 letters long, is
  replaced by the most frequent word (seen at least 20 times) that is one edit away (deletion,
  adjacent swap, substitution or insertion). The map is built from train captions only and applied
  to train and test. Duplicate and unusually short/long captions are kept and counted, not removed.
- Images are converted to 3-channel RGB in the transforms (grayscale/RGBA files).

### Findings
- Train captions, lowercase only: {_fmt(s_low)}
- Train captions, punctuation removed: {_fmt(s_norm)}
- Train captions, typos corrected (final): {_fmt(s_tr)}
- Test captions (final): {_fmt(s_te)}
- Typo map: {len(typo_map)} distinct words corrected, {n_changed} captions changed
  ({n_words} words in the train captions). First 20: {shown}
- Non-RGB train images: {n_gray} of {len(df)}.
- Train image pairs with identical perceptual hash: {len(dups)}.
- Image quality (train, 128x128 thumbnail): brightness min/median/max =
  {df['brightness'].min():.0f}/{df['brightness'].median():.0f}/{df['brightness'].max():.0f};
  {dark} images darker than 40, {bright} brighter than 215; contrast median {df['contrast'].median():.0f};
  sharpness (variance of Laplacian) 1st percentile {df['sharpness'].quantile(0.01):.0f},
  median {df['sharpness'].median():.0f}; {small} images with a side under 200 px.
- Dataset channel mean (RGB) {[round(m, 3) for m in mean]} and std {[round(s, 3) for s in std]};
  ImageNet values used for normalisation: mean {list(IMAGENET_MEAN)}, std {list(IMAGENET_STD)}.
"""
    update_notes(notes, body)
