"""Caption splitting, cleaning and statistics. `description` holds ten captions joined by newlines.

Cleaning has two stages. `clean_caption` works on one caption (case, punctuation, whitespace).
`build_typo_map` / `correct_typos` work on the whole training corpus, because a typo is only
recognisable by comparing a rare word with the frequent words around it.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

LETTERS = "abcdefghijklmnopqrstuvwxyz"
_APOSTROPHES = re.compile(r"['’`]")
_NOT_WORD = re.compile(r"[^a-z0-9]+")


def split_captions(description: str) -> list[str]:
    """Split on newlines and drop blank lines (incl. a trailing newline)."""
    return [line for line in description.split("\n") if line.strip()]


def lowercase_caption(text: str) -> str:
    """Lowercase, strip and collapse whitespace only (the stage before punctuation removal)."""
    return " ".join(text.lower().split())


def clean_caption(text: str) -> str:
    """Normalise one caption: Unicode NFKC, lowercase, delete apostrophes ("bird's" -> "birds"),
    turn every other non-alphanumeric character (punctuation, hyphens, quotes) into a space and
    collapse whitespace. "long-tailed bird." -> "long tailed bird"."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _APOSTROPHES.sub("", text)
    return " ".join(_NOT_WORD.sub(" ", text).split())


def _word_counts(captions: list[list[str]]) -> Counter:
    return Counter(w for caps in captions for c in caps for w in c.split())


def caption_stats(captions: list[list[str]], short: int = 3, long: int = 40) -> dict:
    """captions: one list of cleaned captions per image. `short`/`long` flag noisy lengths."""
    counts = _word_counts(captions)
    lengths = [len(c.split()) for caps in captions for c in caps]
    # duplicates = lines repeated within the same image's caption list
    dups = sum(len(caps) - len(set(caps)) for caps in captions)
    return {
        "n_images": len(captions),
        "n_not_ten": sum(len(caps) != 10 for caps in captions),
        "mean_len": sum(lengths) / len(lengths) if lengths else 0.0,
        "min_len": min(lengths, default=0),
        "max_len": max(lengths, default=0),
        "n_short": sum(n <= short for n in lengths),  # captions of at most `short` words
        "n_long": sum(n >= long for n in lengths),    # captions of at least `long` words
        "vocab_size": len(counts),
        "rare_words": sum(n == 1 for n in counts.values()),  # words seen exactly once
        "duplicates": dups,
    }


def _edits1(word: str) -> set[str]:
    """Every string one edit away: deletion, adjacent swap, substitution or insertion."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    out = {a + b[1:] for a, b in splits if b}
    out |= {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    out |= {a + c + b[1:] for a, b in splits if b for c in LETTERS}
    out |= {a + c + b for a, b in splits for c in LETTERS}
    out.discard(word)
    return out


def build_typo_map(captions: list[list[str]], min_common: int = 20, max_rare: int = 2,
                   min_len: int = 6) -> dict[str, str]:
    """{typo: correction}. A word seen at most `max_rare` times is mapped to the most frequent
    word (seen at least `min_common` times) that is one edit away. Words shorter than `min_len`
    are never touched (too many valid short words differ by one letter), and neither are words
    that are themselves frequent. Build it from the training captions only."""
    counts = _word_counts(captions)
    typo_map = {}
    for word, n in counts.items():
        if n > max_rare or len(word) < min_len or not word.isalpha():
            continue
        candidates = [w for w in _edits1(word) if counts.get(w, 0) >= min_common]
        if candidates:
            typo_map[word] = max(candidates, key=lambda w: counts[w])
    return typo_map


def correct_typos(caption: str, typo_map: dict[str, str]) -> str:
    return " ".join(typo_map.get(w, w) for w in caption.split())


def write_clean_captions(records, path) -> None:
    """records: {file_name: [captions]} (or iterable of pairs); saved as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(records), indent=1, ensure_ascii=False), encoding="utf-8")
