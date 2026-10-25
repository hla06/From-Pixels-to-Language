"""Caption splitting, cleaning and statistics. `description` holds ten captions joined by newlines."""
import difflib
import json
from collections import Counter
from pathlib import Path


def split_captions(description: str) -> list[str]:
    """Split on newlines and drop blank lines (incl. a trailing newline)."""
    return [line for line in description.split("\n") if line.strip()]


def clean_caption(text: str) -> str:
    """Lowercase, strip and collapse whitespace. Punctuation and typos are left untouched."""
    return " ".join(text.lower().split())


def _word_counts(captions: list[list[str]]) -> Counter:
    return Counter(w for caps in captions for c in caps for w in c.split())


def caption_stats(captions: list[list[str]]) -> dict:
    """captions: one list of cleaned captions per image."""
    counts = _word_counts(captions)
    lengths = [len(c.split()) for caps in captions for c in caps]
    # duplicates = lines repeated within the same image's caption list
    dups = sum(len(caps) - len(set(caps)) for caps in captions)
    return {
        "n_images": len(captions),
        "n_not_ten": sum(len(caps) != 10 for caps in captions),
        "mean_len": sum(lengths) / len(lengths) if lengths else 0.0,
        "vocab_size": len(counts),
        "rare_words": sum(n == 1 for n in counts.values()),  # words seen exactly once
        "duplicates": dups,
    }


def probable_typos(captions: list[list[str]], min_common: int = 20) -> list[tuple[str, str]]:
    """(rare word, frequent look-alike) pairs: a word seen once that nearly matches a common one."""
    counts = _word_counts(captions)
    common = [w for w, n in counts.items() if n >= min_common]
    pairs = []
    for w, n in counts.items():
        if n == 1:
            match = difflib.get_close_matches(w, common, n=1, cutoff=0.85)
            if match:
                pairs.append((w, match[0]))
    return pairs


def write_clean_captions(records, path) -> None:
    """records: {file_name: [captions]} (or iterable of pairs); saved as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(records), indent=1, ensure_ascii=False), encoding="utf-8")
