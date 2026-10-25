import json

from src.data.captions import (caption_stats, clean_caption, probable_typos,
                               split_captions, write_clean_captions)


def test_split_captions_drops_blank_and_trailing():
    assert split_captions("a\n\nb\n") == ["a", "b"]


def test_clean_caption_lowercases_and_collapses_spaces():
    assert clean_caption("  A  Red   BIRD. ") == "a red bird."


def test_stats_flags_non_ten():
    caps = [[f"bird {i}" for i in range(10)], [f"bird {i}" for i in range(9)]]
    s = caption_stats(caps)
    assert s["n_images"] == 2 and s["n_not_ten"] == 1


def test_stats_vocab_rare_and_duplicates():
    caps = [["a red bird", "a red bird", "a blue bird"]]
    s = caption_stats(caps)
    assert s["vocab_size"] == 4            # a, red, bird, blue
    assert s["rare_words"] == 1            # "blue" appears once
    assert s["duplicates"] == 1            # one repeated line inside the image
    assert s["mean_len"] == 3.0


def test_probable_typos_finds_near_miss_of_frequent_word():
    caps = [["the yellow bird"] * 30, ["the yelow bird"]]
    assert ("yelow", "yellow") in probable_typos(caps, min_common=20)


def test_write_clean_captions_roundtrip(tmp_path):
    p = tmp_path / "sub" / "c.json"
    write_clean_captions({"x.jpg": ["a", "b"]}, p)
    assert json.loads(p.read_text(encoding="utf-8")) == {"x.jpg": ["a", "b"]}
