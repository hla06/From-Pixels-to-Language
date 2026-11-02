import json

from src.data.captions import (build_typo_map, caption_stats, clean_caption, correct_typos,
                               lowercase_caption, split_captions, write_clean_captions)


def test_split_captions_drops_blank_and_trailing():
    assert split_captions("a\n\nb\n") == ["a", "b"]


def test_lowercase_caption_keeps_punctuation():
    assert lowercase_caption("  A  Red   BIRD. ") == "a red bird."


def test_clean_caption_strips_punctuation_and_hyphens():
    assert clean_caption("  A  Red   BIRD. ") == "a red bird"
    assert clean_caption('Long-tailed, "grey" bird;') == "long tailed grey bird"


def test_clean_caption_joins_apostrophes():
    assert clean_caption("the bird's tail isn’t long") == "the birds tail isnt long"


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


def test_typo_map_fixes_rare_near_miss_of_frequent_word():
    caps = [["the yellow bird"] * 30, ["the yelow bird", "the yellwo bird"]]
    tm = build_typo_map(caps, min_common=20, min_len=5)
    assert tm == {"yelow": "yellow", "yellwo": "yellow"}  # deletion and adjacent swap
    assert build_typo_map(caps, min_common=20) == {"yellwo": "yellow"}  # default skips 5-letter words
    assert correct_typos("the yelow bird", tm) == "the yellow bird"


def test_typo_map_leaves_short_frequent_and_far_words_alone():
    caps = [["a wing is red"] * 30, ["a wind is red", "a wingspan is red", "a wings are red"]]
    tm = build_typo_map(caps, min_common=20, min_len=4)
    assert "wings" in tm             # one edit from the frequent "wing"
    assert "wind" in tm              # (a valid word: why short words are skipped by default)
    tm = build_typo_map(caps, min_common=20)
    assert "wind" not in tm and "wings" not in tm  # shorter than min_len
    assert "wingspan" not in tm      # more than one edit from any frequent word
    assert "wing" not in tm          # frequent itself


def test_stats_flag_short_and_long_captions():
    s = caption_stats([["bird", "a b c d e f g h"]], short=3, long=8)
    assert s["n_short"] == 1 and s["n_long"] == 1
    assert s["min_len"] == 1 and s["max_len"] == 8


def test_write_clean_captions_roundtrip(tmp_path):
    p = tmp_path / "sub" / "c.json"
    write_clean_captions({"x.jpg": ["a", "b"]}, p)
    assert json.loads(p.read_text(encoding="utf-8")) == {"x.jpg": ["a", "b"]}
