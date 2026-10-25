from collections import Counter

from src.data.splits import stratified_split

LABELS = [c for c in range(10) for _ in range(30)]


def test_split_disjoint_and_complete():
    tr, va = stratified_split(LABELS)
    assert set(tr).isdisjoint(va)
    assert sorted(tr + va) == list(range(len(LABELS)))


def test_split_every_class_in_both():
    tr, va = stratified_split(LABELS, val_fraction=0.2)
    assert Counter(LABELS[i] for i in va) == {c: 6 for c in range(10)}
    assert {LABELS[i] for i in tr} == set(range(10))


def test_split_deterministic_by_seed():
    assert stratified_split(LABELS, seed=1) == stratified_split(LABELS, seed=1)
    assert stratified_split(LABELS, seed=1) != stratified_split(LABELS, seed=2)


def test_single_image_class_goes_to_train():
    labels = LABELS + [99]
    tr, va = stratified_split(labels)
    assert len(labels) - 1 in tr
    assert len(labels) - 1 not in va


def test_random_split_is_disjoint_complete_and_seeded():
    from src.data.splits import random_split
    tr, va = random_split(10, 0.2, seed=1)
    assert len(va) == 2 and sorted(tr + va) == list(range(10))
    assert (tr, va) == random_split(10, 0.2, seed=1)
