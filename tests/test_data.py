from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.data import make_splits


def test_singleton_class_goes_to_gallery_only():
    pairs = [(Path(f"data/solo/img{i}.jpg"), "solo") for i in range(1)]
    samples = make_splits(pairs)
    assert all(s.split == "gallery" for s in samples)


def test_stratified_split_holds_out_queries():
    pairs = []
    for cls in ("a", "b", "c"):
        pairs.extend((Path(f"data/{cls}/img{i}.jpg"), cls) for i in range(10))
    samples = make_splits(pairs, query_frac=0.2, seed=0)

    by_split: dict[str, Counter] = {
        "gallery": Counter(),
        "query": Counter(),
    }
    for s in samples:
        by_split[s.split][s.label] += 1

    for cls in ("a", "b", "c"):
        assert by_split["query"][cls] >= 1
        assert by_split["gallery"][cls] >= 1
        assert by_split["query"][cls] + by_split["gallery"][cls] == 10


def test_split_is_deterministic_given_seed():
    pairs = [(Path(f"data/x/img{i}.jpg"), "x") for i in range(8)]
    a = make_splits(pairs, seed=123)
    b = make_splits(pairs, seed=123)
    assert [(str(s.path), s.split) for s in a] == [(str(s.path), s.split) for s in b]
