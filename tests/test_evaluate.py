from __future__ import annotations

from src.evaluate import (
    average_precision,
    mean_average_precision,
    precision_at_k,
    topk_accuracy,
)
from src.index import Hit


def h(label: str, score: float = 1.0) -> Hit:
    return Hit(score=score, label=label, path=f"fake/{label}")


def test_top1_accuracy_counts_only_first_result():
    hits = [[h("a"), h("b")], [h("c"), h("c")]]
    truths = ["a", "b"]
    assert topk_accuracy(hits, truths, k=1) == 0.5


def test_top5_accuracy_counts_any_of_top5():
    hits = [[h("x"), h("y"), h("z"), h("a"), h("b")]]
    truths = ["a"]
    assert topk_accuracy(hits, truths, k=5) == 1.0
    assert topk_accuracy(hits, truths, k=1) == 0.0


def test_precision_at_k_is_fraction_correct_in_topk():
    hits = [[h("a"), h("a"), h("b"), h("a"), h("c")]]
    truths = ["a"]
    assert precision_at_k(hits, truths, k=5) == 0.6


def test_average_precision_rewards_early_hits():
    ap_good = average_precision([h("a"), h("x"), h("x")], "a")
    ap_bad = average_precision([h("x"), h("x"), h("a")], "a")
    assert ap_good > ap_bad
    assert ap_good == 1.0
    assert round(ap_bad, 3) == round(1 / 3, 3)


def test_map_averages_per_query():
    hits = [[h("a"), h("a")], [h("x"), h("b")]]
    truths = ["a", "b"]
    m = mean_average_precision(hits, truths)
    assert round(m, 3) == round((1.0 + 0.5) / 2, 3)
