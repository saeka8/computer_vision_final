"""Retrieval metrics and latency measurement.

All metrics treat a retrieved result as "correct" iff its label matches
the query's label (place recognition, not instance retrieval).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.index import Hit, RetrievalIndex


@dataclass
class EvalResult:
    top1: float
    top5: float
    precision_at_5: float
    mean_average_precision: float
    mean_query_ms: float
    p95_query_ms: float
    n_queries: int


def topk_accuracy(hits_per_query: list[list[Hit]], truths: list[str], k: int) -> float:
    if not hits_per_query:
        return 0.0
    correct = 0
    for hits, truth in zip(hits_per_query, truths):
        top_labels = [h.label for h in hits[:k]]
        if truth in top_labels:
            correct += 1
    return correct / len(hits_per_query)


def precision_at_k(hits_per_query: list[list[Hit]], truths: list[str], k: int) -> float:
    if not hits_per_query:
        return 0.0
    scores = []
    for hits, truth in zip(hits_per_query, truths):
        top = hits[:k]
        if not top:
            scores.append(0.0)
            continue
        scores.append(sum(1 for h in top if h.label == truth) / len(top))
    return float(np.mean(scores))


def average_precision(hits: list[Hit], truth: str) -> float:
    """AP over the returned list. Treats the returned list as the full ranking."""
    if not hits:
        return 0.0
    hits_correct = [h.label == truth for h in hits]
    total_relevant = sum(hits_correct)
    if total_relevant == 0:
        return 0.0
    score = 0.0
    seen = 0
    for i, is_rel in enumerate(hits_correct, start=1):
        if is_rel:
            seen += 1
            score += seen / i
    return score / total_relevant


def mean_average_precision(
    hits_per_query: list[list[Hit]], truths: list[str]
) -> float:
    if not hits_per_query:
        return 0.0
    return float(
        np.mean([average_precision(h, t) for h, t in zip(hits_per_query, truths)])
    )


def time_queries(
    index: RetrievalIndex,
    queries: np.ndarray,
    k: int = 5,
    warmup: int = 3,
) -> tuple[list[list[Hit]], list[float]]:
    """Run each query one-at-a-time so the latency numbers reflect the
    real single-query path, not the batched path."""
    for _ in range(min(warmup, queries.shape[0])):
        index.search(queries[0], k=k)

    hits_per_query: list[list[Hit]] = []
    per_query_ms: list[float] = []
    for i in range(queries.shape[0]):
        t0 = time.perf_counter()
        hits = index.search(queries[i], k=k)
        per_query_ms.append((time.perf_counter() - t0) * 1000)
        hits_per_query.append(hits)
    return hits_per_query, per_query_ms


def evaluate(
    index: RetrievalIndex,
    queries: np.ndarray,
    truths: list[str],
    k: int = 5,
) -> EvalResult:
    hits_per_query, per_query_ms = time_queries(index, queries, k=k)
    return EvalResult(
        top1=topk_accuracy(hits_per_query, truths, k=1),
        top5=topk_accuracy(hits_per_query, truths, k=5),
        precision_at_5=precision_at_k(hits_per_query, truths, k=5),
        mean_average_precision=mean_average_precision(hits_per_query, truths),
        mean_query_ms=float(np.mean(per_query_ms)),
        p95_query_ms=float(np.percentile(per_query_ms, 95)),
        n_queries=len(truths),
    )
