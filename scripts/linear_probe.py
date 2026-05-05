"""Linear probe on top of frozen DINOv2 features.

    python scripts/linear_probe.py

Standard self-supervised eval protocol: freeze the backbone, train a
logistic regression on the embeddings, report classification accuracy.
With ~100 images we can't fine-tune the ViT itself, but the linear
probe is the right way to show the features are class-discriminative
and to validate the retrieval result with an independent classifier.

Writes ``results/linear_probe.eval.json`` next to the retrieval evals.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import by_split, load_manifest  # noqa: E402
from src.features.deep import DinoV2Embedder  # noqa: E402
from src.retrieve import embed_paths  # noqa: E402


@dataclass
class ProbeResult:
    top1: float
    top5: float
    n_queries: int
    n_classes: int
    n_train: int
    best_C: float
    cv_top1: float
    mean_query_ms: float
    p95_query_ms: float


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument(
        "--cache",
        default=str(REPO_ROOT / "results" / "deep_embeddings.npz"),
        help="Cache file for DINOv2 embeddings — recompute if missing.",
    )
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    samples = load_manifest()
    gallery = by_split(samples, "gallery")
    queries = by_split(samples, "query")
    if not queries:
        print("ERROR: no query samples in manifest", file=sys.stderr)
        return 1

    cache = Path(args.cache)
    if cache.exists() and not args.no_cache:
        print(f"loading cached embeddings: {cache}")
        z = np.load(cache, allow_pickle=True)
        X_gal, y_gal = z["X_gal"], z["y_gal"]
        X_qry, y_qry = z["X_qry"], z["y_qry"]
    else:
        embedder = DinoV2Embedder()
        print(f"[{embedder.name}] embedding {len(gallery)} gallery + {len(queries)} query images ...")
        t0 = time.perf_counter()
        X_gal = embed_paths([s.path for s in gallery], embedder)
        X_qry = embed_paths([s.path for s in queries], embedder)
        print(f"  embed done in {time.perf_counter() - t0:.1f}s")
        y_gal = np.array([s.label for s in gallery])
        y_qry = np.array([s.label for s in queries])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X_gal=X_gal, y_gal=y_gal, X_qry=X_qry, y_qry=y_qry)
        print(f"cached -> {cache}")

    classes = np.unique(y_gal)
    print(f"gallery: {len(X_gal)} vecs, {len(classes)} classes; query: {len(X_qry)} vecs")

    # Tune regularization with stratified CV on the gallery. Skip CV if
    # any class has <2 samples (StratifiedKFold can't split it).
    counts = np.array([np.sum(y_gal == c) for c in classes])
    can_cv = counts.min() >= 2
    Cs = [0.1, 1.0, 10.0, 100.0]
    if can_cv:
        n_splits = int(min(3, counts.min()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for C in Cs:
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
            s = cross_val_score(clf, X_gal, y_gal, cv=cv, scoring="accuracy")
            scores.append((C, float(s.mean())))
            print(f"  C={C:>6}: cv_top1={s.mean():.3f}")
        best_C, best_cv = max(scores, key=lambda t: t[1])
    else:
        print("  (skipping CV — some classes have <2 gallery samples)")
        best_C, best_cv = 1.0, float("nan")

    print(f"fitting logistic regression on full gallery (C={best_C}) ...")
    clf = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced")
    clf.fit(X_gal, y_gal)

    # Inference timing per single query (matches the retrieval eval format).
    per_query_ms: list[float] = []
    proba_rows: list[np.ndarray] = []
    for i in range(X_qry.shape[0]):
        t0 = time.perf_counter()
        p = clf.predict_proba(X_qry[i : i + 1])[0]
        per_query_ms.append((time.perf_counter() - t0) * 1000)
        proba_rows.append(p)
    proba = np.stack(proba_rows)
    cls_order = clf.classes_

    # Top-1 and Top-5 by predicted probability.
    top1_idx = proba.argmax(axis=1)
    top1_pred = cls_order[top1_idx]
    top1 = float(np.mean(top1_pred == y_qry))

    k5 = min(5, proba.shape[1])
    top5_idx = np.argsort(-proba, axis=1)[:, :k5]
    top5_correct = 0
    for i, true in enumerate(y_qry):
        if true in cls_order[top5_idx[i]]:
            top5_correct += 1
    top5 = top5_correct / len(y_qry)

    result = ProbeResult(
        top1=top1,
        top5=top5,
        n_queries=int(len(y_qry)),
        n_classes=int(len(classes)),
        n_train=int(len(X_gal)),
        best_C=float(best_C),
        cv_top1=float(best_cv),
        mean_query_ms=float(np.mean(per_query_ms)),
        p95_query_ms=float(np.percentile(per_query_ms, 95)),
    )

    out_path = Path(args.out_dir) / "linear_probe.eval.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))
    print(json.dumps(asdict(result), indent=2))
    print(f"\nsaved: {out_path}")

    # Per-query breakdown for the report — useful for the confusion table.
    detail_path = Path(args.out_dir) / "linear_probe.predictions.json"
    detail = []
    for i, s in enumerate(queries):
        detail.append(
            {
                "path": str(s.path.relative_to(REPO_ROOT)),
                "true": s.label,
                "pred": str(top1_pred[i]),
                "top5": [str(c) for c in cls_order[top5_idx[i]]],
                "confidence": float(proba[i, top1_idx[i]]),
            }
        )
    detail_path.write_text(json.dumps(detail, indent=2))
    print(f"saved: {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
