"""Evaluate a saved index against the held-out query split.

    python scripts/run_eval.py --method deep

Writes ``results/<method>.eval.json`` with Top-K, mAP, and latency.

OWNER: Person 4.
"""

from __future__ import annotations

import os

# Must be set before torch/faiss import — macOS ships duplicate libomp copies
# with torch and faiss-cpu, and refuses to run both without this override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import by_split, load_manifest  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.index import RetrievalIndex  # noqa: E402
from src.retrieve import build_method, embed_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["deep", "classical"])
    ap.add_argument("--in-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    samples = load_manifest()
    queries = by_split(samples, "query")
    if not queries:
        print("ERROR: no query samples in the manifest", file=sys.stderr)
        return 1

    index = RetrievalIndex(dim=0)  # dim is overwritten on load
    index.load(Path(args.in_dir) / args.method)
    print(f"loaded index: {len(index)} gallery vectors, dim={index.dim}")

    embedder = build_method(args.method)
    embedder.fit([s.path for s in by_split(samples, "gallery")])
    query_vecs = embed_paths([s.path for s in queries], embedder)

    truths = [s.label for s in queries]
    result = evaluate(index, query_vecs, truths, k=args.k)

    out_path = Path(args.in_dir) / f"{args.method}.eval.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))
    print(json.dumps(asdict(result), indent=2))
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
