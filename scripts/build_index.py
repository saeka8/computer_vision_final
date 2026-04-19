"""Build and persist a retrieval index for a given method.

    python scripts/build_index.py --method deep
    python scripts/build_index.py --method classical

Writes ``results/<method>.faiss`` and ``results/<method>.json``.

OWNER: Person 4.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import by_split, load_manifest  # noqa: E402
from src.index import RetrievalIndex  # noqa: E402
from src.retrieve import build_method, embed_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["deep", "classical"])
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    samples = load_manifest()
    gallery = by_split(samples, "gallery")
    train = by_split(samples, "gallery")  # reuse gallery for any fitting step

    embedder = build_method(args.method)
    print(f"[{embedder.name}] fitting on {len(train)} images ...")
    t0 = time.perf_counter()
    embedder.fit([s.path for s in train])
    print(f"  fit done in {time.perf_counter() - t0:.1f}s")

    print(f"[{embedder.name}] embedding {len(gallery)} gallery images ...")
    t0 = time.perf_counter()
    vectors = embed_paths([s.path for s in gallery], embedder)
    print(f"  embed done in {time.perf_counter() - t0:.1f}s -> {vectors.shape}")

    index = RetrievalIndex(dim=embedder.dim)
    index.build(vectors, [s.label for s in gallery], [str(s.path) for s in gallery])

    out = Path(args.out_dir) / args.method
    index.save(out)
    print(f"saved index: {out}.faiss (+ {out}.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
