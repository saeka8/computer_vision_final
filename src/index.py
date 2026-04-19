"""FAISS-backed retrieval index with a labels+paths sidecar.

Vectors are expected to be L2-normalized, so inner product equals cosine
similarity. Start with ``IndexFlatIP`` — switch to ``IndexIVFPQ`` later
for the scalability story if we have time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Pre-load torch before faiss: on macOS Python 3.13, importing faiss first
# leaves the OpenMP runtime in a state that segfaults any later torch op.
# Safe to skip if torch isn't installed (classical track doesn't need it).
try:
    import torch  # noqa: F401
except ImportError:
    pass

import faiss
import numpy as np


@dataclass
class Hit:
    score: float
    label: str
    path: str


class RetrievalIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._index: faiss.Index | None = None
        self._labels: list[str] = []
        self._paths: list[str] = []

    def build(
        self,
        vectors: np.ndarray,
        labels: list[str],
        paths: list[str],
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected (N, {self.dim}) vectors, got {vectors.shape}"
            )
        if not (len(labels) == len(paths) == vectors.shape[0]):
            raise ValueError("vectors, labels, and paths must align")
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vectors.astype(np.float32))
        self._labels = list(labels)
        self._paths = list(paths)

    def search(self, query: np.ndarray, k: int = 5) -> list[Hit]:
        if self._index is None:
            raise RuntimeError("index not built or loaded")
        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
        scores, ids = self._index.search(q, k)
        return [
            Hit(float(scores[0, i]), self._labels[int(ids[0, i])], self._paths[int(ids[0, i])])
            for i in range(min(k, ids.shape[1]))
            if ids[0, i] != -1
        ]

    def search_batch(self, queries: np.ndarray, k: int = 5) -> list[list[Hit]]:
        if self._index is None:
            raise RuntimeError("index not built or loaded")
        q = np.ascontiguousarray(queries.astype(np.float32))
        scores, ids = self._index.search(q, k)
        out: list[list[Hit]] = []
        for r in range(q.shape[0]):
            row = [
                Hit(float(scores[r, i]), self._labels[int(ids[r, i])], self._paths[int(ids[r, i])])
                for i in range(k)
                if ids[r, i] != -1
            ]
            out.append(row)
        return out

    def save(self, path: Path | str) -> None:
        if self._index is None:
            raise RuntimeError("nothing to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        path.with_suffix(".json").write_text(
            json.dumps({"dim": self.dim, "labels": self._labels, "paths": self._paths})
        )

    def load(self, path: Path | str) -> None:
        path = Path(path)
        self._index = faiss.read_index(str(path.with_suffix(".faiss")))
        meta = json.loads(path.with_suffix(".json").read_text())
        self.dim = int(meta["dim"])
        self._labels = list(meta["labels"])
        self._paths = list(meta["paths"])

    def __len__(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)
