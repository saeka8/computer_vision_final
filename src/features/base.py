"""Shared contract both feature tracks implement.

Any class that satisfies this protocol can be plugged into
``src/index.py`` and ``src/evaluate.py`` without changes — which is what
makes the classical-vs-deep comparison fair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Produces a single L2-normalized vector per image."""

    name: str
    dim: int

    def fit(self, image_paths: list[Path]) -> None:
        """Optional fitting step (e.g. learning a VLAD codebook)."""
        ...

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Embed one RGB uint8 image. Returns shape (dim,), L2-normalized."""
        ...

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Embed a batch. Returns shape (N, dim), each row L2-normalized."""
        ...


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)
