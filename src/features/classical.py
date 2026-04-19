"""Classical local-feature track (SIFT/ORB + VLAD).

OWNER: Person 2.

Responsibilities for the owner:
  - Implement ``SiftVladEmbedder`` (and optionally an ``OrbVladEmbedder``)
    satisfying the ``Embedder`` protocol in ``src/features/base.py``.
  - Learn a KMeans codebook in ``fit()`` from a subset of training images.
  - In ``embed()`` extract SIFT descriptors, accumulate residuals per
    visual word, flatten to a single vector, then power-law + L2 normalize.
  - Typical settings: k ∈ {64, 128}, PCA-whitening to ~128–256 dims.

Reference:
  - Jégou et al., "Aggregating Local Descriptors into a Compact Image
    Representation" (VLAD), 2010.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.base import Embedder, l2_normalize  # noqa: F401


class SiftVladEmbedder:
    name = "sift_vlad"

    def __init__(self, n_clusters: int = 64, pca_dim: int | None = 128):
        self.n_clusters = n_clusters
        self.pca_dim = pca_dim
        self.dim = pca_dim if pca_dim is not None else 128 * n_clusters
        # TODO(person-2): codebook (KMeans), optional PCA matrix

    def fit(self, image_paths: list[Path]) -> None:
        raise NotImplementedError("TODO(person-2): train KMeans codebook on SIFT descriptors")

    def embed(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO(person-2): SIFT + VLAD aggregation")

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.embed(img) for img in images], axis=0)
