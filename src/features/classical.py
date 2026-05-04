"""Classical local-feature track: SIFT descriptors aggregated with VLAD.

Reference:
  - Jégou et al., "Aggregating Local Descriptors into a Compact Image
    Representation" (VLAD), 2010.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from src.features.base import Embedder, l2_normalize  # noqa: F401
from src.data import load_image

_SIFT_DIM = 128  # fixed by the SIFT descriptor


def _extract_sift(image: np.ndarray, max_kp: int = 500) -> np.ndarray | None:
    """Return (N, 128) SIFT descriptors or None if none found."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_kp)
    _, descs = sift.detectAndCompute(gray, None)
    return descs  # None when no keypoints detected


def _vlad(descs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Aggregate (N, 128) descriptors into a single VLAD vector."""
    k = centers.shape[0]
    # assign each descriptor to nearest cluster
    diffs = descs[:, None, :] - centers[None, :, :]   # (N, k, 128)
    assignments = np.argmin(np.linalg.norm(diffs, axis=-1), axis=1)  # (N,)
    vlad = np.zeros((k, _SIFT_DIM), dtype=np.float32)
    for c in range(k):
        mask = assignments == c
        if mask.any():
            vlad[c] = descs[mask].sum(axis=0) - centers[c] * mask.sum()
    # power-law (intra-normalization) then L2
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    return l2_normalize(vlad.flatten())


class SiftVladEmbedder:
    name = "sift_vlad"

    def __init__(self, n_clusters: int = 64, pca_dim: int | None = 128):
        self.n_clusters = n_clusters
        self.pca_dim = pca_dim
        self.dim = pca_dim if pca_dim is not None else _SIFT_DIM * n_clusters
        self._centers: np.ndarray | None = None
        self._pca: PCA | None = None

    def fit(self, image_paths: list[Path]) -> None:
        all_descs: list[np.ndarray] = []
        for p in image_paths:
            img = load_image(p)
            d = _extract_sift(img)
            if d is not None:
                all_descs.append(d)

        descs = np.concatenate(all_descs, axis=0).astype(np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=4096,
            n_init=3,
        )
        kmeans.fit(descs)
        self._centers = kmeans.cluster_centers_.astype(np.float32)

        if self.pca_dim is not None:
            vlads = []
            for p in image_paths:
                d = _extract_sift(load_image(p))
                if d is None:
                    d = np.zeros((1, _SIFT_DIM), dtype=np.float32)
                vlads.append(_vlad(d.astype(np.float32), self._centers))
            vlads = np.stack(vlads)
            n_components = min(self.pca_dim, vlads.shape[0], vlads.shape[1])
            self._pca = PCA(n_components=n_components, whiten=True)
            self._pca.fit(vlads)
            self.dim = n_components

    def embed(self, image: np.ndarray) -> np.ndarray:
        if self._centers is None:
            raise RuntimeError("call fit() before embed()")
        descs = _extract_sift(image)
        if descs is None:
            descs = np.zeros((1, _SIFT_DIM), dtype=np.float32)
        vec = _vlad(descs.astype(np.float32), self._centers)
        if self._pca is not None:
            vec = l2_normalize(self._pca.transform(vec[None])[0])
        return vec

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.embed(img) for img in images], axis=0)
