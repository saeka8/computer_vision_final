"""End-to-end query: load an image → embed → search the index → return hits.

Thin wrapper — useful for both the Streamlit app and the eval script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import load_image
from src.features.base import Embedder
from src.index import Hit, RetrievalIndex


def query_from_path(
    image_path: Path | str,
    embedder: Embedder,
    index: RetrievalIndex,
    k: int = 5,
) -> list[Hit]:
    img = load_image(image_path)
    vec = embedder.embed(img)
    return index.search(vec, k=k)


def embed_paths(
    paths: list[Path], embedder: Embedder, batch_size: int = 16
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = [load_image(p) for p in paths[i : i + batch_size]]
        vectors.append(embedder.embed_batch(batch))
    return np.concatenate(vectors, axis=0)


def build_method(method: str) -> Embedder:
    """Resolve a CLI-level method name to an Embedder instance."""
    if method == "deep":
        from src.features.deep import DinoV2Embedder

        return DinoV2Embedder()
    if method == "classical":
        from src.features.classical import SiftVladEmbedder

        return SiftVladEmbedder()
    raise ValueError(f"unknown method: {method!r}")
