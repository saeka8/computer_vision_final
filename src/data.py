"""Dataset manifest, image loading, and train/gallery/query splits.

Conventions:
- Every image belongs to exactly one class (the folder name under ``data/``).
- The manifest CSV has columns: path, label, split.
- "gallery" images are indexed and searched against; "query" images are
  held out and used at eval time; "train" images may be used by methods
  that need fitting (e.g. VLAD's KMeans codebook).
- Classes with a single image go into the gallery only — they can be
  retrieved but are never used as queries. This is flagged in the report.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    split: str  # "train" | "gallery" | "query"


def discover_images(data_dir: Path = DATA_DIR) -> list[tuple[Path, str]]:
    """Walk ``data_dir`` and return (path, label) for every image file."""
    pairs: list[tuple[Path, str]] = []
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        for f in sorted(entry.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                pairs.append((f, entry.name))
    return pairs


def make_splits(
    pairs: list[tuple[Path, str]],
    query_frac: float = 0.25,
    seed: int = 42,
) -> list[Sample]:
    """Stratified split. 1-image classes go fully to gallery (no query).

    For classes with >=2 images we hold out ``max(1, round(n*query_frac))``
    images as queries; the rest go to the gallery. We do not carve out a
    dedicated "train" split by default — methods that need one can sample
    from the gallery.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[Path]] = defaultdict(list)
    for path, label in pairs:
        by_label[label].append(path)

    samples: list[Sample] = []
    for label, paths in by_label.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        if len(paths) < 2:
            samples.extend(Sample(p, label, "gallery") for p in paths)
            continue
        n_query = max(1, round(len(paths) * query_frac))
        n_query = min(n_query, len(paths) - 1)  # leave at least 1 in gallery
        for p in paths[:n_query]:
            samples.append(Sample(p, label, "query"))
        for p in paths[n_query:]:
            samples.append(Sample(p, label, "gallery"))
    return samples


def write_manifest(samples: list[Sample], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for s in samples:
            w.writerow([str(s.path.relative_to(REPO_ROOT)), s.label, s.split])


def load_manifest(path: Path = MANIFEST_PATH) -> list[Sample]:
    samples: list[Sample] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            samples.append(
                Sample(
                    path=REPO_ROOT / row["path"],
                    label=row["label"],
                    split=row["split"],
                )
            )
    return samples


def by_split(samples: list[Sample], split: str) -> list[Sample]:
    return [s for s in samples if s.split == split]


def load_image(path: Path | str, max_side: int | None = 1024) -> np.ndarray:
    """Return an RGB uint8 image array. Optionally bounds the long side."""
    img = Image.open(path).convert("RGB")
    if max_side is not None and max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.BILINEAR)
    return np.asarray(img)
