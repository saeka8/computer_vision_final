"""Dataset helpers for training data and held-out test images.

Conventions:
- Every image belongs to exactly one class (the folder name under ``data/``).
- Held-out evaluation images live under ``test/`` with the same folder naming.
- The manifest CSV has columns: path, label, split.
- The current workflow uses every image in ``data/`` as ``gallery`` so
  indexing and fitting can use the full training set.
"""

from __future__ import annotations

import csv
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
TEST_DIR = REPO_ROOT / "test"
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


def make_gallery_samples(pairs: list[tuple[Path, str]]) -> list[Sample]:
    """Assign every discovered training image to the gallery split."""
    return [Sample(path=path, label=label, split="gallery") for path, label in pairs]


def write_manifest(samples: list[Sample], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for s in samples:
            w.writerow([str(s.path.relative_to(REPO_ROOT)), s.label, s.split])


def load_manifest(path: Path = MANIFEST_PATH) -> list[Sample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Create the dataset under "
            f"{DATA_DIR}/<place_name>/... and run `python scripts/prepare_data.py`."
        )
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
