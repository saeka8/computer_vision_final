"""One-shot data normalization + manifest generation.

What it does:
  1. Renames ``data/outside _caleido`` to ``data/outside_caleido`` if present.
  2. Deletes any 0-byte placeholder files at the top level of ``data/``
     (remnants of drag-and-drop for locations that were never photographed).
  3. Walks ``data/`` to discover every image.
  4. Stratified split → writes ``data/manifest.csv``.
  5. Prints per-class counts and flags data-quality issues.

Run from the repo root:

    python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    DATA_DIR,
    MANIFEST_PATH,
    discover_images,
    make_splits,
    write_manifest,
)

KNOWN_STUBS = {
    "bathroom_pictures",
    "cafeteria_photos",
    "elevator_pictures",
    "gym_photos",
    "outside_fifth_floor",
    "robotics_photos",
}


def normalize_folder_names() -> list[str]:
    """Rename folders with awkward names. Returns list of actions taken."""
    actions = []
    spaced = DATA_DIR / "outside _caleido"
    target = DATA_DIR / "outside_caleido"
    if spaced.exists() and not target.exists():
        spaced.rename(target)
        actions.append(f"renamed {spaced.name!r} -> {target.name!r}")
    return actions


def remove_zero_byte_stubs() -> list[str]:
    """Delete 0-byte placeholder files at the top level of data/."""
    actions = []
    for entry in sorted(DATA_DIR.iterdir()):
        if entry.is_file() and entry.stat().st_size == 0:
            actions.append(f"removed 0-byte stub {entry.name!r}")
            entry.unlink()
    return actions


def main() -> int:
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist", file=sys.stderr)
        return 1

    print(f"== Preparing {DATA_DIR} ==")

    for a in normalize_folder_names():
        print(f"  {a}")
    for a in remove_zero_byte_stubs():
        print(f"  {a}")

    pairs = discover_images(DATA_DIR)
    if not pairs:
        print("ERROR: no images found", file=sys.stderr)
        return 1

    samples = make_splits(pairs)
    write_manifest(samples)

    counts = Counter(s.label for s in samples)
    gallery = Counter(s.label for s in samples if s.split == "gallery")
    queries = Counter(s.label for s in samples if s.split == "query")

    print()
    print(f"Manifest written: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"Total images: {len(samples)} across {len(counts)} classes")
    print()
    print(f"{'class':<30} {'total':>6} {'gallery':>8} {'query':>6}")
    print("-" * 54)
    for label in sorted(counts):
        print(
            f"{label:<30} {counts[label]:>6} {gallery[label]:>8} {queries[label]:>6}"
        )

    print()
    issues: list[str] = []
    missing_stubs = KNOWN_STUBS - set(counts)
    if missing_stubs:
        issues.append(
            f"no photos yet for: {sorted(missing_stubs)} — person 1 to follow up"
        )
    single_image_classes = [c for c, n in counts.items() if n < 2]
    if single_image_classes:
        issues.append(
            f"{len(single_image_classes)} classes have <2 photos "
            f"(gallery-only, never queried): {single_image_classes}"
        )
    thin_classes = [c for c, n in counts.items() if 2 <= n < 4]
    if thin_classes:
        issues.append(
            f"{len(thin_classes)} classes have 2–3 photos (need more intra-class "
            f"variation): {thin_classes}"
        )

    if issues:
        print("Issues to address:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("No data-quality issues detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
