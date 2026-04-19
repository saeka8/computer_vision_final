# IE Tower Visual Place Recognition

Fast image-retrieval system for place recognition in the IE Tower area.
Given a query photo, returns the top-K most similar images from a gallery
and predicts the location.

Two retrieval tracks share one data loader, index, and evaluation harness:

- **Classical** — SIFT/ORB local features aggregated with VLAD.
- **Deep** — DINOv2 ViT-S/14 global embeddings (ResNet50 fallback).

Both produce L2-normalized vectors that plug into the same FAISS index.

## Repo layout

```
src/            # library code (data, features, index, retrieve, evaluate, app)
scripts/        # CLI entry points (prepare_data, build_index, run_eval)
data/           # captured photos, one folder per location
results/        # built indices + eval JSONs (gitignored)
tests/          # pytest suite
report/         # final write-up + figures
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
# 1. Normalize data and write the manifest.
python scripts/prepare_data.py

# 2. Build both indices.
python scripts/build_index.py --method deep
python scripts/build_index.py --method classical

# 3. Evaluate on the held-out query split.
python scripts/run_eval.py --method deep
python scripts/run_eval.py --method classical

# 4. Launch the demo UI.
streamlit run src/app.py
```

## Ownership

| Person | Files |
|---|---|
| 1 — Data / infra | `src/data.py`, `scripts/prepare_data.py`, `requirements.txt` |
| 2 — Classical track | `src/features/classical.py` |
| 3 — Deep track | `src/features/deep.py` |
| 4 — Index + evaluation | `src/index.py`, `src/retrieve.py`, `src/evaluate.py`, `scripts/build_index.py`, `scripts/run_eval.py` |
| 5 — Demo + report | `src/app.py`, `report/` |

See `/Users/ryanmuenker/.claude/plans/soft-wobbling-cascade.md` for the full plan.
