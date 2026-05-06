# IE Tower Visual Place Recognition

Fast image-retrieval system for place recognition in the IE Tower area.
Given a query photo, returns the top-K most similar images from a gallery
and predicts the location.

Three retrieval tracks share one data loader, index, and evaluation harness:

- **Classical** — SIFT/ORB local features aggregated with VLAD.
- **Deep** — DINOv2 ViT-S/14 global embeddings (ResNet50 fallback).
- **CNN baseline** — small supervised CNN trained on gallery labels, then
  reused as an embedding extractor for retrieval.

All three produce L2-normalized vectors that plug into the same FAISS index.

## Repo layout

```
src/            # library code (data, features, index, retrieve, evaluate, app)
scripts/        # CLI entry points (prepare_data, build_index, run_eval)
data/           # training / gallery photos, one folder per location
test/           # held-out query photos, one folder per location
results/        # built indices + eval JSONs (gitignored)
tests/          # pytest suite
report/         # final write-up + figures
```

## Image data access

Image folders (`data/` and `test/`) are gitignored. Download them from Dropbox:

- Gallery / training images (`data/`): https://www.dropbox.com/scl/fo/bc68mnw9oyms5iq9ognig/ABuEVZpKPCAByLYPezAbAlU?rlkey=rjqw4izex08cjbkqj3tzs2l73&st=f4wxtvir&dl=0
- Test / query images (`test/`): https://www.dropbox.com/scl/fo/5e7vdwn8wyfqnfwn0tv5r/AOdUS4TgyuWhGJSySAKc89o?rlkey=tjqslr9xh6fup4sbgntu3s510&dl=0

## Demo video access
https://drive.google.com/file/d/1Q5s8w_TstVHfq4S6FNW8oj2HWxPAhA-M/view?usp=sharing

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
# 1. Normalize data and write the manifest for the full training set.
python scripts/prepare_data.py

# 2. Build all indices.
python scripts/build_index.py --method deep
python scripts/build_index.py --method cnn
python scripts/build_index.py --method classical

# 3. Evaluate against the separate test folder.
python scripts/run_eval.py --method deep
python scripts/run_eval.py --method cnn
python scripts/run_eval.py --method classical

# 4. Launch the demo UI.
streamlit run src/app.py
```

## Ownership

| Person | Files |
|---|---|
| 1 — Data / infra | `src/data.py`, `scripts/prepare_data.py`, `requirements.txt` |
| 2 — Classical track | `src/features/classical.py` |
| 3 — Deep track | `src/features/deep.py`, `src/features/cnn.py` |
| 4 — Index + evaluation | `src/index.py`, `src/retrieve.py`, `src/evaluate.py`, `scripts/build_index.py`, `scripts/run_eval.py` |
| 5 — Demo + report | `src/app.py`, `report/` |

See `/Users/ryanmuenker/.claude/plans/soft-wobbling-cascade.md` for the full plan.
