"""Streamlit demo: upload a photo, see the top-K retrieved locations.

OWNER: Person 5.

    streamlit run src/app.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.index import RetrievalIndex  # noqa: E402
from src.retrieve import build_method  # noqa: E402

st.set_page_config(page_title="IE Tower Place Recognition", layout="wide")
st.title("IE Tower — Visual Place Recognition")

method = st.sidebar.selectbox("Method", ["deep", "classical"])
k = st.sidebar.slider("Top-K", 1, 10, 5)

index_path = REPO_ROOT / "results" / method
if not index_path.with_suffix(".faiss").exists():
    st.warning(f"No index at {index_path}.faiss — run build_index.py first.")
    st.stop()

index = RetrievalIndex(dim=0)
index.load(index_path)

@st.cache_resource
def get_embedder(name: str):
    e = build_method(name)
    # classical needs a codebook — load from gallery
    from src.data import by_split, load_manifest

    e.fit([s.path for s in by_split(load_manifest(), "gallery")])
    return e

embedder = get_embedder(method)

upload = st.file_uploader("Upload a query photo", type=["jpg", "jpeg", "png"])
if upload is not None:
    img = Image.open(upload).convert("RGB")
    st.image(img, caption="Query", width=400)
    vec = embedder.embed(np.asarray(img))
    hits = index.search(vec, k=k)

    st.subheader("Top-K matches")
    cols = st.columns(min(k, 5))
    for col, hit in zip(cols, hits):
        with col:
            try:
                st.image(hit.path, caption=f"{hit.label}\nscore={hit.score:.3f}")
            except Exception:  # noqa: BLE001
                st.write(f"{hit.label} ({hit.score:.3f})")
