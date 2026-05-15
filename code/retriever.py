"""
retriever.py — Cosine-similarity retrieval over pre-built embedding index.

Embeddings were L2-normalised at index time, so cosine similarity = np.dot.
The fastembed model is loaded once at module level; masks are cached per company.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Resolve index paths relative to this file ─────────────────────────────────
_CODE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODE_DIR.parent
_INDEX_DIR = _CODE_DIR / "index"
_EMB_PATH = _INDEX_DIR / "embeddings.npy"
_META_PATH = _INDEX_DIR / "metadata.jsonl"

# ── Load index ONCE ───────────────────────────────────────────────────────────
print("retriever: loading embeddings...", file=sys.stderr, flush=True)
_embeddings: np.ndarray = np.load(_EMB_PATH)  # shape (N, 384), already L2-normed

print("retriever: loading metadata...", file=sys.stderr, flush=True)
with _META_PATH.open(encoding="utf-8") as _f:
    _metadata: list[dict] = [json.loads(line) for line in _f]

assert len(_embeddings) == len(_metadata), "embeddings / metadata length mismatch"
_N = len(_embeddings)

# ── Load embedding model ONCE ─────────────────────────────────────────────────
print("retriever: loading fastembed model...", file=sys.stderr, flush=True)
from fastembed import TextEmbedding
_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
print("retriever: ready.", file=sys.stderr, flush=True)

# ── Company values in the corpus (lower-cased for matching) ───────────────────
_KNOWN_COMPANIES = {"hackerrank", "claude", "visa"}

# ── Boolean mask cache: company_lower -> np.ndarray[bool] ─────────────────────
_mask_cache: dict[str, np.ndarray] = {}


def _company_mask(company: str) -> np.ndarray:
    key = company.lower()
    if key not in _mask_cache:
        _mask_cache[key] = np.array(
            [m["company"].lower() == key for m in _metadata], dtype=bool
        )
    return _mask_cache[key]


def _embed_query(query: str) -> np.ndarray:
    """Embed a single query string; returns L2-normalised float32 vector."""
    # fastembed ≥0.3 supports query_prefix kwarg; try it, fall back silently
    try:
        vecs = list(_model.query_embed(query))
    except AttributeError:
        vecs = list(_model.embed([query]))
    vec = np.array(vecs[0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def retrieve(
    query: str,
    company: Optional[str] = None,
    k: int = 5,
) -> list[dict]:
    """
    Returns top-k chunks ranked by cosine similarity.
    Each dict: text, company, product_area, file_path, chunk_id, score.
    If company is provided, filters to that company before scoring.
    If company is None, scores across all chunks.
    """
    q_vec = _embed_query(query)  # shape (384,)

    if company is not None:
        mask = _company_mask(company)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        scores = _embeddings[indices] @ q_vec  # (M,)
        top_local = np.argsort(scores)[::-1][: k]
        top_indices = indices[top_local]
        top_scores = scores[top_local]
    else:
        scores = _embeddings @ q_vec  # (N,)
        top_local = np.argsort(scores)[::-1][: k]
        top_indices = top_local
        top_scores = scores[top_local]

    results = []
    for idx, score in zip(top_indices, top_scores):
        m = _metadata[int(idx)]
        results.append(
            {
                "text": m["text"],
                "company": m["company"],
                "product_area": m["product_area"],
                "file_path": m["file_path"],
                "chunk_id": m["chunk_id"],
                "score": float(score),
            }
        )
    return results
