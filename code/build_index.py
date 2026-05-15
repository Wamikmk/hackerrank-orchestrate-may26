"""
build_index.py — Embed chunks from corpus_loader and save to code/index/.

Outputs:
  code/index/embeddings.npy   — float32 array, shape (N, 384)
  code/index/metadata.jsonl   — one JSON object per line, matches row order

Usage:
  python code/build_index.py [--data-dir data/] [--index-dir code/index/]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# Add code/ to path so corpus_loader imports cleanly regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_loader import load_chunks, Chunk

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build embedding index for support corpus")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    p.add_argument("--index-dir", default=str(REPO_ROOT / "code" / "index"))
    return p.parse_args()


def embed_chunks(chunks: list[Chunk], model) -> np.ndarray:
    """Embed all chunks; returns float32 ndarray (N, dim)."""
    texts = [c.text for c in chunks]
    print(f"  Embedding {len(texts)} chunks...", flush=True)
    vectors = list(model.embed(texts))
    return np.array(vectors, dtype=np.float32)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and chunk corpus ──────────────────────────────────────────────
    print(f"\nLoading corpus from {data_dir} ...")
    t0 = time.time()
    chunks = load_chunks(data_dir)
    print(f"Loaded {len(chunks)} chunks in {time.time() - t0:.1f}s")

    # ── 2. Per-company summary ────────────────────────────────────────────────
    from collections import Counter
    company_counts = Counter(c.company for c in chunks)
    print("\nChunks per company:")
    for company, count in sorted(company_counts.items()):
        print(f"  {company:20s}: {count}")
    print(f"  {'TOTAL':20s}: {len(chunks)}")

    # ── 3. Load embedding model ───────────────────────────────────────────────
    print("\nLoading fastembed model (downloads ONNX weights on first run)...")
    from fastembed import TextEmbedding
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    print("Model ready (all-MiniLM-L6-v2, 384-dim, ONNX runtime)")

    # ── 4. Embed ──────────────────────────────────────────────────────────────
    print("\nEmbedding chunks...")
    t1 = time.time()
    embeddings = embed_chunks(chunks, model)
    print(f"Embedding complete in {time.time() - t1:.1f}s  shape={embeddings.shape}")

    # ── 5. Normalise (for cosine similarity via dot product) ──────────────────
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    # ── 6. Save ───────────────────────────────────────────────────────────────
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "metadata.jsonl"

    np.save(emb_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    print(f"\nSaved: {emb_path}  ({emb_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved: {meta_path}  ({meta_path.stat().st_size / 1e3:.1f} KB)")

    # ── 7. Sanity-check: print 3 random chunks ────────────────────────────────
    rng = random.Random(42)
    sample = rng.sample(chunks, min(3, len(chunks)))
    print("\n" + "=" * 70)
    print("RANDOM CHUNK SAMPLES (seed=42)")
    print("=" * 70)
    for i, c in enumerate(sample, 1):
        print(f"\n── Chunk {i} ──────────────────────────────────────────────────────")
        print(f"  company      : {c.company}")
        print(f"  product_area : {c.product_area}")
        print(f"  file_path    : {c.file_path}")
        print(f"  source_url   : {c.source_url}")
        print(f"  title        : {c.title}")
        print(f"  chunk_id     : {c.chunk_id}")
        print(f"  word_count   : {len(c.text.split())}")
        print(f"  text preview :")
        preview = c.text[:600] + ("..." if len(c.text) > 600 else "")
        for line in preview.split("\n"):
            print(f"    {line}")
    print("\n" + "=" * 70)
    print("Index build complete. Run python code/build_index.py to rebuild.")


if __name__ == "__main__":
    main()
