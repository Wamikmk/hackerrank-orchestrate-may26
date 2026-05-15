"""
corpus_loader.py — Walk data/, parse markdown files, chunk into retrieval units.

Chunk strategy:
  - Strip YAML frontmatter and "Last updated / Last modified" lines.
  - Split body into paragraphs (double-newline boundaries).
  - Accumulate paragraphs up to CHUNK_WORDS words, then emit a chunk.
  - Carry ~OVERLAP_WORDS words of context into the next chunk.

Note on all-MiniLM-L6-v2: max sequence length is 256 tokens (~190 words).
CHUNK_WORDS=180 keeps us safely under that limit after tokenisation overhead.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

# Tuning knobs — adjust here only
CHUNK_WORDS = 180
OVERLAP_WORDS = 35
MIN_WORDS = 20  # drop chunks below this word count (stubs, escaped chars, orphan headings)

# Regex: YAML frontmatter block (greedy-safe, anchored to start of file)
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

# Regex: "Last updated / Last modified" italic line
_LAST_UPDATED_RE = re.compile(r"^_Last (?:updated|modified):.*?_\s*$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str          # "{article_slug}_{n}"
    company: str           # "hackerrank" | "claude" | "visa"
    product_area: str      # top-level subdir under data/{company}/
    file_path: str         # relative path from repo root
    source_url: str        # from frontmatter, or ""
    title: str             # from frontmatter, or ""
    text: str              # cleaned chunk text

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Return (metadata_dict, body_without_frontmatter)."""
    meta: dict = {}
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return meta, raw

    fm_block = m.group(0)
    body = raw[m.end():]

    # Pull known scalar fields with simple regex (avoids yaml dependency)
    for field in ("title", "source_url", "final_url", "article_slug", "title_slug"):
        pat = re.search(rf'^{field}:\s*["\']?(.*?)["\']?\s*$', fm_block, re.MULTILINE)
        if pat:
            meta[field] = pat.group(1).strip().strip('"\'')

    return meta, body


def _clean_body(body: str) -> str:
    """Strip noise lines, collapse whitespace."""
    body = _LAST_UPDATED_RE.sub("", body)
    # Collapse 3+ blank lines to 2
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """
    Split text into overlapping chunks by paragraph boundaries.
    Falls back to hard word-split for paragraphs longer than chunk_words.
    """
    paragraphs: list[str] = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0
    overlap_buf: str = ""  # carried-over tail from previous chunk

    def emit():
        nonlocal current_parts, current_words, overlap_buf
        if not current_parts:
            return
        chunk_text = "\n\n".join(current_parts)
        chunks.append(chunk_text)
        # Build overlap buffer: last ~overlap_words words of this chunk
        words = chunk_text.split()
        overlap_buf = " ".join(words[-overlap_words:]) if len(words) > overlap_words else chunk_text
        current_parts = []
        current_words = 0

    for para in paragraphs:
        para_words = _word_count(para)

        # Paragraph itself exceeds chunk_words — hard-split it
        if para_words > chunk_words:
            # Flush current buffer first
            if current_parts:
                emit()
            words = para.split()
            i = 0
            while i < len(words):
                slice_words = words[i: i + chunk_words]
                chunk_str = " ".join(slice_words)
                if overlap_buf:
                    chunk_str = overlap_buf + "\n\n" + chunk_str
                    overlap_buf = ""
                chunks.append(chunk_str)
                overlap_buf = " ".join(slice_words[-overlap_words:])
                i += chunk_words
            continue

        # Would overflow — emit what we have, start fresh with overlap
        if current_words + para_words > chunk_words and current_parts:
            emit()
            if overlap_buf:
                current_parts = [overlap_buf]
                current_words = _word_count(overlap_buf)
                overlap_buf = ""

        current_parts.append(para)
        current_words += para_words

    if current_parts:
        emit()

    return chunks if chunks else [text[:1000]]  # last-resort fallback


def _derive_product_area(rel_path: Path) -> str | None:
    """
    Given path relative to data/{company}/, return the top-level subdirectory
    as product_area. Returns None for files sitting directly in data/{company}/.
    """
    parts = rel_path.parts
    if len(parts) < 2:
        return None  # file is directly in data/{company}/ — skip
    return parts[0]  # first subdirectory = product_area


def load_chunks(data_dir: Path) -> list[Chunk]:
    """Walk data_dir, parse all .md files, return flat list of Chunks."""
    all_chunks: list[Chunk] = []

    for company_dir in sorted(data_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        company = company_dir.name  # "hackerrank" | "claude" | "visa"

        for md_file in sorted(company_dir.rglob("*.md")):
            rel_to_company = md_file.relative_to(company_dir)
            product_area = _derive_product_area(rel_to_company)
            if product_area is None:
                continue  # skip navigation index files

            rel_path = str(md_file.relative_to(data_dir.parent))

            raw = md_file.read_text(encoding="utf-8", errors="replace")
            meta, body = _parse_frontmatter(raw)
            body = _clean_body(body)

            if not body:
                continue

            source_url = meta.get("source_url") or meta.get("final_url") or ""
            title = meta.get("title") or ""
            article_slug = meta.get("article_slug") or md_file.stem

            text_chunks = _chunk_text(body, CHUNK_WORDS, OVERLAP_WORDS)
            for n, text in enumerate(text_chunks):
                if _word_count(text) < MIN_WORDS:
                    continue
                all_chunks.append(Chunk(
                    chunk_id=f"{article_slug}_{n}",
                    company=company,
                    product_area=product_area,
                    file_path=rel_path,
                    source_url=source_url,
                    title=title,
                    text=text,
                ))

    return all_chunks
