# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

---

## Commands

**Setup (one-time):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

**Build the embedding index (one-time, ~15 min, requires data/ corpus):**
```bash
python code/build_index.py
# Outputs: code/index/embeddings.npy  and  code/index/metadata.jsonl
```

**Run the agent on the full ticket set:**
```bash
python code/main.py
# Override defaults:
python code/main.py --input support_tickets/support_tickets.csv \
                    --output support_tickets/output.csv \
                    --debug --limit 10
```

**Evaluate against labeled sample tickets:**
```bash
python code/evaluation.py
# Reads: support_tickets/sample_support_tickets.csv (has expected columns)
# Prints: accuracy table (Status / Request Type / Product Area / Grounding) + per-row diff
```

**Run individual unit tests:**
```bash
python -m pytest code/test_rules.py -v
python -m pytest code/test_classifier.py -v
python -m pytest code/test_decide.py -v
python -m pytest code/test_retriever.py -v
python -m pytest code/test_generator.py -v
```

**Run all tests:**
```bash
python -m pytest code/ -v
```

---

## Architecture

The pipeline is a linear, 6-stage per-ticket flow orchestrated by `code/pipeline.py:process()`:

```
TicketInput (issue, subject, company)
     │
     ├─ [1] rules.py:check_rules()          — regex safety net, no LLM
     ├─ [2] classifier.py:classify()         — Claude Haiku 4.5, strict JSON
     ├─ [3] retriever.py:retrieve()          — MiniLM-L6-v2 via fastembed, numpy cosine
     ├─ [4] decide.py:decide()               — pure logic, no LLM, first-match wins
     ├─ [5] generator.py:generate()          — Claude Sonnet 4.5 (grounded) or static canned response
     └─ [6] product_area.py:map_product_area() — deterministic dir→vocabulary mapping
```

**Key design decisions:**

- **Two-model split**: Haiku for cheap classification, Sonnet only for grounded generation. Canned strings short-circuit Sonnet for escalation/polite/out-of-scope modes.
- **Retrieval index**: Pre-built offline (`build_index.py`). Vectors are L2-normalised at build time so retrieval is a plain `np.dot`. No vector DB — numpy is faster end-to-end at this corpus size (~4026 chunks).
- **Decision logic** (`decide.py`) has a strict priority order: rules → hard sensitivity ceiling (>0.92) → soft sensitivity+grounding gate → invalid routing → grounding threshold. First match wins.
- **Generator force-escalate**: If the generator returns `INSUFFICIENT_CONTEXT`, `pipeline.py` overrides the decision to Escalated/low_grounding.
- **Anti-injection**: Ticket fields are wrapped in XML tags in all prompts; system messages declare them as data.
- **Output CSV columns**: `Issue, Subject, Company, Response, Product Area, Status, Request Type` — Title Case, written incrementally (flush after each row).

**Module map:**

| File | Responsibility |
|---|---|
| `main.py` | CLI entry point, CSV I/O, progress printing |
| `pipeline.py` | Wires all stages, owns the `process()` function |
| `schemas.py` | `TicketInput` and `TriageOutput` Pydantic models |
| `rules.py` | Compiled regex patterns, `check_rules()` |
| `classifier.py` | Haiku API call with tenacity retry, JSON parse + Pydantic validation |
| `retriever.py` | Loads index at import time, `retrieve()` with optional company filter |
| `decide.py` | Pure routing logic, no I/O |
| `generator.py` | Sonnet API call; static strings for non-grounded modes |
| `product_area.py` | Maps top-chunk corpus directory to output vocabulary |
| `corpus_loader.py` | Walks `data/` to yield `Chunk` objects for indexing |
| `build_index.py` | One-shot script: load corpus → embed → L2-normalise → save |
| `evaluation.py` | Dev harness: runs pipeline against sample CSV with expected outputs |

**Data layout:**
```
data/
  hackerrank/   — HackerRank documentation chunks
  claude/       — Claude/Anthropic documentation chunks
  visa/         — Visa documentation chunks
code/index/
  embeddings.npy     — float32 (N, 384), L2-normalised
  metadata.jsonl     — one JSON object per chunk (text, company, product_area, file_path, chunk_id)
support_tickets/
  support_tickets.csv        — evaluation input
  sample_support_tickets.csv — labeled sample (has Status/Request Type/Product Area/Response columns)
  output.csv                 — agent output
```
