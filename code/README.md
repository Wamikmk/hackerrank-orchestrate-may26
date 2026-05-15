# Multi-Domain Support Triage Agent

Terminal-based agent that triages support tickets across HackerRank, Claude, and Visa ecosystems using a local-first RAG pipeline.

## Architecture

5-stage pipeline per ticket:

1. **Rules layer** (`rules.py`) — regex-based safety net for fraud, identity, billing, legal, production-bug, self-harm, and prompt-injection patterns. Forces escalation on match.
2. **Classifier** (`classifier.py`) — Claude Haiku 4.5 with strict JSON output. Returns `request_type`, `sensitivity` (0-1), `inferred_company`, `is_polite_chatter`.
3. **Retriever** (`retriever.py`) — MiniLM-L6-v2 embeddings via fastembed (no torch dep), cosine similarity over 4026 corpus chunks, optional company filter.
4. **Decision** (`decide.py`) — combines rules, sensitivity, and retrieval similarity into Replied/Escalated routing with reason codes.
5. **Generator** (`generator.py`) — Claude Sonnet 4.5 with strict grounding constraints. Short-circuits to canned responses for escalation/polite/out-of-scope modes.

Product Area is derived deterministically (`product_area.py`) from the top-1 retrieved chunk's directory, mapped to the controlled vocabulary observed in sample data.

## Install

Requires Python 3.10+ and ~200MB disk for ONNX embedding model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

Environment variables (in `.env`):

## Build the index (first run only, ~15 min)

```bash
python code/build_index.py
```

Produces `code/index/embeddings.npy` and `code/index/metadata.jsonl`.

## Run the agent

```bash
python code/main.py
```

Default I/O paths can be overridden:

```bash
python code/main.py \
    --input support_tickets/support_tickets.csv \
    --output support_tickets/output.csv \
    --data-dir data/ \
    --debug
```

## Output schema

CSV with Title-Case columns: `Issue, Subject, Company, Response, Product Area, Status, Request Type`.

- `Status` ∈ {Replied, Escalated}
- `Request Type` ∈ {product_issue, feature_request, bug, invalid}
- `Product Area` is empty when Status is Escalated or Request Type is invalid.

## Design choices

- **Local embeddings via fastembed** — chosen over sentence-transformers to avoid 800MB torch dep on 4GB-RAM machines.
- **Two-LLM split** — Haiku for cheap enum classification, Sonnet only for grounded generation. Short-circuits on canned responses.
- **Three-layer escalation** — deterministic rules > sensitivity ceiling > soft sensitivity gate (only escalates sensitive topics that lack documentation grounding).
- **No vector DB** — numpy cosine over normalized vectors is faster end-to-end at this corpus size.
- **Anti-injection** — XML delimiters around ticket fields, system-prompt-level instructions to treat ticket text as data not commands.
