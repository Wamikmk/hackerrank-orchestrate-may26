"""
pipeline.py — Wired triage pipeline orchestrator.

Stage order per row:
  normalize → rules → classify → retrieve → decide → generate → infer_product_area

Module 8 (product_area lookup table) is still a stub — returns the raw
corpus directory name from the top-1 chunk. Module 8 replaces infer_product_area.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import TicketInput
import rules as _rules_mod
import classifier as _classifier_mod
from retriever import retrieve as _retrieve_chunks
import decide as _decide_mod
import generator as _generator_mod
from product_area import map_product_area as _map_product_area

_KNOWN_COMPANIES = {"hackerrank", "claude", "visa"}


# ── Stage 1: Normalize ────────────────────────────────────────────────────────

def normalize(row: TicketInput) -> str:
    parts = []
    if row.subject.strip():
        parts.append(row.subject.strip())
    if row.issue.strip():
        parts.append(row.issue.strip())
    return " ".join(parts)


# ── Stage 3: Retrieve ─────────────────────────────────────────────────────────

def retrieve(text: str, company: str) -> list[dict]:
    company_filter = company if company.lower() in _KNOWN_COMPANIES else None
    return _retrieve_chunks(text, company=company_filter, k=5)


# ── Stage 6: Infer product area (STUB — Module 8 replaces) ───────────────────

def infer_product_area(chunks: list[dict], status: str) -> str:
    """Temporary stub: raw directory name from top-1 chunk. Blank when Escalated."""
    if status == "Escalated" or not chunks:
        return ""
    return chunks[0].get("product_area", "")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def process(row: TicketInput) -> dict:
    """
    Runs all pipeline stages for one ticket. Returns a dict with:
      status, request_type, response, product_area, justification,
      reason_code, response_mode, force_escalate, top_sim, cost_usd
    """
    text = normalize(row)

    # ── Rules (no LLM) ───────────────────────────────────────────────────────
    rules_result = _rules_mod.check_rules(row.issue, row.subject)

    # ── Classify (Haiku) ─────────────────────────────────────────────────────
    classify_result = _classifier_mod.classify(
        issue=row.issue, subject=row.subject, company=row.company
    )
    classify_cost = classify_result.get("_usage", {}).get("cost_usd", 0.0)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    chunks = retrieve(text, row.company)
    max_sim = chunks[0]["score"] if chunks else 0.0

    # ── Decide (pure logic) ───────────────────────────────────────────────────
    decision = _decide_mod.decide(
        rules_result=rules_result,
        classifier_result=classify_result,
        retrieval_max_sim=max_sim,
        company=row.company,
    )
    status        = decision["status"]
    reason_code   = decision["reason_code"]
    response_mode = decision["response_mode"]
    just_seed     = decision["justification_seed"]

    # ── Generate (Sonnet for grounded; static otherwise) ─────────────────────
    gen_result = _generator_mod.generate(
        issue=row.issue,
        subject=row.subject,
        company=row.company,
        response_mode=response_mode,
        retrieved_chunks=chunks,
        justification_seed=just_seed,
    )
    gen_cost = gen_result.get("_usage", {}).get("cost_usd", 0.0)

    # ── Override: INSUFFICIENT_CONTEXT → escalate ────────────────────────────
    if gen_result.get("force_escalate"):
        status        = "Escalated"
        reason_code   = "low_grounding"
        response_mode = "escalation"
        gen_result["response"] = "Escalate to a human"

    # ── Product area (Module 8) ───────────────────────────────────────────────
    top_chunk    = chunks[0] if chunks else {}
    product_area = _map_product_area(
        top_chunk_dir=top_chunk.get("product_area", ""),
        file_path=top_chunk.get("file_path", ""),
        request_type=classify_result.get("request_type", "invalid"),
        status=status,
    )

    return {
        "status":        status,
        "request_type":  classify_result.get("request_type", "invalid"),
        "response":      gen_result["response"],
        "product_area":  product_area,
        "justification": gen_result["justification"],
        "reason_code":   reason_code,
        "response_mode": response_mode,
        "force_escalate": gen_result.get("force_escalate", False),
        "top_sim":       max_sim,
        "cost_usd":      classify_cost + gen_cost,
        "chunks":        chunks,   # kept for evaluation grounding check
    }
