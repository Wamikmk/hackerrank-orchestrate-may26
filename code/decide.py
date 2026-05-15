"""
decide.py — Pure decision logic. No LLM calls, no I/O.

Combines rules result + classifier result + retrieval similarity
into a routing decision. First match wins (priority order).

Decision tree (7 steps):
  1. Rules triggered                            → Escalated, rules:<category>
  2. sensitivity > 0.92                         → Escalated, sensitivity_hard   (hard ceiling)
  3. sensitivity > 0.7  AND  max_sim < 0.50     → Escalated, sensitivity_unsupported
  4. request_type == invalid  AND  polite        → Replied,   polite_chatter
  5. request_type == invalid                     → Replied,   off_topic
  6. max_sim < 0.30                              → Escalated, low_grounding
  7. else                                        → Replied,   grounded
"""


def decide(
    rules_result: tuple,        # (should_escalate: bool, reason: str|None, category: str|None)
    classifier_result: dict,    # from classifier.classify()
    retrieval_max_sim: float,   # top-1 cosine similarity score
    company: str,
) -> dict:
    """
    Returns:
      {
        "status": "Replied" | "Escalated",
        "reason_code": str,
        "response_mode": str,     # escalation | polite | out_of_scope | grounded
        "justification_seed": str,
      }
    """
    should_escalate, _reason, category = rules_result
    request_type = classifier_result.get("request_type", "invalid")
    sensitivity  = float(classifier_result.get("sensitivity", 0.0))
    is_polite    = bool(classifier_result.get("is_polite_chatter", False))

    # ── 1. Hard rules (fraud, injection, self-harm, etc.) ────────────────────
    if should_escalate:
        cat = category or "unknown"
        return {
            "status": "Escalated",
            "reason_code": f"rules:{cat}",
            "response_mode": "escalation",
            "justification_seed": (
                f"Escalated due to rule match ({cat}); requires human review."
            ),
        }

    # ── 2. Hard sensitivity ceiling (> 0.92, even when docs exist) ───────────
    if sensitivity > 0.92:
        return {
            "status": "Escalated",
            "reason_code": "sensitivity_hard",
            "response_mode": "escalation",
            "justification_seed": (
                f"Classifier flagged very high sensitivity ({sensitivity:.2f}); "
                "routed to human for safety."
            ),
        }

    # ── 3. Elevated sensitivity without strong doc grounding ──────────────────
    if sensitivity > 0.7 and retrieval_max_sim < 0.50:
        return {
            "status": "Escalated",
            "reason_code": "sensitivity_unsupported",
            "response_mode": "escalation",
            "justification_seed": (
                f"Sensitive topic with insufficient documentation grounding "
                f"(sim={retrieval_max_sim:.2f}); routed to human."
            ),
        }

    # ── 4. Invalid + polite chatter ───────────────────────────────────────────
    if request_type == "invalid" and is_polite:
        return {
            "status": "Replied",
            "reason_code": "polite_chatter",
            "response_mode": "polite",
            "justification_seed": "Polite/greeting message; replied with brief acknowledgment.",
        }

    # ── 5. Invalid + off-topic (not polite) ──────────────────────────────────
    if request_type == "invalid":
        return {
            "status": "Replied",
            "reason_code": "off_topic",
            "response_mode": "out_of_scope",
            "justification_seed": "Question is outside the scope of the support corpus.",
        }

    # ── 6. Insufficient retrieval grounding ──────────────────────────────────
    if retrieval_max_sim < 0.30:
        return {
            "status": "Escalated",
            "reason_code": "low_grounding",
            "response_mode": "escalation",
            "justification_seed": (
                f"Retrieval similarity {retrieval_max_sim:.2f} below threshold; "
                "insufficient grounding to answer safely."
            ),
        }

    # ── 7. Grounded reply ─────────────────────────────────────────────────────
    return {
        "status": "Replied",
        "reason_code": "grounded",
        "response_mode": "grounded",
        "justification_seed": (
            f"Answer grounded in retrieved documentation (similarity {retrieval_max_sim:.2f})."
        ),
    }
