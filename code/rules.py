"""
rules.py — Keyword/regex safety net. Runs before any LLM call.

Priority order (highest first):
  self_harm > fraud > identity > legal > production_bug > billing_dispute > prompt_injection

First match wins; returns on first hit.
"""

import re
from typing import Optional

# Each entry: (category, [raw_pattern_strings])
# Patterns are compiled once at import time with IGNORECASE.
# Text is also lowercased before matching (belt-and-suspenders).
_RAW_RULES: list[tuple[str, list[str]]] = [
    ("self_harm", [
        r"kill myself",
        r"end my life",
        r"want to die",
    ]),
    ("fraud", [
        r"fraud",
        r"unauthorized transaction",
        r"stolen card",
        r"lost my card",
        r"card was stolen",
        r"didn.t make this transaction",
        r"didn.t authorize",
    ]),
    ("identity", [
        r"account.{0,10}hacked",
        r"account.{0,10}compromised",
        r"someone accessed my account",
        r"lost access to.{0,10}2fa",
        r"lost.{0,20}authenticator",
        r"verify my identity",
        r"password reset.{0,30}different email",
    ]),
    ("legal", [
        r"lawsuit",
        r"\blegal action\b",
        r"\bgdpr\b",
        r"\bdmca\b",
        r"privacy violation",
        r"data breach",
    ]),
    ("production_bug", [
        r"production down",
        r"site is down",
        r"all my candidates can.t",
        r"whole team.s blocked",
        r"everyone affected",
    ]),
    ("billing_dispute", [
        r"charged twice",
        r"double charge",
        r"chargeback",
        r"refund.{0,20}denied",
        r"wrongly charged",
        r"subscription.{0,20}won.t cancel",
    ]),
    # "act as" is narrowed to AI/system personas to avoid false positives
    # like "I want to act as a candidate".
    ("prompt_injection", [
        r"ignore previous instructions",
        r"ignore all previous",
        r"you are now",
        r"act as (?:if|though|an? (?:ai|assistant|llm|gpt|chatbot|bot|system|admin|unrestricted|jailbreak|dan|evil|hacker|new persona))",
        r"\bsystem prompt\b",
        r"your real instructions",
        r"disregard.{0,20}above",
    ]),
]

# Compile once at module load
_RULES: list[tuple[str, list[re.Pattern]]] = [
    (cat, [re.compile(p, re.IGNORECASE) for p in patterns])
    for cat, patterns in _RAW_RULES
]


def check_rules(
    issue: str, subject: str = ""
) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Returns (should_escalate, reason, category).
    Categories: "fraud", "identity", "billing_dispute", "legal",
                "production_bug", "self_harm", "prompt_injection", None.
    Inputs are checked case-insensitively against the combined text.
    """
    text = (subject + " " + issue).lower()

    for category, patterns in _RULES:
        for pattern in patterns:
            m = pattern.search(text)
            if m:
                return True, m.group(0), category

    return False, None, None
