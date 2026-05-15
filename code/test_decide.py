"""
test_decide.py — Acceptance test for Module 6 (Decision logic).

14 cases covering every branch + boundary edges.
Cases 1-8: original suite.
Cases 9-11: Module 9 calibration — initial Visa fix.
Cases 12-14: Patch 4 — hard ceiling raised 0.85→0.92, soft gate 0.55→0.50.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decide import decide

# Helper builders
def _rules(triggered=False, category=None):
    return (triggered, category, category)

def _cls(request_type="product_issue", sensitivity=0.3, polite=False):
    return {
        "request_type": request_type,
        "sensitivity": sensitivity,
        "is_polite_chatter": polite,
        "inferred_company": "HackerRank",
    }

# (label, rules_result, classifier_result, max_sim, company,
#  expect_status, expect_reason_prefix, expect_mode)
CASES = [
    (
        "1  rules:fraud fired",
        _rules(True, "fraud"), _cls(sensitivity=0.1), 0.80, "Visa",
        "Escalated", "rules:fraud", "escalation",
    ),
    (
        "2  sensitivity > 0.92 hard ceiling (max_sim=0.80)",
        _rules(False), _cls(sensitivity=0.95), 0.80, "HackerRank",
        "Escalated", "sensitivity_hard", "escalation",
    ),
    (
        "3  invalid + polite chatter",
        _rules(False), _cls(request_type="invalid", sensitivity=0.2, polite=True), 0.10, "None",
        "Replied", "polite_chatter", "polite",
    ),
    (
        "4  invalid + off-topic (not polite)",
        _rules(False), _cls(request_type="invalid", sensitivity=0.0, polite=False), 0.10, "None",
        "Replied", "off_topic", "out_of_scope",
    ),
    (
        "5  low grounding (max_sim=0.15)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.3), 0.15, "HackerRank",
        "Escalated", "low_grounding", "escalation",
    ),
    (
        "6  grounded reply (max_sim=0.55)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.3), 0.55, "HackerRank",
        "Replied", "grounded", "grounded",
    ),
    (
        "7  edge: sensitivity == 0.7 (must NOT escalate)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.7), 0.55, "HackerRank",
        "Replied", "grounded", "grounded",
    ),
    (
        "8  edge: max_sim == 0.30 (must NOT escalate)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.3), 0.30, "HackerRank",
        "Replied", "grounded", "grounded",
    ),
    # ── Module 9 calibration — Visa sensitivity fix ───────────────────────────
    (
        "9  sensitivity=0.95, max_sim=0.65 → ESCALATE (hard ceiling at 0.92)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.95), 0.65, "Visa",
        "Escalated", "sensitivity_hard", "escalation",
    ),
    (
        "10 sensitivity=0.75, max_sim=0.65 → REPLY (doc-backed, below 0.92 ceiling)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.75), 0.65, "Visa",
        "Replied", "grounded", "grounded",
    ),
    (
        "11 sensitivity=0.75, max_sim=0.40 → ESCALATE (sensitive + sim<0.50)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.75), 0.40, "Visa",
        "Escalated", "sensitivity_unsupported", "escalation",
    ),
    # ── Patch 4 — raised ceiling 0.85→0.92, soft gate 0.55→0.50 ─────────────
    (
        "12 sensitivity=0.95, max_sim=0.65 → ESCALATE (hard ceiling 0.92)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.95), 0.65, "Visa",
        "Escalated", "sensitivity_hard", "escalation",
    ),
    (
        "13 sensitivity=0.88, max_sim=0.65 → REPLY (0.7<sens<0.92, sim>=0.50)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.88), 0.65, "Visa",
        "Replied", "grounded", "grounded",
    ),
    (
        "14 sensitivity=0.88, max_sim=0.40 → ESCALATE (0.7<sens<0.92, sim<0.50)",
        _rules(False), _cls(request_type="product_issue", sensitivity=0.88), 0.40, "Visa",
        "Escalated", "sensitivity_unsupported", "escalation",
    ),
]

SEP = "-" * 72

def main() -> None:
    passed = failed = 0

    print(SEP)
    for label, rules_r, cls_r, max_sim, company, exp_status, exp_reason, exp_mode in CASES:
        result = decide(rules_r, cls_r, max_sim, company)

        reason_ok = result["reason_code"] == exp_reason
        ok = (
            result["status"] == exp_status
            and reason_ok
            and result["response_mode"] == exp_mode
        )

        tag = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n{tag}  {label}")
        print(f"  input   rules_triggered={rules_r[0]}  sensitivity={cls_r['sensitivity']}"
              f"  request_type={cls_r['request_type']!r}"
              f"  polite={cls_r['is_polite_chatter']}  max_sim={max_sim}")
        print(f"  output  {result}")
        if not ok:
            print(f"  EXPECTED  status={exp_status!r}  reason_code={exp_reason!r}"
                  f"  response_mode={exp_mode!r}")

    print(f"\n{SEP}")
    print(f"Passed: {passed}/{len(CASES)}   Failed: {failed}/{len(CASES)}")

if __name__ == "__main__":
    main()
