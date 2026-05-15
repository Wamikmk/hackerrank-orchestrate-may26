"""
evaluation.py — Dev-only diagnostic harness for Module 11.

Runs the full pipeline against sample_support_tickets.csv (which carries
expected outputs) and prints a summary table + per-row diff + failure report.

Usage:
    python code/evaluation.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import process
from schemas import TicketInput

REPO_ROOT   = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"

# Columns that carry expected values in the sample CSV
EXP_STATUS   = "Status"
EXP_RT       = "Request Type"
EXP_PA       = "Product Area"
EXP_RESPONSE = "Response"

GROUNDED_MODES = {"grounded"}     # only check grounding for these response modes


# ── Grounding heuristic ───────────────────────────────────────────────────────

def _is_grounded(response: str, top_chunk_text: str) -> bool:
    """
    True if the predicted response contains at least one 10-char substring
    that appears verbatim (case-insensitive) in the top retrieved chunk.
    Samples every 5 chars for efficiency.
    """
    resp_lower  = response.lower()
    chunk_lower = top_chunk_text.lower()
    step = 5
    window = 10
    for i in range(0, max(0, len(chunk_lower) - window + 1), step):
        if chunk_lower[i : i + window] in resp_lower:
            return True
    return False


# ── Formatting helpers ────────────────────────────────────────────────────────

def _tick(ok: bool) -> str:
    return "✓" if ok else "✗"


def _trunc(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(SAMPLE_PATH, dtype=str).fillna("")
    n  = len(df)
    print(f"Loaded {n} rows from {SAMPLE_PATH.name}\n", flush=True)

    records = []   # one dict per row
    total_cost = 0.0
    wall_t0    = time.perf_counter()

    for idx, row in df.iterrows():
        rownum = int(idx) + 1
        ticket = TicketInput(
            issue=row.get("Issue", ""),
            subject=row.get("Subject", ""),
            company=row.get("Company", ""),
        )
        print(f"  [{rownum:2d}/{n}] processing…", end="\r", flush=True)

        try:
            out = process(ticket)
        except Exception as exc:
            print(f"\n  [ERROR] Row {rownum}: {exc}")
            out = {
                "status": "Escalated", "request_type": "invalid",
                "response": "Escalate to a human.", "product_area": "",
                "reason_code": "error", "response_mode": "escalation",
                "top_sim": 0.0, "cost_usd": 0.0, "chunks": [],
            }

        total_cost += out.get("cost_usd", 0.0)

        exp_status = row.get(EXP_STATUS, "").strip()
        exp_rt     = row.get(EXP_RT, "").strip()
        exp_pa     = row.get(EXP_PA, "").strip()
        exp_resp   = row.get(EXP_RESPONSE, "").strip()

        pred_status = out["status"]
        pred_rt     = out["request_type"]
        pred_pa     = out["product_area"]
        pred_resp   = out["response"]

        status_match = pred_status.strip().lower() == exp_status.lower()
        rt_match     = pred_rt.strip().lower()     == exp_rt.lower()
        pa_match     = pred_pa.strip().lower()     == exp_pa.strip().lower()

        # Grounding check — only for grounded mode
        chunks = out.get("chunks", [])
        top_chunk_text = chunks[0]["text"] if chunks else ""
        if out.get("response_mode") in GROUNDED_MODES and top_chunk_text:
            grounded = _is_grounded(pred_resp, top_chunk_text)
            grounded_applicable = True
        else:
            grounded = None          # N/A
            grounded_applicable = False

        records.append({
            "rownum":                rownum,
            "issue":                 row.get("Issue", ""),
            "company":               row.get("Company", ""),
            "reason_code":           out.get("reason_code", ""),
            "response_mode":         out.get("response_mode", ""),
            "top_sim":               out.get("top_sim", 0.0),
            "pred_status":           pred_status,
            "exp_status":            exp_status,
            "status_match":          status_match,
            "pred_rt":               pred_rt,
            "exp_rt":                exp_rt,
            "rt_match":              rt_match,
            "pred_pa":               pred_pa,
            "exp_pa":                exp_pa,
            "pa_match":              pa_match,
            "pred_resp":             pred_resp,
            "exp_resp":              exp_resp,
            "grounded":              grounded,
            "grounded_applicable":   grounded_applicable,
        })

    wall_s = time.perf_counter() - wall_t0
    print(f"  Done in {wall_s:.1f}s\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    status_matches = sum(r["status_match"] for r in records)
    rt_matches     = sum(r["rt_match"]     for r in records)
    pa_matches     = sum(r["pa_match"]     for r in records)

    grounded_rows = [r for r in records if r["grounded_applicable"]]
    grounded_ok   = sum(1 for r in grounded_rows if r["grounded"])
    grounded_total = len(grounded_rows)

    mean_sim = sum(r["top_sim"] for r in records) / n if n else 0.0

    SEP = "-" * 56
    print(SEP)
    print(f"{'METRIC':<24} {'MATCHES':>7}  {'TOTAL':>5}  {'ACCURACY':>8}")
    print(SEP)
    print(f"{'Status':<24} {status_matches:>7}  {n:>5}  {status_matches/n*100:>7.1f}%")
    print(f"{'Request Type':<24} {rt_matches:>7}  {n:>5}  {rt_matches/n*100:>7.1f}%")
    print(f"{'Product Area':<24} {pa_matches:>7}  {n:>5}  {pa_matches/n*100:>7.1f}%")
    if grounded_total:
        print(f"{'Response Grounded':<24} {grounded_ok:>7}  {grounded_total:>5}  {grounded_ok/grounded_total*100:>7.1f}%")
    else:
        print(f"{'Response Grounded':<24} {'N/A':>7}  {'N/A':>5}  {'N/A':>8}")
    print(SEP)
    print(f"{'Mean retrieval sim':<24} {'':>7}  {'':>5}  {mean_sim:>8.3f}")
    print(f"{'Total cost':<24} {'':>7}  {'':>5}  ${total_cost:>7.4f}")
    print(SEP)

    # ── Per-row diff table ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PER-ROW DIFF")
    print("=" * 70)

    for r in records:
        g_tag = ""
        if r["grounded_applicable"]:
            g_tag = f" [{_tick(r['grounded'])} grounded]"

        flags = (
            f"[{_tick(r['status_match'])} status]"
            f" [{_tick(r['rt_match'])} rt]"
            f" [{_tick(r['pa_match'])} pa]"
            f"{g_tag}"
        )
        print(f"\nROW {r['rownum']} — {flags}")
        print(f"  Issue:    {_trunc(r['issue'], 80)!r}")
        print(f"  Company:  {r['company']}")
        print(f"  Reason:   {r['reason_code']} (sim={r['top_sim']:.2f})")

        def _row(label, pred, exp, match):
            tick = _tick(match)
            print(f"  {label:<10} pred={pred:<22} exp={exp:<22} {tick}")

        _row("Status:",   r["pred_status"], r["exp_status"],   r["status_match"])
        _row("ReqType:",  r["pred_rt"],     r["exp_rt"],       r["rt_match"])
        _row("ProdArea:", r["pred_pa"],     r["exp_pa"],       r["pa_match"])

        print(f"  Response (predicted, first 200 chars):")
        print(f"    {_trunc(r['pred_resp'], 200)!r}")
        print(f"  Response (expected,  first 200 chars):")
        print(f"    {_trunc(r['exp_resp'], 200)!r}")
        print("  ---")

    # ── Failure report ────────────────────────────────────────────────────────
    failures = [
        r for r in records
        if not r["status_match"] or not r["rt_match"] or not r["pa_match"]
        or (r["grounded_applicable"] and not r["grounded"])
    ]

    print()
    print("=" * 70)
    print(f"FAILURE REPORT — {len(failures)} rows with at least one mismatch")
    print("=" * 70)

    if not failures:
        print("  (none — all rows match on all metrics)")
    else:
        for r in failures:
            mismatches = []
            if not r["status_match"]:
                mismatches.append(f"status(pred={r['pred_status']!r} exp={r['exp_status']!r})")
            if not r["rt_match"]:
                mismatches.append(f"rt(pred={r['pred_rt']!r} exp={r['exp_rt']!r})")
            if not r["pa_match"]:
                mismatches.append(f"pa(pred={r['pred_pa']!r} exp={r['exp_pa']!r})")
            if r["grounded_applicable"] and not r["grounded"]:
                mismatches.append("not-grounded")
            print(f"\n  ROW {r['rownum']} [{r['company']}] reason={r['reason_code']}  sim={r['top_sim']:.2f}")
            print(f"    Issue: {_trunc(r['issue'], 72)!r}")
            for m in mismatches:
                print(f"    ✗ {m}")

    print()


if __name__ == "__main__":
    main()
