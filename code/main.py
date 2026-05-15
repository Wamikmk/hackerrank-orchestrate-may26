"""
main.py — HackerRank Orchestrate entry point.
Usage:
    python code/main.py [--input PATH] [--output PATH] [--data-dir PATH] [--debug] [--limit N]
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_COLUMNS = [
    "Issue",
    "Subject",
    "Company",
    "Response",
    "Product Area",
    "Status",
    "Request Type",
]

VALID_STATUSES      = {"Replied", "Escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Support Triage Agent — HackerRank Orchestrate"
    )
    parser.add_argument("--input",    default=str(REPO_ROOT / "support_tickets" / "support_tickets.csv"))
    parser.add_argument("--output",   default=str(REPO_ROOT / "support_tickets" / "output.csv"))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--debug",    action="store_true")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Process only the first N rows (fast testing)")
    return parser.parse_args()


def print_banner(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("  Support Triage Agent — HackerRank Orchestrate 2026")
    print("=" * 60)
    print(f"  Input  : {Path(args.input).resolve()}")
    print(f"  Output : {Path(args.output).resolve()}")
    if args.limit:
        print(f"  Limit  : {args.limit} rows")
    print("=" * 60)


def read_input(path: str, limit: int | None = None) -> list[dict]:
    df = pd.read_csv(path, dtype=str).fillna("")
    if limit is not None:
        df = df.head(limit)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "issue":   r.get("Issue",   r.get("issue",   "")),
            "subject": r.get("Subject", r.get("subject", "")),
            "company": r.get("Company", r.get("company", "")),
        })
    return rows


def main() -> None:
    args = parse_args()
    print_banner(args)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline import process
    from schemas import TicketInput

    rows = read_input(args.input, limit=args.limit)
    n = len(rows)
    print(f"\nProcessing {n} tickets...\n", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status_counts   = Counter()
    rt_counts       = Counter()
    total_cost      = 0.0
    wall_t0         = time.perf_counter()

    # Open output file and write header immediately — incremental writes
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        out_f.flush()

        for i, raw in enumerate(rows, 1):
            row_t0 = time.perf_counter()
            try:
                ticket = TicketInput(**raw)
                out    = process(ticket)
            except Exception as exc:
                print(f"  [ERROR] Row {i}: {exc}", flush=True)
                out = {
                    "status":       "Escalated",
                    "request_type": "invalid",
                    "response":     "Escalate to a human.",
                    "product_area": "",
                    "justification": f"Pipeline error: {exc}",
                    "reason_code":  "error",
                    "cost_usd":     0.0,
                    "top_sim":      0.0,
                }

            row_ms = (time.perf_counter() - row_t0) * 1000
            status_counts[out["status"]] += 1
            rt_counts[out["request_type"]] += 1
            total_cost += out.get("cost_usd", 0.0)

            writer.writerow({
                "Issue":        raw["issue"],
                "Subject":      raw["subject"],
                "Company":      raw["company"],
                "Response":     out["response"],
                "Product Area": out["product_area"],
                "Status":       out["status"],
                "Request Type": out["request_type"],
            })
            out_f.flush()

            sim_str = f"sim={out.get('top_sim', 0):.2f}"
            print(
                f"  [{i:2d}/{n}] {raw['company']:10s} | {out['status']:9s} | "
                f"{out['reason_code']:20s} | {sim_str}  ({row_ms:.0f} ms)",
                flush=True,
            )
            if args.debug:
                print(f"         rt={out['request_type']}  pa={out['product_area']!r}  "
                      f"cost=${out['cost_usd']:.4f}")

    wall_s = time.perf_counter() - wall_t0

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {n} rows in {wall_s:.1f}s")
    print(f"{'='*60}")
    print(f"  Status distribution:")
    for k, v in sorted(status_counts.items()):
        print(f"    {k:10s}: {v}")
    print(f"  Request type distribution:")
    for k, v in sorted(rt_counts.items()):
        print(f"    {k:20s}: {v}")
    print(f"  Total cost : ${total_cost:.4f}")
    print(f"  Output     : {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
