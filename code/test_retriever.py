"""
test_retriever.py — Acceptance test for Module 2 (Retriever).

Runs 6 hand-crafted queries and prints top-1 result for each.
Last query is the canary (off-topic — expect low score).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retriever import retrieve

QUERIES = [
    ("How do I reset my HackerRank password?",          "HackerRank",  "settings/general help"),
    ("My Visa card was charged twice for the same transaction", "Visa", "support/disputes"),
    ("What models are available in the Claude API?",    "Claude",      "claude-api-and-console"),
    ("I can't access my account",                       None,          "any company, account-related"),
    ("How do I export interview data?",                 "HackerRank",  "interviews/library"),
    ("What is the meaning of life?",                    None,          "low score canary"),
]

SEP = "-" * 72

print(SEP)
print(f"{'#':<3}  {'Query':<48}  {'Co':<11}  Score   product_area")
print(SEP)

for i, (query, company, expected) in enumerate(QUERIES, 1):
    results = retrieve(query, company=company, k=1)
    if not results:
        print(f"{i:<3}  {query[:48]:<48}  {str(company):<11}  NO RESULTS")
        continue
    top = results[0]
    score = top["score"]
    pa = top["product_area"]
    snippet = top["text"].replace("\n", " ")[:100]
    co_label = str(company) if company else "(all)"
    print(f"{i:<3}  {query[:48]:<48}  {co_label:<11}  {score:.4f}  {pa}")
    print(f"     expected : {expected}")
    print(f"     snippet  : {snippet}")
    print()

print(SEP)
