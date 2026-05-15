# PROGRESS.md — Module Checklist

Update after each module finishes. Two-line retrospective per completed module.

**Status legend:** `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Modules

- [x] **Module 0 — Bootstrap** (15 min)  
  _Notes:_ `main.py` with argparse (--input/--output/--data-dir/--debug) passes both acceptance tests. Created `requirements.txt` and `.env.example`.

- [x] **Module 1 — Corpus indexer** (60 min)  
  _Notes:_ `corpus_loader.py` (load+chunk, 180-word chunks, 35-word overlap, MIN_WORDS=20 filter) + `build_index.py` (fastembed ONNX embed + save). 4026 chunks total after filter (HackerRank: 2320, Claude: 1669, Visa: 37). Saved to `code/index/embeddings.npy` (6.2 MB, shape 4026×384) and `code/index/metadata.jsonl` (8.4 MB). All 6 identified noise chunks (bare `\`, orphan headings, index stub) confirmed absent. Min chunk = 21 words.

- [x] **Module 2 — Retriever** (30 min)  
  _Notes:_ `retriever.py` with `retrieve(query, company, k)`. Loads embeddings+metadata+model once at module level. Company masks cached. `query_embed` with fallback. Cosine sim via `np.dot`. 6-query acceptance test passed: scores 0.76/0.60/0.75/0.50/0.75/0.19 (canary correctly lowest). `test_retriever.py` written.

- [x] **Module 3 — Schemas + skeleton pipeline** (30 min) ← VERTICAL SLICE  
  _Notes:_ `schemas.py` (TicketInput, TriageOutput with Title Case Status enum). `pipeline.py` (6 stub stages, real retrieve call wired in). `main.py` updated with CSV reader (Title Case → lowercase), CSV writer (exact column order, QUOTE_ALL), `--limit` flag. Acceptance: `--limit 10` → 10 rows, columns `['Issue', 'Subject', 'Company', 'Response', 'Product Area', 'Status', 'Request Type']`, all Status=Escalated, Response="Stub response", Product Area blank.

- [x] **Module 4 — Rules-based safety net** (30 min)  
  _Notes:_ `rules.py` with `check_rules(issue, subject)`. 7 categories, priority order: self_harm > fraud > identity > legal > production_bug > billing_dispute > prompt_injection. Patterns compiled once at import. Key fix: `account.{0,10}hacked` (not literal "account hacked") to match "account was hacked". "act as" narrowed to AI personas to avoid "act as a candidate" false positive. 15/15 tests pass in 1.4 ms.

- [x] **Module 5 — Classifier (Haiku)** (45 min)  
  _Notes:_ `classifier.py` with `classify(issue, subject, company)`. Model: claude-haiku-4-5. System prompt with anti-injection XML delimiter rule + sensitivity scale + 5 few-shot examples. Pydantic validation of JSON response; one parse-retry then safe defaults. Tenacity wrap (3 attempts, exp backoff). 6/6 acceptance cases pass: bug (0.5), polite chatter (is_polite_chatter=true), off-topic (invalid), product_issue (HackerRank), stolen Visa card (sensitivity 0.9), injection attempt (correctly classified as product_issue, HackerRank — JSON not derailed). Total: 5643in/339out, $0.0059.

- [x] **Module 6 — Decision logic** (20 min)  
  _Notes:_ `decide.py` — pure function, no I/O. Priority order: rules > sensitivity>0.7 > invalid+polite > invalid+off-topic > max_sim<0.30 > grounded. Returns status/reason_code/response_mode/justification_seed. Both boundary edges confirmed strict (sensitivity==0.7 → Replied, max_sim==0.30 → Replied). 8/8 tests pass.

- [x] **Module 7 — Grounded generator (Sonnet)** (45 min)  
  _Notes:_ `generator.py` with `generate(issue, subject, company, response_mode, retrieved_chunks, justification_seed)`. Short-circuits escalation/polite/out_of_scope with zero API calls (latency ~0 ms). Grounded mode calls claude-sonnet-4-5 with anti-injection system prompt + XML-delimited ticket. Pydantic parse, one JSON-only retry, force_escalate on double failure. Detects INSUFFICIENT_CONTEXT case-insensitively. 5/5 acceptance cases pass: grounded answer (test active, no hallucination), INSUFFICIENT_CONTEXT (swallow question correctly refused), injection attempt (system prompt not leaked, deletion steps answered correctly), escalation/polite short-circuits both under 1 ms. Total 3 API calls, 6831in/408out, $0.027.

- [ ] **Module 8 — Product area inference** (15 min)  
  _Notes:_ 

- [~] **Module 9 — Wire together + sample run** (60 min)  
  _Notes:_ Pipeline wired (stubs replaced). 10-row sample run: 7 Replied / 3 Escalated, 80.7 s, $0.0945. CSV validates: 7 columns, correct Title Case headers, Status ∈ {Replied,Escalated}, Request Type ∈ {product_issue,bug,invalid}. Evaluation harness (Module 11) runs next for per-row diff analysis.

- [ ] **Module 10 — Edge cases pass** (45 min)  
  _Notes:_ 

- [x] **Module 11 — Evaluation harness** (30 min)  
  _Notes:_ `evaluation.py` reads sample CSV (10 rows), runs full pipeline, computes status/rt/pa/grounded matches, prints summary table + per-row diff + failure report. First run: Status 80% (8/10), Request Type 100% (10/10), Product Area 30% (3/10), Response Grounded 100% (5/5). 7 rows with mismatches, all are PA mismatches (stub returning raw dir names) + 2 status errors (Visa rows over-escalating on sensitivity). Saved to /tmp/eval_report.txt.

- [ ] **Module 12 — Final run + submission** (30 min)  
  _Notes:_ 

---

## Current blocker (if any)

_None._

---

## Decisions log (append-only, newest at top)

- [2026-05-01] Discovered sample_support_tickets.csv has 10 rows (not 108 as previously assumed). Sample is too small for percentage-based accuracy targets — Module 9 switches to per-row defensibility check with 1-line rationale for any diff. Product Area vocab is sample-derived (7 values: community, conversation_management, general_support, privacy, screen, travel_support, empty) and likely incomplete for the 56-row test set; Module 8 needs a robust fallback (lowercase-with-underscores of directory name) rather than hardcoded `general_support`.
- [2026-05-01] Schema corrections from sample CSV inspection: (1) Column names are Title Case with spaces — `Issue, Subject, Company, Response, Product Area, Status, Request Type` — not snake_case. (2) `Status` values are Title Case: `"Replied"` / `"Escalated"`. (3) `Product Area` uses a controlled vocabulary from the sample CSV, not raw corpus directory names. (4) Escalated rows have EMPTY `Product Area`. (5) Response style varies by status+request_type: escalated=terse, polite/greeting=warm one-liner, out-of-scope=apologetic refusal, grounded=multi-paragraph with URLs.
- [2026-05-01] Switched embedding library from sentence-transformers to fastembed. Same model (MiniLM-L6-v2, 384-dim), no torch dependency, fits 4GB RAM. Hybrid BM25+dense retrieval considered but deferred — may add as Module 1.5 if Module 9 evaluation shows accuracy gaps.
- [2026-05-01] Corpus is heavily imbalanced: HackerRank 2351 chunks, Claude 1671, Visa 40. Visa coverage is ~1% of corpus. Expect more Visa tickets to escalate due to low retrieval similarity, even when questions are reasonable. Consider Visa-specific threshold tuning in Module 6 if sample evaluation shows excess Visa false-escalations.

---

## Surprises encountered

_(unexpected things — corpus quirks, API behavior, evaluator feedback)_