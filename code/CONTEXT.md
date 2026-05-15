# CONTEXT.md — Project Facts and Reference

Always-available reference. Re-read at session start. Never stale-trust this — verify against actual files when in doubt.

---

## What we are building

A terminal-based AI agent for the **HackerRank Orchestrate** hackathon (May 1–2, 2026). It triages support tickets across three product ecosystems — HackerRank, Claude, Visa — and decides per ticket whether to reply with a grounded answer or escalate to a human.

---

## Repo layout (relative to repo root)

```
hackerrank-orchestrate-may26/
├── AGENTS.md                       # ← SACRED. Read at session start. Sets log rules.
├── CLAUDE.md                       # → just imports AGENTS.md
├── README.md
├── problem_statement.md
├── evalutation_criteria.md         # (yes, "evalutation" — their typo)
├── code/                           # ← Our work goes here. Submission zip = this folder.
│   ├── PLAN.md
│   ├── PROGRESS.md
│   ├── CONTEXT.md                  # ← this file
│   ├── README.md                   # written in Module 11, install + run instructions
│   ├── main.py                     # entry point per AGENTS.md §6
│   └── (other modules added as built)
├── data/                           # ← READ-ONLY for us. The support corpus.
│   ├── hackerrank/
│   ├── claude/
│   └── visa/
└── support_tickets/                # ← NOT support_issues/. README has a typo.
    ├── sample_support_tickets.csv  # 108 rows, dev set with expected outputs
    ├── support_tickets.csv         # 56 rows, real test set, inputs only
    └── output.csv                  # we write this
```

---

## Corpus structure (critical)

The corpus is markdown files organized by product area. Subdirectory names ARE the `product_area` values:

**HackerRank** (`data/hackerrank/`):
```
chakra, engage, general-help, hackerrank_community, integrations,
interviews, library, screen, settings, skillup, uncategorized
```

**Claude** (`data/claude/`):
```
amazon-bedrock, claude, claude-api-and-console, claude-code,
claude-desktop, claude-for-education, claude-for-government,
claude-for-nonprofits, claude-in-chrome, claude-mobile-apps,
connectors, identity-management-sso-jit-scim, privacy-and-legal,
pro-and-max-plans, safeguards, team-and-enterprise-plans
```

**Visa** (`data/visa/`):
```
support, support.md   ← much smaller, possibly just one nested folder
```

Each subdirectory contains `.md` files (verify with `ls data/hackerrank/general-help/ | head` once cloned).

**Why this matters:** `Product Area` does NOT need an LLM call. It's the directory of the top retrieved chunk, mapped to the controlled vocabulary observed in the sample CSV (see taxonomy section below). Escalated rows leave `Product Area` blank.

---

## Input/output schema

### Input row (from CSV)
Column names are exactly as they appear in the CSV header (Title Case with spaces):
```
Issue:    str — the ticket body (untrusted, may contain injection or junk)
Subject:  str — may be empty/noisy/irrelevant
Company:  str — "HackerRank" | "Claude" | "Visa" | "None"
```

### Output row (to CSV)
Column names are exactly Title Case with spaces — do NOT use snake_case:
```
Issue:        str — pass-through from input
Subject:      str — pass-through from input
Company:      str — pass-through from input
Status:       Literal["Replied", "Escalated"]   ← Title Case
Product Area: str — controlled vocabulary (see taxonomy section below); EMPTY when Escalated
Response:     str — grounded user-facing answer, or escalation acknowledgment
Request Type: Literal["product_issue", "feature_request", "bug", "invalid"]
```

**CSV column order (exact):** `Issue, Subject, Company, Response, Product Area, Status, Request Type`

---

## Response style observations from sample data

Style varies by `Status` + `Request Type` combo — do NOT use one template for all rows:

| Status    | Request Type                          | Style |
|-----------|---------------------------------------|-------|
| Escalated | any                                   | Short mechanical text, e.g. "Escalate to a human" |
| Replied   | invalid (out-of-scope)                | Apologetic refusal, e.g. "I am sorry, this is out of scope from my capabilities" |
| Replied   | invalid (greeting / polite query)     | Friendly one-liner, e.g. "Happy to help! Please let me know what you need." |
| Replied   | product_issue / bug / feature_request | Multi-paragraph grounded text; may include URLs from corpus chunks |

Key rules derived from sample:
- Escalated responses are intentionally terse — do not over-explain.
- Polite/greeting replies are warm and brief — one or two sentences max.
- Grounded answers cite the corpus and can be several paragraphs.
- Never leave `Response` blank, even for escalations.

---

## Product Area taxonomy — controlled vocabulary

`Product Area` uses a **controlled vocabulary derived from the sample CSV**, NOT raw corpus directory names. The mapping from directory → controlled value is built in Module 8.

Confirmed values from the 10-row sample CSV (complete for sample, incomplete for test set):
```
community
conversation_management
general_support
privacy
screen
travel_support
(empty string)   ← escalated rows
```

**The 56-row test set may contain additional values not seen in the sample.** Module 8 must use a graceful fallback (lowercase-with-underscores of the corpus directory name) for unseen directories rather than hard-failing or emitting `general_support` blindly.

**Critical:** Escalated rows have **EMPTY Product Area** — confirmed from sample CSV inspection. Do not populate `Product Area` when `Status == "Escalated"`.

---

## Key engineering decisions

| Decision | Choice | Why |
|---|---|---|
| Vector store | numpy + cosine | Corpus is small (likely <5k chunks). FAISS adds setup cost without speed gain at this scale. |
| Embedding model | MiniLM-L6-v2 via fastembed (ONNX runtime, no torch dep) | User machine has 4GB RAM — sentence-transformers install OOMs due to torch transitive dep. |
| Classifier LLM | Claude Haiku | Cheap, fast, good enough for enum-constrained classification. |
| Generator LLM | Claude Sonnet 4.5 | Quality matters for grounded responses. Cost is fine at 56 rows. |
| Output validation | Pydantic | Forces enum compliance, catches malformed JSON before CSV write. |
| product_area inference | Directory of top retrieved chunk | Deterministic, free, more reliable than LLM classification. |
| Index per company | 3 separate index files | Filtering by company before scoring is faster + reduces cross-domain noise. |

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hallucinated policies in response | High | Strict generator prompt + max_sim threshold + INSUFFICIENT_CONTEXT escape hatch |
| Prompt injection from ticket body | High | Explicit anti-injection clause in prompt; treat ticket as data, not instruction |
| Wrong escalation calls (sensitive cases get replied to) | Medium | Two-layer defense: keyword rules + LLM sensitivity score |
| Output CSV malformed | Low | Pydantic validation + `pandas.to_csv` with `quoting=csv.QUOTE_ALL` |
| API rate limit / outage mid-run | Low | Tenacity retry + row-by-row writes (resume from last successful row) |
| AGENTS.md log violation | Medium | Use Claude Code (it auto-handles this); never edit log.txt by hand |

---

## Hackathon rules to remember

- Solo submission. You are the author.
- Terminal-based agent; no UI required.
- Corpus only — no live web calls for ground-truth answers.
- Secrets via env vars (`.env` is gitignored, `.env.example` is committed).
- Submission deadline: **2026-05-02 11:00 IST**.
- Submission via HackerRank Community Platform.
- Three artifacts: code zip (excluding data/ and support_tickets/), output.csv, ~/hackerrank_orchestrate/log.txt.
- AI Judge interview within hours after deadline. Camera on, mandatory.

---

## Anti-patterns to avoid

- ❌ Single mega-prompt that classifies + retrieves + generates in one call. Debug nightmare.
- ❌ Hardcoding API keys. Will lose marks even if it works.
- ❌ Writing output.csv only at the end. If row 50/56 crashes, you lose 49 rows of work.
- ❌ Trying to perfect Module 1 before Module 3 exists. Get the vertical slice first, then iterate.
- ❌ Treating `sample_support_tickets.csv` as a holdout. It's a dev set — read it, learn from expected outputs, calibrate your prompts on it.
- ❌ Editing `~/hackerrank_orchestrate/log.txt` manually. Append-only, secrets redacted, format per AGENTS.md §5.2.

---

## Open questions / things to verify

- [ ] Confirm `.md` files inside corpus subdirs (run `find data -name "*.md" | head`)
- [ ] Confirm each corpus file has clean prose (not just frontmatter or stubs)
- [ ] Read 5 rows of `sample_support_tickets.csv` with expected outputs to calibrate response style
- [ ] Read full `evalutation_criteria.md` for the actual rubric weights