# Claims Document Extractor: Plan

## Context
This is a portfolio/FDE-style project. Insurance claims arrive as messy PDFs: contractor estimates, fire department incident reports, and invoices. Adjusters re-key them by hand. The goal is unstructured PDF in, validated system-ready JSON out, plus a human-in-the-loop review screen where a person approves or corrects every field before the data is considered final. The API budget is about $5–15. The directory `P:\source_code\dev\extractor` is empty (greenfield).

**Decisions made:** Python FastAPI + React (Vite). Real redacted samples plus synthetic docs with ground truth. PDFs are sent directly to Claude, with no OCR stage.

---

## Architecture

```
Upload PDF ──► FastAPI ──► store file + doc row (status=queued)
                  │
                  ▼
        1. classify doc type   (estimate | fire_report | invoice | other)
        2. extract with type-specific Pydantic schema (Claude, structured output)
        3. validate  (Pydantic + business rules: line items sum to totals, dates sane)
        4. status = needs_review  (confidence + validation flags per field)
                  │
                  ▼
React review screen: PDF viewer (left) | editable fields (right)
  low-confidence / failed-rule fields highlighted, click field → jump to page
  Approve → status=approved, store corrected JSON + diff vs. model output
                  │
                  ▼
GET /documents/{id}/export → final JSON  (the "system-ready" payload)
```

**Claude API approach (Python `anthropic` SDK):**
- Model: `claude-opus-5-5` ($4 / $20 per MTok). Set `output_config.effort` explicitly: start at `medium`, and try `low` in the eval.
- PDF goes in as a `document` content block (base64). Claude reads both the text layer and the page images, so scans work.
- Use structured outputs: `client.messages.parse(..., output_format=<Pydantic model>)`, the SDK helper that wraps `output_config.format`. Don't use forced `tool_choice`, because Opus 5.5 rejects it.
- **Provenance:** native Citations can't be combined with structured outputs (the API returns a 400). So every extracted field is wrapped as `{value, page, source_text, confidence}` inside the schema itself. The review UI uses `page` and `source_text` to jump to and highlight the evidence.
- Enable server-side `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) so a classifier refusal doesn't fail a document. Handle the `stop_reason` values `refusal` and `max_tokens`.
- Prompt caching on the static system prompt and schema instructions.
- Message Batches API (50% off) for eval runs over the whole sample set.

## Data model (Pydantic, `backend/app/schemas.py`)
- `Field[T]`: value, page, source_text, confidence (0–1)
- `ClaimCommon`: claim/policy number, insured name, property address, date_of_loss, cause_of_loss (enum: fire, water, wind, hail, theft, other, plus free text)
- `LineItem`: description, quantity, unit, unit_price, amount, category
- `ContractorEstimate`: common + contractor info, line_items[], subtotal, tax, overhead_profit, total
- `FireReport`: common + incident number, department, alarm/arrival times, origin/cause determination, injuries, narrative summary
- `Invoice`: common + vendor, invoice number/date, line_items[], subtotal, tax, total, amount due
- Validation rules (`validation.py`): the sum of line items equals the subtotal within ±$0.01, subtotal + tax (+ O&P) equals the total, date_of_loss is not in the future and comes before the invoice date, and required fields are present. Each failed rule becomes a field flag.

## Project layout
```
extractor/
  backend/
    app/
      main.py            FastAPI routes
      schemas.py         Pydantic doc schemas
      extract.py         classify + extract via Claude
      prompts.py         system prompts per doc type
      validation.py      business-rule checks
      db.py              SQLite (SQLModel): documents, extractions, reviews
      storage.py         save PDFs to ./data/uploads
    tests/               pytest (validation rules, schema, mocked Claude)
    pyproject.toml
  frontend/              Vite + React + TS
    src/pages/Queue.tsx        list of docs + status
    src/pages/Review.tsx       react-pdf viewer + field form
    src/components/FieldRow.tsx  value, confidence badge, flag, "go to page"
  samples/
    real/                your redacted PDFs (gitignored)
    synthetic/           generated PDFs + *.truth.json
  scripts/
    gen_synthetic.py     reportlab/Faker → PDFs + ground truth; some rasterized/rotated/noisy to mimic scans
    eval.py              run extractor over samples, field-level accuracy report
```

## API endpoints
- `POST /documents` (upload, then run the pipeline in a background task)
- `GET /documents` (queue with status: queued, processing, needs_review, approved, failed)
- `GET /documents/{id}` (extraction, flags, and PDF URL)
- `GET /documents/{id}/pdf`
- `PUT /documents/{id}/review` (corrected fields plus approve; stores the diff)
- `POST /documents/{id}/reextract`
- `GET /documents/{id}/export` (final approved JSON)

## Build milestones
1. **Extraction core (CLI first):** schemas, prompts, and `extract.py`. Run it on 3 PDFs and print the JSON. No UI yet.
2. **Synthetic data + eval:** `gen_synthetic.py` (about 30 docs: 10 per type, about a third scan-like) and `eval.py` (per-field exact or normalized match, numeric tolerance, and dates normalized). This gives a baseline accuracy number.
3. **Validation layer:** math and date rules, and flags.
4. **Backend API + SQLite.**
5. **Review UI:** queue, then a side-by-side review with highlighted low-confidence or flagged fields, edit, and approve.
6. **Polish for portfolio:** README with an architecture diagram, eval results table, cost per doc, and a short demo GIF.

## Git branching and environments
Use lightweight trunk-based development. With one developer and a single staging environment, GitFlow (develop/release/hotfix) adds ceremony and gives nothing back.

| Branch | Purpose | Deploys to |
|---|---|---|
| `main` | Always green, always deployable | **Staging**, automatically on merge |
| `feat/<short-name>`, `fix/<short-name>` | One milestone or change each, short-lived (days, not weeks) | Local only (optionally a preview) |
| tags `v0.1.0`, `v0.2.0`… | Snapshot of `main` you'd call a release | **Prod**, later, when one exists |

Rules:
- Never commit directly to `main`. Each change goes feat branch → PR → squash-merge, so one commit per feature and a clean history.
- CI on every PR (GitHub Actions): `ruff`, `pytest` (Claude mocked, $0), frontend `tsc` + `npm run build`. Branch protection on `main` requires CI to pass.
- **No live Claude calls in CI**: the eval (`scripts/eval.py`) is run manually or by a `workflow_dispatch` job so the API budget can't be drained by pushes.
- Merging to `main` triggers a deploy-to-staging workflow. Promoting to prod later means tagging a commit already verified on staging (no separate prod branch).
- Config by environment, not by branch: `.env.local` / staging secrets (`ANTHROPIC_API_KEY`, DB URL, storage path) live in the host's secret store. The same image runs in both places.
- `.gitignore` from the first commit: `.env*`, `samples/real/`, `data/uploads/`, `*.db`, `node_modules/`, `.venv/`. Real claim PDFs never enter git history.

Suggested first branches, matching the milestones: `feat/extraction-core` → `feat/synthetic-eval` → `feat/validation` → `feat/api` → `feat/review-ui` → `chore/ci-staging-deploy` (this one can come earlier, right after the API exists).

## Cost estimate
About 2–3K input tokens per PDF page. A typical 2–4 page doc is roughly 10K input and 3K output including thinking, so about **$0.10 per doc**. Dev iteration (around 50 calls), plus 4–5 eval passes over about 40 docs through Batches, comes to about **$8–12**, which is inside the budget. `eval.py` logs `usage` per call and totals the cost.

## Verification
- `pytest backend/tests`: validation rules, schema round-trips, and the extractor with a mocked client.
- `python scripts/eval.py samples/synthetic`: field-level accuracy per doc type. The target is ≥95% on clean docs, and the number on scans is reported separately.
- Manual end-to-end: run `uvicorn app.main:app` and `npm run dev`, upload a real redacted estimate, confirm flagged fields are highlighted, fix one field, approve, and check that `/export` returns the corrected JSON and the diff is stored.
