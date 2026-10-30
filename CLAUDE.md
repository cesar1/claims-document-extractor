# Claims Document Extractor

Claim PDFs (contractor estimates, fire department reports, invoices) → structured JSON via Claude →
human review screen (approve/fix fields) → export. FastAPI backend in `backend/`, React (Vite) review UI
in `frontend/` (not built yet).

## Commands (run from `backend/`)
- Tests: `.venv/Scripts/python -m pytest -q`
- Lint/format: `.venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format .`
- Extract (live, costs money): `.venv/Scripts/python -m app.extract ../samples/real/<file>.pdf`

## Rules
- **API budget is ~$5–15 total.** Never run live extraction, the eval, or anything that calls Claude
  without asking first; say the expected cost (~$0.10/doc). Tests must mock the client.
- **Real claim documents are sensitive.** `samples/real/` and `data/` are gitignored — never commit,
  copy, or paste their contents elsewhere. Synthetic docs live in `samples/synthetic/`.
- **Git:** trunk-based. Never commit to `main` directly; work on `feat/*` / `fix/*` / `chore/*`,
  merge via PR. `main` auto-deploys to staging (once CI is set up). Commit only when asked.

## Claude API notes (things that 400 or surprise)
- Model `claude-opus-5-5`; effort set explicitly via `output_config.effort` (its default is `medium`).
  Thinking can't be disabled on this model — lower effort instead.
- Structured output via `client.beta.messages.stream(..., output_format=PydanticModel)`; read
  `message.parsed_output`. Don't use forced `tool_choice` (`any`/`tool`) — rejected on Opus 5.5.
- Native citations are incompatible with structured outputs, so provenance is in the schema:
  every field is `{value, page, source_text, confidence}` (`app/schemas.py`).
- `fallbacks="default"` + beta `server-side-fallback-2026-07-01` on live calls; the Batches API rejects
  `fallbacks`, so strip it for batch eval runs. Always check `stop_reason` (`refusal`, `max_tokens`).
- Extraction prompt rule: extract what is printed, never compute missing values. Math/date
  consistency checks belong in `app/validation.py`, not the prompt.

## Plan
Milestones: extraction core → synthetic docs + eval → validation rules → API + SQLite → review UI →
CI/staging deploy → README polish. Full plan: `docs/plan.md`.
