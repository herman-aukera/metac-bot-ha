---
applyTo: "**/tests/**/*.py"
---

## ✅ Testing Strategy – Metaculus Bot

- Use `pytest` only
- Use `AsyncMock` for LLM or API calls
- Structure:
  - `tests/unit/` → isolated unit tests per layer
  - `tests/integration/` → multi-layer flows (e.g. dispatcher + ingestion)
  - `tests/e2e/` → ingest → forecast → output logged
- Aim for ≥ 80% test coverage
- Focus on:
  - Validation edge cases
  - Bad input, malformed questions
  - Brier score correctness
- Commit per layer or behavior