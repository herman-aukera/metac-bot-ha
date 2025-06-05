---
applyTo: "**/domain/*.py"
---

## 🧠 Domain Logic – Forecasting Intelligence

- Pure logic for reasoning, validation, and scoring
- Models: `ForecastResult`, `Question`, `ConfidenceScore`
- Utils: `brier_score`, `decompose_question`

### Principles
- No side effects, no IO, no LLM calls
- Use dataclasses or pydantic
- Fully unit-tested and immutable

### Commit Tags
- `feat(domain): add composite forecast type`
- `test(domain): validate scoring bounds`