---
applyTo: "**/forecasting_pipeline.py"
---

## 🔁 Forecasting Pipeline

- Avoid rewriting this file unless tests force it
- Add `run_ensemble_forecast` only if test demands it
- Reuse domain/application services (do not re-implement logic)
- Avoid monolithic methods — split into composable stages
- Add logging for:
  - Question type
  - LLM model used
  - Confidence score
- Handle missing methods with minimal additions
- Use `@dataclass` + pure orchestration