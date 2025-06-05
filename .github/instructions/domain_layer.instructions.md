---
applyTo: "**/domain/*.py"
---

## 🧱 Clean Domain Logic – Metaculus Bot

- Write pure, deterministic logic (no I/O, no side effects)
- Model domain concepts as Python `dataclass`es or `pydantic` models
- Use type hints (`Optional`, `Literal`, `Union`) for validation clarity
- Define a small vocabulary of reusable domain types:
  - `Question`, `Forecast`, `BrierScore`, `QuestionStatus`
- Favor immutability; don’t mutate models in-place
- Include logic like:
  - Scoring functions (`brier_score`)
  - Timestamp normalization
  - Type refinements (`BinaryQuestion`, `NumericQuestion`)
- Each domain function should be:
  - Unit tested
  - Independent of infrastructure or UI

# 🧠 Domain Layer Instructions

## Goal
Encapsulate all core business logic. This layer is *pure*, isolated from I/O, frameworks, or APIs.

## Responsibilities
- Represent core entities: `Question`, `ForecastResult`
- Implement logic like Brier score calculation
- Enforce validation, domain constraints, and business rules

## Principles
- Pure Python, no external I/O or dependencies
- Must be 100% covered with unit tests
- All domain types must use `pydantic.BaseModel` or equivalent for safety

## Files
- `forecast.py`: Brier score and forecast evaluation
- `question.py`: Core domain model with types, enums, and status logic

## Commit Guide
- `feat(domain): add QuestionStatus enum`
- `test(domain): add Brier score edge case test`

## Test Targets
- `tests/unit/domain/test_forecast.py`
- `tests/unit/domain/test_question.py`