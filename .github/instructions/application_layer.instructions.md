---
applyTo: "**/application/*.py"
---

## ⚙️ Application Layer – Orchestration Rules

- Use this layer to:
  - Parse, dispatch, and call domain logic
  - Aggregate and validate question data
- All dependencies must be passed as arguments (Dependency Injection)
- Never import `requests`, `os`, or any infrastructure code
- Main classes:
  - `ForecastService`: wraps domain logic and decision-making
  - `Dispatcher`: receives questions, routes them
  - `IngestionService`: maps JSON → domain model
- Follow TDD:
  - Write the test first
  - Implement just enough to pass

# 🔧 Application Layer Instructions

## Goal
Coordinate domain logic and orchestration logic. No external APIs, but can orchestrate internal use cases.

## Responsibilities
- `ForecastService`: Coordinates domain logic and combines forecast data
- `Dispatcher`: Routes questions to the appropriate forecast method
- `IngestionService`: Converts raw JSON to domain-safe `Question` objects

## Rules
- Use only domain types, services, and pure Python modules
- Each class = one use-case
- No external I/O (only accept data as arguments)

## Files
- `forecast_service.py`
- `dispatcher.py`
- `ingestion_service.py`

## Commits
- `feat(app): initial ForecastService with binary forecast`
- `test(app): dispatcher routes questions correctly`

## Test Files
- `test_forecast_service.py`
- `test_dispatcher.py`
- `test_ingestion_service.py`