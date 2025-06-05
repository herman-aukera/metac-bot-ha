---
applyTo: "**/infrastructure/*.py"
---

## 🌐 Infrastructure Layer – IO, APIs, LLMs

- This layer handles all external systems:
  - Metaculus API (mock or real)
  - LLMs (`OpenAI`, `OpenRouter`)
  - `.env` files, proxies, retries
- Follow the adapter pattern:
  - `class MetaculusAPI: def fetch_json(self) → Dict`
  - `class OpenRouterClient: def predict(...) → str`
- Add `TODO` markers for real API integration
- Mock all external calls for tests
- Use `pydantic` models for response schemas

# 🛠 Infrastructure Layer Instructions

## Goal
Interface external services (APIs, files, models). Wrap external dependencies behind stable interfaces.

## Responsibilities
- `metaculus_api.py`: Mocked fetch for questions
- Wrappers for OpenRouter, AskNews, etc. (future)
- Must mock external APIs for testing

## Files
- `metaculus_api.py` (mocked only for now)

## Commit Conventions
- `feat(infra): add mock Metaculus fetcher`
- `test(infra): validate mocked question schema`

## Test Targets
- `tests/unit/infrastructure/test_metaculus_api.py`