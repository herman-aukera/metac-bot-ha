---
applyTo: ".github/workflows/*.yml"
---

## 🌀 GitHub CI – Metaculus Forecast Bot

- Use Python 3.11+ only
- Add jobs for:
  - `make test` — run full pytest suite
  - `make lint` — check code quality
  - `make forecast` — generate local predictions (if ready)
  - Export coverage to a badge
  - Log number of forecasts submitted (once CLI exists)
- Schedule CI every 30min using:
```yaml
on:
  schedule:
    - cron: '*/30 * * * *'

---
# 🚀 CI / Workflow Instructions

## Setup Targets
- `make test` — runs unit + integration tests
- `make run` — local mode forecast with dummy input
- `make forecast` — production submission
- GitHub Actions — CI every 30 mins

## Tools
- Python 3.10+ (use `python3`, never plain `python`)
- Poetry for dependency management

## Paths
Fix this:
```text
WARNING: The script streamlit is installed in '/Users/herman/Library/Python/3.13/bin' which is not on PATH.