# Metaculus Agentic Bot — v1.0.0-rc1 Release Candidate

## Overview

The Metaculus Agentic Bot is a modular, extensible forecasting agent supporting:

- Binary, multi-choice, and numeric forecasting
- Tool integration: Wikipedia, Math, and runtime plugin loader
- Plugin lifecycle hooks: `pre_forecast`, `post_submit`
- Structured reasoning trace for all steps, tool calls, and plugin hooks
- CLI modes: dryrun, submit, batch, plugin-dir, show-trace, version
- CI-safe, fully tested (unit, integration, plugin lifecycle)

## Installation

1. Clone the repository:
   ```zsh
   git clone <repo-url>
   cd metac-agent-ha
   ```
2. Install dependencies:
   ```zsh
   poetry install
   ```

## CLI Usage

Run the agent in dry-run mode (no API token required):

```zsh
PYTHONPATH=$(pwd) poetry run python main_agent.py --mode batch --limit 3 --show-trace --dryrun
```

Show version:

```zsh
poetry run python main_agent.py --version
```

Run with plugins:

```zsh
PYTHONPATH=$(pwd) poetry run python main_agent.py --plugin-dir ./plugins --enable-webhooks --show-trace --dryrun
```

## Plugin System

- Plugins can be loaded at runtime from a directory
- Support for both Python and webhook plugins
- Lifecycle hooks: `pre_forecast`, `post_submit` (see docs/tools.instructions.md)

## Trace Logging

- Every step, tool call, and plugin hook is logged in the output trace
- See `docs/forecast_pipeline.instructions.md` for trace format and examples

## Testing

- All features are covered by unit and integration tests
- Run tests with:
  ```zsh
  poetry run pytest
  ```

## Release Checklist

- All tests pass
- Docs are current
- CLI flags verified
- CI workflows passing
- No new features after this tag (only bugfix/docs)

## Changelog

See `CHANGELOG.md` for full release notes.

## License

MIT
