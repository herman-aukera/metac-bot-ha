# RELEASE NOTES — v1.0.0-rc1

## Features

- Binary, multi-choice, and numeric forecasting (Metaculus-style)
- Modular tool integration: Wikipedia, Math, and runtime plugin loader
- Plugin lifecycle hooks: `pre_forecast`, `post_submit` (trace-logged)
- Structured reasoning trace for all steps, tool calls, and plugin hooks
- CLI modes: dryrun, submit, batch, plugin-dir, show-trace, version
- CI-safe, fully tested (unit, integration, plugin lifecycle)
- Documentation: tools, pipeline, plugin system, lifecycle, CLI usage

## Known Limitations

- Plugins run in-process (no sandboxing or privilege separation)
- Webhook plugins require network access if enabled
- No built-in plugin repository or auto-update
- Only basic error handling for plugin exceptions
- LLM and search tool integration assumes compatible interface

## Upgrade Notes

- This is a release candidate. Only bugfixes and doc updates will be accepted before v1.0.0.
- See `docs/instructions/release_checklist.md` for release QA steps.

## How to Run

- See `main_agent.py --help` for CLI options
- Use `--version` to print the current release tag
- Use `--plugin-dir` and `--enable-webhooks` to load external tools

---

Metaculus Agentic Bot v1.0.0-rc1 — https://github.com/l1dr/metac-agent-ha
