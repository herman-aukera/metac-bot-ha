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

## Runtime Validation (v1.0.0-rc1)

### Dry-Run Batch Test

Command:

```zsh
PYTHONPATH=$(pwd) poetry run python main_agent.py --mode batch --limit 3 --show-trace --dryrun
```

Result:

- No crashes
- All question types (binary, MC, numeric) handled
- Full trace output for each forecast
- Plugin/tool flows visible in trace

Example trace snippet:

```json
{
  "step": 3,
  "type": "tool",
  "input": { "tool": "WikipediaTool", "query": "Which city will win the bid?" },
  "output": "[WikipediaTool] No Wikipedia page found for 'Which city will win the bid?'.",
  "timestamp": "..."
}
```

### Batch Output Robustness

- `run_batch` supports all forecast result keys: `forecast`, `prediction`, `value`, etc.
- Fallback: "No forecast result available." if missing

### CLI Version

- `--version` prints: `Metaculus Agentic Bot version: v1.0.0-rc1`

### Release Verdict

- ✅ All tests pass
- ✅ CLI and docs verified
- ✅ Ready for v1.0.0-rc1 tag

---

Metaculus Agentic Bot v1.0.0-rc1 — https://github.com/l1dr/metac-agent-ha
