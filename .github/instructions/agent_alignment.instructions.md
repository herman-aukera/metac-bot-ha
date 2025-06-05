# Agent Alignment Instructions
# This document outlines the protocol for aligning the forecasting agent's implementation with the original meta-prompt.
# 🧠 Forecast Agent – Prompt-Aligned Development Protocol

## 🎯 Goal

Ensure every forecasting agent behavior defined in the original meta-prompt is:
- Implemented in code
- Covered by tests
- Documented with input/output & reasoning
- Traceable from prompt → code → test → doc

---

## ✅ Agent Requirements Recap (from meta-prompt)

| Capability                | Source File                         | Test File                          | Doc File                            |
|---------------------------|-------------------------------------|------------------------------------|--------------------------------------|
| Search Questions          | `metaculus_client.py`               | `test_metaculus_client.py`         | `/docs/api/metaculus_client.md`     |
| Evidence Retrieval (SearchTool) | `tools/search.py`             | `test_tools/test_search.py`        | `/docs/tools/search.md`             |
| CoT Forecast Logic        | `chains/forecast_chain.py`          | `test_chains/test_forecast_chain.py`| `/docs/chains/forecast_chain.md`    |
| Prediction Submission     | `forecast/submitter.py`             | `test_forecast/test_submitter.py`  | `/docs/forecast/submitter.md`       |
| Agent Composition         | `agent_runner.py`                   | `test_agent_runner.py`             | `/docs/agent/forecast_agent.md`     |
| CLI Entrypoint            | `main_agent.py`                     | `test_main_agent.py`               | `/docs/entrypoints/main_agent.md`   |
| Automation (CI)           | `.github/workflows/daily_run_agent.yaml` | N/A                          | `/docs/infra/ci_workflow.md`        |

---

## 🔄 Test & Documentation Alignment Rules

For each **LangChain Tool**, **Chain**, or **Agent behavior**:

- [ ] Is **tested** in `/tests/` with clear scenario + assertion
- [ ] Has **matching `.md` doc** in `/docs/` with:
  - 🧾 Input/Output schema
  - 🔁 Example I/O
  - 🤔 Reasoning logic (CoT steps, search paths, aggregation logic)
  - 🧪 Linked test (`../tests/...#Lxx`)

---

## 📘 Docs Must Contain:

```md
## Purpose
What does this module/chain/tool do?

## Input
```json
{ "example_input": "..." }

Output

{ "forecast": 0.82, "justification": "...based on evidence A, B, C..." }

Reasoning Logic

Step-by-step explanation (how it searches, how it thinks)

Related Tests
	•	test_…py#L23

---

## 🧪 Testing Rules

- Unit tests for:
  - Each tool (AskNews, math)
  - Chain logic (reasoning, structure)
  - Metaculus I/O
- Integration test for:
  - Full run: question → evidence → forecast
- `--dryrun` mode prints full forecast JSON

---

## 🧠 Prompt Sync Meta-Test

Every `README.md`, `/docs/*.md`, and prompt fragment must be auditable:
- Does this behavior exist in code?
- Is it tested?
- Is the result visible in dry-run?

> If a forecast feature isn’t tested and explained, it doesn’t exist.

---

## 🧩 Agent Evolution Checklist (MVP+)

| Feature                  | Status  | Doc? | Test? |
|--------------------------|---------|------|-------|
| AskNews tool             | ✅       | ✅    | ✅     |
| CoT ForecastChain        | 🔄       | ⏳    | 🔄     |
| CLI Entrypoint           | ✅       | ✅    | ✅     |
| LangChain ForecastAgent  | ✅       | ✅    | ✅     |
| Submitter                | 🔄       | ⏳    | 🔄     |
| Perplexity Tool          | ⏳       | ⏳    | ⏳     |
| Ensemble (Optional)      | ⏳       | ⏳    | ⏳     |

---

💡 Use `docs/traceability.md` to visualize prompt → code → test → doc for all behaviors.


⸻

Let me know if you want a CI job that checks this matrix automatically and flags any missing links.