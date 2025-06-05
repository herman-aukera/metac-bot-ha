---
applyTo: "**/forecasting_pipeline.py"
---

## 🔁 Forecasting Pipeline – Agent Execution

- Top-level entrypoint that connects:
  - Dispatcher → Tools → Prediction
- Compose `AgentOrchestrator`, `LLMTool`, and search tools
- Must log:
  - Forecast type
  - Agent strategy used
  - Source confidence levels

### Commit Tags
- `feat(pipeline): add ensemble agent strategy`
- `log(pipeline): log model used and score`