# forecast_pipeline.instructions.md

## Purpose

Describes the ForecastChain pipeline: how a Metaculus question is processed using CoT, evidence, and LLM to produce a forecast.

## Input

```json
{
  "question_id": 1,
  "question_text": "Will it rain tomorrow?"
}
```

## Output

```json
{
  "question_id": 1,
  "forecast": 0.88,
  "justification": "Based on evidence."
}
```

## Reasoning Logic

1. Extract question details (id, text)
2. Gather evidence using SearchTool (AskNews/Perplexity)
3. Build a modular CoT prompt (MCP style)
4. Invoke LLM (LangChain Runnable/LLMChain)
5. Parse LLM output as JSON with forecast and justification

## Related Tests

- `tests/unit/agent/chains/test_forecast_chain.py`
- `tests/integration/test_agent_full_path.py`

---

## Example Prompt

```
You are a forecasting agent.
Question: Will it rain tomorrow?
Evidence: Evidence for: Will it rain tomorrow? (mocked)
Think step by step and output a probability (0-1) and justification as JSON: {'forecast': float, 'justification': str}
```
