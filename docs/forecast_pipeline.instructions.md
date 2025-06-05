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

## Multi-Choice (MC) Forecasts

### Input Example

```json
{
  "question_id": 2,
  "question_text": "Which city will win the bid?",
  "type": "mc",
  "options": ["London", "Paris", "Berlin"]
}
```

### Output Example

```json
{
  "question_id": 2,
  "forecast": [0.1, 0.3, 0.6],
  "justification": "Based on evidence for all options."
}
```

### Reasoning Logic (MC)

1. Extract question details (id, text, options)
2. Gather evidence using SearchTool
3. Build a modular CoT prompt for MC (requesting a probability list)
4. Invoke LLM (LangChain Runnable/LLMChain)
5. Parse LLM output as JSON with forecast (list) and justification

### Example MC Prompt

```
You are a forecasting agent.
Question: Which city will win the bid?
Options: ['London', 'Paris', 'Berlin']
Evidence: Evidence for: Which city will win the bid? (mocked)
Think step by step and output a probability list (summing to 1) and justification as JSON: {'forecast': [float, ...], 'justification': str}
```

### Related Tests (MC)

- `tests/unit/agent/chains/test_forecast_chain.py` (MC)
- `tests/unit/api/test_metaculus_client.py` (MC)
- `tests/integration/test_submission_pipeline.py` (MC)

## Numeric Forecasts

### Input Example

```json
{
  "question_id": 3,
  "question_text": "How many widgets will be sold?",
  "type": "numeric"
}
```

### Output Example

```json
{
  "question_id": 3,
  "prediction": 42.0,
  "low": 30.0,
  "high": 60.0,
  "justification": "Based on numeric evidence."
}
```

### Reasoning Logic (Numeric)

1. Extract question details (id, text, type)
2. Gather evidence using SearchTool
3. Build a modular CoT prompt for numeric (requesting prediction and interval)
4. Invoke LLM (LangChain Runnable/LLMChain)
5. Parse LLM output as JSON with prediction, low, high, justification

### Example Numeric Prompt

```
You are a forecasting agent.
Question: How many widgets will be sold?
Evidence: Evidence for: How many widgets will be sold? (mocked)
Think step by step and output a numeric prediction and 90% confidence interval as JSON: {'prediction': float, 'low': float, 'high': float, 'justification': str}
```

### Related Tests (Numeric)

- `tests/unit/agent/chains/test_forecast_chain.py` (numeric)
- `tests/unit/api/test_metaculus_client.py` (numeric)

## Tool Routing

- The agent can use WikipediaTool and MathTool for evidence and calculation.
- Tools are available in `tool_list` and can be injected into chains or called from the agent.
- Example output includes `"tools_used": ["WikipediaTool", "MathTool"]` for traceability.

### Example Reasoning

```json
{
  "question": "What is the population of France?",
  "tools_used": ["WikipediaTool"],
  "justification": "Based on Wikipedia's summary of France..."
}
```

## Model Selection & CLI Usage

All runs must use the project root as PYTHONPATH to resolve `src.*` imports:

```bash
PYTHONPATH=$(pwd) poetry run python main_agent.py --model openai/gpt-4 --mode batch --dryrun
```

- The `--model` flag selects the LLM backend (e.g. `openai/gpt-4`, `anthropic/claude-3`, `mistral/mixtral-8x7b`).
- If `PYTHONPATH` is not set, you will see a warning and imports may fail.
- This is required for all CLI and CI runs using the `src/` import path strategy.

See CLI help for more details.
