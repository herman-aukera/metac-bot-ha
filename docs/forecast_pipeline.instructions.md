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

## Tool Routing & Integration

The ForecastChain automatically invokes relevant tools (WikipediaTool, MathTool) during reasoning:

- WikipediaTool: fetches short summaries for named entities in the question
- MathTool: evaluates simple math expressions in the question
- Tool outputs are injected into the LLM's evidence/context
- If a tool fails or returns empty, reasoning continues with fallback evidence

### Example

```json
{
  "question": "What is quantum computing?",
  "tools_used": ["WikipediaTool"],
  "justification": "Based on Wikipedia's summary of quantum computing..."
}
```

See also: `tools.instructions.md` for tool details and test strategy.

## Model Selection & CLI Usage

All runs must use the project root as PYTHONPATH to resolve `src.*` imports:

```bash
PYTHONPATH=$(pwd) poetry run python main_agent.py --model openai/gpt-4 --mode batch --dryrun
```

- The `--model` flag selects the LLM backend (e.g. `openai/gpt-4`, `anthropic/claude-3`, `mistral/mixtral-8x7b`).
- If `PYTHONPATH` is not set, you will see a warning and imports may fail.
- This is required for all CLI and CI runs using the `src/` import path strategy.

See CLI help for more details.

## CLI Plugin Usage

To run with plugins:

```bash
PYTHONPATH=$(pwd) poetry run python main_agent.py --plugin-dir plugins/ --enable-webhooks --model openai/gpt-4 --dryrun
```

- `--plugin-dir`: Directory to load PluginTool and webhook plugins from
- `--enable-webhooks`: Allow loading webhook plugins from JSON
- Loaded plugins are listed at startup

### Plugin Trace Example

```json
{
  "step": 3,
  "type": "tool",
  "input": { "tool": "MyPlugin", "input": "What is the capital of France?" },
  "output": "Paris",
  "timestamp": "2025-06-06T12:34:56Z"
}
```

See `tools.instructions.md` for plugin schema and test details.

## Plugin Lifecycle Flow

Plugins can participate in the forecast process at two points:

- **pre_forecast(question)**: Called before reasoning. Can annotate, mutate, or log. Return value (if any) is logged in the trace.
- **post_submit(forecast)**: Called after submission. Can log, alert, or trigger side effects. Return value (if any) is logged in the trace.

### Example Trace

```json
{
  "step": 1,
  "type": "plugin_pre_forecast",
  "input": {"tool": "TaggingPlugin", "input": {"question_id": 1, ...}},
  "output": "tagged for science",
  "timestamp": "..."
}
{
  "step": 8,
  "type": "plugin_post_submit",
  "input": {"tool": "TaggingPlugin", "input": {"status": "ok"}},
  "output": "notified Discord",
  "timestamp": "..."
}
```

### When Hooks Fire

- All plugins are called in order for each hook.
- Hooks are optional; plugins may define only one or neither.
- If a hook returns a value, it is logged in the trace for auditability.
- No plugin can override or break core forecast logic.

## Batch Forecast Output Robustness

- `run_batch` now supports all forecast result keys: `forecast`, `prediction`, `value`, etc.
- Uses `.get()` to avoid `KeyError` and prints a fallback if no forecast is found.
- Justification is also robustly extracted.
- All question types (binary, MC, numeric) are handled in batch mode.
- Example:

```json
{
  "question_id": 2,
  "forecast": [0.33, 0.33, 0.33],
  "justification": "Mock LLM justification.",
  "trace": [ ... ]
}
```

If no forecast is found:

```json
{
  "question_id": 99,
  "forecast": "No forecast result available.",
  "justification": "No justification.",
  "trace": [ ... ]
}
```
