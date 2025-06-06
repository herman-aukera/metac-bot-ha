# agentic-usecases.md

# Metaculus Agentic Bot — Use Cases & Example Workflows

## Example Input JSON

```json
{
  "question_id": 123,
  "question_text": "Will it rain in London on July 1, 2025?"
}
```

## Example Forecast Output

```json
{
  "question_id": 123,
  "forecast": 0.74,
  "justification": "Reasoning for: Will it rain in London on July 1, 2025? based on evidence. Evidence: Evidence for: Will it rain in London on July 1, 2025? (stubbed)"
}
```

## Reasoning Explanation Sample

- The agent fetches evidence (AskNews/Perplexity)
- Performs chain-of-thought reasoning
- Outputs a justified forecast with traceable logic

## Workflow

1. Receive new Metaculus question JSON
2. Fetch evidence using search tool
3. Run CoT reasoning chain
4. Output forecast + justification
5. (If not dryrun) Submit to Metaculus

## Test Traceability

- Each step is covered by unit/integration tests
- Example: question → forecast → justification → output

## ForecastChain from JSON → forecast dict

### Input Example

```json
{
  "question_id": 1,
  "question_text": "Will it rain tomorrow?"
}
```

### Output Example

```json
{
  "question_id": 1,
  "forecast": 0.88,
  "justification": "Based on evidence."
}
```

### Reasoning Explanation

- Extracts question details
- Gathers evidence using SearchTool
- Builds a CoT prompt
- Invokes LLM
- Parses and returns forecast dict

### Related Tests

- `tests/unit/agent/chains/test_forecast_chain.py`
- `tests/integration/test_agent_full_path.py`

## End-to-end: question → CoT → submission

### Input Example

```json
{
  "question_id": 123,
  "question_text": "Will it rain tomorrow?"
}
```

### Output Example (after submission)

```json
{
  "status": "success",
  "question_id": 123,
  "response": { "result": "ok" }
}
```

### Reasoning Explanation

- ForecastChain produces forecast dict
- MetaculusClient submits forecast
- Handles errors and returns status

### Related Tests

- `tests/unit/api/test_metaculus_client.py`

## Multi-Choice (MC) Example

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

### End-to-end: MC question → CoT → submission

- ForecastChain produces MC forecast dict
- MetaculusClient submits MC forecast (as list)
- Handles errors and returns status

### Related Tests

- `tests/unit/agent/chains/test_forecast_chain.py` (MC)
- `tests/unit/api/test_metaculus_client.py` (MC)
- `tests/integration/test_submission_pipeline.py` (MC)

## Example: Tool-Augmented Reasoning

Question: "What is quantum computing?"

- ForecastChain detects Wikipedia entity
- WikipediaTool fetches summary
- LLM uses summary in justification

Output sample:

```json
{
  "question_id": 1,
  "forecast": 0.5,
  "justification": "Based on Wikipedia's summary of quantum computing..."
}
```

## Example: Forecast with PluginTool

- PluginTool loaded from plugins directory
- Called during reasoning, result appears in trace

```json
{
  "question_id": 123,
  "forecast": 0.7,
  "justification": "Based on plugin output...",
  "trace": [
    {
      "step": 3,
      "type": "tool",
      "input": { "tool": "MyPlugin", "input": "..." },
      "output": "Plugin result",
      "timestamp": "..."
    }
  ]
}
```

## Example: Plugin Lifecycle Hooks

- PluginTool can tag, filter, or log before/after forecast
- Hooks are logged in the trace for auditability

```json
{
  "step": 1,
  "type": "plugin_pre_forecast",
  "input": {"tool": "TaggingPlugin", "input": {"question_id": 1, ...}},
  "output": {"question_id": 1, "question_text": "...", "tag": "science"},
  "timestamp": "..."
}
{
  "step": 7,
  "type": "plugin_post_submit",
  "input": {"tool": "TaggingPlugin", "input": {"status": "ok"}},
  "output": null,
  "timestamp": "..."
}
```

## Example: Plugin Lifecycle Trace

```json
{
  "step": 1,
  "type": "plugin_pre_forecast",
  "input": {"tool": "PrePlugin", "input": {"question_id": 1, "question_text": "foo"}},
  "output": "pre_hook_called",
  "timestamp": "..."
}
{
  "step": 8,
  "type": "plugin_post_submit",
  "input": {"tool": "PostPlugin", "input": {"status": "ok"}},
  "output": "post_hook_called",
  "timestamp": "..."
}
```

### Example Use Cases

- Tagging questions for downstream routing
- Logging submissions to a file or webhook
- Alerting on high-impact forecasts

## CLI Example: Model Routing

To run the agent with a specific LLM model (e.g. GPT-4) and correct import path:

```bash
PYTHONPATH=$(pwd) poetry run python main_agent.py --model openai/gpt-4 --mode batch --dryrun
```

- The `--model` flag controls which LLM backend is used for reasoning.
- See `forecast_pipeline.instructions.md` for more details on model selection and import requirements.
