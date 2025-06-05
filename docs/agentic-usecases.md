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
