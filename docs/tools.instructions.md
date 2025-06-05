# tools.instructions.md

## Purpose

LangChain-compatible tools for the Metaculus Agentic Bot. Extend agent reasoning with Wikipedia and math capabilities.

## Tools

### WikipediaTool

- **Purpose:** Query Wikipedia for short, source-attributed summaries.
- **Input:** `{ "query": "Quantum computing" }`
- **Output:** `"[Wikipedia: Quantum computing] Quantum computing is..."`
- **Reasoning Logic:** Used for named entity background, fact-checking, and context.
- **Related Tests:** `tests/unit/agents/tools/test_wikipedia.py`

### MathTool

- **Purpose:** Evaluate safe math expressions for quick calculations.
- **Input:** `{ "expr": "sqrt(16) + 2" }`
- **Output:** `"6.0"`
- **Reasoning Logic:** Used for probability, statistics, and numeric reasoning.
- **Related Tests:** `tests/unit/agents/tools/test_math.py`

## Tool Routing

- Tools are available in `tool_list` and can be injected into chains or called from the agent.
- All tools are CI-safe and can be mocked in tests.

## Example CLI Output

```json
{
  "question": "What is the probability of X if Y?",
  "tools_used": ["WikipediaTool", "MathTool"],
  "justification": "Based on Wikipedia's summary of X and quick math calculation..."
}
```
