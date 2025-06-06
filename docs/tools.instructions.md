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

## PluginTool System

### Interface

A PluginTool must implement:

- `name: str`
- `description: str`
- `invoke(input: str) -> str`

### Dynamic Loading

- Plugins are loaded from a directory via `load_plugins(plugin_dir)`
- Webhook plugins can be registered by subclassing `WebhookPluginTool`
- All tools are registered into the `tool_list` and injected into `ForecastChain`

### Example PluginTool

```python
from src.agents.tools.plugin_loader import PluginTool
class MyPlugin(PluginTool):
    name = "MyPlugin"
    description = "Returns a canned response."
    def invoke(self, input_str):
        return f"Hello from plugin: {input_str}"
```

### Webhook Plugin Example

```json
{
  "name": "MyWebhook",
  "url": "http://localhost:8000/hook",
  "description": "Demo webhook tool"
}
```

### Test Strategy

- Unit: test plugin loading, schema, and error handling
- Integration: run ForecastChain with a plugin, check trace
- CI: all plugins can be mocked, no real network required

## Plugin Lifecycle Hooks

Plugins can optionally implement lifecycle hooks:

- `pre_forecast(question_dict) -> Optional[question_dict]`: Called before reasoning. Can mutate/annotate the question (e.g., add tags, filter, log).
- `post_submit(submission_response) -> None`: Called after forecast submission. Can log, alert, or trigger side effects (e.g., Discord webhook).

### Example Plugin with Hooks

```python
from src.agents.tools.plugin_loader import PluginTool
class TaggingPlugin(PluginTool):
    name = "TaggingPlugin"
    def pre_forecast(self, q):
        return {**q, "tag": "science"}
    def post_submit(self, resp):
        print(f"Submitted: {resp}")
```

### Trace Format for Hooks

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

### Use Cases

- Tagging, annotation, or filtering questions before reasoning
- Logging or alerting after submission (e.g., Discord, Slack, file log)
- Auditing plugin activity in the trace
