---
applyTo: "**/application/*.py"
---

## ⚙️ Application Layer – Agent Orchestration Logic

- This layer coordinates LangChain chains or LangGraph flows.
- Orchestrate agent actions: search, predict, explain.
- Inject dependencies via constructors.
- No direct API or LLM calls here — only orchestrate tools.

### Main Classes
- `AgentOrchestrator`: Selects and runs agent strategies
- `InputRouter`: Maps questions to agent workflows
- `PromptContextBuilder`: Assembles input prompts

### Patterns
- Call LangChain tools via injected wrappers
- Use `LangGraph` or custom FSM-style orchestration

### Test Strategy
- Unit test routing logic and prompt construction
- Use mocks for LangChain components

### Commit Tags
- `feat(app): initial agent orchestrator`
- `test(app): routing for binary vs numeric`