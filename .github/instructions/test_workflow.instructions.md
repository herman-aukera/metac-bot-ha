## ✅ Test Strategy

- Unit tests for:
  - Domain logic
  - Prompt composition
  - Dispatcher routing

- Integration tests for:
  - Agent chains
  - Prompt → LLM → forecast path

- Use `pytest`, `LangChainTester`, `mock.AsyncMock`

### Commit Tags
- `test(unit): dispatcher routing to tools`
- `test(integration): end-to-end with AskNews`