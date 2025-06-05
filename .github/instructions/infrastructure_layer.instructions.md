---
applyTo: "**/infrastructure/*.py"
---

## 🌐 Infrastructure – Tools & LLM Adapters

- Implement LangChain tools, LangGraph chains
- Wrap LLMs, APIs, search (AskNews, Exa, Perplexity)

### Components
- `LLMTool`: A class that wraps one OpenAI/Claude call
- `SearchTool`: Wraps AskNews/Exa
- `ForecastingChain`: LangChain chain that does RAG + prompt

### Principles
- Follow Adapter pattern
- Fully mockable
- No domain logic here

### Commit Tags
- `feat(infra): wrap OpenAI search tool`
- `test(infra): mock AskNews client`