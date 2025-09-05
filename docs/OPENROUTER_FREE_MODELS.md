OpenRouter routing and free models

- All third-party LLM calls go through OpenRouter (https://openrouter.ai/docs/quickstart). We set attribution headers via environment variables:
  - OPENROUTER_HTTP_REFERER
  - OPENROUTER_APP_TITLE
- Provider-prefixed model IDs are required, e.g.:
  - openai/gpt-oss-20b:free
  - moonshotai/kimi-k2:free
  - openai/gpt-5, openai/gpt-5-mini, openai/gpt-5-nano
- Free-tier models are treated as zero-cost in BudgetManager and TokenTracker.
- AskNews research is attempted first; free models synthesize only when AskNews is unavailable or empty. Perplexity is disabled by default and only allowed if feature-flagged.

References
- OpenRouter Quickstart: https://openrouter.ai/docs/quickstart
- LiteLLM OpenRouter provider: https://docs.litellm.ai/docs/providers/openrouter
