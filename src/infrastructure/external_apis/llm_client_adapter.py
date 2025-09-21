"""Adapters to use the hardened LLMClient in places expecting a GeneralLlm-like interface.

This provides a minimal wrapper exposing an `.invoke(prompt)` coroutine and a
`.model` attribute so existing code paths (tri_model_router, validation pipeline)
can transparently call through our resilient OpenRouter client with backoff,
quota circuit breaker, and diagnostics.
"""

from __future__ import annotations

from typing import Any


from .llm_client import LLMClient


class HardenedOpenRouterModel:
    """Thin adapter that forwards `.invoke()` to LLMClient.generate().

    Attributes:
        model: Provider-prefixed model name (e.g. "openai/gpt-5-mini").
        temperature: Default temperature to use if not overridden.
        timeout: Unused here; kept for compatibility with GeneralLlm signature.
        allowed_tries: Unused here; internal client already handles retries/backoff.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        *,
        temperature: float = 0.1,
        timeout: int | float = 60,
        allowed_tries: int = 2,
    ) -> None:
        self._llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.allowed_tries = allowed_tries

    async def invoke(self, content: str, **kwargs: Any) -> str:
        """Invoke the underlying hardened client for a single prompt.

        Extra kwargs are forwarded to the OpenRouter payload when supported.
        """
        # LLMClient.generate routes by provider in its config; we always pass `model`
        # so per-call model choice is respected.
        return await self._llm_client.generate(
            prompt=content,
            model=self.model,
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )

    # Optional: convenience alias used in some code paths
    async def ainvoke(self, content: str, **kwargs: Any) -> str:  # pragma: no cover
        return await self.invoke(content, **kwargs)

    # Graceful close passthrough if caller tries to close models
    async def aclose(self) -> None:  # pragma: no cover - best effort
        try:
            if hasattr(self._llm_client, "client"):
                await self._llm_client.client.aclose()
        except Exception:
            pass

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"HardenedOpenRouterModel(model={self.model})"
