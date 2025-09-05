"""Helper factory to create GeneralLlm instances with correct provider normalization
and OpenRouter routing across the codebase.

This centralizes:
- Provider-prefixed model normalization (e.g., gpt-5 -> openai/gpt-5)
- OpenRouter base_url + attribution headers
- Proxy handling for metaculus/* models (no API key)

Usage: prefer create_llm(...) instead of instantiating GeneralLlm directly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, TYPE_CHECKING

# Avoid importing forecasting_tools at module import time to prevent heavy transitive
# imports (e.g., streamlit) during pytest collection. Import inside functions.
if TYPE_CHECKING:  # pragma: no cover - type checking only
    from forecasting_tools import GeneralLlm  # noqa: F401


_OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def _openrouter_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = referer
    if title := os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-Title"] = title
    return headers


def normalize_model_id(model_name: str) -> str:
    """Ensure model IDs are provider-prefixed for litellm/OpenRouter.

    Examples:
    - gpt-5-mini -> openai/gpt-5-mini
    - gpt-5:floor -> openai/gpt-5:floor
    - claude-3-5-sonnet -> anthropic/claude-3-5-sonnet
    - kimi-k2:free -> moonshotai/kimi-k2:free
    - perplexity/sonar-pro -> perplexity/sonar-pro (unchanged)
    - metaculus/* -> keep as-is (proxy decides concrete model)
    """
    if not model_name:
        return model_name
    if "/" in model_name or model_name.startswith("metaculus/"):
        return model_name

    base, suffix = (model_name.split(":", 1) + [""])[:2]
    lower = base.lower()

    if lower.startswith("gpt-5") or lower.startswith("gpt-oss") or lower.startswith("gpt-4o"):
        base = f"openai/{base}"
    elif lower.startswith("claude"):
        base = f"anthropic/{base}"
    elif lower.startswith("kimi") or lower.startswith("k2"):
        base = f"moonshotai/{base}"

    return f"{base}{(':' + suffix) if suffix else ''}"


def create_llm(
    model: str,
    *,
    temperature: float = 0.1,
    timeout: int | float = 60,
    allowed_tries: int = 2,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a GeneralLlm with correct OpenRouter wiring and normalization.

    - metaculus/* models: no API key, no base_url
    - all other provider-prefixed models (openai/*, anthropic/*, moonshotai/*, perplexity/*):
            route via OpenRouter with base_url and attribution headers using OPENROUTER_API_KEY.

        Notes / rationale:
        - LiteLLM + OpenRouter: prefer setting custom_llm_provider='openrouter' so models like
            'moonshotai/kimi-k2:free' or 'openai/gpt-oss-120b:free' don’t trigger
            "LLM Provider NOT provided" (see LiteLLM OpenRouter docs: https://docs.litellm.ai/docs/providers/openrouter).
        - OpenRouter attribution headers per Quickstart/API docs: HTTP-Referer and X-Title
            (https://openrouter.ai/docs/quickstart, https://openrouter.ai/docs/app-attribution).
    """
    model = normalize_model_id(model)
    extra_kwargs = extra_kwargs or {}

    # Import here to avoid heavy import cost at module import time
    from forecasting_tools import GeneralLlm  # type: ignore

    if model.startswith("metaculus/"):
        return GeneralLlm(
            model=model,
            api_key=None,
            temperature=temperature,
            timeout=timeout,
            allowed_tries=allowed_tries,
            **extra_kwargs,
        )

    # Route through OpenRouter by default for supported providers
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key or openrouter_key.startswith("dummy_"):
        # Create without API key/base_url so callers can still fail gracefully
        return GeneralLlm(
            model=model,
            temperature=temperature,
            timeout=timeout,
            allowed_tries=allowed_tries,
            **extra_kwargs,
        )

    # Ensure LiteLLM treats this as an OpenRouter call path; avoid provider ambiguity.
    # We keep the provider-aware model name (e.g., openai/gpt-oss-120b:free) and set
    # custom_llm_provider to 'openrouter' so routing + auth + headers are correct.
    openrouter_kwargs = {**extra_kwargs}
    openrouter_kwargs.setdefault("custom_llm_provider", "openrouter")

    return GeneralLlm(
        model=model,
        api_key=openrouter_key,
        base_url=_OPENROUTER_BASE_URL,
        extra_headers=_openrouter_headers(),
        temperature=temperature,
        timeout=timeout,
        allowed_tries=allowed_tries,
        **openrouter_kwargs,
    )


def create_perplexity_llm(use_openrouter: bool = True) -> Any:
    """Create a Perplexity LLM, preferring OpenRouter routing for consistency.

    If use_openrouter is False, this will still return a Perplexity model but without
    OpenRouter base_url. LiteLLM can pick up PERPLEXITY_API_KEY from env if present.
    """
    model = "perplexity/sonar-reasoning" if use_openrouter else "perplexity/sonar-pro"
    # Prefer OpenRouter path to avoid provider ambiguity
    return create_llm(model, temperature=0.1, timeout=60, allowed_tries=2)
