import os
from unittest.mock import patch

import pytest

from src.infrastructure.config.budget_manager import BudgetManager
from src.infrastructure.external_apis.llm_client import LLMClient
from src.infrastructure.config.settings import LLMConfig


def test_free_models_zero_cost():
    bm = BudgetManager(budget_limit=1.0)
    assert abs(bm.estimate_cost("openai/gpt-oss-20b:free", 1000, 500) - 0.0) < 1e-9
    assert abs(bm.estimate_cost("moonshotai/kimi-k2:free", 2000, 1000) - 0.0) < 1e-9


@pytest.mark.asyncio
async def test_llm_client_openrouter_headers_from_env(monkeypatch):
    # Ensure env-based attribution headers are used
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.test/app")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Test App Title")

    cfg = LLMConfig(provider="openrouter", model="openai/gpt-oss-20b:free", api_key="sk-test")
    client = LLMClient(cfg)

    with patch.object(client.client, "post") as mock_post:
        mock_resp = type("R", (), {"raise_for_status": lambda self: None, "json": lambda self: {"choices": [{"message": {"content": "ok"}}]}})()
        mock_post.return_value = mock_resp

        out = await client.generate("ping", max_tokens=1)
        assert out == "ok"

    # Verify headers passed include env-derived attribution
    _, kwargs = mock_post.call_args
    headers = kwargs.get("headers", {})
    assert headers.get("HTTP-Referer") == "https://example.test/app"
    assert headers.get("X-Title") == "Test App Title"
