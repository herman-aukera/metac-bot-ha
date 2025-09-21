import pytest

import src.infrastructure.external_apis.llm_client as llm_mod
from src.infrastructure.config.settings import LLMConfig


class DummyResp:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.headers = {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class DummyClient:
    def __init__(self, responses):
        self._responses = responses
        self._i = 0

    async def post(self, *a, **kw):
        r = self._responses[min(self._i, len(self._responses)-1)]
        self._i += 1
        return r


@pytest.mark.asyncio
async def test_quota_circuit_breaker_trips_and_fast_fails(monkeypatch):
    # Reset module state
    llm_mod.OPENROUTER_QUOTA_EXCEEDED = False
    llm_mod.OPENROUTER_QUOTA_MESSAGE = None

    cfg = LLMConfig(provider="openrouter", model="openrouter/auto", api_key="k")
    client = llm_mod.LLMClient(cfg)
    # Inject dummy http client returning 403 with key limit message
    resp = DummyResp(403, {"error": {"message": "Key limit exceeded"}})
    client.client = DummyClient([resp])
    with pytest.raises(RuntimeError):
        await client._call_openrouter("ping", cfg.model, 0.0, 8)
    # Next call should fast-fail due to circuit
    with pytest.raises(RuntimeError):
        await client._call_openrouter("ping", cfg.model, 0.0, 8)
    assert llm_mod.OPENROUTER_QUOTA_EXCEEDED is True
