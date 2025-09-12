"""Minimal OpenRouter probe to test auth, connectivity, and headers.

Run: python -m diagnostics.openrouter_probe "hello"
"""
from __future__ import annotations
import os
import sys
import json
import time
import asyncio
import httpx

BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
ENDPOINT = f"{BASE_URL.rstrip('/')}/chat/completions"
MODEL = os.getenv("OPENROUTER_PROBE_MODEL", "openrouter/auto")
PROMPT = " ".join(sys.argv[1:]) or "ping"

async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/metac-bot-ha"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Metaculus Forecasting Bot HA"),
    }
    payload = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "temperature": 0.0, "max_tokens": 8}
    async with httpx.AsyncClient(timeout=30.0) as client:
        t0 = time.time()
        try:
            resp = await client.post(ENDPOINT, headers=headers, json=payload)
        except Exception as e:
            print(f"NETWORK_ERROR: {e}")
            return
        dt = (time.time() - t0) * 1000
        print(f"status={resp.status_code} latency_ms={dt:.1f}")
        for k, v in resp.headers.items():
            lk = k.lower()
            if lk.startswith("x-ratelimit") or lk in ("retry-after",):
                print(f"header:{k}={v}")
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:400]}
        snippet = json.dumps(data)[:500]
        print(f"body_snippet={snippet}")
        if resp.status_code == 401:
            print("AUTH_FAILURE: Check API key")
        elif resp.status_code == 429:
            print("RATE_LIMIT: back off")
        elif resp.status_code >= 500:
            print("SERVER_ERROR: transient")

if __name__ == "__main__":
    asyncio.run(main())
