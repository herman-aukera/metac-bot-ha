"""Central model policy helpers for GPT-5 enforcement.

Provides utilities to purge deprecated GPT-4o family references while allowing
tests to assert behavior deterministically. All code selecting model lists
should pass them through `enforce_model_policy` prior to use.
"""

from __future__ import annotations

from typing import Iterable, List
import os

DEPRECATED_MODELS = {
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "metaculus/gpt-4o",
    "metaculus/gpt-4o-mini",
}

GPT5_REPLACEMENTS_ORDER = [
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-oss-20b:free",
    "moonshotai/kimi-k2:free",
]


def enforce_model_policy(models: Iterable[str]) -> List[str]:
    """Return list with deprecated models removed & replaced.

    If a deprecated model is the first entry, first available replacement is
    inserted at its position (deduplicated). Controlled by env var
    GPT_MODEL_POLICY (default 'enforce'). Set to 'legacy' to bypass.
    """
    mode = os.getenv("GPT_MODEL_POLICY", "enforce").lower()
    out: List[str] = []
    for m in models:
        if mode != "enforce" or m not in DEPRECATED_MODELS:
            if m not in out:
                out.append(m)
        else:
            # Insert first viable replacement maintaining order semantics
            for repl in GPT5_REPLACEMENTS_ORDER:
                if repl not in out:
                    out.append(repl)
                    break
    # Ensure at least one GPT-5 tier present
    if mode == "enforce" and not any(x.startswith("openai/gpt-5") for x in out):
        out.insert(0, "openai/gpt-5-mini")
    return out


__all__ = ["DEPRECATED_MODELS", "enforce_model_policy"]
