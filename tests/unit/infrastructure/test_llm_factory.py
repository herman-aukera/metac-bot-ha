import os

import pytest

from src.infrastructure.config.llm_factory import normalize_model_id


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("gpt-5", "openai/gpt-5"),
        ("gpt-5:floor", "openai/gpt-5:floor"),
        ("gpt-5-mini", "openai/gpt-5-mini"),
        ("gpt-4o-mini", "openai/gpt-4o-mini"),
        ("claude-3-5-sonnet", "anthropic/claude-3-5-sonnet"),
        ("kimi-k2:free", "moonshotai/kimi-k2:free"),
        ("perplexity/sonar-pro", "perplexity/sonar-pro"),
        ("metaculus/gpt-4o-mini", "metaculus/gpt-4o-mini"),
    ],
)
def test_normalize_model_id(inp, expected):
    assert normalize_model_id(inp) == expected
