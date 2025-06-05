# test_search.py
# Unit tests for SearchTool

import pytest
from src.agents.search import SearchTool

def test_search_stub():
    tool = SearchTool()
    query = "What is the weather forecast for London?"
    result = tool.search(query)
    assert "Stubbed evidence for" in result
    assert query in result
