# test_wikipedia.py
# Unit tests for WikipediaTool
import pytest
from src.agents.tools.wikipedia import WikipediaTool

class MockResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}
    def json(self):
        return self._json

def test_wikipedia_success(monkeypatch):
    def mock_get(url, timeout):
        return MockResponse(200, {"title": "Quantum computing", "extract": "Quantum computing is..."})
    monkeypatch.setattr("requests.get", mock_get)
    tool = WikipediaTool()
    result = tool.run("Quantum computing")
    assert "Quantum computing is" in result
    assert "Wikipedia: Quantum computing" in result

def test_wikipedia_no_result(monkeypatch):
    def mock_get(url, timeout):
        return MockResponse(404)
    monkeypatch.setattr("requests.get", mock_get)
    tool = WikipediaTool()
    result = tool.run("NonexistentPage12345")
    assert "No Wikipedia page found" in result

def test_wikipedia_ambiguous(monkeypatch):
    def mock_get(url, timeout):
        return MockResponse(200, {"title": "Mercury", "extract": None})
    monkeypatch.setattr("requests.get", mock_get)
    tool = WikipediaTool()
    result = tool.run("Mercury")
    assert "No summary available" in result
