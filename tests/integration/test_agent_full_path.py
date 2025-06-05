# test_agent_full_path.py
# Integration test: ForecastAgent end-to-end with mocks
import pytest
from unittest.mock import Mock
from src.agents.forecast_agent import ForecastAgent
from src.agents.chains.forecast_chain import ForecastChain
from src.agents.search import SearchTool

@pytest.fixture
def mock_search():
    tool = Mock(spec=SearchTool)
    tool.search.return_value = "Evidence for: integration test (mocked)"
    return tool

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.invoke.return_value = {'forecast': 0.99, 'justification': 'Integration justified.'}
    return llm

def test_forecast_agent_predict(monkeypatch, mock_llm, mock_search):
    # Patch ForecastAgent to use our chain with mocks
    def chain_factory(llm=None, search_tool=None, tools=None):
        return ForecastChain(mock_llm, mock_search, tools=tools)
    monkeypatch.setattr('src.agents.forecast_agent.ForecastChain', chain_factory)
    agent = ForecastAgent()
    question = {'question_id': 42, 'question_text': 'Will AGI arrive by 2030?'}
    result = agent.invoke(question)
    assert isinstance(result, dict)
    assert 'question_id' in result
    assert 'forecast' in result or 'prediction' in result
    assert 'justification' in result
    assert 'trace' in result
    # Check trace structure
    assert isinstance(result['trace'], list)
    assert any(step['type'] == 'input' for step in result['trace'])
    assert any(step['type'] == 'llm' for step in result['trace'])
