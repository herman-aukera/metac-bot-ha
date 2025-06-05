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
    monkeypatch.setattr('src.agents.forecast_agent.ForecastChain', lambda: ForecastChain(mock_llm, mock_search))
    agent = ForecastAgent()
    question = {'question_id': 42, 'question_text': 'Will AGI arrive by 2030?'}
    result = agent.invoke(question)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'question_id', 'forecast', 'justification'}
    assert result['forecast'] == 0.99
    assert 'Integration justified' in result['justification']
