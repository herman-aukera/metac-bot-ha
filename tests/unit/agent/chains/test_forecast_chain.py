# test_forecast_chain.py
# Unit tests for ForecastChain (CoT + Evidence)
import pytest
from unittest.mock import Mock
from src.agents.chains.forecast_chain import ForecastChain
from src.agents.search import SearchTool

@pytest.fixture
def mock_search():
    tool = Mock(spec=SearchTool)
    tool.search.return_value = "Evidence for: test question (mocked)"
    return tool

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.invoke.return_value = {'forecast': 0.88, 'justification': 'Based on evidence.'}
    return llm

def test_run_binary_question(mock_llm, mock_search):
    chain = ForecastChain(llm=mock_llm, search_tool=mock_search)
    question = {'question_id': 1, 'question_text': 'Will it rain tomorrow?'}
    result = chain.run(question)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'question_id', 'forecast', 'justification'}
    assert result['forecast'] == 0.88
    assert 'Based on evidence' in result['justification']

def test_run_missing_input(mock_llm, mock_search):
    chain = ForecastChain(llm=mock_llm, search_tool=mock_search)
    result = chain.run({'question_text': 'Missing id'})
    assert 'error' in result
    result2 = chain.run({})
    assert 'error' in result2

def test_run_llm_json_string(mock_search):
    llm = Mock()
    llm.invoke.return_value = '{"forecast": 0.42, "justification": "LLM string output"}'
    chain = ForecastChain(llm=llm, search_tool=mock_search)
    question = {'question_id': 2, 'question_text': 'Will the stock go up?'}
    result = chain.run(question)
    assert result['forecast'] == 0.42
    assert 'LLM string output' in result['justification']
