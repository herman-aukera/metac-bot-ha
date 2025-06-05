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

def test_run_multi_choice_question(mock_search):
    mock_llm = Mock()
    mock_llm.invoke.return_value = {'forecast': [0.1, 0.3, 0.6], 'justification': 'MC evidence.'}
    chain = ForecastChain(llm=mock_llm, search_tool=mock_search)
    question = {
        'question_id': 10,
        'question_text': 'Which city will win?',
        'type': 'mc',
        'options': ['London', 'Paris', 'Berlin']
    }
    result = chain.run(question)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'question_id', 'forecast', 'justification'}
    assert isinstance(result['forecast'], list)
    assert len(result['forecast']) == 3
    assert abs(sum(result['forecast']) - 1.0) < 1e-6 or sum(result['forecast']) <= 1.0
    assert 'MC evidence' in result['justification']

def test_run_numeric_question(mock_search):
    mock_llm = Mock()
    mock_llm.invoke.return_value = {
        'prediction': 42.0,
        'low': 30.0,
        'high': 60.0,
        'justification': 'Numeric evidence.'
    }
    chain = ForecastChain(llm=mock_llm, search_tool=mock_search)
    question = {
        'question_id': 20,
        'question_text': 'How many widgets will be sold?',
        'type': 'numeric'
    }
    result = chain.run(question)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'question_id', 'prediction', 'low', 'high', 'justification'}
    assert result['prediction'] == 42.0
    assert result['low'] == 30.0
    assert result['high'] == 60.0
    assert 'Numeric evidence' in result['justification']
