# test_forecast_chain.py
# Unit tests for ForecastChain logic

import pytest
from src.agents.chains.forecast_chain import ForecastChain

@pytest.fixture
def sample_question():
    return {
        'question_id': 456,
        'question_text': 'Will the S&P 500 close above 5000 by Dec 31, 2025?'
    }

def test_forecast_chain_output(sample_question):
    chain = ForecastChain()
    result = chain.run(sample_question)
    assert 'question_id' in result
    assert 'forecast' in result
    assert 'justification' in result
    assert isinstance(result['forecast'], float)
    assert isinstance(result['justification'], str)
    assert result['question_id'] == sample_question['question_id']

def test_forecast_chain_justification(sample_question):
    chain = ForecastChain()
    result = chain.run(sample_question)
    assert 'Reasoning for' in result['justification']
    assert 'Evidence for' in result['justification']
