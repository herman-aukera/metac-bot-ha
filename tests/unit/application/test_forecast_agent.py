# test_forecast_agent.py
# Unit and integration tests for ForecastAgent

import pytest
from src.agents.forecast_agent import ForecastAgent

@pytest.fixture
def sample_question():
    return {
        'question_id': 123,
        'question_text': 'Will it rain in London on July 1, 2025?'
    }

def test_forecast_agent_output(sample_question):
    agent = ForecastAgent()
    result = agent.invoke(sample_question)
    assert 'question_id' in result
    assert 'forecast' in result
    assert 'justification' in result
    assert isinstance(result['forecast'], float)
    assert isinstance(result['justification'], str)
    assert result['question_id'] == sample_question['question_id']

def test_forecast_agent_reasoning(sample_question):
    agent = ForecastAgent()
    result = agent.invoke(sample_question)
    # Accept either legacy stub or mock LLM output
    assert any(s in result['justification'] for s in ['Reasoning for', 'justification'])
