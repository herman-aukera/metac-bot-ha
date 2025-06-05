# test_integration_forecast_pipeline.py
# Integration test: question → forecast → output

import pytest
from src.agents.agent_runner import run_agent

@pytest.fixture
def sample_question():
    return {
        'question_id': 789,
        'question_text': 'Will a major hurricane make landfall in Florida in 2025?'
    }

def test_run_agent_dryrun(capsys, sample_question):
    result = run_agent(sample_question, dryrun=True)
    assert 'question_id' in result
    assert 'forecast' in result
    assert 'justification' in result
    assert isinstance(result['forecast'], float)
    assert isinstance(result['justification'], str)
    assert result['question_id'] == sample_question['question_id']
    # Check output to stdout
    captured = capsys.readouterr()
    assert 'forecast' in captured.out
