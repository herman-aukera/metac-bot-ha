# test_batch_executor.py
# Unit tests for batch_executor logic
import pytest
from unittest.mock import patch, Mock
from src.agents.batch_executor import run_batch

def test_run_batch_dryrun():
    questions = [
        {"question_id": 1, "question_text": "Q1"},
        {"question_id": 2, "question_text": "Q2"}
    ]
    with patch("src.agents.forecast_agent.ForecastAgent.invoke") as mock_invoke:
        mock_invoke.side_effect = [
            {"question_id": 1, "forecast": 0.5, "justification": "J1"},
            {"question_id": 2, "forecast": 0.7, "justification": "J2"}
        ]
        logs = run_batch(questions, submit=False)
        assert len(logs) == 2
        assert logs[0]["forecast"] == 0.5
        assert logs[1]["forecast"] == 0.7

import os
import contextlib

def test_run_batch_submit():
    questions = [{"question_id": 1, "question_text": "Q1"}]
    with patch("src.agents.forecast_agent.ForecastAgent.invoke") as mock_invoke, \
         patch("src.api.metaculus_client.MetaculusClient.submit_forecast") as mock_submit, \
         patch.dict(os.environ, {"METACULUS_TOKEN": "FAKE_TOKEN"}):
        mock_invoke.return_value = {"question_id": 1, "forecast": 0.5, "justification": "J1"}
        mock_submit.return_value = {"status": "success", "question_id": 1}
        logs = run_batch(questions, submit=True)
        assert logs[0]["submission"]["status"] == "success"
