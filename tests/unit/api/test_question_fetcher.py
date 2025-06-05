# test_question_fetcher.py
# Unit tests for question_fetcher filtering logic
import pytest
from unittest.mock import patch, Mock
from src.api.question_fetcher import fetch_new_questions

def test_fetch_new_questions_filters_answered():
    fake_response = {
        "results": [
            {"id": 1, "title": "Q1", "user_prediction": None, "skipped": False},
            {"id": 2, "title": "Q2", "user_prediction": 0.5, "skipped": False},
            {"id": 3, "title": "Q3", "user_prediction": None, "skipped": True},
            {"id": 4, "title": "Q4", "user_prediction": None, "skipped": False}
        ],
        "next": None
    }
    with patch("requests.Session.get") as mock_get:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_get.return_value = mock_resp
        questions = fetch_new_questions(token="FAKE", limit=10)
        assert len(questions) == 2
        ids = [q["question_id"] for q in questions]
        assert 1 in ids and 4 in ids

def test_fetch_new_questions_includes_mc():
    fake_response = {
        "results": [
            {"id": 1, "title": "Q1", "user_prediction": None, "skipped": False, "type": "mc", "options": ["A", "B"]},
            {"id": 2, "title": "Q2", "user_prediction": None, "skipped": False, "type": "binary"}
        ],
        "next": None
    }
    with patch("requests.Session.get") as mock_get:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_get.return_value = mock_resp
        questions = fetch_new_questions(token="FAKE", limit=10)
        assert len(questions) == 2
        assert questions[0]["type"] == "mc"
        assert questions[0]["options"] == ["A", "B"]
        assert "type" not in questions[1] or questions[1]["type"] == "binary"
