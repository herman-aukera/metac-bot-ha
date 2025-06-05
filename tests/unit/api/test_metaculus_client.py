# test_metaculus_client.py
# Unit tests for MetaculusClient submission logic
import pytest
from unittest.mock import patch, Mock
from src.api.metaculus_client import MetaculusClient

@pytest.fixture
def client():
    return MetaculusClient(token="FAKE_TOKEN")

def test_submit_success(client):
    with patch.object(client.session, 'post') as mock_post:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_resp
        forecast = {"question_id": 123, "forecast": 0.7, "justification": "Test"}
        result = client.submit(forecast)
        assert result["status"] == "success"
        assert result["question_id"] == 123

def test_submit_auth_fail(client):
    with patch.object(client.session, 'post') as mock_post:
        mock_resp = Mock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.return_value = mock_resp
        forecast = {"question_id": 123, "forecast": 0.7, "justification": "Test"}
        result = client.submit(forecast)
        assert result["status"] == "error"
        assert result["code"] == 401

def test_submit_malformed_forecast(client):
    # Missing forecast key
    result = client.submit({"question_id": 1})
    assert result["status"] == "error"
    assert "forecast" in result["error"] or "KeyError" in result["error"]
