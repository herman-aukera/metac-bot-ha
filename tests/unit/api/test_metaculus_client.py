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

def test_submit_forecast_success(client):
    with patch.object(client.session, 'post') as mock_post:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_resp
        result = client.submit_forecast(123, 0.7, "Test justification")
        assert result["status"] == "success"
        assert result["question_id"] == 123

def test_submit_forecast_auth_fail(client):
    with patch.object(client.session, 'post') as mock_post:
        mock_resp = Mock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.return_value = mock_resp
        result = client.submit_forecast(123, 0.7, "Test justification")
        assert result["status"] == "error"
        assert result["code"] == 401

def test_submit_forecast_invalid_payload(client):
    result = client.submit_forecast(123, 1.2, "Test justification")
    assert result["status"] == "error"
    assert "Invalid payload" in result["error"]

def test_submit_forecast_timeout(client):
    with patch.object(client.session, 'post', side_effect=Exception("Timeout")):
        result = client.submit_forecast(123, 0.5, "Test justification")
        assert result["status"] == "error"
        assert "Timeout" in result["error"]

def test_submit_forecast_empty_justification(client):
    with patch.object(client.session, 'post') as mock_post:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_resp
        result = client.submit_forecast(123, 0.5, "")
        assert result["status"] == "success"

def test_submit_forecast_probability_boundaries(client):
    with patch.object(client.session, 'post') as mock_post:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_resp
        for prob in [0.0, 1.0]:
            result = client.submit_forecast(123, prob, "Boundary test")
            assert result["status"] == "success"
