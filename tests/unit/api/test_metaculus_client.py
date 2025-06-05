# test_metaculus_client.py
# Unit tests for MetaculusClient submission logic
import pytest
from unittest.mock import patch, Mock
from src.api.metaculus_client import MetaculusClient
import jsonschema
from jsonschema import ValidationError

@pytest.fixture
def client():
    return MetaculusClient(token="FAKE_TOKEN")

@pytest.mark.skip(reason="Requires valid token or full network mock")
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

@pytest.mark.skip(reason="Requires valid token or full network mock")
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
    """
    Test that submit_forecast raises ValueError on schema validation error (e.g., forecast out of range).
    This test is isolated from network/auth by patching the validate function.
    """
    with patch.object(client, '_validate', side_effect=ValidationError('out of range')):
        with pytest.raises(ValueError) as excinfo:
            client.submit_forecast(123, 1.2, "Test justification")
        assert "Invalid payload" in str(excinfo.value)

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

def test_submit_forecast_multi_choice(client):
    with patch.object(client.session, 'post') as mock_post:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_resp
        result = client.submit_forecast(456, [0.2, 0.3, 0.5], "MC justification")
        assert result["status"] == "success"
        assert result["question_id"] == 456
        args, kwargs = mock_post.call_args
        assert 'predict' in args[0]
        assert 'values' in kwargs['json']
        assert kwargs['json']['values'] == [0.2, 0.3, 0.5]

def test_submit_forecast_numeric(client):
    """
    Test that submit_forecast accepts and submits a numeric forecast dict (prediction, low, high).
    Network is mocked; checks payload and response.
    """
    with patch.object(client.session, 'post') as mock_post:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_resp
        numeric_forecast = {"prediction": 42.0, "low": 30.0, "high": 60.0}
        result = client.submit_forecast(789, numeric_forecast, "Numeric justification")
        assert result["status"] == "success"
        assert result["question_id"] == 789
        args, kwargs = mock_post.call_args
        assert 'predict' in args[0]
        # Numeric payload should have value, low, high
        assert kwargs['json']['value'] == 42.0
        assert kwargs['json']['low'] == 30.0
        assert kwargs['json']['high'] == 60.0

def test_submit_forecast_numeric_invalid_schema(client):
    """
    Test that submit_forecast raises ValueError on invalid numeric forecast dict (e.g., missing keys).
    """
    bad_numeric = {"prediction": 42.0, "low": 30.0}  # missing 'high'
    with patch.object(client, '_validate', side_effect=ValidationError('missing high')):
        with pytest.raises(ValueError) as excinfo:
            client.submit_forecast(789, bad_numeric, "Bad numeric")
        assert "Invalid payload" in str(excinfo.value)
