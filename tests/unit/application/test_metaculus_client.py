# test_metaculus_client.py
# Unit test for MetaculusClient stub

from src.agents.metaculus_client import MetaculusClient

def test_submit_forecast_stub(capsys):
    client = MetaculusClient(api_key="TEST_KEY")
    result = client.submit_forecast(123, 0.5, "Test justification")
    assert result["status"] == "success"
    assert result["question_id"] == 123
    assert result["forecast"] == 0.5
    captured = capsys.readouterr()
    assert "Submitting forecast" in captured.out
    assert "Justification" in captured.out
