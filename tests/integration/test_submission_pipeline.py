# test_submission_pipeline.py
# Integration test: ForecastChain → MetaculusClient submission
import pytest
from unittest.mock import patch, Mock
from src.agents.llm import MockLLM
from src.agents.search import SearchTool
from src.agents.chains.forecast_chain import ForecastChain
from src.api.metaculus_client import MetaculusClient

def test_submission_pipeline():
    llm = MockLLM()
    search = SearchTool()
    chain = ForecastChain(llm, search)
    forecast = chain.run({'question_id': 101, 'question_text': 'Will it rain?'})
    client = MetaculusClient(token="FAKE_TOKEN")
    with patch.object(client.session, 'post') as mock_post:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_resp
        result = client.submit_forecast(forecast['question_id'], forecast['forecast'], forecast['justification'])
        assert result['status'] == 'success'
        assert result['question_id'] == 101
        # Confirm payload
        args, kwargs = mock_post.call_args
        assert 'predict' in args[0]
        assert kwargs['json']['value'] == forecast['forecast']
