"""Integration test: ingestion → dispatcher → forecast pipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.infrastructure.metaculus_api import MetaculusAPI
from src.application.ingestion_service import IngestionService
from src.application.dispatcher import Dispatcher
from src.application.forecast_service import ForecastService
from src.domain.entities.question import QuestionType

def test_ingestion_dispatcher_forecast_pipeline(monkeypatch):
    # Mock MetaculusAPI to return a sample question JSON
    sample_json = {
        "id": 1,
        "title": "Will it rain tomorrow?",
        "question_type": "binary",
        "description": "A simple binary weather question.",
        "min_value": None,
        "max_value": None,
        "choices": None,
        "status": "open"
    }
    api = MagicMock()
    api.fetch_questions.return_value = [sample_json]

    # IngestionService parses JSON to Question
    ingestion = IngestionService()
    questions, _ = ingestion.parse_questions([sample_json])
    assert len(questions) == 1
    q = questions[0]
    assert q.title == "Will it rain tomorrow?"
    assert q.question_type == QuestionType.BINARY

    # ForecastService is mocked to always return a dummy forecast
    forecast_service = MagicMock()
    forecast_service.generate_forecast.return_value = "dummy-forecast"

    # Dispatcher routes the question to ForecastService
    dispatcher = Dispatcher(forecast_service)
    result = dispatcher.dispatch(q)
    assert result == "dummy-forecast"
