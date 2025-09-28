"""Tests for the Dispatcher application service."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from src.application.dispatcher import (
    Dispatcher,
    DispatcherConfig,
    DispatcherError,
    DispatcherStats,
)
from src.application.ingestion_service import IngestionStats, ValidationLevel
from src.domain.entities.forecast import Forecast
from src.domain.entities.question import Question, QuestionStatus, QuestionType
from src.infrastructure.metaculus_api import APIConfig, MetaculusAPIError


@pytest.fixture
def dispatcher_config():
    """Create a dispatcher configuration."""
    return DispatcherConfig(
        batch_size=2,
        validation_level=ValidationLevel.LENIENT,
        max_retries=2,
        enable_dry_run=False,
    )


@pytest.fixture
def api_config():
    """Create API configuration."""
    return APIConfig(base_url="https://test.metaculus.com")


@pytest.fixture
def sample_raw_questions():
    """Create sample raw question data from API."""
    return [
        {
            "id": 101,
            "title": "Will AI achieve AGI by 2030?",
            "description": "Question about AGI timeline",
            "type": "binary",
            "status": "open",
            "url": "https://example.com/question/101",
            "close_time": "2030-01-01T00:00:00Z",
            "resolve_time": "2030-01-01T00:00:00Z",
            "categories": ["ai", "technology"],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": 102,
            "title": "What will be the global temperature anomaly in 2025?",
            "description": "Question about climate",
            "type": "numeric",
            "status": "open",
            "url": "https://example.com/question/102",
            "close_time": "2025-01-01T00:00:00Z",
            "resolve_time": "2025-01-01T00:00:00Z",
            "categories": ["climate"],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "min_value": 0.0,
            "max_value": 5.0,
        },
    ]


@pytest.fixture
def sample_questions():
    """Create sample Question domain objects."""
    return [
        Question(
            id=uuid4(),
            metaculus_id=101,
            title="Will AI achieve AGI by 2030?",
            description="Question about AGI timeline",
            question_type=QuestionType.BINARY,
            status=QuestionStatus.OPEN,
            url="https://example.com/question/101",
            close_time=datetime(2030, 1, 1, tzinfo=timezone.utc),
            resolve_time=datetime(2030, 1, 1, tzinfo=timezone.utc),
            categories=["ai", "technology"],
            metadata={},
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        Question(
            id=uuid4(),
            metaculus_id=102,
            title="What will be the global temperature anomaly in 2025?",
            description="Question about climate",
            question_type=QuestionType.NUMERIC,
            status=QuestionStatus.OPEN,
            url="https://example.com/question/102",
            close_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            resolve_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            categories=["climate"],
            metadata={},
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            min_value=-2.0,
            max_value=5.0,
        ),
    ]


@pytest.fixture
def sample_forecasts():
    """Create sample forecasts."""
    return [Mock(spec=Forecast), Mock(spec=Forecast)]


class TestDispatcherConfig:
    """Test DispatcherConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DispatcherConfig()

        assert config.batch_size == 10
        assert config.max_retries == 3
        assert config.validation_level == ValidationLevel.LENIENT
        assert config.enable_dry_run is False
        assert config.api_config is None

    def test_custom_config(self):
        """Test custom configuration values."""
        api_config = APIConfig(base_url="https://test.metaculus.com")
        config = DispatcherConfig(
            batch_size=5,
            max_retries=2,
            validation_level=ValidationLevel.STRICT,
            enable_dry_run=True,
            api_config=api_config,
        )

        assert config.batch_size == 5
        assert config.max_retries == 2
        assert config.validation_level == ValidationLevel.STRICT
        assert config.enable_dry_run is True
        assert config.api_config == api_config


class TestDispatcherStats:
    """Test DispatcherStats class."""

    def test_initial_stats(self):
        """Test initial stats values."""
        stats = DispatcherStats()

        assert stats.total_questions_fetched == 0
        assert stats.questions_successfully_parsed == 0
        assert stats.questions_failed_parsing == 0
        assert stats.forecasts_generated == 0
        assert stats.forecasts_failed == 0
        assert stats.total_processing_time_seconds == 0.0
        assert stats.errors == []

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = DispatcherStats()

        # Test with no questions
        assert stats.success_rate == 0.0

        # Test with questions
        stats.total_questions_fetched = 10
        stats.forecasts_generated = 7
        assert stats.success_rate == 70.0

        # Test perfect success
        stats.total_questions_fetched = 5
        stats.forecasts_generated = 5
        assert stats.success_rate == 100.0


class TestDispatcher:
    """Test Dispatcher class."""

    def test_init_default_config(self):
        """Test dispatcher initialization with default config."""
        with (
            patch("src.application.dispatcher.MetaculusAPI"),
            patch("src.application.dispatcher.IngestionService"),
            patch("src.application.dispatcher.ForecastService"),
        ):
            dispatcher = Dispatcher()

            assert isinstance(dispatcher.config, DispatcherConfig)
            assert isinstance(dispatcher.stats, DispatcherStats)

    def test_init_with_config(self, dispatcher_config):
        """Test dispatcher initialization with custom config."""
        with (
            patch("src.application.dispatcher.MetaculusAPI"),
            patch("src.application.dispatcher.IngestionService"),
            patch("src.application.dispatcher.ForecastService"),
        ):
            dispatcher = Dispatcher(config=dispatcher_config)

            assert dispatcher.config == dispatcher_config
            assert isinstance(dispatcher.stats, DispatcherStats)

    @patch("src.application.dispatcher.MetaculusAPI")
    @patch("src.application.dispatcher.IngestionService")
    @patch("src.application.dispatcher.ForecastService")
    def test_run_success(
        self,
        mock_forecast_service_cls,
        mock_ingestion_service_cls,
        mock_api_cls,
        sample_raw_questions,
        sample_questions,
        sample_forecasts,
    ):
        """Test successful run of the dispatcher."""
        # Setup mocks
        mock_api = Mock()
        mock_api.fetch_questions.return_value = sample_raw_questions
        mock_api_cls.return_value = mock_api

        mock_ingestion_service = Mock()
        mock_ingestion_stats = IngestionStats(
            successful_parsed=len(sample_questions), failed_parsing=0
        )
        mock_ingestion_service.parse_questions.return_value = (
            sample_questions,
            mock_ingestion_stats,
        )
        mock_ingestion_service_cls.return_value = mock_ingestion_service

        mock_forecast_service = Mock()
        mock_forecast_service.generate_forecast.side_effect = sample_forecasts
        mock_forecast_service_cls.return_value = mock_forecast_service

        # Create dispatcher and run
        dispatcher = Dispatcher()
        forecasts, stats = dispatcher.run(limit=2, status="open")

        # Verify results
        assert len(forecasts) == 2
        assert stats.total_questions_fetched == 2
        assert stats.questions_successfully_parsed == 2
        assert stats.questions_failed_parsing == 0
        assert stats.forecasts_generated == 2
        assert stats.forecasts_failed == 0
        assert stats.success_rate == 100.0

        # Verify method calls
        mock_api.fetch_questions.assert_called_once_with(
            limit=2, status="open", category=None
        )
        mock_ingestion_service.parse_questions.assert_called_once_with(
            sample_raw_questions
        )
        assert mock_forecast_service.generate_forecast.call_count == 2

    @patch("src.application.dispatcher.MetaculusAPI")
    @patch("src.application.dispatcher.IngestionService")
    @patch("src.application.dispatcher.ForecastService")
    def test_run_api_error(
        self, mock_forecast_service_cls, mock_ingestion_service_cls, mock_api_cls
    ):
        """Test run with API error."""
        # Setup mocks
        mock_api = Mock()
        mock_api.fetch_questions.side_effect = MetaculusAPIError("API Error")
        mock_api_cls.return_value = mock_api

        mock_ingestion_service_cls.return_value = Mock()
        mock_forecast_service_cls.return_value = Mock()

        # Create dispatcher and run
        dispatcher = Dispatcher()
        forecasts, stats = dispatcher.run(limit=2)

        # Verify results
        assert len(forecasts) == 0
        assert stats.total_questions_fetched == 0
        assert len(stats.errors) > 0
        assert "Failed to fetch questions" in stats.errors[0]

    @patch("src.application.dispatcher.MetaculusAPI")
    @patch("src.application.dispatcher.IngestionService")
    @patch("src.application.dispatcher.ForecastService")
    def test_run_dry_run_mode(
        self,
        mock_forecast_service_cls,
        mock_ingestion_service_cls,
        mock_api_cls,
        sample_raw_questions,
        sample_questions,
    ):
        """Test run in dry run mode."""
        # Setup mocks
        mock_api = Mock()
        mock_api.fetch_questions.return_value = sample_raw_questions
        mock_api_cls.return_value = mock_api

        mock_ingestion_service = Mock()
        mock_ingestion_stats = IngestionStats(
            successful_parsed=len(sample_questions), failed_parsing=0
        )
        mock_ingestion_service.parse_questions.return_value = (
            sample_questions,
            mock_ingestion_stats,
        )
        mock_ingestion_service_cls.return_value = mock_ingestion_service

        mock_forecast_service = Mock()
        mock_forecast_service_cls.return_value = mock_forecast_service

        # Create dispatcher with dry run enabled
        config = DispatcherConfig(enable_dry_run=True)
        dispatcher = Dispatcher(config=config)
        forecasts, stats = dispatcher.run(limit=2)

        # Verify no forecasts generated in dry run
        assert len(forecasts) == 0
        assert stats.forecasts_generated == 0
        assert mock_forecast_service.generate_forecast.call_count == 0

    @patch("src.application.dispatcher.MetaculusAPI")
    @patch("src.application.dispatcher.IngestionService")
    @patch("src.application.dispatcher.ForecastService")
    def test_run_forecast_failures(
        self,
        mock_forecast_service_cls,
        mock_ingestion_service_cls,
        mock_api_cls,
        sample_raw_questions,
        sample_questions,
    ):
        """Test run with some forecast failures."""
        # Setup mocks
        mock_api = Mock()
        mock_api.fetch_questions.return_value = sample_raw_questions
        mock_api_cls.return_value = mock_api

        mock_ingestion_service = Mock()
        mock_ingestion_stats = IngestionStats(
            successful_parsed=len(sample_questions), failed_parsing=0
        )
        mock_ingestion_service.parse_questions.return_value = (
            sample_questions,
            mock_ingestion_stats,
        )
        mock_ingestion_service_cls.return_value = mock_ingestion_service

        mock_forecast_service = Mock()
        # First call succeeds, second fails
        mock_forecast = Mock(spec=Forecast)
        mock_forecast_service.generate_forecast.side_effect = [
            mock_forecast,
            Exception("Forecast error"),
        ]
        mock_forecast_service_cls.return_value = mock_forecast_service

        # Create dispatcher and run
        dispatcher = Dispatcher()
        forecasts, stats = dispatcher.run(limit=2)

        # Verify results
        assert len(forecasts) == 1
        assert stats.forecasts_generated == 1
        assert stats.forecasts_failed == 1
        assert len(stats.errors) > 0

    @patch("src.application.dispatcher.MetaculusAPI")
    @patch("src.application.dispatcher.IngestionService")
    @patch("src.application.dispatcher.ForecastService")
    def test_run_batch(
        self,
        mock_forecast_service_cls,
        mock_ingestion_service_cls,
        mock_api_cls,
        sample_raw_questions,
        sample_questions,
        sample_forecasts,
    ):
        """Test batch processing."""
        # Setup mocks
        mock_api = Mock()
        mock_api.fetch_questions.return_value = sample_raw_questions
        mock_api_cls.return_value = mock_api

        mock_ingestion_service = Mock()
        mock_ingestion_stats = IngestionStats(
            successful_parsed=len(sample_questions), failed_parsing=0
        )
        mock_ingestion_service.parse_questions.return_value = (
            sample_questions,
            mock_ingestion_stats,
        )
        mock_ingestion_service_cls.return_value = mock_ingestion_service

        mock_forecast_service = Mock()
        mock_forecast_service.generate_forecast.side_effect = (
            sample_forecasts * 2
        )  # For multiple batches
        mock_forecast_service_cls.return_value = mock_forecast_service

        # Create dispatcher with small batch size
        config = DispatcherConfig(batch_size=1)
        dispatcher = Dispatcher(config=config)

        # Run batch processing
        forecasts, stats = dispatcher.run_batch(total_limit=2)

        # Verify results - should have processed 2 batches
        assert len(forecasts) == 4  # 2 forecasts per batch * 2 batches
        assert stats.total_questions_fetched == 4
        assert mock_api.fetch_questions.call_count == 2

    def test_get_status(self):
        """Test getting dispatcher status."""
        with (
            patch("src.application.dispatcher.MetaculusAPI"),
            patch("src.application.dispatcher.IngestionService"),
            patch("src.application.dispatcher.ForecastService"),
        ):
            config = DispatcherConfig(batch_size=5, max_retries=2)
            dispatcher = Dispatcher(config=config)

            # Update some stats
            dispatcher.stats.total_questions_fetched = 10
            dispatcher.stats.forecasts_generated = 8

            status = dispatcher.get_status()

            # Verify status structure
            assert "config" in status
            assert "stats" in status
            assert "last_updated" in status

            assert status["config"]["batch_size"] == 5
            assert status["config"]["max_retries"] == 2
            assert status["stats"]["total_questions_fetched"] == 10
            assert status["stats"]["forecasts_generated"] == 8
            assert status["stats"]["success_rate"] == 80.0


class TestDispatcherError:
    """Test DispatcherError exception."""

    def test_dispatcher_error(self):
        """Test DispatcherError creation."""
        error = DispatcherError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
