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
from src.application.forecast_service import ForecastService
from src.application.ingestion_service import (
    IngestionService,
    IngestionStats,
    ValidationLevel,
)
from src.domain.entities.forecast import Forecast
from src.domain.entities.question import Question, QuestionStatus, QuestionType
from src.infrastructure.metaculus_api import MetaculusAPI, MetaculusAPIError


@pytest.fixture
def sample_questions():
    """Create sample questions for testing."""
    return [
        Question(
            id=uuid4(),
            metaculus_id=101,
            title="Will AI achieve AGI by 2030?",
            description="Question about AGI timeline",
            question_type=QuestionType.BINARY,
            status=QuestionStatus.OPEN,
            url="https://example.com/question/101",
            close_time=datetime.now(timezone.utc),
            resolve_time=None,
            categories=["ai", "technology"],
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        Question(
            id=uuid4(),
            metaculus_id=102,
            title="What will be the global temperature anomaly in 2025?",
            description="Question about climate",
            question_type=QuestionType.NUMERIC,
            status=QuestionStatus.OPEN,
            url="https://example.com/question/102",
            close_time=datetime.now(timezone.utc),
            resolve_time=None,
            categories=["climate"],
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            min_value=0.0,
            max_value=5.0,
        ),
    ]


@pytest.fixture
def sample_forecast():
    """Create a sample forecast for testing."""
    forecast = Mock(spec=Forecast)
    forecast.id = uuid4()
    return forecast


class TestDispatcherConfig:
    """Test DispatcherConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DispatcherConfig()

        assert config.batch_size == 10
        assert config.validation_level == ValidationLevel.LENIENT
        assert config.max_retries == 3
        assert config.enable_dry_run is False
        assert config.api_config is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DispatcherConfig(
            batch_size=5,
            validation_level=ValidationLevel.STRICT,
            max_retries=2,
            enable_dry_run=True,
        )

        assert config.batch_size == 5
        assert config.validation_level == ValidationLevel.STRICT
        assert config.max_retries == 2
        assert config.enable_dry_run is True


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

        # Test zero case
        assert stats.success_rate == 0.0

        # Test with data
        stats.total_questions_fetched = 10
        stats.forecasts_generated = 7
        assert stats.success_rate == 70.0

    def test_stats_operations(self):
        """Test stats update operations."""
        stats = DispatcherStats()

        stats.total_questions_fetched = 5
        stats.questions_successfully_parsed = 4
        stats.questions_failed_parsing = 1
        stats.forecasts_generated = 3
        stats.forecasts_failed = 1
        stats.errors.append("Test error")
        stats.total_processing_time_seconds = 10.5

        assert stats.total_questions_fetched == 5
        assert stats.questions_successfully_parsed == 4
        assert stats.questions_failed_parsing == 1
        assert stats.forecasts_generated == 3
        assert stats.forecasts_failed == 1
        assert len(stats.errors) == 1
        assert stats.total_processing_time_seconds == 10.5


class TestDispatcher:
    """Test Dispatcher class."""

    def test_init_default(self):
        """Test dispatcher initialization with defaults."""
        dispatcher = Dispatcher()

        assert isinstance(dispatcher.config, DispatcherConfig)
        assert isinstance(dispatcher.api, MetaculusAPI)
        assert isinstance(dispatcher.ingestion_service, IngestionService)
        assert isinstance(dispatcher.forecast_service, ForecastService)
        assert isinstance(dispatcher.stats, DispatcherStats)

    def test_init_with_config(self):
        """Test dispatcher initialization with custom config."""
        config = DispatcherConfig(batch_size=5)
        dispatcher = Dispatcher(config=config)

        assert dispatcher.config == config
        assert dispatcher.config.batch_size == 5

    def test_fetch_questions_success(self):
        """Test successful question fetching."""
        dispatcher = Dispatcher()

        # Mock API response
        mock_questions_json = [
            {"id": 101, "title": "Test Question 1"},
            {"id": 102, "title": "Test Question 2"},
        ]

        with patch.object(
            dispatcher.api, "fetch_questions", return_value=mock_questions_json
        ):
            result = dispatcher._fetch_questions(limit=2, status="open", category=None)

            assert result == mock_questions_json
            assert len(result) == 2

    def test_fetch_questions_api_error(self):
        """Test question fetching with API error."""
        dispatcher = Dispatcher()

        with patch.object(
            dispatcher.api,
            "fetch_questions",
            side_effect=MetaculusAPIError("API Error"),
        ):
            result = dispatcher._fetch_questions(limit=2, status="open", category=None)

            assert result == []
            assert len(dispatcher.stats.errors) == 1
            assert "Failed to fetch questions" in dispatcher.stats.errors[0]

    def test_parse_questions_success(self, sample_questions):
        """Test successful question parsing."""
        dispatcher = Dispatcher()

        # Mock ingestion service
        mock_questions_json = [{"id": 101}, {"id": 102}]
        mock_ingestion_stats = IngestionStats(
            total_processed=2, successful_parsed=2, failed_parsing=0
        )

        with patch.object(
            dispatcher.ingestion_service,
            "parse_questions",
            return_value=(sample_questions, mock_ingestion_stats),
        ):
            result = dispatcher._parse_questions(mock_questions_json)

            assert result == sample_questions
            assert dispatcher.stats.questions_successfully_parsed == 2
            assert dispatcher.stats.questions_failed_parsing == 0

    def test_parse_questions_with_errors(self):
        """Test question parsing with some errors."""
        dispatcher = Dispatcher()

        mock_questions_json = [{"id": 101}, {"id": 102}]
        mock_ingestion_stats = IngestionStats(
            total_processed=2, successful_parsed=1, failed_parsing=1
        )

        with patch.object(
            dispatcher.ingestion_service,
            "parse_questions",
            return_value=([], mock_ingestion_stats),
        ):
            result = dispatcher._parse_questions(mock_questions_json)

            assert result == []
            assert dispatcher.stats.questions_successfully_parsed == 1
            assert dispatcher.stats.questions_failed_parsing == 1
            assert len(dispatcher.stats.errors) == 1

    def test_generate_forecasts_success(self, sample_questions, sample_forecast):
        """Test successful forecast generation."""
        dispatcher = Dispatcher()

        with patch.object(
            dispatcher.forecast_service,
            "generate_forecast",
            return_value=sample_forecast,
        ):
            result = dispatcher._generate_forecasts(sample_questions)

            assert len(result) == len(sample_questions)
            assert dispatcher.stats.forecasts_generated == len(sample_questions)
            assert dispatcher.stats.forecasts_failed == 0

    def test_generate_forecasts_with_errors(self, sample_questions):
        """Test forecast generation with some errors."""
        dispatcher = Dispatcher()

        with patch.object(
            dispatcher.forecast_service,
            "generate_forecast",
            side_effect=Exception("Forecast error"),
        ):
            result = dispatcher._generate_forecasts(sample_questions)

            assert len(result) == 0
            assert dispatcher.stats.forecasts_generated == 0
            assert dispatcher.stats.forecasts_failed == len(sample_questions)
            assert len(dispatcher.stats.errors) == len(sample_questions)

    def test_generate_forecasts_dry_run(self, sample_questions):
        """Test forecast generation in dry run mode."""
        config = DispatcherConfig(enable_dry_run=True)
        dispatcher = Dispatcher(config=config)

        result = dispatcher._generate_forecasts(sample_questions)

        assert len(result) == 0
        assert dispatcher.stats.forecasts_generated == 0
        assert dispatcher.stats.forecasts_failed == 0

    def test_run_success(self, sample_questions, sample_forecast):
        """Test successful end-to-end pipeline execution."""
        dispatcher = Dispatcher()

        # Mock all dependencies
        mock_questions_json = [{"id": 101}, {"id": 102}]
        mock_ingestion_stats = IngestionStats(
            total_processed=2, successful_parsed=2, failed_parsing=0
        )

        with (
            patch.object(
                dispatcher.api, "fetch_questions", return_value=mock_questions_json
            ),
            patch.object(
                dispatcher.ingestion_service,
                "parse_questions",
                return_value=(sample_questions, mock_ingestion_stats),
            ),
            patch.object(
                dispatcher.forecast_service,
                "generate_forecast",
                return_value=sample_forecast,
            ),
        ):
            forecasts, stats = dispatcher.run(limit=2, status="open", category=None)

            assert isinstance(stats, DispatcherStats)
            assert stats.total_questions_fetched == 2
            assert stats.questions_successfully_parsed == 2
            assert len(forecasts) == len(sample_questions)
            assert stats.total_processing_time_seconds > 0

    def test_run_no_questions_fetched(self):
        """Test run when no questions are fetched."""
        dispatcher = Dispatcher()

        with patch.object(dispatcher.api, "fetch_questions", return_value=[]):
            forecasts, stats = dispatcher.run(limit=10)

            assert len(forecasts) == 0
            assert stats.total_questions_fetched == 0

    def test_run_with_critical_error(self):
        """Test run with critical error."""
        dispatcher = Dispatcher()

        with patch.object(
            dispatcher.api, "fetch_questions", side_effect=Exception("Critical error")
        ):
            with pytest.raises(DispatcherError, match="Critical error in dispatcher"):
                dispatcher.run(limit=10)

    def test_run_batch_success(self, sample_questions, sample_forecast):
        """Test batch processing functionality."""
        config = DispatcherConfig(batch_size=1)
        dispatcher = Dispatcher(config=config)

        # Mock the run method to simulate batch processing
        mock_stats = DispatcherStats()
        mock_stats.total_questions_fetched = 1
        mock_stats.forecasts_generated = 1

        with patch.object(
            dispatcher, "run", return_value=([sample_forecast], mock_stats)
        ):
            forecasts, combined_stats = dispatcher.run_batch(
                total_limit=2, status="open"
            )

            assert len(forecasts) == 2  # Two batches
            assert combined_stats.total_questions_fetched == 2
            assert combined_stats.forecasts_generated == 2

    def test_merge_stats(self):
        """Test statistics merging."""
        dispatcher = Dispatcher()

        combined = DispatcherStats()
        batch = DispatcherStats()
        batch.total_questions_fetched = 5
        batch.questions_successfully_parsed = 4
        batch.forecasts_generated = 3
        batch.total_processing_time_seconds = 10.5
        batch.errors.append("Test error")

        dispatcher._merge_stats(combined, batch)

        assert combined.total_questions_fetched == 5
        assert combined.questions_successfully_parsed == 4
        assert combined.forecasts_generated == 3
        assert combined.total_processing_time_seconds == 10.5
        assert len(combined.errors) == 1

    def test_get_status(self):
        """Test getting dispatcher status."""
        config = DispatcherConfig(batch_size=5, enable_dry_run=True)
        dispatcher = Dispatcher(config=config)

        status = dispatcher.get_status()

        assert "config" in status
        assert "stats" in status
        assert "last_updated" in status
        assert status["config"]["batch_size"] == 5
        assert status["config"]["enable_dry_run"] is True
        assert status["stats"]["total_questions_fetched"] == 0


class TestDispatcherError:
    """Test DispatcherError exception."""

    def test_dispatcher_error_creation(self):
        """Test DispatcherError exception creation."""
        error = DispatcherError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
