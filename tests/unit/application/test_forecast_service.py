"""Tests for the ForecastService application service."""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4, UUID
from datetime import datetime, timezone, timedelta

from src.application.forecast_service import ForecastService, ForecastValidationError
from src.domain.entities.question import Question, QuestionType, QuestionStatus
from src.domain.entities.forecast import Forecast, ForecastStatus
from src.domain.entities.prediction import Prediction, PredictionResult, PredictionConfidence, PredictionMethod
from src.domain.entities.research_report import ResearchReport, ResearchQuality
from src.domain.value_objects.probability import Probability
from src.domain.value_objects.confidence import ConfidenceLevel


@pytest.fixture
def forecast_service():
    """Create a ForecastService instance."""
    return ForecastService()


@pytest.fixture
def sample_binary_question():
    """Create a sample binary question for testing."""
    return Question(
        id=uuid4(),
        metaculus_id=101,
        title="Will AI achieve AGI by 2030?",
        description="Question about AGI timeline",
        question_type=QuestionType.BINARY,
        status=QuestionStatus.OPEN,
        url="https://metaculus.com/questions/101/",
        close_time=datetime.now(timezone.utc) + timedelta(days=30),
        resolve_time=datetime.now(timezone.utc) + timedelta(days=365),
        categories=["AI", "Technology"],
        metadata={"community_prediction": 0.6},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_numeric_question():
    """Create a sample numeric question for testing."""
    return Question(
        id=uuid4(),
        metaculus_id=102,
        title="What will be the global temperature anomaly in 2025?",
        description="Question about climate",
        question_type=QuestionType.NUMERIC,
        status=QuestionStatus.OPEN,
        url="https://metaculus.com/questions/102/",
        close_time=datetime.now(timezone.utc) + timedelta(days=30),
        resolve_time=datetime.now(timezone.utc) + timedelta(days=365),
        categories=["Climate"],
        metadata={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        min_value=0.0,
        max_value=5.0
    )


@pytest.fixture
def sample_closed_question():
    """Create a sample closed question for testing."""
    return Question(
        id=uuid4(),
        metaculus_id=103,
        title="Closed Question",
        description="This question is closed",
        question_type=QuestionType.BINARY,
        status=QuestionStatus.CLOSED,
        url="https://metaculus.com/questions/103/",
        close_time=datetime.now(timezone.utc) - timedelta(days=1),
        resolve_time=datetime.now(timezone.utc),
        categories=["Test"],
        metadata={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_forecast():
    """Create a sample forecast for testing."""
    research_report = Mock(spec=ResearchReport)
    research_report.id = uuid4()
    
    prediction = Mock(spec=Prediction)
    prediction.result = PredictionResult(binary_probability=0.7)
    prediction.confidence = PredictionConfidence.MEDIUM
    
    forecast = Mock(spec=Forecast)
    forecast.id = uuid4()
    forecast.question_id = uuid4()
    forecast.status = ForecastStatus.SUBMITTED
    forecast.final_prediction = prediction
    forecast.research_reports = [research_report]
    forecast.predictions = [prediction]
    
    return forecast


class TestForecastValidationError:
    """Test ForecastValidationError exception."""
    
    def test_forecast_validation_error(self):
        """Test ForecastValidationError exception creation."""
        error = ForecastValidationError("Test validation error")
        assert str(error) == "Test validation error"
        assert isinstance(error, Exception)


class TestForecastService:
    """Test ForecastService class."""
    
    def test_init(self, forecast_service):
        """Test ForecastService initialization."""
        assert isinstance(forecast_service, ForecastService)
    
    # Test validate_forecast method
    def test_validate_forecast_success(self, forecast_service, sample_binary_question):
        """Test successful forecast validation."""
        probability = Probability(0.7)
        confidence = ConfidenceLevel(0.8)
        reasoning = "Based on historical trends and expert analysis."
        
        # Should not raise any exception
        forecast_service.validate_forecast(sample_binary_question, probability, confidence, reasoning)
    
    def test_validate_forecast_closed_question(self, forecast_service, sample_closed_question):
        """Test validation with closed question."""
        probability = Probability(0.7)
        confidence = ConfidenceLevel(0.8)
        reasoning = "Some reasoning"
        
        with pytest.raises(ForecastValidationError, match="Question is closed"):
            forecast_service.validate_forecast(sample_closed_question, probability, confidence, reasoning)
    
    def test_validate_forecast_non_binary_question(self, forecast_service, sample_numeric_question):
        """Test validation with non-binary question."""
        probability = Probability(0.7)
        confidence = ConfidenceLevel(0.8)
        reasoning = "Some reasoning"
        
        with pytest.raises(ForecastValidationError, match="Only binary questions are supported"):
            forecast_service.validate_forecast(sample_numeric_question, probability, confidence, reasoning)
    
    def test_validate_forecast_empty_reasoning(self, forecast_service, sample_binary_question):
        """Test validation with empty reasoning."""
        probability = Probability(0.7)
        confidence = ConfidenceLevel(0.8)
        reasoning = ""
        
        with pytest.raises(ForecastValidationError, match="Reasoning cannot be empty"):
            forecast_service.validate_forecast(sample_binary_question, probability, confidence, reasoning)
    
    def test_validate_forecast_extreme_probability(self, forecast_service, sample_binary_question):
        """Test validation with extreme probability values."""
        probability_low = Probability(0.02)
        probability_high = Probability(0.98)
        confidence = ConfidenceLevel(0.8)
        reasoning = "Some reasoning"
        
        with pytest.raises(ForecastValidationError, match="Extreme probability values"):
            forecast_service.validate_forecast(sample_binary_question, probability_low, confidence, reasoning)
        
        with pytest.raises(ForecastValidationError, match="Extreme probability values"):
            forecast_service.validate_forecast(sample_binary_question, probability_high, confidence, reasoning)
    
    # Test create_forecast method
    def test_create_forecast_success(self, forecast_service, sample_binary_question):
        """Test successful forecast creation."""
        forecaster_id = uuid4()
        probability = Probability(0.7)
        confidence = ConfidenceLevel(0.8)
        reasoning = "Based on historical trends and expert analysis."
        
        forecast = forecast_service.create_forecast(
            sample_binary_question, forecaster_id, probability, confidence, reasoning
        )
        
        assert isinstance(forecast, Forecast)
        assert forecast.question_id == sample_binary_question.id
        assert len(forecast.predictions) == 1
        assert forecast.final_prediction.result.binary_probability == probability.value
        assert forecast.reasoning_summary == reasoning
    
    def test_create_forecast_validation_error(self, forecast_service, sample_closed_question):
        """Test forecast creation with validation error."""
        forecaster_id = uuid4()
        probability = Probability(0.7)
        confidence = ConfidenceLevel(0.8)
        reasoning = "Some reasoning"
        
        with pytest.raises(ForecastValidationError):
            forecast_service.create_forecast(
                sample_closed_question, forecaster_id, probability, confidence, reasoning
            )
    
    # Test generate_forecast method
    def test_generate_forecast_success(self, forecast_service, sample_binary_question):
        """Test successful forecast generation."""
        forecast = forecast_service.generate_forecast(sample_binary_question)
        
        assert isinstance(forecast, Forecast)
        assert forecast.question_id == sample_binary_question.id
        assert len(forecast.predictions) > 0
        assert forecast.final_prediction is not None
        assert forecast.final_prediction.result.binary_probability is not None
        assert 0.0 <= forecast.final_prediction.result.binary_probability <= 1.0
        assert len(forecast.research_reports) > 0
    
    def test_generate_forecast_with_community_prediction(self, forecast_service, sample_binary_question):
        """Test forecast generation using community prediction."""
        # Set community prediction in metadata
        sample_binary_question.metadata = {"community_prediction": 0.8}
        
        forecast = forecast_service.generate_forecast(sample_binary_question)
        
        assert isinstance(forecast, Forecast)
        # The generated forecast should be influenced by the community prediction
        final_prob = forecast.final_prediction.result.binary_probability
        assert final_prob is not None
        # Should be within reasonable range of community prediction (allowing for variation)
        assert 0.6 <= final_prob <= 1.0
    
    def test_generate_forecast_closed_question(self, forecast_service, sample_closed_question):
        """Test forecast generation with closed question."""
        with pytest.raises(ForecastValidationError, match="Cannot generate forecast for closed question"):
            forecast_service.generate_forecast(sample_closed_question)
    
    def test_generate_forecast_non_binary_question(self, forecast_service, sample_numeric_question):
        """Test forecast generation with non-binary question."""
        # The service now supports numeric questions, so this should succeed
        forecast = forecast_service.generate_forecast(sample_numeric_question)
        
        assert isinstance(forecast, Forecast)
        assert forecast.question_id == sample_numeric_question.id
        assert len(forecast.predictions) > 0
        assert forecast.final_prediction is not None
        assert forecast.final_prediction.result.numeric_value is not None
        assert sample_numeric_question.min_value <= forecast.final_prediction.result.numeric_value <= sample_numeric_question.max_value
        assert len(forecast.research_reports) > 0
    
    # Test score_forecast method
    def test_score_forecast_unresolved_question(self, forecast_service, sample_forecast, sample_binary_question):
        """Test scoring with unresolved question."""
        score = forecast_service.score_forecast(sample_forecast, sample_binary_question)
        
        assert score is None  # Should return None for unresolved questions
    
    def test_score_forecast_resolved_question(self, forecast_service, sample_forecast):
        """Test scoring with resolved question (placeholder test)."""
        # Create a mock resolved question
        resolved_question = Mock()
        resolved_question.is_resolved.return_value = True
        
        # This is a placeholder test since the resolution logic is not fully implemented
        score = forecast_service.score_forecast(sample_forecast, resolved_question)
        
        # Should return None since outcome extraction is not implemented
        assert score is None
    
    # Test batch_score_forecasts method
    def test_batch_score_forecasts(self, forecast_service, sample_binary_question, sample_forecast):
        """Test batch scoring of forecasts."""
        forecasts = [sample_forecast() for _ in range(3)]
        questions = [sample_binary_question for _ in range(3)]
        
        scores = forecast_service.batch_score_forecasts(forecasts, questions)
        
        assert len(scores) == len(forecasts)
        # All should be None since questions are not resolved
        assert all(score is None for score in scores)
    
    def test_batch_score_forecasts_missing_questions(self, forecast_service, sample_forecast):
        """Test batch scoring with missing questions."""
        forecasts = [sample_forecast]
        questions = []  # No questions provided
        
        scores = forecast_service.batch_score_forecasts(forecasts, questions)
        
        assert len(scores) == len(forecasts)
        assert scores[0] is None  # Should be None for missing question
    
    # Test calculate_average_score method
    def test_calculate_average_score_valid_scores(self, forecast_service):
        """Test average score calculation with valid scores."""
        scores = [0.1, 0.2, 0.3, 0.4]
        
        average = forecast_service.calculate_average_score(scores)
        
        assert average == 0.25  # (0.1 + 0.2 + 0.3 + 0.4) / 4
    
    def test_calculate_average_score_with_none_values(self, forecast_service):
        """Test average score calculation with None values."""
        scores = [0.1, None, 0.3, None, 0.5]
        
        average = forecast_service.calculate_average_score(scores)
        
        assert average == 0.3  # (0.1 + 0.3 + 0.5) / 3
    
    def test_calculate_average_score_all_none(self, forecast_service):
        """Test average score calculation with all None values."""
        scores = [None, None, None]
        
        average = forecast_service.calculate_average_score(scores)
        
        assert average is None
    
    def test_calculate_average_score_empty_list(self, forecast_service):
        """Test average score calculation with empty list."""
        scores = []
        
        average = forecast_service.calculate_average_score(scores)
        
        assert average is None
    
    # Test get_forecast_summary method
    def test_get_forecast_summary(self, forecast_service, sample_binary_question, sample_forecast):
        """Test forecast summary generation."""
        forecasts = [sample_forecast() for _ in range(3)]
        questions = [sample_binary_question for _ in range(3)]
        
        summary = forecast_service.get_forecast_summary(forecasts, questions)
        
        assert isinstance(summary, dict)
        assert "total_forecasts" in summary
        assert "scored_forecasts" in summary
        assert "average_score" in summary
        assert "scores" in summary
        
        assert summary["total_forecasts"] == 3
        assert summary["scored_forecasts"] == 0  # No resolved questions
        assert summary["average_score"] is None
        assert len(summary["scores"]) == 3
    
    # Test _is_extreme_probability method
    def test_is_extreme_probability(self, forecast_service):
        """Test extreme probability detection."""
        low_prob = Probability(0.03)
        high_prob = Probability(0.97)
        normal_prob = Probability(0.5)
        
        assert forecast_service._is_extreme_probability(low_prob) is True
        assert forecast_service._is_extreme_probability(high_prob) is True
        assert forecast_service._is_extreme_probability(normal_prob) is False
    
    # Test _generate_mock_reasoning method
    def test_generate_mock_reasoning(self, forecast_service, sample_binary_question):
        """Test mock reasoning generation."""
        ai_probability = 0.7
        base_probability = 0.6
        
        reasoning = forecast_service._generate_mock_reasoning(
            sample_binary_question, ai_probability, base_probability
        )
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert sample_binary_question.title in reasoning
        assert f"{ai_probability:.1%}" in reasoning
        assert f"{base_probability:.1%}" in reasoning
    
    # Test _create_mock_research_report method
    def test_create_mock_research_report(self, forecast_service, sample_binary_question):
        """Test mock research report creation."""
        base_probability = 0.6
        
        research_report = forecast_service._create_mock_research_report(
            sample_binary_question, base_probability
        )
        
        assert isinstance(research_report, ResearchReport)
        assert research_report.question_id == sample_binary_question.id
        assert sample_binary_question.title in research_report.title
        assert f"{base_probability:.1%}" in research_report.executive_summary
        assert len(research_report.sources) > 0
        assert research_report.quality == ResearchQuality.MEDIUM
        assert len(research_report.key_factors) > 0
        assert "historical_rate" in research_report.base_rates
        assert research_report.base_rates["historical_rate"] == base_probability


class TestForecastServiceIntegration:
    """Integration tests for ForecastService."""
    
    def test_end_to_end_forecast_generation(self, forecast_service, sample_binary_question):
        """Test complete forecast generation workflow."""
        # Generate a forecast
        forecast = forecast_service.generate_forecast(sample_binary_question)
        
        # Verify the forecast structure
        assert isinstance(forecast, Forecast)
        assert forecast.question_id == sample_binary_question.id
        assert forecast.status == ForecastStatus.DRAFT
        assert len(forecast.predictions) >= 1
        assert forecast.final_prediction is not None
        assert len(forecast.research_reports) >= 1
        
        # Verify prediction details
        final_prediction = forecast.final_prediction
        assert final_prediction.question_id == sample_binary_question.id
        assert final_prediction.result.binary_probability is not None
        assert 0.0 <= final_prediction.result.binary_probability <= 1.0
        assert final_prediction.confidence in PredictionConfidence
        assert final_prediction.method in PredictionMethod
        assert len(final_prediction.reasoning) > 0
        
        # Verify research report details
        research_report = forecast.research_reports[0]
        assert research_report.question_id == sample_binary_question.id
        assert len(research_report.sources) > 0
        assert research_report.quality in ResearchQuality
        assert len(research_report.key_factors) > 0
    
    def test_forecast_consistency(self, forecast_service, sample_binary_question):
        """Test that forecast generation is reasonably consistent."""
        # Generate multiple forecasts for the same question
        forecasts = [
            forecast_service.generate_forecast(sample_binary_question)
            for _ in range(5)
        ]
        
        # Extract final probabilities
        probabilities = [
            f.final_prediction.result.binary_probability 
            for f in forecasts
        ]
        
        # All probabilities should be valid
        assert all(0.0 <= p <= 1.0 for p in probabilities)
        
        # Probabilities should be reasonably consistent (not wildly different)
        # Allow for some variation due to randomness in mock generation
        prob_range = max(probabilities) - min(probabilities)
        assert prob_range <= 0.5  # Should not vary by more than 50%