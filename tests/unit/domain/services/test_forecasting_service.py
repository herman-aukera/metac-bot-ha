"""Comprehensive unit tests for ForecastingService."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from src.domain.entities.forecast import Forecast, ForecastStatus
from src.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
)
from src.domain.entities.research_report import ResearchReport
from src.domain.services.forecasting_service import ForecastingService
from src.domain.value_objects.probability import Probability


class TestForecastingService:
    """Test cases for ForecastingService."""

    @pytest.fixture
    def forecasting_service(self):
        """Create ForecastingService instance."""
        return ForecastingService()

    @pytest.fixture
    def sample_question_id(self):
        """Sample question ID for testing."""
        return uuid4()

    @pytest.fixture
    def sample_research_report_id(self):
        """Sample research report ID for testing."""
        return uuid4()

    @pytest.fixture
    def binary_prediction_high_conf(
        self, sample_question_id, sample_research_report_id
    ):
        """Create a binary prediction with high confidence."""
        return Prediction.create_binary_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Strong evidence suggests positive outcome",
            created_by="test_agent_1",
        )

    @pytest.fixture
    def binary_prediction_medium_conf(
        self, sample_question_id, sample_research_report_id
    ):
        """Create a binary prediction with medium confidence."""
        return Prediction.create_binary_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            probability=0.6,
            confidence=PredictionConfidence.MEDIUM,
            method=PredictionMethod.TREE_OF_THOUGHT,
            reasoning="Moderate evidence for positive outcome",
            created_by="test_agent_2",
        )

    @pytest.fixture
    def binary_prediction_low_conf(self, sample_question_id, sample_research_report_id):
        """Create a binary prediction with low confidence."""
        return Prediction.create_binary_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            probability=0.8,
            confidence=PredictionConfidence.LOW,
            method=PredictionMethod.REACT,
            reasoning="Limited evidence but suggests positive outcome",
            created_by="test_agent_3",
        )

    @pytest.fixture
    def sample_forecast(self, sample_question_id, binary_prediction_high_conf):
        """Sample forecast for testing."""
        return Forecast(
            id=uuid4(),
            question_id=sample_question_id,
            research_reports=[],
            predictions=[binary_prediction_high_conf],
            final_prediction=binary_prediction_high_conf,
            status=ForecastStatus.DRAFT,
            confidence_score=0.75,
            reasoning_summary="Test forecast",
            submission_timestamp=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            ensemble_method="single",
            weight_distribution={"test_agent_1": 1.0},
            consensus_strength=1.0,
            metadata={},
        )

    def test_init_creates_service_with_supported_methods(self, forecasting_service):
        """Test service initialization with supported prediction methods."""
        assert isinstance(forecasting_service, ForecastingService)
        assert len(forecasting_service.supported_methods) == 5
        assert (
            PredictionMethod.CHAIN_OF_THOUGHT in forecasting_service.supported_methods
        )
        assert PredictionMethod.TREE_OF_THOUGHT in forecasting_service.supported_methods
        assert PredictionMethod.REACT in forecasting_service.supported_methods
        assert PredictionMethod.AUTO_COT in forecasting_service.supported_methods
        assert (
            PredictionMethod.SELF_CONSISTENCY in forecasting_service.supported_methods
        )

    def test_aggregate_predictions_empty_list_raises_error(self, forecasting_service):
        """Test aggregating empty prediction list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot aggregate empty prediction list"):
            forecasting_service.aggregate_predictions([])

    def test_aggregate_predictions_different_questions_raises_error(
        self,
        forecasting_service,
        binary_prediction_high_conf,
        sample_research_report_id,
    ):
        """Test aggregating predictions from different questions raises ValueError."""
        # Create prediction for different question
        different_question_prediction = Prediction.create_binary_prediction(
            question_id=uuid4(),  # Different question ID
            research_report_id=sample_research_report_id,
            probability=0.5,
            confidence=PredictionConfidence.MEDIUM,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Different question",
            created_by="test_agent",
        )

        predictions = [binary_prediction_high_conf, different_question_prediction]

        with pytest.raises(
            ValueError, match="All predictions must be for the same question"
        ):
            forecasting_service.aggregate_predictions(predictions)

    def test_aggregate_predictions_weighted_average(
        self,
        forecasting_service,
        binary_prediction_high_conf,
        binary_prediction_medium_conf,
    ):
        """Test weighted average aggregation."""
        predictions = [binary_prediction_high_conf, binary_prediction_medium_conf]

        result = forecasting_service.aggregate_predictions(
            predictions, method="weighted_average"
        )

        assert isinstance(result, Prediction)
        assert result.question_id == binary_prediction_high_conf.question_id
        assert result.method == PredictionMethod.ENSEMBLE
        assert result.created_by == "ensemble_service"
        assert "Weighted average of 2 predictions" in result.reasoning

        # Check aggregated probability is weighted by confidence
        # High conf (0.75) * 0.7 + Medium conf (0.6) * 0.6 = weighted average
        high_weight = binary_prediction_high_conf.get_confidence_score()  # 0.75
        medium_weight = binary_prediction_medium_conf.get_confidence_score()  # 0.6
        total_weight = high_weight + medium_weight
        expected_prob = (0.7 * high_weight + 0.6 * medium_weight) / total_weight

        assert abs(result.result.binary_probability - expected_prob) < 0.001
        assert result.method_metadata["aggregation_method"] == "weighted_average"
        assert result.method_metadata["component_predictions"] == 2

    def test_aggregate_predictions_median(
        self,
        forecasting_service,
        binary_prediction_high_conf,
        binary_prediction_medium_conf,
        binary_prediction_low_conf,
    ):
        """Test median aggregation."""
        predictions = [
            binary_prediction_high_conf,
            binary_prediction_medium_conf,
            binary_prediction_low_conf,
        ]

        result = forecasting_service.aggregate_predictions(predictions, method="median")

        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        assert "Median of 3 predictions" in result.reasoning

        # Median of [0.7, 0.6, 0.8] = 0.7
        assert result.result.binary_probability == 0.7
        assert result.method_metadata["aggregation_method"] == "median"

    def test_aggregate_predictions_median_even_count(
        self,
        forecasting_service,
        binary_prediction_high_conf,
        binary_prediction_medium_conf,
    ):
        """Test median aggregation with even number of predictions."""
        predictions = [binary_prediction_high_conf, binary_prediction_medium_conf]

        result = forecasting_service.aggregate_predictions(predictions, method="median")

        # Median of [0.7, 0.6] = (0.6 + 0.7) / 2 = 0.65
        assert abs(result.result.binary_probability - 0.65) < 0.001

    def test_aggregate_predictions_confidence_weighted(
        self,
        forecasting_service,
        binary_prediction_high_conf,
        binary_prediction_medium_conf,
    ):
        """Test confidence-weighted aggregation."""
        predictions = [binary_prediction_high_conf, binary_prediction_medium_conf]

        result = forecasting_service.aggregate_predictions(
            predictions, method="confidence_weighted"
        )

        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        assert "Confidence-weighted average of 2 predictions" in result.reasoning

        # Should weight by squared confidence scores
        high_weight = binary_prediction_high_conf.get_confidence_score() ** 2  # 0.75^2
        medium_weight = (
            binary_prediction_medium_conf.get_confidence_score() ** 2
        )  # 0.6^2
        total_weight = high_weight + medium_weight
        expected_prob = (0.7 * high_weight + 0.6 * medium_weight) / total_weight

        assert abs(result.result.binary_probability - expected_prob) < 0.001
        assert result.method_metadata["aggregation_method"] == "confidence_weighted"

    def test_aggregate_predictions_unsupported_method_raises_error(
        self, forecasting_service, binary_prediction_high_conf
    ):
        """Test unsupported aggregation method raises ValueError."""
        predictions = [binary_prediction_high_conf]

        with pytest.raises(
            ValueError, match="Unsupported aggregation method: invalid_method"
        ):
            forecasting_service.aggregate_predictions(
                predictions, method="invalid_method"
            )

    def test_aggregate_predictions_zero_confidence_weights(
        self, forecasting_service, sample_question_id, sample_research_report_id
    ):
        """Test aggregation with zero confidence weights falls back to equal weights."""
        # Create predictions with very low confidence (maps to 0.2 score)
        zero_conf_pred1 = Prediction.create_binary_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            probability=0.3,
            confidence=PredictionConfidence.VERY_LOW,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Very uncertain",
            created_by="agent1",
        )
        zero_conf_pred2 = Prediction.create_binary_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            probability=0.7,
            confidence=PredictionConfidence.VERY_LOW,
            method=PredictionMethod.TREE_OF_THOUGHT,
            reasoning="Also very uncertain",
            created_by="agent2",
        )

        predictions = [zero_conf_pred1, zero_conf_pred2]
        result = forecasting_service.aggregate_predictions(
            predictions, method="weighted_average"
        )

        # Should use equal weights when all have zero confidence
        expected_prob = (0.3 + 0.7) / 2  # Equal weights
        assert abs(result.result.binary_probability - expected_prob) < 0.001

    def test_confidence_weighted_average_empty_list_raises_error(
        self, forecasting_service
    ):
        """Test confidence weighted average with empty list raises error."""
        with pytest.raises(
            ValueError, match="Cannot calculate average of empty prediction list"
        ):
            forecasting_service.confidence_weighted_average([])

    def test_confidence_weighted_average_no_binary_predictions_returns_default(
        self, forecasting_service, sample_question_id, sample_research_report_id
    ):
        """Test confidence weighted average with no binary predictions returns 0.5."""
        # Create numeric prediction (no binary_probability)
        numeric_pred = Prediction.create_numeric_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            value=100.0,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Numeric prediction",
            created_by="agent",
        )

        result = forecasting_service.confidence_weighted_average([numeric_pred])

        assert isinstance(result, Probability)
        assert result.value == 0.5

    def test_confidence_weighted_average_single_prediction(
        self, forecasting_service, binary_prediction_high_conf
    ):
        """Test confidence weighted average with single prediction."""
        result = forecasting_service.confidence_weighted_average(
            [binary_prediction_high_conf]
        )

        assert isinstance(result, Probability)
        assert result.value == 0.7

    def test_confidence_weighted_average_multiple_predictions(
        self,
        forecasting_service,
        binary_prediction_high_conf,
        binary_prediction_medium_conf,
    ):
        """Test confidence weighted average with multiple predictions."""
        predictions = [binary_prediction_high_conf, binary_prediction_medium_conf]

        result = forecasting_service.confidence_weighted_average(predictions)

        assert isinstance(result, Probability)

        # Calculate expected weighted average
        high_weight = binary_prediction_high_conf.get_confidence_score() ** 2
        medium_weight = binary_prediction_medium_conf.get_confidence_score() ** 2
        total_weight = high_weight + medium_weight
        expected = (0.7 * high_weight + 0.6 * medium_weight) / total_weight

        assert abs(result.value - expected) < 0.001

    def test_confidence_weighted_average_zero_total_weight(
        self, forecasting_service, sample_question_id, sample_research_report_id
    ):
        """Test confidence weighted average with zero total weight uses equal weights."""
        # Mock get_confidence_score to return 0
        pred1 = Prediction.create_binary_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            probability=0.3,
            confidence=PredictionConfidence.VERY_LOW,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Test",
            created_by="agent1",
        )
        pred2 = Prediction.create_binary_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            probability=0.7,
            confidence=PredictionConfidence.VERY_LOW,
            method=PredictionMethod.TREE_OF_THOUGHT,
            reasoning="Test",
            created_by="agent2",
        )

        with (
            patch.object(pred1, "get_confidence_score", return_value=0.0),
            patch.object(pred2, "get_confidence_score", return_value=0.0),
        ):
            result = forecasting_service.confidence_weighted_average([pred1, pred2])

            # Should use equal weights: (0.3 + 0.7) / 2 = 0.5
            assert result.value == 0.5

    def test_calculate_average_confidence(
        self,
        forecasting_service,
        binary_prediction_high_conf,
        binary_prediction_medium_conf,
    ):
        """Test calculating average confidence from predictions."""
        predictions = [binary_prediction_high_conf, binary_prediction_medium_conf]

        avg_confidence = forecasting_service._calculate_average_confidence(predictions)

        # Average of HIGH (0.75) and MEDIUM (0.6) = 0.675, which maps to HIGH
        assert avg_confidence == PredictionConfidence.HIGH

    def test_score_to_confidence_mapping(self, forecasting_service):
        """Test confidence score to enum mapping."""
        assert (
            forecasting_service._score_to_confidence(0.1)
            == PredictionConfidence.VERY_LOW
        )
        assert forecasting_service._score_to_confidence(0.3) == PredictionConfidence.LOW
        assert (
            forecasting_service._score_to_confidence(0.5) == PredictionConfidence.MEDIUM
        )
        assert (
            forecasting_service._score_to_confidence(0.7) == PredictionConfidence.HIGH
        )
        assert (
            forecasting_service._score_to_confidence(0.9)
            == PredictionConfidence.VERY_HIGH
        )

        # Boundary conditions
        assert (
            forecasting_service._score_to_confidence(0.2)
            == PredictionConfidence.VERY_LOW
        )
        assert forecasting_service._score_to_confidence(0.4) == PredictionConfidence.LOW
        assert (
            forecasting_service._score_to_confidence(0.6) == PredictionConfidence.MEDIUM
        )
        assert (
            forecasting_service._score_to_confidence(0.8) == PredictionConfidence.HIGH
        )

    def test_validate_forecast_quality_no_research_reports(
        self, forecasting_service, sample_question_id
    ):
        """Test forecast validation with no research reports."""
        forecast = Forecast(
            id=uuid4(),
            question_id=sample_question_id,
            research_reports=[],  # Empty
            predictions=[],
            final_prediction=None,
            status=ForecastStatus.DRAFT,
            confidence_score=0.5,
            reasoning_summary="Test",
            submission_timestamp=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            ensemble_method="single",
            weight_distribution={},
            consensus_strength=0.0,
            metadata={},
        )

        quality = forecasting_service.validate_forecast_quality(forecast)

        assert quality["is_valid"] is False
        assert "No research reports provided" in quality["issues"]

    def test_validate_forecast_quality_no_predictions(
        self, forecasting_service, sample_question_id
    ):
        """Test forecast validation with no predictions."""
        mock_research_report = Mock(spec=ResearchReport)
        mock_research_report.confidence_level = 0.8

        forecast = Forecast(
            id=uuid4(),
            question_id=sample_question_id,
            research_reports=[mock_research_report],
            predictions=[],  # Empty
            final_prediction=None,
            status=ForecastStatus.DRAFT,
            confidence_score=0.5,
            reasoning_summary="Test",
            submission_timestamp=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            ensemble_method="single",
            weight_distribution={},
            consensus_strength=0.0,
            metadata={},
        )

        quality = forecasting_service.validate_forecast_quality(forecast)

        assert quality["is_valid"] is False
        assert "No predictions provided" in quality["issues"]

    def test_validate_forecast_quality_high_variance_warning(
        self, forecasting_service, sample_forecast
    ):
        """Test forecast validation with high prediction variance."""
        # Mock calculate_prediction_variance to return high variance
        with patch.object(
            sample_forecast, "calculate_prediction_variance", return_value=0.15
        ):
            quality = forecasting_service.validate_forecast_quality(sample_forecast)

            assert any(
                "High prediction variance" in warning for warning in quality["warnings"]
            )

    def test_validate_forecast_quality_low_research_quality_warning(
        self, forecasting_service, sample_forecast
    ):
        """Test forecast validation with low research quality."""
        # Add mock research report with low quality
        mock_research_report = Mock(spec=ResearchReport)
        mock_research_report.confidence_level = 0.3  # Low quality
        sample_forecast.research_reports = [mock_research_report]

        with patch.object(
            sample_forecast, "calculate_prediction_variance", return_value=0.05
        ):
            quality = forecasting_service.validate_forecast_quality(sample_forecast)

            assert any(
                "Low average research quality" in warning
                for warning in quality["warnings"]
            )

    def test_validate_forecast_quality_high_quality_forecast(
        self,
        forecasting_service,
        sample_forecast,
        binary_prediction_high_conf,
        binary_prediction_medium_conf,
        binary_prediction_low_conf,
    ):
        """Test forecast validation with high quality forecast."""
        # Add multiple predictions and high-quality research
        sample_forecast.predictions = [
            binary_prediction_high_conf,
            binary_prediction_medium_conf,
            binary_prediction_low_conf,
        ]

        mock_research_report = Mock(spec=ResearchReport)
        mock_research_report.confidence_level = 0.8  # High quality
        sample_forecast.research_reports = [mock_research_report]

        with patch.object(
            sample_forecast, "calculate_prediction_variance", return_value=0.03
        ):  # High consensus
            quality = forecasting_service.validate_forecast_quality(sample_forecast)

            assert quality["is_valid"] is True
            assert len(quality["issues"]) == 0
            # Quality score should be high: 0.5 + 0.2 (research) + 0.1 (3+ predictions) + 0.1 (consensus) + 0.1 (quality) = 1.0
            assert abs(quality["quality_score"] - 1.0) < 0.001

    def test_validate_forecast_quality_medium_quality_forecast(
        self, forecasting_service, sample_forecast
    ):
        """Test forecast validation with medium quality forecast."""
        mock_research_report = Mock(spec=ResearchReport)
        mock_research_report.confidence_level = 0.6  # Medium quality
        sample_forecast.research_reports = [mock_research_report]

        with patch.object(
            sample_forecast, "calculate_prediction_variance", return_value=0.08
        ):  # Medium consensus
            quality = forecasting_service.validate_forecast_quality(sample_forecast)

            assert quality["is_valid"] is True
            # Base score: 0.5 + 0.2 (has research) = 0.7
            assert quality["quality_score"] == 0.7

    def test_aggregate_predictions_only_binary_supported_error(
        self, forecasting_service, sample_question_id, sample_research_report_id
    ):
        """Test that only binary predictions are currently supported."""
        # Create numeric prediction
        numeric_pred = Prediction.create_numeric_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            value=100.0,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Numeric prediction",
            created_by="agent",
        )

        with pytest.raises(
            NotImplementedError, match="Only binary predictions supported currently"
        ):
            forecasting_service.aggregate_predictions(
                [numeric_pred], method="weighted_average"
            )

    def test_weighted_average_aggregation_metadata_includes_weights(
        self,
        forecasting_service,
        binary_prediction_high_conf,
        binary_prediction_medium_conf,
    ):
        """Test that weighted average aggregation includes weights in metadata."""
        predictions = [binary_prediction_high_conf, binary_prediction_medium_conf]

        result = forecasting_service.aggregate_predictions(
            predictions, method="weighted_average"
        )

        assert "weights" in result.method_metadata
        assert len(result.method_metadata["weights"]) == 2
        # Weights should correspond to confidence scores
        expected_weights = [p.get_confidence_score() for p in predictions]
        assert result.method_metadata["weights"] == expected_weights

    def test_confidence_weighted_aggregation_includes_confidence_weights(
        self,
        forecasting_service,
        binary_prediction_high_conf,
        binary_prediction_medium_conf,
    ):
        """Test that confidence-weighted aggregation includes confidence weights in metadata."""
        predictions = [binary_prediction_high_conf, binary_prediction_medium_conf]

        result = forecasting_service.aggregate_predictions(
            predictions, method="confidence_weighted"
        )

        assert "confidence_weights" in result.method_metadata
        assert len(result.method_metadata["confidence_weights"]) == 2
        # Weights should be squared confidence scores
        expected_weights = [p.get_confidence_score() ** 2 for p in predictions]
        assert result.method_metadata["confidence_weights"] == expected_weights

    def test_confidence_weighted_aggregation_falls_back_to_weighted_average(
        self, forecasting_service, sample_question_id, sample_research_report_id
    ):
        """Test confidence-weighted aggregation falls back to weighted average when total weight is zero."""
        # Create predictions with zero confidence
        pred1 = Prediction.create_binary_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            probability=0.3,
            confidence=PredictionConfidence.VERY_LOW,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Test",
            created_by="agent1",
        )
        pred2 = Prediction.create_binary_prediction(
            question_id=sample_question_id,
            research_report_id=sample_research_report_id,
            probability=0.7,
            confidence=PredictionConfidence.VERY_LOW,
            method=PredictionMethod.TREE_OF_THOUGHT,
            reasoning="Test",
            created_by="agent2",
        )

        with (
            patch.object(pred1, "get_confidence_score", return_value=0.0),
            patch.object(pred2, "get_confidence_score", return_value=0.0),
        ):
            # Mock the _weighted_average_aggregation method to verify it's called
            with patch.object(
                forecasting_service, "_weighted_average_aggregation"
            ) as mock_weighted:
                mock_result = Mock(spec=Prediction)
                mock_weighted.return_value = mock_result

                result = forecasting_service.aggregate_predictions(
                    [pred1, pred2], method="confidence_weighted"
                )

                mock_weighted.assert_called_once_with(
                    [pred1, pred2], sample_question_id
                )
                assert result == mock_result

    def test_probability_value_bounds_enforced(self, forecasting_service):
        """Test that probability values are kept within valid bounds."""
        # Test with values that might go out of bounds during calculation
        predictions = []
        question_id = uuid4()
        research_report_id = uuid4()

        # Create predictions with extreme values
        for prob in [0.0, 1.0]:
            pred = Prediction.create_binary_prediction(
                question_id=question_id,
                research_report_id=research_report_id,
                probability=prob,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning=f"Extreme probability {prob}",
                created_by="test_agent",
            )
            predictions.append(pred)

        result = forecasting_service.confidence_weighted_average(predictions)

        # Result should be within bounds
        assert 0.0 <= result.value <= 1.0
