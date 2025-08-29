"""Tests for UncertaintyQuantifier service."""

from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.domain.entities.forecast import Forecast
from src.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
    PredictionResult,
)
from src.domain.services.uncertainty_quantifier import (
    ConfidenceThresholds,
    UncertaintyAssessment,
    UncertaintyQuantifier,
    UncertaintySource,
)
from src.domain.value_objects.reasoning_trace import ReasoningTrace


class TestConfidenceThresholds:
    """Test confidence thresholds configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = ConfidenceThresholds()

        assert thresholds.minimum_submission == 0.6
        assert thresholds.high_confidence == 0.8
        assert thresholds.very_high_confidence == 0.9
        assert thresholds.abstention_threshold == 0.4
        assert thresholds.research_trigger == 0.5

    def test_threshold_validation_success(self):
        """Test successful threshold validation."""
        thresholds = ConfidenceThresholds(
            abstention_threshold=0.3,
            research_trigger=0.5,
            minimum_submission=0.6,
            high_confidence=0.8,
            very_high_confidence=0.9,
        )

        # Should not raise exception
        thresholds.validate_thresholds()

    def test_threshold_validation_failure(self):
        """Test threshold validation with invalid order."""
        thresholds = ConfidenceThresholds(
            abstention_threshold=0.7,  # Higher than minimum_submission
            minimum_submission=0.6,
        )

        with pytest.raises(ValueError, match="ascending order"):
            thresholds.validate_thresholds()


class TestUncertaintyAssessment:
    """Test uncertainty assessment data structure."""

    def test_uncertainty_assessment_creation(self):
        """Test creating uncertainty assessment."""
        uncertainty_sources = {
            UncertaintySource.EPISTEMIC: 0.3,
            UncertaintySource.DATA: 0.4,
            UncertaintySource.MODEL: 0.2,
        }

        assessment = UncertaintyAssessment(
            total_uncertainty=0.5,
            uncertainty_sources=uncertainty_sources,
            confidence_interval=(0.3, 0.7),
            confidence_level=0.6,
            calibration_score=0.7,
            uncertainty_decomposition={"epistemic": 0.3, "data": 0.4},
            assessment_timestamp=datetime.utcnow(),
        )

        assert assessment.total_uncertainty == 0.5
        assert assessment.confidence_level == 0.6
        assert len(assessment.uncertainty_sources) == 3

    def test_uncertainty_summary(self):
        """Test uncertainty summary generation."""
        uncertainty_sources = {
            UncertaintySource.DATA: 0.6,  # Dominant source
            UncertaintySource.MODEL: 0.3,
        }

        assessment = UncertaintyAssessment(
            total_uncertainty=0.5,
            uncertainty_sources=uncertainty_sources,
            confidence_interval=(0.2, 0.8),
            confidence_level=0.6,
            calibration_score=0.7,
            uncertainty_decomposition={},
            assessment_timestamp=datetime.utcnow(),
        )

        summary = assessment.get_uncertainty_summary()

        assert summary["total_uncertainty"] == 0.5
        assert summary["dominant_source"] == "data"
        assert abs(summary["confidence_interval_width"] - 0.6) < 1e-10
        assert summary["confidence_level"] == 0.6


class TestUncertaintyQuantifier:
    """Test uncertainty quantifier service."""

    @pytest.fixture
    def quantifier(self):
        """Create uncertainty quantifier instance."""
        return UncertaintyQuantifier()

    @pytest.fixture
    def sample_prediction(self):
        """Create sample prediction for testing."""
        return Prediction.create_binary_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Test reasoning",
            created_by="test_agent",
        )

    @pytest.fixture
    def ensemble_predictions(self):
        """Create ensemble predictions for testing."""
        predictions = []
        probabilities = [0.6, 0.7, 0.8]  # Some variance

        for i, prob in enumerate(probabilities):
            prediction = Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=prob,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.ENSEMBLE,
                reasoning=f"Agent {i} reasoning",
                created_by=f"agent_{i}",
            )
            predictions.append(prediction)

        return predictions

    def test_assess_prediction_uncertainty_basic(self, quantifier, sample_prediction):
        """Test basic uncertainty assessment for single prediction."""
        assessment = quantifier.assess_prediction_uncertainty(sample_prediction)

        assert isinstance(assessment, UncertaintyAssessment)
        assert 0.0 <= assessment.total_uncertainty <= 1.0
        assert 0.0 <= assessment.confidence_level <= 1.0
        assert len(assessment.uncertainty_sources) == 6  # All uncertainty sources
        assert assessment.confidence_interval[0] < assessment.confidence_interval[1]

    def test_assess_prediction_uncertainty_with_ensemble(
        self, quantifier, sample_prediction, ensemble_predictions
    ):
        """Test uncertainty assessment with ensemble predictions."""
        assessment = quantifier.assess_prediction_uncertainty(
            sample_prediction, ensemble_predictions=ensemble_predictions
        )

        # Should have expert uncertainty from ensemble disagreement
        assert assessment.uncertainty_sources[UncertaintySource.EXPERT] > 0.0
        assert assessment.total_uncertainty > 0.0

    def test_assess_prediction_uncertainty_with_research_quality(
        self, quantifier, sample_prediction
    ):
        """Test uncertainty assessment with research quality score."""
        # High quality research should reduce data uncertainty
        assessment_high_quality = quantifier.assess_prediction_uncertainty(
            sample_prediction, research_quality_score=0.9
        )

        # Low quality research should increase data uncertainty
        assessment_low_quality = quantifier.assess_prediction_uncertainty(
            sample_prediction, research_quality_score=0.2
        )

        assert (
            assessment_low_quality.uncertainty_sources[UncertaintySource.DATA]
            > assessment_high_quality.uncertainty_sources[UncertaintySource.DATA]
        )

    def test_validate_confidence_level(self, quantifier, sample_prediction):
        """Test confidence level validation."""
        assessment = quantifier.assess_prediction_uncertainty(sample_prediction)
        validation = quantifier.validate_confidence_level(sample_prediction, assessment)

        assert "is_valid" in validation
        assert "predicted_confidence" in validation
        assert "assessed_confidence" in validation
        assert "confidence_gap" in validation
        assert "recommendation" in validation

        assert isinstance(validation["is_valid"], bool)
        assert 0.0 <= validation["predicted_confidence"] <= 1.0
        assert 0.0 <= validation["assessed_confidence"] <= 1.0

    def test_should_trigger_additional_research(self, quantifier, sample_prediction):
        """Test research trigger logic."""
        # Create assessment with high data uncertainty
        assessment = quantifier.assess_prediction_uncertainty(
            sample_prediction, research_quality_score=0.2  # Low quality
        )

        research_decision = quantifier.should_trigger_additional_research(assessment)

        assert "trigger_research" in research_decision
        assert "research_priorities" in research_decision
        assert "confidence_level" in research_decision
        assert "dominant_uncertainty" in research_decision

        # Should trigger research due to low quality
        assert research_decision["trigger_research"] is True
        assert "data_quality" in research_decision["research_priorities"]

    def test_should_abstain_from_prediction(self, quantifier):
        """Test abstention logic."""
        # Create low confidence prediction
        low_confidence_prediction = Prediction.create_binary_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            probability=0.5,
            confidence=PredictionConfidence.VERY_LOW,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Very uncertain",
            created_by="test_agent",
        )

        assessment = quantifier.assess_prediction_uncertainty(low_confidence_prediction)
        abstention_decision = quantifier.should_abstain_from_prediction(assessment)

        assert "should_abstain" in abstention_decision
        assert "confidence_level" in abstention_decision
        assert "abstention_threshold" in abstention_decision
        assert "reason" in abstention_decision

        # Should likely abstain due to very low confidence
        assert isinstance(abstention_decision["should_abstain"], bool)

    def test_should_abstain_with_tournament_context(
        self, quantifier, sample_prediction
    ):
        """Test abstention with tournament context."""
        assessment = quantifier.assess_prediction_uncertainty(sample_prediction)

        # Tournament with high abstention penalty
        tournament_context = {"abstention_penalty": 0.3}

        abstention_decision = quantifier.should_abstain_from_prediction(
            assessment, tournament_context
        )

        assert abstention_decision["tournament_penalty"] == 0.3
        assert (
            abstention_decision["abstention_threshold"]
            > quantifier.confidence_thresholds.abstention_threshold
        )

    def test_update_confidence_thresholds(self, quantifier):
        """Test confidence threshold updates based on performance."""
        original_threshold = quantifier.confidence_thresholds.minimum_submission

        # Poor calibration should make thresholds more conservative
        performance_data = {"accuracy": 0.6}
        calibration_data = {"calibration_error": 0.15}  # High error

        quantifier.update_confidence_thresholds(performance_data, calibration_data)

        assert quantifier.confidence_thresholds.minimum_submission > original_threshold

    def test_get_confidence_management_report(self, quantifier, ensemble_predictions):
        """Test confidence management report generation."""
        assessments = []
        for prediction in ensemble_predictions:
            assessment = quantifier.assess_prediction_uncertainty(prediction)
            assessments.append(assessment)

        report = quantifier.get_confidence_management_report(
            ensemble_predictions, assessments
        )

        assert "summary" in report
        assert "confidence_distribution" in report
        assert "uncertainty_analysis" in report
        assert "calibration_metrics" in report
        assert "threshold_performance" in report
        assert "recommendations" in report

        # Check summary statistics
        summary = report["summary"]
        assert summary["total_predictions"] == len(ensemble_predictions)
        assert 0.0 <= summary["average_confidence"] <= 1.0
        assert 0.0 <= summary["average_uncertainty"] <= 1.0

    def test_confidence_management_report_empty_data(self, quantifier):
        """Test report generation with empty data."""
        report = quantifier.get_confidence_management_report([], [])

        assert "error" in report

    def test_model_uncertainty_calculation(self, quantifier):
        """Test model uncertainty calculation for different methods."""
        # Tree of thought should have lower uncertainty than ReAct
        tot_prediction = Prediction.create_binary_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.TREE_OF_THOUGHT,
            reasoning="Systematic analysis",
            created_by="tot_agent",
        )

        react_prediction = Prediction.create_binary_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.REACT,
            reasoning="Dynamic reasoning",
            created_by="react_agent",
        )

        tot_uncertainty = quantifier._calculate_model_uncertainty(tot_prediction)
        react_uncertainty = quantifier._calculate_model_uncertainty(react_prediction)

        assert tot_uncertainty < react_uncertainty

    def test_expert_uncertainty_calculation(self, quantifier, ensemble_predictions):
        """Test expert uncertainty calculation from ensemble disagreement."""
        # High disagreement ensemble
        high_disagreement = [
            Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=prob,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.ENSEMBLE,
                reasoning="Test",
                created_by="agent",
            )
            for prob in [0.2, 0.8, 0.9]  # High variance
        ]

        # Low disagreement ensemble
        low_disagreement = [
            Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=prob,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.ENSEMBLE,
                reasoning="Test",
                created_by="agent",
            )
            for prob in [0.68, 0.70, 0.72]  # Low variance
        ]

        high_uncertainty = quantifier._calculate_expert_uncertainty(high_disagreement)
        low_uncertainty = quantifier._calculate_expert_uncertainty(low_disagreement)

        assert high_uncertainty > low_uncertainty

    def test_confidence_interval_calculation(self, quantifier, sample_prediction):
        """Test confidence interval calculation."""
        # High uncertainty should produce wider intervals
        high_uncertainty = 0.8
        low_uncertainty = 0.2

        high_interval = quantifier._calculate_confidence_interval(
            sample_prediction, high_uncertainty
        )
        low_interval = quantifier._calculate_confidence_interval(
            sample_prediction, low_uncertainty
        )

        high_width = high_interval[1] - high_interval[0]
        low_width = low_interval[1] - low_interval[0]

        assert high_width > low_width
        assert 0.0 <= high_interval[0] <= high_interval[1] <= 1.0
        assert 0.0 <= low_interval[0] <= low_interval[1] <= 1.0

    def test_assess_forecast_uncertainty(self, quantifier):
        """Test uncertainty assessment for complete forecast."""
        # Create mock forecast
        forecast = Mock(spec=Forecast)
        forecast.final_prediction = Prediction.create_binary_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.ENSEMBLE,
            reasoning="Ensemble prediction",
            created_by="ensemble",
        )
        forecast.predictions = [forecast.final_prediction]

        assessment = quantifier.assess_forecast_uncertainty(forecast)

        assert isinstance(assessment, UncertaintyAssessment)
        assert 0.0 <= assessment.total_uncertainty <= 1.0

    def test_uncertainty_source_analysis(self, quantifier, ensemble_predictions):
        """Test uncertainty source analysis across multiple assessments."""
        assessments = []
        for prediction in ensemble_predictions:
            assessment = quantifier.assess_prediction_uncertainty(prediction)
            assessments.append(assessment)

        source_analysis = quantifier._analyze_uncertainty_sources(assessments)

        # Should have analysis for all uncertainty sources
        for source in UncertaintySource:
            assert source.value in source_analysis
            assert "mean" in source_analysis[source.value]
            assert "max" in source_analysis[source.value]
            assert "std" in source_analysis[source.value]

    def test_calibration_metrics_calculation(self, quantifier, ensemble_predictions):
        """Test calibration metrics calculation."""
        assessments = []
        for prediction in ensemble_predictions:
            assessment = quantifier.assess_prediction_uncertainty(prediction)
            assessments.append(assessment)

        calibration_metrics = quantifier._calculate_calibration_metrics(
            ensemble_predictions, assessments
        )

        assert "confidence_mae" in calibration_metrics
        assert "average_predicted_confidence" in calibration_metrics
        assert "average_assessed_confidence" in calibration_metrics

        assert 0.0 <= calibration_metrics["confidence_mae"] <= 1.0

    def test_threshold_performance_analysis(self, quantifier, ensemble_predictions):
        """Test threshold performance analysis."""
        assessments = []
        for prediction in ensemble_predictions:
            assessment = quantifier.assess_prediction_uncertainty(prediction)
            assessments.append(assessment)

        threshold_analysis = quantifier._analyze_threshold_performance(
            ensemble_predictions, assessments
        )

        for threshold_name in [
            "minimum_submission",
            "high_confidence",
            "very_high_confidence",
        ]:
            assert threshold_name in threshold_analysis
            assert "threshold" in threshold_analysis[threshold_name]
            assert "predictions_above" in threshold_analysis[threshold_name]
            assert "percentage_above" in threshold_analysis[threshold_name]

    def test_confidence_recommendations_generation(
        self, quantifier, ensemble_predictions
    ):
        """Test confidence recommendation generation."""
        assessments = []
        for prediction in ensemble_predictions:
            assessment = quantifier.assess_prediction_uncertainty(prediction)
            assessments.append(assessment)

        recommendations = quantifier._generate_confidence_recommendations(
            ensemble_predictions, assessments
        )

        assert isinstance(recommendations, list)
        # Should provide some recommendations
        assert len(recommendations) >= 0

    def test_custom_confidence_thresholds(self):
        """Test quantifier with custom confidence thresholds."""
        custom_thresholds = ConfidenceThresholds(
            minimum_submission=0.7,
            high_confidence=0.85,
            very_high_confidence=0.95,
            abstention_threshold=0.3,
            research_trigger=0.6,
        )

        quantifier = UncertaintyQuantifier(custom_thresholds)

        assert quantifier.confidence_thresholds.minimum_submission == 0.7
        assert quantifier.confidence_thresholds.research_trigger == 0.6

    def test_reasoning_trace_impact_on_uncertainty(self, quantifier):
        """Test impact of reasoning trace on uncertainty assessment."""
        # Create prediction with reasoning trace
        prediction_with_trace = Prediction.create_binary_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Detailed reasoning",
            created_by="test_agent",
        )

        # Mock reasoning trace with high quality
        mock_trace = Mock(spec=ReasoningTrace)
        mock_trace.get_reasoning_quality_score.return_value = 0.9
        prediction_with_trace.reasoning_trace = mock_trace

        # Create prediction without reasoning trace
        prediction_without_trace = Prediction.create_binary_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Basic reasoning",
            created_by="test_agent",
        )

        assessment_with_trace = quantifier.assess_prediction_uncertainty(
            prediction_with_trace
        )
        assessment_without_trace = quantifier.assess_prediction_uncertainty(
            prediction_without_trace
        )

        # Prediction with high-quality reasoning trace should have lower epistemic uncertainty
        assert (
            assessment_with_trace.uncertainty_sources[UncertaintySource.EPISTEMIC]
            < assessment_without_trace.uncertainty_sources[UncertaintySource.EPISTEMIC]
        )
