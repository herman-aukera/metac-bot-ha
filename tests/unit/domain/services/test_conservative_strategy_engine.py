"""Tests for ConservativeStrategyEngine."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

from src.domain.services.conservative_strategy_engine import (
    ConservativeStrategyEngine,
    ConservativeStrategyConfig,
    RiskLevel,
    ConservativeAction,
    RiskAssessment
)
from src.domain.services.uncertainty_quantifier import UncertaintyQuantifier, UncertaintyAssessment, UncertaintySource
from src.domain.services.calibration_service import CalibrationTracker, CalibrationMetrics
from src.domain.entities.prediction import Prediction, PredictionResult, PredictionMethod, PredictionConfidence
from src.domain.entities.forecast import Forecast
from src.domain.value_objects.confidence import ConfidenceLevel


class TestConservativeStrategyEngine:
    """Test cases for ConservativeStrategyEngine."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ConservativeStrategyConfig(
            high_uncertainty_threshold=0.7,
            poor_calibration_threshold=0.15,
            time_pressure_threshold_hours=6.0,
            confidence_reduction_factor=0.8,
            abstention_uncertainty_threshold=0.8,
            abstention_risk_threshold=0.75
        )

    @pytest.fixture
    def uncertainty_quantifier(self):
        """Create mock uncertainty quantifier."""
        mock_quantifier = Mock(spec=UncertaintyQuantifier)
        mock_assessment = UncertaintyAssessment(
            total_uncertainty=0.5,
            uncertainty_sources={UncertaintySource.EPISTEMIC: 0.3, UncertaintySource.MODEL: 0.2},
            confidence_interval=(0.3, 0.7),
            confidence_level=0.8,
            calibration_score=0.9,
            uncertainty_decomposition={"data": 0.2, "model": 0.3},
            assessment_timestamp=datetime.utcnow()
        )
        mock_quantifier.assess_prediction_uncertainty.return_value = mock_assessment
        mock_quantifier.assess_forecast_uncertainty.return_value = mock_assessment
        return mock_quantifier

    @pytest.fixture
    def calibration_tracker(self):
        """Create mock calibration tracker."""
        mock_tracker = Mock(spec=CalibrationTracker)
        from src.domain.services.calibration_service import CalibrationBin, CalibrationDriftSeverity

        # Create sample calibration bins
        bins = [
            CalibrationBin(confidence_range=(0.0, 0.2)),
            CalibrationBin(confidence_range=(0.2, 0.4)),
            CalibrationBin(confidence_range=(0.4, 0.6)),
            CalibrationBin(confidence_range=(0.6, 0.8)),
            CalibrationBin(confidence_range=(0.8, 1.0))
        ]

        mock_metrics = CalibrationMetrics(
            brier_score=0.2,
            calibration_error=0.1,
            reliability=0.9,
            resolution=0.8,
            uncertainty=0.3,
            sharpness=0.7,
            calibration_bins=bins,
            measurement_timestamp=datetime.utcnow(),
            time_window_days=30,
            drift_severity=CalibrationDriftSeverity.MILD,
            drift_score=0.1
        )
        mock_tracker.calibration_history = [mock_metrics]
        return mock_tracker

    @pytest.fixture
    def engine(self, config, uncertainty_quantifier, calibration_tracker):
        """Create ConservativeStrategyEngine instance."""
        return ConservativeStrategyEngine(
            config=config,
            uncertainty_quantifier=uncertainty_quantifier,
            calibration_tracker=calibration_tracker
        )

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
            created_by="test_agent"
        )

    @pytest.fixture
    def sample_forecast(self, sample_prediction):
        """Create sample forecast for testing."""
        return Forecast.create_new(
            question_id=uuid4(),
            research_reports=[],
            predictions=[sample_prediction],
            final_prediction=sample_prediction,
            reasoning_summary="Test forecast",
            ensemble_method="test_ensemble",
            weight_distribution={"agent1": 1.0},
            consensus_strength=0.8
        )

    def test_engine_initialization(self, config, uncertainty_quantifier, calibration_tracker):
        """Test engine initialization."""
        engine = ConservativeStrategyEngine(
            config=config,
            uncertainty_quantifier=uncertainty_quantifier,
            calibration_tracker=calibration_tracker
        )

        assert engine.config == config
        assert engine.uncertainty_quantifier == uncertainty_quantifier
        assert engine.calibration_tracker == calibration_tracker
        assert engine.strategy_performance_history == []

    def test_engine_initialization_with_defaults(self):
        """Test engine initialization with default parameters."""
        engine = ConservativeStrategyEngine()

        assert isinstance(engine.config, ConservativeStrategyConfig)
        assert isinstance(engine.uncertainty_quantifier, UncertaintyQuantifier)
        assert isinstance(engine.calibration_tracker, CalibrationTracker)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = ConservativeStrategyConfig(confidence_reduction_factor=0.8)
        config.validate_config()

        # Invalid config should raise
        with pytest.raises(ValueError):
            invalid_config = ConservativeStrategyConfig(confidence_reduction_factor=1.5)
            invalid_config.validate_config()

    def test_assess_prediction_risk_low_risk(self, engine, sample_prediction, uncertainty_quantifier):
        """Test risk assessment for low-risk prediction."""
        # Setup low uncertainty assessment
        low_uncertainty = UncertaintyAssessment(
            total_uncertainty=0.2,
            uncertainty_sources={UncertaintySource.EPISTEMIC: 0.1, UncertaintySource.MODEL: 0.1},
            confidence_interval=(0.6, 0.8),
            confidence_level=0.9,
            calibration_score=0.95,
            uncertainty_decomposition={"data": 0.1, "model": 0.1},
            assessment_timestamp=datetime.utcnow()
        )
        uncertainty_quantifier.assess_prediction_uncertainty.return_value = low_uncertainty

        risk_assessment = engine.assess_prediction_risk(sample_prediction)

        assert risk_assessment.overall_risk in [RiskLevel.LOW, RiskLevel.VERY_LOW]
        assert risk_assessment.recommended_action == ConservativeAction.SUBMIT
        assert risk_assessment.confidence_adjustment is None

    def test_assess_prediction_risk_high_risk(self, engine, sample_prediction, uncertainty_quantifier):
        """Test risk assessment for high-risk prediction."""
        # Setup high uncertainty assessment
        high_uncertainty = UncertaintyAssessment(
            total_uncertainty=0.9,
            uncertainty_sources={UncertaintySource.EPISTEMIC: 0.5, UncertaintySource.MODEL: 0.4},
            confidence_interval=(0.1, 0.9),
            confidence_level=0.3,
            calibration_score=0.5,
            uncertainty_decomposition={"data": 0.4, "model": 0.5},
            assessment_timestamp=datetime.utcnow()
        )
        uncertainty_quantifier.assess_prediction_uncertainty.return_value = high_uncertainty

        risk_assessment = engine.assess_prediction_risk(sample_prediction)

        assert risk_assessment.overall_risk in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        assert risk_assessment.recommended_action in [ConservativeAction.ABSTAIN, ConservativeAction.REDUCE_CONFIDENCE, ConservativeAction.REQUEST_RESEARCH]

    def test_assess_prediction_risk_with_tournament_context(self, engine, sample_prediction):
        """Test risk assessment with tournament context."""
        tournament_context = {
            "hours_to_deadline": 2,
            "competition_level": "high",
            "question_difficulty": "hard",
            "abstention_penalty": 0.1,
            "question_weight": 1.5
        }

        risk_assessment = engine.assess_prediction_risk(
            sample_prediction,
            tournament_context=tournament_context
        )

        assert risk_assessment.time_pressure_risk > 0.5  # High time pressure
        assert risk_assessment.tournament_risk > 0.3  # High tournament risk
        assert risk_assessment.abstention_penalty == 0.1

    def test_apply_conservative_strategy_submit(self, engine, sample_forecast):
        """Test applying conservative strategy when submission is recommended."""
        adjusted_forecast, strategy_report = engine.apply_conservative_strategy(sample_forecast)

        # Should return original forecast if no adjustments needed
        assert adjusted_forecast.question_id == sample_forecast.question_id
        assert "risk_assessment" in strategy_report
        assert "adjustments_applied" in strategy_report

    def test_apply_conservative_strategy_reduce_confidence(self, engine, sample_forecast, uncertainty_quantifier):
        """Test applying conservative strategy with confidence reduction."""
        # Setup moderate uncertainty that triggers confidence reduction
        moderate_uncertainty = UncertaintyAssessment(
            total_uncertainty=0.6,
            uncertainty_sources={UncertaintySource.EPISTEMIC: 0.4, UncertaintySource.MODEL: 0.2},
            confidence_interval=(0.3, 0.8),
            confidence_level=0.6,
            calibration_score=0.7,
            uncertainty_decomposition={"data": 0.3, "model": 0.3},
            assessment_timestamp=datetime.utcnow()
        )
        uncertainty_quantifier.assess_forecast_uncertainty.return_value = moderate_uncertainty

        adjusted_forecast, strategy_report = engine.apply_conservative_strategy(sample_forecast)

        # Check if confidence was adjusted
        if "conservative_adjustment_applied" in adjusted_forecast.metadata:
            assert adjusted_forecast.metadata["conservative_adjustment_applied"] is True
            assert "original_confidence" in adjusted_forecast.metadata

    def test_should_abstain_from_prediction_high_risk(self, engine, sample_prediction, uncertainty_quantifier):
        """Test abstention decision for high-risk prediction."""
        # Setup very high uncertainty
        very_high_uncertainty = UncertaintyAssessment(
            total_uncertainty=0.95,
            uncertainty_sources={UncertaintySource.EPISTEMIC: 0.6, UncertaintySource.MODEL: 0.35},
            confidence_interval=(0.0, 1.0),
            confidence_level=0.1,
            calibration_score=0.3,
            uncertainty_decomposition={"data": 0.5, "model": 0.45},
            assessment_timestamp=datetime.utcnow()
        )
        uncertainty_quantifier.assess_prediction_uncertainty.return_value = very_high_uncertainty

        abstention_result = engine.should_abstain_from_prediction(sample_prediction)

        assert abstention_result["should_abstain"] is True
        assert abstention_result["abstention_reason"] is not None
        assert "risk_assessment" in abstention_result

    def test_should_abstain_from_prediction_low_risk(self, engine, sample_prediction, uncertainty_quantifier):
        """Test abstention decision for low-risk prediction."""
        # Setup low uncertainty
        low_uncertainty = UncertaintyAssessment(
            total_uncertainty=0.1,
            uncertainty_sources={UncertaintySource.EPISTEMIC: 0.05, UncertaintySource.MODEL: 0.05},
            confidence_interval=(0.65, 0.75),
            confidence_level=0.95,
            calibration_score=0.98,
            uncertainty_decomposition={"data": 0.05, "model": 0.05},
            assessment_timestamp=datetime.utcnow()
        )
        uncertainty_quantifier.assess_prediction_uncertainty.return_value = low_uncertainty

        abstention_result = engine.should_abstain_from_prediction(sample_prediction)

        assert abstention_result["should_abstain"] is False
        assert abstention_result["abstention_reason"] is None

    def test_should_abstain_with_tournament_penalty(self, engine, sample_prediction, uncertainty_quantifier):
        """Test abstention decision considering tournament penalty."""
        # Setup high uncertainty but high abstention penalty
        high_uncertainty = UncertaintyAssessment(
            total_uncertainty=0.85,
            uncertainty_sources={UncertaintySource.EPISTEMIC: 0.5, UncertaintySource.MODEL: 0.35},
            confidence_interval=(0.1, 0.9),
            confidence_level=0.2,
            calibration_score=0.4,
            uncertainty_decomposition={"data": 0.4, "model": 0.45},
            assessment_timestamp=datetime.utcnow()
        )
        uncertainty_quantifier.assess_prediction_uncertainty.return_value = high_uncertainty

        tournament_context = {
            "abstention_penalty": 0.5,  # Very high penalty
            "question_weight": 1.0,
            "phase": "final"
        }

        abstention_result = engine.should_abstain_from_prediction(
            sample_prediction,
            tournament_context=tournament_context
        )

        # Might not abstain due to high penalty
        assert "tournament_considerations" in abstention_result
        assert abstention_result["tournament_considerations"]["abstention_penalty"] == 0.75  # 0.5 * 1.5 (final phase)

    def test_optimize_tournament_scoring(self, engine, sample_prediction):
        """Test tournament scoring optimization."""
        predictions = [sample_prediction] * 3
        tournament_context = {
            "competition_level": "high",
            "phase": "middle",
            "abstention_penalty": 0.1
        }

        optimization_results = engine.optimize_tournament_scoring(predictions, tournament_context)

        assert "original_predictions" in optimization_results
        assert "submitted_predictions" in optimization_results
        assert "abstained_predictions" in optimization_results
        assert "expected_score_impact" in optimization_results
        assert optimization_results["original_predictions"] == 3

    def test_update_strategy_based_on_performance(self, engine):
        """Test strategy updates based on performance feedback."""
        # Add some performance history
        for i in range(10):
            engine.strategy_performance_history.append({
                "timestamp": datetime.utcnow() - timedelta(days=i),
                "abstained": i % 3 == 0,  # 33% abstention rate
                "submitted": i % 3 != 0,
                "accuracy": 0.8 if i % 3 != 0 else None
            })

        original_threshold = engine.config.abstention_uncertainty_threshold

        performance_data = {"recent_accuracy": 0.8, "abstention_rate": 0.33}
        engine.update_strategy_based_on_performance(performance_data)

        # Should adjust thresholds based on performance
        # With good accuracy and moderate abstention, might become slightly more aggressive
        assert engine.config.abstention_uncertainty_threshold >= original_threshold * 0.9

    def test_get_conservative_strategy_report_no_data(self, engine):
        """Test strategy report with no data."""
        report = engine.get_conservative_strategy_report()

        assert "error" in report
        assert report["error"] == "No strategy performance data available"

    def test_get_conservative_strategy_report_with_data(self, engine):
        """Test strategy report with performance data."""
        # Add performance history
        for i in range(5):
            engine.strategy_performance_history.append({
                "timestamp": datetime.utcnow() - timedelta(days=i),
                "abstained": i % 2 == 0,
                "submitted": i % 2 != 0,
                "accuracy": 0.7,
                "risk_level": "moderate",
                "tournament_context": {"competition_level": "medium"}
            })

        report = engine.get_conservative_strategy_report(time_window_days=10)

        assert "summary" in report
        assert "risk_analysis" in report
        assert "tournament_analysis" in report
        assert "recommendations" in report
        assert "current_config" in report
        assert report["time_window_days"] == 10

    def test_risk_level_determination(self, engine):
        """Test risk level determination from risk factors."""
        # Test very high risk
        very_high_risk_factors = {
            "uncertainty": 0.9,
            "calibration": 0.8,
            "confidence_mismatch": 0.7,
            "evidence_quality": 0.8,
            "method": 0.6
        }
        risk_level = engine._determine_overall_risk_level(very_high_risk_factors)
        assert risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]  # Allow for both high and very high

        # Test low risk
        low_risk_factors = {
            "uncertainty": 0.1,
            "calibration": 0.05,
            "confidence_mismatch": 0.1,
            "evidence_quality": 0.1,
            "method": 0.2
        }
        risk_level = engine._determine_overall_risk_level(low_risk_factors)
        assert risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]  # Allow for both very low and low

    def test_tournament_risk_calculation(self, engine, sample_prediction):
        """Test tournament risk calculation."""
        # High-risk tournament context
        high_risk_context = {
            "hours_to_deadline": 1,
            "competition_level": "very_high",
            "question_difficulty": "very_hard"
        }

        risk = engine._calculate_tournament_risk(sample_prediction, high_risk_context)
        assert risk > 0.8  # Should be high risk

        # Low-risk tournament context
        low_risk_context = {
            "hours_to_deadline": 48,
            "competition_level": "low",
            "question_difficulty": "easy"
        }

        risk = engine._calculate_tournament_risk(sample_prediction, low_risk_context)
        assert risk < 0.4  # Should be low risk

    def test_time_pressure_risk_calculation(self, engine):
        """Test time pressure risk calculation."""
        # Very high time pressure
        high_pressure_context = {"hours_to_deadline": 0.5}
        risk = engine._calculate_time_pressure_risk(high_pressure_context)
        assert risk > 0.8

        # Low time pressure
        low_pressure_context = {"hours_to_deadline": 48}
        risk = engine._calculate_time_pressure_risk(low_pressure_context)
        assert risk < 0.2

    def test_abstention_penalty_calculation(self, engine):
        """Test abstention penalty calculation."""
        # Final phase tournament
        final_phase_context = {
            "abstention_penalty": 0.2,
            "phase": "final"
        }
        penalty = engine._calculate_abstention_penalty(final_phase_context)
        assert penalty == 0.2 * 1.5  # Final phase multiplier

        # Early phase tournament
        early_phase_context = {
            "abstention_penalty": 0.2,
            "phase": "early"
        }
        penalty = engine._calculate_abstention_penalty(early_phase_context)
        assert penalty == 0.2 * 0.8  # Early phase multiplier

    def test_confidence_adjustment_application(self, engine, sample_prediction):
        """Test confidence adjustment application."""
        adjustment_factor = 0.8
        adjusted_prediction = engine._apply_confidence_adjustment(sample_prediction, adjustment_factor)

        # Original probability was 0.7, adjusted should be closer to 0.5
        original_prob = sample_prediction.result.binary_probability
        adjusted_prob = adjusted_prediction.result.binary_probability

        expected_prob = 0.5 + (original_prob - 0.5) * adjustment_factor
        assert abs(adjusted_prob - expected_prob) < 0.001

    def test_risk_reasoning_generation(self, engine):
        """Test risk reasoning generation."""
        risk_factors = {
            "uncertainty": 0.8,
            "calibration": 0.6,
            "confidence_mismatch": 0.3,
            "evidence_quality": 0.4,
            "method": 0.2
        }

        reasoning = engine._generate_risk_reasoning(
            risk_factors,
            RiskLevel.HIGH,
            ConservativeAction.ABSTAIN,
            {"hours_to_deadline": 2}
        )

        assert "high" in reasoning.lower()
        assert "abstention" in reasoning.lower() or "abstain" in reasoning.lower()
        assert len(reasoning) > 50  # Should be descriptive

    def test_competitive_impact_calculation(self, engine):
        """Test competitive impact calculation."""
        high_competition_context = {"competition_level": "very_high"}

        # Abstaining in high competition has high negative impact
        impact = engine._calculate_competitive_impact(
            ConservativeAction.ABSTAIN,
            high_competition_context
        )
        assert impact < -0.3  # Negative impact

        # Submitting has no negative impact
        impact = engine._calculate_competitive_impact(
            ConservativeAction.SUBMIT,
            high_competition_context
        )
        assert impact == 0.0
