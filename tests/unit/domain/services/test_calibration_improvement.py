"""
Tests for calibration improvement and overconfidence reduction.
"""
import pytest
import math
from unittest.mock import Mock, patch
from datetime import datetime

from src.domain.services.tournament_calibration_service import (
    TournamentCalibrationService, CalibrationAdjustment, CommunityPredictionData
)
from src.domain.entities.prediction import Prediction, PredictionConfidence
from src.domain.value_objects.probability import Probability


class TestCalibrationImprovement:
    """Test calibration improvement mechanisms."""

    def setup_method(self):
        """Set up test environment."""
        self.calibration_service = TournamentCalibrationService()

    def test_overconfidence_mitigation(self):
        """Test overconfidence mitigation for extreme predictions."""
        # Create overconfident prediction (too close to extreme)
        overconfident_prediction = Mock(spec=Prediction)
        overconfident_prediction.probability = Probability(0.98)  # Very high confidence
        overconfident_prediction.confidence = PredictionConfidence.VERY_HIGH

        # Apply calibration
        calibrated_pred, adjustment = self.calibration_service.calibrate_prediction(
            overconfident_prediction
        )

        # Should reduce extreme confidence
        calibrated_value = self.calibration_service._extract_prediction_value(calibrated_pred)
        original_value = 0.98

        assert calibrated_value < original_value, "Should reduce overconfident prediction"
        assert calibrated_value > 0.5, "Should still indicate positive prediction"
        assert adjustment.adjustment_type in ["overconfidence", "extreme_avoidance"]

    def test_extreme_value_avoidance(self):
        """Test avoidance of extreme values for log scoring optimization."""
        extreme_cases = [
            (0.01, "very low"),
            (0.99, "very high"),
            (0.001, "extremely low"),
            (0.999, "extremely high")
        ]

        for extreme_value, description in extreme_cases:
            # Create prediction with extreme value
            extreme_prediction = Mock(spec=Prediction)
            extreme_prediction.probability = Probability(extreme_value)
            extreme_prediction.confidence = PredictionConfidence.HIGH

            # Apply calibration
            calibrated_pred, adjustment = self.calibration_service.calibrate_prediction(
                extreme_prediction
            )

            calibrated_value = self.calibration_service._extract_prediction_value(calibrated_pred)

            # Should pull back from extremes
            if extreme_value < 0.5:
                assert calibrated_value > extreme_value, f"Should increase {description} value"
                assert calibrated_value < 0.5, f"Should maintain {description} direction"
            else:
                assert calibrated_value < extreme_value, f"Should decrease {description} value"
                assert calibrated_value > 0.5, f"Should maintain {description} direction"
    def test_community_anchoring_adjustment(self):
        """Test community anchoring for calibration improvement."""
        # Create individual prediction
        individual_prediction = Mock(spec=Prediction)
        individual_prediction.probability = Probability(0.8)
        individual_prediction.confidence = PredictionConfidence.MEDIUM

        # Create community data
        community_data = CommunityPredictionData(
            median_prediction=0.6,
            mean_prediction=0.65,
            prediction_count=50,
            confidence_interval=(0.5, 0.7),
            last_updated=datetime.now()
        )

        # Apply calibration with community anchoring
        calibrated_pred, adjustment = self.calibration_service.calibrate_prediction(
            individual_prediction, community_data=community_data
        )

        calibrated_value = self.calibration_service._extract_prediction_value(calibrated_pred)

        # Should move toward community consensus
        assert calibrated_value < 0.8, "Should adjust toward community median"
        assert calibrated_value > 0.6, "Should not fully adopt community median"
        assert "anchoring" in adjustment.adjustment_type.lower() or "community" in adjustment.reasoning.lower()

    def test_confidence_level_adjustments(self):
        """Test different adjustment rates based on confidence levels."""
        base_prediction_value = 0.85
        confidence_levels = [
            PredictionConfidence.VERY_HIGH,
            PredictionConfidence.HIGH,
            PredictionConfidence.MEDIUM,
            PredictionConfidence.LOW,
            PredictionConfidence.VERY_LOW
        ]

        adjustments = []

        for confidence in confidence_levels:
            prediction = Mock(spec=Prediction)
            prediction.probability = Probability(base_prediction_value)
            prediction.confidence = confidence

            calibrated_pred, adjustment = self.calibration_service.calibrate_prediction(prediction)
            calibrated_value = self.calibration_service._extract_prediction_value(calibrated_pred)

            adjustment_magnitude = abs(calibrated_value - base_prediction_value)
            adjustments.append((confidence, adjustment_magnitude))

        # Higher confidence should receive larger adjustments
        very_high_adj = next(adj for conf, adj in adjustments if conf == PredictionConfidence.VERY_HIGH)
        low_adj = next(adj for conf, adj in adjustments if conf == PredictionConfidence.LOW)

        assert very_high_adj > low_adj, "Very high confidence should be adjusted more than low confidence"

    def test_log_scoring_optimization(self):
        """Test calibration adjustments optimize log scoring."""
        # Test predictions that would have poor log scores
        risky_predictions = [
            (0.95, True),   # High confidence, correct
            (0.95, False),  # High confidence, incorrect (should be heavily penalized)
            (0.05, True),   # Low confidence, correct (should be heavily penalized)
            (0.05, False),  # Low confidence, incorrect
        ]

        for pred_value, actual_outcome in risky_predictions:
            prediction = Mock(spec=Prediction)
            prediction.probability = Probability(pred_value)
            prediction.confidence = PredictionConfidence.HIGH

            calibrated_pred, adjustment = self.calibration_service.calibrate_prediction(prediction)
            calibrated_value = self.calibration_service._extract_prediction_value(calibrated_pred)

            # Calculate log scores
            original_log_score = self._calculate_log_score(pred_value, actual_outcome)
            calibrated_log_score = self._calculate_log_score(calibrated_value, actual_outcome)

            # Calibrated version should have better (less negative) log score on average
            # This is especially important for incorrect extreme predictions
            if (pred_value > 0.9 and not actual_outcome) or (pred_value < 0.1 and actual_outcome):
                assert calibrated_log_score > original_log_score, \
                    f"Calibration should improve log score for extreme incorrect prediction"

    def _calculate_log_score(self, prediction: float, outcome: bool) -> float:
        """Calculate log score for a prediction."""
        # Avoid log(0) by using small epsilon
        epsilon = 1e-10
        prob = max(epsilon, min(1 - epsilon, prediction))

        if outcome:
            return math.log(prob)
        else:
            return math.log(1 - prob)
    def test_historical_performance_adjustment(self):
        """Test calibration based on historical performance."""
        # Mock historical performance data
        historical_performance = {
            "overall_calibration_error": 0.15,  # 15% calibration error
            "overconfidence_bias": 0.08,        # Tends to be overconfident
            "recent_accuracy": 0.72              # 72% accuracy
        }

        prediction = Mock(spec=Prediction)
        prediction.probability = Probability(0.9)
        prediction.confidence = PredictionConfidence.HIGH

        # Apply calibration with historical performance
        calibrated_pred, adjustment = self.calibration_service.calibrate_prediction(
            prediction, historical_performance=historical_performance
        )

        calibrated_value = self.calibration_service._extract_prediction_value(calibrated_pred)

        # Should adjust based on historical overconfidence
        assert calibrated_value < 0.9, "Should reduce prediction due to historical overconfidence"
        assert calibrated_value > 0.5, "Should maintain prediction direction"

    def test_no_adjustment_for_well_calibrated_predictions(self):
        """Test that well-calibrated predictions receive minimal adjustment."""
        # Create well-calibrated prediction (moderate confidence, reasonable value)
        well_calibrated_prediction = Mock(spec=Prediction)
        well_calibrated_prediction.probability = Probability(0.65)
        well_calibrated_prediction.confidence = PredictionConfidence.MEDIUM

        # Apply calibration
        calibrated_pred, adjustment = self.calibration_service.calibrate_prediction(
            well_calibrated_prediction
        )

        calibrated_value = self.calibration_service._extract_prediction_value(calibrated_pred)

        # Should have minimal adjustment
        adjustment_magnitude = abs(calibrated_value - 0.65)
        assert adjustment_magnitude < 0.1, "Well-calibrated prediction should have minimal adjustment"

    def test_calibration_reasoning_quality(self):
        """Test that calibration adjustments include clear reasoning."""
        prediction = Mock(spec=Prediction)
        prediction.probability = Probability(0.95)
        prediction.confidence = PredictionConfidence.VERY_HIGH

        calibrated_pred, adjustment = self.calibration_service.calibrate_prediction(prediction)

        # Should provide clear reasoning
        assert len(adjustment.reasoning) > 10, "Should provide meaningful reasoning"
        assert any(keyword in adjustment.reasoning.lower() for keyword in
                  ["overconfidence", "extreme", "calibration", "adjustment"]), \
               "Reasoning should explain the type of adjustment"

    def teardown_method(self):
        """Clean up test environment."""
        pass
