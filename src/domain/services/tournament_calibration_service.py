"""Tournament calibration service for overconfidence mitigation and log scoring optimization."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..entities.forecast import Forecast
from ..entities.prediction import Prediction, PredictionConfidence, PredictionResult


@dataclass
class CalibrationAdjustment:
    """Represents a calibration adjustment applied to a prediction."""

    original_value: float
    adjusted_value: float
    adjustment_type: str  # "overconfidence", "anchoring", "extreme_avoidance"
    adjustment_factor: float
    reasoning: str


@dataclass
class CommunityPredictionData:
    """Community prediction data for anchoring strategies."""

    median_prediction: Optional[float] = None
    mean_prediction: Optional[float] = None
    prediction_count: int = 0
    confidence_interval: Optional[Tuple[float, float]] = None
    last_updated: Optional[datetime] = None


class TournamentCalibrationService:
    """Service for calibrating predictions to optimize tournament performance."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Calibration parameters for tournament optimization
        self.overconfidence_threshold = (
            0.05  # Avoid predictions closer than 5% to extremes
        )
        self.extreme_avoidance_factor = 0.1  # Pull back from extremes by 10%
        self.anchoring_weight = 0.2  # Weight for community anchoring
        self.confidence_adjustment_rates = {
            PredictionConfidence.VERY_HIGH: 0.15,  # Reduce very high confidence more
            PredictionConfidence.HIGH: 0.10,
            PredictionConfidence.MEDIUM: 0.05,
            PredictionConfidence.LOW: 0.02,
            PredictionConfidence.VERY_LOW: 0.01,
        }

    def calibrate_prediction(
        self,
        prediction: Prediction,
        community_data: Optional[CommunityPredictionData] = None,
        historical_performance: Optional[Dict[str, float]] = None,
    ) -> Tuple[Prediction, CalibrationAdjustment]:
        """Calibrate a prediction for tournament performance optimization."""

        original_value = self._extract_prediction_value(prediction)
        if original_value is None:
            # Can't calibrate without a numeric value
            return prediction, CalibrationAdjustment(
                original_value=0.0,
                adjusted_value=0.0,
                adjustment_type="no_adjustment",
                adjustment_factor=1.0,
                reasoning="No numeric value to calibrate",
            )

        adjusted_value = original_value
        adjustments_applied = []

        # 1. Apply overconfidence mitigation
        overconfidence_adjustment = self._apply_overconfidence_mitigation(
            adjusted_value, prediction.confidence
        )
        if overconfidence_adjustment != adjusted_value:
            adjustments_applied.append("overconfidence_mitigation")
            adjusted_value = overconfidence_adjustment

        # 2. Apply extreme value avoidance for log scoring
        extreme_adjustment = self._apply_extreme_avoidance(adjusted_value)
        if extreme_adjustment != adjusted_value:
            adjustments_applied.append("extreme_avoidance")
            adjusted_value = extreme_adjustment

        # 3. Apply community anchoring if available
        if community_data and community_data.median_prediction is not None:
            anchored_value = self._apply_community_anchoring(
                adjusted_value, community_data, prediction.confidence
            )
            if anchored_value != adjusted_value:
                adjustments_applied.append("community_anchoring")
                adjusted_value = anchored_value

        # 4. Apply historical performance adjustment if available
        if historical_performance:
            performance_adjusted = self._apply_historical_performance_adjustment(
                adjusted_value, historical_performance, prediction.method
            )
            if performance_adjusted != adjusted_value:
                adjustments_applied.append("historical_performance")
                adjusted_value = performance_adjusted

        # Create calibrated prediction
        calibrated_prediction = self._create_calibrated_prediction(
            prediction, adjusted_value
        )

        # Create adjustment record
        adjustment = CalibrationAdjustment(
            original_value=original_value,
            adjusted_value=adjusted_value,
            adjustment_type=(
                "+".join(adjustments_applied)
                if adjustments_applied
                else "no_adjustment"
            ),
            adjustment_factor=(
                adjusted_value / original_value if original_value != 0 else 1.0
            ),
            reasoning=self._generate_adjustment_reasoning(
                original_value, adjusted_value, adjustments_applied, community_data
            ),
        )

        self.logger.info(
            "Applied calibration adjustments",
            original_value=original_value,
            adjusted_value=adjusted_value,
            adjustments=adjustments_applied,
            adjustment_factor=adjustment.adjustment_factor,
        )

        return calibrated_prediction, adjustment

    def calibrate_forecast(
        self,
        forecast: Forecast,
        community_data: Optional[CommunityPredictionData] = None,
        historical_performance: Optional[Dict[str, float]] = None,
    ) -> Tuple[Forecast, List[CalibrationAdjustment]]:
        """Calibrate all predictions in a forecast."""

        calibrated_predictions = []
        adjustments = []

        for prediction in forecast.predictions:
            calibrated_pred, adjustment = self.calibrate_prediction(
                prediction, community_data, historical_performance
            )
            calibrated_predictions.append(calibrated_pred)
            adjustments.append(adjustment)

        # Create calibrated forecast
        calibrated_forecast = Forecast(
            id=forecast.id,
            question_id=forecast.question_id,
            predictions=calibrated_predictions,
            research_reports=forecast.research_reports,
            created_at=forecast.created_at,
            updated_at=datetime.utcnow(),
            ensemble_method=forecast.ensemble_method,
            weight_distribution=forecast.weight_distribution,
            reasoning_summary=self._update_reasoning_with_calibration(
                forecast.reasoning_summary, adjustments
            ),
            tournament_strategy=forecast.tournament_strategy,
            reasoning_traces=forecast.reasoning_traces,
        )

        return calibrated_forecast, adjustments

    def calculate_log_score_risk(self, prediction_value: float) -> float:
        """Calculate the log scoring risk for a prediction value."""
        if prediction_value <= 0 or prediction_value >= 1:
            return float("inf")  # Infinite risk for extreme values

        # Log score risk is higher near extremes
        # Risk = -log(p) for correct predictions, -log(1-p) for incorrect
        # We calculate expected risk assuming 50% chance of being correct
        risk_if_correct = -math.log(prediction_value)
        risk_if_incorrect = -math.log(1 - prediction_value)
        expected_risk = 0.5 * (risk_if_correct + risk_if_incorrect)

        return expected_risk

    def optimize_for_log_scoring(
        self, prediction_value: float, confidence: PredictionConfidence
    ) -> float:
        """Optimize prediction value specifically for log scoring performance."""

        # Calculate current risk
        current_risk = self.calculate_log_score_risk(prediction_value)

        # Test different adjustments to minimize risk
        best_value = prediction_value
        best_risk = current_risk

        # Test conservative adjustments based on confidence
        confidence_factor = self.confidence_adjustment_rates.get(confidence, 0.05)

        # Test moving toward 0.5 (most conservative for log scoring)
        conservative_adjustments = (
            [0.1, 0.2, 0.3] if confidence_factor > 0.05 else [0.05, 0.1]
        )

        for adjustment in conservative_adjustments:
            if prediction_value > 0.5:
                test_value = prediction_value - (prediction_value - 0.5) * adjustment
            else:
                test_value = prediction_value + (0.5 - prediction_value) * adjustment

            test_risk = self.calculate_log_score_risk(test_value)
            if test_risk < best_risk:
                best_value = test_value
                best_risk = test_risk

        return best_value

    def _extract_prediction_value(self, prediction: Prediction) -> Optional[float]:
        """Extract numeric prediction value from prediction result."""
        if prediction.result.binary_probability is not None:
            return prediction.result.binary_probability
        elif prediction.result.numeric_value is not None:
            return prediction.result.numeric_value
        elif prediction.result.multiple_choice_probabilities:
            # For multiple choice, use the highest probability
            return max(prediction.result.multiple_choice_probabilities.values())
        return None

    def _apply_overconfidence_mitigation(
        self, value: float, confidence: PredictionConfidence
    ) -> float:
        """Apply overconfidence mitigation based on confidence level."""

        # Higher confidence predictions get more adjustment
        adjustment_rate = self.confidence_adjustment_rates.get(confidence, 0.05)

        # Pull extreme values toward center
        if value > 0.5:
            # For values > 0.5, reduce by adjustment rate
            adjusted = value - (value - 0.5) * adjustment_rate
        else:
            # For values < 0.5, increase by adjustment rate
            adjusted = value + (0.5 - value) * adjustment_rate

        # Ensure we stay within bounds
        return max(0.01, min(0.99, adjusted))

    def _apply_extreme_avoidance(self, value: float) -> float:
        """Apply extreme value avoidance for log scoring protection."""

        # Avoid values too close to 0 or 1
        if value < self.overconfidence_threshold:
            return self.overconfidence_threshold
        elif value > (1 - self.overconfidence_threshold):
            return 1 - self.overconfidence_threshold

        return value

    def _apply_community_anchoring(
        self,
        value: float,
        community_data: CommunityPredictionData,
        confidence: PredictionConfidence,
    ) -> float:
        """Apply community prediction anchoring strategy."""

        if community_data.median_prediction is None:
            return value

        community_median = community_data.median_prediction

        # Adjust anchoring weight based on confidence and community size
        base_weight = self.anchoring_weight

        # Reduce anchoring for high confidence predictions
        if confidence in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]:
            base_weight *= 0.5

        # Increase anchoring if community has many predictions (more reliable)
        if community_data.prediction_count > 50:
            base_weight *= 1.2
        elif community_data.prediction_count < 10:
            base_weight *= 0.7

        # Apply weighted average
        anchored_value = (1 - base_weight) * value + base_weight * community_median

        return max(0.01, min(0.99, anchored_value))

    def _apply_historical_performance_adjustment(
        self, value: float, historical_performance: Dict[str, float], method
    ) -> float:
        """Apply adjustment based on historical performance of the method."""

        method_name = method.value if hasattr(method, "value") else str(method)

        # Get calibration factor for this method
        calibration_factor = historical_performance.get(
            f"{method_name}_calibration", 1.0
        )
        overconfidence_factor = historical_performance.get(
            f"{method_name}_overconfidence", 0.0
        )

        # Apply calibration adjustment
        if calibration_factor != 1.0:
            # Adjust toward 0.5 based on historical overconfidence
            if value > 0.5:
                adjusted = value - (value - 0.5) * overconfidence_factor
            else:
                adjusted = value + (0.5 - value) * overconfidence_factor

            return max(0.01, min(0.99, adjusted))

        return value

    def _create_calibrated_prediction(
        self, original: Prediction, adjusted_value: float
    ) -> Prediction:
        """Create a new prediction with calibrated value."""

        # Create new result with adjusted value
        if original.result.binary_probability is not None:
            new_result = PredictionResult(binary_probability=adjusted_value)
        elif original.result.numeric_value is not None:
            new_result = PredictionResult(numeric_value=adjusted_value)
        else:
            new_result = original.result  # Keep original if can't adjust

        # Create new prediction with calibrated result
        calibrated = Prediction(
            id=original.id,
            question_id=original.question_id,
            research_report_id=original.research_report_id,
            result=new_result,
            confidence=original.confidence,
            method=original.method,
            reasoning=original.reasoning,
            reasoning_steps=original.reasoning_steps
            + [f"Applied tournament calibration: {original.result} → {new_result}"],
            created_at=original.created_at,
            created_by=original.created_by,
            method_metadata={
                **original.method_metadata,
                "calibration_applied": True,
                "original_value": self._extract_prediction_value(original),
                "calibration_timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Copy other attributes
        for attr in [
            "lower_bound",
            "upper_bound",
            "confidence_interval",
            "internal_consistency_score",
            "evidence_strength",
            "reasoning_trace",
            "bias_checks_performed",
            "uncertainty_quantification",
            "calibration_data",
        ]:
            if hasattr(original, attr):
                setattr(calibrated, attr, getattr(original, attr))

        return calibrated

    def _generate_adjustment_reasoning(
        self,
        original_value: float,
        adjusted_value: float,
        adjustments_applied: List[str],
        community_data: Optional[CommunityPredictionData],
    ) -> str:
        """Generate reasoning for calibration adjustments."""

        if not adjustments_applied:
            return "No calibration adjustments applied"

        reasoning_parts = [
            f"Tournament calibration applied: {original_value:.3f} → {adjusted_value:.3f}"
        ]

        if "overconfidence_mitigation" in adjustments_applied:
            reasoning_parts.append(
                "Applied overconfidence mitigation to improve calibration"
            )

        if "extreme_avoidance" in adjustments_applied:
            reasoning_parts.append(
                "Applied extreme value avoidance for log scoring protection"
            )

        if "community_anchoring" in adjustments_applied and community_data:
            reasoning_parts.append(
                f"Applied community anchoring (median: {community_data.median_prediction:.3f}, "
                f"n={community_data.prediction_count})"
            )

        if "historical_performance" in adjustments_applied:
            reasoning_parts.append("Applied historical performance adjustment")

        return ". ".join(reasoning_parts)

    def _update_reasoning_with_calibration(
        self,
        original_reasoning: Optional[str],
        adjustments: List[CalibrationAdjustment],
    ) -> str:
        """Update forecast reasoning to include calibration information."""

        base_reasoning = original_reasoning or "Ensemble forecast"

        if not adjustments or all(
            adj.adjustment_type == "no_adjustment" for adj in adjustments
        ):
            return base_reasoning

        calibration_summary = []
        for i, adj in enumerate(adjustments):
            if adj.adjustment_type != "no_adjustment":
                calibration_summary.append(
                    f"Prediction {i + 1}: {adj.original_value:.3f} → {adj.adjusted_value:.3f} "
                    f"({adj.adjustment_type})"
                )

        if calibration_summary:
            calibration_text = "\n\nTournament Calibration Applied:\n" + "\n".join(
                calibration_summary
            )
            return base_reasoning + calibration_text

        return base_reasoning
