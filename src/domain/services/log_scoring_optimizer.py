"""Log scoring optimizer for tournament performance."""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..entities.prediction import Prediction


@dataclass
class LogScoringAnalysis:
    """Analysis of log scoring performance for a prediction."""

    prediction_value: float
    expected_log_score: float
    risk_level: str  # "low", "medium", "high", "extreme"
    optimal_range: Tuple[float, float]
    risk_factors: List[str]
    recommendations: List[str]


class LogScoringOptimizer:
    """Optimizer for log scoring performance in tournaments."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Log scoring optimization parameters
        self.safe_range = (0.1, 0.9)  # Safe range to avoid extreme penalties
        self.conservative_range = (0.2, 0.8)  # More conservative range
        self.extreme_penalty_threshold = (
            0.05  # Values closer than this to 0/1 are extreme
        )

        # Risk thresholds for log scoring
        self.risk_thresholds = {
            "low": 2.0,  # Expected log score better than -2.0
            "medium": 3.0,  # Expected log score better than -3.0
            "high": 4.0,  # Expected log score better than -4.0
            "extreme": 4.0,  # Expected log score worse than -4.0
        }

    def analyze_log_scoring_risk(self, prediction_value: float) -> LogScoringAnalysis:
        """Analyze the log scoring risk for a prediction value."""

        # Calculate expected log score (assuming 50% chance of being correct)
        expected_score = self._calculate_expected_log_score(prediction_value)

        # Determine risk level
        risk_level = self._determine_risk_level(expected_score)

        # Find optimal range based on risk tolerance
        optimal_range = self._find_optimal_range(prediction_value, risk_level)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(prediction_value)

        # Generate recommendations
        recommendations = self._generate_recommendations(prediction_value, risk_level)

        return LogScoringAnalysis(
            prediction_value=prediction_value,
            expected_log_score=expected_score,
            risk_level=risk_level,
            optimal_range=optimal_range,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )

    def optimize_prediction_for_log_scoring(
        self,
        prediction: Prediction,
        risk_tolerance: str = "medium",
        preserve_direction: bool = True,
    ) -> Tuple[Prediction, LogScoringAnalysis]:
        """Optimize a prediction for log scoring performance."""

        original_value = self._extract_prediction_value(prediction)
        if original_value is None:
            return prediction, LogScoringAnalysis(
                prediction_value=0.5,
                expected_log_score=math.log(2),
                risk_level="medium",
                optimal_range=(0.4, 0.6),
                risk_factors=["no_numeric_value"],
                recommendations=["Cannot optimize without numeric prediction value"],
            )

        # Analyze current risk
        analysis = self.analyze_log_scoring_risk(original_value)

        # Optimize if needed
        optimized_value = self._optimize_value(
            original_value, risk_tolerance, preserve_direction
        )

        # Create optimized prediction if value changed
        if abs(optimized_value - original_value) > 0.001:
            optimized_prediction = self._create_optimized_prediction(
                prediction, optimized_value, analysis
            )

            # Update analysis with optimized value
            analysis = self.analyze_log_scoring_risk(optimized_value)

            self.logger.info(
                "Optimized prediction for log scoring",
                original_value=original_value,
                optimized_value=optimized_value,
                risk_improvement=analysis.risk_level,
            )
        else:
            optimized_prediction = prediction

        return optimized_prediction, analysis

    def calculate_log_score(
        self, prediction_value: float, actual_outcome: bool
    ) -> float:
        """Calculate the actual log score for a prediction and outcome."""

        if prediction_value <= 0 or prediction_value >= 1:
            return float("-inf")  # Infinite penalty for impossible predictions

        if actual_outcome:
            return math.log(prediction_value)
        else:
            return math.log(1 - prediction_value)

    def simulate_log_scoring_performance(
        self, prediction_values: List[float], accuracy_rate: float = 0.6
    ) -> Dict[str, float]:
        """Simulate log scoring performance for a set of predictions."""

        total_score = 0.0
        scores = []

        for value in prediction_values:
            # Simulate outcome based on accuracy rate
            # Higher accuracy for predictions closer to the truth
            if value > 0.5:
                outcome_prob = accuracy_rate + (value - 0.5) * 0.2
            else:
                outcome_prob = accuracy_rate - (0.5 - value) * 0.2

            outcome_prob = max(0.1, min(0.9, outcome_prob))

            # Calculate expected score
            expected_score = outcome_prob * math.log(value) + (
                1 - outcome_prob
            ) * math.log(1 - value)

            scores.append(expected_score)
            total_score += expected_score

        return {
            "total_score": total_score,
            "average_score": (
                total_score / len(prediction_values) if prediction_values else 0
            ),
            "best_score": max(scores) if scores else 0,
            "worst_score": min(scores) if scores else 0,
            "score_variance": np.var(scores) if scores else 0,
        }

    def _calculate_expected_log_score(self, prediction_value: float) -> float:
        """Calculate expected log score assuming unknown true probability."""

        if prediction_value <= 0 or prediction_value >= 1:
            return float("-inf")

        # Assume 50% chance of being correct (worst case for log scoring)
        expected_score = 0.5 * math.log(prediction_value) + 0.5 * math.log(
            1 - prediction_value
        )
        return expected_score

    def _determine_risk_level(self, expected_score: float) -> str:
        """Determine risk level based on expected log score."""

        if expected_score >= -self.risk_thresholds["low"]:
            return "low"
        elif expected_score >= -self.risk_thresholds["medium"]:
            return "medium"
        elif expected_score >= -self.risk_thresholds["high"]:
            return "high"
        else:
            return "extreme"

    def _find_optimal_range(
        self, prediction_value: float, risk_level: str
    ) -> Tuple[float, float]:
        """Find optimal range for prediction based on risk level."""

        if risk_level == "low":
            return (0.3, 0.7)  # Conservative range
        elif risk_level == "medium":
            return (0.2, 0.8)  # Moderate range
        elif risk_level == "high":
            return (0.15, 0.85)  # Wider range but still safe
        else:  # extreme
            return (0.1, 0.9)  # Maximum safe range

    def _identify_risk_factors(self, prediction_value: float) -> List[str]:
        """Identify risk factors for log scoring."""

        risk_factors = []

        if prediction_value < self.extreme_penalty_threshold:
            risk_factors.append("extreme_low_prediction")
        elif prediction_value > (1 - self.extreme_penalty_threshold):
            risk_factors.append("extreme_high_prediction")

        if prediction_value < 0.1 or prediction_value > 0.9:
            risk_factors.append("outside_safe_range")

        if prediction_value < 0.2 or prediction_value > 0.8:
            risk_factors.append("high_confidence_risk")

        # Calculate distance from optimal (0.5 for unknown true probability)
        distance_from_optimal = abs(prediction_value - 0.5)
        if distance_from_optimal > 0.3:
            risk_factors.append("far_from_optimal")

        return risk_factors

    def _generate_recommendations(
        self, prediction_value: float, risk_level: str
    ) -> List[str]:
        """Generate recommendations for improving log scoring performance."""

        recommendations = []

        if risk_level == "extreme":
            recommendations.append("URGENT: Move prediction away from extreme values")
            if prediction_value < 0.1:
                recommendations.append(
                    f"Increase prediction to at least 0.1 (currently {prediction_value:.3f})"
                )
            elif prediction_value > 0.9:
                recommendations.append(
                    f"Decrease prediction to at most 0.9 (currently {prediction_value:.3f})"
                )

        elif risk_level == "high":
            recommendations.append(
                "Consider moving prediction toward safer range (0.2-0.8)"
            )

        elif risk_level == "medium":
            recommendations.append(
                "Prediction is in acceptable range but could be more conservative"
            )

        else:  # low risk
            recommendations.append("Prediction is in low-risk range for log scoring")

        # Specific recommendations based on value
        if prediction_value > 0.8:
            recommendations.append(
                "High confidence prediction - ensure strong evidence supports this"
            )
        elif prediction_value < 0.2:
            recommendations.append(
                "Low confidence prediction - ensure this reflects true uncertainty"
            )

        return recommendations

    def _optimize_value(
        self, original_value: float, risk_tolerance: str, preserve_direction: bool
    ) -> float:
        """Optimize prediction value for log scoring."""

        # Define target ranges based on risk tolerance
        target_ranges = {
            "conservative": (0.25, 0.75),
            "medium": (0.15, 0.85),
            "aggressive": (0.1, 0.9),
        }

        target_range = target_ranges.get(risk_tolerance, target_ranges["medium"])
        min_val, max_val = target_range

        # If already in range, minimal adjustment
        if min_val <= original_value <= max_val:
            # Small adjustment toward center if very close to edges
            if original_value < min_val + 0.05:
                return min_val + 0.05
            elif original_value > max_val - 0.05:
                return max_val - 0.05
            else:
                return original_value

        # If outside range, move to nearest safe value
        if original_value < min_val:
            optimized = min_val
        else:  # original_value > max_val
            optimized = max_val

        # If preserve_direction is True, try to maintain the direction of confidence
        if preserve_direction:
            if original_value > 0.5 and optimized < 0.5:
                optimized = 0.5 + (0.5 - optimized)  # Flip to maintain direction
            elif original_value < 0.5 and optimized > 0.5:
                optimized = 0.5 - (optimized - 0.5)  # Flip to maintain direction

        return optimized

    def _extract_prediction_value(self, prediction: Prediction) -> Optional[float]:
        """Extract numeric prediction value."""

        if prediction.result.binary_probability is not None:
            return prediction.result.binary_probability
        elif prediction.result.numeric_value is not None:
            return prediction.result.numeric_value
        elif prediction.result.multiple_choice_probabilities:
            return max(prediction.result.multiple_choice_probabilities.values())

        return None

    def _create_optimized_prediction(
        self, original: Prediction, optimized_value: float, analysis: LogScoringAnalysis
    ) -> Prediction:
        """Create optimized prediction with new value."""

        # Create new result with optimized value
        if original.result.binary_probability is not None:
            from ..entities.prediction import PredictionResult

            new_result = PredictionResult(binary_probability=optimized_value)
        elif original.result.numeric_value is not None:
            from ..entities.prediction import PredictionResult

            new_result = PredictionResult(numeric_value=optimized_value)
        else:
            new_result = original.result

        # Create optimized prediction
        optimized = Prediction(
            id=original.id,
            question_id=original.question_id,
            research_report_id=original.research_report_id,
            result=new_result,
            confidence=original.confidence,
            method=original.method,
            reasoning=original.reasoning,
            reasoning_steps=original.reasoning_steps
            + [
                f"Applied log scoring optimization: {self._extract_prediction_value(original):.3f} → {optimized_value:.3f}"
            ],
            created_at=original.created_at,
            created_by=original.created_by,
            method_metadata={
                **original.method_metadata,
                "log_scoring_optimized": True,
                "original_value": self._extract_prediction_value(original),
                "optimization_risk_level": analysis.risk_level,
                "expected_log_score": analysis.expected_log_score,
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
                setattr(optimized, attr, getattr(original, attr))

        return optimized
