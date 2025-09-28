"""Uncertainty quantification and confidence management service."""

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..entities.forecast import Forecast
from ..entities.prediction import Prediction


class UncertaintySource(Enum):
    """Types of uncertainty sources."""

    EPISTEMIC = "epistemic"  # Knowledge uncertainty
    ALEATORY = "aleatory"  # Inherent randomness
    MODEL = "model"  # Model uncertainty
    DATA = "data"  # Data quality uncertainty
    TEMPORAL = "temporal"  # Time-related uncertainty
    EXPERT = "expert"  # Expert disagreement


@dataclass
class UncertaintyAssessment:
    """Comprehensive uncertainty assessment."""

    total_uncertainty: float
    uncertainty_sources: Dict[UncertaintySource, float]
    confidence_interval: Tuple[float, float]
    confidence_level: float
    calibration_score: float
    uncertainty_decomposition: Dict[str, float]
    assessment_timestamp: datetime

    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of uncertainty assessment."""
        return {
            "total_uncertainty": self.total_uncertainty,
            "dominant_source": max(
                self.uncertainty_sources.items(), key=lambda x: x[1]
            )[0].value,
            "confidence_interval_width": self.confidence_interval[1]
            - self.confidence_interval[0],
            "confidence_level": self.confidence_level,
            "calibration_score": self.calibration_score,
            "assessment_time": self.assessment_timestamp.isoformat(),
        }


@dataclass
class ConfidenceThresholds:
    """Configurable confidence thresholds for decision making."""

    minimum_submission: float = 0.6
    high_confidence: float = 0.8
    very_high_confidence: float = 0.9
    abstention_threshold: float = 0.4
    research_trigger: float = 0.5

    def validate_thresholds(self) -> None:
        """Validate threshold consistency."""
        thresholds = [
            self.abstention_threshold,
            self.research_trigger,
            self.minimum_submission,
            self.high_confidence,
            self.very_high_confidence,
        ]

        if thresholds != sorted(thresholds):
            raise ValueError("Confidence thresholds must be in ascending order")


class UncertaintyQuantifier:
    """Service for quantifying uncertainty and managing confidence levels."""

    def __init__(self, confidence_thresholds: Optional[ConfidenceThresholds] = None):
        """Initialize uncertainty quantifier."""
        self.confidence_thresholds = confidence_thresholds or ConfidenceThresholds()
        self.confidence_thresholds.validate_thresholds()

        # Historical calibration data for adjustment
        self.calibration_history: List[Dict[str, Any]] = []

    def assess_prediction_uncertainty(
        self,
        prediction: Prediction,
        ensemble_predictions: Optional[List[Prediction]] = None,
        research_quality_score: Optional[float] = None,
    ) -> UncertaintyAssessment:
        """Assess uncertainty for a single prediction."""
        uncertainty_sources = {}

        # Model uncertainty from prediction method
        uncertainty_sources[UncertaintySource.MODEL] = (
            self._calculate_model_uncertainty(prediction)
        )

        # Data uncertainty from evidence quality
        uncertainty_sources[UncertaintySource.DATA] = self._calculate_data_uncertainty(
            prediction, research_quality_score
        )

        # Expert uncertainty from ensemble disagreement
        if ensemble_predictions:
            uncertainty_sources[UncertaintySource.EXPERT] = (
                self._calculate_expert_uncertainty(ensemble_predictions)
            )
        else:
            uncertainty_sources[UncertaintySource.EXPERT] = 0.0

        # Epistemic uncertainty from reasoning quality
        uncertainty_sources[UncertaintySource.EPISTEMIC] = (
            self._calculate_epistemic_uncertainty(prediction)
        )

        # Temporal uncertainty (default moderate for now)
        uncertainty_sources[UncertaintySource.TEMPORAL] = 0.3

        # Aleatory uncertainty (inherent randomness - moderate default)
        uncertainty_sources[UncertaintySource.ALEATORY] = 0.2

        # Calculate total uncertainty
        total_uncertainty = self._aggregate_uncertainties(uncertainty_sources)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            prediction, total_uncertainty
        )

        # Assess confidence level
        confidence_level = self._assess_confidence_level(prediction, total_uncertainty)

        # Calculate calibration score
        calibration_score = self._calculate_calibration_score(prediction)

        # Create uncertainty decomposition
        uncertainty_decomposition = {
            source.value: value for source, value in uncertainty_sources.items()
        }

        return UncertaintyAssessment(
            total_uncertainty=total_uncertainty,
            uncertainty_sources=uncertainty_sources,
            confidence_interval=confidence_interval,
            confidence_level=confidence_level,
            calibration_score=calibration_score,
            uncertainty_decomposition=uncertainty_decomposition,
            assessment_timestamp=datetime.utcnow(),
        )

    def assess_forecast_uncertainty(
        self,
        forecast: Forecast,
        research_quality_scores: Optional[Dict[str, float]] = None,
    ) -> UncertaintyAssessment:
        """Assess uncertainty for a complete forecast."""
        # Use ensemble predictions for comprehensive assessment
        return self.assess_prediction_uncertainty(
            forecast.final_prediction,
            forecast.predictions,
            research_quality_scores.get("average") if research_quality_scores else None,
        )

    def validate_confidence_level(
        self, prediction: Prediction, uncertainty_assessment: UncertaintyAssessment
    ) -> Dict[str, Any]:
        """Validate if confidence level is appropriate given uncertainty."""
        predicted_confidence = prediction.get_confidence_score()
        assessed_confidence = uncertainty_assessment.confidence_level

        confidence_gap = abs(predicted_confidence - assessed_confidence)

        validation_result = {
            "is_valid": confidence_gap < 0.2,  # Allow 20% tolerance
            "predicted_confidence": predicted_confidence,
            "assessed_confidence": assessed_confidence,
            "confidence_gap": confidence_gap,
            "recommendation": self._get_confidence_recommendation(
                predicted_confidence, assessed_confidence, uncertainty_assessment
            ),
        }

        return validation_result

    def should_trigger_additional_research(
        self, uncertainty_assessment: UncertaintyAssessment
    ) -> Dict[str, Any]:
        """Determine if additional research is needed based on uncertainty."""
        trigger_research = (
            uncertainty_assessment.confidence_level
            < self.confidence_thresholds.research_trigger
            or uncertainty_assessment.uncertainty_sources[UncertaintySource.DATA] > 0.6
            or uncertainty_assessment.uncertainty_sources[UncertaintySource.EPISTEMIC]
            > 0.7
        )

        research_priorities = []

        # Identify research priorities based on uncertainty sources
        if uncertainty_assessment.uncertainty_sources[UncertaintySource.DATA] > 0.5:
            research_priorities.append("data_quality")
        if (
            uncertainty_assessment.uncertainty_sources[UncertaintySource.EPISTEMIC]
            > 0.5
        ):
            research_priorities.append("domain_knowledge")
        if uncertainty_assessment.uncertainty_sources[UncertaintySource.EXPERT] > 0.5:
            research_priorities.append("expert_consensus")

        return {
            "trigger_research": trigger_research,
            "research_priorities": research_priorities,
            "confidence_level": uncertainty_assessment.confidence_level,
            "dominant_uncertainty": max(
                uncertainty_assessment.uncertainty_sources.items(), key=lambda x: x[1]
            )[0].value,
        }

    def should_abstain_from_prediction(
        self,
        uncertainty_assessment: UncertaintyAssessment,
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Determine if prediction should be abstained based on uncertainty."""

        # Consider tournament context
        tournament_penalty = 0.0
        if tournament_context:
            # Higher penalty for abstention in competitive tournaments
            tournament_penalty = tournament_context.get("abstention_penalty", 0.0)

        adjusted_threshold = (
            self.confidence_thresholds.abstention_threshold + tournament_penalty
        )

        should_abstain = uncertainty_assessment.confidence_level < adjusted_threshold

        return {
            "should_abstain": should_abstain,
            "confidence_level": uncertainty_assessment.confidence_level,
            "abstention_threshold": adjusted_threshold,
            "tournament_penalty": tournament_penalty,
            "reason": self._get_abstention_reason(
                uncertainty_assessment, should_abstain
            ),
        }

    def update_confidence_thresholds(
        self, performance_data: Dict[str, float], calibration_data: Dict[str, float]
    ) -> None:
        """Update confidence thresholds based on performance feedback."""
        # Adjust thresholds based on calibration performance
        calibration_error = calibration_data.get("calibration_error", 0.0)

        if calibration_error > 0.1:  # Poor calibration
            # Be more conservative
            self.confidence_thresholds.minimum_submission += 0.05
            self.confidence_thresholds.high_confidence += 0.05
        elif calibration_error < 0.05:  # Good calibration
            # Can be slightly more aggressive
            self.confidence_thresholds.minimum_submission = max(
                0.5, self.confidence_thresholds.minimum_submission - 0.02
            )

        # Ensure thresholds remain valid
        self.confidence_thresholds.validate_thresholds()

    def get_confidence_management_report(
        self,
        predictions: List[Prediction],
        uncertainty_assessments: List[UncertaintyAssessment],
    ) -> Dict[str, Any]:
        """Generate comprehensive confidence management report."""
        if not predictions or not uncertainty_assessments:
            return {"error": "No data provided for report generation"}

        confidence_scores = [p.get_confidence_score() for p in predictions]
        uncertainty_scores = [ua.total_uncertainty for ua in uncertainty_assessments]

        report = {
            "summary": {
                "total_predictions": len(predictions),
                "average_confidence": statistics.mean(confidence_scores),
                "average_uncertainty": statistics.mean(uncertainty_scores),
                "confidence_std": (
                    statistics.stdev(confidence_scores)
                    if len(confidence_scores) > 1
                    else 0.0
                ),
                "uncertainty_std": (
                    statistics.stdev(uncertainty_scores)
                    if len(uncertainty_scores) > 1
                    else 0.0
                ),
            },
            "confidence_distribution": self._analyze_confidence_distribution(
                confidence_scores
            ),
            "uncertainty_analysis": self._analyze_uncertainty_sources(
                uncertainty_assessments
            ),
            "calibration_metrics": self._calculate_calibration_metrics(
                predictions, uncertainty_assessments
            ),
            "threshold_performance": self._analyze_threshold_performance(
                predictions, uncertainty_assessments
            ),
            "recommendations": self._generate_confidence_recommendations(
                predictions, uncertainty_assessments
            ),
        }

        return report

    def _calculate_model_uncertainty(self, prediction: Prediction) -> float:
        """Calculate uncertainty from prediction method."""
        method_uncertainties = {
            "chain_of_thought": 0.3,
            "tree_of_thought": 0.2,  # More systematic, lower uncertainty
            "react": 0.4,  # Dynamic, higher uncertainty
            "auto_cot": 0.35,
            "self_consistency": 0.25,
            "ensemble": 0.15,  # Ensemble reduces uncertainty
        }

        return method_uncertainties.get(prediction.method.value, 0.4)

    def _calculate_data_uncertainty(
        self, prediction: Prediction, research_quality_score: Optional[float]
    ) -> float:
        """Calculate uncertainty from data/evidence quality."""
        if research_quality_score is None:
            # Use evidence strength if available
            if prediction.evidence_strength:
                return 1.0 - prediction.evidence_strength
            return 0.5  # Default moderate uncertainty

        return 1.0 - research_quality_score

    def _calculate_expert_uncertainty(
        self, ensemble_predictions: List[Prediction]
    ) -> float:
        """Calculate uncertainty from expert/ensemble disagreement."""
        if len(ensemble_predictions) < 2:
            return 0.0

        # Calculate variance in binary predictions
        binary_probs = [
            p.result.binary_probability
            for p in ensemble_predictions
            if p.result.binary_probability is not None
        ]

        if not binary_probs:
            return 0.0

        if len(binary_probs) == 1:
            return 0.0

        variance = statistics.variance(binary_probs)
        # Scale variance to uncertainty (0-1 range)
        return min(1.0, variance * 4)  # Scale factor for reasonable range

    def _calculate_epistemic_uncertainty(self, prediction: Prediction) -> float:
        """Calculate epistemic (knowledge) uncertainty."""
        base_uncertainty = 0.4

        # Reduce uncertainty based on reasoning quality
        if prediction.reasoning_trace:
            reasoning_quality = prediction.reasoning_trace.get_reasoning_quality_score()
            base_uncertainty *= 1.0 - reasoning_quality * 0.5

        # Reduce uncertainty based on multi-step reasoning
        if prediction.multi_step_reasoning:
            reasoning_depth_bonus = min(
                0.2, len(prediction.multi_step_reasoning) * 0.02
            )
            base_uncertainty *= 1.0 - reasoning_depth_bonus

        return max(0.1, base_uncertainty)  # Minimum epistemic uncertainty

    def _aggregate_uncertainties(
        self, uncertainty_sources: Dict[UncertaintySource, float]
    ) -> float:
        """Aggregate different uncertainty sources."""
        # Use root sum of squares for independent uncertainties
        sum_of_squares = sum(u**2 for u in uncertainty_sources.values())
        return min(1.0, math.sqrt(sum_of_squares / len(uncertainty_sources)))

    def _calculate_confidence_interval(
        self, prediction: Prediction, total_uncertainty: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""
        if prediction.result.binary_probability is None:
            return (0.0, 1.0)

        center = prediction.result.binary_probability

        # Calculate interval width based on uncertainty
        # Higher uncertainty = wider interval
        interval_width = total_uncertainty * 0.5  # Scale factor

        lower_bound = max(0.0, center - interval_width)
        upper_bound = min(1.0, center + interval_width)

        return (lower_bound, upper_bound)

    def _assess_confidence_level(
        self, prediction: Prediction, total_uncertainty: float
    ) -> float:
        """Assess appropriate confidence level given uncertainty."""
        # Start with base confidence from prediction
        base_confidence = prediction.get_confidence_score()

        # Adjust based on total uncertainty
        uncertainty_penalty = total_uncertainty * 0.5
        adjusted_confidence = base_confidence * (1.0 - uncertainty_penalty)

        return max(0.1, min(1.0, adjusted_confidence))

    def _calculate_calibration_score(self, prediction: Prediction) -> float:
        """Calculate calibration score for prediction."""
        # Use historical calibration data if available
        if prediction.calibration_data:
            return prediction.calibration_data.get("calibration_score", 0.5)

        # Default moderate calibration
        return 0.6

    def _get_confidence_recommendation(
        self,
        predicted_confidence: float,
        assessed_confidence: float,
        uncertainty_assessment: UncertaintyAssessment,
    ) -> str:
        """Get recommendation for confidence adjustment."""
        if predicted_confidence > assessed_confidence + 0.2:
            return "Consider reducing confidence due to high uncertainty"
        elif predicted_confidence < assessed_confidence - 0.2:
            return "Consider increasing confidence given uncertainty assessment"
        else:
            return "Confidence level appears appropriate"

    def _get_abstention_reason(
        self, uncertainty_assessment: UncertaintyAssessment, should_abstain: bool
    ) -> str:
        """Get reason for abstention recommendation."""
        if not should_abstain:
            return "Confidence sufficient for prediction"

        dominant_source = max(
            uncertainty_assessment.uncertainty_sources.items(), key=lambda x: x[1]
        )[0]

        reasons = {
            UncertaintySource.DATA: "Insufficient or low-quality data",
            UncertaintySource.EPISTEMIC: "Insufficient domain knowledge",
            UncertaintySource.EXPERT: "High expert disagreement",
            UncertaintySource.MODEL: "High model uncertainty",
            UncertaintySource.TEMPORAL: "High temporal uncertainty",
            UncertaintySource.ALEATORY: "High inherent randomness",
        }

        return f"High uncertainty due to: {reasons.get(dominant_source, 'multiple factors')}"

    def _analyze_confidence_distribution(
        self, confidence_scores: List[float]
    ) -> Dict[str, Any]:
        """Analyze distribution of confidence scores."""
        if not confidence_scores:
            return {}

        return {
            "mean": statistics.mean(confidence_scores),
            "median": statistics.median(confidence_scores),
            "std": (
                statistics.stdev(confidence_scores)
                if len(confidence_scores) > 1
                else 0.0
            ),
            "min": min(confidence_scores),
            "max": max(confidence_scores),
            "quartiles": {
                "q1": (
                    statistics.quantiles(confidence_scores, n=4)[0]
                    if len(confidence_scores) >= 4
                    else min(confidence_scores)
                ),
                "q3": (
                    statistics.quantiles(confidence_scores, n=4)[2]
                    if len(confidence_scores) >= 4
                    else max(confidence_scores)
                ),
            },
        }

    def _analyze_uncertainty_sources(
        self, uncertainty_assessments: List[UncertaintyAssessment]
    ) -> Dict[str, Any]:
        """Analyze uncertainty sources across assessments."""
        if not uncertainty_assessments:
            return {}

        source_averages = {}
        for source in UncertaintySource:
            values = [
                ua.uncertainty_sources.get(source, 0.0)
                for ua in uncertainty_assessments
            ]
            source_averages[source.value] = {
                "mean": statistics.mean(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

        return source_averages

    def _calculate_calibration_metrics(
        self,
        predictions: List[Prediction],
        uncertainty_assessments: List[UncertaintyAssessment],
    ) -> Dict[str, float]:
        """Calculate calibration metrics."""
        if not predictions or not uncertainty_assessments:
            return {}

        confidence_scores = [p.get_confidence_score() for p in predictions]
        assessed_confidences = [ua.confidence_level for ua in uncertainty_assessments]

        # Calculate mean absolute error between predicted and assessed confidence
        mae = statistics.mean(
            [
                abs(pred - assess)
                for pred, assess in zip(confidence_scores, assessed_confidences)
            ]
        )

        return {
            "confidence_mae": mae,
            "average_predicted_confidence": statistics.mean(confidence_scores),
            "average_assessed_confidence": statistics.mean(assessed_confidences),
        }

    def _analyze_threshold_performance(
        self,
        predictions: List[Prediction],
        uncertainty_assessments: List[UncertaintyAssessment],
    ) -> Dict[str, Any]:
        """Analyze performance of confidence thresholds."""
        confidence_scores = [p.get_confidence_score() for p in predictions]

        threshold_analysis = {}

        for threshold_name in [
            "minimum_submission",
            "high_confidence",
            "very_high_confidence",
        ]:
            threshold_value = getattr(self.confidence_thresholds, threshold_name)
            above_threshold = sum(1 for c in confidence_scores if c >= threshold_value)

            threshold_analysis[threshold_name] = {
                "threshold": threshold_value,
                "predictions_above": above_threshold,
                "percentage_above": above_threshold / len(confidence_scores) * 100,
            }

        return threshold_analysis

    def _generate_confidence_recommendations(
        self,
        predictions: List[Prediction],
        uncertainty_assessments: List[UncertaintyAssessment],
    ) -> List[str]:
        """Generate recommendations for confidence management."""
        recommendations = []

        confidence_scores = [p.get_confidence_score() for p in predictions]
        uncertainty_scores = [ua.total_uncertainty for ua in uncertainty_assessments]

        avg_confidence = statistics.mean(confidence_scores)
        avg_uncertainty = statistics.mean(uncertainty_scores)

        if avg_confidence > 0.8 and avg_uncertainty > 0.6:
            recommendations.append(
                "Consider being more conservative with confidence given high uncertainty"
            )

        if avg_confidence < 0.4:
            recommendations.append(
                "Consider additional research or abstention for low-confidence predictions"
            )

        # Analyze uncertainty sources
        source_averages = self._analyze_uncertainty_sources(uncertainty_assessments)
        if source_averages:
            dominant_source = max(source_averages.items(), key=lambda x: x[1]["mean"])
            if dominant_source[1]["mean"] > 0.6:
                recommendations.append(
                    f"Focus on reducing {dominant_source[0]} uncertainty"
                )

        return recommendations
