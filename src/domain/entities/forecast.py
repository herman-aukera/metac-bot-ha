"""Forecast domain entity."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..value_objects.reasoning_trace import ReasoningTrace
from ..value_objects.tournament_strategy import (
    CompetitiveIntelligence,
    QuestionPriority,
    TournamentStrategy,
)
from .prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
    PredictionResult,
)
from .research_report import ResearchReport

if TYPE_CHECKING:
    from src.domain.value_objects.probability import Probability


class ForecastStatus(Enum):
    """Status of a forecast."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


@dataclass
class Forecast:
    """
    Domain entity representing a complete forecast for a question.

    Aggregates multiple predictions and research reports to form
    the final forecast that will be submitted to Metaculus.
    """

    id: UUID
    question_id: UUID
    research_reports: List[ResearchReport]
    predictions: List[Prediction]
    final_prediction: Prediction
    status: ForecastStatus
    confidence_score: float
    reasoning_summary: str
    submission_timestamp: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    # Ensemble and aggregation metadata
    ensemble_method: str
    weight_distribution: Dict[str, float]
    consensus_strength: float

    # Additional metadata for pipeline interface
    metadata: Dict[str, Any]

    # Performance tracking
    metaculus_response: Optional[Dict[str, Any]] = None

    # Tournament-specific enhancements
    tournament_strategy: Optional[TournamentStrategy] = None
    question_priority: Optional[QuestionPriority] = None
    competitive_analysis: Optional[Dict[str, Any]] = None
    competitive_intelligence: Optional[CompetitiveIntelligence] = None
    risk_assessment: Optional[Dict[str, float]] = None
    calibration_metrics: Optional[Dict[str, float]] = None
    submission_timing_data: Optional[Dict[str, Any]] = None

    # Advanced reasoning capabilities
    reasoning_traces: Optional[List[ReasoningTrace]] = None
    bias_mitigation_applied: Optional[List[str]] = None
    uncertainty_sources: Optional[List[str]] = None
    confidence_calibration_history: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def create_new(
        cls,
        question_id: UUID,
        research_reports: List[ResearchReport],
        predictions: List[Prediction],
        final_prediction: Prediction,
        **kwargs,
    ) -> "Forecast":
        """Factory method to create a new forecast."""
        now = datetime.utcnow()

        # Calculate confidence score from predictions
        confidence_scores = [p.get_confidence_score() for p in predictions]
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.5
        )

        return cls(
            id=uuid4(),
            question_id=question_id,
            research_reports=research_reports,
            predictions=predictions,
            final_prediction=final_prediction,
            status=ForecastStatus.DRAFT,
            confidence_score=avg_confidence,
            reasoning_summary=kwargs.get("reasoning_summary", ""),
            submission_timestamp=None,
            created_at=now,
            updated_at=now,
            ensemble_method=kwargs.get("ensemble_method", "weighted_average"),
            weight_distribution=kwargs.get("weight_distribution", {}),
            consensus_strength=kwargs.get("consensus_strength", 0.0),
            metadata=kwargs.get("metadata", {}),
            metaculus_response=kwargs.get("metaculus_response"),
        )

    @classmethod
    def create(
        cls,
        question_id: UUID,
        predictions: List[Prediction],
        final_probability: "Probability",
        aggregation_method: str = "single",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "Forecast":
        """Factory method to create a forecast compatible with pipeline interface."""
        from src.domain.value_objects.probability import Probability

        now = datetime.utcnow()

        # Create a final prediction from the probability
        # We need a dummy research_report_id for now - this should come from the pipeline
        dummy_research_report_id = (
            uuid4()
        )  # TODO: Get actual research report ID from pipeline

        # Convert final_probability to PredictionResult
        if isinstance(final_probability, (int, float)):
            prediction_result = PredictionResult(
                binary_probability=float(final_probability)
            )
        elif hasattr(final_probability, "value"):
            # It's a Probability object with .value attribute
            prediction_result = PredictionResult(
                binary_probability=float(final_probability.value)
            )
        else:
            # Assume it's already a PredictionResult or similar
            prediction_result = final_probability

        final_prediction = Prediction(
            id=uuid4(),
            question_id=question_id,
            research_report_id=dummy_research_report_id,
            result=prediction_result,
            confidence=PredictionConfidence.MEDIUM,
            method=PredictionMethod.ENSEMBLE,  # Since this is aggregated from pipeline
            reasoning="Aggregated prediction from pipeline",
            reasoning_steps=["Pipeline aggregation of multiple agent predictions"],
            created_at=now,
            created_by=(
                metadata.get("agent_used", "pipeline") if metadata else "pipeline"
            ),
        )

        # Calculate confidence score from predictions
        if predictions:
            confidence_scores = [p.get_confidence_score() for p in predictions]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            avg_confidence = 0.5

        return cls(
            id=uuid4(),
            question_id=question_id,
            research_reports=kwargs.get("research_reports", []),
            predictions=predictions,
            final_prediction=final_prediction,
            status=ForecastStatus.DRAFT,
            confidence_score=avg_confidence,
            reasoning_summary=kwargs.get(
                "reasoning_summary", f"Forecast using {aggregation_method} aggregation"
            ),
            submission_timestamp=None,
            created_at=now,
            updated_at=now,
            ensemble_method=aggregation_method,
            weight_distribution=kwargs.get("weight_distribution", {}),
            consensus_strength=kwargs.get("consensus_strength", 0.0),
            metadata=metadata or {},
            metaculus_response=kwargs.get("metaculus_response"),
        )

    def submit(self) -> None:
        """Mark the forecast as submitted."""
        self.status = ForecastStatus.SUBMITTED
        self.submission_timestamp = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def resolve(self, metaculus_response: Dict[str, Any]) -> None:
        """Mark the forecast as resolved with Metaculus response."""
        self.status = ForecastStatus.RESOLVED
        self.metaculus_response = metaculus_response
        self.updated_at = datetime.utcnow()

    def calculate_prediction_variance(self) -> float:
        """Calculate the variance in predictions to assess consensus."""
        if not self.predictions:
            return 0.0

        # For binary predictions
        binary_probs = [
            p.result.binary_probability
            for p in self.predictions
            if p.result.binary_probability is not None
        ]

        if binary_probs:
            mean_prob = sum(binary_probs) / len(binary_probs)
            variance = sum((p - mean_prob) ** 2 for p in binary_probs) / len(
                binary_probs
            )
            return variance

        return 0.0

    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get a summary of all predictions for analysis."""
        summary = {
            "total_predictions": len(self.predictions),
            "methods_used": list(set(p.method.value for p in self.predictions)),
            "confidence_levels": [p.confidence.value for p in self.predictions],
            "prediction_variance": self.calculate_prediction_variance(),
            "consensus_strength": self.consensus_strength,
        }

        # Add prediction values by type
        binary_probs = [
            p.result.binary_probability
            for p in self.predictions
            if p.result.binary_probability is not None
        ]
        if binary_probs:
            summary["binary_predictions"] = {
                "values": binary_probs,
                "mean": sum(binary_probs) / len(binary_probs),
                "min": min(binary_probs),
                "max": max(binary_probs),
            }

        return summary

    # Backward compatibility properties for tests
    @property
    def prediction(self) -> float:
        """Get the final prediction probability for backward compatibility."""
        if (
            self.final_prediction
            and self.final_prediction.result.binary_probability is not None
        ):
            return self.final_prediction.result.binary_probability
        return 0.5  # Default fallback

    @property
    def confidence(self) -> float:
        """Get the confidence score for backward compatibility."""
        if hasattr(self.final_prediction, "confidence"):
            # Convert confidence enum to numeric value
            if hasattr(self.final_prediction.confidence, "value"):
                confidence_map = {
                    "very_low": 0.2,
                    "low": 0.4,
                    "medium": 0.6,
                    "high": 0.75,
                    "very_high": 0.95,
                }
                return confidence_map.get(self.final_prediction.confidence.value, 0.6)
        return self.confidence_score

    @property
    def method(self) -> str:
        """Get the prediction method for backward compatibility."""
        if hasattr(self, "_method_override") and self._method_override:
            return self._method_override
        if self.final_prediction and hasattr(self.final_prediction, "method"):
            return self.final_prediction.method.value
        return "unknown"

    @method.setter
    def method(self, value: str) -> None:
        """Set the prediction method for backward compatibility."""
        self._method_override = value

    def apply_tournament_strategy(self, strategy: TournamentStrategy) -> None:
        """Apply tournament strategy to the forecast."""
        self.tournament_strategy = strategy
        self.updated_at = datetime.utcnow()

    def add_competitive_intelligence(
        self, intelligence: CompetitiveIntelligence
    ) -> None:
        """Add competitive intelligence data."""
        self.competitive_intelligence = intelligence
        self.updated_at = datetime.utcnow()

    def add_reasoning_trace(self, trace: ReasoningTrace) -> None:
        """Add reasoning trace for transparency."""
        if self.reasoning_traces is None:
            self.reasoning_traces = []
        self.reasoning_traces.append(trace)
        self.updated_at = datetime.utcnow()

    def apply_bias_mitigation(self, mitigation_type: str) -> None:
        """Record bias mitigation applied."""
        if self.bias_mitigation_applied is None:
            self.bias_mitigation_applied = []
        self.bias_mitigation_applied.append(mitigation_type)
        self.updated_at = datetime.utcnow()

    def add_uncertainty_source(self, source: str) -> None:
        """Add identified uncertainty source."""
        if self.uncertainty_sources is None:
            self.uncertainty_sources = []
        self.uncertainty_sources.append(source)
        self.updated_at = datetime.utcnow()

    def update_calibration_history(self, calibration_data: Dict[str, Any]) -> None:
        """Update confidence calibration history."""
        if self.confidence_calibration_history is None:
            self.confidence_calibration_history = []
        calibration_data["timestamp"] = datetime.utcnow()
        self.confidence_calibration_history.append(calibration_data)
        self.updated_at = datetime.utcnow()

    def set_question_priority(self, priority: QuestionPriority) -> None:
        """Set question priority for resource allocation."""
        self.question_priority = priority
        self.updated_at = datetime.utcnow()

    def add_competitive_analysis(self, analysis: Dict[str, Any]) -> None:
        """Add competitive analysis data."""
        self.competitive_analysis = analysis
        self.updated_at = datetime.utcnow()

    def calculate_risk_assessment(self) -> Dict[str, float]:
        """Calculate risk assessment for the forecast."""
        risk_factors = {
            "prediction_variance": self.calculate_prediction_variance(),
            "research_quality": self._calculate_research_quality_risk(),
            "time_pressure": self._calculate_time_pressure_risk(),
            "confidence_calibration": self._calculate_calibration_risk(),
            "ensemble_disagreement": self._calculate_ensemble_disagreement_risk(),
        }

        # Overall risk score (higher is riskier)
        risk_factors["overall_risk"] = sum(risk_factors.values()) / len(risk_factors)

        self.risk_assessment = risk_factors
        return risk_factors

    def _calculate_research_quality_risk(self) -> float:
        """Calculate risk based on research quality."""
        if not self.research_reports:
            return 0.8  # High risk with no research

        avg_quality = sum(
            (
                0.8
                if report.quality.value == "high"
                else 0.5 if report.quality.value == "medium" else 0.2
            )
            for report in self.research_reports
        ) / len(self.research_reports)

        return 1.0 - avg_quality  # Convert quality to risk

    def _calculate_time_pressure_risk(self) -> float:
        """Calculate risk based on time pressure."""
        if not self.submission_timing_data:
            return 0.5  # Default moderate risk

        time_to_deadline = self.submission_timing_data.get("hours_to_deadline", 24)
        if time_to_deadline < 2:
            return 0.9  # Very high risk
        elif time_to_deadline < 12:
            return 0.7  # High risk
        elif time_to_deadline < 48:
            return 0.4  # Moderate risk
        else:
            return 0.2  # Low risk

    def _calculate_calibration_risk(self) -> float:
        """Calculate risk based on confidence calibration."""
        if not self.calibration_metrics:
            return 0.5  # Default moderate risk

        calibration_error = self.calibration_metrics.get("calibration_error", 0.1)
        return min(1.0, calibration_error * 5)  # Scale calibration error to risk

    def _calculate_ensemble_disagreement_risk(self) -> float:
        """Calculate risk based on ensemble disagreement."""
        variance = self.calculate_prediction_variance()
        if variance > 0.1:
            return 0.8  # High risk with high disagreement
        elif variance > 0.05:
            return 0.6  # Moderate risk
        else:
            return 0.3  # Low risk with good agreement

    def should_submit_prediction(
        self, strategy: Optional[TournamentStrategy] = None
    ) -> bool:
        """Determine if prediction should be submitted based on strategy and risk."""
        current_strategy = strategy or self.tournament_strategy

        if not current_strategy:
            # Default conservative approach
            return (
                self.confidence_score > 0.6
                and self.calculate_prediction_variance() < 0.1
            )

        # Check confidence threshold
        min_confidence = current_strategy.confidence_thresholds.get(
            "minimum_submission", 0.6
        )
        if self.confidence_score < min_confidence:
            return False

        # Check risk assessment
        risk_assessment = self.risk_assessment or self.calculate_risk_assessment()
        max_risk = 0.8 if current_strategy.risk_profile.value == "aggressive" else 0.6

        if risk_assessment["overall_risk"] > max_risk:
            return False

        return True

    def optimize_submission_timing(
        self, tournament_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize submission timing based on tournament strategy."""
        current_time = datetime.utcnow()
        deadline = tournament_context.get("deadline")

        if not deadline:
            return {
                "recommended_action": "submit_now",
                "reason": "No deadline information",
            }

        hours_to_deadline = (deadline - current_time).total_seconds() / 3600

        strategy = self.tournament_strategy
        if not strategy:
            return {"recommended_action": "submit_now", "reason": "No strategy defined"}

        timing_strategy = strategy.submission_timing_strategy

        if timing_strategy == "early_advantage":
            if hours_to_deadline > 24:
                return {
                    "recommended_action": "submit_now",
                    "reason": "Early submission for competitive advantage",
                }
            else:
                return {
                    "recommended_action": "submit_now",
                    "reason": "Close to deadline",
                }

        elif timing_strategy == "late_validation":
            if hours_to_deadline > 12:
                return {
                    "recommended_action": "wait",
                    "reason": "Allow time for additional validation",
                }
            else:
                return {
                    "recommended_action": "submit_now",
                    "reason": "Approaching deadline",
                }

        elif timing_strategy == "optimal_window":
            if hours_to_deadline > 48:
                return {
                    "recommended_action": "wait",
                    "reason": "Too early, wait for optimal window",
                }
            elif hours_to_deadline > 6:
                return {
                    "recommended_action": "submit_now",
                    "reason": "In optimal submission window",
                }
            else:
                return {
                    "recommended_action": "submit_now",
                    "reason": "Deadline approaching",
                }

        return {"recommended_action": "submit_now", "reason": "Default action"}

    def get_tournament_performance_metrics(self) -> Dict[str, Any]:
        """Get tournament-specific performance metrics."""
        metrics = {
            "confidence_score": self.confidence_score,
            "prediction_variance": self.calculate_prediction_variance(),
            "consensus_strength": self.consensus_strength,
            "research_quality_score": self._get_research_quality_score(),
            "reasoning_quality_score": self._get_reasoning_quality_score(),
            "risk_score": (
                self.risk_assessment.get("overall_risk", 0.5)
                if self.risk_assessment
                else 0.5
            ),
        }

        if self.question_priority:
            metrics["priority_score"] = (
                self.question_priority.get_overall_priority_score()
            )
            metrics["scoring_potential"] = self.question_priority.scoring_potential

        return metrics

    def _get_research_quality_score(self) -> float:
        """Get average research quality score."""
        if not self.research_reports:
            return 0.0

        quality_scores = []
        for report in self.research_reports:
            if report.quality.value == "high":
                quality_scores.append(0.8)
            elif report.quality.value == "medium":
                quality_scores.append(0.5)
            else:
                quality_scores.append(0.2)

        return sum(quality_scores) / len(quality_scores)

    def _get_reasoning_quality_score(self) -> float:
        """Get average reasoning quality score from predictions."""
        if not self.predictions:
            return 0.0

        quality_scores = []
        for prediction in self.predictions:
            if hasattr(prediction, "calculate_prediction_quality_score"):
                quality_scores.append(prediction.calculate_prediction_quality_score())
            else:
                quality_scores.append(0.5)  # Default score

        return sum(quality_scores) / len(quality_scores)


def calculate_brier_score(forecast: float, outcome: int) -> float:
    """
    Calculates the Brier score for a binary forecast.

    Args:
        forecast: The predicted probability of the event occurring (must be between 0.0 and 1.0).
        outcome: The actual outcome (0 if the event did not occur, 1 if it did).

    Returns:
        The Brier score.

    Raises:
        ValueError: If the forecast probability is outside the [0.0, 1.0] range
                    or if the outcome is not 0 or 1.
    """
    if not (0.0 <= forecast <= 1.0):
        raise ValueError("Forecast probability must be between 0.0 and 1.0")
    if outcome not in (0, 1):
        raise ValueError("Outcome must be 0 or 1")

    return (forecast - outcome) ** 2


# TODO: Consider extending to multiclass Brier score or other scoring rules like Log Score.
