"""Forecast domain entity and Brier score utilities (clean)."""

from dataclasses import dataclass
from datetime import datetime, timezone
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
    DRAFT = "draft"
    SUBMITTED = "submitted"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


@dataclass
class Forecast:
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

    ensemble_method: str
    weight_distribution: Dict[str, float]
    consensus_strength: float

    metadata: Dict[str, Any]

    metaculus_response: Optional[Dict[str, Any]] = None

    tournament_strategy: Optional[TournamentStrategy] = None
    question_priority: Optional[QuestionPriority] = None
    competitive_analysis: Optional[Dict[str, Any]] = None
    competitive_intelligence: Optional[CompetitiveIntelligence] = None
    risk_assessment: Optional[Dict[str, float]] = None
    calibration_metrics: Optional[Dict[str, float]] = None
    submission_timing_data: Optional[Dict[str, Any]] = None

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
        **kwargs: Any,
    ) -> "Forecast":
        now = datetime.now(timezone.utc)
        confidence_scores = [p.get_confidence_score() for p in predictions]
        avg_conf = (
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
            confidence_score=avg_conf,
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
        final_probability: "Probability | float",
        aggregation_method: str = "single",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "Forecast":
        now = datetime.now(timezone.utc)
        dummy_research_report_id = uuid4()
        # normalize to PredictionResult
        if isinstance(final_probability, (int, float)):
            pr = PredictionResult(binary_probability=float(final_probability))
        elif isinstance(final_probability, PredictionResult):
            pr = final_probability
        elif hasattr(final_probability, "binary_probability"):
            pr = PredictionResult(
                binary_probability=float(
                    getattr(final_probability, "binary_probability")
                )
            )
        elif hasattr(final_probability, "value"):
            pr = PredictionResult(
                binary_probability=float(getattr(final_probability, "value"))
            )
        else:
            raise TypeError(
                "final_probability must be float-like, Probability, or PredictionResult-like"
            )

        final_prediction = Prediction(
            id=uuid4(),
            question_id=question_id,
            research_report_id=dummy_research_report_id,
            result=pr,
            confidence=PredictionConfidence.MEDIUM,
            method=PredictionMethod.ENSEMBLE,
            reasoning="Aggregated prediction from pipeline",
            reasoning_steps=["Pipeline aggregation of multiple agent predictions"],
            created_at=now,
            created_by=(
                metadata.get("agent_used", "pipeline") if metadata else "pipeline"
            ),
        )

        if predictions:
            cs = [p.get_confidence_score() for p in predictions]
            avg_conf = sum(cs) / len(cs)
        else:
            avg_conf = 0.5

        return cls(
            id=uuid4(),
            question_id=question_id,
            research_reports=kwargs.get("research_reports", []),
            predictions=predictions,
            final_prediction=final_prediction,
            status=ForecastStatus.DRAFT,
            confidence_score=avg_conf,
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
        self.status = ForecastStatus.SUBMITTED
        self.submission_timestamp = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def resolve(self, metaculus_response: Dict[str, Any]) -> None:
        self.status = ForecastStatus.RESOLVED
        self.metaculus_response = metaculus_response
        self.updated_at = datetime.now(timezone.utc)

    def calculate_prediction_variance(self) -> float:
        if not self.predictions:
            return 0.0
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
            return float(variance)
        return 0.0

    def get_prediction_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total_predictions": len(self.predictions),
            "methods_used": list(set(p.method.value for p in self.predictions)),
            "confidence_levels": [p.confidence.value for p in self.predictions],
            "prediction_variance": self.calculate_prediction_variance(),
            "consensus_strength": self.consensus_strength,
        }
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

    @property
    def prediction(self) -> float:
        if (
            self.final_prediction
            and self.final_prediction.result.binary_probability is not None
        ):
            return self.final_prediction.result.binary_probability
        return 0.5

    @property
    def confidence(self) -> float:
        if hasattr(self.final_prediction, "confidence") and hasattr(
            self.final_prediction.confidence, "value"
        ):
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
        if hasattr(self, "_method_override") and getattr(self, "_method_override"):
            return getattr(self, "_method_override")
        if self.final_prediction and hasattr(self.final_prediction, "method"):
            return self.final_prediction.method.value
        return "unknown"

    @method.setter
    def method(self, value: str) -> None:
        setattr(self, "_method_override", value)

    def apply_tournament_strategy(self, strategy: TournamentStrategy) -> None:
        self.tournament_strategy = strategy
        self.updated_at = datetime.now(timezone.utc)

    def add_competitive_intelligence(
        self, intelligence: CompetitiveIntelligence
    ) -> None:
        self.competitive_intelligence = intelligence
        self.updated_at = datetime.now(timezone.utc)

    def add_reasoning_trace(self, trace: ReasoningTrace) -> None:
        if self.reasoning_traces is None:
            self.reasoning_traces = []
        self.reasoning_traces.append(trace)
        self.updated_at = datetime.now(timezone.utc)

    def apply_bias_mitigation(self, mitigation_type: str) -> None:
        if self.bias_mitigation_applied is None:
            self.bias_mitigation_applied = []
        self.bias_mitigation_applied.append(mitigation_type)
        self.updated_at = datetime.now(timezone.utc)

    def add_uncertainty_source(self, source: str) -> None:
        if self.uncertainty_sources is None:
            self.uncertainty_sources = []
        self.uncertainty_sources.append(source)
        self.updated_at = datetime.now(timezone.utc)

    def update_calibration_history(self, calibration_data: Dict[str, Any]) -> None:
        if self.confidence_calibration_history is None:
            self.confidence_calibration_history = []
        calibration_data["timestamp"] = datetime.now(timezone.utc)
        self.confidence_calibration_history.append(calibration_data)
        self.updated_at = datetime.now(timezone.utc)

    def set_question_priority(self, priority: QuestionPriority) -> None:
        self.question_priority = priority
        self.updated_at = datetime.now(timezone.utc)

    def add_competitive_analysis(self, analysis: Dict[str, Any]) -> None:
        self.competitive_analysis = analysis
        self.updated_at = datetime.now(timezone.utc)

    def calculate_risk_assessment(self) -> Dict[str, float]:
        risk_factors = {
            "prediction_variance": self.calculate_prediction_variance(),
            "research_quality": self._calculate_research_quality_risk(),
            "time_pressure": self._calculate_time_pressure_risk(),
            "confidence_calibration": self._calculate_calibration_risk(),
            "ensemble_disagreement": self._calculate_ensemble_disagreement_risk(),
        }
        risk_factors["overall_risk"] = sum(risk_factors.values()) / len(risk_factors)
        self.risk_assessment = risk_factors
        return risk_factors

    def _calculate_research_quality_risk(self) -> float:
        if not self.research_reports:
            return 0.8
        avg_quality = sum(
            (
                0.8
                if r.quality.value == "high"
                else 0.5
                if r.quality.value == "medium"
                else 0.2
            )
            for r in self.research_reports
        ) / len(self.research_reports)
        return float(1.0 - avg_quality)

    def _calculate_time_pressure_risk(self) -> float:
        if not self.submission_timing_data:
            return 0.5
        hours = self.submission_timing_data.get("hours_to_deadline", 24)
        if hours < 2:
            return 0.9
        if hours < 12:
            return 0.7
        if hours < 48:
            return 0.4
        return 0.2

    def _calculate_calibration_risk(self) -> float:
        if not self.calibration_metrics:
            return 0.5
        calibration_error = self.calibration_metrics.get("calibration_error", 0.1)
        return float(min(1.0, calibration_error * 5))

    def _calculate_ensemble_disagreement_risk(self) -> float:
        variance = self.calculate_prediction_variance()
        if variance > 0.1:
            return 0.8
        if variance > 0.05:
            return 0.6
        return 0.3

    def should_submit_prediction(
        self, strategy: Optional[TournamentStrategy] = None
    ) -> bool:
        current_strategy = strategy or self.tournament_strategy
        if not current_strategy:
            return (
                self.confidence_score > 0.6
                and self.calculate_prediction_variance() < 0.1
            )
        min_confidence = current_strategy.confidence_thresholds.get(
            "minimum_submission", 0.6
        )
        if self.confidence_score < min_confidence:
            return False
        risk_assessment = self.risk_assessment or self.calculate_risk_assessment()
        max_risk = 0.8 if current_strategy.risk_profile.value == "aggressive" else 0.6
        return risk_assessment.get("overall_risk", 1.0) <= max_risk

    def optimize_submission_timing(
        self, tournament_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        deadline = tournament_context.get("deadline")
        if not isinstance(deadline, datetime):
            return {
                "recommended_action": "submit_now",
                "reason": "No or invalid deadline",
            }
        if deadline.tzinfo is None or deadline.tzinfo.utcoffset(deadline) is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        hours = (deadline - now_utc).total_seconds() / 3600
        strategy = self.tournament_strategy
        if not strategy:
            return {"recommended_action": "submit_now", "reason": "No strategy defined"}
        s = strategy.submission_timing_strategy
        if s == "early_advantage":
            return (
                {
                    "recommended_action": "submit_now",
                    "reason": "Early submission for competitive advantage",
                }
                if hours > 24
                else {"recommended_action": "submit_now", "reason": "Close to deadline"}
            )
        if s == "late_validation":
            return (
                {
                    "recommended_action": "wait",
                    "reason": "Allow time for additional validation",
                }
                if hours > 12
                else {
                    "recommended_action": "submit_now",
                    "reason": "Approaching deadline",
                }
            )
        if s == "optimal_window":
            if hours > 48:
                return {
                    "recommended_action": "wait",
                    "reason": "Too early, wait for optimal window",
                }
            if hours > 6:
                return {
                    "recommended_action": "submit_now",
                    "reason": "In optimal submission window",
                }
            return {
                "recommended_action": "submit_now",
                "reason": "Deadline approaching",
            }
        return {"recommended_action": "submit_now", "reason": "Default action"}


def calculate_brier_score(forecast: float, outcome: int) -> float:
    """Calculate the Brier score for a binary forecast.

    Args:
        forecast: Predicted probability in [0.0, 1.0].
        outcome: Actual outcome (0 or 1).
    """
    if not (0.0 <= forecast <= 1.0):
        raise ValueError("Forecast probability must be between 0.0 and 1.0")
    if outcome not in (0, 1):
        raise ValueError("Outcome must be 0 or 1")
    return float((forecast - outcome) ** 2)
