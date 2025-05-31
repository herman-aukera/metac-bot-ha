"""Forecast domain entity."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from .prediction import Prediction
from .research_report import ResearchReport


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
    
    # Performance tracking
    metaculus_response: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create_new(
        cls,
        question_id: UUID,
        research_reports: List[ResearchReport],
        predictions: List[Prediction],
        final_prediction: Prediction,
        **kwargs
    ) -> "Forecast":
        """Factory method to create a new forecast."""
        now = datetime.utcnow()
        
        # Calculate confidence score from predictions
        confidence_scores = [p.get_confidence_score() for p in predictions]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
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
            p.result.binary_probability for p in self.predictions
            if p.result.binary_probability is not None
        ]
        
        if binary_probs:
            mean_prob = sum(binary_probs) / len(binary_probs)
            variance = sum((p - mean_prob) ** 2 for p in binary_probs) / len(binary_probs)
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
        binary_probs = [p.result.binary_probability for p in self.predictions if p.result.binary_probability is not None]
        if binary_probs:
            summary["binary_predictions"] = {
                "values": binary_probs,
                "mean": sum(binary_probs) / len(binary_probs),
                "min": min(binary_probs),
                "max": max(binary_probs),
            }
        
        return summary

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
