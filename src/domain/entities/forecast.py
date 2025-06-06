"""Forecast domain entity."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from uuid import UUID, uuid4

from .prediction import Prediction, PredictionResult, PredictionConfidence, PredictionMethod
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
        **kwargs
    ) -> "Forecast":
        """Factory method to create a forecast compatible with pipeline interface."""
        from src.domain.value_objects.probability import Probability
        
        now = datetime.utcnow()
        
        # Create a final prediction from the probability
        # We need a dummy research_report_id for now - this should come from the pipeline
        dummy_research_report_id = uuid4()  # TODO: Get actual research report ID from pipeline
        
        # Convert final_probability to PredictionResult
        if isinstance(final_probability, (int, float)):
            prediction_result = PredictionResult(binary_probability=float(final_probability))
        elif hasattr(final_probability, 'value'):
            # It's a Probability object with .value attribute
            prediction_result = PredictionResult(binary_probability=float(final_probability.value))
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
            created_by=metadata.get("agent_used", "pipeline") if metadata else "pipeline"
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
            reasoning_summary=kwargs.get("reasoning_summary", f"Forecast using {aggregation_method} aggregation"),
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
    
    # Backward compatibility properties for tests
    @property
    def prediction(self) -> float:
        """Get the final prediction probability for backward compatibility."""
        if self.final_prediction and self.final_prediction.result.binary_probability is not None:
            return self.final_prediction.result.binary_probability
        return 0.5  # Default fallback
    
    @property
    def confidence(self) -> float:
        """Get the confidence score for backward compatibility."""
        if hasattr(self.final_prediction, 'confidence'):
            # Convert confidence enum to numeric value
            if hasattr(self.final_prediction.confidence, 'value'):
                confidence_map = {
                    'very_low': 0.2,
                    'low': 0.4, 
                    'medium': 0.6,
                    'high': 0.75,
                    'very_high': 0.95
                }
                return confidence_map.get(self.final_prediction.confidence.value, 0.6)
        return self.confidence_score
    
    @property
    def method(self) -> str:
        """Get the prediction method for backward compatibility."""
        if self.final_prediction and hasattr(self.final_prediction, 'method'):
            return self.final_prediction.method.value
        return "unknown"

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
