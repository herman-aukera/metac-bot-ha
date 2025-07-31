"""Forecast entity for representing predictions and their metadata."""

from dataclasses import dataclass
from datetime import datetime
from typing import Union, List, Dict, Any, Optional
from ..value_objects.reasoning_step import ReasoningStep
from ..value_objects.confidence import Confidence


@dataclass
class Forecast:
    """Represents a forecast/prediction for a tournament question.

    Attributes:
        question_id: ID of the question being forecasted
        prediction: The actual prediction value(s)
        confidence: Overall confidence in this forecast
        reasoning_trace: Chain of reasoning steps that led to this prediction
        evidence_sources: Sources of evidence used in making this prediction
        timestamp: When this forecast was made
        agent_id: ID of the agent that made this forecast
        metadata: Additional forecast-specific data
        submission_id: ID if this forecast was submitted to tournament
        is_final: Whether this is the final forecast for this question
    """
    question_id: int
    prediction: Union[float, List[float], Dict[str, float]]
    confidence: Confidence
    reasoning_trace: List[ReasoningStep]
    evidence_sources: List[str]
    timestamp: datetime
    agent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    submission_id: Optional[str] = None
    is_final: bool = False

    def __post_init__(self) -> None:
        """Validate forecast data."""
        if self.question_id <= 0:
            raise ValueError(f"Question ID must be positive, got {self.question_id}")

        if not isinstance(self.reasoning_trace, list):
            raise ValueError("Reasoning trace must be a list")

        if not isinstance(self.evidence_sources, list):
            raise ValueError("Evidence sources must be a list")

        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

        # Validate prediction format
        self._validate_prediction()

    def _validate_prediction(self) -> None:
        """Validate the prediction format."""
        if isinstance(self.prediction, float):
            # Binary or numeric prediction
            if not 0.0 <= self.prediction <= 1.0:
                # Allow numeric predictions outside 0-1 range for non-probability predictions
                pass
        elif isinstance(self.prediction, list):
            # Multiple predictions (e.g., for ensemble or time series)
            if not self.prediction:
                raise ValueError("Prediction list cannot be empty")
            if not all(isinstance(p, (int, float)) for p in self.prediction):
                raise ValueError("All predictions in list must be numeric")
        elif isinstance(self.prediction, dict):
            # Multiple choice or conditional predictions
            if not self.prediction:
                raise ValueError("Prediction dictionary cannot be empty")
            if not all(isinstance(v, (int, float)) for v in self.prediction.values()):
                raise ValueError("All prediction values must be numeric")
        else:
            raise ValueError(f"Invalid prediction type: {type(self.prediction)}")

    @classmethod
    def create_binary(
        cls,
        question_id: int,
        probability: float,
        confidence_level: float,
        confidence_basis: str,
        reasoning_trace: List[ReasoningStep],
        evidence_sources: List[str],
        agent_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> "Forecast":
        """Create a binary forecast.

        Args:
            question_id: ID of the question
            probability: Probability between 0.0 and 1.0
            confidence_level: Confidence in this forecast (0.0-1.0)
            confidence_basis: Basis for confidence level
            reasoning_trace: Steps in reasoning process
            evidence_sources: Sources of evidence used
            agent_id: Optional agent identifier
            timestamp: Optional timestamp, defaults to now

        Returns:
            New Forecast instance for binary question
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Binary probability must be between 0.0 and 1.0, got {probability}")

        if timestamp is None:
            timestamp = datetime.utcnow()

        confidence = Confidence(level=confidence_level, basis=confidence_basis)

        return cls(
            question_id=question_id,
            prediction=probability,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
            evidence_sources=evidence_sources,
            timestamp=timestamp,
            agent_id=agent_id
        )

    @classmethod
    def create_numeric(
        cls,
        question_id: int,
        value: float,
        confidence_level: float,
        confidence_basis: str,
        reasoning_trace: List[ReasoningStep],
        evidence_sources: List[str],
        agent_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> "Forecast":
        """Create a numeric forecast.

        Args:
            question_id: ID of the question
            value: Numeric prediction value
            confidence_level: Confidence in this forecast (0.0-1.0)
            confidence_basis: Basis for confidence level
            reasoning_trace: Steps in reasoning process
            evidence_sources: Sources of evidence used
            agent_id: Optional agent identifier
            timestamp: Optional timestamp, defaults to now

        Returns:
            New Forecast instance for numeric question
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        confidence = Confidence(level=confidence_level, basis=confidence_basis)

        return cls(
            question_id=question_id,
            prediction=value,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
            evidence_sources=evidence_sources,
            timestamp=timestamp,
            agent_id=agent_id
        )

    @classmethod
    def create_multiple_choice(
        cls,
        question_id: int,
        choice_probabilities: Dict[str, float],
        confidence_level: float,
        confidence_basis: str,
        reasoning_trace: List[ReasoningStep],
        evidence_sources: List[str],
        agent_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> "Forecast":
        """Create a multiple choice forecast.

        Args:
            question_id: ID of the question
            choice_probabilities: Probabilities for each choice
            confidence_level: Confidence in this forecast (0.0-1.0)
            confidence_basis: Basis for confidence level
            reasoning_trace: Steps in reasoning process
            evidence_sources: Sources of evidence used
            agent_id: Optional agent identifier
            timestamp: Optional timestamp, defaults to now

        Returns:
            New Forecast instance for multiple choice question
        """
        # Validate probabilities sum to approximately 1.0
        total_prob = sum(choice_probabilities.values())
        if not 0.99 <= total_prob <= 1.01:
            raise ValueError(f"Choice probabilities must sum to 1.0, got {total_prob}")

        if timestamp is None:
            timestamp = datetime.utcnow()

        confidence = Confidence(level=confidence_level, basis=confidence_basis)

        return cls(
            question_id=question_id,
            prediction=choice_probabilities,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
            evidence_sources=evidence_sources,
            timestamp=timestamp,
            agent_id=agent_id
        )

    def is_binary_forecast(self) -> bool:
        """Check if this is a binary forecast."""
        return isinstance(self.prediction, float) and 0.0 <= self.prediction <= 1.0

    def is_numeric_forecast(self) -> bool:
        """Check if this is a numeric forecast."""
        return isinstance(self.prediction, float) and not self.is_binary_forecast()

    def is_multiple_choice_forecast(self) -> bool:
        """Check if this is a multiple choice forecast."""
        return isinstance(self.prediction, dict)

    def get_binary_probability(self) -> float:
        """Get binary probability, raises error if not binary forecast."""
        if not self.is_binary_forecast():
            raise ValueError("Not a binary forecast")
        return self.prediction

    def get_numeric_value(self) -> float:
        """Get numeric value, raises error if not numeric forecast."""
        if not self.is_numeric_forecast():
            raise ValueError("Not a numeric forecast")
        return self.prediction

    def get_choice_probabilities(self) -> Dict[str, float]:
        """Get choice probabilities, raises error if not multiple choice forecast."""
        if not self.is_multiple_choice_forecast():
            raise ValueError("Not a multiple choice forecast")
        return self.prediction

    def get_most_likely_choice(self) -> str:
        """Get the most likely choice for multiple choice forecasts."""
        if not self.is_multiple_choice_forecast():
            raise ValueError("Not a multiple choice forecast")
        return max(self.prediction.items(), key=lambda x: x[1])[0]

    def has_high_confidence(self) -> bool:
        """Check if forecast has high confidence."""
        return self.confidence.is_high()

    def get_reasoning_summary(self) -> str:
        """Get a summary of the reasoning process."""
        if not self.reasoning_trace:
            return "No reasoning trace available"

        steps = [step.to_summary() for step in self.reasoning_trace]
        return " -> ".join(steps)

    def mark_as_final(self) -> "Forecast":
        """Mark this forecast as final."""
        return Forecast(
            question_id=self.question_id,
            prediction=self.prediction,
            confidence=self.confidence,
            reasoning_trace=self.reasoning_trace,
            evidence_sources=self.evidence_sources,
            timestamp=self.timestamp,
            agent_id=self.agent_id,
            metadata=self.metadata,
            submission_id=self.submission_id,
            is_final=True
        )

    def with_submission_id(self, submission_id: str) -> "Forecast":
        """Create a copy with submission ID set."""
        return Forecast(
            question_id=self.question_id,
            prediction=self.prediction,
            confidence=self.confidence,
            reasoning_trace=self.reasoning_trace,
            evidence_sources=self.evidence_sources,
            timestamp=self.timestamp,
            agent_id=self.agent_id,
            metadata=self.metadata,
            submission_id=submission_id,
            is_final=self.is_final
        )

    def to_summary(self) -> str:
        """Create a brief summary of the forecast."""
        pred_str = ""
        if self.is_binary_forecast():
            pred_str = f"{self.prediction:.3f}"
        elif self.is_numeric_forecast():
            pred_str = f"{self.prediction:.2f}"
        elif self.is_multiple_choice_forecast():
            most_likely = self.get_most_likely_choice()
            prob = self.prediction[most_likely]
            pred_str = f"{most_likely} ({prob:.3f})"

        confidence_desc = "high" if self.confidence.is_high() else "medium" if self.confidence.is_medium() else "low"
        final_str = " [FINAL]" if self.is_final else ""

        return f"Q{self.question_id}: {pred_str} (confidence: {confidence_desc}){final_str}"
