"""Prediction entity for representing individual agent predictions with metadata."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Union
from uuid import UUID, uuid4
from ..value_objects.confidence import Confidence
from ..value_objects.reasoning_step import ReasoningStep


@dataclass
class Prediction:
    """Individual agent prediction with comprehensive metadata.

    Attributes:
        id: Unique identifier for the prediction
        question_id: ID of the question being predicted
        result: The prediction result value(s)
        confidence: Confidence level in this prediction
        method: Method used to generate prediction
        reasoning: Explanation of prediction reasoning
        created_by: ID of agent that created this prediction
        timestamp: When prediction was created
        metadata: Additional prediction-specific data
        reasoning_steps: Detailed reasoning trace
        evidence_sources: Sources used for this prediction
    """
    id: UUID
    question_id: int
    result: Union[float, Dict[str, float]]
    confidence: Confidence
    method: str
    reasoning: str
    created_by: str
    timestamp: datetime
    metadata: Dict[str, Any]
    reasoning_steps: list[ReasoningStep]
    evidence_sources: list[str]

    def __post_init__(self) -> None:
        """Validate prediction data and set defaults."""
        if self.id is None:
            object.__setattr__(self, 'id', uuid4())

        if self.question_id <= 0:
            raise ValueError(f"Question ID must be positive, got {self.question_id}")

        if not self.created_by or not self.created_by.strip():
            raise ValueError("Created by cannot be empty")

        if not self.method or not self.method.strip():
            raise ValueError("Method cannot be empty")

        if not self.reasoning or not self.reasoning.strip():
            raise ValueError("Reasoning cannot be empty")

        if not isinstance(self.metadata, dict):
            object.__setattr__(self, 'metadata', {})

        if not isinstance(self.reasoning_steps, list):
            object.__setattr__(self, 'reasoning_steps', [])

        if not isinstance(self.evidence_sources, list):
            object.__setattr__(self, 'evidence_sources', [])

        # Validate prediction result format
        self._validate_result()

    def _validate_result(self) -> None:
        """Validate the prediction result format."""
        if isinstance(self.result, float):
            # Binary or numeric prediction
            pass  # Allow any float value
        elif isinstance(self.result, dict):
            # Multiple choice predictions
            if not self.result:
                raise ValueError("Prediction result dictionary cannot be empty")
            if not all(isinstance(v, (int, float)) for v in self.result.values()):
                raise ValueError("All prediction values must be numeric")
        else:
            raise ValueError(f"Invalid prediction result type: {type(self.result)}")

    @classmethod
    def create_binary(
        cls,
        question_id: int,
        probability: float,
        confidence_level: float,
        confidence_basis: str,
        method: str,
        reasoning: str,
        created_by: str,
        reasoning_steps: Optional[list[ReasoningStep]] = None,
        evidence_sources: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> "Prediction":
        """Create a binary prediction.

        Args:
            question_id: ID of the question
            probability: Probability between 0.0 and 1.0
            confidence_level: Confidence in this prediction (0.0-1.0)
            confidence_basis: Basis for confidence level
            method: Method used to generate prediction
            reasoning: Explanation of prediction reasoning
            created_by: Agent identifier
            reasoning_steps: Optional reasoning steps
            evidence_sources: Optional evidence sources
            metadata: Optional additional metadata
            timestamp: Optional timestamp, defaults to now

        Returns:
            New Prediction instance for binary question
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Binary probability must be between 0.0 and 1.0, got {probability}")

        if timestamp is None:
            timestamp = datetime.utcnow()

        confidence = Confidence(level=confidence_level, basis=confidence_basis)

        return cls(
            id=uuid4(),
            question_id=question_id,
            result=probability,
            confidence=confidence,
            method=method,
            reasoning=reasoning,
            created_by=created_by,
            timestamp=timestamp,
            metadata=metadata or {},
            reasoning_steps=reasoning_steps or [],
            evidence_sources=evidence_sources or []
        )

    @classmethod
    def create_numeric(
        cls,
        question_id: int,
        value: float,
        confidence_level: float,
        confidence_basis: str,
        method: str,
        reasoning: str,
        created_by: str,
        reasoning_steps: Optional[list[ReasoningStep]] = None,
        evidence_sources: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> "Prediction":
        """Create a numeric prediction.

        Args:
            question_id: ID of the question
            value: Numeric prediction value
            confidence_level: Confidence in this prediction (0.0-1.0)
            confidence_basis: Basis for confidence level
            method: Method used to generate prediction
            reasoning: Explanation of prediction reasoning
            created_by: Agent identifier
            reasoning_steps: Optional reasoning steps
            evidence_sources: Optional evidence sources
            metadata: Optional additional metadata
            timestamp: Optional timestamp, defaults to now

        Returns:
            New Prediction instance for numeric question
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        confidence = Confidence(level=confidence_level, basis=confidence_basis)

        return cls(
            id=uuid4(),
            question_id=question_id,
            result=value,
            confidence=confidence,
            method=method,
            reasoning=reasoning,
            created_by=created_by,
            timestamp=timestamp,
            metadata=metadata or {},
            reasoning_steps=reasoning_steps or [],
            evidence_sources=evidence_sources or []
        )

    @classmethod
    def create_multiple_choice(
        cls,
        question_id: int,
        choice_probabilities: Dict[str, float],
        confidence_level: float,
        confidence_basis: str,
        method: str,
        reasoning: str,
        created_by: str,
        reasoning_steps: Optional[list[ReasoningStep]] = None,
        evidence_sources: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> "Prediction":
        """Create a multiple choice prediction.

        Args:
            question_id: ID of the question
            choice_probabilities: Probabilities for each choice
            confidence_level: Confidence in this prediction (0.0-1.0)
            confidence_basis: Basis for confidence level
            method: Method used to generate prediction
            reasoning: Explanation of prediction reasoning
            created_by: Agent identifier
            reasoning_steps: Optional reasoning steps
            evidence_sources: Optional evidence sources
            metadata: Optional additional metadata
            timestamp: Optional timestamp, defaults to now

        Returns:
            New Prediction instance for multiple choice question
        """
        # Validate probabilities sum to approximately 1.0
        total_prob = sum(choice_probabilities.values())
        if not 0.99 <= total_prob <= 1.01:
            raise ValueError(f"Choice probabilities must sum to 1.0, got {total_prob}")

        if timestamp is None:
            timestamp = datetime.utcnow()

        confidence = Confidence(level=confidence_level, basis=confidence_basis)

        return cls(
            id=uuid4(),
            question_id=question_id,
            result=choice_probabilities,
            confidence=confidence,
            method=method,
            reasoning=reasoning,
            created_by=created_by,
            timestamp=timestamp,
            metadata=metadata or {},
            reasoning_steps=reasoning_steps or [],
            evidence_sources=evidence_sources or []
        )

    def is_binary_prediction(self) -> bool:
        """Check if this is a binary prediction."""
        return isinstance(self.result, float) and 0.0 <= self.result <= 1.0

    def is_numeric_prediction(self) -> bool:
        """Check if this is a numeric prediction."""
        return isinstance(self.result, float) and not self.is_binary_prediction()

    def is_multiple_choice_prediction(self) -> bool:
        """Check if this is a multiple choice prediction."""
        return isinstance(self.result, dict)

    def get_binary_probability(self) -> float:
        """Get binary probability, raises error if not binary prediction."""
        if not self.is_binary_prediction():
            raise ValueError("Not a binary prediction")
        return self.result

    def get_numeric_value(self) -> float:
        """Get numeric value, raises error if not numeric prediction."""
        if not self.is_numeric_prediction():
            raise ValueError("Not a numeric prediction")
        return self.result

    def get_choice_probabilities(self) -> Dict[str, float]:
        """Get choice probabilities, raises error if not multiple choice prediction."""
        if not self.is_multiple_choice_prediction():
            raise ValueError("Not a multiple choice prediction")
        return self.result

    def get_most_likely_choice(self) -> str:
        """Get the most likely choice for multiple choice predictions."""
        if not self.is_multiple_choice_prediction():
            raise ValueError("Not a multiple choice prediction")
        return max(self.result.items(), key=lambda x: x[1])[0]

    def has_high_confidence(self) -> bool:
        """Check if prediction has high confidence."""
        return self.confidence.is_high()

    def get_reasoning_summary(self) -> str:
        """Get a summary of the reasoning process."""
        if not self.reasoning_steps:
            return self.reasoning[:100] + "..." if len(self.reasoning) > 100 else self.reasoning

        steps = [step.to_summary() for step in self.reasoning_steps]
        return " -> ".join(steps)

    def to_submission_format(self) -> Dict[str, Any]:
        """Convert to Metaculus submission format."""
        if self.is_binary_prediction():
            return {
                "prediction": self.get_binary_probability(),
                "confidence": self.confidence.level,
                "reasoning": self.reasoning
            }
        elif self.is_numeric_prediction():
            return {
                "prediction": self.get_numeric_value(),
                "confidence": self.confidence.level,
                "reasoning": self.reasoning
            }
        elif self.is_multiple_choice_prediction():
            return {
                "prediction": self.get_choice_probabilities(),
                "confidence": self.confidence.level,
                "reasoning": self.reasoning
            }
        else:
            raise ValueError("Unknown prediction type")

    def to_summary(self) -> str:
        """Create a brief summary of the prediction."""
        pred_str = ""
        if self.is_binary_prediction():
            pred_str = f"{self.result:.3f}"
        elif self.is_numeric_prediction():
            pred_str = f"{self.result:.2f}"
        elif self.is_multiple_choice_prediction():
            most_likely = self.get_most_likely_choice()
            prob = self.result[most_likely]
            pred_str = f"{most_likely} ({prob:.3f})"

        confidence_desc = "high" if self.confidence.is_high() else "medium" if self.confidence.is_medium() else "low"

        return f"Q{self.question_id}: {pred_str} (confidence: {confidence_desc}) by {self.created_by}"
