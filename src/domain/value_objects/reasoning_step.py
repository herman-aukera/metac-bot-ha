"""Reasoning step value object for documenting reasoning processes."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from .confidence import Confidence


@dataclass(frozen=True)
class ReasoningStep:
    """Represents a single step in a reasoning process.

    Attributes:
        step_number: Sequential number of this step in the reasoning chain
        description: Human-readable description of what this step accomplishes
        input_data: Data that was input to this reasoning step
        output_data: Data that was produced by this reasoning step
        confidence: Confidence level in this step's reasoning
        timestamp: When this reasoning step was executed
        reasoning_type: Type of reasoning used (e.g., 'deduction', 'induction', 'abduction')
    """
    step_number: int
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: Confidence
    timestamp: datetime
    reasoning_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate reasoning step data."""
        if self.step_number < 1:
            raise ValueError(f"Step number must be positive, got {self.step_number}")

        if not self.description or not self.description.strip():
            raise ValueError("Description cannot be empty")

        if not isinstance(self.input_data, dict):
            raise ValueError("Input data must be a dictionary")

        if not isinstance(self.output_data, dict):
            raise ValueError("Output data must be a dictionary")

    @classmethod
    def create(
        cls,
        step_number: int,
        description: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence_level: float,
        confidence_basis: str,
        reasoning_type: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> "ReasoningStep":
        """Create a reasoning step with automatic timestamp and confidence creation.

        Args:
            step_number: Sequential step number
            description: Description of the reasoning step
            input_data: Input data for this step
            output_data: Output data from this step
            confidence_level: Confidence level (0.0-1.0)
            confidence_basis: Basis for the confidence level
            reasoning_type: Optional type of reasoning used
            timestamp: Optional timestamp, defaults to now

        Returns:
            New ReasoningStep instance
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        confidence = Confidence(level=confidence_level, basis=confidence_basis)

        return cls(
            step_number=step_number,
            description=description,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            timestamp=timestamp,
            reasoning_type=reasoning_type
        )

    def has_high_confidence(self) -> bool:
        """Check if this step has high confidence."""
        return self.confidence.is_high()

    def get_key_outputs(self) -> Dict[str, Any]:
        """Get the most important outputs from this step."""
        # Filter out internal/debug keys and return main results
        return {k: v for k, v in self.output_data.items()
                if not k.startswith('_') and k in ['result', 'conclusion', 'prediction', 'analysis']}

    def to_summary(self) -> str:
        """Create a brief summary of this reasoning step."""
        confidence_desc = "high" if self.confidence.is_high() else "medium" if self.confidence.is_medium() else "low"
        return f"Step {self.step_number}: {self.description} (confidence: {confidence_desc})"
