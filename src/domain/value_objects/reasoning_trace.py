"""Reasoning trace value objects for transparent decision-making."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class ReasoningStepType(Enum):
    """Types of reasoning steps."""

    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"
    BIAS_CHECK = "bias_check"
    UNCERTAINTY_ASSESSMENT = "uncertainty_assessment"


@dataclass(frozen=True)
class ReasoningStep:
    """Individual step in a reasoning process."""

    id: UUID
    step_type: ReasoningStepType
    content: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate reasoning step."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

    @classmethod
    def create(
        cls,
        step_type: ReasoningStepType,
        content: str,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ReasoningStep":
        """Factory method to create a reasoning step."""
        return cls(
            id=uuid4(),
            step_type=step_type,
            content=content,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class ReasoningTrace:
    """Complete trace of reasoning process for transparency."""

    id: UUID
    question_id: UUID
    agent_id: str
    reasoning_method: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    bias_checks: List[str]
    uncertainty_sources: List[str]
    created_at: datetime

    def __post_init__(self):
        """Validate reasoning trace."""
        if not 0.0 <= self.overall_confidence <= 1.0:
            raise ValueError(
                f"Overall confidence must be between 0 and 1, got {self.overall_confidence}"
            )
        if not self.steps:
            raise ValueError("Reasoning trace must have at least one step")

    @classmethod
    def create(
        cls,
        question_id: UUID,
        agent_id: str,
        reasoning_method: str,
        steps: List[ReasoningStep],
        final_conclusion: str,
        overall_confidence: float,
        bias_checks: Optional[List[str]] = None,
        uncertainty_sources: Optional[List[str]] = None,
    ) -> "ReasoningTrace":
        """Factory method to create a reasoning trace."""
        return cls(
            id=uuid4(),
            question_id=question_id,
            agent_id=agent_id,
            reasoning_method=reasoning_method,
            steps=steps,
            final_conclusion=final_conclusion,
            overall_confidence=overall_confidence,
            bias_checks=bias_checks or [],
            uncertainty_sources=uncertainty_sources or [],
            created_at=datetime.utcnow(),
        )

    def get_step_by_type(self, step_type: ReasoningStepType) -> List[ReasoningStep]:
        """Get all steps of a specific type."""
        return [step for step in self.steps if step.step_type == step_type]

    def get_confidence_progression(self) -> List[float]:
        """Get confidence levels throughout the reasoning process."""
        return [step.confidence for step in self.steps]

    def has_bias_checks(self) -> bool:
        """Check if bias checks were performed."""
        return len(self.bias_checks) > 0 or any(
            step.step_type == ReasoningStepType.BIAS_CHECK for step in self.steps
        )

    def has_uncertainty_assessment(self) -> bool:
        """Check if uncertainty was assessed."""
        return len(self.uncertainty_sources) > 0 or any(
            step.step_type == ReasoningStepType.UNCERTAINTY_ASSESSMENT
            for step in self.steps
        )

    def get_reasoning_quality_score(self) -> float:
        """Calculate a quality score for the reasoning process."""
        base_score = 0.5

        # Bonus for having multiple step types
        unique_step_types = len(set(step.step_type for step in self.steps))
        step_diversity_bonus = min(0.2, unique_step_types * 0.05)

        # Bonus for bias checks
        bias_check_bonus = 0.1 if self.has_bias_checks() else 0.0

        # Bonus for uncertainty assessment
        uncertainty_bonus = 0.1 if self.has_uncertainty_assessment() else 0.0

        # Bonus for step count (more thorough reasoning)
        step_count_bonus = min(0.1, len(self.steps) * 0.01)

        return min(
            1.0,
            base_score
            + step_diversity_bonus
            + bias_check_bonus
            + uncertainty_bonus
            + step_count_bonus,
        )
