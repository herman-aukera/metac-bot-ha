"""Strategy result value object for representing tournament strategy outcomes."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from .confidence import Confidence


class StrategyType(Enum):
    """Types of tournament strategies."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    CONTRARIAN = "contrarian"
    MOMENTUM = "momentum"


class StrategyOutcome(Enum):
    """Possible outcomes of strategy execution."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    PENDING = "pending"


@dataclass(frozen=True)
class StrategyResult:
    """Represents the result of executing a tournament strategy.

    Attributes:
        strategy_type: Type of strategy that was executed
        outcome: Whether the strategy succeeded, failed, or is pending
        confidence: Confidence in the strategy's effectiveness
        expected_score: Expected scoring impact of this strategy
        actual_score: Actual score achieved (None if not yet resolved)
        reasoning: Explanation of why this strategy was chosen
        metadata: Additional strategy-specific data
        timestamp: When this strategy result was created
        question_ids: IDs of questions this strategy applies to
    """
    strategy_type: StrategyType
    outcome: StrategyOutcome
    confidence: Confidence
    expected_score: float
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime
    question_ids: List[int]
    actual_score: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate strategy result data."""
        if not self.reasoning or not self.reasoning.strip():
            raise ValueError("Strategy reasoning cannot be empty")

        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        if not isinstance(self.question_ids, list):
            raise ValueError("Question IDs must be a list")

        if any(not isinstance(qid, int) or qid <= 0 for qid in self.question_ids):
            raise ValueError("All question IDs must be positive integers")

    @classmethod
    def create(
        cls,
        strategy_type: StrategyType,
        expected_score: float,
        reasoning: str,
        question_ids: List[int],
        confidence_level: float,
        confidence_basis: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> "StrategyResult":
        """Create a new strategy result with pending outcome.

        Args:
            strategy_type: Type of strategy being executed
            expected_score: Expected scoring impact
            reasoning: Explanation for choosing this strategy
            question_ids: Questions this strategy applies to
            confidence_level: Confidence in strategy effectiveness (0.0-1.0)
            confidence_basis: Basis for confidence level
            metadata: Optional additional data
            timestamp: Optional timestamp, defaults to now

        Returns:
            New StrategyResult instance
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if metadata is None:
            metadata = {}

        confidence = Confidence(level=confidence_level, basis=confidence_basis)

        return cls(
            strategy_type=strategy_type,
            outcome=StrategyOutcome.PENDING,
            confidence=confidence,
            expected_score=expected_score,
            reasoning=reasoning,
            metadata=metadata,
            timestamp=timestamp,
            question_ids=question_ids
        )

    def mark_success(self, actual_score: float) -> "StrategyResult":
        """Mark strategy as successful with actual score."""
        return StrategyResult(
            strategy_type=self.strategy_type,
            outcome=StrategyOutcome.SUCCESS,
            confidence=self.confidence,
            expected_score=self.expected_score,
            actual_score=actual_score,
            reasoning=self.reasoning,
            metadata=self.metadata,
            timestamp=self.timestamp,
            question_ids=self.question_ids
        )

    def mark_failure(self, actual_score: float) -> "StrategyResult":
        """Mark strategy as failed with actual score."""
        return StrategyResult(
            strategy_type=self.strategy_type,
            outcome=StrategyOutcome.FAILURE,
            confidence=self.confidence,
            expected_score=self.expected_score,
            actual_score=actual_score,
            reasoning=self.reasoning,
            metadata=self.metadata,
            timestamp=self.timestamp,
            question_ids=self.question_ids
        )

    def is_successful(self) -> bool:
        """Check if strategy was successful."""
        return self.outcome == StrategyOutcome.SUCCESS

    def is_pending(self) -> bool:
        """Check if strategy outcome is still pending."""
        return self.outcome == StrategyOutcome.PENDING

    def get_score_difference(self) -> Optional[float]:
        """Get difference between actual and expected score."""
        if self.actual_score is None:
            return None
        return self.actual_score - self.expected_score

    def to_summary(self) -> str:
        """Create a brief summary of this strategy result."""
        status = self.outcome.value
        score_info = f"expected: {self.expected_score:.3f}"
        if self.actual_score is not None:
            score_info += f", actual: {self.actual_score:.3f}"

        return f"{self.strategy_type.value} strategy ({status}) - {score_info}"
