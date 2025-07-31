"""Source credibility value object for evaluating source reliability and authority."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class SourceCredibility:
    """Source reliability and authority scoring with comprehensive evaluation.

    Attributes:
        authority_score: Score based on source authority and reputation (0.0-1.0)
        recency_score: Score based on how recent the information is (0.0-1.0)
        relevance_score: Score based on relevance to the question (0.0-1.0)
        cross_validation_score: Score based on cross-validation with other sources (0.0-1.0)
        bias_score: Score indicating potential bias (0.0=high bias, 1.0=low bias)
        methodology_score: Score for research methodology quality (0.0-1.0)
        metadata: Additional scoring metadata
    """
    authority_score: float
    recency_score: float
    relevance_score: float
    cross_validation_score: float
    bias_score: float = 0.5
    methodology_score: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate credibility scores."""
        scores = [
            ("authority_score", self.authority_score),
            ("recency_score", self.recency_score),
            ("relevance_score", self.relevance_score),
            ("cross_validation_score", self.cross_validation_score),
            ("bias_score", self.bias_score),
            ("methodology_score", self.methodology_score)
        ]

        for score_name, score_value in scores:
            if not 0.0 <= score_value <= 1.0:
                raise ValueError(f"{score_name} must be between 0.0 and 1.0, got {score_value}")

        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

    @classmethod
    def create(
        cls,
        authority_score: float,
        recency_score: float,
        relevance_score: float,
        cross_validation_score: float,
        bias_score: float = 0.5,
        methodology_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "SourceCredibility":
        """Create a source credibility assessment.

        Args:
            authority_score: Authority and reputation score
            recency_score: Information recency score
            relevance_score: Relevance to question score
            cross_validation_score: Cross-validation score
            bias_score: Bias assessment score (higher = less biased)
            methodology_score: Research methodology quality score
            metadata: Additional metadata

        Returns:
            New SourceCredibility instance
        """
        return cls(
            authority_score=authority_score,
            recency_score=recency_score,
            relevance_score=relevance_score,
            cross_validation_score=cross_validation_score,
            bias_score=bias_score,
            methodology_score=methodology_score,
            metadata=metadata
        )

    @classmethod
    def create_high_credibility(
        cls,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "SourceCredibility":
        """Create a high credibility assessment."""
        return cls(
            authority_score=0.9,
            recency_score=0.8,
            relevance_score=0.9,
            cross_validation_score=0.8,
            bias_score=0.8,
            methodology_score=0.9,
            metadata=metadata
        )

    @classmethod
    def create_medium_credibility(
        cls,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "SourceCredibility":
        """Create a medium credibility assessment."""
        return cls(
            authority_score=0.6,
            recency_score=0.6,
            relevance_score=0.7,
            cross_validation_score=0.6,
            bias_score=0.6,
            methodology_score=0.6,
            metadata=metadata
        )

    @classmethod
    def create_low_credibility(
        cls,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "SourceCredibility":
        """Create a low credibility assessment."""
        return cls(
            authority_score=0.3,
            recency_score=0.4,
            relevance_score=0.5,
            cross_validation_score=0.3,
            bias_score=0.4,
            methodology_score=0.3,
            metadata=metadata
        )

    @property
    def overall_score(self) -> float:
        """Calculate overall credibility score using weighted average.

        Weights:
        - Authority: 25%
        - Recency: 15%
        - Relevance: 25%
        - Cross-validation: 20%
        - Bias: 10%
        - Methodology: 5%
        """
        return (
            self.authority_score * 0.25 +
            self.recency_score * 0.15 +
            self.relevance_score * 0.25 +
            self.cross_validation_score * 0.20 +
            self.bias_score * 0.10 +
            self.methodology_score * 0.05
        )

    @property
    def core_score(self) -> float:
        """Calculate core credibility score using only primary factors.

        Uses authority, relevance, and cross-validation as core factors.
        """
        return (self.authority_score + self.relevance_score + self.cross_validation_score) / 3.0

    def is_high_credibility(self, threshold: float = 0.7) -> bool:
        """Check if source has high credibility."""
        return self.overall_score >= threshold

    def is_medium_credibility(self, low_threshold: float = 0.4, high_threshold: float = 0.7) -> bool:
        """Check if source has medium credibility."""
        return low_threshold <= self.overall_score < high_threshold

    def is_low_credibility(self, threshold: float = 0.4) -> bool:
        """Check if source has low credibility."""
        return self.overall_score < threshold

    def has_authority_concerns(self, threshold: float = 0.5) -> bool:
        """Check if source has authority concerns."""
        return self.authority_score < threshold

    def has_recency_concerns(self, threshold: float = 0.3) -> bool:
        """Check if source has recency concerns."""
        return self.recency_score < threshold

    def has_relevance_concerns(self, threshold: float = 0.5) -> bool:
        """Check if source has relevance concerns."""
        return self.relevance_score < threshold

    def has_bias_concerns(self, threshold: float = 0.4) -> bool:
        """Check if source has significant bias concerns."""
        return self.bias_score < threshold

    def has_methodology_concerns(self, threshold: float = 0.4) -> bool:
        """Check if source has methodology concerns."""
        return self.methodology_score < threshold

    def get_credibility_level(self) -> str:
        """Get credibility level as string."""
        if self.is_high_credibility():
            return "high"
        elif self.is_medium_credibility():
            return "medium"
        else:
            return "low"

    def get_primary_concerns(self) -> list[str]:
        """Get list of primary credibility concerns."""
        concerns = []

        if self.has_authority_concerns():
            concerns.append("authority")
        if self.has_recency_concerns():
            concerns.append("recency")
        if self.has_relevance_concerns():
            concerns.append("relevance")
        if self.cross_validation_score < 0.5:
            concerns.append("cross_validation")
        if self.has_bias_concerns():
            concerns.append("bias")
        if self.has_methodology_concerns():
            concerns.append("methodology")

        return concerns

    def combine_with(self, other: "SourceCredibility", weight: float = 0.5) -> "SourceCredibility":
        """Combine this credibility assessment with another using weighted average.

        Args:
            other: Another credibility assessment
            weight: Weight for this assessment (0.0-1.0), other gets (1-weight)

        Returns:
            New combined credibility assessment
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")

        other_weight = 1.0 - weight

        combined_metadata = {}
        if self.metadata:
            combined_metadata.update(self.metadata)
        if other.metadata:
            combined_metadata.update(other.metadata)
        combined_metadata["combination_weight"] = weight

        return SourceCredibility(
            authority_score=self.authority_score * weight + other.authority_score * other_weight,
            recency_score=self.recency_score * weight + other.recency_score * other_weight,
            relevance_score=self.relevance_score * weight + other.relevance_score * other_weight,
            cross_validation_score=self.cross_validation_score * weight + other.cross_validation_score * other_weight,
            bias_score=self.bias_score * weight + other.bias_score * other_weight,
            methodology_score=self.methodology_score * weight + other.methodology_score * other_weight,
            metadata=combined_metadata
        )

    def adjust_for_consensus(self, consensus_factor: float) -> "SourceCredibility":
        """Adjust credibility based on consensus with other sources.

        Args:
            consensus_factor: Factor indicating consensus level (0.0-1.0)

        Returns:
            New credibility assessment adjusted for consensus
        """
        if not 0.0 <= consensus_factor <= 1.0:
            raise ValueError(f"Consensus factor must be between 0.0 and 1.0, got {consensus_factor}")

        # Boost cross-validation score based on consensus
        adjusted_cross_validation = min(1.0, self.cross_validation_score + (consensus_factor * 0.2))

        # Slightly boost overall scores for high consensus
        boost_factor = 1.0 + (consensus_factor * 0.1)

        adjusted_metadata = dict(self.metadata) if self.metadata else {}
        adjusted_metadata["consensus_adjustment"] = consensus_factor

        return SourceCredibility(
            authority_score=min(1.0, self.authority_score * boost_factor),
            recency_score=self.recency_score,  # Recency not affected by consensus
            relevance_score=min(1.0, self.relevance_score * boost_factor),
            cross_validation_score=adjusted_cross_validation,
            bias_score=min(1.0, self.bias_score * boost_factor),
            methodology_score=min(1.0, self.methodology_score * boost_factor),
            metadata=adjusted_metadata
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "authority_score": self.authority_score,
            "recency_score": self.recency_score,
            "relevance_score": self.relevance_score,
            "cross_validation_score": self.cross_validation_score,
            "bias_score": self.bias_score,
            "methodology_score": self.methodology_score,
            "overall_score": self.overall_score,
            "core_score": self.core_score,
            "credibility_level": self.get_credibility_level(),
            "primary_concerns": self.get_primary_concerns(),
            "metadata": self.metadata
        }

    def to_summary(self) -> str:
        """Create a brief summary of the credibility assessment."""
        level = self.get_credibility_level()
        concerns = self.get_primary_concerns()
        concerns_str = f" (concerns: {', '.join(concerns)})" if concerns else ""

        return f"Credibility: {level} ({self.overall_score:.2f}){concerns_str}"
