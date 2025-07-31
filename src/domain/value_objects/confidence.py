"""Confidence value object for representing prediction confidence levels."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Confidence:
    """Represents confidence level in a prediction or reasoning step.

    Attributes:
        level: Confidence level between 0.0 and 1.0
        basis: Explanation of what the confidence level is based on
    """
    level: float
    basis: str

    def __post_init__(self) -> None:
        """Validate confidence level is within valid range."""
        if not 0.0 <= self.level <= 1.0:
            raise ValueError(f"Confidence level must be between 0.0 and 1.0, got {self.level}")

        if not self.basis or not self.basis.strip():
            raise ValueError("Confidence basis cannot be empty")

    @classmethod
    def high(cls, basis: str) -> "Confidence":
        """Create high confidence (0.8-1.0)."""
        return cls(level=0.9, basis=basis)

    @classmethod
    def medium(cls, basis: str) -> "Confidence":
        """Create medium confidence (0.4-0.8)."""
        return cls(level=0.6, basis=basis)

    @classmethod
    def low(cls, basis: str) -> "Confidence":
        """Create low confidence (0.0-0.4)."""
        return cls(level=0.3, basis=basis)

    def is_high(self) -> bool:
        """Check if confidence is high (>= 0.8)."""
        return self.level >= 0.8

    def is_medium(self) -> bool:
        """Check if confidence is medium (0.4 <= level < 0.8)."""
        return 0.4 <= self.level < 0.8

    def is_low(self) -> bool:
        """Check if confidence is low (< 0.4)."""
        return self.level < 0.4

    def combine_with(self, other: "Confidence", weight: float = 0.5) -> "Confidence":
        """Combine this confidence with another using weighted average.

        Args:
            other: Another confidence to combine with
            weight: Weight for this confidence (0.0-1.0), other gets (1-weight)

        Returns:
            New confidence with combined level and basis
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")

        combined_level = self.level * weight + other.level * (1 - weight)
        combined_basis = f"Combined: {self.basis} (weight: {weight:.2f}) + {other.basis} (weight: {1-weight:.2f})"

        return Confidence(level=combined_level, basis=combined_basis)
