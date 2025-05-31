"""Confidence level value object."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ConfidenceLevel:
    """Represents a confidence level for predictions."""
    
    value: float
    
    def __post_init__(self):
        """Validate confidence level value."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence level must be between 0 and 1, got {self.value}")
    
    @classmethod
    def from_percentage(cls, percentage: Union[int, float]) -> "ConfidenceLevel":
        """Create confidence level from percentage (0-100)."""
        return cls(value=percentage / 100.0)
    
    @classmethod
    def from_decimal(cls, decimal: float) -> "ConfidenceLevel":
        """Create confidence level from decimal (0.0-1.0)."""
        return cls(value=decimal)
    
    def to_percentage(self) -> float:
        """Convert to percentage (0-100)."""
        return self.value * 100.0
    
    def to_decimal(self) -> float:
        """Convert to decimal (0.0-1.0)."""
        return self.value
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.to_percentage():.1f}%"
    
    def __repr__(self) -> str:
        """Representation for debugging."""
        return f"ConfidenceLevel({self.value})"
