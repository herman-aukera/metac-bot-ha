"""Probability value object."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Probability:
    """
    Value object representing a probability value.
    
    Ensures probabilities are always between 0 and 1.
    """
    value: float
    
    def __post_init__(self):
        """Validate probability value."""
        if not 0 <= self.value <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {self.value}")
    
    @classmethod
    def from_percentage(cls, percentage: float) -> "Probability":
        """Create probability from percentage (0-100)."""
        return cls(percentage / 100.0)
    
    def to_percentage(self) -> float:
        """Convert to percentage."""
        return self.value * 100.0
    
    def complement(self) -> "Probability":
        """Get the complement probability (1 - p)."""
        return Probability(1.0 - self.value)
    
    def __str__(self) -> str:
        return f"{self.to_percentage():.1f}%"
    
    def __float__(self) -> float:
        return self.value
    
    def __add__(self, other: Union["Probability", float]) -> "Probability":
        if isinstance(other, Probability):
            return Probability(min(1.0, self.value + other.value))
        return Probability(min(1.0, self.value + other))
    
    def __mul__(self, other: Union["Probability", float]) -> "Probability":
        if isinstance(other, Probability):
            return Probability(self.value * other.value)
        return Probability(self.value * other)
    
    def __lt__(self, other: Union["Probability", float]) -> bool:
        if isinstance(other, Probability):
            return self.value < other.value
        return self.value < other
    
    def __le__(self, other: Union["Probability", float]) -> bool:
        if isinstance(other, Probability):
            return self.value <= other.value
        return self.value <= other
    
    def __gt__(self, other: Union["Probability", float]) -> bool:
        if isinstance(other, Probability):
            return self.value > other.value
        return self.value > other
    
    def __ge__(self, other: Union["Probability", float]) -> bool:
        if isinstance(other, Probability):
            return self.value >= other.value
        return self.value >= other
