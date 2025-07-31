"""Prediction result value object for structured prediction results with validation."""

from dataclasses import dataclass
from typing import Union, Dict, Optional, Any
from enum import Enum


class PredictionType(Enum):
    """Types of predictions."""
    BINARY = "binary"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"
    DATE = "date"
    CONDITIONAL = "conditional"


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction result with comprehensive validation.

    Attributes:
        value: The prediction value(s) - float for binary/numeric, dict for multiple choice
        prediction_type: Type of prediction being made
        bounds: Optional bounds for numeric predictions (min, max)
        metadata: Additional prediction-specific metadata
    """
    value: Union[float, Dict[str, float]]
    prediction_type: PredictionType
    bounds: Optional[tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate prediction result data."""
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

        # Validate prediction value based on type
        self._validate_value()

        # Validate bounds if provided
        if self.bounds is not None:
            self._validate_bounds()

    def _validate_value(self) -> None:
        """Validate prediction value based on prediction type."""
        if self.prediction_type == PredictionType.BINARY:
            if not isinstance(self.value, (int, float)):
                raise ValueError(f"Binary prediction must be numeric, got {type(self.value)}")
            if not 0.0 <= self.value <= 1.0:
                raise ValueError(f"Binary prediction must be between 0.0 and 1.0, got {self.value}")

        elif self.prediction_type == PredictionType.NUMERIC:
            if not isinstance(self.value, (int, float)):
                raise ValueError(f"Numeric prediction must be numeric, got {type(self.value)}")
            # Numeric predictions can be any real number

        elif self.prediction_type == PredictionType.MULTIPLE_CHOICE:
            if not isinstance(self.value, dict):
                raise ValueError(f"Multiple choice prediction must be dict, got {type(self.value)}")
            if not self.value:
                raise ValueError("Multiple choice prediction cannot be empty")

            # Validate all values are numeric
            for choice, prob in self.value.items():
                if not isinstance(prob, (int, float)):
                    raise ValueError(f"Choice probability must be numeric, got {type(prob)} for {choice}")
                if not 0.0 <= prob <= 1.0:
                    raise ValueError(f"Choice probability must be between 0.0 and 1.0, got {prob} for {choice}")

            # Validate probabilities sum to approximately 1.0
            total_prob = sum(self.value.values())
            if not 0.99 <= total_prob <= 1.01:
                raise ValueError(f"Choice probabilities must sum to 1.0, got {total_prob}")

        elif self.prediction_type == PredictionType.DATE:
            if not isinstance(self.value, (int, float)):
                raise ValueError(f"Date prediction must be numeric timestamp, got {type(self.value)}")

        elif self.prediction_type == PredictionType.CONDITIONAL:
            if not isinstance(self.value, dict):
                raise ValueError(f"Conditional prediction must be dict, got {type(self.value)}")
            if not self.value:
                raise ValueError("Conditional prediction cannot be empty")

    def _validate_bounds(self) -> None:
        """Validate bounds if provided."""
        if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
            raise ValueError("Bounds must be a tuple of (min, max)")

        min_val, max_val = self.bounds
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise ValueError("Bounds values must be numeric")

        if min_val >= max_val:
            raise ValueError(f"Minimum bound must be less than maximum, got {min_val} >= {max_val}")

        # Validate prediction is within bounds for numeric predictions
        if self.prediction_type == PredictionType.NUMERIC and isinstance(self.value, (int, float)):
            if not min_val <= self.value <= max_val:
                raise ValueError(f"Prediction {self.value} is outside bounds [{min_val}, {max_val}]")

    @classmethod
    def create_binary(cls, probability: float, metadata: Optional[Dict[str, Any]] = None) -> "PredictionResult":
        """Create a binary prediction result.

        Args:
            probability: Probability between 0.0 and 1.0
            metadata: Optional additional metadata

        Returns:
            PredictionResult for binary prediction
        """
        return cls(
            value=probability,
            prediction_type=PredictionType.BINARY,
            metadata=metadata
        )

    @classmethod
    def create_numeric(
        cls,
        value: float,
        bounds: Optional[tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "PredictionResult":
        """Create a numeric prediction result.

        Args:
            value: Numeric prediction value
            bounds: Optional bounds (min, max)
            metadata: Optional additional metadata

        Returns:
            PredictionResult for numeric prediction
        """
        return cls(
            value=value,
            prediction_type=PredictionType.NUMERIC,
            bounds=bounds,
            metadata=metadata
        )

    @classmethod
    def create_multiple_choice(
        cls,
        choice_probabilities: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> "PredictionResult":
        """Create a multiple choice prediction result.

        Args:
            choice_probabilities: Probabilities for each choice
            metadata: Optional additional metadata

        Returns:
            PredictionResult for multiple choice prediction
        """
        return cls(
            value=choice_probabilities,
            prediction_type=PredictionType.MULTIPLE_CHOICE,
            metadata=metadata
        )

    def validate(self) -> bool:
        """Validate prediction format and constraints.

        Returns:
            True if prediction is valid

        Raises:
            ValueError: If prediction is invalid
        """
        try:
            self._validate_value()
            if self.bounds is not None:
                self._validate_bounds()
            return True
        except ValueError:
            raise

    def is_binary(self) -> bool:
        """Check if this is a binary prediction."""
        return self.prediction_type == PredictionType.BINARY

    def is_numeric(self) -> bool:
        """Check if this is a numeric prediction."""
        return self.prediction_type == PredictionType.NUMERIC

    def is_multiple_choice(self) -> bool:
        """Check if this is a multiple choice prediction."""
        return self.prediction_type == PredictionType.MULTIPLE_CHOICE

    def get_binary_probability(self) -> float:
        """Get binary probability value."""
        if not self.is_binary():
            raise ValueError("Not a binary prediction")
        return float(self.value)

    def get_numeric_value(self) -> float:
        """Get numeric prediction value."""
        if not self.is_numeric():
            raise ValueError("Not a numeric prediction")
        return float(self.value)

    def get_choice_probabilities(self) -> Dict[str, float]:
        """Get choice probabilities."""
        if not self.is_multiple_choice():
            raise ValueError("Not a multiple choice prediction")
        return dict(self.value)

    def get_most_likely_choice(self) -> str:
        """Get the most likely choice for multiple choice predictions."""
        if not self.is_multiple_choice():
            raise ValueError("Not a multiple choice prediction")
        return max(self.value.items(), key=lambda x: x[1])[0]

    def is_within_bounds(self) -> bool:
        """Check if prediction is within specified bounds."""
        if self.bounds is None:
            return True

        if not self.is_numeric():
            return True

        min_val, max_val = self.bounds
        return min_val <= self.get_numeric_value() <= max_val

    def get_confidence_from_distribution(self) -> float:
        """Calculate confidence based on prediction distribution.

        For binary: distance from 0.5
        For multiple choice: entropy-based measure
        For numeric: inverse of relative uncertainty (if bounds available)
        """
        if self.is_binary():
            prob = self.get_binary_probability()
            # Distance from maximum uncertainty (0.5)
            return abs(prob - 0.5) * 2.0

        elif self.is_multiple_choice():
            probs = list(self.get_choice_probabilities().values())
            # Calculate entropy and convert to confidence
            import math
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(probs))
            return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        elif self.is_numeric() and self.bounds is not None:
            # Confidence based on position within bounds
            min_val, max_val = self.bounds
            range_size = max_val - min_val
            if range_size == 0:
                return 1.0

            # Simple heuristic: confidence decreases as we move toward extremes
            value = self.get_numeric_value()
            center = (min_val + max_val) / 2.0
            distance_from_center = abs(value - center)
            max_distance = range_size / 2.0
            return 1.0 - (distance_from_center / max_distance)

        else:
            return 0.5  # Default moderate confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "value": self.value,
            "prediction_type": self.prediction_type.value,
            "bounds": self.bounds,
            "metadata": self.metadata
        }

    def to_summary(self) -> str:
        """Create a brief summary of the prediction result."""
        if self.is_binary():
            return f"Binary: {self.get_binary_probability():.3f}"
        elif self.is_numeric():
            bounds_str = f" [{self.bounds[0]:.2f}, {self.bounds[1]:.2f}]" if self.bounds else ""
            return f"Numeric: {self.get_numeric_value():.2f}{bounds_str}"
        elif self.is_multiple_choice():
            most_likely = self.get_most_likely_choice()
            prob = self.value[most_likely]
            return f"Choice: {most_likely} ({prob:.3f})"
        else:
            return f"{self.prediction_type.value}: {self.value}"
