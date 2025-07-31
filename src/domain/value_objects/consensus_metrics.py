"""Consensus metrics value object for measuring ensemble consensus quality."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import statistics


@dataclass(frozen=True)
class ConsensusMetrics:
    """Ensemble consensus quality metrics with comprehensive analysis.

    Attributes:
        consensus_strength: Overall consensus strength (0.0-1.0, higher = more consensus)
        prediction_variance: Variance in predictions across agents
        agent_diversity_score: Diversity of agent approaches (0.0-1.0, higher = more diverse)
        confidence_alignment: Alignment of confidence levels across agents (0.0-1.0)
        disagreement_score: Level of disagreement between agents (0.0-1.0, higher = more disagreement)
        outlier_count: Number of outlier predictions
        metadata: Additional consensus-related metadata
    """
    consensus_strength: float
    prediction_variance: float
    agent_diversity_score: float
    confidence_alignment: float
    disagreement_score: float = 0.0
    outlier_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate consensus metrics."""
        if not 0.0 <= self.consensus_strength <= 1.0:
            raise ValueError(f"Consensus strength must be between 0.0 and 1.0, got {self.consensus_strength}")

        if self.prediction_variance < 0.0:
            raise ValueError(f"Prediction variance cannot be negative, got {self.prediction_variance}")

        if not 0.0 <= self.agent_diversity_score <= 1.0:
            raise ValueError(f"Agent diversity score must be between 0.0 and 1.0, got {self.agent_diversity_score}")

        if not 0.0 <= self.confidence_alignment <= 1.0:
            raise ValueError(f"Confidence alignment must be between 0.0 and 1.0, got {self.confidence_alignment}")

        if not 0.0 <= self.disagreement_score <= 1.0:
            raise ValueError(f"Disagreement score must be between 0.0 and 1.0, got {self.disagreement_score}")

        if self.outlier_count < 0:
            raise ValueError(f"Outlier count cannot be negative, got {self.outlier_count}")

        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

    @classmethod
    def calculate_from_predictions(
        cls,
        predictions: List[float],
        confidences: List[float],
        agent_types: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ConsensusMetrics":
        """Calculate consensus metrics from prediction data.

        Args:
            predictions: List of prediction values
            confidences: List of confidence levels
            agent_types: Optional list of agent type identifiers
            metadata: Optional additional metadata

        Returns:
            ConsensusMetrics calculated from the data
        """
        if not predictions:
            raise ValueError("Cannot calculate consensus metrics from empty predictions")

        if len(predictions) != len(confidences):
            raise ValueError("Predictions and confidences must have same length")

        if agent_types and len(agent_types) != len(predictions):
            raise ValueError("Agent types must match predictions length")

        # Calculate prediction variance
        pred_variance = statistics.variance(predictions) if len(predictions) > 1 else 0.0

        # Calculate consensus strength (inverse of normalized variance)
        if len(predictions) == 1:
            consensus_strength = 1.0
        else:
            # Normalize variance by prediction range
            pred_range = max(predictions) - min(predictions)
            if pred_range == 0:
                consensus_strength = 1.0
            else:
                normalized_variance = pred_variance / (pred_range ** 2)
                consensus_strength = max(0.0, 1.0 - normalized_variance)

        # Calculate confidence alignment
        conf_variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0
        confidence_alignment = max(0.0, 1.0 - (conf_variance * 4))  # Scale variance to 0-1

        # Calculate agent diversity score
        if agent_types:
            unique_types = len(set(agent_types))
            total_types = len(agent_types)
            agent_diversity_score = unique_types / total_types if total_types > 0 else 0.0
        else:
            # Use prediction spread as proxy for diversity
            if len(predictions) <= 1:
                agent_diversity_score = 0.0
            else:
                pred_std = statistics.stdev(predictions)
                # Normalize by reasonable prediction range (0-1 for probabilities)
                agent_diversity_score = min(1.0, pred_std * 2.0)

        # Calculate disagreement score
        if len(predictions) <= 1:
            disagreement_score = 0.0
        else:
            # Based on coefficient of variation
            mean_pred = statistics.mean(predictions)
            if mean_pred == 0:
                disagreement_score = 0.0
            else:
                cv = statistics.stdev(predictions) / abs(mean_pred)
                disagreement_score = min(1.0, cv)

        # Count outliers (predictions more than 2 standard deviations from mean)
        outlier_count = 0
        if len(predictions) > 2:
            mean_pred = statistics.mean(predictions)
            std_pred = statistics.stdev(predictions)
            if std_pred > 0:
                outlier_count = sum(
                    1 for pred in predictions
                    if abs(pred - mean_pred) > 2 * std_pred
                )

        return cls(
            consensus_strength=consensus_strength,
            prediction_variance=pred_variance,
            agent_diversity_score=agent_diversity_score,
            confidence_alignment=confidence_alignment,
            disagreement_score=disagreement_score,
            outlier_count=outlier_count,
            metadata=metadata or {}
        )

    @classmethod
    def create_high_consensus(
        cls,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ConsensusMetrics":
        """Create metrics representing high consensus."""
        return cls(
            consensus_strength=0.9,
            prediction_variance=0.01,
            agent_diversity_score=0.6,  # Some diversity is good
            confidence_alignment=0.9,
            disagreement_score=0.1,
            outlier_count=0,
            metadata=metadata
        )

    @classmethod
    def create_medium_consensus(
        cls,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ConsensusMetrics":
        """Create metrics representing medium consensus."""
        return cls(
            consensus_strength=0.6,
            prediction_variance=0.05,
            agent_diversity_score=0.7,
            confidence_alignment=0.6,
            disagreement_score=0.4,
            outlier_count=1,
            metadata=metadata
        )

    @classmethod
    def create_low_consensus(
        cls,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ConsensusMetrics":
        """Create metrics representing low consensus."""
        return cls(
            consensus_strength=0.3,
            prediction_variance=0.15,
            agent_diversity_score=0.9,
            confidence_alignment=0.3,
            disagreement_score=0.7,
            outlier_count=2,
            metadata=metadata
        )

    def is_high_consensus(self, threshold: float = 0.8) -> bool:
        """Check if consensus is high."""
        return self.consensus_strength >= threshold

    def is_medium_consensus(self, low_threshold: float = 0.4, high_threshold: float = 0.8) -> bool:
        """Check if consensus is medium."""
        return low_threshold <= self.consensus_strength < high_threshold

    def is_low_consensus(self, threshold: float = 0.4) -> bool:
        """Check if consensus is low."""
        return self.consensus_strength < threshold

    def is_diverse_ensemble(self, threshold: float = 0.6) -> bool:
        """Check if ensemble has good diversity."""
        return self.agent_diversity_score >= threshold

    def has_high_disagreement(self, threshold: float = 0.6) -> bool:
        """Check if there's high disagreement between agents."""
        return self.disagreement_score >= threshold

    def has_confidence_alignment(self, threshold: float = 0.7) -> bool:
        """Check if agents have aligned confidence levels."""
        return self.confidence_alignment >= threshold

    def has_outliers(self) -> bool:
        """Check if there are outlier predictions."""
        return self.outlier_count > 0

    def get_consensus_quality(self) -> str:
        """Get overall consensus quality as string."""
        if self.is_high_consensus():
            return "high"
        elif self.is_medium_consensus():
            return "medium"
        else:
            return "low"

    def get_ensemble_health_score(self) -> float:
        """Calculate overall ensemble health score (0.0-1.0).

        Combines consensus strength, diversity, and confidence alignment
        with appropriate weighting.
        """
        # Weight factors
        consensus_weight = 0.4
        diversity_weight = 0.3
        confidence_weight = 0.2
        disagreement_penalty = 0.1

        health_score = (
            self.consensus_strength * consensus_weight +
            self.agent_diversity_score * diversity_weight +
            self.confidence_alignment * confidence_weight -
            self.disagreement_score * disagreement_penalty
        )

        return max(0.0, min(1.0, health_score))

    def get_reliability_indicators(self) -> Dict[str, bool]:
        """Get reliability indicators for the ensemble."""
        return {
            "high_consensus": self.is_high_consensus(),
            "diverse_ensemble": self.is_diverse_ensemble(),
            "confidence_aligned": self.has_confidence_alignment(),
            "low_disagreement": not self.has_high_disagreement(),
            "no_outliers": not self.has_outliers(),
            "healthy_ensemble": self.get_ensemble_health_score() >= 0.7
        }

    def get_warning_flags(self) -> List[str]:
        """Get list of warning flags for ensemble quality."""
        warnings = []

        if self.is_low_consensus():
            warnings.append("low_consensus")

        if self.has_high_disagreement():
            warnings.append("high_disagreement")

        if not self.has_confidence_alignment():
            warnings.append("confidence_misalignment")

        if self.has_outliers():
            warnings.append(f"outliers_detected_{self.outlier_count}")

        if not self.is_diverse_ensemble():
            warnings.append("low_diversity")

        if self.prediction_variance > 0.1:
            warnings.append("high_variance")

        return warnings

    def compare_with(self, other: "ConsensusMetrics") -> Dict[str, str]:
        """Compare this consensus with another.

        Args:
            other: Another ConsensusMetrics to compare with

        Returns:
            Dictionary with comparison results
        """
        def compare_score(name: str, score1: float, score2: float) -> str:
            diff = score1 - score2
            if abs(diff) < 0.05:
                return "similar"
            elif diff > 0:
                return "higher"
            else:
                return "lower"

        return {
            "consensus_strength": compare_score("consensus", self.consensus_strength, other.consensus_strength),
            "diversity": compare_score("diversity", self.agent_diversity_score, other.agent_diversity_score),
            "confidence_alignment": compare_score("confidence", self.confidence_alignment, other.confidence_alignment),
            "disagreement": compare_score("disagreement", self.disagreement_score, other.disagreement_score),
            "overall_health": compare_score("health", self.get_ensemble_health_score(), other.get_ensemble_health_score())
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "consensus_strength": self.consensus_strength,
            "prediction_variance": self.prediction_variance,
            "agent_diversity_score": self.agent_diversity_score,
            "confidence_alignment": self.confidence_alignment,
            "disagreement_score": self.disagreement_score,
            "outlier_count": self.outlier_count,
            "consensus_quality": self.get_consensus_quality(),
            "ensemble_health_score": self.get_ensemble_health_score(),
            "reliability_indicators": self.get_reliability_indicators(),
            "warning_flags": self.get_warning_flags(),
            "metadata": self.metadata
        }

    def to_summary(self) -> str:
        """Create a brief summary of the consensus metrics."""
        quality = self.get_consensus_quality()
        health = self.get_ensemble_health_score()
        warnings = self.get_warning_flags()
        warning_str = f" (warnings: {len(warnings)})" if warnings else ""

        return f"Consensus: {quality} (strength: {self.consensus_strength:.2f}, health: {health:.2f}){warning_str}"
