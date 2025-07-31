"""Agent entity for representing forecasting agents and their characteristics."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class ReasoningStyle(Enum):
    """Different reasoning approaches used by agents."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"
    ENSEMBLE = "ensemble"
    BAYESIAN = "bayesian"
    FREQUENTIST = "frequentist"
    INTUITIVE = "intuitive"
    ANALYTICAL = "analytical"


class AggregationMethod(Enum):
    """Methods for aggregating ensemble predictions."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    META_REASONING = "meta_reasoning"


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent.

    Attributes:
        total_predictions: Total number of predictions made
        correct_predictions: Number of correct predictions
        average_confidence: Average confidence level
        calibration_score: How well-calibrated the agent is
        brier_score: Average Brier score
        log_score: Average log score
        accuracy_by_category: Accuracy broken down by question category
        confidence_intervals: Confidence interval performance
        last_updated: When these metrics were last calculated
    """
    total_predictions: int = 0
    correct_predictions: int = 0
    average_confidence: float = 0.0
    calibration_score: float = 0.0
    brier_score: float = 0.0
    log_score: float = 0.0
    accuracy_by_category: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, float]] = None
    last_updated: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate performance metrics."""
        if self.total_predictions < 0:
            raise ValueError(f"Total predictions cannot be negative, got {self.total_predictions}")

        if self.correct_predictions < 0:
            raise ValueError(f"Correct predictions cannot be negative, got {self.correct_predictions}")

        if self.correct_predictions > self.total_predictions:
            raise ValueError("Correct predictions cannot exceed total predictions")

        if not 0.0 <= self.average_confidence <= 1.0:
            raise ValueError(f"Average confidence must be between 0.0 and 1.0, got {self.average_confidence}")

        if self.accuracy_by_category is None:
            object.__setattr__(self, 'accuracy_by_category', {})

        if self.confidence_intervals is None:
            object.__setattr__(self, 'confidence_intervals', {})

        if self.last_updated is None:
            object.__setattr__(self, 'last_updated', datetime.utcnow())

    def get_accuracy(self) -> float:
        """Calculate overall accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def is_well_calibrated(self, threshold: float = 0.1) -> bool:
        """Check if agent is well-calibrated (calibration score close to 0)."""
        return abs(self.calibration_score) <= threshold

    def has_sufficient_data(self, min_predictions: int = 10) -> bool:
        """Check if there's sufficient data for reliable metrics."""
        return self.total_predictions >= min_predictions


@dataclass
class PerformanceHistory:
    """Historical performance data for an agent.

    Attributes:
        current_metrics: Current performance metrics
        historical_metrics: Historical performance over time
        tournament_performance: Performance in specific tournaments
        question_type_performance: Performance by question type
        recent_trend: Recent performance trend (improving/declining)
        peak_performance: Best historical performance
        metadata: Additional performance-related data
    """
    current_metrics: PerformanceMetrics
    historical_metrics: List[PerformanceMetrics]
    tournament_performance: Dict[int, PerformanceMetrics]
    question_type_performance: Dict[str, PerformanceMetrics]
    recent_trend: Optional[str] = None
    peak_performance: Optional[PerformanceMetrics] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate performance history."""
        if not isinstance(self.historical_metrics, list):
            raise ValueError("Historical metrics must be a list")

        if not isinstance(self.tournament_performance, dict):
            raise ValueError("Tournament performance must be a dictionary")

        if not isinstance(self.question_type_performance, dict):
            raise ValueError("Question type performance must be a dictionary")

        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

    def get_performance_trend(self, window_size: int = 5) -> str:
        """Analyze recent performance trend."""
        if len(self.historical_metrics) < window_size:
            return "insufficient_data"

        recent_metrics = self.historical_metrics[-window_size:]
        accuracies = [m.get_accuracy() for m in recent_metrics]

        # Simple trend analysis
        if len(accuracies) < 2:
            return "stable"

        trend_sum = sum(accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies)))

        if trend_sum > 0.05:
            return "improving"
        elif trend_sum < -0.05:
            return "declining"
        else:
            return "stable"

    def get_best_tournament_performance(self) -> Optional[tuple[int, PerformanceMetrics]]:
        """Get the tournament with best performance."""
        if not self.tournament_performance:
            return None

        best_tournament = max(
            self.tournament_performance.items(),
            key=lambda x: x[1].get_accuracy()
        )
        return best_tournament

    def get_strongest_question_type(self) -> Optional[str]:
        """Get the question type with best performance."""
        if not self.question_type_performance:
            return None

        return max(
            self.question_type_performance.items(),
            key=lambda x: x[1].get_accuracy()
        )[0]


@dataclass
class Agent:
    """Represents a forecasting agent with its characteristics and performance.

    Attributes:
        id: Unique identifier for the agent
        name: Human-readable name of the agent
        reasoning_style: Primary reasoning approach used
        knowledge_domains: Areas of specialized knowledge
        performance_history: Historical performance data
        configuration: Agent-specific configuration parameters
        created_at: When this agent was created
        last_active: When this agent was last used
        is_active: Whether this agent is currently active
        version: Version of the agent implementation
        metadata: Additional agent-specific data
    """
    id: str
    name: str
    reasoning_style: ReasoningStyle
    knowledge_domains: List[str]
    performance_history: PerformanceHistory
    configuration: Dict[str, Any]
    created_at: datetime
    last_active: Optional[datetime] = None
    is_active: bool = True
    version: str = "1.0.0"
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate agent data."""
        if not self.id or not self.id.strip():
            raise ValueError("Agent ID cannot be empty")

        if not self.name or not self.name.strip():
            raise ValueError("Agent name cannot be empty")

        if not isinstance(self.knowledge_domains, list):
            raise ValueError("Knowledge domains must be a list")

        if not isinstance(self.configuration, dict):
            raise ValueError("Configuration must be a dictionary")

        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

    def has_domain_expertise(self, domain: str) -> bool:
        """Check if agent has expertise in a specific domain."""
        return domain.lower() in [d.lower() for d in self.knowledge_domains]

    def get_current_accuracy(self) -> float:
        """Get current overall accuracy."""
        return self.performance_history.current_metrics.get_accuracy()

    def is_well_calibrated(self) -> bool:
        """Check if agent is well-calibrated."""
        return self.performance_history.current_metrics.is_well_calibrated()

    def has_sufficient_performance_data(self) -> bool:
        """Check if agent has sufficient performance data."""
        return self.performance_history.current_metrics.has_sufficient_data()

    def get_specialization_score(self, question_category: str) -> float:
        """Get specialization score for a question category."""
        # Check if agent has domain expertise
        domain_match = self.has_domain_expertise(question_category)

        # Check historical performance in this category
        category_performance = self.performance_history.current_metrics.accuracy_by_category.get(
            question_category, 0.0
        )

        # Combine domain expertise and historical performance
        base_score = 0.5  # Neutral score
        if domain_match:
            base_score += 0.3

        # Weight historical performance
        if category_performance > 0:
            base_score += category_performance * 0.2

        return min(base_score, 1.0)

    def is_suitable_for_question(self, question_category: str, required_confidence: float = 0.6) -> bool:
        """Check if agent is suitable for a specific question category."""
        specialization = self.get_specialization_score(question_category)
        return specialization >= required_confidence and self.is_active

    def update_last_active(self) -> "Agent":
        """Create updated agent with current timestamp as last_active."""
        return Agent(
            id=self.id,
            name=self.name,
            reasoning_style=self.reasoning_style,
            knowledge_domains=self.knowledge_domains,
            performance_history=self.performance_history,
            configuration=self.configuration,
            created_at=self.created_at,
            last_active=datetime.utcnow(),
            is_active=self.is_active,
            version=self.version,
            metadata=self.metadata
        )

    def deactivate(self) -> "Agent":
        """Create deactivated version of this agent."""
        return Agent(
            id=self.id,
            name=self.name,
            reasoning_style=self.reasoning_style,
            knowledge_domains=self.knowledge_domains,
            performance_history=self.performance_history,
            configuration=self.configuration,
            created_at=self.created_at,
            last_active=self.last_active,
            is_active=False,
            version=self.version,
            metadata=self.metadata
        )

    def update_performance(self, new_performance: PerformanceHistory) -> "Agent":
        """Create updated agent with new performance data."""
        return Agent(
            id=self.id,
            name=self.name,
            reasoning_style=self.reasoning_style,
            knowledge_domains=self.knowledge_domains,
            performance_history=new_performance,
            configuration=self.configuration,
            created_at=self.created_at,
            last_active=self.last_active,
            is_active=self.is_active,
            version=self.version,
            metadata=self.metadata
        )

    def get_agent_summary(self) -> Dict[str, Any]:
        """Get comprehensive agent summary."""
        return {
            "id": self.id,
            "name": self.name,
            "reasoning_style": self.reasoning_style.value,
            "knowledge_domains": self.knowledge_domains,
            "is_active": self.is_active,
            "current_accuracy": self.get_current_accuracy(),
            "is_well_calibrated": self.is_well_calibrated(),
            "total_predictions": self.performance_history.current_metrics.total_predictions,
            "performance_trend": self.performance_history.get_performance_trend(),
            "strongest_domain": self.performance_history.get_strongest_question_type(),
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None
        }

    def to_summary(self) -> str:
        """Create a brief summary of the agent."""
        accuracy = self.get_current_accuracy()
        status = "active" if self.is_active else "inactive"
        domains = ", ".join(self.knowledge_domains[:3])  # Show first 3 domains
        if len(self.knowledge_domains) > 3:
            domains += "..."

        return f"Agent {self.id}: {self.name} ({self.reasoning_style.value}) - {accuracy:.3f} accuracy, {status} [{domains}]"
