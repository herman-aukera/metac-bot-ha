"""
DivergenceAnalyzer for analyzing agent disagreement and consensus patterns.

This service analyzes disagreement between forecasting agents, identifies sources
of divergence, and provides strategies for resolving conflicts in predictions.
"""

from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from uuid import UUID
import statistics
import math
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import structlog

from ..entities.prediction import Prediction, PredictionMethod, PredictionConfidence
from ..value_objects.probability import Probability
from ..value_objects.confidence import ConfidenceLevel


logger = structlog.get_logger(__name__)


class DivergenceLevel(Enum):
    """Enumeration of divergence levels between predictions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DivergenceSource(Enum):
    """Enumeration of potential sources of divergence."""
    METHODOLOGY = "methodology"
    CONFIDENCE = "confidence"
    REASONING_QUALITY = "reasoning_quality"
    INFORMATION_ACCESS = "information_access"
    BIAS = "bias"
    UNCERTAINTY = "uncertainty"
    OUTLIER = "outlier"


@dataclass
class DivergenceMetrics:
    """Metrics describing divergence between predictions."""
    variance: float
    standard_deviation: float
    range_spread: float
    interquartile_range: float
    coefficient_of_variation: float
    entropy: float
    consensus_strength: float
    outlier_count: int


@dataclass
class AgentDivergenceProfile:
    """Profile of an agent's divergence patterns."""
    agent_name: str
    avg_distance_from_consensus: float
    outlier_frequency: float
    confidence_calibration: float
    reasoning_consistency: float
    method_bias: Optional[str]
    typical_divergence_sources: List[DivergenceSource]


@dataclass
class DivergenceAnalysis:
    """Complete analysis of divergence between predictions."""
    divergence_level: DivergenceLevel
    primary_sources: List[DivergenceSource]
    metrics: DivergenceMetrics
    agent_profiles: List[AgentDivergenceProfile]
    consensus_prediction: float
    confidence_adjustment: float
    resolution_strategy: str
    explanation: str


class DivergenceAnalyzer:
    """
    Service for analyzing disagreement between forecasting agents.

    Provides detailed analysis of prediction divergence, identifies sources
    of disagreement, and suggests resolution strategies.
    """

    def __init__(self):
        self.divergence_thresholds = {
            DivergenceLevel.VERY_LOW: 0.005,
            DivergenceLevel.LOW: 0.02,
            DivergenceLevel.MODERATE: 0.05,
            DivergenceLevel.HIGH: 0.1,
            DivergenceLevel.VERY_HIGH: float('inf')
        }

        self.resolution_strategies = {
            DivergenceLevel.VERY_LOW: "simple_average",
            DivergenceLevel.LOW: "confidence_weighted",
            DivergenceLevel.MODERATE: "meta_reasoning",
            DivergenceLevel.HIGH: "outlier_robust_mean",
            DivergenceLevel.VERY_HIGH: "expert_review_required"
        }

        # Historical divergence patterns for learning
        self.divergence_history: List[DivergenceAnalysis] = []
        self.agent_performance_patterns: Dict[str, List[float]] = {}

    def analyze_divergence(
        self,
        predictions: List[Prediction],
        include_agent_profiles: bool = True
    ) -> DivergenceAnalysis:
        """
        Perform comprehensive divergence analysis on predictions.

        Args:
            predictions: List of predictions to analyze
            include_agent_profiles: Whether to include detailed agent profiles

        Returns:
            Complete divergence analysis
        """
        if len(predictions) < 2:
            return self._create_minimal_analysis(predictions)

        logger.info(
            "Analyzing prediction divergence",
            prediction_count=len(predictions),
            question_id=str(predictions[0].question_id)
        )

        # Extract probabilities and basic info
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if len(probabilities) < 2:
            return self._create_minimal_analysis(predictions)

        # Calculate divergence metrics
        metrics = self._calculate_divergence_metrics(probabilities)

        # Determine divergence level
        divergence_level = self._classify_divergence_level(metrics.variance)

        # Identify divergence sources
        primary_sources = self._identify_divergence_sources(predictions, metrics)

        # Calculate consensus prediction
        consensus_prediction = self._calculate_consensus_prediction(predictions, divergence_level)

        # Calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(metrics, divergence_level)

        # Generate agent profiles if requested
        agent_profiles = []
        if include_agent_profiles:
            agent_profiles = self._generate_agent_profiles(predictions, consensus_prediction)

        # Select resolution strategy
        resolution_strategy = self._select_resolution_strategy(divergence_level, primary_sources)

        # Generate explanation
        explanation = self._generate_divergence_explanation(
            divergence_level, primary_sources, metrics, len(predictions)
        )

        analysis = DivergenceAnalysis(
            divergence_level=divergence_level,
            primary_sources=primary_sources,
            metrics=metrics,
            agent_profiles=agent_profiles,
            consensus_prediction=consensus_prediction,
            confidence_adjustment=confidence_adjustment,
            resolution_strategy=resolution_strategy,
            explanation=explanation
        )

        # Store for learning
        self.divergence_history.append(analysis)
        if len(self.divergence_history) > 100:  # Keep recent history
            self.divergence_history = self.divergence_history[-100:]

        logger.info(
            "Divergence analysis completed",
            divergence_level=divergence_level.value,
            primary_sources=[s.value for s in primary_sources],
            consensus_prediction=consensus_prediction
        )

        return analysis

    def _calculate_divergence_metrics(self, probabilities: List[float]) -> DivergenceMetrics:
        """Calculate comprehensive divergence metrics."""
        if len(probabilities) < 2:
            return DivergenceMetrics(
                variance=0.0, standard_deviation=0.0, range_spread=0.0,
                interquartile_range=0.0, coefficient_of_variation=0.0,
                entropy=0.0, consensus_strength=1.0, outlier_count=0
            )

        # Basic statistical measures
        mean_prob = statistics.mean(probabilities)
        variance = statistics.variance(probabilities)
        std_dev = math.sqrt(variance)
        range_spread = max(probabilities) - min(probabilities)

        # Interquartile range
        sorted_probs = sorted(probabilities)
        n = len(sorted_probs)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        iqr = sorted_probs[q3_idx] - sorted_probs[q1_idx] if n >= 4 else range_spread

        # Coefficient of variation (normalized standard deviation)
        cv = std_dev / mean_prob if mean_prob > 0 else 0.0

        # Entropy measure (information-theoretic divergence)
        entropy = self._calculate_prediction_entropy(probabilities)

        # Consensus strength (inverse of normalized variance)
        consensus_strength = max(0.0, 1.0 - (variance / 0.25))  # Normalize by max possible variance

        # Outlier detection
        outlier_count = self._count_outliers(probabilities)

        return DivergenceMetrics(
            variance=variance,
            standard_deviation=std_dev,
            range_spread=range_spread,
            interquartile_range=iqr,
            coefficient_of_variation=cv,
            entropy=entropy,
            consensus_strength=consensus_strength,
            outlier_count=outlier_count
        )

    def _calculate_prediction_entropy(self, probabilities: List[float]) -> float:
        """Calculate entropy measure of prediction divergence."""
        if len(probabilities) < 2:
            return 0.0

        # Bin probabilities into ranges to calculate entropy
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_counts = [0] * (len(bins) - 1)

        for prob in probabilities:
            for i in range(len(bins) - 1):
                if bins[i] <= prob < bins[i + 1]:
                    bin_counts[i] += 1
                    break
            else:
                bin_counts[-1] += 1  # Handle prob = 1.0

        # Calculate entropy
        total = len(probabilities)
        entropy = 0.0
        for count in bin_counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _count_outliers(self, probabilities: List[float]) -> int:
        """Count outliers using IQR method."""
        if len(probabilities) < 4:
            return 0

        sorted_probs = sorted(probabilities)
        n = len(sorted_probs)

        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_probs[q1_idx]
        q3 = sorted_probs[q3_idx]
        iqr = q3 - q1

        if iqr == 0:
            return 0

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = 0
        for prob in probabilities:
            if prob < lower_bound or prob > upper_bound:
                outlier_count += 1

        return outlier_count

    def _classify_divergence_level(self, variance: float) -> DivergenceLevel:
        """Classify divergence level based on variance."""
        for level, threshold in self.divergence_thresholds.items():
            if variance <= threshold:
                return level
        return DivergenceLevel.VERY_HIGH

    def _identify_divergence_sources(
        self,
        predictions: List[Prediction],
        metrics: DivergenceMetrics
    ) -> List[DivergenceSource]:
        """Identify likely sources of divergence between predictions."""
        sources = []

        # Analyze methodology differences
        methods = [p.method for p in predictions]
        if len(set(methods)) > 1:
            sources.append(DivergenceSource.METHODOLOGY)

        # Analyze confidence differences
        confidences = [p.get_confidence_score() for p in predictions]
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0
        if confidence_variance > 0.1:
            sources.append(DivergenceSource.CONFIDENCE)

        # Analyze reasoning quality differences
        reasoning_qualities = [self._assess_reasoning_quality(p) for p in predictions]
        quality_variance = statistics.variance(reasoning_qualities) if len(reasoning_qualities) > 1 else 0.0
        if quality_variance > 0.2:
            sources.append(DivergenceSource.REASONING_QUALITY)

        # Check for outliers
        if metrics.outlier_count > 0:
            sources.append(DivergenceSource.OUTLIER)

        # Check for high uncertainty
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]
        if probabilities:
            # High uncertainty if many predictions are near 0.5
            near_uncertain = sum(1 for p in probabilities if 0.4 <= p <= 0.6)
            if near_uncertain / len(probabilities) > 0.5:
                sources.append(DivergenceSource.UNCERTAINTY)

        # Check for potential bias (systematic deviations)
        if self._detect_systematic_bias(predictions):
            sources.append(DivergenceSource.BIAS)

        # If no specific sources identified but high divergence, assume information access differences
        if not sources and metrics.variance > 0.05:
            sources.append(DivergenceSource.INFORMATION_ACCESS)

        return sources[:3]  # Return top 3 sources

    def _assess_reasoning_quality(self, prediction: Prediction) -> float:
        """Assess the quality of reasoning in a prediction (0-1 scale)."""
        reasoning = prediction.reasoning or ""

        # Length factor (diminishing returns)
        length_score = min(1.0, len(reasoning) / 500.0)

        # Evidence indicators
        evidence_keywords = ["research", "study", "data", "evidence", "analysis", "according to"]
        evidence_score = sum(0.1 for keyword in evidence_keywords if keyword.lower() in reasoning.lower())
        evidence_score = min(0.5, evidence_score)

        # Logical structure
        structure_keywords = ["because", "therefore", "however", "furthermore", "in contrast"]
        structure_score = sum(0.05 for keyword in structure_keywords if keyword.lower() in reasoning.lower())
        structure_score = min(0.3, structure_score)

        # Uncertainty acknowledgment
        uncertainty_keywords = ["uncertain", "unclear", "might", "could", "possibly"]
        uncertainty_score = sum(0.02 for keyword in uncertainty_keywords if keyword.lower() in reasoning.lower())
        uncertainty_score = min(0.2, uncertainty_score)

        total_score = length_score + evidence_score + structure_score + uncertainty_score
        return min(1.0, total_score)

    def _detect_systematic_bias(self, predictions: List[Prediction]) -> bool:
        """Detect if there's systematic bias in predictions."""
        # Check if certain agents consistently predict higher/lower
        agent_predictions = {}
        for pred in predictions:
            agent_name = pred.created_by
            if agent_name not in agent_predictions:
                agent_predictions[agent_name] = []
            if pred.result.binary_probability is not None:
                agent_predictions[agent_name].append(pred.result.binary_probability)

        if len(agent_predictions) < 2:
            return False

        # Calculate mean prediction for each agent
        agent_means = {}
        for agent, probs in agent_predictions.items():
            if probs:
                agent_means[agent] = statistics.mean(probs)

        if len(agent_means) < 2:
            return False

        # Check if there's significant difference in means
        mean_values = list(agent_means.values())
        overall_variance = statistics.variance(mean_values)

        # Bias detected if variance in agent means is high
        return overall_variance > 0.05

    def _calculate_consensus_prediction(
        self,
        predictions: List[Prediction],
        divergence_level: DivergenceLevel
    ) -> float:
        """Calculate consensus prediction based on divergence level."""
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if not probabilities:
            return 0.5

        if divergence_level in [DivergenceLevel.VERY_LOW, DivergenceLevel.LOW]:
            # Simple average for low divergence
            return statistics.mean(probabilities)
        elif divergence_level == DivergenceLevel.MODERATE:
            # Median for moderate divergence
            return statistics.median(probabilities)
        else:
            # Trimmed mean for high divergence
            return self._trimmed_mean(probabilities, trim_percent=0.2)

    def _trimmed_mean(self, values: List[float], trim_percent: float = 0.1) -> float:
        """Calculate trimmed mean by removing extreme values."""
        if len(values) <= 2:
            return statistics.mean(values)

        sorted_values = sorted(values)
        trim_count = int(len(values) * trim_percent)

        if trim_count == 0:
            return statistics.mean(values)

        trimmed_values = sorted_values[trim_count:-trim_count]
        return statistics.mean(trimmed_values) if trimmed_values else statistics.mean(values)

    def _calculate_confidence_adjustment(
        self,
        metrics: DivergenceMetrics,
        divergence_level: DivergenceLevel
    ) -> float:
        """Calculate confidence adjustment based on divergence."""
        base_adjustment = {
            DivergenceLevel.VERY_LOW: 0.05,   # Increase confidence
            DivergenceLevel.LOW: 0.02,        # Slight increase
            DivergenceLevel.MODERATE: 0.0,    # No change
            DivergenceLevel.HIGH: -0.05,      # Decrease confidence
            DivergenceLevel.VERY_HIGH: -0.1   # Significant decrease
        }

        adjustment = base_adjustment.get(divergence_level, 0.0)

        # Additional adjustment based on outliers
        if metrics.outlier_count > 0:
            adjustment -= 0.02 * metrics.outlier_count

        # Consensus strength bonus
        if metrics.consensus_strength > 0.8:
            adjustment += 0.02

        return max(-0.2, min(0.1, adjustment))  # Clamp to reasonable range

    def _generate_agent_profiles(
        self,
        predictions: List[Prediction],
        consensus: float
    ) -> List[AgentDivergenceProfile]:
        """Generate divergence profiles for each agent."""
        profiles = []

        # Group predictions by agent
        agent_predictions = {}
        for pred in predictions:
            agent_name = pred.created_by
            if agent_name not in agent_predictions:
                agent_predictions[agent_name] = []
            agent_predictions[agent_name].append(pred)

        for agent_name, agent_preds in agent_predictions.items():
            profile = self._create_agent_profile(agent_name, agent_preds, consensus)
            profiles.append(profile)

        return profiles

    def _create_agent_profile(
        self,
        agent_name: str,
        predictions: List[Prediction],
        consensus: float
    ) -> AgentDivergenceProfile:
        """Create divergence profile for a single agent."""
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if not probabilities:
            return AgentDivergenceProfile(
                agent_name=agent_name,
                avg_distance_from_consensus=0.0,
                outlier_frequency=0.0,
                confidence_calibration=0.5,
                reasoning_consistency=0.5,
                method_bias=None,
                typical_divergence_sources=[]
            )

        # Calculate average distance from consensus
        distances = [abs(p - consensus) for p in probabilities]
        avg_distance = statistics.mean(distances)

        # Calculate outlier frequency
        outlier_count = sum(1 for d in distances if d > 0.2)  # Threshold for outlier
        outlier_frequency = outlier_count / len(distances)

        # Assess confidence calibration (simplified)
        confidences = [p.get_confidence_score() for p in predictions]
        confidence_calibration = statistics.mean(confidences) if confidences else 0.5

        # Assess reasoning consistency
        reasoning_qualities = [self._assess_reasoning_quality(p) for p in predictions]
        reasoning_consistency = 1.0 - statistics.stdev(reasoning_qualities) if len(reasoning_qualities) > 1 else 1.0

        # Identify method bias
        methods = [p.method for p in predictions]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        most_common_method = max(method_counts.items(), key=lambda x: x[1])[0] if method_counts else None
        method_bias = most_common_method.value if most_common_method else None

        # Identify typical divergence sources (simplified)
        typical_sources = []
        if outlier_frequency > 0.3:
            typical_sources.append(DivergenceSource.OUTLIER)
        if confidence_calibration < 0.3:
            typical_sources.append(DivergenceSource.CONFIDENCE)
        if reasoning_consistency < 0.5:
            typical_sources.append(DivergenceSource.REASONING_QUALITY)

        return AgentDivergenceProfile(
            agent_name=agent_name,
            avg_distance_from_consensus=avg_distance,
            outlier_frequency=outlier_frequency,
            confidence_calibration=confidence_calibration,
            reasoning_consistency=reasoning_consistency,
            method_bias=method_bias,
            typical_divergence_sources=typical_sources
        )

    def _select_resolution_strategy(
        self,
        divergence_level: DivergenceLevel,
        sources: List[DivergenceSource]
    ) -> str:
        """Select appropriate resolution strategy based on divergence analysis."""
        base_strategy = self.resolution_strategies.get(divergence_level, "meta_reasoning")

        # Adjust strategy based on divergence sources
        if DivergenceSource.OUTLIER in sources:
            return "outlier_robust_mean"
        elif DivergenceSource.CONFIDENCE in sources:
            return "confidence_weighted"
        elif DivergenceSource.METHODOLOGY in sources:
            return "meta_reasoning"
        elif DivergenceSource.BIAS in sources:
            return "bayesian_model_averaging"

        return base_strategy

    def _generate_divergence_explanation(
        self,
        level: DivergenceLevel,
        sources: List[DivergenceSource],
        metrics: DivergenceMetrics,
        prediction_count: int
    ) -> str:
        """Generate human-readable explanation of divergence analysis."""
        explanation = f"Divergence Analysis: {level.value.replace('_', ' ').title()} divergence detected among {prediction_count} predictions.\n\n"

        explanation += f"Key Metrics:\n"
        explanation += f"- Variance: {metrics.variance:.4f}\n"
        explanation += f"- Range: {metrics.range_spread:.3f}\n"
        explanation += f"- Consensus Strength: {metrics.consensus_strength:.2f}\n"
        explanation += f"- Outliers: {metrics.outlier_count}\n\n"

        if sources:
            explanation += f"Primary Divergence Sources:\n"
            for source in sources:
                explanation += f"- {source.value.replace('_', ' ').title()}\n"
            explanation += "\n"

        # Interpretation
        if level in [DivergenceLevel.VERY_LOW, DivergenceLevel.LOW]:
            explanation += "High agreement among agents suggests strong consensus. Confidence in ensemble prediction is increased."
        elif level == DivergenceLevel.MODERATE:
            explanation += "Moderate disagreement suggests some uncertainty. Standard ensemble methods should work well."
        else:
            explanation += "High disagreement suggests significant uncertainty or conflicting information. Robust aggregation methods recommended."

        return explanation

    def _create_minimal_analysis(self, predictions: List[Prediction]) -> DivergenceAnalysis:
        """Create minimal analysis for edge cases (single prediction, etc.)."""
        if not predictions:
            consensus = 0.5
        else:
            prob = predictions[0].result.binary_probability
            consensus = prob if prob is not None else 0.5

        analysis = DivergenceAnalysis(
            divergence_level=DivergenceLevel.VERY_LOW,
            primary_sources=[],
            metrics=DivergenceMetrics(
                variance=0.0, standard_deviation=0.0, range_spread=0.0,
                interquartile_range=0.0, coefficient_of_variation=0.0,
                entropy=0.0, consensus_strength=1.0, outlier_count=0
            ),
            agent_profiles=[],
            consensus_prediction=consensus,
            confidence_adjustment=0.0,
            resolution_strategy="simple_average",
            explanation="Insufficient predictions for divergence analysis."
        )

        # Store for learning even for minimal analyses
        self.divergence_history.append(analysis)
        if len(self.divergence_history) > 100:  # Keep recent history
            self.divergence_history = self.divergence_history[-100:]

        return analysis

    def get_divergence_patterns(self) -> Dict[str, Any]:
        """Get patterns from historical divergence analyses."""
        if not self.divergence_history:
            return {}

        # Analyze patterns in divergence levels
        level_counts = {}
        source_counts = {}

        for analysis in self.divergence_history:
            level = analysis.divergence_level
            level_counts[level] = level_counts.get(level, 0) + 1

            for source in analysis.primary_sources:
                source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "total_analyses": len(self.divergence_history),
            "divergence_level_distribution": {k.value: v for k, v in level_counts.items()},
            "common_divergence_sources": {k.value: v for k, v in source_counts.items()},
            "average_consensus_strength": statistics.mean([a.metrics.consensus_strength for a in self.divergence_history]),
            "average_confidence_adjustment": statistics.mean([a.confidence_adjustment for a in self.divergence_history])
        }

    def update_agent_performance_pattern(self, agent_name: str, accuracy_score: float) -> None:
        """Update agent performance patterns for better divergence analysis."""
        if agent_name not in self.agent_performance_patterns:
            self.agent_performance_patterns[agent_name] = []

        self.agent_performance_patterns[agent_name].append(accuracy_score)

        # Keep recent history
        if len(self.agent_performance_patterns[agent_name]) > 50:
            self.agent_performance_patterns[agent_name] = self.agent_performance_patterns[agent_name][-50:]
