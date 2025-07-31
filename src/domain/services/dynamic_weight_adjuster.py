"""
DynamicWeightAdjuster for performance-based adaptation of ensemble weights.

This service tracks agent performance over time, detects performance degradation,
and dynamically adjusts weights for optimal ensemble composition.
"""

from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from uuid import UUID
import statistics
import math
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import structlog

from ..entities.prediction import Prediction, PredictionMethod, PredictionConfidence
from ..value_objects.probability import Probability
from ..value_objects.confidence import ConfidenceLevel


logger = structlog.get_logger(__name__)


class PerformanceMetric(Enum):
    """Types of performance metrics to track."""
    BRIER_SCORE = "brier_score"
    ACCURACY = "accuracy"
    CALIBRATION = "calibration"
    LOG_SCORE = "log_score"
    CONFIDENCE_CORRELATION = "confidence_correlation"


class WeightAdjustmentStrategy(Enum):
    """Strategies for adjusting weights based on performance."""
    EXPONENTIAL_DECAY = "exponential_decay"
    LINEAR_DECAY = "linear_decay"
    THRESHOLD_BASED = "threshold_based"
    RELATIVE_RANKING = "relative_ranking"
    ADAPTIVE_LEARNING_RATE = "adaptive_learning_rate"


@dataclass
class PerformanceRecord:
    """Record of agent performance for a specific prediction."""
    agent_name: str
    prediction_id: UUID
    question_id: UUID
    timestamp: datetime
    predicted_probability: float
    actual_outcome: Optional[bool]
    brier_score: Optional[float]
    accuracy: Optional[float]
    confidence_score: float
    method: PredictionMethod


@dataclass
class AgentPerformanceProfile:
    """Comprehensive performance profile for an agent."""
    agent_name: str
    total_predictions: int
    recent_predictions: int
    overall_brier_score: float
    recent_brier_score: float
    overall_accuracy: float
    recent_accuracy: float
    calibration_score: float
    confidence_correlation: float
    performance_trend: float  # Positive = improving, negative = degrading
    consistency_score: float
    specialization_areas: List[str]
    current_weight: float
    recommended_weight: float
    last_updated: datetime


@dataclass
class EnsembleComposition:
    """Recommended ensemble composition with weights."""
    agent_weights: Dict[str, float]
    total_agents: int
    active_agents: int
    diversity_score: float
    expected_performance: float
    confidence_level: float
    composition_rationale: str


class DynamicWeightAdjuster:
    """
    Service for dynamically adjusting ensemble weights based on agent performance.

    Tracks historical performance, detects trends, and optimizes ensemble composition
    for maximum prediction accuracy and reliability.
    """

    def __init__(self,
                 lookback_window: int = 50,
                 min_predictions_for_weight: int = 5,
                 performance_decay_factor: float = 0.95):
        """
        Initialize the dynamic weight adjuster.

        Args:
            lookback_window: Number of recent predictions to consider
            min_predictions_for_weight: Minimum predictions needed for weight calculation
            performance_decay_factor: Factor for exponential decay of old performance
        """
        self.lookback_window = lookback_window
        self.min_predictions_for_weight = min_predictions_for_weight
        self.performance_decay_factor = performance_decay_factor

        # Performance tracking
        self.performance_records: List[PerformanceRecord] = []
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}

        # Weight adjustment parameters
        self.weight_adjustment_strategies = {
            WeightAdjustmentStrategy.EXPONENTIAL_DECAY: self._exponential_decay_weights,
            WeightAdjustmentStrategy.LINEAR_DECAY: self._linear_decay_weights,
            WeightAdjustmentStrategy.THRESHOLD_BASED: self._threshold_based_weights,
            WeightAdjustmentStrategy.RELATIVE_RANKING: self._relative_ranking_weights,
            WeightAdjustmentStrategy.ADAPTIVE_LEARNING_RATE: self._adaptive_learning_rate_weights
        }

        # Performance thresholds
        self.performance_thresholds = {
            "excellent": 0.15,  # Brier score threshold
            "good": 0.20,
            "average": 0.25,
            "poor": 0.30,
            "very_poor": 0.40
        }

        # Ensemble composition history
        self.composition_history: List[EnsembleComposition] = []

    def record_performance(self,
                          prediction: Prediction,
                          actual_outcome: bool,
                          timestamp: Optional[datetime] = None) -> None:
        """
        Record performance for a prediction.

        Args:
            prediction: The prediction that was made
            actual_outcome: The actual outcome (True/False)
            timestamp: When the outcome was resolved (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate performance metrics
        predicted_prob = prediction.result.binary_probability or 0.5
        brier_score = (predicted_prob - (1.0 if actual_outcome else 0.0)) ** 2
        accuracy = 1.0 if ((predicted_prob > 0.5) == actual_outcome) else 0.0

        # Create performance record
        record = PerformanceRecord(
            agent_name=prediction.created_by,
            prediction_id=prediction.id,
            question_id=prediction.question_id,
            timestamp=timestamp,
            predicted_probability=predicted_prob,
            actual_outcome=actual_outcome,
            brier_score=brier_score,
            accuracy=accuracy,
            confidence_score=prediction.get_confidence_score(),
            method=prediction.method
        )

        # Store record
        self.performance_records.append(record)

        # Keep only recent records
        cutoff_time = timestamp - timedelta(days=30)  # Keep 30 days of history
        self.performance_records = [
            r for r in self.performance_records
            if r.timestamp >= cutoff_time
        ]

        # Update agent profile
        self._update_agent_profile(record.agent_name)

        logger.info(
            "Performance recorded",
            agent=record.agent_name,
            brier_score=brier_score,
            accuracy=accuracy,
            predicted_prob=predicted_prob,
            actual_outcome=actual_outcome
        )

    def _update_agent_profile(self, agent_name: str) -> None:
        """Update performance profile for an agent."""
        agent_records = [r for r in self.performance_records if r.agent_name == agent_name]

        if not agent_records:
            return

        # Sort by timestamp
        agent_records.sort(key=lambda r: r.timestamp)

        # Calculate overall metrics
        total_predictions = len(agent_records)
        recent_records = agent_records[-self.lookback_window:]
        recent_predictions = len(recent_records)

        # Brier scores
        overall_brier = statistics.mean([r.brier_score for r in agent_records if r.brier_score is not None])
        recent_brier = statistics.mean([r.brier_score for r in recent_records if r.brier_score is not None])

        # Accuracy
        overall_accuracy = statistics.mean([r.accuracy for r in agent_records if r.accuracy is not None])
        recent_accuracy = statistics.mean([r.accuracy for r in recent_records if r.accuracy is not None])

        # Calibration score (simplified)
        calibration_score = self._calculate_calibration_score(agent_records)

        # Confidence correlation
        confidence_correlation = self._calculate_confidence_correlation(agent_records)

        # Performance trend
        performance_trend = self._calculate_performance_trend(agent_records)

        # Consistency score
        consistency_score = self._calculate_consistency_score(agent_records)

        # Specialization areas (simplified)
        specialization_areas = self._identify_specialization_areas(agent_records)

        # Current weight (if exists)
        current_weight = self.agent_profiles.get(agent_name, AgentPerformanceProfile(
            agent_name="", total_predictions=0, recent_predictions=0,
            overall_brier_score=0.25, recent_brier_score=0.25,
            overall_accuracy=0.5, recent_accuracy=0.5,
            calibration_score=0.5, confidence_correlation=0.0,
            performance_trend=0.0, consistency_score=0.5,
            specialization_areas=[], current_weight=1.0,
            recommended_weight=1.0, last_updated=datetime.now()
        )).current_weight

        # Calculate recommended weight
        recommended_weight = self._calculate_recommended_weight(
            recent_brier, performance_trend, consistency_score, recent_predictions
        )

        # Update profile
        self.agent_profiles[agent_name] = AgentPerformanceProfile(
            agent_name=agent_name,
            total_predictions=total_predictions,
            recent_predictions=recent_predictions,
            overall_brier_score=overall_brier,
            recent_brier_score=recent_brier,
            overall_accuracy=overall_accuracy,
            recent_accuracy=recent_accuracy,
            calibration_score=calibration_score,
            confidence_correlation=confidence_correlation,
            performance_trend=performance_trend,
            consistency_score=consistency_score,
            specialization_areas=specialization_areas,
            current_weight=current_weight,
            recommended_weight=recommended_weight,
            last_updated=datetime.now()
        )

    def _calculate_calibration_score(self, records: List[PerformanceRecord]) -> float:
        """Calculate calibration score for an agent."""
        if len(records) < 5:
            return 0.5  # Default for insufficient data

        # Group predictions by confidence bins
        bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        bin_data = {i: [] for i in range(len(bins))}

        for record in records:
            if record.actual_outcome is not None:
                for i, (low, high) in enumerate(bins):
                    if low <= record.predicted_probability < high:
                        bin_data[i].append((record.predicted_probability, record.actual_outcome))
                        break

        # Calculate calibration error
        calibration_error = 0.0
        total_predictions = 0

        for i, predictions in bin_data.items():
            if predictions:
                avg_predicted = statistics.mean([p[0] for p in predictions])
                avg_actual = statistics.mean([1.0 if p[1] else 0.0 for p in predictions])
                error = abs(avg_predicted - avg_actual)
                calibration_error += error * len(predictions)
                total_predictions += len(predictions)

        if total_predictions == 0:
            return 0.5

        calibration_error /= total_predictions

        # Convert to score (lower error = higher score)
        return max(0.0, 1.0 - calibration_error * 2)

    def _calculate_confidence_correlation(self, records: List[PerformanceRecord]) -> float:
        """Calculate correlation between confidence and accuracy."""
        if len(records) < 5:
            return 0.0

        confidences = [r.confidence_score for r in records]
        accuracies = [r.accuracy for r in records if r.accuracy is not None]

        if len(confidences) != len(accuracies):
            return 0.0

        # Calculate Pearson correlation
        try:
            mean_conf = statistics.mean(confidences)
            mean_acc = statistics.mean(accuracies)

            numerator = sum((c - mean_conf) * (a - mean_acc) for c, a in zip(confidences, accuracies))

            conf_var = sum((c - mean_conf) ** 2 for c in confidences)
            acc_var = sum((a - mean_acc) ** 2 for a in accuracies)

            if conf_var == 0 or acc_var == 0:
                return 0.0

            correlation = numerator / math.sqrt(conf_var * acc_var)
            return max(-1.0, min(1.0, correlation))

        except (ValueError, ZeroDivisionError):
            return 0.0

    def _calculate_performance_trend(self, records: List[PerformanceRecord]) -> float:
        """Calculate performance trend (positive = improving, negative = degrading)."""
        if len(records) < 10:
            return 0.0

        # Use recent vs older performance
        mid_point = len(records) // 2
        older_records = records[:mid_point]
        newer_records = records[mid_point:]

        older_brier = statistics.mean([r.brier_score for r in older_records if r.brier_score is not None])
        newer_brier = statistics.mean([r.brier_score for r in newer_records if r.brier_score is not None])

        # Lower Brier score is better, so improvement is negative change
        trend = older_brier - newer_brier

        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, trend * 10))

    def _calculate_consistency_score(self, records: List[PerformanceRecord]) -> float:
        """Calculate consistency score based on performance variance."""
        if len(records) < 5:
            return 0.5

        brier_scores = [r.brier_score for r in records if r.brier_score is not None]

        if len(brier_scores) < 2:
            return 0.5

        variance = statistics.variance(brier_scores)

        # Lower variance = higher consistency
        # Normalize assuming max reasonable variance is 0.1
        consistency = max(0.0, 1.0 - variance * 10)

        return consistency

    def _identify_specialization_areas(self, records: List[PerformanceRecord]) -> List[str]:
        """Identify areas where agent performs particularly well."""
        # Simplified implementation - could be enhanced with question categorization
        method_performance = {}

        for record in records:
            if record.brier_score is not None:
                method = record.method.value
                if method not in method_performance:
                    method_performance[method] = []
                method_performance[method].append(record.brier_score)

        specializations = []
        for method, scores in method_performance.items():
            if len(scores) >= 3:  # Minimum sample size
                avg_score = statistics.mean(scores)
                if avg_score < 0.2:  # Good performance threshold
                    specializations.append(method)

        return specializations

    def _calculate_recommended_weight(self,
                                    recent_brier: float,
                                    performance_trend: float,
                                    consistency_score: float,
                                    recent_predictions: int) -> float:
        """Calculate recommended weight for an agent."""
        if recent_predictions < self.min_predictions_for_weight:
            return 0.1  # Low weight for insufficient data

        # Base weight from performance (inverse of Brier score)
        base_weight = max(0.1, 1.0 - recent_brier * 2)

        # Trend adjustment
        trend_adjustment = 1.0 + (performance_trend * 0.2)

        # Consistency adjustment
        consistency_adjustment = 0.8 + (consistency_score * 0.4)

        # Sample size adjustment
        sample_adjustment = min(1.0, recent_predictions / 20.0)

        # Combine adjustments
        recommended_weight = base_weight * trend_adjustment * consistency_adjustment * sample_adjustment

        # Clamp to reasonable range
        return max(0.05, min(2.0, recommended_weight))

    def get_dynamic_weights(self,
                           agent_names: List[str],
                           strategy: WeightAdjustmentStrategy = WeightAdjustmentStrategy.ADAPTIVE_LEARNING_RATE) -> Dict[str, float]:
        """
        Get dynamically adjusted weights for a list of agents.

        Args:
            agent_names: List of agent names to get weights for
            strategy: Weight adjustment strategy to use

        Returns:
            Dictionary mapping agent names to weights
        """
        if strategy not in self.weight_adjustment_strategies:
            logger.warning(f"Unknown strategy {strategy}, using adaptive_learning_rate")
            strategy = WeightAdjustmentStrategy.ADAPTIVE_LEARNING_RATE

        return self.weight_adjustment_strategies[strategy](agent_names)

    def _exponential_decay_weights(self, agent_names: List[str]) -> Dict[str, float]:
        """Calculate weights using exponential decay based on recent performance."""
        weights = {}

        for agent_name in agent_names:
            profile = self.agent_profiles.get(agent_name)
            if profile and profile.recent_predictions >= self.min_predictions_for_weight:
                # Exponential decay based on Brier score
                decay_factor = math.exp(-profile.recent_brier_score * 5)
                weights[agent_name] = decay_factor
            else:
                weights[agent_name] = 0.5  # Default weight

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(agent_names)
            weights = {name: equal_weight for name in agent_names}

        return weights

    def _linear_decay_weights(self, agent_names: List[str]) -> Dict[str, float]:
        """Calculate weights using linear decay based on performance."""
        weights = {}

        for agent_name in agent_names:
            profile = self.agent_profiles.get(agent_name)
            if profile and profile.recent_predictions >= self.min_predictions_for_weight:
                # Linear decay: better performance (lower Brier) = higher weight
                weight = max(0.1, 1.0 - profile.recent_brier_score * 2)
                weights[agent_name] = weight
            else:
                weights[agent_name] = 0.5

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            equal_weight = 1.0 / len(agent_names)
            weights = {name: equal_weight for name in agent_names}

        return weights

    def _threshold_based_weights(self, agent_names: List[str]) -> Dict[str, float]:
        """Calculate weights using performance thresholds."""
        weights = {}

        for agent_name in agent_names:
            profile = self.agent_profiles.get(agent_name)
            if profile and profile.recent_predictions >= self.min_predictions_for_weight:
                brier = profile.recent_brier_score

                if brier <= self.performance_thresholds["excellent"]:
                    weight = 2.0
                elif brier <= self.performance_thresholds["good"]:
                    weight = 1.5
                elif brier <= self.performance_thresholds["average"]:
                    weight = 1.0
                elif brier <= self.performance_thresholds["poor"]:
                    weight = 0.5
                else:
                    weight = 0.1

                weights[agent_name] = weight
            else:
                weights[agent_name] = 1.0

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            equal_weight = 1.0 / len(agent_names)
            weights = {name: equal_weight for name in agent_names}

        return weights

    def _relative_ranking_weights(self, agent_names: List[str]) -> Dict[str, float]:
        """Calculate weights based on relative ranking of agents."""
        # Get performance scores for ranking
        agent_scores = {}
        for agent_name in agent_names:
            profile = self.agent_profiles.get(agent_name)
            if profile and profile.recent_predictions >= self.min_predictions_for_weight:
                # Combined score: lower Brier + positive trend + consistency
                score = (1.0 - profile.recent_brier_score) + profile.performance_trend * 0.2 + profile.consistency_score * 0.3
                agent_scores[agent_name] = score
            else:
                agent_scores[agent_name] = 0.5

        # Rank agents
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

        # Assign weights based on rank
        weights = {}
        total_agents = len(sorted_agents)

        for i, (agent_name, score) in enumerate(sorted_agents):
            # Higher rank = higher weight
            rank_weight = (total_agents - i) / total_agents
            weights[agent_name] = rank_weight

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}

        return weights

    def _adaptive_learning_rate_weights(self, agent_names: List[str]) -> Dict[str, float]:
        """Calculate weights using adaptive learning rate based on multiple factors."""
        weights = {}

        for agent_name in agent_names:
            profile = self.agent_profiles.get(agent_name)
            if profile and profile.recent_predictions >= self.min_predictions_for_weight:
                # Use the pre-calculated recommended weight
                weights[agent_name] = profile.recommended_weight
            else:
                weights[agent_name] = 0.5

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            equal_weight = 1.0 / len(agent_names)
            weights = {name: equal_weight for name in agent_names}

        return weights

    def detect_performance_degradation(self, agent_name: str) -> Tuple[bool, str]:
        """
        Detect if an agent's performance is degrading.

        Args:
            agent_name: Name of the agent to check

        Returns:
            Tuple of (is_degrading, explanation)
        """
        profile = self.agent_profiles.get(agent_name)
        if not profile:
            return False, "No performance data available"

        if profile.recent_predictions < 5:  # Lowered threshold for more sensitive detection
            return False, "Insufficient recent predictions for degradation analysis"

        # Check multiple indicators with enhanced sensitivity
        degradation_indicators = []

        # Performance trend (more sensitive threshold)
        if profile.performance_trend < -0.2:
            degradation_indicators.append("Negative performance trend")

        # Recent vs overall performance (more sensitive)
        if profile.recent_brier_score > profile.overall_brier_score * 1.15:
            degradation_indicators.append("Recent performance significantly worse than overall")

        # Absolute performance threshold
        if profile.recent_brier_score > 0.35:
            degradation_indicators.append("Recent performance below acceptable threshold")

        # Consistency drop (more sensitive)
        if profile.consistency_score < 0.4:
            degradation_indicators.append("Low consistency in recent predictions")

        # Confidence correlation (more sensitive)
        if profile.confidence_correlation < -0.1:
            degradation_indicators.append("Poor confidence calibration")

        # Recent accuracy drop
        if profile.recent_accuracy < 0.4:
            degradation_indicators.append("Low recent accuracy")

        # Require only 1 indicator for degradation (more sensitive)
        is_degrading = len(degradation_indicators) >= 1
        explanation = "; ".join(degradation_indicators) if degradation_indicators else "Performance appears stable"

        return is_degrading, explanation

    def recommend_ensemble_composition(self,
                                     available_agents: List[str],
                                     target_size: Optional[int] = None,
                                     diversity_weight: float = 0.3) -> EnsembleComposition:
        """
        Recommend optimal ensemble composition.

        Args:
            available_agents: List of available agent names
            target_size: Target number of agents (None for automatic)
            diversity_weight: Weight given to diversity vs performance

        Returns:
            Recommended ensemble composition
        """
        if not available_agents:
            return EnsembleComposition(
                agent_weights={}, total_agents=0, active_agents=0,
                diversity_score=0.0, expected_performance=0.0,
                confidence_level=0.0, composition_rationale="No agents available"
            )

        # Get performance-based weights
        performance_weights = self.get_dynamic_weights(
            available_agents, WeightAdjustmentStrategy.ADAPTIVE_LEARNING_RATE
        )

        # Calculate diversity scores
        diversity_scores = self._calculate_agent_diversity_scores(available_agents)

        # Combine performance and diversity
        combined_scores = {}
        for agent in available_agents:
            perf_score = performance_weights.get(agent, 0.5)
            div_score = diversity_scores.get(agent, 0.5)
            combined_scores[agent] = (1 - diversity_weight) * perf_score + diversity_weight * div_score

        # Select agents
        if target_size is None:
            # Automatic selection: include agents above threshold
            threshold = statistics.mean(combined_scores.values()) * 0.8
            selected_agents = [agent for agent, score in combined_scores.items() if score >= threshold]
            # Ensure at least 2 agents
            if len(selected_agents) < 2:
                selected_agents = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                selected_agents = [agent for agent, _ in selected_agents]
        else:
            # Select top N agents
            sorted_agents = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            selected_agents = [agent for agent, _ in sorted_agents[:target_size]]

        # Calculate final weights for selected agents
        final_weights = {}
        total_score = sum(combined_scores[agent] for agent in selected_agents)

        for agent in selected_agents:
            if total_score > 0:
                final_weights[agent] = combined_scores[agent] / total_score
            else:
                final_weights[agent] = 1.0 / len(selected_agents)

        # Calculate ensemble metrics
        diversity_score = self._calculate_ensemble_diversity(selected_agents)
        expected_performance = self._estimate_ensemble_performance(final_weights)
        confidence_level = self._calculate_ensemble_confidence(selected_agents)

        # Generate rationale
        rationale = self._generate_composition_rationale(
            selected_agents, final_weights, diversity_score, expected_performance
        )

        composition = EnsembleComposition(
            agent_weights=final_weights,
            total_agents=len(available_agents),
            active_agents=len(selected_agents),
            diversity_score=diversity_score,
            expected_performance=expected_performance,
            confidence_level=confidence_level,
            composition_rationale=rationale
        )

        # Store in history
        self.composition_history.append(composition)
        if len(self.composition_history) > 50:  # Keep recent history
            self.composition_history = self.composition_history[-50:]

        return composition

    def _calculate_agent_diversity_scores(self, agent_names: List[str]) -> Dict[str, float]:
        """Calculate diversity scores for agents."""
        diversity_scores = {}

        for agent in agent_names:
            profile = self.agent_profiles.get(agent)
            if profile:
                # Diversity based on specialization and method variety
                diversity = len(profile.specialization_areas) * 0.3

                # Add method diversity (simplified)
                agent_records = [r for r in self.performance_records if r.agent_name == agent]
                methods = set(r.method for r in agent_records[-20:])  # Recent methods
                method_diversity = len(methods) * 0.2

                diversity_scores[agent] = min(1.0, diversity + method_diversity + 0.5)
            else:
                diversity_scores[agent] = 0.5

        return diversity_scores

    def _calculate_ensemble_diversity(self, selected_agents: List[str]) -> float:
        """Calculate overall diversity of the ensemble."""
        if len(selected_agents) < 2:
            return 0.0

        # Method diversity
        all_methods = set()
        for agent in selected_agents:
            agent_records = [r for r in self.performance_records if r.agent_name == agent]
            agent_methods = set(r.method for r in agent_records[-10:])
            all_methods.update(agent_methods)

        method_diversity = len(all_methods) / 5.0  # Normalize by max expected methods

        # Performance diversity (different strengths)
        performance_variance = 0.0
        if len(selected_agents) > 1:
            performances = []
            for agent in selected_agents:
                profile = self.agent_profiles.get(agent)
                if profile:
                    performances.append(profile.recent_brier_score)

            if len(performances) > 1:
                performance_variance = statistics.variance(performances)

        # Combine diversity measures
        diversity = min(1.0, method_diversity + performance_variance * 2)

        return diversity

    def _estimate_ensemble_performance(self, agent_weights: Dict[str, float]) -> float:
        """Estimate expected performance of the ensemble."""
        if not agent_weights:
            return 0.5

        weighted_performance = 0.0
        total_weight = 0.0

        for agent, weight in agent_weights.items():
            profile = self.agent_profiles.get(agent)
            if profile:
                # Use inverse of Brier score as performance measure
                performance = max(0.1, 1.0 - profile.recent_brier_score)
                weighted_performance += performance * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_performance / total_weight
        else:
            return 0.5

    def _calculate_ensemble_confidence(self, selected_agents: List[str]) -> float:
        """Calculate confidence in the ensemble composition."""
        if not selected_agents:
            return 0.0

        # Base confidence on data availability and performance consistency
        confidence_factors = []

        for agent in selected_agents:
            profile = self.agent_profiles.get(agent)
            if profile:
                # Data availability factor
                data_factor = min(1.0, profile.recent_predictions / 20.0)

                # Performance consistency factor
                consistency_factor = profile.consistency_score

                # Trend factor
                trend_factor = max(0.0, 0.5 + profile.performance_trend * 0.5)

                agent_confidence = (data_factor + consistency_factor + trend_factor) / 3.0
                confidence_factors.append(agent_confidence)

        if confidence_factors:
            return statistics.mean(confidence_factors)
        else:
            return 0.5

    def _generate_composition_rationale(self,
                                      selected_agents: List[str],
                                      weights: Dict[str, float],
                                      diversity_score: float,
                                      expected_performance: float) -> str:
        """Generate human-readable rationale for ensemble composition."""
        rationale = f"Selected {len(selected_agents)} agents for ensemble:\n\n"

        # Sort agents by weight for presentation
        sorted_agents = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        for agent, weight in sorted_agents:
            profile = self.agent_profiles.get(agent)
            if profile:
                rationale += f"- {agent} (weight: {weight:.3f}): "
                rationale += f"Brier: {profile.recent_brier_score:.3f}, "
                rationale += f"Trend: {profile.performance_trend:+.2f}, "
                rationale += f"Consistency: {profile.consistency_score:.2f}\n"

        rationale += f"\nEnsemble Metrics:\n"
        rationale += f"- Diversity Score: {diversity_score:.3f}\n"
        rationale += f"- Expected Performance: {expected_performance:.3f}\n"

        if diversity_score > 0.7:
            rationale += "- High diversity provides robust predictions across different scenarios\n"
        elif diversity_score < 0.3:
            rationale += "- Low diversity may indicate similar approaches; consider adding diverse agents\n"

        if expected_performance > 0.8:
            rationale += "- High expected performance based on recent agent track records\n"
        elif expected_performance < 0.6:
            rationale += "- Moderate expected performance; monitor and adjust as needed\n"

        return rationale

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "total_agents": len(self.agent_profiles),
            "total_predictions": len(self.performance_records),
            "agent_profiles": {},
            "overall_metrics": {},
            "degradation_alerts": []
        }

        # Agent profiles
        for agent_name, profile in self.agent_profiles.items():
            summary["agent_profiles"][agent_name] = {
                "total_predictions": profile.total_predictions,
                "recent_predictions": profile.recent_predictions,
                "recent_brier_score": profile.recent_brier_score,
                "recent_accuracy": profile.recent_accuracy,
                "performance_trend": profile.performance_trend,
                "consistency_score": profile.consistency_score,
                "current_weight": profile.current_weight,
                "recommended_weight": profile.recommended_weight
            }

            # Check for degradation
            is_degrading, explanation = self.detect_performance_degradation(agent_name)
            if is_degrading:
                summary["degradation_alerts"].append({
                    "agent": agent_name,
                    "explanation": explanation
                })

        # Overall metrics
        if self.performance_records:
            all_brier_scores = [r.brier_score for r in self.performance_records if r.brier_score is not None]
            all_accuracies = [r.accuracy for r in self.performance_records if r.accuracy is not None]

            if all_brier_scores:
                summary["overall_metrics"]["mean_brier_score"] = statistics.mean(all_brier_scores)
                summary["overall_metrics"]["brier_score_std"] = statistics.stdev(all_brier_scores) if len(all_brier_scores) > 1 else 0.0

            if all_accuracies:
                summary["overall_metrics"]["mean_accuracy"] = statistics.mean(all_accuracies)

        return summary

    def reset_agent_performance(self, agent_name: str) -> None:
        """Reset performance tracking for a specific agent."""
        # Remove records for the agent
        self.performance_records = [r for r in self.performance_records if r.agent_name != agent_name]

        # Remove profile
        if agent_name in self.agent_profiles:
            del self.agent_profiles[agent_name]

        logger.info(f"Reset performance tracking for agent: {agent_name}")

    def get_weight_adjustment_history(self) -> List[Dict[str, Any]]:
        """Get history of weight adjustments."""
        history = []

        for composition in self.composition_history:
            history.append({
                "agent_weights": composition.agent_weights,
                "active_agents": composition.active_agents,
                "diversity_score": composition.diversity_score,
                "expected_performance": composition.expected_performance,
                "confidence_level": composition.confidence_level
            })

        return history

    def should_trigger_rebalancing(self, current_agents: List[str]) -> Tuple[bool, str]:
        """
        Determine if ensemble rebalancing should be triggered.

        Args:
            current_agents: List of currently active agents

        Returns:
            Tuple of (should_rebalance, reason)
        """
        rebalancing_reasons = []

        # Check for performance degradation in any agent
        degraded_agents = []
        for agent in current_agents:
            is_degrading, _ = self.detect_performance_degradation(agent)
            if is_degrading:
                degraded_agents.append(agent)

        if degraded_agents:
            rebalancing_reasons.append(f"Performance degradation detected in agents: {', '.join(degraded_agents)}")

        # Check for significant performance variance
        if len(current_agents) > 1:
            recent_scores = []
            for agent in current_agents:
                profile = self.agent_profiles.get(agent)
                if profile and profile.recent_predictions >= 3:
                    recent_scores.append(profile.recent_brier_score)

            if len(recent_scores) > 1:
                score_variance = statistics.variance(recent_scores)
                if score_variance > 0.05:  # High variance threshold
                    rebalancing_reasons.append("High performance variance between agents")

        # Check for new high-performing agents not in current ensemble
        all_agents = list(self.agent_profiles.keys())
        available_agents = [a for a in all_agents if a not in current_agents]

        if available_agents:
            current_avg_performance = 0.0
            if current_agents:
                current_scores = []
                for agent in current_agents:
                    profile = self.agent_profiles.get(agent)
                    if profile:
                        current_scores.append(1.0 - profile.recent_brier_score)  # Convert to performance score
                if current_scores:
                    current_avg_performance = statistics.mean(current_scores)

            # Check if any available agent significantly outperforms current ensemble
            for agent in available_agents:
                profile = self.agent_profiles.get(agent)
                if profile and profile.recent_predictions >= 5:
                    agent_performance = 1.0 - profile.recent_brier_score
                    if agent_performance > current_avg_performance * 1.2:
                        rebalancing_reasons.append(f"High-performing agent {agent} available for inclusion")
                        break

        # Check time since last rebalancing
        if self.composition_history:
            # Simplified: trigger rebalancing if we have reasons
            pass

        should_rebalance = len(rebalancing_reasons) > 0
        reason = "; ".join(rebalancing_reasons) if rebalancing_reasons else "No rebalancing needed"

        return should_rebalance, reason

    def select_optimal_agents_realtime(self,
                                     available_agents: List[str],
                                     max_agents: int = 5,
                                     performance_weight: float = 0.7,
                                     diversity_weight: float = 0.2,
                                     recency_weight: float = 0.1) -> List[str]:
        """
        Select optimal agents for real-time ensemble composition.

        Args:
            available_agents: List of available agent names
            max_agents: Maximum number of agents to select
            performance_weight: Weight for performance score
            diversity_weight: Weight for diversity score
            recency_weight: Weight for recent activity

        Returns:
            List of selected agent names
        """
        if not available_agents:
            return []

        agent_scores = {}

        for agent in available_agents:
            profile = self.agent_profiles.get(agent)
            if not profile:
                agent_scores[agent] = 0.1  # Low score for unknown agents
                continue

            # Performance score (inverse of Brier score)
            if profile.recent_predictions >= 3:
                performance_score = max(0.0, 1.0 - profile.recent_brier_score)

                # Adjust for trend
                trend_adjustment = 1.0 + (profile.performance_trend * 0.3)
                performance_score *= trend_adjustment

                # Adjust for consistency
                consistency_adjustment = 0.7 + (profile.consistency_score * 0.3)
                performance_score *= consistency_adjustment
            else:
                performance_score = 0.3  # Default for insufficient data

            # Diversity score
            diversity_score = len(profile.specialization_areas) * 0.2 + 0.5
            diversity_score = min(1.0, diversity_score)

            # Recency score (based on recent activity)
            if profile.recent_predictions > 0:
                recency_score = min(1.0, profile.recent_predictions / 10.0)
            else:
                recency_score = 0.0

            # Combined score
            combined_score = (
                performance_weight * performance_score +
                diversity_weight * diversity_score +
                recency_weight * recency_score
            )

            agent_scores[agent] = combined_score

        # Sort agents by score and select top performers
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        selected_agents = [agent for agent, score in sorted_agents[:max_agents] if score > 0.2]

        # Ensure minimum diversity if possible
        if len(selected_agents) < 2 and len(available_agents) >= 2:
            # Add second best agent even if score is low
            for agent, score in sorted_agents:
                if agent not in selected_agents:
                    selected_agents.append(agent)
                    break

        logger.info(
            "Real-time agent selection completed",
            available_agents=len(available_agents),
            selected_agents=len(selected_agents),
            selected=selected_agents,
            scores={agent: agent_scores[agent] for agent in selected_agents}
        )

        return selected_agents

    def trigger_automatic_rebalancing(self,
                                    current_agents: List[str],
                                    available_agents: List[str]) -> Optional[EnsembleComposition]:
        """
        Trigger automatic rebalancing if conditions are met.

        Args:
            current_agents: Currently active agents
            available_agents: All available agents

        Returns:
            New ensemble composition if rebalancing is triggered, None otherwise
        """
        should_rebalance, reason = self.should_trigger_rebalancing(current_agents)

        if not should_rebalance:
            logger.debug("Automatic rebalancing not triggered", reason=reason)
            return None

        logger.info("Triggering automatic rebalancing", reason=reason)

        # Select optimal agents
        optimal_agents = self.select_optimal_agents_realtime(available_agents)

        if not optimal_agents:
            logger.warning("No optimal agents found for rebalancing")
            return None

        # Generate new composition
        new_composition = self.recommend_ensemble_composition(
            optimal_agents,
            target_size=min(5, len(optimal_agents)),
            diversity_weight=0.3
        )

        # Add rebalancing metadata
        new_composition.composition_rationale += f"\n\nRebalancing triggered: {reason}"

        logger.info(
            "Automatic rebalancing completed",
            new_agents=list(new_composition.agent_weights.keys()),
            previous_agents=current_agents,
            reason=reason
        )

        return new_composition

    def get_rebalancing_recommendations(self, current_agents: List[str]) -> Dict[str, Any]:
        """
        Get recommendations for ensemble rebalancing.

        Args:
            current_agents: Currently active agents

        Returns:
            Dictionary with rebalancing recommendations
        """
        recommendations = {
            "should_rebalance": False,
            "reason": "",
            "degraded_agents": [],
            "recommended_additions": [],
            "recommended_removals": [],
            "performance_summary": {}
        }

        # Check rebalancing need
        should_rebalance, reason = self.should_trigger_rebalancing(current_agents)
        recommendations["should_rebalance"] = should_rebalance
        recommendations["reason"] = reason

        # Identify degraded agents
        for agent in current_agents:
            is_degrading, explanation = self.detect_performance_degradation(agent)
            if is_degrading:
                recommendations["degraded_agents"].append({
                    "agent": agent,
                    "explanation": explanation
                })

        # Find potential additions
        all_agents = list(self.agent_profiles.keys())
        available_agents = [a for a in all_agents if a not in current_agents]

        for agent in available_agents:
            profile = self.agent_profiles.get(agent)
            if profile and profile.recent_predictions >= 5:
                if profile.recent_brier_score < 0.2:  # Good performance threshold
                    recommendations["recommended_additions"].append({
                        "agent": agent,
                        "recent_brier_score": profile.recent_brier_score,
                        "performance_trend": profile.performance_trend,
                        "recent_predictions": profile.recent_predictions
                    })

        # Recommend removals based on poor performance
        for agent in current_agents:
            profile = self.agent_profiles.get(agent)
            if profile:
                if (profile.recent_brier_score > 0.4 or
                    profile.performance_trend < -0.3 or
                    profile.consistency_score < 0.2):
                    recommendations["recommended_removals"].append({
                        "agent": agent,
                        "recent_brier_score": profile.recent_brier_score,
                        "performance_trend": profile.performance_trend,
                        "consistency_score": profile.consistency_score
                    })

        # Performance summary
        if current_agents:
            brier_scores = []
            trends = []
            for agent in current_agents:
                profile = self.agent_profiles.get(agent)
                if profile:
                    brier_scores.append(profile.recent_brier_score)
                    trends.append(profile.performance_trend)

            if brier_scores:
                recommendations["performance_summary"] = {
                    "mean_brier_score": statistics.mean(brier_scores),
                    "brier_score_variance": statistics.variance(brier_scores) if len(brier_scores) > 1 else 0.0,
                    "mean_trend": statistics.mean(trends) if trends else 0.0,
                    "agents_count": len(current_agents)
                }

        return recommendations
