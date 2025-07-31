"""Real-time Learning and Adaptation System for tournament optimization.

This module implements a comprehensive learning system that includes:
- ML-based pattern recognition for prediction accuracy analysis
- Adaptive strategy refinement based on tournament performance feedback
- Dynamic prediction updating system for incorporating new information
- Tournament dynamics monitoring with real-time strategy adjustment
- Performance-based agent weighting and selection optimization
- Calibration monitoring and automatic correction mechanisms
- Historical performance analysis for identifying successful patterns
- A/B testing framework for strategy optimization validation
"""

import asyncio
import logging
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from enum import Enum
import json
from statistics import mean, median, stdev

from ...domain.entities.prediction import Prediction
from ...domain.entities.question import Question, QuestionType, QuestionCategory
from ...domain.entities.tournament import Tournament
from ...domain.value_objects.confidence import Confidence
from ...domain.value_objects.strategy_result import StrategyResult, StrategyType, StrategyOutcome
from ...domain.value_objects.prediction_result import PredictionResult


logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for different optimization approaches."""
    ACCURACY_FOCUSED = "accuracy_focused"
    CALIBRATION_FOCUSED = "calibration_focused"
    SCORE_FOCUSED = "score_focused"
    ADAPTIVE = "adaptive"


class PatternType(Enum):
    """Types of patterns that can be identified."""
    CATEGORY_PERFORMANCE = "category_performance"
    TIMING_PATTERNS = "timing_patterns"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    STRATEGY_EFFECTIVENESS = "strategy_effectiveness"
    AGENT_SPECIALIZATION = "agent_specialization"
    MARKET_DYNAMICS = "market_dynamics"


class AdaptationTrigger(Enum):
    """Triggers for strategy adaptation."""
    PERFORMANCE_DECLINE = "performance_decline"
    CALIBRATION_DRIFT = "calibration_drift"
    PATTERN_CHANGE = "pattern_change"
    NEW_INFORMATION = "new_information"
    TOURNAMENT_PHASE = "tournament_phase"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for analysis."""
    accuracy_score: float
    calibration_score: float
    brier_score: float
    log_score: float
    confidence_correlation: float
    prediction_count: int
    category_breakdown: Dict[QuestionCategory, float]
    time_period: Tuple[datetime, datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternInsight:
    """Represents a discovered pattern in performance data."""
    pattern_type: PatternType
    description: str
    confidence: float
    impact_score: float
    supporting_evidence: List[str]
    recommendations: List[str]
    discovered_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceProfile:
    """Performance profile for individual agents."""
    agent_id: str
    overall_accuracy: float
    category_specializations: Dict[QuestionCategory, float]
    confidence_calibration: float
    prediction_speed: float
    consistency_score: float
    recent_trend: float
    optimal_question_types: List[QuestionType]
    weight_recommendation: float
    last_updated: datetime


@dataclass
class CalibrationAnalysis:
    """Analysis of prediction calibration."""
    overall_calibration: float
    confidence_bins: Dict[str, Tuple[float, float]]  # bin -> (avg_confidence, accuracy)
    overconfidence_bias: float
    underconfidence_bias: float
    calibration_trend: List[Tuple[datetime, float]]
    recommendations: List[str]


@dataclass
class StrategyAdaptation:
    """Represents a strategy adaptation decision."""
    trigger: AdaptationTrigger
    old_strategy: StrategyType
    new_strategy: StrategyType
    confidence: float
    expected_improvement: float
    reasoning: str
    adaptation_time: datetime
    validation_period: timedelta


@dataclass
class ABTestResult:
    """Results from A/B testing of strategies."""
    test_id: str
    strategy_a: StrategyType
    strategy_b: StrategyType
    sample_size_a: int
    sample_size_b: int
    performance_a: float
    performance_b: float
    statistical_significance: float
    winner: Optional[StrategyType]
    confidence_interval: Tuple[float, float]
    test_duration: timedelta
    completed_at: datetime


class LearningService:
    """Advanced learning service with ML-based pattern recognition."""

    def __init__(self,
                 learning_mode: LearningMode = LearningMode.ADAPTIVE,
                 history_window_days: int = 30,
                 min_samples_for_learning: int = 10,
                 adaptation_threshold: float = 0.1,
                 calibration_target: float = 0.95):
        """Initialize learning service."""
        self.learning_mode = learning_mode
        self.history_window_days = history_window_days
        self.min_samples_for_learning = min_samples_for_learning
        self.adaptation_threshold = adaptation_threshold
        self.calibration_target = calibration_target

        # Performance tracking
        self.prediction_history: List[Tuple[Prediction, Optional[float]]] = []
        self.strategy_history: List[StrategyResult] = []
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.pattern_insights: List[PatternInsight] = []
        self.adaptation_history: List[StrategyAdaptation] = []

        # A/B testing framework
        self.active_ab_tests: Dict[str, Dict[str, Any]] = {}
        self.completed_ab_tests: List[ABTestResult] = []

        # Calibration monitoring
        self.calibration_history: deque = deque(maxlen=1000)
        self.last_calibration_check: Optional[datetime] = None

        # Pattern recognition models (placeholder for ML models)
        self.pattern_models: Dict[PatternType, Any] = {}

        logger.info(f"Initialized LearningService with mode: {learning_mode.value}")

    async def analyze_prediction_accuracy(self, predictions: List[Tuple[Prediction, Optional[float]]]) -> PerformanceMetrics:
        """Analyze prediction accuracy with detailed performance attribution."""
        if not predictions:
            raise ValueError("No predictions provided for analysis")

        resolved_predictions = [(p, actual) for p, actual in predictions if actual is not None]
        if len(resolved_predictions) < self.min_samples_for_learning:
            logger.warning(f"Insufficient resolved predictions for analysis: {len(resolved_predictions)}")
            return self._create_empty_metrics()

        # Calculate accuracy metrics
        accuracy_scores = []
        brier_scores = []
        log_scores = []
        confidence_scores = []
        category_breakdown = defaultdict(list)

        for prediction, actual_outcome in resolved_predictions:
            # Binary accuracy
            predicted_prob = self._extract_probability(prediction.result)
            accuracy = 1.0 if (predicted_prob > 0.5) == (actual_outcome > 0.5) else 0.0
            accuracy_scores.append(accuracy)

            # Brier score (lower is better)
            brier_score = (predicted_prob - actual_outcome) ** 2
            brier_scores.append(brier_score)

            # Log score (higher is better)
            epsilon = 1e-15  # Prevent log(0)
            prob_clamped = max(epsilon, min(1 - epsilon, predicted_prob))
            if actual_outcome == 1:
                log_score = np.log(prob_clamped)
            else:
                log_score = np.log(1 - prob_clamped)
            log_scores.append(log_score)

            confidence_scores.append(prediction.confidence.level)

            # Category breakdown
            if hasattr(prediction, 'question_category'):
                category_breakdown[prediction.question_category].append(accuracy)

        # Calculate overall metrics
        overall_accuracy = mean(accuracy_scores)
        overall_brier = mean(brier_scores)
        overall_log_score = mean(log_scores)
        confidence_correlation = np.corrcoef(accuracy_scores, confidence_scores)[0, 1] if len(accuracy_scores) > 1 else 0.0

        # Calculate calibration score
        calibration_score = self._calculate_calibration_score(resolved_predictions)

        # Category breakdown averages
        category_avg = {cat: mean(scores) for cat, scores in category_breakdown.items()}

        time_period = (
            min(p.timestamp for p, _ in resolved_predictions),
            max(p.timestamp for p, _ in resolved_predictions)
        )

        return PerformanceMetrics(
            accuracy_score=overall_accuracy,
            calibration_score=calibration_score,
            brier_score=overall_brier,
            log_score=overall_log_score,
            confidence_correlation=confidence_correlation,
            prediction_count=len(resolved_predictions),
            category_breakdown=category_avg,
            time_period=time_period,
            metadata={
                'confidence_std': stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0,
                'accuracy_std': stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0.0
            }
        )

    async def identify_improvement_opportunities(self, metrics: PerformanceMetrics) -> List[PatternInsight]:
        """Identify improvement opportunities from performance analysis."""
        insights = []

        # Accuracy improvement opportunities
        if metrics.accuracy_score < 0.6:
            insights.append(PatternInsight(
                pattern_type=PatternType.CATEGORY_PERFORMANCE,
                description=f"Overall accuracy is low at {metrics.accuracy_score:.2%}",
                confidence=0.9,
                impact_score=0.8,
                supporting_evidence=[f"Accuracy: {metrics.accuracy_score:.2%}", f"Sample size: {metrics.prediction_count}"],
                recommendations=[
                    "Increase research depth for predictions",
                    "Consider ensemble methods with more diverse agents",
                    "Review question categorization and specialized strategies"
                ],
                discovered_at=datetime.utcnow()
            ))

        # Calibration improvement opportunities
        if metrics.calibration_score < self.calibration_target:
            insights.append(PatternInsight(
                pattern_type=PatternType.CONFIDENCE_CALIBRATION,
                description=f"Calibration needs improvement: {metrics.calibration_score:.2%}",
                confidence=0.85,
                impact_score=0.7,
                supporting_evidence=[f"Calibration score: {metrics.calibration_score:.2%}"],
                recommendations=[
                    "Implement confidence adjustment mechanisms",
                    "Review confidence assignment algorithms",
                    "Add calibration-focused training data"
                ],
                discovered_at=datetime.utcnow()
            ))

        # Category-specific insights
        for category, accuracy in metrics.category_breakdown.items():
            if accuracy < metrics.accuracy_score - 0.1:  # Significantly below average
                insights.append(PatternInsight(
                    pattern_type=PatternType.CATEGORY_PERFORMANCE,
                    description=f"Poor performance in {category.value}: {accuracy:.2%}",
                    confidence=0.8,
                    impact_score=0.6,
                    supporting_evidence=[f"{category.value} accuracy: {accuracy:.2%}"],
                    recommendations=[
                        f"Develop specialized strategies for {category.value} questions",
                        f"Increase training data for {category.value} category",
                        f"Consider domain expert consultation for {category.value}"
                    ],
                    discovered_at=datetime.utcnow()
                ))

        # Confidence correlation insights
        if abs(metrics.confidence_correlation) < 0.3:
            insights.append(PatternInsight(
                pattern_type=PatternType.CONFIDENCE_CALIBRATION,
                description=f"Poor confidence-accuracy correlation: {metrics.confidence_correlation:.3f}",
                confidence=0.75,
                impact_score=0.5,
                supporting_evidence=[f"Correlation: {metrics.confidence_correlation:.3f}"],
                recommendations=[
                    "Recalibrate confidence scoring mechanisms",
                    "Implement uncertainty quantification improvements",
                    "Review confidence assignment criteria"
                ],
                discovered_at=datetime.utcnow()
            ))

        self.pattern_insights.extend(insights)
        logger.info(f"Identified {len(insights)} improvement opportunities")
        return insights

    async def refine_strategy_based_on_feedback(self, tournament: Tournament,
                                              recent_results: List[StrategyResult]) -> Optional[StrategyAdaptation]:
        """Implement adaptive strategy refinement based on tournament performance feedback."""
        if not recent_results:
            return None

        # Analyze recent strategy performance
        strategy_performance = defaultdict(list)
        for result in recent_results:
            if result.actual_score is not None:
                effectiveness = result.get_score_difference()
                if effectiveness is not None:
                    strategy_performance[result.strategy_type].append(effectiveness)

        # Calculate average effectiveness per strategy
        strategy_averages = {
            strategy: mean(scores) for strategy, scores in strategy_performance.items()
            if len(scores) >= 3  # Minimum sample size
        }

        if not strategy_averages:
            return None

        # Find best and worst performing strategies
        best_strategy = max(strategy_averages.keys(), key=lambda s: strategy_averages[s])
        worst_strategy = min(strategy_averages.keys(), key=lambda s: strategy_averages[s])

        current_strategy = self._get_current_strategy(tournament)
        performance_gap = strategy_averages[best_strategy] - strategy_averages.get(current_strategy, 0)

        # Decide if adaptation is needed
        if performance_gap > self.adaptation_threshold:
            adaptation = StrategyAdaptation(
                trigger=AdaptationTrigger.PERFORMANCE_DECLINE,
                old_strategy=current_strategy,
                new_strategy=best_strategy,
                confidence=min(0.9, performance_gap / self.adaptation_threshold),
                expected_improvement=performance_gap,
                reasoning=f"Strategy {best_strategy.value} outperforming current by {performance_gap:.3f}",
                adaptation_time=datetime.utcnow(),
                validation_period=timedelta(days=7)
            )

            self.adaptation_history.append(adaptation)
            logger.info(f"Strategy adaptation recommended: {current_strategy.value} -> {best_strategy.value}")
            return adaptation

        return None

    async def update_predictions_with_new_information(self,
                                                    question_id: str,
                                                    new_information: Dict[str, Any],
                                                    current_predictions: List[Prediction]) -> List[Prediction]:
        """Create dynamic prediction updating system for incorporating new information."""
        if not current_predictions:
            return []

        updated_predictions = []
        information_impact = self._assess_information_impact(new_information)

        for prediction in current_predictions:
            if prediction.question_id != question_id:
                updated_predictions.append(prediction)
                continue

            # Calculate confidence adjustment based on new information
            confidence_adjustment = self._calculate_confidence_adjustment(
                prediction, new_information, information_impact
            )

            # Update prediction if significant new information
            if abs(confidence_adjustment) > 0.05:  # 5% threshold
                new_confidence = Confidence(
                    level=max(0.0, min(1.0, prediction.confidence.level + confidence_adjustment)),
                    basis=f"Updated with new information: {prediction.confidence.basis}"
                )

                # Create updated prediction
                updated_prediction = Prediction(
                    id=prediction.id,
                    question_id=prediction.question_id,
                    result=prediction.result,
                    confidence=new_confidence,
                    method=f"{prediction.method}_updated",
                    reasoning=f"{prediction.reasoning}\n\nUpdated based on: {new_information.get('summary', 'new information')}",
                    created_by=prediction.created_by,
                    timestamp=datetime.utcnow(),
                    metadata={
                        **prediction.metadata,
                        'update_trigger': 'new_information',
                        'original_confidence': prediction.confidence.level,
                        'confidence_adjustment': confidence_adjustment
                    }
                )
                updated_predictions.append(updated_prediction)
                logger.info(f"Updated prediction {prediction.id} with confidence adjustment: {confidence_adjustment:+.3f}")
            else:
                updated_predictions.append(prediction)

        return updated_predictions

    async def monitor_tournament_dynamics(self, tournament: Tournament) -> Dict[str, Any]:
        """Add tournament dynamics monitoring with real-time strategy adjustment."""
        dynamics = {
            'tournament_phase': self._determine_tournament_phase(tournament),
            'competition_intensity': self._calculate_competition_intensity(tournament),
            'question_difficulty_trend': self._analyze_question_difficulty_trend(tournament),
            'participant_behavior_patterns': self._analyze_participant_patterns(tournament),
            'market_efficiency': self._assess_market_efficiency(tournament),
            'strategic_recommendations': []
        }

        # Generate strategic recommendations based on dynamics
        phase = dynamics['tournament_phase']
        if phase == 'early':
            dynamics['strategic_recommendations'].extend([
                "Focus on high-confidence predictions to establish early lead",
                "Prioritize research quality over speed",
                "Build reputation with consistent performance"
            ])
        elif phase == 'middle':
            dynamics['strategic_recommendations'].extend([
                "Balance risk-taking with consistency",
                "Monitor competitor strategies and adapt",
                "Focus on questions with highest scoring potential"
            ])
        elif phase == 'late':
            dynamics['strategic_recommendations'].extend([
                "Optimize for final ranking position",
                "Take calculated risks on high-impact questions",
                "Consider defensive strategies if leading"
            ])

        # Competition intensity adjustments
        if dynamics['competition_intensity'] > 0.8:
            dynamics['strategic_recommendations'].append(
                "High competition detected - consider contrarian strategies"
            )

        logger.info(f"Tournament dynamics analysis completed for {tournament.name}")
        return dynamics

    async def optimize_agent_weights(self, agents: List[str],
                                   recent_performance: Dict[str, PerformanceMetrics]) -> Dict[str, float]:
        """Implement performance-based agent weighting and selection optimization."""
        if not agents or not recent_performance:
            return {agent: 1.0 / len(agents) for agent in agents}

        # Update agent profiles
        for agent_id in agents:
            if agent_id in recent_performance:
                await self._update_agent_profile(agent_id, recent_performance[agent_id])

        # Calculate weights based on multiple factors
        weights = {}
        total_score = 0

        for agent_id in agents:
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                weights[agent_id] = 0.1  # Minimal weight for unknown agents
                continue

            # Multi-factor scoring
            accuracy_score = profile.overall_accuracy * 0.4
            calibration_score = profile.confidence_calibration * 0.3
            consistency_score = profile.consistency_score * 0.2
            trend_score = max(0, profile.recent_trend) * 0.1

            total_score_agent = accuracy_score + calibration_score + consistency_score + trend_score
            weights[agent_id] = total_score_agent
            total_score += total_score_agent

        # Normalize weights
        if total_score > 0:
            weights = {agent: weight / total_score for agent, weight in weights.items()}
        else:
            # Fallback to equal weights
            weights = {agent: 1.0 / len(agents) for agent in agents}

        # Apply minimum weight constraint (no agent below 5%)
        min_weight = 0.05
        for agent in weights:
            if weights[agent] < min_weight:
                weights[agent] = min_weight

        # Renormalize after minimum weight application
        total_weight = sum(weights.values())
        weights = {agent: weight / total_weight for agent, weight in weights.items()}

        logger.info(f"Optimized agent weights: {weights}")
        return weights

    async def monitor_calibration(self, recent_predictions: List[Tuple[Prediction, Optional[float]]]) -> CalibrationAnalysis:
        """Create calibration monitoring and automatic correction mechanisms."""
        resolved_predictions = [(p, actual) for p, actual in recent_predictions if actual is not None]

        if len(resolved_predictions) < self.min_samples_for_learning:
            return self._create_empty_calibration_analysis()

        # Create confidence bins
        confidence_bins = {
            '0.0-0.2': ([], []),
            '0.2-0.4': ([], []),
            '0.4-0.6': ([], []),
            '0.6-0.8': ([], []),
            '0.8-1.0': ([], [])
        }

        # Populate bins
        for prediction, actual in resolved_predictions:
            confidence = prediction.confidence.level
            bin_key = self._get_confidence_bin(confidence)
            confidence_bins[bin_key][0].append(confidence)
            confidence_bins[bin_key][1].append(actual)

        # Calculate bin statistics
        bin_stats = {}
        overall_calibration_error = 0
        total_predictions = 0

        for bin_key, (confidences, actuals) in confidence_bins.items():
            if confidences:
                avg_confidence = mean(confidences)
                avg_accuracy = mean(actuals)
                bin_stats[bin_key] = (avg_confidence, avg_accuracy)

                # Contribution to overall calibration error
                bin_error = abs(avg_confidence - avg_accuracy) * len(confidences)
                overall_calibration_error += bin_error
                total_predictions += len(confidences)

        overall_calibration = 1.0 - (overall_calibration_error / total_predictions) if total_predictions > 0 else 0.0

        # Calculate bias measures
        all_confidences = [p.confidence.level for p, _ in resolved_predictions]
        all_accuracies = [float(actual) for _, actual in resolved_predictions]

        overconfidence_bias = mean(all_confidences) - mean(all_accuracies)
        underconfidence_bias = -overconfidence_bias if overconfidence_bias < 0 else 0.0

        # Generate recommendations
        recommendations = []
        if overconfidence_bias > 0.1:
            recommendations.append("Reduce confidence levels - showing overconfidence bias")
        elif underconfidence_bias > 0.1:
            recommendations.append("Increase confidence levels - showing underconfidence bias")

        if overall_calibration < 0.8:
            recommendations.append("Implement confidence recalibration mechanisms")

        # Update calibration history
        self.calibration_history.append((datetime.utcnow(), overall_calibration))

        return CalibrationAnalysis(
            overall_calibration=overall_calibration,
            confidence_bins=bin_stats,
            overconfidence_bias=max(0, overconfidence_bias),
            underconfidence_bias=max(0, underconfidence_bias),
            calibration_trend=list(self.calibration_history),
            recommendations=recommendations
        )

    async def analyze_historical_patterns(self,
                                        historical_data: List[Tuple[Prediction, Optional[float]]],
                                        time_window_days: int = 90) -> List[PatternInsight]:
        """Build historical performance analysis for identifying successful patterns."""
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
        relevant_data = [
            (p, actual) for p, actual in historical_data
            if p.timestamp >= cutoff_date and actual is not None
        ]

        if len(relevant_data) < self.min_samples_for_learning:
            return []

        patterns = []

        # Temporal patterns
        temporal_patterns = await self._analyze_temporal_patterns(relevant_data)
        patterns.extend(temporal_patterns)

        # Category performance patterns
        category_patterns = await self._analyze_category_patterns(relevant_data)
        patterns.extend(category_patterns)

        # Confidence patterns
        confidence_patterns = await self._analyze_confidence_patterns(relevant_data)
        patterns.extend(confidence_patterns)

        # Method effectiveness patterns
        method_patterns = await self._analyze_method_patterns(relevant_data)
        patterns.extend(method_patterns)

        logger.info(f"Identified {len(patterns)} historical patterns")
        return patterns

    async def run_ab_test(self,
                         test_id: str,
                         strategy_a: StrategyType,
                         strategy_b: StrategyType,
                         test_duration_days: int = 14) -> str:
        """Add A/B testing framework for strategy optimization validation."""
        if test_id in self.active_ab_tests:
            raise ValueError(f"A/B test {test_id} is already running")

        test_config = {
            'test_id': test_id,
            'strategy_a': strategy_a,
            'strategy_b': strategy_b,
            'start_time': datetime.utcnow(),
            'end_time': datetime.utcnow() + timedelta(days=test_duration_days),
            'results_a': [],
            'results_b': [],
            'current_assignment': 'A'  # Simple alternating assignment
        }

        self.active_ab_tests[test_id] = test_config
        logger.info(f"Started A/B test {test_id}: {strategy_a.value} vs {strategy_b.value}")
        return test_id

    async def record_ab_test_result(self, test_id: str, strategy_used: StrategyType,
                                  performance_score: float) -> None:
        """Record result for ongoing A/B test."""
        if test_id not in self.active_ab_tests:
            logger.warning(f"A/B test {test_id} not found")
            return

        test_config = self.active_ab_tests[test_id]

        if strategy_used == test_config['strategy_a']:
            test_config['results_a'].append(performance_score)
        elif strategy_used == test_config['strategy_b']:
            test_config['results_b'].append(performance_score)

        # Check if test should be completed
        if datetime.utcnow() >= test_config['end_time']:
            await self._complete_ab_test(test_id)

    async def get_ab_test_assignment(self, test_id: str) -> Optional[StrategyType]:
        """Get strategy assignment for A/B test."""
        if test_id not in self.active_ab_tests:
            return None

        test_config = self.active_ab_tests[test_id]

        # Simple alternating assignment
        if test_config['current_assignment'] == 'A':
            test_config['current_assignment'] = 'B'
            return test_config['strategy_a']
        else:
            test_config['current_assignment'] = 'A'
            return test_config['strategy_b']

    # Helper methods
    def _extract_probability(self, result: Union[float, Dict[str, float], Any]) -> float:
        """Extract probability from prediction result."""
        if isinstance(result, float):
            return result
        elif hasattr(result, 'value'):
            if isinstance(result.value, float):
                return result.value
            elif isinstance(result.value, dict) and 'probability' in result.value:
                return result.value['probability']
        elif isinstance(result, dict) and 'probability' in result:
            return result['probability']

        return 0.5  # Default neutral probability

    def _calculate_calibration_score(self, predictions: List[Tuple[Prediction, float]]) -> float:
        """Calculate calibration score using reliability diagram approach."""
        if not predictions:
            return 0.0

        # Create bins and calculate calibration
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(len(bins) - 1):
            bin_predictions = [
                (p, actual) for p, actual in predictions
                if bins[i] <= p.confidence.level < bins[i + 1]
            ]

            if bin_predictions:
                avg_confidence = mean([p.confidence.level for p, _ in bin_predictions])
                avg_accuracy = mean([actual for _, actual in bin_predictions])
                bin_confidences.append(avg_confidence)
                bin_accuracies.append(avg_accuracy)
                bin_counts.append(len(bin_predictions))

        if not bin_confidences:
            return 0.0

        # Calculate Expected Calibration Error (ECE)
        total_predictions = sum(bin_counts)
        ece = sum(
            (count / total_predictions) * abs(conf - acc)
            for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts)
        )

        return max(0.0, 1.0 - ece)  # Convert to calibration score (higher is better)

    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty performance metrics."""
        return PerformanceMetrics(
            accuracy_score=0.0,
            calibration_score=0.0,
            brier_score=1.0,
            log_score=-np.inf,
            confidence_correlation=0.0,
            prediction_count=0,
            category_breakdown={},
            time_period=(datetime.utcnow(), datetime.utcnow())
        )

    def _get_current_strategy(self, tournament: Tournament) -> StrategyType:
        """Get current strategy for tournament."""
        # This would typically be retrieved from tournament state
        return StrategyType.BALANCED  # Default strategy

    def _assess_information_impact(self, new_information: Dict[str, Any]) -> float:
        """Assess the impact level of new information."""
        # Simple heuristic - could be enhanced with ML
        impact_factors = {
            'source_credibility': new_information.get('credibility', 0.5),
            'information_novelty': new_information.get('novelty', 0.5),
            'relevance_score': new_information.get('relevance', 0.5)
        }
        return mean(impact_factors.values())

    def _calculate_confidence_adjustment(self,
                                       prediction: Prediction,
                                       new_information: Dict[str, Any],
                                       impact: float) -> float:
        """Calculate confidence adjustment based on new information."""
        # Simple adjustment logic - could be enhanced
        base_adjustment = impact * 0.1  # Max 10% adjustment

        # Adjust based on information sentiment
        sentiment = new_information.get('sentiment', 'neutral')
        if sentiment == 'positive':
            return base_adjustment
        elif sentiment == 'negative':
            return -base_adjustment
        else:
            return 0.0

    def _determine_tournament_phase(self, tournament: Tournament) -> str:
        """Determine current phase of tournament."""
        now = datetime.utcnow()
        total_duration = (tournament.end_date - tournament.start_date).total_seconds()
        elapsed = (now - tournament.start_date).total_seconds()
        progress = elapsed / total_duration

        if progress < 0.3:
            return 'early'
        elif progress < 0.7:
            return 'middle'
        else:
            return 'late'

    def _calculate_competition_intensity(self, tournament: Tournament) -> float:
        """Calculate competition intensity metric."""
        # Simple heuristic based on participant count and question activity
        base_intensity = min(1.0, tournament.get_participant_count() / 100)

        # Adjust based on active questions
        active_questions = len(tournament.get_active_questions())
        question_factor = min(1.0, active_questions / 50)

        return (base_intensity + question_factor) / 2

    def _analyze_question_difficulty_trend(self, tournament: Tournament) -> str:
        """Analyze trend in question difficulty."""
        # Placeholder - would analyze historical question resolution rates
        return 'stable'

    def _analyze_participant_patterns(self, tournament: Tournament) -> Dict[str, Any]:
        """Analyze participant behavior patterns."""
        return {
            'average_predictions_per_participant': 10.0,  # Placeholder
            'prediction_timing_pattern': 'early_heavy',
            'consensus_level': 0.7
        }

    def _assess_market_efficiency(self, tournament: Tournament) -> float:
        """Assess market efficiency level."""
        # Placeholder - would analyze prediction convergence and accuracy
        return 0.75

    async def _update_agent_profile(self, agent_id: str, metrics: PerformanceMetrics) -> None:
        """Update agent performance profile."""
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentPerformanceProfile(
                agent_id=agent_id,
                overall_accuracy=metrics.accuracy_score,
                category_specializations=metrics.category_breakdown,
                confidence_calibration=metrics.calibration_score,
                prediction_speed=1.0,  # Placeholder
                consistency_score=0.8,  # Placeholder
                recent_trend=0.0,
                optimal_question_types=[],
                weight_recommendation=1.0,
                last_updated=datetime.utcnow()
            )
        else:
            profile = self.agent_profiles[agent_id]
            # Update with exponential moving average
            alpha = 0.3
            profile.overall_accuracy = alpha * metrics.accuracy_score + (1 - alpha) * profile.overall_accuracy
            profile.confidence_calibration = alpha * metrics.calibration_score + (1 - alpha) * profile.confidence_calibration
            profile.last_updated = datetime.utcnow()

    def _create_empty_calibration_analysis(self) -> CalibrationAnalysis:
        """Create empty calibration analysis."""
        return CalibrationAnalysis(
            overall_calibration=0.0,
            confidence_bins={},
            overconfidence_bias=0.0,
            underconfidence_bias=0.0,
            calibration_trend=[],
            recommendations=["Insufficient data for calibration analysis"]
        )

    def _get_confidence_bin(self, confidence: float) -> str:
        """Get confidence bin for given confidence level."""
        if confidence < 0.2:
            return '0.0-0.2'
        elif confidence < 0.4:
            return '0.2-0.4'
        elif confidence < 0.6:
            return '0.4-0.6'
        elif confidence < 0.8:
            return '0.6-0.8'
        else:
            return '0.8-1.0'

    async def _analyze_temporal_patterns(self, data: List[Tuple[Prediction, float]]) -> List[PatternInsight]:
        """Analyze temporal patterns in predictions."""
        patterns = []

        # Group by hour of day
        hourly_performance = defaultdict(list)
        for prediction, actual in data:
            hour = prediction.timestamp.hour
            accuracy = 1.0 if (self._extract_probability(prediction.result) > 0.5) == (actual > 0.5) else 0.0
            hourly_performance[hour].append(accuracy)

        # Find best and worst performing hours
        hourly_averages = {hour: mean(accuracies) for hour, accuracies in hourly_performance.items() if len(accuracies) >= 3}

        if hourly_averages:
            best_hour = max(hourly_averages.keys(), key=lambda h: hourly_averages[h])
            worst_hour = min(hourly_averages.keys(), key=lambda h: hourly_averages[h])

            if hourly_averages[best_hour] - hourly_averages[worst_hour] > 0.1:
                patterns.append(PatternInsight(
                    pattern_type=PatternType.TIMING_PATTERNS,
                    description=f"Performance varies by time: best at {best_hour}:00 ({hourly_averages[best_hour]:.2%}), worst at {worst_hour}:00 ({hourly_averages[worst_hour]:.2%})",
                    confidence=0.7,
                    impact_score=0.4,
                    supporting_evidence=[f"Hour {best_hour}: {hourly_averages[best_hour]:.2%}", f"Hour {worst_hour}: {hourly_averages[worst_hour]:.2%}"],
                    recommendations=[f"Schedule more predictions around {best_hour}:00", f"Review prediction quality at {worst_hour}:00"],
                    discovered_at=datetime.utcnow()
                ))

        return patterns

    async def _analyze_category_patterns(self, data: List[Tuple[Prediction, float]]) -> List[PatternInsight]:
        """Analyze category-specific patterns."""
        patterns = []

        # This would require category information in predictions
        # Placeholder implementation
        return patterns

    async def _analyze_confidence_patterns(self, data: List[Tuple[Prediction, float]]) -> List[PatternInsight]:
        """Analyze confidence-related patterns."""
        patterns = []

        # Analyze confidence vs accuracy relationship
        confidences = [p.confidence.level for p, _ in data]
        accuracies = [1.0 if (self._extract_probability(p.result) > 0.5) == (actual > 0.5) else 0.0 for p, actual in data]

        if len(confidences) > 10:
            correlation = np.corrcoef(confidences, accuracies)[0, 1]

            if abs(correlation) < 0.3:
                patterns.append(PatternInsight(
                    pattern_type=PatternType.CONFIDENCE_CALIBRATION,
                    description=f"Weak confidence-accuracy correlation: {correlation:.3f}",
                    confidence=0.8,
                    impact_score=0.6,
                    supporting_evidence=[f"Correlation: {correlation:.3f}", f"Sample size: {len(confidences)}"],
                    recommendations=["Recalibrate confidence assignment", "Review confidence calculation methods"],
                    discovered_at=datetime.utcnow()
                ))

        return patterns

    async def _analyze_method_patterns(self, data: List[Tuple[Prediction, float]]) -> List[PatternInsight]:
        """Analyze method effectiveness patterns."""
        patterns = []

        # Group by prediction method
        method_performance = defaultdict(list)
        for prediction, actual in data:
            accuracy = 1.0 if (self._extract_probability(prediction.result) > 0.5) == (actual > 0.5) else 0.0
            method_performance[prediction.method].append(accuracy)

        # Find best and worst methods
        method_averages = {method: mean(accuracies) for method, accuracies in method_performance.items() if len(accuracies) >= 5}

        if len(method_averages) > 1:
            best_method = max(method_averages.keys(), key=lambda m: method_averages[m])
            worst_method = min(method_averages.keys(), key=lambda m: method_averages[m])

            if method_averages[best_method] - method_averages[worst_method] > 0.15:
                patterns.append(PatternInsight(
                    pattern_type=PatternType.STRATEGY_EFFECTIVENESS,
                    description=f"Method effectiveness varies: {best_method} ({method_averages[best_method]:.2%}) vs {worst_method} ({method_averages[worst_method]:.2%})",
                    confidence=0.85,
                    impact_score=0.7,
                    supporting_evidence=[f"{best_method}: {method_averages[best_method]:.2%}", f"{worst_method}: {method_averages[worst_method]:.2%}"],
                    recommendations=[f"Prioritize {best_method} method", f"Investigate issues with {worst_method} method"],
                    discovered_at=datetime.utcnow()
                ))

        return patterns

    async def _complete_ab_test(self, test_id: str) -> ABTestResult:
        """Complete A/B test and calculate results."""
        test_config = self.active_ab_tests[test_id]

        results_a = test_config['results_a']
        results_b = test_config['results_b']

        if not results_a or not results_b:
            logger.warning(f"A/B test {test_id} completed with insufficient data")
            return None

        # Calculate statistics
        mean_a = mean(results_a)
        mean_b = mean(results_b)

        # Simple statistical significance test (t-test approximation)
        if len(results_a) > 1 and len(results_b) > 1:
            std_a = stdev(results_a)
            std_b = stdev(results_b)
            pooled_std = np.sqrt((std_a**2 / len(results_a)) + (std_b**2 / len(results_b)))
            t_stat = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
            # Simplified p-value approximation
            p_value = max(0.01, 1.0 / (1.0 + t_stat))
            significance = 1.0 - p_value
        else:
            significance = 0.5

        # Determine winner
        winner = None
        if significance > 0.95:  # 95% confidence
            winner = test_config['strategy_a'] if mean_a > mean_b else test_config['strategy_b']

        # Calculate confidence interval for difference
        diff = mean_a - mean_b
        margin_of_error = 1.96 * pooled_std if 'pooled_std' in locals() else abs(diff) * 0.1
        confidence_interval = (diff - margin_of_error, diff + margin_of_error)

        result = ABTestResult(
            test_id=test_id,
            strategy_a=test_config['strategy_a'],
            strategy_b=test_config['strategy_b'],
            sample_size_a=len(results_a),
            sample_size_b=len(results_b),
            performance_a=mean_a,
            performance_b=mean_b,
            statistical_significance=significance,
            winner=winner,
            confidence_interval=confidence_interval,
            test_duration=test_config['end_time'] - test_config['start_time'],
            completed_at=datetime.utcnow()
        )

        # Move to completed tests
        self.completed_ab_tests.append(result)
        del self.active_ab_tests[test_id]

        logger.info(f"A/B test {test_id} completed: {result.winner.value if result.winner else 'No clear winner'}")
        return result
