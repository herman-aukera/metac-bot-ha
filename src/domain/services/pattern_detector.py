"""Pattern detector for tournament adaptation and competitive intelligence."""

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import structlog

from ..entities.forecast import Forecast, ForecastStatus
from ..entities.prediction import Prediction, PredictionConfidence, PredictionMethod
from ..entities.question import Question, QuestionType
from ..value_objects.tournament_strategy import TournamentStrategy

logger = structlog.get_logger(__name__)


class PatternType(Enum):
    """Types of patterns that can be detected."""

    QUESTION_TYPE_PERFORMANCE = "question_type_performance"
    TEMPORAL_PERFORMANCE = "temporal_performance"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    METHOD_EFFECTIVENESS = "method_effectiveness"
    TOURNAMENT_DYNAMICS = "tournament_dynamics"
    COMPETITIVE_POSITIONING = "competitive_positioning"
    MARKET_INEFFICIENCY = "market_inefficiency"
    SEASONAL_TRENDS = "seasonal_trends"
    COMPLEXITY_CORRELATION = "complexity_correlation"
    ENSEMBLE_SYNERGY = "ensemble_synergy"


class AdaptationStrategy(Enum):
    """Types of adaptation strategies."""

    INCREASE_CONFIDENCE = "increase_confidence"
    DECREASE_CONFIDENCE = "decrease_confidence"
    CHANGE_METHOD_PREFERENCE = "change_method_preference"
    ADJUST_ENSEMBLE_WEIGHTS = "adjust_ensemble_weights"
    MODIFY_RESEARCH_DEPTH = "modify_research_depth"
    ALTER_SUBMISSION_TIMING = "alter_submission_timing"
    FOCUS_QUESTION_TYPES = "focus_question_types"
    EXPLOIT_MARKET_GAP = "exploit_market_gap"
    INCREASE_CONSERVATISM = "increase_conservatism"
    INCREASE_AGGRESSIVENESS = "increase_aggressiveness"


@dataclass
class DetectedPattern:
    """A detected pattern in forecasting performance or tournament dynamics."""

    pattern_type: PatternType
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    strength: float  # How strong the pattern is
    frequency: float  # How often it occurs
    context: Dict[str, Any]  # Context-specific data
    affected_questions: List[UUID]
    affected_agents: List[str]
    first_observed: datetime
    last_observed: datetime
    trend_direction: str  # "improving", "declining", "stable"
    statistical_significance: float
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationRecommendation:
    """Recommendation for strategy adaptation based on detected patterns."""

    strategy_type: AdaptationStrategy
    title: str
    description: str
    rationale: str
    expected_impact: float  # Expected improvement
    confidence: float  # Confidence in recommendation
    priority: float  # Implementation priority
    implementation_complexity: float  # 0.0 to 1.0
    affected_contexts: List[str]
    specific_actions: List[str]
    success_metrics: List[str]
    timeline: str  # "immediate", "short_term", "long_term"
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompetitiveIntelligence:
    """Competitive intelligence derived from tournament analysis."""

    tournament_id: str
    market_gaps: List[Dict[str, Any]]
    competitor_weaknesses: List[Dict[str, Any]]
    optimal_positioning: Dict[str, Any]
    timing_opportunities: List[Dict[str, Any]]
    question_type_advantages: Dict[str, float]
    confidence_level_opportunities: Dict[str, float]
    meta_game_insights: List[str]
    strategic_recommendations: List[str]
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternDetector:
    """
    Service for detecting patterns in forecasting performance and tournament dynamics.

    Provides question type pattern recognition, tournament dynamics detection,
    competitive intelligence, and meta-pattern identification for strategy evolution.
    """

    def __init__(self):
        self.detected_patterns: List[DetectedPattern] = []
        self.adaptation_recommendations: List[AdaptationRecommendation] = []
        self.competitive_intelligence: List[CompetitiveIntelligence] = []

        # Configuration
        self.min_samples_for_pattern = 5
        self.pattern_confidence_threshold = 0.6
        self.statistical_significance_threshold = 0.05
        self.pattern_detection_window_days = 30

        # Pattern detection methods
        self.pattern_detectors = {
            PatternType.QUESTION_TYPE_PERFORMANCE: self._detect_question_type_patterns,
            PatternType.TEMPORAL_PERFORMANCE: self._detect_temporal_patterns,
            PatternType.CONFIDENCE_CALIBRATION: self._detect_calibration_patterns,
            PatternType.METHOD_EFFECTIVENESS: self._detect_method_patterns,
            PatternType.TOURNAMENT_DYNAMICS: self._detect_tournament_patterns,
            PatternType.COMPETITIVE_POSITIONING: self._detect_competitive_patterns,
            PatternType.MARKET_INEFFICIENCY: self._detect_market_inefficiencies,
            PatternType.SEASONAL_TRENDS: self._detect_seasonal_patterns,
            PatternType.COMPLEXITY_CORRELATION: self._detect_complexity_patterns,
            PatternType.ENSEMBLE_SYNERGY: self._detect_ensemble_patterns,
        }

    def detect_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]] = None,
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect patterns in forecasting data for tournament adaptation.

        Args:
            forecasts: List of forecasts to analyze
            questions: List of questions for context
            ground_truth: Optional ground truth for resolved questions
            tournament_context: Optional tournament-specific context

        Returns:
            Comprehensive pattern analysis results
        """
        if not forecasts:
            return {"message": "No forecasts provided for pattern detection"}

        logger.info(
            "Detecting patterns",
            forecast_count=len(forecasts),
            question_count=len(questions),
            has_ground_truth=ground_truth is not None,
            analysis_timestamp=datetime.utcnow(),
        )

        # Detect patterns using all available detectors
        all_patterns = []
        for pattern_type, detector_func in self.pattern_detectors.items():
            try:
                patterns = detector_func(
                    forecasts, questions, ground_truth, tournament_context
                )
                all_patterns.extend(patterns)
            except Exception as e:
                logger.warning(
                    "Pattern detection failed",
                    pattern_type=pattern_type.value,
                    error=str(e),
                )

        # Filter patterns by confidence and significance
        significant_patterns = [
            pattern
            for pattern in all_patterns
            if pattern.confidence >= self.pattern_confidence_threshold
            and pattern.statistical_significance
            <= self.statistical_significance_threshold
        ]

        # Generate adaptation recommendations
        recommendations = self._generate_adaptation_recommendations(
            significant_patterns, tournament_context
        )

        # Generate competitive intelligence
        competitive_intel = self._generate_competitive_intelligence(
            significant_patterns, forecasts, questions, tournament_context
        )

        # Store patterns for historical tracking
        self.detected_patterns.extend(significant_patterns)
        self.adaptation_recommendations.extend(recommendations)
        if competitive_intel:
            self.competitive_intelligence.append(competitive_intel)

        # Analyze meta-patterns
        meta_patterns = self._detect_meta_patterns(significant_patterns)

        results = {
            "analysis_timestamp": datetime.utcnow(),
            "total_patterns_detected": len(all_patterns),
            "significant_patterns": len(significant_patterns),
            "patterns_by_type": self._group_patterns_by_type(significant_patterns),
            "detected_patterns": [
                self._serialize_pattern(p) for p in significant_patterns
            ],
            "adaptation_recommendations": [
                self._serialize_recommendation(r) for r in recommendations
            ],
            "competitive_intelligence": (
                self._serialize_competitive_intelligence(competitive_intel)
                if competitive_intel
                else None
            ),
            "meta_patterns": meta_patterns,
            "strategy_evolution_suggestions": self._generate_strategy_evolution_suggestions(
                significant_patterns
            ),
        }

        logger.info(
            "Pattern detection completed",
            significant_patterns=len(significant_patterns),
            recommendations=len(recommendations),
            has_competitive_intel=competitive_intel is not None,
        )

        return results

    def _detect_question_type_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect patterns related to question type performance."""
        patterns = []

        # Group forecasts by question type
        question_type_map = {q.id: q.question_type for q in questions}
        type_performance = defaultdict(list)

        for i, forecast in enumerate(forecasts):
            question_type = question_type_map.get(forecast.question_id)
            if question_type and ground_truth and i < len(ground_truth):
                accuracy = (
                    1.0 if (forecast.prediction > 0.5) == ground_truth[i] else 0.0
                )
                type_performance[question_type].append(
                    {
                        "forecast": forecast,
                        "accuracy": accuracy,
                        "confidence": forecast.confidence,
                        "brier_score": (
                            forecast.prediction - (1.0 if ground_truth[i] else 0.0)
                        )
                        ** 2,
                    }
                )

        # Analyze performance by question type
        for question_type, performance_data in type_performance.items():
            if len(performance_data) < self.min_samples_for_pattern:
                continue

            accuracies = [p["accuracy"] for p in performance_data]
            brier_scores = [p["brier_score"] for p in performance_data]

            avg_accuracy = statistics.mean(accuracies)
            avg_brier = statistics.mean(brier_scores)

            # Compare to overall performance
            overall_accuracy = (
                statistics.mean(
                    [
                        1.0 if (f.prediction > 0.5) == truth else 0.0
                        for f, truth in zip(forecasts, ground_truth or [])
                        if truth is not None
                    ]
                )
                if ground_truth
                else 0.5
            )

            performance_diff = avg_accuracy - overall_accuracy

            if abs(performance_diff) > 0.1:  # Significant difference
                trend = "improving" if performance_diff > 0 else "declining"

                pattern = DetectedPattern(
                    pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
                    title=f"Question Type Performance: {question_type.value}",
                    description=f"Performance on {question_type.value} questions is {performance_diff:+.2f} compared to overall average",
                    confidence=min(0.9, 0.5 + abs(performance_diff)),
                    strength=abs(performance_diff),
                    frequency=len(performance_data) / len(forecasts),
                    context={
                        "question_type": question_type.value,
                        "sample_size": len(performance_data),
                        "avg_accuracy": avg_accuracy,
                        "avg_brier_score": avg_brier,
                        "performance_difference": performance_diff,
                    },
                    affected_questions=[
                        p["forecast"].question_id for p in performance_data
                    ],
                    affected_agents=list(
                        set(
                            p["forecast"].final_prediction.created_by
                            for p in performance_data
                        )
                    ),
                    first_observed=min(
                        p["forecast"].created_at for p in performance_data
                    ),
                    last_observed=max(
                        p["forecast"].created_at for p in performance_data
                    ),
                    trend_direction=trend,
                    statistical_significance=(
                        0.01 if abs(performance_diff) > 0.15 else 0.05
                    ),
                    examples=[
                        {
                            "question_id": str(p["forecast"].question_id),
                            "accuracy": p["accuracy"],
                            "confidence": p["confidence"],
                        }
                        for p in performance_data[:3]
                    ],
                )
                patterns.append(pattern)

        return patterns

    def _detect_temporal_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect temporal patterns in performance."""
        patterns = []

        if not ground_truth or len(forecasts) < 10:
            return patterns

        # Sort forecasts by creation time
        sorted_forecasts = sorted(
            [
                (f, truth)
                for f, truth in zip(forecasts, ground_truth)
                if truth is not None
            ],
            key=lambda x: x[0].created_at,
        )

        if len(sorted_forecasts) < 10:
            return patterns

        # Analyze performance over time using sliding windows
        window_size = max(5, len(sorted_forecasts) // 4)
        windows = []

        for i in range(0, len(sorted_forecasts) - window_size + 1, window_size // 2):
            window_data = sorted_forecasts[i : i + window_size]
            window_accuracy = statistics.mean(
                [
                    1.0 if (f.prediction > 0.5) == truth else 0.0
                    for f, truth in window_data
                ]
            )
            window_confidence = statistics.mean([f.confidence for f, _ in window_data])
            window_time = statistics.mean(
                [f.created_at.timestamp() for f, _ in window_data]
            )

            windows.append(
                {
                    "time": datetime.fromtimestamp(window_time),
                    "accuracy": window_accuracy,
                    "confidence": window_confidence,
                    "sample_size": len(window_data),
                }
            )

        if len(windows) < 3:
            return patterns

        # Detect trends
        accuracies = [w["accuracy"] for w in windows]
        times = list(range(len(windows)))

        # Simple linear trend detection
        if len(accuracies) > 2:
            # Calculate correlation between time and accuracy
            mean_time = statistics.mean(times)
            mean_acc = statistics.mean(accuracies)

            numerator = sum(
                (t - mean_time) * (a - mean_acc) for t, a in zip(times, accuracies)
            )
            denominator = math.sqrt(
                sum((t - mean_time) ** 2 for t in times)
                * sum((a - mean_acc) ** 2 for a in accuracies)
            )

            if denominator > 0:
                correlation = numerator / denominator

                if abs(correlation) > 0.5:  # Significant trend
                    trend_direction = "improving" if correlation > 0 else "declining"

                    pattern = DetectedPattern(
                        pattern_type=PatternType.TEMPORAL_PERFORMANCE,
                        title=f"Temporal Performance Trend: {trend_direction.title()}",
                        description=f"Performance is {trend_direction} over time with correlation {correlation:.2f}",
                        confidence=min(0.9, abs(correlation)),
                        strength=abs(correlation),
                        frequency=1.0,  # Temporal patterns affect all forecasts
                        context={
                            "correlation": correlation,
                            "trend_strength": abs(correlation),
                            "window_count": len(windows),
                            "time_span_days": (
                                windows[-1]["time"] - windows[0]["time"]
                            ).days,
                            "accuracy_range": [min(accuracies), max(accuracies)],
                        },
                        affected_questions=[f.question_id for f in forecasts],
                        affected_agents=list(
                            set(f.final_prediction.created_by for f in forecasts)
                        ),
                        first_observed=windows[0]["time"],
                        last_observed=windows[-1]["time"],
                        trend_direction=trend_direction,
                        statistical_significance=(
                            0.01 if abs(correlation) > 0.7 else 0.05
                        ),
                        examples=[
                            {
                                "time_window": w["time"].isoformat(),
                                "accuracy": w["accuracy"],
                                "confidence": w["confidence"],
                            }
                            for w in windows[:3]
                        ],
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_calibration_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect confidence calibration patterns."""
        patterns = []

        if not ground_truth:
            return patterns

        # Group forecasts by confidence level
        confidence_bins = {"low": [], "medium": [], "high": []}

        for forecast, truth in zip(forecasts, ground_truth):
            if truth is None:
                continue

            accuracy = 1.0 if (forecast.prediction > 0.5) == truth else 0.0

            if forecast.confidence < 0.4:
                confidence_bins["low"].append(accuracy)
            elif forecast.confidence < 0.7:
                confidence_bins["medium"].append(accuracy)
            else:
                confidence_bins["high"].append(accuracy)

        # Analyze calibration for each confidence level
        for conf_level, accuracies in confidence_bins.items():
            if len(accuracies) < self.min_samples_for_pattern:
                continue

            avg_accuracy = statistics.mean(accuracies)
            expected_accuracy = (
                0.3 if conf_level == "low" else 0.6 if conf_level == "medium" else 0.8
            )

            calibration_error = abs(avg_accuracy - expected_accuracy)

            if calibration_error > 0.15:  # Significant miscalibration
                miscalibration_type = (
                    "overconfident"
                    if avg_accuracy < expected_accuracy
                    else "underconfident"
                )

                pattern = DetectedPattern(
                    pattern_type=PatternType.CONFIDENCE_CALIBRATION,
                    title=f"Calibration Issue: {miscalibration_type.title()} at {conf_level.title()} Confidence",
                    description=f"{conf_level.title()} confidence predictions show {miscalibration_type} pattern with {calibration_error:.2f} calibration error",
                    confidence=min(0.9, calibration_error * 2),
                    strength=calibration_error,
                    frequency=len(accuracies) / len(forecasts),
                    context={
                        "confidence_level": conf_level,
                        "sample_size": len(accuracies),
                        "actual_accuracy": avg_accuracy,
                        "expected_accuracy": expected_accuracy,
                        "calibration_error": calibration_error,
                        "miscalibration_type": miscalibration_type,
                    },
                    affected_questions=[],  # Would need to track which forecasts
                    affected_agents=list(
                        set(f.final_prediction.created_by for f in forecasts)
                    ),
                    first_observed=min(f.created_at for f in forecasts),
                    last_observed=max(f.created_at for f in forecasts),
                    trend_direction="stable",  # Would need temporal analysis
                    statistical_significance=0.01 if calibration_error > 0.25 else 0.05,
                )
                patterns.append(pattern)

        return patterns

    def _detect_method_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect patterns in method effectiveness."""
        patterns = []

        if not ground_truth:
            return patterns

        # Group forecasts by method
        method_performance = defaultdict(list)

        for forecast, truth in zip(forecasts, ground_truth):
            if truth is None:
                continue

            method = forecast.method
            accuracy = 1.0 if (forecast.prediction > 0.5) == truth else 0.0
            brier_score = (forecast.prediction - (1.0 if truth else 0.0)) ** 2

            method_performance[method].append(
                {
                    "accuracy": accuracy,
                    "brier_score": brier_score,
                    "confidence": forecast.confidence,
                    "forecast": forecast,
                }
            )

        # Analyze method performance
        method_stats = {}
        for method, performance_data in method_performance.items():
            if len(performance_data) < self.min_samples_for_pattern:
                continue

            method_stats[method] = {
                "accuracy": statistics.mean([p["accuracy"] for p in performance_data]),
                "brier_score": statistics.mean(
                    [p["brier_score"] for p in performance_data]
                ),
                "confidence": statistics.mean(
                    [p["confidence"] for p in performance_data]
                ),
                "sample_size": len(performance_data),
                "data": performance_data,
            }

        if len(method_stats) < 2:
            return patterns

        # Find best and worst performing methods
        best_method = max(
            method_stats.keys(), key=lambda m: method_stats[m]["accuracy"]
        )
        worst_method = min(
            method_stats.keys(), key=lambda m: method_stats[m]["accuracy"]
        )

        performance_gap = (
            method_stats[best_method]["accuracy"]
            - method_stats[worst_method]["accuracy"]
        )

        if performance_gap > 0.15:  # Significant difference
            pattern = DetectedPattern(
                pattern_type=PatternType.METHOD_EFFECTIVENESS,
                title=f"Method Performance Gap: {best_method} vs {worst_method}",
                description=f"Significant performance difference between {best_method} ({method_stats[best_method]['accuracy']:.2f}) and {worst_method} ({method_stats[worst_method]['accuracy']:.2f})",
                confidence=min(0.9, performance_gap * 2),
                strength=performance_gap,
                frequency=1.0,  # Method patterns affect all forecasts
                context={
                    "best_method": best_method,
                    "worst_method": worst_method,
                    "performance_gap": performance_gap,
                    "method_stats": {
                        method: {
                            "accuracy": stats["accuracy"],
                            "brier_score": stats["brier_score"],
                            "sample_size": stats["sample_size"],
                        }
                        for method, stats in method_stats.items()
                    },
                },
                affected_questions=[f.question_id for f in forecasts],
                affected_agents=list(
                    set(f.final_prediction.created_by for f in forecasts)
                ),
                first_observed=min(f.created_at for f in forecasts),
                last_observed=max(f.created_at for f in forecasts),
                trend_direction="stable",  # Would need temporal analysis
                statistical_significance=0.01 if performance_gap > 0.25 else 0.05,
                examples=[
                    {
                        "method": method,
                        "accuracy": stats["accuracy"],
                        "sample_size": stats["sample_size"],
                    }
                    for method, stats in list(method_stats.items())[:3]
                ],
            )
            patterns.append(pattern)

        return patterns

    def _detect_tournament_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect tournament-specific dynamics patterns."""
        patterns = []

        if not tournament_context:
            return patterns

        # Analyze submission timing patterns
        if "deadlines" in tournament_context:
            timing_pattern = self._analyze_submission_timing(
                forecasts, tournament_context["deadlines"]
            )
            if timing_pattern:
                patterns.append(timing_pattern)

        # Analyze competitive pressure patterns
        if "competitor_data" in tournament_context:
            competitive_pattern = self._analyze_competitive_pressure(
                forecasts, tournament_context["competitor_data"]
            )
            if competitive_pattern:
                patterns.append(competitive_pattern)

        # Analyze question difficulty patterns
        difficulty_pattern = self._analyze_question_difficulty(
            forecasts, questions, ground_truth
        )
        if difficulty_pattern:
            patterns.append(difficulty_pattern)

        return patterns

    def _detect_competitive_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect competitive positioning patterns."""
        patterns = []

        # This would require competitor data which isn't available in the current setup
        # In a real tournament, this would analyze:
        # - Market consensus vs our predictions
        # - Competitor prediction patterns
        # - Market inefficiencies
        # - Optimal differentiation strategies

        return patterns

    def _detect_market_inefficiencies(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect market inefficiencies for exploitation."""
        patterns = []

        # This would require market data which isn't available in the current setup
        # In a real tournament, this would analyze:
        # - Prediction market prices vs our forecasts
        # - Crowd wisdom vs expert predictions
        # - Systematic biases in market predictions
        # - Arbitrage opportunities

        return patterns

    def _detect_seasonal_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect seasonal or cyclical patterns."""
        patterns = []

        if len(forecasts) < 20:  # Need sufficient data for seasonal analysis
            return patterns

        # Group forecasts by time periods
        monthly_performance = defaultdict(list)
        weekly_performance = defaultdict(list)

        for i, forecast in enumerate(forecasts):
            if ground_truth and i < len(ground_truth) and ground_truth[i] is not None:
                accuracy = (
                    1.0 if (forecast.prediction > 0.5) == ground_truth[i] else 0.0
                )

                month = forecast.created_at.month
                weekday = forecast.created_at.weekday()

                monthly_performance[month].append(accuracy)
                weekly_performance[weekday].append(accuracy)

        # Analyze monthly patterns
        if len(monthly_performance) >= 3:
            month_accuracies = {
                month: statistics.mean(accuracies)
                for month, accuracies in monthly_performance.items()
                if len(accuracies) >= 3
            }

            if len(month_accuracies) >= 3:
                accuracy_values = list(month_accuracies.values())
                if (
                    max(accuracy_values) - min(accuracy_values) > 0.2
                ):  # Significant seasonal variation
                    best_month = max(
                        month_accuracies.keys(), key=lambda m: month_accuracies[m]
                    )
                    worst_month = min(
                        month_accuracies.keys(), key=lambda m: month_accuracies[m]
                    )

                    pattern = DetectedPattern(
                        pattern_type=PatternType.SEASONAL_TRENDS,
                        title="Monthly Performance Variation",
                        description=f"Performance varies significantly by month: best in month {best_month} ({month_accuracies[best_month]:.2f}), worst in month {worst_month} ({month_accuracies[worst_month]:.2f})",
                        confidence=0.7,
                        strength=max(accuracy_values) - min(accuracy_values),
                        frequency=1.0,
                        context={
                            "monthly_accuracies": month_accuracies,
                            "best_month": best_month,
                            "worst_month": worst_month,
                            "variation": max(accuracy_values) - min(accuracy_values),
                        },
                        affected_questions=[f.question_id for f in forecasts],
                        affected_agents=list(
                            set(f.final_prediction.created_by for f in forecasts)
                        ),
                        first_observed=min(f.created_at for f in forecasts),
                        last_observed=max(f.created_at for f in forecasts),
                        trend_direction="cyclical",
                        statistical_significance=0.05,
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_complexity_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect patterns related to question complexity."""
        patterns = []

        # This would require question complexity metrics
        # In a real implementation, this would analyze:
        # - Performance vs question complexity
        # - Confidence calibration vs complexity
        # - Method effectiveness vs complexity
        # - Research depth requirements vs complexity

        return patterns

    def _detect_ensemble_patterns(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect ensemble synergy patterns."""
        patterns = []

        # Analyze ensemble vs individual agent performance
        ensemble_forecasts = [f for f in forecasts if f.method == "ensemble"]
        individual_forecasts = [f for f in forecasts if f.method != "ensemble"]

        if (
            len(ensemble_forecasts) < 5
            or len(individual_forecasts) < 5
            or not ground_truth
        ):
            return patterns

        # Calculate performance metrics
        ensemble_accuracy = statistics.mean(
            [
                1.0 if (f.prediction > 0.5) == ground_truth[i] else 0.0
                for i, f in enumerate(forecasts)
                if f.method == "ensemble"
                and i < len(ground_truth)
                and ground_truth[i] is not None
            ]
        )

        individual_accuracy = statistics.mean(
            [
                1.0 if (f.prediction > 0.5) == ground_truth[i] else 0.0
                for i, f in enumerate(forecasts)
                if f.method != "ensemble"
                and i < len(ground_truth)
                and ground_truth[i] is not None
            ]
        )

        ensemble_advantage = ensemble_accuracy - individual_accuracy

        if abs(ensemble_advantage) > 0.1:  # Significant difference
            advantage_type = "positive" if ensemble_advantage > 0 else "negative"

            pattern = DetectedPattern(
                pattern_type=PatternType.ENSEMBLE_SYNERGY,
                title=f"Ensemble {advantage_type.title()} Synergy",
                description=f"Ensemble methods show {advantage_type} synergy with {ensemble_advantage:+.2f} accuracy difference vs individual methods",
                confidence=min(0.9, abs(ensemble_advantage) * 2),
                strength=abs(ensemble_advantage),
                frequency=len(ensemble_forecasts) / len(forecasts),
                context={
                    "ensemble_accuracy": ensemble_accuracy,
                    "individual_accuracy": individual_accuracy,
                    "ensemble_advantage": ensemble_advantage,
                    "ensemble_count": len(ensemble_forecasts),
                    "individual_count": len(individual_forecasts),
                },
                affected_questions=[f.question_id for f in ensemble_forecasts],
                affected_agents=list(
                    set(f.final_prediction.created_by for f in ensemble_forecasts)
                ),
                first_observed=min(f.created_at for f in ensemble_forecasts),
                last_observed=max(f.created_at for f in ensemble_forecasts),
                trend_direction="stable",
                statistical_significance=(
                    0.01 if abs(ensemble_advantage) > 0.2 else 0.05
                ),
            )
            patterns.append(pattern)

        return patterns

    def _analyze_submission_timing(
        self, forecasts: List[Forecast], deadlines: Dict[str, datetime]
    ) -> Optional[DetectedPattern]:
        """Analyze submission timing patterns."""
        # This would analyze optimal submission timing
        # For now, return None as we don't have deadline data structure
        return None

    def _analyze_competitive_pressure(
        self, forecasts: List[Forecast], competitor_data: Dict[str, Any]
    ) -> Optional[DetectedPattern]:
        """Analyze competitive pressure effects."""
        # This would analyze how competitive pressure affects performance
        # For now, return None as we don't have competitor data structure
        return None

    def _analyze_question_difficulty(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        ground_truth: Optional[List[bool]],
    ) -> Optional[DetectedPattern]:
        """Analyze question difficulty patterns."""
        # This would require question difficulty metrics
        # For now, return None as we don't have difficulty scoring
        return None

    def _generate_adaptation_recommendations(
        self,
        patterns: List[DetectedPattern],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[AdaptationRecommendation]:
        """Generate adaptation recommendations based on detected patterns."""
        recommendations = []

        for pattern in patterns:
            if pattern.pattern_type == PatternType.QUESTION_TYPE_PERFORMANCE:
                rec = self._recommend_question_type_adaptation(pattern)
                if rec:
                    recommendations.append(rec)

            elif pattern.pattern_type == PatternType.CONFIDENCE_CALIBRATION:
                rec = self._recommend_calibration_adaptation(pattern)
                if rec:
                    recommendations.append(rec)

            elif pattern.pattern_type == PatternType.METHOD_EFFECTIVENESS:
                rec = self._recommend_method_adaptation(pattern)
                if rec:
                    recommendations.append(rec)

            elif pattern.pattern_type == PatternType.TEMPORAL_PERFORMANCE:
                rec = self._recommend_temporal_adaptation(pattern)
                if rec:
                    recommendations.append(rec)

            elif pattern.pattern_type == PatternType.ENSEMBLE_SYNERGY:
                rec = self._recommend_ensemble_adaptation(pattern)
                if rec:
                    recommendations.append(rec)

        return recommendations

    def _recommend_question_type_adaptation(
        self, pattern: DetectedPattern
    ) -> Optional[AdaptationRecommendation]:
        """Recommend adaptations for question type patterns."""
        context = pattern.context
        performance_diff = context.get("performance_difference", 0)
        question_type = context.get("question_type", "unknown")

        if performance_diff > 0.1:  # Strong performance
            return AdaptationRecommendation(
                strategy_type=AdaptationStrategy.FOCUS_QUESTION_TYPES,
                title=f"Focus on {question_type} Questions",
                description=f"Prioritize {question_type} questions due to strong performance advantage",
                rationale=f"Performance is {performance_diff:+.2f} above average on {question_type} questions",
                expected_impact=performance_diff * 0.5,
                confidence=pattern.confidence,
                priority=0.8,
                implementation_complexity=0.3,
                affected_contexts=[question_type],
                specific_actions=[
                    f"Increase resource allocation for {question_type} questions",
                    f"Develop specialized strategies for {question_type}",
                    f"Train agents specifically on {question_type} patterns",
                ],
                success_metrics=[
                    f"Increased accuracy on {question_type} questions",
                    "Higher tournament ranking in relevant categories",
                ],
                timeline="short_term",
            )

        elif performance_diff < -0.1:  # Poor performance
            return AdaptationRecommendation(
                strategy_type=AdaptationStrategy.MODIFY_RESEARCH_DEPTH,
                title=f"Improve {question_type} Performance",
                description=f"Address underperformance on {question_type} questions",
                rationale=f"Performance is {performance_diff:+.2f} below average on {question_type} questions",
                expected_impact=abs(performance_diff) * 0.3,
                confidence=pattern.confidence,
                priority=0.7,
                implementation_complexity=0.6,
                affected_contexts=[question_type],
                specific_actions=[
                    f"Increase research depth for {question_type} questions",
                    f"Develop specialized methodologies for {question_type}",
                    f"Consider abstaining from low-confidence {question_type} questions",
                ],
                success_metrics=[
                    f"Reduced performance gap on {question_type} questions",
                    "Improved overall tournament performance",
                ],
                timeline="medium_term",
            )

        return None

    def _recommend_calibration_adaptation(
        self, pattern: DetectedPattern
    ) -> Optional[AdaptationRecommendation]:
        """Recommend adaptations for calibration patterns."""
        context = pattern.context
        miscalibration_type = context.get("miscalibration_type", "unknown")
        confidence_level = context.get("confidence_level", "unknown")

        if miscalibration_type == "overconfident":
            return AdaptationRecommendation(
                strategy_type=AdaptationStrategy.DECREASE_CONFIDENCE,
                title=f"Reduce {confidence_level.title()} Confidence Overconfidence",
                description=f"Calibrate {confidence_level} confidence predictions to reduce overconfidence",
                rationale=f"Systematic overconfidence detected in {confidence_level} confidence predictions",
                expected_impact=context.get("calibration_error", 0.1) * 0.5,
                confidence=pattern.confidence,
                priority=0.8,
                implementation_complexity=0.4,
                affected_contexts=[f"{confidence_level}_confidence"],
                specific_actions=[
                    f"Apply confidence penalty for {confidence_level} predictions",
                    "Implement calibration training",
                    "Add uncertainty quantification",
                ],
                success_metrics=[
                    "Improved calibration error",
                    "Better confidence-accuracy correlation",
                ],
                timeline="short_term",
            )

        elif miscalibration_type == "underconfident":
            return AdaptationRecommendation(
                strategy_type=AdaptationStrategy.INCREASE_CONFIDENCE,
                title=f"Increase {confidence_level.title()} Confidence Appropriately",
                description=f"Adjust {confidence_level} confidence predictions to reduce underconfidence",
                rationale=f"Systematic underconfidence detected in {confidence_level} confidence predictions",
                expected_impact=context.get("calibration_error", 0.1) * 0.3,
                confidence=pattern.confidence,
                priority=0.6,
                implementation_complexity=0.5,
                affected_contexts=[f"{confidence_level}_confidence"],
                specific_actions=[
                    f"Boost confidence for well-supported {confidence_level} predictions",
                    "Improve evidence quality assessment",
                    "Enhance reasoning validation",
                ],
                success_metrics=[
                    "Improved calibration error",
                    "Better utilization of high-quality evidence",
                ],
                timeline="medium_term",
            )

        return None

    def _recommend_method_adaptation(
        self, pattern: DetectedPattern
    ) -> Optional[AdaptationRecommendation]:
        """Recommend adaptations for method effectiveness patterns."""
        context = pattern.context
        best_method = context.get("best_method", "unknown")
        worst_method = context.get("worst_method", "unknown")
        performance_gap = context.get("performance_gap", 0)

        return AdaptationRecommendation(
            strategy_type=AdaptationStrategy.CHANGE_METHOD_PREFERENCE,
            title=f"Optimize Method Selection: Favor {best_method}",
            description=f"Adjust method preferences based on performance analysis",
            rationale=f"Significant performance gap detected: {best_method} outperforms {worst_method} by {performance_gap:.2f}",
            expected_impact=performance_gap * 0.4,
            confidence=pattern.confidence,
            priority=0.9,
            implementation_complexity=0.3,
            affected_contexts=["method_selection"],
            specific_actions=[
                f"Increase weight for {best_method} in ensemble",
                f"Reduce reliance on {worst_method}",
                f"Investigate why {best_method} performs better",
                f"Consider retiring {worst_method} if consistently poor",
            ],
            success_metrics=[
                "Improved overall accuracy",
                "Better method performance distribution",
            ],
            timeline="immediate",
        )

    def _recommend_temporal_adaptation(
        self, pattern: DetectedPattern
    ) -> Optional[AdaptationRecommendation]:
        """Recommend adaptations for temporal patterns."""
        context = pattern.context
        correlation = context.get("correlation", 0)

        if pattern.trend_direction == "improving":
            return AdaptationRecommendation(
                strategy_type=AdaptationStrategy.INCREASE_AGGRESSIVENESS,
                title="Capitalize on Improving Performance",
                description="Increase aggressiveness to capitalize on improving performance trend",
                rationale=f"Performance is improving over time with correlation {correlation:.2f}",
                expected_impact=0.05,
                confidence=pattern.confidence,
                priority=0.7,
                implementation_complexity=0.4,
                affected_contexts=["temporal_strategy"],
                specific_actions=[
                    "Increase confidence in recent predictions",
                    "Allocate more resources to current strategies",
                    "Consider more aggressive tournament positioning",
                ],
                success_metrics=[
                    "Continued performance improvement",
                    "Better tournament ranking",
                ],
                timeline="short_term",
            )

        elif pattern.trend_direction == "declining":
            return AdaptationRecommendation(
                strategy_type=AdaptationStrategy.INCREASE_CONSERVATISM,
                title="Address Declining Performance",
                description="Implement conservative measures to address declining performance",
                rationale=f"Performance is declining over time with correlation {correlation:.2f}",
                expected_impact=0.08,
                confidence=pattern.confidence,
                priority=0.9,
                implementation_complexity=0.6,
                affected_contexts=["temporal_strategy"],
                specific_actions=[
                    "Review and update forecasting methodologies",
                    "Increase validation and quality checks",
                    "Consider strategy reset or major adjustments",
                ],
                success_metrics=["Stabilized performance", "Reversed declining trend"],
                timeline="immediate",
            )

        return None

    def _recommend_ensemble_adaptation(
        self, pattern: DetectedPattern
    ) -> Optional[AdaptationRecommendation]:
        """Recommend adaptations for ensemble patterns."""
        context = pattern.context
        ensemble_advantage = context.get("ensemble_advantage", 0)

        if ensemble_advantage > 0.1:  # Positive synergy
            return AdaptationRecommendation(
                strategy_type=AdaptationStrategy.ADJUST_ENSEMBLE_WEIGHTS,
                title="Increase Ensemble Usage",
                description="Increase reliance on ensemble methods due to positive synergy",
                rationale=f"Ensemble methods show {ensemble_advantage:+.2f} accuracy advantage",
                expected_impact=ensemble_advantage * 0.6,
                confidence=pattern.confidence,
                priority=0.8,
                implementation_complexity=0.3,
                affected_contexts=["ensemble_strategy"],
                specific_actions=[
                    "Increase ensemble method usage",
                    "Optimize ensemble composition",
                    "Invest in ensemble methodology improvements",
                ],
                success_metrics=[
                    "Increased overall accuracy",
                    "Better ensemble performance",
                ],
                timeline="short_term",
            )

        elif ensemble_advantage < -0.1:  # Negative synergy
            return AdaptationRecommendation(
                strategy_type=AdaptationStrategy.ADJUST_ENSEMBLE_WEIGHTS,
                title="Fix Ensemble Issues",
                description="Address negative ensemble synergy",
                rationale=f"Ensemble methods underperform by {ensemble_advantage:+.2f}",
                expected_impact=abs(ensemble_advantage) * 0.4,
                confidence=pattern.confidence,
                priority=0.9,
                implementation_complexity=0.7,
                affected_contexts=["ensemble_strategy"],
                specific_actions=[
                    "Debug ensemble aggregation methods",
                    "Review individual agent selection",
                    "Consider simpler aggregation strategies",
                    "Investigate ensemble weight optimization",
                ],
                success_metrics=[
                    "Improved ensemble performance",
                    "Positive ensemble synergy",
                ],
                timeline="medium_term",
            )

        return None

    def _generate_competitive_intelligence(
        self,
        patterns: List[DetectedPattern],
        forecasts: List[Forecast],
        questions: List[Question],
        tournament_context: Optional[Dict[str, Any]],
    ) -> Optional[CompetitiveIntelligence]:
        """Generate competitive intelligence from patterns."""
        if not tournament_context:
            return None

        # This would require actual tournament and competitor data
        # For now, return basic intelligence based on patterns

        tournament_id = tournament_context.get("tournament_id", "unknown")

        # Extract insights from patterns
        market_gaps = []
        strategic_recommendations = []

        for pattern in patterns:
            if pattern.pattern_type == PatternType.QUESTION_TYPE_PERFORMANCE:
                if pattern.context.get("performance_difference", 0) > 0.1:
                    market_gaps.append(
                        {
                            "type": "question_type_advantage",
                            "description": f"Strong performance on {pattern.context.get('question_type')} questions",
                            "opportunity_score": pattern.strength,
                        }
                    )

        return CompetitiveIntelligence(
            tournament_id=tournament_id,
            market_gaps=market_gaps,
            competitor_weaknesses=[],  # Would need competitor data
            optimal_positioning={
                "focus_areas": [gap["description"] for gap in market_gaps],
                "confidence_level": "medium",
            },
            timing_opportunities=[],  # Would need timing analysis
            question_type_advantages={},  # Would extract from patterns
            confidence_level_opportunities={},  # Would extract from patterns
            meta_game_insights=[
                "Focus on identified question type advantages",
                "Maintain current successful methodologies",
            ],
            strategic_recommendations=strategic_recommendations,
            timestamp=datetime.utcnow(),
            confidence=0.6,
        )

    def _detect_meta_patterns(
        self, patterns: List[DetectedPattern]
    ) -> List[Dict[str, Any]]:
        """Detect meta-patterns across different pattern types."""
        meta_patterns = []

        # Pattern frequency analysis
        pattern_type_counts = Counter([p.pattern_type for p in patterns])
        if len(pattern_type_counts) > 1:
            most_common_type = pattern_type_counts.most_common(1)[0]
            meta_patterns.append(
                {
                    "type": "pattern_frequency",
                    "description": f"Most common pattern type: {most_common_type[0].value} ({most_common_type[1]} occurrences)",
                    "confidence": 0.8,
                }
            )

        # Pattern strength correlation
        high_strength_patterns = [p for p in patterns if p.strength > 0.2]
        if len(high_strength_patterns) > 2:
            meta_patterns.append(
                {
                    "type": "high_impact_patterns",
                    "description": f"Multiple high-strength patterns detected ({len(high_strength_patterns)} patterns)",
                    "confidence": 0.7,
                    "patterns": [p.title for p in high_strength_patterns],
                }
            )

        return meta_patterns

    def _generate_strategy_evolution_suggestions(
        self, patterns: List[DetectedPattern]
    ) -> List[str]:
        """Generate high-level strategy evolution suggestions."""
        suggestions = []

        # Analyze pattern implications
        performance_patterns = [
            p
            for p in patterns
            if p.pattern_type
            in [
                PatternType.QUESTION_TYPE_PERFORMANCE,
                PatternType.METHOD_EFFECTIVENESS,
                PatternType.TEMPORAL_PERFORMANCE,
            ]
        ]

        if len(performance_patterns) > 2:
            suggestions.append(
                "Consider comprehensive strategy review based on multiple performance patterns"
            )

        calibration_patterns = [
            p for p in patterns if p.pattern_type == PatternType.CONFIDENCE_CALIBRATION
        ]
        if calibration_patterns:
            suggestions.append(
                "Implement systematic calibration training and monitoring"
            )

        method_patterns = [
            p for p in patterns if p.pattern_type == PatternType.METHOD_EFFECTIVENESS
        ]
        if method_patterns:
            suggestions.append("Optimize method selection and ensemble composition")

        temporal_patterns = [
            p for p in patterns if p.pattern_type == PatternType.TEMPORAL_PERFORMANCE
        ]
        if temporal_patterns:
            suggestions.append(
                "Implement adaptive strategy that responds to performance trends"
            )

        return suggestions

    def _group_patterns_by_type(
        self, patterns: List[DetectedPattern]
    ) -> Dict[str, int]:
        """Group patterns by type for summary statistics."""
        return dict(Counter([p.pattern_type.value for p in patterns]))

    def _serialize_pattern(self, pattern: DetectedPattern) -> Dict[str, Any]:
        """Serialize pattern for JSON output."""
        return {
            "type": pattern.pattern_type.value,
            "title": pattern.title,
            "description": pattern.description,
            "confidence": pattern.confidence,
            "strength": pattern.strength,
            "frequency": pattern.frequency,
            "trend_direction": pattern.trend_direction,
            "statistical_significance": pattern.statistical_significance,
            "affected_questions_count": len(pattern.affected_questions),
            "affected_agents": pattern.affected_agents,
            "first_observed": pattern.first_observed.isoformat(),
            "last_observed": pattern.last_observed.isoformat(),
            "examples_count": len(pattern.examples),
            "context": pattern.context,
        }

    def _serialize_recommendation(
        self, recommendation: AdaptationRecommendation
    ) -> Dict[str, Any]:
        """Serialize recommendation for JSON output."""
        return {
            "strategy_type": recommendation.strategy_type.value,
            "title": recommendation.title,
            "description": recommendation.description,
            "rationale": recommendation.rationale,
            "expected_impact": recommendation.expected_impact,
            "confidence": recommendation.confidence,
            "priority": recommendation.priority,
            "implementation_complexity": recommendation.implementation_complexity,
            "affected_contexts": recommendation.affected_contexts,
            "specific_actions": recommendation.specific_actions,
            "success_metrics": recommendation.success_metrics,
            "timeline": recommendation.timeline,
            "dependencies": recommendation.dependencies,
            "risks": recommendation.risks,
        }

    def _serialize_competitive_intelligence(
        self, intelligence: CompetitiveIntelligence
    ) -> Dict[str, Any]:
        """Serialize competitive intelligence for JSON output."""
        return {
            "tournament_id": intelligence.tournament_id,
            "market_gaps": intelligence.market_gaps,
            "competitor_weaknesses": intelligence.competitor_weaknesses,
            "optimal_positioning": intelligence.optimal_positioning,
            "timing_opportunities": intelligence.timing_opportunities,
            "question_type_advantages": intelligence.question_type_advantages,
            "confidence_level_opportunities": intelligence.confidence_level_opportunities,
            "meta_game_insights": intelligence.meta_game_insights,
            "strategic_recommendations": intelligence.strategic_recommendations,
            "timestamp": intelligence.timestamp.isoformat(),
            "confidence": intelligence.confidence,
        }

    def get_pattern_history(self, days: int = 30) -> Dict[str, Any]:
        """Get pattern detection history for the last N days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        recent_patterns = [
            pattern
            for pattern in self.detected_patterns
            if pattern.last_observed >= cutoff_date
        ]

        return {
            "period_days": days,
            "total_patterns": len(recent_patterns),
            "patterns_by_type": self._group_patterns_by_type(recent_patterns),
            "high_confidence_patterns": len(
                [p for p in recent_patterns if p.confidence > 0.8]
            ),
            "actionable_patterns": len(
                [p for p in recent_patterns if p.strength > 0.15]
            ),
        }

    def get_adaptation_tracking(self) -> Dict[str, Any]:
        """Get tracking of adaptation recommendations and their implementation."""
        recent_recommendations = [
            rec
            for rec in self.adaptation_recommendations
            if rec.timeline in ["immediate", "short_term"]
        ]

        return {
            "total_recommendations": len(self.adaptation_recommendations),
            "high_priority_recommendations": len(
                [r for r in recent_recommendations if r.priority > 0.8]
            ),
            "recommendations_by_strategy": dict(
                Counter([r.strategy_type.value for r in recent_recommendations])
            ),
            "expected_total_impact": sum(
                r.expected_impact for r in recent_recommendations
            ),
        }
