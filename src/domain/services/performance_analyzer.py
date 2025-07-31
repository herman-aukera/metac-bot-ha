"""Performance analyzer for continuous improvement and learning."""

from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math
from uuid import UUID
import structlog

from ..entities.forecast import Forecast, ForecastStatus
from ..entities.prediction import Prediction, PredictionMethod, PredictionConfidence
from ..entities.question import Question
from ..value_objects.probability import Probability


logger = structlog.get_logger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics tracked."""
    ACCURACY = "accuracy"
    CALIBRATION = "calibration"
    BRIER_SCORE = "brier_score"
    LOG_SCORE = "log_score"
    RESOLUTION = "resolution"
    RELIABILITY = "reliability"
    SHARPNESS = "sharpness"
    DISCRIMINATION = "discrimination"


class ImprovementOpportunityType(Enum):
    """Types of improvement opportunities identified."""
    OVERCONFIDENCE = "overconfidence"
    UNDERCONFIDENCE = "underconfidence"
    POOR_CALIBRATION = "poor_calibration"
    LOW_RESOLUTION = "low_resolution"
    INCONSISTENT_REASONING = "inconsistent_reasoning"
    INSUFFICIENT_RESEARCH = "insufficient_research"
    BIAS_DETECTION = "bias_detection"
    METHOD_SELECTION = "method_selection"
    ENSEMBLE_WEIGHTING = "ensemble_weighting"
    TIMING_OPTIMIZATION = "timing_optimization"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    question_id: Optional[UUID] = None
    agent_id: Optional[str] = None
    method: Optional[str] = None
    confidence_level: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementOpportunity:
    """Identified improvement opportunity."""
    opportunity_type: ImprovementOpportunityType
    description: str
    severity: float  # 0.0 to 1.0
    affected_questions: List[UUID]
    affected_agents: List[str]
    recommended_actions: List[str]
    potential_impact: float  # Expected improvement in performance
    implementation_difficulty: float  # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformancePattern:
    """Detected performance pattern."""
    pattern_type: str
    description: str
    frequency: float
    confidence: float
    affected_contexts: List[str]
    performance_impact: float
    first_observed: datetime
    last_observed: datetime
    examples: List[UUID] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningInsight:
    """Learning insight derived from performance analysis."""
    insight_type: str
    title: str
    description: str
    evidence: List[str]
    confidence: float
    actionable_recommendations: List[str]
    expected_improvement: float
    priority: float  # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceAnalyzer:
    """
    Service for analyzing forecasting performance and identifying improvement opportunities.

    Provides comprehensive analysis of resolved predictions, accuracy attribution,
    improvement opportunity identification, and strategy refinement recommendations.
    """

    def __init__(self):
        self.performance_history: List[PerformanceMetric] = []
        self.improvement_opportunities: List[ImprovementOpportunity] = []
        self.detected_patterns: List[PerformancePattern] = []
        self.learning_insights: List[LearningInsight] = []

        # Configuration
        self.min_samples_for_analysis = 10
        self.calibration_bins = 10
        self.pattern_detection_window_days = 30
        self.improvement_threshold = 0.05  # Minimum improvement to consider significant

        # Performance tracking by context
        self.agent_performance: Dict[str, List[PerformanceMetric]] = {}
        self.method_performance: Dict[str, List[PerformanceMetric]] = {}
        self.question_type_performance: Dict[str, List[PerformanceMetric]] = {}
        self.confidence_level_performance: Dict[str, List[PerformanceMetric]] = {}

    def analyze_resolved_predictions(
        self,
        resolved_forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> Dict[str, Any]:
        """
        Analyze resolved predictions for accuracy attribution and performance insights.

        Args:
            resolved_forecasts: List of resolved forecasts
            ground_truth: Corresponding ground truth values

        Returns:
            Comprehensive performance analysis results
        """
        if not resolved_forecasts:
            raise ValueError("Cannot analyze empty forecast list")

        if len(resolved_forecasts) != len(ground_truth):
            raise ValueError("Forecasts and ground truth must have same length")

        logger.info(
            "Analyzing resolved predictions",
            forecast_count=len(resolved_forecasts),
            analysis_timestamp=datetime.utcnow()
        )

        # Calculate performance metrics
        overall_metrics = self._calculate_overall_metrics(resolved_forecasts, ground_truth)
        agent_metrics = self._calculate_agent_metrics(resolved_forecasts, ground_truth)
        method_metrics = self._calculate_method_metrics(resolved_forecasts, ground_truth)
        calibration_analysis = self._analyze_calibration(resolved_forecasts, ground_truth)

        # Store metrics for historical tracking
        self._store_performance_metrics(overall_metrics, agent_metrics, method_metrics)

        # Identify improvement opportunities
        opportunities = self._identify_improvement_opportunities(
            resolved_forecasts, ground_truth, overall_metrics
        )

        # Detect performance patterns
        patterns = self._detect_performance_patterns(resolved_forecasts, ground_truth)

        # Generate learning insights
        insights = self._generate_learning_insights(
            overall_metrics, agent_metrics, method_metrics, opportunities, patterns
        )

        analysis_results = {
            "analysis_timestamp": datetime.utcnow(),
            "sample_size": len(resolved_forecasts),
            "overall_metrics": overall_metrics,
            "agent_performance": agent_metrics,
            "method_performance": method_metrics,
            "calibration_analysis": calibration_analysis,
            "improvement_opportunities": [self._serialize_opportunity(opp) for opp in opportunities],
            "performance_patterns": [self._serialize_pattern(pattern) for pattern in patterns],
            "learning_insights": [self._serialize_insight(insight) for insight in insights],
            "recommendations": self._generate_actionable_recommendations(opportunities, insights)
        }

        logger.info(
            "Performance analysis completed",
            opportunities_found=len(opportunities),
            patterns_detected=len(patterns),
            insights_generated=len(insights)
        )

        return analysis_results

    def _calculate_overall_metrics(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        predictions = [f.prediction for f in forecasts]

        # Brier Score
        brier_scores = [
            (pred - (1.0 if truth else 0.0)) ** 2
            for pred, truth in zip(predictions, ground_truth)
        ]
        brier_score = statistics.mean(brier_scores)

        # Log Score (avoiding log(0) and log(1))
        log_scores = []
        for pred, truth in zip(predictions, ground_truth):
            # Clip predictions to avoid log(0)
            clipped_pred = max(0.001, min(0.999, pred))
            if truth:
                log_scores.append(-math.log(clipped_pred))
            else:
                log_scores.append(-math.log(1 - clipped_pred))
        log_score = statistics.mean(log_scores)

        # Accuracy (using 0.5 threshold)
        correct_predictions = [
            (pred > 0.5) == truth
            for pred, truth in zip(predictions, ground_truth)
        ]
        accuracy = sum(correct_predictions) / len(correct_predictions)

        # Resolution and Reliability (components of Brier Score decomposition)
        base_rate = sum(ground_truth) / len(ground_truth)
        resolution = self._calculate_resolution(predictions, ground_truth, base_rate)
        reliability = self._calculate_reliability(predictions, ground_truth)

        # Sharpness (average distance from base rate)
        sharpness = statistics.mean([abs(pred - base_rate) for pred in predictions])

        # Discrimination (ability to distinguish between outcomes)
        discrimination = self._calculate_discrimination(predictions, ground_truth)

        return {
            "brier_score": brier_score,
            "log_score": log_score,
            "accuracy": accuracy,
            "resolution": resolution,
            "reliability": reliability,
            "sharpness": sharpness,
            "discrimination": discrimination,
            "base_rate": base_rate,
            "sample_size": len(forecasts)
        }

    def _calculate_agent_metrics(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by agent."""
        agent_metrics = {}

        # Group forecasts by agent
        agent_forecasts = {}
        for i, forecast in enumerate(forecasts):
            agent_id = forecast.final_prediction.created_by
            if agent_id not in agent_forecasts:
                agent_forecasts[agent_id] = []
            agent_forecasts[agent_id].append((forecast, ground_truth[i]))

        # Calculate metrics for each agent
        for agent_id, agent_data in agent_forecasts.items():
            if len(agent_data) < 3:  # Need minimum samples
                continue

            agent_forecasts_list = [item[0] for item in agent_data]
            agent_ground_truth = [item[1] for item in agent_data]

            agent_metrics[agent_id] = self._calculate_overall_metrics(
                agent_forecasts_list, agent_ground_truth
            )

            # Add agent-specific metrics
            agent_metrics[agent_id]["prediction_count"] = len(agent_data)
            agent_metrics[agent_id]["confidence_correlation"] = self._calculate_confidence_correlation(
                agent_forecasts_list, agent_ground_truth
            )

        return agent_metrics

    def _calculate_method_metrics(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by prediction method."""
        method_metrics = {}

        # Group forecasts by method
        method_forecasts = {}
        for i, forecast in enumerate(forecasts):
            method = forecast.method
            if method not in method_forecasts:
                method_forecasts[method] = []
            method_forecasts[method].append((forecast, ground_truth[i]))

        # Calculate metrics for each method
        for method, method_data in method_forecasts.items():
            if len(method_data) < 3:  # Need minimum samples
                continue

            method_forecasts_list = [item[0] for item in method_data]
            method_ground_truth = [item[1] for item in method_data]

            method_metrics[method] = self._calculate_overall_metrics(
                method_forecasts_list, method_ground_truth
            )

            method_metrics[method]["prediction_count"] = len(method_data)

        return method_metrics

    def _analyze_calibration(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> Dict[str, Any]:
        """Analyze prediction calibration."""
        predictions = [f.prediction for f in forecasts]

        # Create calibration bins
        bin_edges = [i / self.calibration_bins for i in range(self.calibration_bins + 1)]
        bin_counts = [0] * self.calibration_bins
        bin_correct = [0] * self.calibration_bins
        bin_predictions = [[] for _ in range(self.calibration_bins)]

        # Assign predictions to bins
        for pred, truth in zip(predictions, ground_truth):
            bin_idx = min(int(pred * self.calibration_bins), self.calibration_bins - 1)
            bin_counts[bin_idx] += 1
            bin_predictions[bin_idx].append(pred)
            if truth:
                bin_correct[bin_idx] += 1

        # Calculate calibration metrics
        calibration_data = []
        total_calibration_error = 0.0

        for i in range(self.calibration_bins):
            if bin_counts[i] > 0:
                bin_accuracy = bin_correct[i] / bin_counts[i]
                bin_confidence = statistics.mean(bin_predictions[i])
                calibration_error = abs(bin_confidence - bin_accuracy)
                total_calibration_error += calibration_error * bin_counts[i]

                calibration_data.append({
                    "bin_start": bin_edges[i],
                    "bin_end": bin_edges[i + 1],
                    "count": bin_counts[i],
                    "accuracy": bin_accuracy,
                    "confidence": bin_confidence,
                    "calibration_error": calibration_error
                })

        # Expected Calibration Error (ECE)
        ece = total_calibration_error / len(predictions)

        # Maximum Calibration Error (MCE)
        mce = max([bin_data["calibration_error"] for bin_data in calibration_data]) if calibration_data else 0.0

        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "calibration_bins": calibration_data,
            "is_well_calibrated": ece < 0.1,  # Threshold for good calibration
            "overconfidence_detected": self._detect_overconfidence(calibration_data),
            "underconfidence_detected": self._detect_underconfidence(calibration_data)
        }

    def _calculate_resolution(
        self,
        predictions: List[float],
        ground_truth: List[bool],
        base_rate: float
    ) -> float:
        """Calculate resolution component of Brier score decomposition."""
        # Group by prediction bins for resolution calculation
        bin_size = 0.1
        bins = {}

        for pred, truth in zip(predictions, ground_truth):
            bin_key = round(pred / bin_size) * bin_size
            if bin_key not in bins:
                bins[bin_key] = []
            bins[bin_key].append(truth)

        resolution = 0.0
        total_count = len(predictions)

        for bin_predictions in bins.values():
            if len(bin_predictions) > 0:
                bin_rate = sum(bin_predictions) / len(bin_predictions)
                bin_weight = len(bin_predictions) / total_count
                resolution += bin_weight * (bin_rate - base_rate) ** 2

        return resolution

    def _calculate_reliability(
        self,
        predictions: List[float],
        ground_truth: List[bool]
    ) -> float:
        """Calculate reliability component of Brier score decomposition."""
        # Group by prediction bins
        bin_size = 0.1
        bins = {}

        for pred, truth in zip(predictions, ground_truth):
            bin_key = round(pred / bin_size) * bin_size
            if bin_key not in bins:
                bins[bin_key] = {"predictions": [], "outcomes": []}
            bins[bin_key]["predictions"].append(pred)
            bins[bin_key]["outcomes"].append(truth)

        reliability = 0.0
        total_count = len(predictions)

        for bin_data in bins.values():
            if len(bin_data["predictions"]) > 0:
                avg_prediction = statistics.mean(bin_data["predictions"])
                actual_rate = sum(bin_data["outcomes"]) / len(bin_data["outcomes"])
                bin_weight = len(bin_data["predictions"]) / total_count
                reliability += bin_weight * (avg_prediction - actual_rate) ** 2

        return reliability

    def _calculate_discrimination(
        self,
        predictions: List[float],
        ground_truth: List[bool]
    ) -> float:
        """Calculate discrimination ability (AUC approximation)."""
        if len(set(ground_truth)) < 2:  # Need both positive and negative cases
            return 0.5

        # Simple AUC calculation
        positive_predictions = [pred for pred, truth in zip(predictions, ground_truth) if truth]
        negative_predictions = [pred for pred, truth in zip(predictions, ground_truth) if not truth]

        if not positive_predictions or not negative_predictions:
            return 0.5

        # Count pairs where positive prediction > negative prediction
        correct_pairs = 0
        total_pairs = 0

        for pos_pred in positive_predictions:
            for neg_pred in negative_predictions:
                total_pairs += 1
                if pos_pred > neg_pred:
                    correct_pairs += 1
                elif pos_pred == neg_pred:
                    correct_pairs += 0.5  # Tie

        return correct_pairs / total_pairs if total_pairs > 0 else 0.5

    def _calculate_confidence_correlation(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> float:
        """Calculate correlation between confidence and accuracy."""
        if len(forecasts) < 3:
            return 0.0

        confidences = [f.confidence for f in forecasts]
        accuracies = [
            1.0 if (f.prediction > 0.5) == truth else 0.0
            for f, truth in zip(forecasts, ground_truth)
        ]

        # Calculate Pearson correlation
        if len(set(confidences)) < 2 or len(set(accuracies)) < 2:
            return 0.0

        mean_conf = statistics.mean(confidences)
        mean_acc = statistics.mean(accuracies)

        numerator = sum(
            (conf - mean_conf) * (acc - mean_acc)
            for conf, acc in zip(confidences, accuracies)
        )

        conf_var = sum((conf - mean_conf) ** 2 for conf in confidences)
        acc_var = sum((acc - mean_acc) ** 2 for acc in accuracies)

        denominator = math.sqrt(conf_var * acc_var)

        return numerator / denominator if denominator > 0 else 0.0

    def _detect_overconfidence(self, calibration_data: List[Dict[str, Any]]) -> bool:
        """Detect systematic overconfidence."""
        overconfident_bins = [
            bin_data for bin_data in calibration_data
            if bin_data["confidence"] > bin_data["accuracy"] + 0.1
            and bin_data["count"] >= 3
        ]

        # Overconfidence if multiple bins show the pattern
        return len(overconfident_bins) >= 2

    def _detect_underconfidence(self, calibration_data: List[Dict[str, Any]]) -> bool:
        """Detect systematic underconfidence."""
        underconfident_bins = [
            bin_data for bin_data in calibration_data
            if bin_data["accuracy"] > bin_data["confidence"] + 0.1
            and bin_data["count"] >= 3
        ]

        # Underconfidence if multiple bins show the pattern
        return len(underconfident_bins) >= 2

    def _store_performance_metrics(
        self,
        overall_metrics: Dict[str, float],
        agent_metrics: Dict[str, Dict[str, float]],
        method_metrics: Dict[str, Dict[str, float]]
    ) -> None:
        """Store performance metrics for historical tracking."""
        timestamp = datetime.utcnow()

        # Store overall metrics
        metric_mapping = {
            "brier_score": PerformanceMetricType.BRIER_SCORE,
            "log_score": PerformanceMetricType.LOG_SCORE,
            "accuracy": PerformanceMetricType.ACCURACY,
            "resolution": PerformanceMetricType.RESOLUTION,
            "reliability": PerformanceMetricType.RELIABILITY
        }

        for metric_name, value in overall_metrics.items():
            if metric_name in metric_mapping:
                metric = PerformanceMetric(
                    metric_type=metric_mapping[metric_name],
                    value=value,
                    timestamp=timestamp,
                    metadata={"context": "overall"}
                )
                self.performance_history.append(metric)

        # Store agent metrics
        for agent_id, metrics in agent_metrics.items():
            if agent_id not in self.agent_performance:
                self.agent_performance[agent_id] = []

            for metric_name, value in metrics.items():
                if metric_name in metric_mapping:
                    metric = PerformanceMetric(
                        metric_type=metric_mapping[metric_name],
                        value=value,
                        timestamp=timestamp,
                        agent_id=agent_id,
                        metadata={"context": "agent"}
                    )
                    self.agent_performance[agent_id].append(metric)

        # Store method metrics
        for method, metrics in method_metrics.items():
            if method not in self.method_performance:
                self.method_performance[method] = []

            for metric_name, value in metrics.items():
                if metric_name in metric_mapping:
                    metric = PerformanceMetric(
                        metric_type=metric_mapping[metric_name],
                        value=value,
                        timestamp=timestamp,
                        method=method,
                        metadata={"context": "method"}
                    )
                    self.method_performance[method].append(metric)

    def _identify_improvement_opportunities(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool],
        overall_metrics: Dict[str, float]
    ) -> List[ImprovementOpportunity]:
        """Identify specific improvement opportunities."""
        opportunities = []

        # Check for calibration issues
        if overall_metrics.get("reliability", 0) > 0.05:
            opportunities.append(ImprovementOpportunity(
                opportunity_type=ImprovementOpportunityType.POOR_CALIBRATION,
                description="Predictions are poorly calibrated - confidence levels don't match actual accuracy",
                severity=min(1.0, overall_metrics["reliability"] * 10),
                affected_questions=[f.id for f in forecasts],
                affected_agents=list(set(f.final_prediction.created_by for f in forecasts)),
                recommended_actions=[
                    "Implement confidence calibration training",
                    "Add calibration feedback loops",
                    "Use temperature scaling for confidence adjustment"
                ],
                potential_impact=overall_metrics["reliability"] * 0.5,
                implementation_difficulty=0.6,
                timestamp=datetime.utcnow()
            ))

        # Check for low resolution
        if overall_metrics.get("resolution", 0) < 0.02:
            opportunities.append(ImprovementOpportunity(
                opportunity_type=ImprovementOpportunityType.LOW_RESOLUTION,
                description="Predictions lack resolution - not sufficiently differentiated",
                severity=0.7,
                affected_questions=[f.id for f in forecasts],
                affected_agents=list(set(f.final_prediction.created_by for f in forecasts)),
                recommended_actions=[
                    "Encourage more diverse prediction ranges",
                    "Improve evidence gathering for differentiation",
                    "Add incentives for well-calibrated extreme predictions"
                ],
                potential_impact=0.1,
                implementation_difficulty=0.8,
                timestamp=datetime.utcnow()
            ))

        # Check for poor discrimination
        if overall_metrics.get("discrimination", 0.5) < 0.6:
            opportunities.append(ImprovementOpportunity(
                opportunity_type=ImprovementOpportunityType.BIAS_DETECTION,
                description="Poor ability to discriminate between positive and negative outcomes",
                severity=0.8,
                affected_questions=[f.id for f in forecasts],
                affected_agents=list(set(f.final_prediction.created_by for f in forecasts)),
                recommended_actions=[
                    "Improve feature selection and evidence quality",
                    "Add bias detection and mitigation",
                    "Enhance reasoning methodology training"
                ],
                potential_impact=0.15,
                implementation_difficulty=0.7,
                timestamp=datetime.utcnow()
            ))

        return opportunities

    def _detect_performance_patterns(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> List[PerformancePattern]:
        """Detect patterns in performance data."""
        patterns = []

        # Pattern: Confidence-accuracy mismatch
        confidence_accuracy_pattern = self._detect_confidence_accuracy_pattern(forecasts, ground_truth)
        if confidence_accuracy_pattern:
            patterns.append(confidence_accuracy_pattern)

        # Pattern: Method performance differences
        method_pattern = self._detect_method_performance_pattern(forecasts, ground_truth)
        if method_pattern:
            patterns.append(method_pattern)

        # Pattern: Time-based performance trends
        time_pattern = self._detect_time_based_pattern(forecasts, ground_truth)
        if time_pattern:
            patterns.append(time_pattern)

        return patterns

    def _detect_confidence_accuracy_pattern(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> Optional[PerformancePattern]:
        """Detect confidence-accuracy mismatch patterns."""
        high_conf_forecasts = [
            (f, truth) for f, truth in zip(forecasts, ground_truth)
            if f.confidence > 0.8
        ]

        if len(high_conf_forecasts) < 5:
            return None

        high_conf_accuracy = sum(
            1 for f, truth in high_conf_forecasts
            if (f.prediction > 0.5) == truth
        ) / len(high_conf_forecasts)

        if high_conf_accuracy < 0.7:  # High confidence but low accuracy
            return PerformancePattern(
                pattern_type="confidence_accuracy_mismatch",
                description=f"High confidence predictions ({len(high_conf_forecasts)}) have low accuracy ({high_conf_accuracy:.2f})",
                frequency=len(high_conf_forecasts) / len(forecasts),
                confidence=0.8,
                affected_contexts=["high_confidence_predictions"],
                performance_impact=-0.1,
                first_observed=min(f.created_at for f, _ in high_conf_forecasts),
                last_observed=max(f.created_at for f, _ in high_conf_forecasts),
                examples=[f.id for f, _ in high_conf_forecasts[:5]]
            )

        return None

    def _detect_method_performance_pattern(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> Optional[PerformancePattern]:
        """Detect method-specific performance patterns."""
        method_accuracy = {}

        for f, truth in zip(forecasts, ground_truth):
            method = f.method
            if method not in method_accuracy:
                method_accuracy[method] = []
            method_accuracy[method].append((f.prediction > 0.5) == truth)

        # Find methods with significantly different performance
        method_scores = {
            method: sum(accuracies) / len(accuracies)
            for method, accuracies in method_accuracy.items()
            if len(accuracies) >= 3
        }

        if len(method_scores) < 2:
            return None

        best_method = max(method_scores.keys(), key=lambda m: method_scores[m])
        worst_method = min(method_scores.keys(), key=lambda m: method_scores[m])

        performance_gap = method_scores[best_method] - method_scores[worst_method]

        if performance_gap > 0.15:  # Significant difference
            return PerformancePattern(
                pattern_type="method_performance_difference",
                description=f"Significant performance gap between {best_method} ({method_scores[best_method]:.2f}) and {worst_method} ({method_scores[worst_method]:.2f})",
                frequency=1.0,
                confidence=0.9,
                affected_contexts=[best_method, worst_method],
                performance_impact=performance_gap,
                first_observed=min(f.created_at for f in forecasts),
                last_observed=max(f.created_at for f in forecasts),
                metadata={"method_scores": method_scores}
            )

        return None

    def _detect_time_based_pattern(
        self,
        forecasts: List[Forecast],
        ground_truth: List[bool]
    ) -> Optional[PerformancePattern]:
        """Detect time-based performance trends."""
        if len(forecasts) < 10:
            return None

        # Sort by creation time
        sorted_data = sorted(zip(forecasts, ground_truth), key=lambda x: x[0].created_at)

        # Split into early and late periods
        split_point = len(sorted_data) // 2
        early_data = sorted_data[:split_point]
        late_data = sorted_data[split_point:]

        early_accuracy = sum(
            1 for f, truth in early_data
            if (f.prediction > 0.5) == truth
        ) / len(early_data)

        late_accuracy = sum(
            1 for f, truth in late_data
            if (f.prediction > 0.5) == truth
        ) / len(late_data)

        improvement = late_accuracy - early_accuracy

        if abs(improvement) > 0.1:  # Significant trend
            trend_type = "improving" if improvement > 0 else "declining"
            return PerformancePattern(
                pattern_type=f"performance_{trend_type}",
                description=f"Performance is {trend_type} over time: {early_accuracy:.2f} → {late_accuracy:.2f}",
                frequency=1.0,
                confidence=0.7,
                affected_contexts=["temporal_trend"],
                performance_impact=improvement,
                first_observed=early_data[0][0].created_at,
                last_observed=late_data[-1][0].created_at,
                metadata={
                    "early_accuracy": early_accuracy,
                    "late_accuracy": late_accuracy,
                    "improvement": improvement
                }
            )

        return None

    def _generate_learning_insights(
        self,
        overall_metrics: Dict[str, float],
        agent_metrics: Dict[str, Dict[str, float]],
        method_metrics: Dict[str, Dict[str, float]],
        opportunities: List[ImprovementOpportunity],
        patterns: List[PerformancePattern]
    ) -> List[LearningInsight]:
        """Generate actionable learning insights."""
        insights = []

        # Insight from overall performance
        if overall_metrics.get("brier_score", 1.0) > 0.25:
            insights.append(LearningInsight(
                insight_type="overall_performance",
                title="Overall Performance Needs Improvement",
                description=f"Brier score of {overall_metrics['brier_score']:.3f} indicates room for improvement",
                evidence=[
                    f"Brier score: {overall_metrics['brier_score']:.3f}",
                    f"Accuracy: {overall_metrics.get('accuracy', 0):.3f}",
                    f"Calibration reliability: {overall_metrics.get('reliability', 0):.3f}"
                ],
                confidence=0.9,
                actionable_recommendations=[
                    "Focus on calibration training",
                    "Improve evidence gathering processes",
                    "Implement ensemble methods"
                ],
                expected_improvement=0.1,
                priority=0.8,
                timestamp=datetime.utcnow()
            ))

        # Insight from agent performance differences
        if len(agent_metrics) > 1:
            agent_scores = {
                agent: metrics.get("brier_score", 1.0)
                for agent, metrics in agent_metrics.items()
            }
            best_agent = min(agent_scores.keys(), key=lambda a: agent_scores[a])
            worst_agent = max(agent_scores.keys(), key=lambda a: agent_scores[a])

            if agent_scores[worst_agent] - agent_scores[best_agent] > 0.1:
                insights.append(LearningInsight(
                    insight_type="agent_performance_gap",
                    title="Significant Agent Performance Differences",
                    description=f"Performance gap between best ({best_agent}) and worst ({worst_agent}) agents",
                    evidence=[
                        f"Best agent ({best_agent}): {agent_scores[best_agent]:.3f}",
                        f"Worst agent ({worst_agent}): {agent_scores[worst_agent]:.3f}",
                        f"Performance gap: {agent_scores[worst_agent] - agent_scores[best_agent]:.3f}"
                    ],
                    confidence=0.8,
                    actionable_recommendations=[
                        f"Study {best_agent}'s methodology",
                        f"Provide additional training for {worst_agent}",
                        "Consider ensemble weighting adjustments"
                    ],
                    expected_improvement=0.05,
                    priority=0.7,
                    timestamp=datetime.utcnow()
                ))

        # Insights from patterns
        for pattern in patterns:
            if pattern.performance_impact < -0.05:  # Negative impact patterns
                insights.append(LearningInsight(
                    insight_type="pattern_based",
                    title=f"Detected Performance Issue: {pattern.pattern_type}",
                    description=pattern.description,
                    evidence=[f"Pattern confidence: {pattern.confidence:.2f}"],
                    confidence=pattern.confidence,
                    actionable_recommendations=[
                        "Investigate root causes of this pattern",
                        "Implement targeted interventions",
                        "Monitor pattern evolution"
                    ],
                    expected_improvement=abs(pattern.performance_impact),
                    priority=min(1.0, abs(pattern.performance_impact) * 2),
                    timestamp=datetime.utcnow(),
                    metadata={"pattern": pattern.pattern_type}
                ))

        return insights

    def _generate_actionable_recommendations(
        self,
        opportunities: List[ImprovementOpportunity],
        insights: List[LearningInsight]
    ) -> List[Dict[str, Any]]:
        """Generate prioritized actionable recommendations."""
        recommendations = []

        # High-priority opportunities
        high_priority_opportunities = [
            opp for opp in opportunities
            if opp.severity > 0.7 and opp.potential_impact > 0.05
        ]

        for opp in high_priority_opportunities:
            recommendations.append({
                "type": "improvement_opportunity",
                "priority": opp.severity * opp.potential_impact,
                "title": f"Address {opp.opportunity_type.value}",
                "description": opp.description,
                "actions": opp.recommended_actions,
                "expected_impact": opp.potential_impact,
                "difficulty": opp.implementation_difficulty
            })

        # High-priority insights
        high_priority_insights = [
            insight for insight in insights
            if insight.priority > 0.6 and insight.expected_improvement > 0.03
        ]

        for insight in high_priority_insights:
            recommendations.append({
                "type": "learning_insight",
                "priority": insight.priority * insight.expected_improvement,
                "title": insight.title,
                "description": insight.description,
                "actions": insight.actionable_recommendations,
                "expected_impact": insight.expected_improvement,
                "confidence": insight.confidence
            })

        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)

        return recommendations[:10]  # Top 10 recommendations

    def _serialize_opportunity(self, opportunity: ImprovementOpportunity) -> Dict[str, Any]:
        """Serialize improvement opportunity for JSON output."""
        return {
            "type": opportunity.opportunity_type.value,
            "description": opportunity.description,
            "severity": opportunity.severity,
            "affected_questions_count": len(opportunity.affected_questions),
            "affected_agents": opportunity.affected_agents,
            "recommended_actions": opportunity.recommended_actions,
            "potential_impact": opportunity.potential_impact,
            "implementation_difficulty": opportunity.implementation_difficulty,
            "timestamp": opportunity.timestamp.isoformat()
        }

    def _serialize_pattern(self, pattern: PerformancePattern) -> Dict[str, Any]:
        """Serialize performance pattern for JSON output."""
        return {
            "type": pattern.pattern_type,
            "description": pattern.description,
            "frequency": pattern.frequency,
            "confidence": pattern.confidence,
            "affected_contexts": pattern.affected_contexts,
            "performance_impact": pattern.performance_impact,
            "first_observed": pattern.first_observed.isoformat(),
            "last_observed": pattern.last_observed.isoformat(),
            "examples_count": len(pattern.examples)
        }

    def _serialize_insight(self, insight: LearningInsight) -> Dict[str, Any]:
        """Serialize learning insight for JSON output."""
        return {
            "type": insight.insight_type,
            "title": insight.title,
            "description": insight.description,
            "evidence": insight.evidence,
            "confidence": insight.confidence,
            "recommendations": insight.actionable_recommendations,
            "expected_improvement": insight.expected_improvement,
            "priority": insight.priority,
            "timestamp": insight.timestamp.isoformat()
        }

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for the last N days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        recent_metrics = [
            metric for metric in self.performance_history
            if metric.timestamp >= cutoff_date
        ]

        if not recent_metrics:
            return {"message": "No recent performance data available"}

        # Group metrics by type
        metrics_by_type = {}
        for metric in recent_metrics:
            metric_type = metric.metric_type.value
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(metric.value)

        # Calculate summary statistics
        summary = {
            "period_days": days,
            "total_metrics": len(recent_metrics),
            "metric_types": {}
        }

        for metric_type, values in metrics_by_type.items():
            summary["metric_types"][metric_type] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else None
            }

        return summary

    def get_improvement_tracking(self) -> Dict[str, Any]:
        """Get tracking of improvement opportunities and their resolution."""
        active_opportunities = [
            opp for opp in self.improvement_opportunities
            if opp.timestamp >= datetime.utcnow() - timedelta(days=90)
        ]

        return {
            "active_opportunities": len(active_opportunities),
            "opportunities_by_type": {
                opp_type.value: len([
                    opp for opp in active_opportunities
                    if opp.opportunity_type == opp_type
                ])
                for opp_type in ImprovementOpportunityType
            },
            "high_severity_count": len([
                opp for opp in active_opportunities
                if opp.severity > 0.7
            ]),
            "total_potential_impact": sum(
                opp.potential_impact for opp in active_opportunities
            )
        }
