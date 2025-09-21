"""Calibration tracking and drift detection service."""

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..entities.prediction import Prediction


class CalibrationDriftSeverity(Enum):
    """Severity levels for calibration drift."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class CalibrationBin:
    """Represents a calibration bin for analysis."""

    confidence_range: Tuple[float, float]
    predicted_probabilities: List[float] = field(default_factory=list)
    actual_outcomes: List[int] = field(default_factory=list)

    @property
    def bin_center(self) -> float:
        """Get center of confidence range."""
        return (self.confidence_range[0] + self.confidence_range[1]) / 2

    @property
    def count(self) -> int:
        """Get number of predictions in bin."""
        return len(self.predicted_probabilities)

    @property
    def observed_frequency(self) -> float:
        """Get observed frequency of positive outcomes."""
        if not self.actual_outcomes:
            return 0.0
        return sum(self.actual_outcomes) / len(self.actual_outcomes)

    @property
    def average_predicted_probability(self) -> float:
        """Get average predicted probability in bin."""
        if not self.predicted_probabilities:
            return self.bin_center
        return statistics.mean(self.predicted_probabilities)

    def add_prediction(self, predicted_prob: float, actual_outcome: int) -> None:
        """Add a prediction to this bin."""
        self.predicted_probabilities.append(predicted_prob)
        self.actual_outcomes.append(actual_outcome)


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration metrics."""

    brier_score: float
    calibration_error: float
    reliability: float
    resolution: float
    uncertainty: float
    sharpness: float

    # Bin-wise analysis
    calibration_bins: List[CalibrationBin]

    # Time-based metrics
    measurement_timestamp: datetime
    time_window_days: int

    # Drift detection
    drift_severity: CalibrationDriftSeverity
    drift_score: float

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration metrics."""
        return {
            "brier_score": self.brier_score,
            "calibration_error": self.calibration_error,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "sharpness": self.sharpness,
            "drift_severity": self.drift_severity.value,
            "drift_score": self.drift_score,
            "measurement_time": self.measurement_timestamp.isoformat(),
            "sample_size": sum(bin.count for bin in self.calibration_bins),
        }


@dataclass
class CalibrationDriftAlert:
    """Alert for calibration drift detection."""

    severity: CalibrationDriftSeverity
    drift_score: float
    affected_categories: List[str]
    detection_timestamp: datetime
    recommended_actions: List[str]

    # Comparison metrics
    current_calibration_error: float
    baseline_calibration_error: float
    error_increase: float

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        return {
            "severity": self.severity.value,
            "drift_score": self.drift_score,
            "affected_categories": self.affected_categories,
            "detection_time": self.detection_timestamp.isoformat(),
            "error_increase": self.error_increase,
            "recommended_actions": self.recommended_actions,
        }


class CalibrationTracker:
    """Service for tracking calibration and detecting drift."""

    def __init__(
        self,
        num_bins: int = 10,
        min_samples_per_bin: int = 5,
        drift_threshold_mild: float = 0.05,
        drift_threshold_moderate: float = 0.10,
        drift_threshold_severe: float = 0.15,
        drift_threshold_critical: float = 0.25,
    ):
        """Initialize calibration tracker."""
        self.num_bins = num_bins
        self.min_samples_per_bin = min_samples_per_bin

        # Drift detection thresholds
        self.drift_thresholds = {
            CalibrationDriftSeverity.MILD: drift_threshold_mild,
            CalibrationDriftSeverity.MODERATE: drift_threshold_moderate,
            CalibrationDriftSeverity.SEVERE: drift_threshold_severe,
            CalibrationDriftSeverity.CRITICAL: drift_threshold_critical,
        }

        # Historical calibration data
        self.calibration_history: List[CalibrationMetrics] = []

        # Category-specific tracking
        self.category_calibration: Dict[str, List[CalibrationMetrics]] = defaultdict(
            list
        )

        # Baseline calibration for drift detection
        self.baseline_calibration: Optional[CalibrationMetrics] = None

    def calculate_calibration_metrics(
        self,
        predictions: List[Prediction],
        actual_outcomes: List[int],
        time_window_days: int = 30,
        category: Optional[str] = None,
    ) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics."""
        if len(predictions) != len(actual_outcomes):
            raise ValueError("Predictions and outcomes must have same length")

        if not predictions:
            raise ValueError("No predictions provided")

        # Extract predicted probabilities
        predicted_probs = []
        for pred in predictions:
            if pred.result.binary_probability is not None:
                predicted_probs.append(pred.result.binary_probability)
            else:
                # Skip non-binary predictions for now
                continue

        if len(predicted_probs) != len(actual_outcomes):
            # Filter outcomes to match valid predictions
            valid_indices = [
                i
                for i, pred in enumerate(predictions)
                if pred.result.binary_probability is not None
            ]
            actual_outcomes = [actual_outcomes[i] for i in valid_indices]

        # Calculate basic metrics
        brier_score = self._calculate_brier_score(predicted_probs, actual_outcomes)
        calibration_bins = self._create_calibration_bins(
            predicted_probs, actual_outcomes
        )

        # Calculate calibration error (reliability)
        calibration_error = self._calculate_calibration_error(calibration_bins)
        reliability = calibration_error  # Same as calibration error

        # Calculate resolution and uncertainty
        resolution = self._calculate_resolution(calibration_bins, actual_outcomes)
        uncertainty = self._calculate_uncertainty(actual_outcomes)

        # Calculate sharpness
        sharpness = self._calculate_sharpness(predicted_probs)

        # Detect drift
        drift_severity, drift_score = self._detect_calibration_drift(
            calibration_error, category
        )

        metrics = CalibrationMetrics(
            brier_score=brier_score,
            calibration_error=calibration_error,
            reliability=reliability,
            resolution=resolution,
            uncertainty=uncertainty,
            sharpness=sharpness,
            calibration_bins=calibration_bins,
            measurement_timestamp=datetime.utcnow(),
            time_window_days=time_window_days,
            drift_severity=drift_severity,
            drift_score=drift_score,
        )

        # Store in history
        self.calibration_history.append(metrics)
        if category:
            self.category_calibration[category].append(metrics)

        # Update baseline if needed
        if self.baseline_calibration is None:
            self.baseline_calibration = metrics

        return metrics

    def detect_calibration_drift(
        self,
        recent_predictions: List[Prediction],
        recent_outcomes: List[int],
        comparison_window_days: int = 90,
        category: Optional[str] = None,
    ) -> Optional[CalibrationDriftAlert]:
        """Detect calibration drift and generate alerts."""
        # Calculate current calibration
        current_metrics = self.calculate_calibration_metrics(
            recent_predictions, recent_outcomes, category=category
        )

        # Get baseline for comparison
        baseline_metrics = self._get_baseline_calibration(
            comparison_window_days, category
        )

        if baseline_metrics is None:
            return None  # Not enough historical data

        # Calculate drift
        error_increase = (
            current_metrics.calibration_error - baseline_metrics.calibration_error
        )
        drift_score = abs(error_increase)

        # Determine severity
        severity = self._classify_drift_severity(drift_score)

        if severity == CalibrationDriftSeverity.NONE:
            return None  # No significant drift

        # Generate recommendations
        recommendations = self._generate_drift_recommendations(
            current_metrics, baseline_metrics, severity
        )

        # Identify affected categories
        affected_categories = self._identify_affected_categories(drift_score)

        return CalibrationDriftAlert(
            severity=severity,
            drift_score=drift_score,
            affected_categories=affected_categories,
            detection_timestamp=datetime.utcnow(),
            recommended_actions=recommendations,
            current_calibration_error=current_metrics.calibration_error,
            baseline_calibration_error=baseline_metrics.calibration_error,
            error_increase=error_increase,
        )

    def apply_calibration_correction(
        self, prediction: Prediction, category: Optional[str] = None
    ) -> Prediction:
        """Apply calibration correction to a prediction."""
        if prediction.result.binary_probability is None:
            return prediction  # Can't correct non-binary predictions

        # Get calibration correction factor
        correction_factor = self._get_calibration_correction_factor(
            prediction.result.binary_probability, category
        )

        # Apply correction
        corrected_prob = self._apply_correction_factor(
            prediction.result.binary_probability, correction_factor
        )

        # Create corrected prediction
        corrected_prediction = Prediction.create_binary_prediction(
            question_id=prediction.question_id,
            research_report_id=prediction.research_report_id,
            probability=corrected_prob,
            confidence=prediction.confidence,
            method=prediction.method,
            reasoning=f"Calibration-corrected: {prediction.reasoning}",
            created_by=f"{prediction.created_by}_calibrated",
            reasoning_steps=prediction.reasoning_steps
            + ["Applied calibration correction"],
            method_metadata={
                **prediction.method_metadata,
                "calibration_correction_applied": True,
                "original_probability": prediction.result.binary_probability,
                "correction_factor": correction_factor,
            },
        )

        return corrected_prediction

    def get_calibration_report(
        self, time_window_days: int = 30, include_categories: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive calibration report."""
        if not self.calibration_history:
            return {"error": "No calibration data available"}

        # Get recent metrics
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
        recent_metrics = [
            m
            for m in self.calibration_history
            if m.measurement_timestamp >= cutoff_date
        ]

        if not recent_metrics:
            return {"error": "No recent calibration data available"}

        # Calculate summary statistics
        summary = self._calculate_calibration_summary(recent_metrics)

        # Analyze trends
        trends = self._analyze_calibration_trends(recent_metrics)

        # Category analysis
        category_analysis = {}
        if include_categories:
            category_analysis = self._analyze_category_calibration(time_window_days)

        # Drift analysis
        drift_analysis = self._analyze_drift_patterns(recent_metrics)

        return {
            "summary": summary,
            "trends": trends,
            "category_analysis": category_analysis,
            "drift_analysis": drift_analysis,
            "recommendations": self._generate_calibration_recommendations(
                recent_metrics
            ),
            "report_timestamp": datetime.utcnow().isoformat(),
            "time_window_days": time_window_days,
        }

    def update_calibration_thresholds(self, performance_data: Dict[str, float]) -> None:
        """Update calibration drift thresholds based on performance."""
        # Adjust thresholds based on overall system performance
        accuracy = performance_data.get("accuracy", 0.7)

        if accuracy > 0.8:
            # High accuracy - can be more sensitive to drift
            scale_factor = 0.8
        elif accuracy < 0.6:
            # Low accuracy - be less sensitive to avoid false alarms
            scale_factor = 1.2
        else:
            scale_factor = 1.0

        for severity in self.drift_thresholds:
            self.drift_thresholds[severity] *= scale_factor

    def _calculate_brier_score(
        self, predicted_probs: List[float], actual_outcomes: List[int]
    ) -> float:
        """Calculate Brier score."""
        if not predicted_probs or not actual_outcomes:
            return 1.0  # Worst possible score

        return statistics.mean(
            [
                (pred - actual) ** 2
                for pred, actual in zip(predicted_probs, actual_outcomes)
            ]
        )

    def _create_calibration_bins(
        self, predicted_probs: List[float], actual_outcomes: List[int]
    ) -> List[CalibrationBin]:
        """Create calibration bins for analysis."""
        bins = []
        bin_width = 1.0 / self.num_bins

        # Create bins
        for i in range(self.num_bins):
            lower = i * bin_width
            upper = (i + 1) * bin_width
            bins.append(CalibrationBin((lower, upper)))

        # Assign predictions to bins
        for pred_prob, outcome in zip(predicted_probs, actual_outcomes):
            bin_index = min(int(pred_prob * self.num_bins), self.num_bins - 1)
            bins[bin_index].add_prediction(pred_prob, outcome)

        return bins

    def _calculate_calibration_error(
        self, calibration_bins: List[CalibrationBin]
    ) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        total_samples = sum(bin.count for bin in calibration_bins)

        if total_samples == 0:
            return 1.0

        weighted_error = 0.0
        for bin in calibration_bins:
            if bin.count >= self.min_samples_per_bin:
                bin_error = abs(
                    bin.average_predicted_probability - bin.observed_frequency
                )
                weight = bin.count / total_samples
                weighted_error += weight * bin_error

        return weighted_error

    def _calculate_resolution(
        self, calibration_bins: List[CalibrationBin], actual_outcomes: List[int]
    ) -> float:
        """Calculate resolution component of Brier score decomposition."""
        if not actual_outcomes:
            return 0.0

        overall_base_rate = statistics.mean(actual_outcomes)
        total_samples = len(actual_outcomes)

        resolution = 0.0
        for bin in calibration_bins:
            if bin.count >= self.min_samples_per_bin:
                weight = bin.count / total_samples
                bin_base_rate = bin.observed_frequency
                resolution += weight * (bin_base_rate - overall_base_rate) ** 2

        return resolution

    def _calculate_uncertainty(self, actual_outcomes: List[int]) -> float:
        """Calculate uncertainty component of Brier score decomposition."""
        if not actual_outcomes:
            return 0.0

        base_rate = statistics.mean(actual_outcomes)
        return base_rate * (1 - base_rate)

    def _calculate_sharpness(self, predicted_probs: List[float]) -> float:
        """Calculate sharpness (how far predictions are from 0.5)."""
        if not predicted_probs:
            return 0.0

        return statistics.mean([abs(prob - 0.5) for prob in predicted_probs])

    def _detect_calibration_drift(
        self, current_calibration_error: float, category: Optional[str]
    ) -> Tuple[CalibrationDriftSeverity, float]:
        """Detect calibration drift severity."""
        baseline_error = self._get_baseline_error(category)

        if baseline_error is None:
            return CalibrationDriftSeverity.NONE, 0.0

        drift_score = abs(current_calibration_error - baseline_error)
        severity = self._classify_drift_severity(drift_score)

        return severity, drift_score

    def _get_baseline_calibration(
        self, comparison_window_days: int, category: Optional[str]
    ) -> Optional[CalibrationMetrics]:
        """Get baseline calibration for comparison."""
        if category and category in self.category_calibration:
            history = self.category_calibration[category]
        else:
            history = self.calibration_history

        if len(history) < 2:
            return None

        # Use metrics from comparison window as baseline
        cutoff_date = datetime.utcnow() - timedelta(days=comparison_window_days)
        baseline_metrics = [m for m in history if m.measurement_timestamp < cutoff_date]

        if not baseline_metrics:
            return history[0]  # Use oldest available

        # Return most recent baseline metric
        return max(baseline_metrics, key=lambda m: m.measurement_timestamp)

    def _get_baseline_error(self, category: Optional[str]) -> Optional[float]:
        """Get baseline calibration error."""
        if self.baseline_calibration is None:
            return None

        if category and category in self.category_calibration:
            category_history = self.category_calibration[category]
            if category_history:
                return category_history[0].calibration_error

        return self.baseline_calibration.calibration_error

    def _classify_drift_severity(self, drift_score: float) -> CalibrationDriftSeverity:
        """Classify drift severity based on score."""
        if drift_score >= self.drift_thresholds[CalibrationDriftSeverity.CRITICAL]:
            return CalibrationDriftSeverity.CRITICAL
        elif drift_score >= self.drift_thresholds[CalibrationDriftSeverity.SEVERE]:
            return CalibrationDriftSeverity.SEVERE
        elif drift_score >= self.drift_thresholds[CalibrationDriftSeverity.MODERATE]:
            return CalibrationDriftSeverity.MODERATE
        elif drift_score >= self.drift_thresholds[CalibrationDriftSeverity.MILD]:
            return CalibrationDriftSeverity.MILD
        else:
            return CalibrationDriftSeverity.NONE

    def _generate_drift_recommendations(
        self,
        current_metrics: CalibrationMetrics,
        baseline_metrics: CalibrationMetrics,
        severity: CalibrationDriftSeverity,
    ) -> List[str]:
        """Generate recommendations for addressing calibration drift."""
        recommendations = []

        if severity in [
            CalibrationDriftSeverity.SEVERE,
            CalibrationDriftSeverity.CRITICAL,
        ]:
            recommendations.append("Immediate recalibration required")
            recommendations.append("Review recent prediction methodology changes")

        if current_metrics.calibration_error > baseline_metrics.calibration_error:
            recommendations.append("Apply conservative confidence adjustments")
            recommendations.append("Increase validation requirements for predictions")
        else:
            recommendations.append("Consider more aggressive confidence levels")

        if current_metrics.sharpness < baseline_metrics.sharpness:
            recommendations.append(
                "Predictions may be too conservative - review confidence thresholds"
            )

        if severity in [
            CalibrationDriftSeverity.MODERATE,
            CalibrationDriftSeverity.SEVERE,
            CalibrationDriftSeverity.CRITICAL,
        ]:
            recommendations.append(
                "Conduct detailed analysis of recent prediction errors"
            )
            recommendations.append("Consider retraining or updating prediction models")

        return recommendations

    def _identify_affected_categories(self, drift_score: float) -> List[str]:
        """Identify categories most affected by calibration drift."""
        affected = []

        for category, metrics_list in self.category_calibration.items():
            if not metrics_list:
                continue

            recent_metric = metrics_list[-1]
            if recent_metric.drift_score >= drift_score * 0.8:  # 80% of overall drift
                affected.append(category)

        return affected

    def _get_calibration_correction_factor(
        self, predicted_probability: float, category: Optional[str]
    ) -> float:
        """Get calibration correction factor for a prediction."""
        # Find appropriate calibration bin
        bin_index = min(int(predicted_probability * self.num_bins), self.num_bins - 1)

        # Get recent calibration metrics
        recent_metrics = self._get_recent_calibration_metrics(category)

        if not recent_metrics or not recent_metrics.calibration_bins:
            return 1.0  # No correction

        calibration_bin = recent_metrics.calibration_bins[bin_index]

        if calibration_bin.count < self.min_samples_per_bin:
            return 1.0  # Not enough data for correction

        # Calculate correction factor
        observed_freq = calibration_bin.observed_frequency
        predicted_freq = calibration_bin.average_predicted_probability

        if predicted_freq == 0:
            return 1.0

        return observed_freq / predicted_freq

    def _apply_correction_factor(
        self, original_probability: float, correction_factor: float
    ) -> float:
        """Apply correction factor to probability."""
        corrected = original_probability * correction_factor
        return max(0.01, min(0.99, corrected))  # Clamp to valid range

    def _get_recent_calibration_metrics(
        self, category: Optional[str]
    ) -> Optional[CalibrationMetrics]:
        """Get most recent calibration metrics."""
        if category and category in self.category_calibration:
            history = self.category_calibration[category]
            return history[-1] if history else None
        else:
            return self.calibration_history[-1] if self.calibration_history else None

    def _calculate_calibration_summary(
        self, recent_metrics: List[CalibrationMetrics]
    ) -> Dict[str, float]:
        """Calculate summary statistics for recent calibration metrics."""
        if not recent_metrics:
            return {}

        return {
            "average_brier_score": statistics.mean(
                [m.brier_score for m in recent_metrics]
            ),
            "average_calibration_error": statistics.mean(
                [m.calibration_error for m in recent_metrics]
            ),
            "average_resolution": statistics.mean(
                [m.resolution for m in recent_metrics]
            ),
            "average_sharpness": statistics.mean([m.sharpness for m in recent_metrics]),
            "total_predictions": sum(
                sum(bin.count for bin in m.calibration_bins) for m in recent_metrics
            ),
        }

    def _analyze_calibration_trends(
        self, recent_metrics: List[CalibrationMetrics]
    ) -> Dict[str, Any]:
        """Analyze calibration trends over time."""
        if len(recent_metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Sort by timestamp
        sorted_metrics = sorted(recent_metrics, key=lambda m: m.measurement_timestamp)

        # Calculate trends
        brier_scores = [m.brier_score for m in sorted_metrics]
        calibration_errors = [m.calibration_error for m in sorted_metrics]

        return {
            "brier_score_trend": self._calculate_trend(brier_scores),
            "calibration_error_trend": self._calculate_trend(calibration_errors),
            "drift_severity_progression": [
                m.drift_severity.value for m in sorted_metrics
            ],
            "measurements_count": len(sorted_metrics),
        }

    def _analyze_category_calibration(self, time_window_days: int) -> Dict[str, Any]:
        """Analyze calibration by category."""
        category_analysis = {}
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)

        for category, metrics_list in self.category_calibration.items():
            recent_metrics = [
                m for m in metrics_list if m.measurement_timestamp >= cutoff_date
            ]

            if recent_metrics:
                category_analysis[category] = {
                    "average_calibration_error": statistics.mean(
                        [m.calibration_error for m in recent_metrics]
                    ),
                    "measurements_count": len(recent_metrics),
                    "latest_drift_severity": recent_metrics[-1].drift_severity.value,
                }

        return category_analysis

    def _analyze_drift_patterns(
        self, recent_metrics: List[CalibrationMetrics]
    ) -> Dict[str, Any]:
        """Analyze drift patterns in recent metrics."""
        if not recent_metrics:
            return {}

        drift_scores = [m.drift_score for m in recent_metrics]
        drift_severities = [m.drift_severity for m in recent_metrics]

        # Count severity occurrences
        severity_counts = {}
        for severity in drift_severities:
            severity_counts[severity.value] = severity_counts.get(severity.value, 0) + 1

        return {
            "average_drift_score": statistics.mean(drift_scores),
            "max_drift_score": max(drift_scores),
            "severity_distribution": severity_counts,
            "drift_trend": self._calculate_trend(drift_scores),
        }

    def _generate_calibration_recommendations(
        self, recent_metrics: List[CalibrationMetrics]
    ) -> List[str]:
        """Generate recommendations based on calibration analysis."""
        recommendations = []

        if not recent_metrics:
            return ["Insufficient calibration data for recommendations"]

        avg_error = statistics.mean([m.calibration_error for m in recent_metrics])
        avg_sharpness = statistics.mean([m.sharpness for m in recent_metrics])

        if avg_error > 0.1:
            recommendations.append(
                "High calibration error detected - consider recalibration"
            )

        if avg_sharpness < 0.1:
            recommendations.append(
                "Low prediction sharpness - consider more confident predictions"
            )

        # Check for consistent drift
        drift_severities = [m.drift_severity for m in recent_metrics]
        severe_drift_count = sum(
            1
            for s in drift_severities
            if s in [CalibrationDriftSeverity.SEVERE, CalibrationDriftSeverity.CRITICAL]
        )

        if severe_drift_count > len(recent_metrics) * 0.3:
            recommendations.append(
                "Persistent severe drift - comprehensive model review needed"
            )

        return recommendations

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear trend
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
