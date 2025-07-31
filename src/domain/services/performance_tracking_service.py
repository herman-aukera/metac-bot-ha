"""
Comprehensive performance tracking service for tournament forecasting.

This service provides detailed metrics logging, reasoning trace preservation,
tournament-specific analytics, and real-time performance monitoring.
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from uuid import UUID, uuid4
import statistics
import threading
from collections import defaultdict, deque

from ..entities.forecast import Forecast
from ..entities.prediction import Prediction
from ..value_objects.reasoning_trace import ReasoningTrace
from ...infrastructure.logging.reasoning_logger import get_reasoning_logger


class MetricType(Enum):
    """Types of performance metrics."""
    ACCURACY = "accuracy"
    CALIBRATION = "calibration"
    BRIER_SCORE = "brier_score"
    LOG_SCORE = "log_score"
    CONFIDENCE = "confidence"
    REASONING_QUALITY = "reasoning_quality"
    TOURNAMENT_RANKING = "tournament_ranking"
    RESPONSE_TIME = "response_time"
    RESOURCE_USAGE = "resource_usage"
    COMPETITIVE_POSITION = "competitive_position"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    id: UUID
    metric_type: MetricType
    value: float
    timestamp: datetime
    question_id: Optional[UUID] = None
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        metric_type: MetricType,
        value: float,
        question_id: Optional[UUID] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "PerformanceMetric":
        """Factory method to create a performance metric."""
        return cls(
            id=uuid4(),
            metric_type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            question_id=question_id,
            agent_id=agent_id,
            metadata=metadata or {}
        )


@dataclass
class TournamentAnalytics:
    """Tournament-specific analytics data."""
    tournament_id: Optional[int]
    current_ranking: Optional[int]
    total_participants: Optional[int]
    questions_answered: int
    questions_resolved: int
    average_brier_score: Optional[float]
    calibration_score: Optional[float]
    competitive_position_percentile: Optional[float]
    market_inefficiencies_detected: int
    strategic_opportunities: List[Dict[str, Any]]
    timestamp: datetime

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get tournament performance summary."""
        return {
            "ranking": self.current_ranking,
            "percentile": self.competitive_position_percentile,
            "questions_answered": self.questions_answered,
            "questions_resolved": self.questions_resolved,
            "average_brier_score": self.average_brier_score,
            "calibration_score": self.calibration_score,
            "opportunities_count": len(self.strategic_opportunities)
        }


@dataclass
class PerformanceAlert:
    """Performance monitoring alert."""
    id: UUID
    level: AlertLevel
    message: str
    metric_type: MetricType
    threshold_value: float
    actual_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        level: AlertLevel,
        message: str,
        metric_type: MetricType,
        threshold_value: float,
        actual_value: float
    ) -> "PerformanceAlert":
        """Factory method to create a performance alert."""
        return cls(
            id=uuid4(),
            level=level,
            message=message,
            metric_type=metric_type,
            threshold_value=threshold_value,
            actual_value=actual_value,
            timestamp=datetime.utcnow()
        )


class PerformanceTrackingService:
    """
    Comprehensive performance tracking service for tournament forecasting.

    Provides detailed metrics logging, reasoning trace preservation,
    tournament-specific analytics, and real-time performance monitoring.
    """

    def __init__(
        self,
        metrics_storage_path: Optional[Path] = None,
        enable_real_time_monitoring: bool = True,
        alert_thresholds: Optional[Dict[MetricType, Dict[str, float]]] = None
    ):
        """
        Initialize the performance tracking service.

        Args:
            metrics_storage_path: Path to store metrics data
            enable_real_time_monitoring: Enable real-time monitoring and alerting
            alert_thresholds: Custom alert thresholds for different metrics
        """
        self.logger = logging.getLogger(__name__)
        self.reasoning_logger = get_reasoning_logger()

        # Storage configuration
        if metrics_storage_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            metrics_storage_path = project_root / "logs" / "performance"

        self.metrics_storage_path = Path(metrics_storage_path)
        self.metrics_storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory storage for real-time monitoring
        self.metrics_buffer: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.alerts_buffer: deque = deque(maxlen=1000)    # Keep last 1k alerts
        self.tournament_analytics: Dict[int, TournamentAnalytics] = {}

        # Real-time monitoring
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.alert_thresholds = alert_thresholds or self._get_default_alert_thresholds()
        self._monitoring_lock = threading.Lock()

        # Performance aggregations
        self.agent_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.question_performance: Dict[UUID, Dict[str, Any]] = {}
        self.tournament_performance: Dict[int, Dict[str, Any]] = defaultdict(dict)

        self.logger.info(f"Performance tracking service initialized with storage at {self.metrics_storage_path}")

    def _get_default_alert_thresholds(self) -> Dict[MetricType, Dict[str, float]]:
        """Get default alert thresholds for different metrics."""
        return {
            MetricType.BRIER_SCORE: {
                "warning": 0.3,
                "error": 0.4,
                "critical": 0.5
            },
            MetricType.CALIBRATION: {
                "warning": 0.1,
                "error": 0.15,
                "critical": 0.2
            },
            MetricType.CONFIDENCE: {
                "warning": 0.4,
                "error": 0.3,
                "critical": 0.2
            },
            MetricType.RESPONSE_TIME: {
                "warning": 300.0,  # 5 minutes
                "error": 600.0,    # 10 minutes
                "critical": 1200.0  # 20 minutes
            },
            MetricType.TOURNAMENT_RANKING: {
                "warning": 0.7,    # Below 70th percentile
                "error": 0.5,      # Below 50th percentile
                "critical": 0.3    # Below 30th percentile
            }
        }

    def track_forecast_performance(
        self,
        forecast: Forecast,
        processing_time: Optional[float] = None,
        resource_usage: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Track comprehensive performance metrics for a forecast.

        Args:
            forecast: The forecast to track
            processing_time: Time taken to generate forecast (seconds)
            resource_usage: Resource usage metrics (CPU, memory, etc.)
        """
        try:
            timestamp = datetime.utcnow()

            # Track basic forecast metrics
            self._track_forecast_metrics(forecast, timestamp)

            # Track reasoning quality
            self._track_reasoning_quality(forecast, timestamp)

            # Track processing performance
            if processing_time is not None:
                self._track_processing_time(forecast, processing_time, timestamp)

            # Track resource usage
            if resource_usage:
                self._track_resource_usage(forecast, resource_usage, timestamp)

            # Track tournament-specific metrics
            self._track_tournament_metrics(forecast, timestamp)

            # Preserve reasoning traces
            self._preserve_reasoning_traces(forecast)

            # Update aggregated performance data
            self._update_performance_aggregations(forecast)

            # Check for alerts if real-time monitoring is enabled
            if self.enable_real_time_monitoring:
                self._check_performance_alerts(forecast)

            self.logger.debug(f"Tracked performance for forecast {forecast.id}")

        except Exception as e:
            self.logger.error(f"Error tracking forecast performance: {e}")

    def _track_forecast_metrics(self, forecast: Forecast, timestamp: datetime) -> None:
        """Track basic forecast metrics."""
        # Confidence score
        confidence_metric = PerformanceMetric.create(
            MetricType.CONFIDENCE,
            forecast.confidence_score,
            question_id=forecast.question_id,
            agent_id=forecast.ensemble_method,
            metadata={
                "forecast_id": str(forecast.id),
                "prediction_variance": forecast.calculate_prediction_variance(),
                "consensus_strength": forecast.consensus_strength
            }
        )
        self._store_metric(confidence_metric)

        # Prediction variance (ensemble disagreement)
        variance_metric = PerformanceMetric.create(
            MetricType.ACCURACY,  # Using accuracy type for variance tracking
            forecast.calculate_prediction_variance(),
            question_id=forecast.question_id,
            agent_id=forecast.ensemble_method,
            metadata={
                "forecast_id": str(forecast.id),
                "metric_subtype": "prediction_variance",
                "ensemble_method": forecast.ensemble_method
            }
        )
        self._store_metric(variance_metric)

    def _track_reasoning_quality(self, forecast: Forecast, timestamp: datetime) -> None:
        """Track reasoning quality metrics."""
        reasoning_quality_score = 0.5  # Default

        # Calculate reasoning quality from predictions
        if forecast.predictions:
            quality_scores = []
            for prediction in forecast.predictions:
                if hasattr(prediction, 'calculate_prediction_quality_score'):
                    quality_scores.append(prediction.calculate_prediction_quality_score())
                else:
                    # Basic quality assessment
                    base_quality = 0.3
                    if len(prediction.reasoning) > 100:
                        base_quality += 0.2
                    if len(prediction.reasoning_steps) > 2:
                        base_quality += 0.2
                    if prediction.reasoning_trace:
                        base_quality += 0.3
                    quality_scores.append(min(1.0, base_quality))

            if quality_scores:
                reasoning_quality_score = statistics.mean(quality_scores)

        # Store reasoning quality metric
        quality_metric = PerformanceMetric.create(
            MetricType.REASONING_QUALITY,
            reasoning_quality_score,
            question_id=forecast.question_id,
            agent_id=forecast.ensemble_method,
            metadata={
                "forecast_id": str(forecast.id),
                "predictions_count": len(forecast.predictions),
                "has_reasoning_traces": any(p.reasoning_trace for p in forecast.predictions),
                "average_reasoning_length": statistics.mean([len(p.reasoning) for p in forecast.predictions]) if forecast.predictions else 0
            }
        )
        self._store_metric(quality_metric)

    def _track_processing_time(self, forecast: Forecast, processing_time: float, timestamp: datetime) -> None:
        """Track processing time metrics."""
        time_metric = PerformanceMetric.create(
            MetricType.RESPONSE_TIME,
            processing_time,
            question_id=forecast.question_id,
            agent_id=forecast.ensemble_method,
            metadata={
                "forecast_id": str(forecast.id),
                "predictions_count": len(forecast.predictions),
                "research_reports_count": len(forecast.research_reports)
            }
        )
        self._store_metric(time_metric)

    def _track_resource_usage(self, forecast: Forecast, resource_usage: Dict[str, float], timestamp: datetime) -> None:
        """Track resource usage metrics."""
        for resource_type, usage_value in resource_usage.items():
            resource_metric = PerformanceMetric.create(
                MetricType.RESOURCE_USAGE,
                usage_value,
                question_id=forecast.question_id,
                agent_id=forecast.ensemble_method,
                metadata={
                    "forecast_id": str(forecast.id),
                    "resource_type": resource_type
                }
            )
            self._store_metric(resource_metric)

    def _track_tournament_metrics(self, forecast: Forecast, timestamp: datetime) -> None:
        """Track tournament-specific metrics."""
        if forecast.tournament_strategy:
            # Track competitive positioning
            if forecast.competitive_intelligence:
                position_metric = PerformanceMetric.create(
                    MetricType.COMPETITIVE_POSITION,
                    forecast.competitive_intelligence.market_position_percentile or 0.5,
                    question_id=forecast.question_id,
                    agent_id=forecast.ensemble_method,
                    metadata={
                        "forecast_id": str(forecast.id),
                        "tournament_strategy": forecast.tournament_strategy.strategy_name,
                        "question_priority": forecast.question_priority.get_overall_priority_score() if forecast.question_priority else 0.5
                    }
                )
                self._store_metric(position_metric)

    def _preserve_reasoning_traces(self, forecast: Forecast) -> None:
        """Preserve detailed reasoning traces for transparency."""
        try:
            # Log individual prediction reasoning traces
            for prediction in forecast.predictions:
                # Always log basic reasoning data
                reasoning_data = {
                    "reasoning": prediction.reasoning,
                    "reasoning_steps": prediction.reasoning_steps,
                    "confidence_analysis": f"Confidence: {prediction.confidence.value}",
                    "method": prediction.method.value
                }

                # Add detailed reasoning trace if available
                if prediction.reasoning_trace:
                    reasoning_data["reasoning_trace"] = {
                        "steps": [
                            {
                                "type": step.step_type.value,
                                "content": step.content,
                                "confidence": step.confidence,
                                "timestamp": step.timestamp.isoformat()
                            }
                            for step in prediction.reasoning_trace.steps
                        ],
                        "final_conclusion": prediction.reasoning_trace.final_conclusion,
                        "overall_confidence": prediction.reasoning_trace.overall_confidence,
                        "bias_checks": prediction.reasoning_trace.bias_checks,
                        "uncertainty_sources": prediction.reasoning_trace.uncertainty_sources
                    }

                prediction_result = {
                    "probability": prediction.result.binary_probability,
                    "confidence": prediction.get_confidence_score(),
                    "method": prediction.method.value
                }

                self.reasoning_logger.log_reasoning_trace(
                    question_id=forecast.question_id,
                    agent_name=prediction.created_by,
                    reasoning_data=reasoning_data,
                    prediction_result=prediction_result
                )

            # Log ensemble reasoning trace
            ensemble_reasoning_data = {
                "reasoning": forecast.reasoning_summary,
                "ensemble_method": forecast.ensemble_method,
                "weight_distribution": forecast.weight_distribution,
                "consensus_strength": forecast.consensus_strength,
                "prediction_variance": forecast.calculate_prediction_variance(),
                "individual_predictions": [
                    {
                        "agent": pred.created_by,
                        "prediction": pred.result.binary_probability,
                        "confidence": pred.get_confidence_score(),
                        "method": pred.method.value
                    }
                    for pred in forecast.predictions
                ]
            }

            ensemble_result = {
                "probability": forecast.prediction,
                "confidence": forecast.confidence_score,
                "method": "ensemble"
            }

            self.reasoning_logger.log_reasoning_trace(
                question_id=forecast.question_id,
                agent_name="ensemble",
                reasoning_data=ensemble_reasoning_data,
                prediction_result=ensemble_result
            )

        except Exception as e:
            self.logger.error(f"Error preserving reasoning traces: {e}")

    def _update_performance_aggregations(self, forecast: Forecast) -> None:
        """Update aggregated performance data."""
        with self._monitoring_lock:
            # Update agent performance
            for prediction in forecast.predictions:
                agent_id = prediction.created_by
                self.agent_performance[agent_id]["confidence"].append(prediction.get_confidence_score())
                if prediction.result.binary_probability is not None:
                    self.agent_performance[agent_id]["predictions"].append(prediction.result.binary_probability)

            # Update question performance
            self.question_performance[forecast.question_id] = {
                "forecast_id": forecast.id,
                "confidence_score": forecast.confidence_score,
                "prediction_variance": forecast.calculate_prediction_variance(),
                "consensus_strength": forecast.consensus_strength,
                "predictions_count": len(forecast.predictions),
                "timestamp": datetime.utcnow()
            }

    def _check_performance_alerts(self, forecast: Forecast) -> None:
        """Check for performance alerts based on thresholds."""
        try:
            # Check confidence threshold
            if forecast.confidence_score < self.alert_thresholds[MetricType.CONFIDENCE]["critical"]:
                alert = PerformanceAlert.create(
                    AlertLevel.CRITICAL,
                    f"Very low confidence score: {forecast.confidence_score:.3f}",
                    MetricType.CONFIDENCE,
                    self.alert_thresholds[MetricType.CONFIDENCE]["critical"],
                    forecast.confidence_score
                )
                self._store_alert(alert)

            # Check prediction variance (ensemble disagreement)
            variance = forecast.calculate_prediction_variance()
            if variance > 0.15:  # High disagreement threshold
                alert = PerformanceAlert.create(
                    AlertLevel.WARNING,
                    f"High ensemble disagreement (variance: {variance:.3f})",
                    MetricType.ACCURACY,
                    0.15,
                    variance
                )
                self._store_alert(alert)

        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")

    def _store_metric(self, metric: PerformanceMetric) -> None:
        """Store a performance metric."""
        with self._monitoring_lock:
            self.metrics_buffer.append(metric)

        # Persist to disk periodically
        self._persist_metrics_if_needed()

    def _store_alert(self, alert: PerformanceAlert) -> None:
        """Store a performance alert."""
        with self._monitoring_lock:
            self.alerts_buffer.append(alert)

        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }[alert.level]

        self.logger.log(log_level, f"Performance Alert: {alert.message}")

    def _persist_metrics_if_needed(self) -> None:
        """Persist metrics to disk if buffer is getting full."""
        if len(self.metrics_buffer) > 8000:  # Persist when 80% full
            self._persist_metrics()

    def _persist_metrics(self) -> None:
        """Persist metrics buffer to disk."""
        try:
            timestamp = datetime.utcnow()
            filename = f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.metrics_storage_path / filename

            with self._monitoring_lock:
                metrics_data = [asdict(metric) for metric in list(self.metrics_buffer)]
                self.metrics_buffer.clear()

            # Convert UUIDs and datetime objects to strings for JSON serialization
            for metric_data in metrics_data:
                metric_data["id"] = str(metric_data["id"])
                if metric_data["question_id"]:
                    metric_data["question_id"] = str(metric_data["question_id"])
                metric_data["timestamp"] = metric_data["timestamp"].isoformat()
                metric_data["metric_type"] = metric_data["metric_type"].value

            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)

            self.logger.debug(f"Persisted {len(metrics_data)} metrics to {filepath}")

        except Exception as e:
            self.logger.error(f"Error persisting metrics: {e}")

    def track_resolved_prediction(
        self,
        forecast: Forecast,
        actual_outcome: int,
        resolution_timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Track performance metrics for a resolved prediction.

        Args:
            forecast: The original forecast
            actual_outcome: The actual outcome (0 or 1 for binary questions)
            resolution_timestamp: When the question was resolved

        Returns:
            Dictionary of calculated performance metrics
        """
        try:
            if resolution_timestamp is None:
                resolution_timestamp = datetime.utcnow()

            prediction_prob = forecast.prediction

            # Calculate Brier score
            brier_score = (prediction_prob - actual_outcome) ** 2

            # Calculate log score (avoiding log(0))
            epsilon = 1e-15
            prob_clamped = max(epsilon, min(1 - epsilon, prediction_prob))
            if actual_outcome == 1:
                log_score = -math.log(prob_clamped)
            else:
                log_score = -math.log(1 - prob_clamped)

            # Calculate calibration contribution
            calibration_bin = int(prediction_prob * 10) / 10  # 0.1 bins

            # Store performance metrics
            metrics = {
                "brier_score": brier_score,
                "log_score": log_score,
                "accuracy": 1.0 if (prediction_prob > 0.5) == (actual_outcome == 1) else 0.0,
                "calibration_bin": calibration_bin
            }

            # Store metrics
            for metric_name, metric_value in metrics.items():
                if metric_name == "brier_score":
                    metric_type = MetricType.BRIER_SCORE
                elif metric_name == "log_score":
                    metric_type = MetricType.LOG_SCORE
                elif metric_name == "accuracy":
                    metric_type = MetricType.ACCURACY
                else:
                    metric_type = MetricType.CALIBRATION

                metric = PerformanceMetric.create(
                    metric_type,
                    metric_value,
                    question_id=forecast.question_id,
                    agent_id=forecast.ensemble_method,
                    metadata={
                        "forecast_id": str(forecast.id),
                        "actual_outcome": actual_outcome,
                        "predicted_probability": prediction_prob,
                        "resolution_timestamp": resolution_timestamp.isoformat()
                    }
                )
                self._store_metric(metric)

            # Check for performance alerts
            if self.enable_real_time_monitoring:
                self._check_resolved_prediction_alerts(metrics, forecast)

            self.logger.info(f"Tracked resolved prediction for forecast {forecast.id}: Brier={brier_score:.3f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Error tracking resolved prediction: {e}")
            return {}

    def _check_resolved_prediction_alerts(self, metrics: Dict[str, float], forecast: Forecast) -> None:
        """Check for alerts based on resolved prediction performance."""
        brier_score = metrics.get("brier_score", 0.0)

        # Check Brier score thresholds
        thresholds = self.alert_thresholds[MetricType.BRIER_SCORE]
        if brier_score >= thresholds["critical"]:
            alert = PerformanceAlert.create(
                AlertLevel.CRITICAL,
                f"Very poor Brier score: {brier_score:.3f} for forecast {forecast.id}",
                MetricType.BRIER_SCORE,
                thresholds["critical"],
                brier_score
            )
            self._store_alert(alert)
        elif brier_score >= thresholds["error"]:
            alert = PerformanceAlert.create(
                AlertLevel.ERROR,
                f"Poor Brier score: {brier_score:.3f} for forecast {forecast.id}",
                MetricType.BRIER_SCORE,
                thresholds["error"],
                brier_score
            )
            self._store_alert(alert)

    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive performance dashboard data.

        Returns:
            Dictionary containing dashboard metrics and visualizations
        """
        try:
            with self._monitoring_lock:
                current_time = datetime.utcnow()

                # Recent metrics (last 24 hours)
                recent_metrics = [
                    metric for metric in self.metrics_buffer
                    if (current_time - metric.timestamp).total_seconds() < 86400
                ]

                # Recent alerts (last 24 hours)
                recent_alerts = [
                    alert for alert in self.alerts_buffer
                    if (current_time - alert.timestamp).total_seconds() < 86400
                ]

                dashboard_data = {
                    "summary": self._get_performance_summary(recent_metrics),
                    "agent_performance": self._get_agent_performance_summary(),
                    "recent_alerts": self._format_alerts_for_dashboard(recent_alerts),
                    "tournament_analytics": self._get_tournament_analytics_summary(),
                    "real_time_metrics": self._get_real_time_metrics(recent_metrics),
                    "calibration_analysis": self._get_calibration_analysis(recent_metrics),
                    "timestamp": current_time.isoformat()
                }

                return dashboard_data

        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def _get_performance_summary(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not metrics:
            return {"message": "No recent metrics available"}

        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric.value)

        summary = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                summary[metric_type.value] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0
                }

        return summary

    def _get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get agent-specific performance summary."""
        agent_summary = {}

        for agent_id, performance_data in self.agent_performance.items():
            agent_summary[agent_id] = {}

            for metric_name, values in performance_data.items():
                if values:
                    agent_summary[agent_id][metric_name] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "recent_trend": self._calculate_trend(values[-10:]) if len(values) >= 5 else "insufficient_data"
                    }

        return agent_summary

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 3:
            return "insufficient_data"

        # Simple linear trend calculation
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"

    def _format_alerts_for_dashboard(self, alerts: List[PerformanceAlert]) -> List[Dict[str, Any]]:
        """Format alerts for dashboard display."""
        return [
            {
                "id": str(alert.id),
                "level": alert.level.value,
                "message": alert.message,
                "metric_type": alert.metric_type.value,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }
            for alert in alerts[-20:]  # Last 20 alerts
        ]

    def _get_tournament_analytics_summary(self) -> Dict[str, Any]:
        """Get tournament analytics summary."""
        if not self.tournament_analytics:
            return {"message": "No tournament data available"}

        summary = {}
        for tournament_id, analytics in self.tournament_analytics.items():
            summary[str(tournament_id)] = analytics.get_performance_summary()

        return summary

    def _get_real_time_metrics(self, recent_metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Get real-time metrics for monitoring."""
        current_time = datetime.utcnow()

        # Metrics from last hour
        last_hour_metrics = [
            metric for metric in recent_metrics
            if (current_time - metric.timestamp).total_seconds() < 3600
        ]

        if not last_hour_metrics:
            return {"message": "No recent metrics in last hour"}

        # Calculate real-time indicators
        confidence_values = [
            metric.value for metric in last_hour_metrics
            if metric.metric_type == MetricType.CONFIDENCE
        ]

        response_times = [
            metric.value for metric in last_hour_metrics
            if metric.metric_type == MetricType.RESPONSE_TIME
        ]

        real_time_data = {
            "metrics_count_last_hour": len(last_hour_metrics),
            "average_confidence": statistics.mean(confidence_values) if confidence_values else None,
            "average_response_time": statistics.mean(response_times) if response_times else None,
            "active_forecasts": len(set(metric.question_id for metric in last_hour_metrics if metric.question_id)),
            "timestamp": current_time.isoformat()
        }

        return real_time_data

    def _get_calibration_analysis(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Get calibration analysis from recent metrics."""
        # This would be enhanced with actual calibration calculations
        # For now, return basic structure
        return {
            "message": "Calibration analysis requires resolved predictions",
            "bins": {},
            "overall_calibration_error": None
        }

    def update_tournament_analytics(
        self,
        tournament_id: int,
        ranking: Optional[int] = None,
        total_participants: Optional[int] = None,
        brier_scores: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Update tournament-specific analytics.

        Args:
            tournament_id: Tournament identifier
            ranking: Current ranking in tournament
            total_participants: Total number of participants
            brier_scores: Recent Brier scores
            **kwargs: Additional tournament data
        """
        try:
            current_analytics = self.tournament_analytics.get(tournament_id)

            if current_analytics is None:
                # Create new analytics entry
                analytics = TournamentAnalytics(
                    tournament_id=tournament_id,
                    current_ranking=ranking,
                    total_participants=total_participants,
                    questions_answered=kwargs.get("questions_answered", 0),
                    questions_resolved=kwargs.get("questions_resolved", 0),
                    average_brier_score=statistics.mean(brier_scores) if brier_scores else None,
                    calibration_score=kwargs.get("calibration_score"),
                    competitive_position_percentile=ranking / total_participants if ranking and total_participants else None,
                    market_inefficiencies_detected=kwargs.get("market_inefficiencies_detected", 0),
                    strategic_opportunities=kwargs.get("strategic_opportunities", []),
                    timestamp=datetime.utcnow()
                )
            else:
                # Update existing analytics
                analytics = TournamentAnalytics(
                    tournament_id=tournament_id,
                    current_ranking=ranking or current_analytics.current_ranking,
                    total_participants=total_participants or current_analytics.total_participants,
                    questions_answered=kwargs.get("questions_answered", current_analytics.questions_answered),
                    questions_resolved=kwargs.get("questions_resolved", current_analytics.questions_resolved),
                    average_brier_score=statistics.mean(brier_scores) if brier_scores else current_analytics.average_brier_score,
                    calibration_score=kwargs.get("calibration_score", current_analytics.calibration_score),
                    competitive_position_percentile=ranking / total_participants if ranking and total_participants else current_analytics.competitive_position_percentile,
                    market_inefficiencies_detected=kwargs.get("market_inefficiencies_detected", current_analytics.market_inefficiencies_detected),
                    strategic_opportunities=kwargs.get("strategic_opportunities", current_analytics.strategic_opportunities),
                    timestamp=datetime.utcnow()
                )

            self.tournament_analytics[tournament_id] = analytics

            # Track tournament ranking metric
            if ranking and total_participants:
                percentile = ranking / total_participants
                ranking_metric = PerformanceMetric.create(
                    MetricType.TOURNAMENT_RANKING,
                    percentile,
                    metadata={
                        "tournament_id": tournament_id,
                        "ranking": ranking,
                        "total_participants": total_participants
                    }
                )
                self._store_metric(ranking_metric)

            self.logger.info(f"Updated tournament analytics for tournament {tournament_id}")

        except Exception as e:
            self.logger.error(f"Error updating tournament analytics: {e}")

    def get_performance_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        tournament_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            start_date: Start date for report period
            end_date: End date for report period
            agent_id: Specific agent to analyze
            tournament_id: Specific tournament to analyze

        Returns:
            Comprehensive performance report
        """
        try:
            if end_date is None:
                end_date = datetime.utcnow()
            if start_date is None:
                start_date = end_date - timedelta(days=7)  # Last week by default

            # Filter metrics by date range
            with self._monitoring_lock:
                filtered_metrics = [
                    metric for metric in self.metrics_buffer
                    if start_date <= metric.timestamp <= end_date
                ]

                if agent_id:
                    filtered_metrics = [m for m in filtered_metrics if m.agent_id == agent_id]

            report = {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "duration_days": (end_date - start_date).days
                },
                "filters": {
                    "agent_id": agent_id,
                    "tournament_id": tournament_id
                },
                "summary": self._get_performance_summary(filtered_metrics),
                "detailed_analysis": self._get_detailed_performance_analysis(filtered_metrics),
                "recommendations": self._generate_performance_recommendations(filtered_metrics),
                "generated_at": datetime.utcnow().isoformat()
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}

    def _get_detailed_performance_analysis(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Get detailed performance analysis."""
        analysis = {
            "total_metrics": len(metrics),
            "unique_questions": len(set(m.question_id for m in metrics if m.question_id)),
            "unique_agents": len(set(m.agent_id for m in metrics if m.agent_id)),
            "metric_types_distribution": {}
        }

        # Metric types distribution
        type_counts = defaultdict(int)
        for metric in metrics:
            type_counts[metric.metric_type.value] += 1

        analysis["metric_types_distribution"] = dict(type_counts)

        return analysis

    def _generate_performance_recommendations(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Analyze confidence levels
        confidence_metrics = [m for m in metrics if m.metric_type == MetricType.CONFIDENCE]
        if confidence_metrics:
            avg_confidence = statistics.mean([m.value for m in confidence_metrics])
            if avg_confidence < 0.5:
                recommendations.append("Consider improving confidence calibration - average confidence is low")

        # Analyze response times
        time_metrics = [m for m in metrics if m.metric_type == MetricType.RESPONSE_TIME]
        if time_metrics:
            avg_time = statistics.mean([m.value for m in time_metrics])
            if avg_time > 300:  # 5 minutes
                recommendations.append("Consider optimizing processing time - average response time is high")

        # Analyze reasoning quality
        quality_metrics = [m for m in metrics if m.metric_type == MetricType.REASONING_QUALITY]
        if quality_metrics:
            avg_quality = statistics.mean([m.value for m in quality_metrics])
            if avg_quality < 0.6:
                recommendations.append("Consider enhancing reasoning documentation and quality")

        if not recommendations:
            recommendations.append("Performance metrics look good - continue current approach")

        return recommendations

    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Clean up old performance data.

        Args:
            days_to_keep: Number of days of data to retain

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            with self._monitoring_lock:
                # Clean metrics buffer
                original_metrics_count = len(self.metrics_buffer)
                self.metrics_buffer = deque(
                    [m for m in self.metrics_buffer if m.timestamp > cutoff_date],
                    maxlen=self.metrics_buffer.maxlen
                )
                metrics_removed = original_metrics_count - len(self.metrics_buffer)

                # Clean alerts buffer
                original_alerts_count = len(self.alerts_buffer)
                self.alerts_buffer = deque(
                    [a for a in self.alerts_buffer if a.timestamp > cutoff_date],
                    maxlen=self.alerts_buffer.maxlen
                )
                alerts_removed = original_alerts_count - len(self.alerts_buffer)

            # Clean old metric files
            files_removed = 0
            for file_path in self.metrics_storage_path.glob("metrics_*.json"):
                try:
                    # Extract date from filename
                    filename = file_path.stem
                    date_str = filename.split('_')[1] + '_' + filename.split('_')[2]
                    file_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')

                    if file_date < cutoff_date:
                        file_path.unlink()
                        files_removed += 1
                except (ValueError, IndexError):
                    # Skip files with unexpected naming
                    continue

            cleanup_stats = {
                "metrics_removed": metrics_removed,
                "alerts_removed": alerts_removed,
                "files_removed": files_removed,
                "cutoff_date": cutoff_date.isoformat()
            }

            self.logger.info(f"Cleaned up old performance data: {cleanup_stats}")
            return cleanup_stats

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return {"error": str(e)}
