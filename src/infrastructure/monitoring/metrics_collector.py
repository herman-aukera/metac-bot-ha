"""
Metrics collection with Prometheus integration.

Provides comprehensive metrics collection for forecasting performance,
system health, and business metrics with Prometheus integration.
"""

import time
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import statistics

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when Prometheus is not available
    class CollectorRegistry:
        pass

from src.infrastructure.logging.structured_logger import get_logger


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = None
    buckets: List[float] = None  # For histograms


class MetricsCollector:
    """
    Comprehensive metrics collector with Prometheus integration.

    Collects forecasting performance, system health, and business metrics
    with support for Prometheus exposition and local aggregation.
    """

    def __init__(
        self,
        enable_prometheus: bool = True,
        prometheus_port: Optional[int] = None,
        registry: Optional[CollectorRegistry] = None
    ):
        self.logger = get_logger("metrics_collector")
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port
        self.registry = registry or CollectorRegistry()

        # Local metrics storage for when Prometheus is not available
        self.local_metrics: Dict[str, Any] = defaultdict(dict)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Thread safety
        self._lock = threading.Lock()

        # Prometheus metrics
        self._prometheus_metrics: Dict[str, Any] = {}

        # Initialize core metrics
        self._initialize_core_metrics()

        # Start Prometheus server if enabled
        if self.enable_prometheus and prometheus_port:
            self._start_prometheus_server()

    def _initialize_core_metrics(self):
        """Initialize core system and forecasting metrics."""

        # Forecasting performance metrics
        forecasting_metrics = [
            MetricDefinition(
                "forecasting_requests_total",
                "Total number of forecasting requests",
                MetricType.COUNTER,
                ["question_type", "agent_type", "status"]
            ),
            MetricDefinition(
                "forecasting_duration_seconds",
                "Time spent on forecasting operations",
                MetricType.HISTOGRAM,
                ["question_type", "agent_type"],
                [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
            ),
            MetricDefinition(
                "prediction_accuracy",
                "Prediction accuracy scores",
                MetricType.HISTOGRAM,
                ["question_type", "agent_type"],
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            MetricDefinition(
                "prediction_confidence",
                "Prediction confidence levels",
                MetricType.HISTOGRAM,
                ["question_type", "agent_type"],
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            MetricDefinition(
                "ensemble_consensus_strength",
                "Ensemble consensus strength",
                MetricType.GAUGE,
                ["question_id"]
            ),
            MetricDefinition(
                "research_sources_count",
                "Number of research sources used",
                MetricType.HISTOGRAM,
                ["question_type"],
                [1, 2, 5, 10, 20, 50]
            ),
        ]

        # System health metrics
        system_metrics = [
            MetricDefinition(
                "system_health_score",
                "Overall system health score",
                MetricType.GAUGE,
                ["component"]
            ),
            MetricDefinition(
                "api_requests_total",
                "Total API requests to external services",
                MetricType.COUNTER,
                ["service", "endpoint", "status"]
            ),
            MetricDefinition(
                "api_request_duration_seconds",
                "API request duration",
                MetricType.HISTOGRAM,
                ["service", "endpoint"],
                [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            MetricDefinition(
                "circuit_breaker_state",
                "Circuit breaker states",
                MetricType.GAUGE,
                ["service", "state"]
            ),
            MetricDefinition(
                "cache_hit_rate",
                "Cache hit rate",
                MetricType.GAUGE,
                ["cache_type"]
            ),
            MetricDefinition(
                "memory_usage_bytes",
                "Memory usage in bytes",
                MetricType.GAUGE,
                ["component"]
            ),
        ]

        # Business metrics
        business_metrics = [
            MetricDefinition(
                "tournament_questions_processed",
                "Tournament questions processed",
                MetricType.COUNTER,
                ["tournament_id", "question_type"]
            ),
            MetricDefinition(
                "tournament_ranking_position",
                "Current tournament ranking position",
                MetricType.GAUGE,
                ["tournament_id"]
            ),
            MetricDefinition(
                "tournament_score",
                "Current tournament score",
                MetricType.GAUGE,
                ["tournament_id"]
            ),
            MetricDefinition(
                "agent_performance_score",
                "Individual agent performance scores",
                MetricType.GAUGE,
                ["agent_id", "metric_type"]
            ),
        ]

        # Register all metrics
        all_metrics = forecasting_metrics + system_metrics + business_metrics
        for metric_def in all_metrics:
            self._register_metric(metric_def)

    def _register_metric(self, metric_def: MetricDefinition):
        """Register a metric with Prometheus if available."""
        if not self.enable_prometheus:
            return

        labels = metric_def.labels or []

        try:
            if metric_def.metric_type == MetricType.COUNTER:
                metric = Counter(
                    metric_def.name,
                    metric_def.description,
                    labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    metric_def.name,
                    metric_def.description,
                    labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    metric_def.name,
                    metric_def.description,
                    labels,
                    buckets=metric_def.buckets,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.SUMMARY:
                metric = Summary(
                    metric_def.name,
                    metric_def.description,
                    labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.INFO:
                metric = Info(
                    metric_def.name,
                    metric_def.description,
                    registry=self.registry
                )
            else:
                self.logger.warning(f"Unknown metric type: {metric_def.metric_type}")
                return

            self._prometheus_metrics[metric_def.name] = metric
            self.logger.debug(f"Registered Prometheus metric: {metric_def.name}")

        except Exception as e:
            self.logger.error(f"Failed to register metric {metric_def.name}", exception=e)

    def _start_prometheus_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.prometheus_port, registry=self.registry)
            self.logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            self.logger.error("Failed to start Prometheus server", exception=e)

    def increment_counter(
        self,
        metric_name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric."""
        with self._lock:
            # Prometheus
            if self.enable_prometheus and metric_name in self._prometheus_metrics:
                try:
                    metric = self._prometheus_metrics[metric_name]
                    if labels:
                        metric.labels(**labels).inc(value)
                    else:
                        metric.inc(value)
                except Exception as e:
                    self.logger.error(f"Failed to increment Prometheus counter {metric_name}", exception=e)

            # Local storage
            key = self._make_metric_key(metric_name, labels)
            if key not in self.local_metrics:
                self.local_metrics[key] = 0.0
            self.local_metrics[key] = self.local_metrics[key] + value
            self._record_metric_history(key, self.local_metrics[key])

    def set_gauge(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric value."""
        with self._lock:
            # Prometheus
            if self.enable_prometheus and metric_name in self._prometheus_metrics:
                try:
                    metric = self._prometheus_metrics[metric_name]
                    if labels:
                        metric.labels(**labels).set(value)
                    else:
                        metric.set(value)
                except Exception as e:
                    self.logger.error(f"Failed to set Prometheus gauge {metric_name}", exception=e)

            # Local storage
            key = self._make_metric_key(metric_name, labels)
            self.local_metrics[key] = value
            self._record_metric_history(key, value)

    def observe_histogram(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Observe a value for a histogram metric."""
        with self._lock:
            # Prometheus
            if self.enable_prometheus and metric_name in self._prometheus_metrics:
                try:
                    metric = self._prometheus_metrics[metric_name]
                    if labels:
                        metric.labels(**labels).observe(value)
                    else:
                        metric.observe(value)
                except Exception as e:
                    self.logger.error(f"Failed to observe Prometheus histogram {metric_name}", exception=e)

            # Local storage (maintain simple statistics)
            key = self._make_metric_key(metric_name, labels)
            if key not in self.local_metrics:
                self.local_metrics[key] = {"count": 0, "sum": 0.0, "values": deque(maxlen=100)}

            self.local_metrics[key]["count"] += 1
            self.local_metrics[key]["sum"] += value
            self.local_metrics[key]["values"].append(value)
            self._record_metric_history(key, value)

    def time_operation(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimingContext(self, metric_name, labels)

    def record_forecasting_request(
        self,
        question_type: str,
        agent_type: str,
        status: str,
        duration: float,
        accuracy: Optional[float] = None,
        confidence: Optional[float] = None
    ):
        """Record forecasting request metrics."""
        labels = {
            "question_type": question_type,
            "agent_type": agent_type,
            "status": status
        }

        self.increment_counter("forecasting_requests_total", labels=labels)
        self.observe_histogram("forecasting_duration_seconds", duration, labels={
            "question_type": question_type,
            "agent_type": agent_type
        })

        if accuracy is not None:
            self.observe_histogram("prediction_accuracy", accuracy, labels={
                "question_type": question_type,
                "agent_type": agent_type
            })

        if confidence is not None:
            self.observe_histogram("prediction_confidence", confidence, labels={
                "question_type": question_type,
                "agent_type": agent_type
            })

    def record_api_request(
        self,
        service: str,
        endpoint: str,
        status: str,
        duration: float
    ):
        """Record API request metrics."""
        self.increment_counter("api_requests_total", labels={
            "service": service,
            "endpoint": endpoint,
            "status": status
        })
        self.observe_histogram("api_request_duration_seconds", duration, labels={
            "service": service,
            "endpoint": endpoint
        })

    def record_tournament_metrics(
        self,
        tournament_id: str,
        ranking_position: Optional[int] = None,
        score: Optional[float] = None,
        questions_processed: Optional[int] = None
    ):
        """Record tournament-specific metrics."""
        if ranking_position is not None:
            self.set_gauge("tournament_ranking_position", ranking_position, labels={
                "tournament_id": tournament_id
            })

        if score is not None:
            self.set_gauge("tournament_score", score, labels={
                "tournament_id": tournament_id
            })

        if questions_processed is not None:
            self.increment_counter("tournament_questions_processed", questions_processed, labels={
                "tournament_id": tournament_id,
                "question_type": "all"
            })

    def record_system_health(self, component: str, health_score: float):
        """Record system health metrics."""
        self.set_gauge("system_health_score", health_score, labels={
            "component": component
        })

    def record_circuit_breaker_state(self, service: str, state: str):
        """Record circuit breaker state."""
        state_value = {"closed": 0, "open": 1, "half_open": 0.5}.get(state.lower(), -1)
        self.set_gauge("circuit_breaker_state", state_value, labels={
            "service": service,
            "state": state
        })

    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self._lock:
            summary = {}

            # Get all keys that match the metric name
            matching_keys = [key for key in self.local_metrics.keys() if key.startswith(metric_name)]

            for key in matching_keys:
                value = self.local_metrics[key]

                if isinstance(value, dict) and "values" in value:
                    # Histogram-like metric
                    values = list(value["values"])
                    if values:
                        summary[key] = {
                            "count": value["count"],
                            "sum": value["sum"],
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                            "min": min(values),
                            "max": max(values),
                            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
                        }
                else:
                    # Simple metric
                    summary[key] = {"current_value": value}

            return summary

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in exposition format."""
        if not self.enable_prometheus:
            return "# Prometheus not available\n"

        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            self.logger.error("Failed to generate Prometheus metrics", exception=e)
            return f"# Error generating metrics: {str(e)}\n"

    def _make_metric_key(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for local metric storage."""
        if not labels:
            return metric_name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}{{{label_str}}}"

    def _record_metric_history(self, key: str, value: float):
        """Record metric value in history for trend analysis."""
        self.metric_history[key].append({
            "timestamp": datetime.utcnow(),
            "value": value
        })

    def get_metric_trends(
        self,
        metric_name: str,
        time_window: timedelta = timedelta(hours=1)
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get metric trends within a time window."""
        cutoff_time = datetime.utcnow() - time_window
        trends = {}

        for key, history in self.metric_history.items():
            if key.startswith(metric_name):
                recent_values = [
                    entry for entry in history
                    if entry["timestamp"] >= cutoff_time
                ]
                if recent_values:
                    trends[key] = recent_values

        return trends


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.observe_histogram(self.metric_name, duration, self.labels)


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector


def configure_metrics(
    enable_prometheus: bool = True,
    prometheus_port: Optional[int] = 8000,
    registry: Optional[CollectorRegistry] = None
) -> MetricsCollector:
    """Configure and return a metrics collector."""
    global metrics_collector
    metrics_collector = MetricsCollector(
        enable_prometheus=enable_prometheus,
        prometheus_port=prometheus_port,
        registry=registry
    )
    return metrics_collector
