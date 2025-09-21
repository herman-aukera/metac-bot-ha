"""
Production-grade metrics and monitoring service for AI forecasting bot.
Provides comprehensive performance tracking, alerting, and observability.
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import HTTPServer
from typing import Any, Dict, List

import psutil
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.exposition import MetricsHandler


@dataclass
class MetricPoint:
    """Individual metric data point."""

    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    condition: str
    threshold: float
    duration: int  # seconds
    severity: str
    message: str
    enabled: bool = True


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self):
        self.registry = CollectorRegistry()
        self._setup_metrics()
        self._metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()

    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_counter = Counter(
            "forecasting_bot_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "forecasting_bot_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # Forecasting metrics
        self.predictions_total = Counter(
            "forecasting_bot_predictions_total",
            "Total number of predictions made",
            ["agent_type", "question_category"],
            registry=self.registry,
        )

        self.accuracy_score = Gauge(
            "forecasting_bot_accuracy_score",
            "Current prediction accuracy score",
            registry=self.registry,
        )

        self.brier_score = Gauge(
            "forecasting_bot_brier_score", "Current Brier score", registry=self.registry
        )

        self.calibration_error = Gauge(
            "forecasting_bot_calibration_error",
            "Current calibration error",
            registry=self.registry,
        )

        # Tournament metrics
        self.tournament_rank = Gauge(
            "forecasting_bot_tournament_rank",
            "Current tournament ranking",
            registry=self.registry,
        )

        self.missed_deadlines = Counter(
            "forecasting_bot_missed_deadlines_total",
            "Total number of missed deadlines",
            registry=self.registry,
        )

        # System metrics
        self.memory_usage = Gauge(
            "forecasting_bot_memory_usage_bytes",
            "Memory usage in bytes",
            registry=self.registry,
        )

        self.cpu_usage = Gauge(
            "forecasting_bot_cpu_usage_percent",
            "CPU usage percentage",
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "forecasting_bot_errors_total",
            "Total number of errors",
            ["error_type", "component"],
            registry=self.registry,
        )

        # Agent performance metrics
        self.agent_accuracy = Gauge(
            "forecasting_bot_agent_accuracy",
            "Individual agent accuracy",
            ["agent_type"],
            registry=self.registry,
        )

        self.ensemble_diversity = Gauge(
            "forecasting_bot_ensemble_diversity",
            "Ensemble prediction diversity",
            registry=self.registry,
        )

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.request_counter.labels(
            method=method, endpoint=endpoint, status=str(status)
        ).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def record_prediction(self, agent_type: str, question_category: str):
        """Record prediction metrics."""
        self.predictions_total.labels(
            agent_type=agent_type, question_category=question_category
        ).inc()

    def update_accuracy(self, accuracy: float):
        """Update accuracy metrics."""
        self.accuracy_score.set(accuracy)
        with self._lock:
            self._metrics_history["accuracy"].append(
                MetricPoint(datetime.now(), accuracy)
            )

    def update_brier_score(self, score: float):
        """Update Brier score."""
        self.brier_score.set(score)
        with self._lock:
            self._metrics_history["brier_score"].append(
                MetricPoint(datetime.now(), score)
            )

    def update_calibration_error(self, error: float):
        """Update calibration error."""
        self.calibration_error.set(error)
        with self._lock:
            self._metrics_history["calibration_error"].append(
                MetricPoint(datetime.now(), error)
            )

    def update_tournament_rank(self, rank: int):
        """Update tournament ranking."""
        self.tournament_rank.set(rank)

    def record_missed_deadline(self):
        """Record missed deadline."""
        self.missed_deadlines.inc()

    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        self.errors_total.labels(error_type=error_type, component=component).inc()

    def update_agent_accuracy(self, agent_type: str, accuracy: float):
        """Update individual agent accuracy."""
        self.agent_accuracy.labels(agent_type=agent_type).set(accuracy)

    def update_ensemble_diversity(self, diversity: float):
        """Update ensemble diversity metric."""
        self.ensemble_diversity.set(diversity)

    def update_system_metrics(self):
        """Update system resource metrics."""
        process = psutil.Process()
        self.memory_usage.set(process.memory_info().rss)
        self.cpu_usage.set(process.cpu_percent())

    def get_metrics_data(self) -> bytes:
        """Get Prometheus metrics data."""
        return generate_latest(self.registry)

    def get_metrics_history(
        self, metric_name: str, duration_minutes: int = 60
    ) -> List[MetricPoint]:
        """Get historical metrics data."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        with self._lock:
            history = self._metrics_history.get(metric_name, deque())
            return [point for point in history if point.timestamp >= cutoff_time]


class AlertManager:
    """Manages alerting based on metrics."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default alert rules."""
        self.alert_rules = [
            AlertRule(
                name="HighErrorRate",
                condition="error_rate > threshold",
                threshold=0.1,
                duration=120,
                severity="warning",
                message="High error rate detected",
            ),
            AlertRule(
                name="LowAccuracy",
                condition="accuracy < threshold",
                threshold=0.6,
                duration=300,
                severity="warning",
                message="Prediction accuracy has dropped",
            ),
            AlertRule(
                name="HighMemoryUsage",
                condition="memory_usage > threshold",
                threshold=1073741824,  # 1GB
                duration=300,
                severity="warning",
                message="High memory usage detected",
            ),
            AlertRule(
                name="MissedDeadline",
                condition="missed_deadlines > threshold",
                threshold=0,
                duration=0,
                severity="critical",
                message="Tournament deadline missed",
            ),
            AlertRule(
                name="CalibrationDrift",
                condition="abs(calibration_error) > threshold",
                threshold=0.1,
                duration=600,
                severity="warning",
                message="Calibration drift detected",
            ),
        ]

    def check_alerts(self):
        """Check all alert conditions."""
        current_time = datetime.now()

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            if self._evaluate_condition(rule):
                if rule.name not in self.active_alerts:
                    self.active_alerts[rule.name] = current_time
                elif (
                    current_time - self.active_alerts[rule.name]
                ).total_seconds() >= rule.duration:
                    self._trigger_alert(rule)
            else:
                if rule.name in self.active_alerts:
                    self._resolve_alert(rule)
                    del self.active_alerts[rule.name]

    def _evaluate_condition(self, rule: AlertRule) -> bool:
        """Evaluate alert condition."""
        try:
            if rule.name == "HighErrorRate":
                # Calculate error rate from recent history
                return False  # Placeholder
            elif rule.name == "LowAccuracy":
                accuracy_history = self.metrics_collector.get_metrics_history(
                    "accuracy", 5
                )
                if accuracy_history:
                    return accuracy_history[-1].value < rule.threshold
            elif rule.name == "HighMemoryUsage":
                process = psutil.Process()
                return process.memory_info().rss > rule.threshold
            elif rule.name == "MissedDeadline":
                # Check if any deadlines were missed recently
                return False  # Placeholder
            elif rule.name == "CalibrationDrift":
                calibration_history = self.metrics_collector.get_metrics_history(
                    "calibration_error", 10
                )
                if calibration_history:
                    return abs(calibration_history[-1].value) > rule.threshold
        except Exception as e:
            logging.error(f"Error evaluating alert condition {rule.name}: {e}")

        return False

    def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert."""
        alert_data = {
            "name": rule.name,
            "severity": rule.severity,
            "message": rule.message,
            "timestamp": datetime.now().isoformat(),
            "status": "firing",
        }

        self.alert_history.append(alert_data)
        self._send_alert_notification(alert_data)

        logging.warning(f"Alert triggered: {rule.name} - {rule.message}")

    def _resolve_alert(self, rule: AlertRule):
        """Resolve an alert."""
        alert_data = {
            "name": rule.name,
            "severity": rule.severity,
            "message": f"{rule.message} - RESOLVED",
            "timestamp": datetime.now().isoformat(),
            "status": "resolved",
        }

        self.alert_history.append(alert_data)
        self._send_alert_notification(alert_data)

        logging.info(f"Alert resolved: {rule.name}")

    def _send_alert_notification(self, alert_data: Dict[str, Any]):
        """Send alert notification."""
        # Implement notification logic (webhook, email, etc.)
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if webhook_url:
            try:
                import requests

                requests.post(webhook_url, json=alert_data, timeout=5)
            except Exception as e:
                logging.error(f"Failed to send alert notification: {e}")


class PerformanceTracker:
    """Tracks and analyzes performance trends."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.performance_data = defaultdict(list)

    def track_prediction_performance(
        self, prediction_id: str, accuracy: float, confidence: float, agent_type: str
    ):
        """Track individual prediction performance."""
        self.performance_data["predictions"].append(
            {
                "id": prediction_id,
                "accuracy": accuracy,
                "confidence": confidence,
                "agent_type": agent_type,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        analysis = {
            "accuracy_trend": self._calculate_trend("accuracy"),
            "calibration_trend": self._calculate_trend("calibration_error"),
            "agent_performance": self._analyze_agent_performance(),
            "system_health": self._analyze_system_health(),
        }
        return analysis

    def _calculate_trend(self, metric_name: str) -> Dict[str, float]:
        """Calculate trend for a metric."""
        history = self.metrics_collector.get_metrics_history(metric_name, 60)
        if len(history) < 2:
            return {"trend": 0.0, "current": 0.0, "change": 0.0}

        recent = sum(point.value for point in history[-10:]) / min(10, len(history))
        older = sum(point.value for point in history[:10]) / min(10, len(history))

        return {
            "trend": (recent - older) / older if older != 0 else 0.0,
            "current": recent,
            "change": recent - older,
        }

    def _analyze_agent_performance(self) -> Dict[str, Any]:
        """Analyze individual agent performance."""
        predictions = self.performance_data.get("predictions", [])
        if not predictions:
            return {}

        agent_stats = defaultdict(list)
        for pred in predictions[-100:]:  # Last 100 predictions
            agent_stats[pred["agent_type"]].append(pred["accuracy"])

        return {
            agent_type: {
                "avg_accuracy": sum(accuracies) / len(accuracies),
                "count": len(accuracies),
            }
            for agent_type, accuracies in agent_stats.items()
        }

    def _analyze_system_health(self) -> Dict[str, Any]:
        """Analyze system health metrics."""
        process = psutil.Process()
        return {
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "uptime_seconds": time.time() - process.create_time(),
        }


class MonitoringService:
    """Main monitoring service that coordinates all monitoring components."""

    def __init__(self, port: int = 8080):
        self.port = port
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self._running = False
        self._monitor_thread = None

    def start(self):
        """Start the monitoring service."""
        self._running = True

        # Start metrics HTTP server
        self._start_metrics_server()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()

        logging.info(f"Monitoring service started on port {self.port}")

    def stop(self):
        """Stop the monitoring service."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logging.info("Monitoring service stopped")

    def _start_metrics_server(self):
        """Start HTTP server for metrics endpoint."""

        class MetricsRequestHandler(MetricsHandler):
            def __init__(self, request, client_address, server, metrics_collector):
                self.metrics_collector = metrics_collector
                super().__init__(request, client_address, server)

            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(self.metrics_collector.get_metrics_data())
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    health_data = json.dumps(
                        {"status": "healthy", "timestamp": datetime.now().isoformat()}
                    )
                    self.wfile.write(health_data.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        def handler_factory(*args):
            return MetricsRequestHandler(*args, self.metrics_collector)

        server = HTTPServer(("", self.port), handler_factory)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Update system metrics
                self.metrics_collector.update_system_metrics()

                # Check alerts
                self.alert_manager.check_alerts()

                # Sleep for monitoring interval
                time.sleep(30)  # 30 seconds

            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "status": "healthy" if self._running else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "accuracy": self.metrics_collector.get_metrics_history("accuracy", 5),
                "system": self.performance_tracker._analyze_system_health(),
            },
            "alerts": {
                "active": len(self.alert_manager.active_alerts),
                "recent": self.alert_manager.alert_history[-10:],
            },
        }
