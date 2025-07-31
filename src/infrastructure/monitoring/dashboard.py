"""
Real-time performance dashboards and alerting system.

Provides comprehensive dashboards for forecasting performance,
system health, and tournament progress with automated alerting.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque

from src.infrastructure.logging.structured_logger import get_logger
from src.infrastructure.monitoring.metrics_collector import get_metrics_collector, MetricsCollector
from src.infrastructure.monitoring.health_check_manager import get_health_check_manager, HealthCheckManager, HealthStatus
from src.infrastructure.monitoring.distributed_tracing import get_tracer, DistributedTracer


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """System alert."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    threshold_value: float
    current_value: float
    component: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "metric_name": self.metric_name,
            "threshold_value": self.threshold_value,
            "current_value": self.current_value,
            "component": self.component,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "metadata": self.metadata
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals"
    threshold: float
    severity: AlertSeverity
    component: Optional[str] = None
    evaluation_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    description: str = ""
    enabled: bool = True


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    id: str
    title: str
    widget_type: str  # "metric", "chart", "table", "status"
    metric_names: List[str]
    refresh_interval: int = 30  # seconds
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard configuration."""
    id: str
    name: str
    description: str
    widgets: List[DashboardWidget]
    refresh_interval: int = 30
    auto_refresh: bool = True


class AlertManager:
    """Manages alerts and alert rules."""

    def __init__(self):
        self.logger = get_logger("alert_manager")
        self.metrics_collector = get_metrics_collector()

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Last evaluation times for cooldown
        self.last_evaluations: Dict[str, datetime] = {}

        # Initialize default alert rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                name="High Forecasting Error Rate",
                metric_name="forecasting_requests_total",
                condition="greater_than",
                threshold=0.1,  # 10% error rate
                severity=AlertSeverity.HIGH,
                component="forecasting",
                description="Forecasting error rate is above acceptable threshold"
            ),
            AlertRule(
                name="Slow Forecasting Response",
                metric_name="forecasting_duration_seconds",
                condition="greater_than",
                threshold=60.0,  # 60 seconds
                severity=AlertSeverity.MEDIUM,
                component="forecasting",
                description="Forecasting operations are taking too long"
            ),
            AlertRule(
                name="Low System Health",
                metric_name="system_health_score",
                condition="less_than",
                threshold=0.5,
                severity=AlertSeverity.HIGH,
                description="System health score is below acceptable threshold"
            ),
            AlertRule(
                name="High API Error Rate",
                metric_name="api_requests_total",
                condition="greater_than",
                threshold=0.05,  # 5% error rate
                severity=AlertSeverity.MEDIUM,
                description="External API error rate is elevated"
            ),
            AlertRule(
                name="Circuit Breaker Open",
                metric_name="circuit_breaker_state",
                condition="equals",
                threshold=1.0,  # Open state
                severity=AlertSeverity.CRITICAL,
                description="Circuit breaker is open, service degraded"
            )
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def evaluate_rules(self):
        """Evaluate all alert rules."""
        current_time = datetime.utcnow()

        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            # Check cooldown period
            last_eval = self.last_evaluations.get(rule_name)
            if last_eval and (current_time - last_eval) < rule.cooldown_period:
                continue

            try:
                self._evaluate_rule(rule)
                self.last_evaluations[rule_name] = current_time
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule_name}", exception=e)

    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        # Get metric value
        metric_summary = self.metrics_collector.get_metric_summary(rule.metric_name)

        if not metric_summary:
            return

        # Find relevant metric value
        current_value = None
        for key, data in metric_summary.items():
            if rule.component and rule.component not in key:
                continue

            if "current_value" in data:
                current_value = data["current_value"]
            elif "mean" in data:
                current_value = data["mean"]

            break

        if current_value is None:
            return

        # Evaluate condition
        alert_triggered = False

        if rule.condition == "greater_than" and current_value > rule.threshold:
            alert_triggered = True
        elif rule.condition == "less_than" and current_value < rule.threshold:
            alert_triggered = True
        elif rule.condition == "equals" and abs(current_value - rule.threshold) < 0.001:
            alert_triggered = True

        # Handle alert
        if alert_triggered:
            self._trigger_alert(rule, current_value)
        else:
            self._resolve_alert(rule.name)

    def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert."""
        alert_id = f"{rule.name}_{rule.component or 'system'}"

        # Check if alert already exists
        if alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_id]
            alert.current_value = current_value
            return

        # Create new alert
        alert = Alert(
            id=alert_id,
            name=rule.name,
            description=rule.description,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            metric_name=rule.metric_name,
            threshold_value=rule.threshold,
            current_value=current_value,
            component=rule.component
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error("Error in alert callback", exception=e)

        self.logger.warning(
            f"Alert triggered: {alert.name}",
            extra={"alert": alert.to_dict()}
        )

    def _resolve_alert(self, rule_name: str):
        """Resolve an alert."""
        alert_id = f"{rule_name}_system"  # Try system first

        if alert_id not in self.active_alerts:
            # Try with component names
            for existing_id in list(self.active_alerts.keys()):
                if existing_id.startswith(f"{rule_name}_"):
                    alert_id = existing_id
                    break

        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()

            del self.active_alerts[alert_id]

            self.logger.info(f"Alert resolved: {alert.name}")

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()

            self.logger.info(f"Alert acknowledged: {alert.name}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]


class PerformanceDashboard:
    """
    Real-time performance dashboard system.

    Provides comprehensive dashboards for monitoring forecasting
    performance, system health, and tournament progress.
    """

    def __init__(self):
        self.logger = get_logger("performance_dashboard")
        self.metrics_collector = get_metrics_collector()
        self.health_manager = get_health_check_manager()
        self.tracer = get_tracer()
        self.alert_manager = AlertManager()

        # Dashboard storage
        self.dashboards: Dict[str, Dashboard] = {}

        # Data cache for dashboard widgets
        self.widget_data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # Initialize default dashboards
        self._initialize_default_dashboards()

        # Setup alert notifications
        self.alert_manager.add_alert_callback(self._handle_alert_notification)

    def _initialize_default_dashboards(self):
        """Initialize default dashboards."""
        # Forecasting Performance Dashboard
        forecasting_dashboard = Dashboard(
            id="forecasting_performance",
            name="Forecasting Performance",
            description="Real-time forecasting performance metrics",
            widgets=[
                DashboardWidget(
                    id="forecasting_requests",
                    title="Forecasting Requests",
                    widget_type="metric",
                    metric_names=["forecasting_requests_total"],
                    config={"display_type": "counter"}
                ),
                DashboardWidget(
                    id="forecasting_duration",
                    title="Forecasting Duration",
                    widget_type="chart",
                    metric_names=["forecasting_duration_seconds"],
                    config={"chart_type": "histogram", "time_window": "1h"}
                ),
                DashboardWidget(
                    id="prediction_accuracy",
                    title="Prediction Accuracy",
                    widget_type="chart",
                    metric_names=["prediction_accuracy"],
                    config={"chart_type": "line", "time_window": "24h"}
                ),
                DashboardWidget(
                    id="ensemble_consensus",
                    title="Ensemble Consensus",
                    widget_type="metric",
                    metric_names=["ensemble_consensus_strength"],
                    config={"display_type": "gauge"}
                )
            ]
        )

        # System Health Dashboard
        system_dashboard = Dashboard(
            id="system_health",
            name="System Health",
            description="Overall system health and component status",
            widgets=[
                DashboardWidget(
                    id="health_overview",
                    title="Health Overview",
                    widget_type="status",
                    metric_names=["system_health_score"],
                    config={"display_type": "status_grid"}
                ),
                DashboardWidget(
                    id="api_performance",
                    title="API Performance",
                    widget_type="chart",
                    metric_names=["api_request_duration_seconds"],
                    config={"chart_type": "line", "time_window": "1h"}
                ),
                DashboardWidget(
                    id="circuit_breakers",
                    title="Circuit Breaker Status",
                    widget_type="table",
                    metric_names=["circuit_breaker_state"],
                    config={"display_type": "status_table"}
                ),
                DashboardWidget(
                    id="memory_usage",
                    title="Memory Usage",
                    widget_type="chart",
                    metric_names=["memory_usage_bytes"],
                    config={"chart_type": "area", "time_window": "1h"}
                )
            ]
        )

        # Tournament Progress Dashboard
        tournament_dashboard = Dashboard(
            id="tournament_progress",
            name="Tournament Progress",
            description="Tournament performance and competitive analysis",
            widgets=[
                DashboardWidget(
                    id="tournament_ranking",
                    title="Tournament Ranking",
                    widget_type="metric",
                    metric_names=["tournament_ranking_position"],
                    config={"display_type": "ranking"}
                ),
                DashboardWidget(
                    id="tournament_score",
                    title="Tournament Score",
                    widget_type="chart",
                    metric_names=["tournament_score"],
                    config={"chart_type": "line", "time_window": "7d"}
                ),
                DashboardWidget(
                    id="questions_processed",
                    title="Questions Processed",
                    widget_type="metric",
                    metric_names=["tournament_questions_processed"],
                    config={"display_type": "counter"}
                ),
                DashboardWidget(
                    id="agent_performance",
                    title="Agent Performance",
                    widget_type="table",
                    metric_names=["agent_performance_score"],
                    config={"display_type": "performance_table"}
                )
            ]
        )

        self.dashboards["forecasting_performance"] = forecasting_dashboard
        self.dashboards["system_health"] = system_dashboard
        self.dashboards["tournament_progress"] = tournament_dashboard

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID."""
        return self.dashboards.get(dashboard_id)

    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get complete dashboard data."""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {}

        dashboard_data = {
            "id": dashboard.id,
            "name": dashboard.name,
            "description": dashboard.description,
            "refresh_interval": dashboard.refresh_interval,
            "widgets": []
        }

        for widget in dashboard.widgets:
            widget_data = self.get_widget_data(widget)
            dashboard_data["widgets"].append(widget_data)

        return dashboard_data

    def get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget."""
        # Check cache
        cache_key = f"{widget.id}_{hash(tuple(widget.metric_names))}"
        cached_time = self.cache_timestamps.get(cache_key)

        if cached_time and (datetime.utcnow() - cached_time).seconds < widget.refresh_interval:
            return self.widget_data_cache.get(cache_key, {})

        # Generate fresh data
        widget_data = {
            "id": widget.id,
            "title": widget.title,
            "type": widget.widget_type,
            "config": widget.config,
            "data": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Collect metric data
        for metric_name in widget.metric_names:
            metric_data = self._get_metric_data_for_widget(metric_name, widget)
            widget_data["data"][metric_name] = metric_data

        # Add health data for status widgets
        if widget.widget_type == "status":
            widget_data["health_data"] = self._get_health_data()

        # Add trace data for performance widgets
        if "performance" in widget.title.lower():
            widget_data["trace_data"] = self._get_trace_data()

        # Cache the data
        self.widget_data_cache[cache_key] = widget_data
        self.cache_timestamps[cache_key] = datetime.utcnow()

        return widget_data

    def _get_metric_data_for_widget(self, metric_name: str, widget: DashboardWidget) -> Dict[str, Any]:
        """Get metric data formatted for widget display."""
        metric_summary = self.metrics_collector.get_metric_summary(metric_name)

        if not metric_summary:
            return {"error": "No data available"}

        # Get time window from config
        time_window_str = widget.config.get("time_window", "1h")
        time_window = self._parse_time_window(time_window_str)

        # Get trends if available
        trends = self.metrics_collector.get_metric_trends(metric_name, time_window)

        return {
            "summary": metric_summary,
            "trends": trends,
            "latest_value": self._extract_latest_value(metric_summary),
            "time_window": time_window_str
        }

    def _get_health_data(self) -> Dict[str, Any]:
        """Get health data for status widgets."""
        health_summary = self.health_manager.get_latest_health_status()

        return {
            "overall_status": health_summary.overall_status.value,
            "component_count": {
                "healthy": health_summary.healthy_components,
                "degraded": health_summary.degraded_components,
                "unhealthy": health_summary.unhealthy_components,
                "unknown": health_summary.unknown_components
            },
            "components": [result.to_dict() for result in health_summary.component_results]
        }

    def _get_trace_data(self) -> Dict[str, Any]:
        """Get trace data for performance analysis."""
        # Get recent traces
        traces = self.tracer.search_traces(limit=50)

        if not traces:
            return {"traces": [], "summary": {}}

        # Calculate summary statistics
        durations = [trace.duration_ms for trace in traces if trace.duration_ms]

        summary = {}
        if durations:
            summary = {
                "total_traces": len(traces),
                "avg_duration_ms": statistics.mean(durations),
                "median_duration_ms": statistics.median(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations)
            }

        return {
            "traces": [trace.to_dict() for trace in traces[:10]],  # Latest 10
            "summary": summary
        }

    def _parse_time_window(self, time_window_str: str) -> timedelta:
        """Parse time window string to timedelta."""
        if time_window_str.endswith('m'):
            return timedelta(minutes=int(time_window_str[:-1]))
        elif time_window_str.endswith('h'):
            return timedelta(hours=int(time_window_str[:-1]))
        elif time_window_str.endswith('d'):
            return timedelta(days=int(time_window_str[:-1]))
        else:
            return timedelta(hours=1)  # Default

    def _extract_latest_value(self, metric_summary: Dict[str, Any]) -> Optional[float]:
        """Extract the latest value from metric summary."""
        for data in metric_summary.values():
            if "current_value" in data:
                return data["current_value"]
            elif "mean" in data:
                return data["mean"]
        return None

    def _handle_alert_notification(self, alert: Alert):
        """Handle alert notifications."""
        self.logger.warning(
            f"Dashboard alert: {alert.name}",
            extra={
                "alert_id": alert.id,
                "severity": alert.severity.value,
                "component": alert.component,
                "current_value": alert.current_value,
                "threshold": alert.threshold_value
            }
        )

        # Here you could integrate with external notification systems
        # like Slack, PagerDuty, email, etc.

    def get_alerts(self) -> Dict[str, Any]:
        """Get current alerts for dashboard display."""
        active_alerts = self.alert_manager.get_active_alerts()
        alert_history = self.alert_manager.get_alert_history(limit=50)

        return {
            "active_alerts": [alert.to_dict() for alert in active_alerts],
            "alert_history": [alert.to_dict() for alert in alert_history],
            "summary": {
                "total_active": len(active_alerts),
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                "medium": len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
                "low": len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
            }
        }

    def start_monitoring(self):
        """Start background monitoring and alerting."""
        asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started dashboard monitoring and alerting")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Evaluate alert rules
                self.alert_manager.evaluate_rules()

                # Clean up old cached data
                self._cleanup_cache()

                # Wait before next evaluation
                await asyncio.sleep(30)  # 30 seconds

            except Exception as e:
                self.logger.error("Error in dashboard monitoring loop", exception=e)
                await asyncio.sleep(60)  # Longer wait on error

    def _cleanup_cache(self):
        """Clean up old cached data."""
        current_time = datetime.utcnow()
        expired_keys = []

        for key, timestamp in self.cache_timestamps.items():
            if (current_time - timestamp).seconds > 300:  # 5 minutes
                expired_keys.append(key)

        for key in expired_keys:
            if key in self.widget_data_cache:
                del self.widget_data_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]


# Global dashboard instance
performance_dashboard = PerformanceDashboard()


def get_dashboard() -> PerformanceDashboard:
    """Get the global performance dashboard instance."""
    return performance_dashboard


def configure_dashboard(auto_start_monitoring: bool = True) -> PerformanceDashboard:
    """Configure and optionally start dashboard monitoring."""
    global performance_dashboard
    performance_dashboard = PerformanceDashboard()

    if auto_start_monitoring:
        performance_dashboard.start_monitoring()

    return performance_dashboard
