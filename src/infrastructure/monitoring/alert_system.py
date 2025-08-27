"""
Advanced alerting system for performance degradation detection.
Provides real-time monitoring and alerting for forecast accuracy and API performance.
"""
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    ACCURACY_DEGRADATION = "accuracy_degradation"
    CALIBRATION_DRIFT = "calibration_drift"
    API_FAILURE_SPIKE = "api_failure_spike"
    FALLBACK_OVERUSE = "fallback_overuse"
    RESPONSE_TIME_SPIKE = "response_time_spike"
    BUDGET_THRESHOLD = "budget_threshold"
    COST_ANOMALY = "cost_anomaly"


@dataclass
class Alert:
    """Performance alert with detailed context."""
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    current_value: float
    threshold_value: float
    context: Dict[str, Any]
    recommendations: List[str]
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['alert_type'] = self.alert_type.value
        data['severity'] = self.severity.value
        if self.resolved_timestamp:
            data['resolved_timestamp'] = self.resolved_timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create from dictionary for JSON deserialization."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['alert_type'] = AlertType(data['alert_type'])
        data['severity'] = AlertSeverity(data['severity'])
        if data.get('resolved_timestamp'):
            data['resolved_timestamp'] = datetime.fromisoformat(data['resolved_timestamp'])
        return cls(**data)
class AlertSystem:
    """Advanced alerting system for performance monitoring."""

    def __init__(self):
        """Initialize alert system."""
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []

        # Alert thresholds
        self.thresholds = {
            "brier_score_degradation": 0.05,  # 5% degradation
            "calibration_error_threshold": 0.15,  # 15% calibration error
            "api_success_rate_warning": 0.9,  # 90% success rate
            "api_success_rate_critical": 0.8,  # 80% success rate
            "fallback_rate_threshold": 0.3,  # 30% fallback usage
            "response_time_spike": 2.0,  # 2x normal response time
            "cost_spike_multiplier": 2.5  # 2.5x normal cost
        }

        # Alert suppression (avoid spam)
        self.suppression_windows = {
            AlertType.ACCURACY_DEGRADATION: timedelta(hours=2),
            AlertType.CALIBRATION_DRIFT: timedelta(hours=1),
            AlertType.API_FAILURE_SPIKE: timedelta(minutes=30),
            AlertType.FALLBACK_OVERUSE: timedelta(hours=1),
            AlertType.RESPONSE_TIME_SPIKE: timedelta(minutes=15),
            AlertType.COST_ANOMALY: timedelta(minutes=30)
        }

        # Data persistence
        self.alerts_file = Path("logs/performance_alerts.json")
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)

        self._load_existing_alerts()
        logger.info(f"Alert system initialized with {len(self.active_alerts)} active alerts")

    def check_accuracy_degradation(self, current_brier: float, historical_brier: float,
                                  sample_size: int) -> Optional[Alert]:
        """Check for forecast accuracy degradation."""
        if sample_size < 10:  # Need sufficient data
            return None

        degradation = current_brier - historical_brier
        threshold = self.thresholds["brier_score_degradation"]

        if degradation > threshold and not self._is_suppressed(AlertType.ACCURACY_DEGRADATION):
            severity = AlertSeverity.CRITICAL if degradation > threshold * 2 else AlertSeverity.WARNING

            alert = Alert(
                timestamp=datetime.now(),
                alert_type=AlertType.ACCURACY_DEGRADATION,
                severity=severity,
                title="Forecast Accuracy Degradation Detected",
                message=f"Brier score increased by {degradation:.4f} ({degradation/historical_brier:.1%})",
                current_value=current_brier,
                threshold_value=historical_brier + threshold,
                context={
                    "historical_brier": historical_brier,
                    "sample_size": sample_size,
                    "degradation_percent": degradation/historical_brier * 100
                },
                recommendations=[
                    "Review recent model changes or prompt modifications",
                    "Check for data quality issues in recent questions",
                    "Consider reverting to previous forecasting methodology",
                    "Increase ensemble diversity to improve robustness"
                ]
            )

            return self._trigger_alert(alert)

        return None

    def check_calibration_drift(self, calibration_error: float,
                               confidence_bins: Dict[str, float]) -> Optional[Alert]:
        """Check for calibration drift."""
        threshold = self.thresholds["calibration_error_threshold"]

        if calibration_error > threshold and not self._is_suppressed(AlertType.CALIBRATION_DRIFT):
            severity = AlertSeverity.WARNING

            # Find worst calibrated bins
            worst_bins = sorted(
                [(bin_name, abs(float(bin_name.split('-')[0])/100 - accuracy))
                 for bin_name, accuracy in confidence_bins.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]

            alert = Alert(
                timestamp=datetime.now(),
                alert_type=AlertType.CALIBRATION_DRIFT,
                severity=severity,
                title="Calibration Drift Detected",
                message=f"Calibration error is {calibration_error:.3f} (threshold: {threshold:.3f})",
                current_value=calibration_error,
                threshold_value=threshold,
                context={
                    "confidence_bins": confidence_bins,
                    "worst_calibrated_bins": worst_bins
                },
                recommendations=[
                    "Adjust confidence estimation methodology",
                    "Review confidence calibration for specific question types",
                    "Consider recalibrating confidence intervals",
                    f"Focus on improving calibration for {worst_bins[0][0]} confidence range"
                ]
            )

            return self._trigger_alert(alert)

        return None
    def check_api_performance(self, success_rate: float, fallback_rate: float,
                             avg_response_time: float, historical_response_time: float) -> List[Alert]:
        """Check API performance metrics."""
        alerts = []

        # Check API success rate
        if success_rate < self.thresholds["api_success_rate_critical"]:
            if not self._is_suppressed(AlertType.API_FAILURE_SPIKE):
                alert = Alert(
                    timestamp=datetime.now(),
                    alert_type=AlertType.API_FAILURE_SPIKE,
                    severity=AlertSeverity.CRITICAL,
                    title="Critical API Failure Rate",
                    message=f"API success rate dropped to {success_rate:.1%}",
                    current_value=success_rate,
                    threshold_value=self.thresholds["api_success_rate_critical"],
                    context={"fallback_rate": fallback_rate},
                    recommendations=[
                        "Check API service status and connectivity",
                        "Review recent API configuration changes",
                        "Implement circuit breaker pattern",
                        "Scale up fallback mechanisms"
                    ]
                )
                alerts.append(self._trigger_alert(alert))

        elif success_rate < self.thresholds["api_success_rate_warning"]:
            if not self._is_suppressed(AlertType.API_FAILURE_SPIKE):
                alert = Alert(
                    timestamp=datetime.now(),
                    alert_type=AlertType.API_FAILURE_SPIKE,
                    severity=AlertSeverity.WARNING,
                    title="Elevated API Failure Rate",
                    message=f"API success rate is {success_rate:.1%}",
                    current_value=success_rate,
                    threshold_value=self.thresholds["api_success_rate_warning"],
                    context={"fallback_rate": fallback_rate},
                    recommendations=[
                        "Monitor API performance closely",
                        "Prepare fallback mechanisms",
                        "Check for rate limiting issues"
                    ]
                )
                alerts.append(self._trigger_alert(alert))

        # Check fallback usage
        if fallback_rate > self.thresholds["fallback_rate_threshold"]:
            if not self._is_suppressed(AlertType.FALLBACK_OVERUSE):
                alert = Alert(
                    timestamp=datetime.now(),
                    alert_type=AlertType.FALLBACK_OVERUSE,
                    severity=AlertSeverity.WARNING,
                    title="High Fallback Usage",
                    message=f"Fallback rate is {fallback_rate:.1%}",
                    current_value=fallback_rate,
                    threshold_value=self.thresholds["fallback_rate_threshold"],
                    context={"success_rate": success_rate},
                    recommendations=[
                        "Investigate primary API reliability",
                        "Review fallback trigger conditions",
                        "Consider adjusting API timeout settings"
                    ]
                )
                alerts.append(self._trigger_alert(alert))

        # Check response time spikes
        if (historical_response_time > 0 and
            avg_response_time > historical_response_time * self.thresholds["response_time_spike"]):
            if not self._is_suppressed(AlertType.RESPONSE_TIME_SPIKE):
                alert = Alert(
                    timestamp=datetime.now(),
                    alert_type=AlertType.RESPONSE_TIME_SPIKE,
                    severity=AlertSeverity.WARNING,
                    title="API Response Time Spike",
                    message=f"Response time increased to {avg_response_time:.2f}s (was {historical_response_time:.2f}s)",
                    current_value=avg_response_time,
                    threshold_value=historical_response_time * self.thresholds["response_time_spike"],
                    context={"historical_response_time": historical_response_time},
                    recommendations=[
                        "Check API service performance",
                        "Review network connectivity",
                        "Consider implementing request timeouts"
                    ]
                )
                alerts.append(self._trigger_alert(alert))

        return [alert for alert in alerts if alert is not None]
    def check_cost_anomaly(self, current_cost: float, historical_avg: float,
                          question_id: str) -> Optional[Alert]:
        """Check for cost anomalies."""
        if historical_avg <= 0:
            return None

        cost_multiplier = current_cost / historical_avg
        threshold = self.thresholds["cost_spike_multiplier"]

        if cost_multiplier > threshold and not self._is_suppressed(AlertType.COST_ANOMALY):
            alert = Alert(
                timestamp=datetime.now(),
                alert_type=AlertType.COST_ANOMALY,
                severity=AlertSeverity.WARNING,
                title="Cost Anomaly Detected",
                message=f"Question cost is {cost_multiplier:.1f}x higher than average",
                current_value=current_cost,
                threshold_value=historical_avg * threshold,
                context={
                    "question_id": question_id,
                    "historical_average": historical_avg,
                    "cost_multiplier": cost_multiplier
                },
                recommendations=[
                    "Review prompt length and complexity",
                    "Check for unusually long model responses",
                    "Consider using more cost-efficient models",
                    "Implement prompt optimization"
                ]
            )

            return self._trigger_alert(alert)

        return None

    def _trigger_alert(self, alert: Alert) -> Alert:
        """Trigger an alert and add to active alerts."""
        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        # Log the alert
        logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")

        # Save alerts
        self._save_alerts()

        return alert

    def _is_suppressed(self, alert_type: AlertType) -> bool:
        """Check if alert type is currently suppressed."""
        suppression_window = self.suppression_windows.get(alert_type, timedelta(minutes=30))
        cutoff_time = datetime.now() - suppression_window

        # Check if we have a recent alert of this type
        for alert in self.active_alerts:
            if alert.alert_type == alert_type and alert.timestamp > cutoff_time:
                return True

        return False

    def resolve_alert(self, alert: Alert, resolution_note: str = "") -> bool:
        """Resolve an active alert."""
        if alert in self.active_alerts:
            alert.resolved = True
            alert.resolved_timestamp = datetime.now()
            alert.context["resolution_note"] = resolution_note

            self.active_alerts.remove(alert)
            self._save_alerts()

            logger.info(f"Alert resolved: {alert.title}")
            return True

        return False

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        if severity:
            return [alert for alert in self.active_alerts if alert.severity == severity]
        return self.active_alerts.copy()

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_by_severity = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.WARNING: 0,
            AlertSeverity.INFO: 0
        }

        for alert in self.active_alerts:
            active_by_severity[alert.severity] += 1

        recent_history = [
            alert for alert in self.alert_history
            if alert.timestamp > datetime.now() - timedelta(hours=24)
        ]

        return {
            "active_alerts": len(self.active_alerts),
            "active_by_severity": {
                "critical": active_by_severity[AlertSeverity.CRITICAL],
                "warning": active_by_severity[AlertSeverity.WARNING],
                "info": active_by_severity[AlertSeverity.INFO]
            },
            "alerts_last_24h": len(recent_history),
            "most_recent_alert": self.alert_history[-1].to_dict() if self.alert_history else None
        }
    def _save_alerts(self):
        """Save alerts to persistent storage."""
        try:
            data = {
                "active_alerts": [alert.to_dict() for alert in self.active_alerts],
                "alert_history": [alert.to_dict() for alert in self.alert_history[-100:]],  # Keep last 100
                "last_updated": datetime.now().isoformat()
            }

            with open(self.alerts_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def _load_existing_alerts(self):
        """Load existing alerts from storage."""
        try:
            if self.alerts_file.exists():
                with open(self.alerts_file, 'r') as f:
                    data = json.load(f)

                # Load active alerts
                active_data = data.get("active_alerts", [])
                self.active_alerts = [Alert.from_dict(alert_data) for alert_data in active_data]

                # Load alert history
                history_data = data.get("alert_history", [])
                self.alert_history = [Alert.from_dict(alert_data) for alert_data in history_data]

                logger.info(f"Loaded {len(self.active_alerts)} active alerts and "
                           f"{len(self.alert_history)} historical alerts")

        except Exception as e:
            logger.warning(f"Failed to load existing alerts: {e}")

    def cleanup_old_alerts(self, days: int = 7):
        """Clean up old resolved alerts."""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Remove old alerts from history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_date or not alert.resolved
        ]

        # Remove old active alerts (shouldn't happen, but safety check)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert.timestamp > cutoff_date
        ]

        self._save_alerts()
        logger.info(f"Cleaned up alerts older than {days} days")


# Global alert system instance
alert_system = AlertSystem()
