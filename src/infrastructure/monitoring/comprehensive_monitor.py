"""
Comprehensive monitoring and performance tracking system.
Integrates budget monitoring, performance tracking, and alerting.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.cost_monitor import cost_monitor
from .budget_dashboard import budget_dashboard
from .performance_tracker import performance_tracker

logger = logging.getLogger(__name__)


class ComprehensiveMonitor:
    """Main monitoring service that coordinates all monitoring components."""

    def __init__(self):
        """Initialize comprehensive monitoring system."""
        self.budget_dashboard = budget_dashboard
        self.performance_tracker = performance_tracker
        self.cost_monitor = cost_monitor

        # Monitoring configuration
        self.monitoring_interval = 60  # seconds
        self.alert_check_interval = 30  # seconds

        # Monitoring state
        self._running = False
        self._monitor_thread = None
        self._alert_thread = None

        # Dashboard data
        self.dashboard_file = Path("logs/comprehensive_dashboard.json")
        self.dashboard_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Comprehensive monitoring system initialized")

    def start_monitoring(self):
        """Start the comprehensive monitoring system."""
        if self._running:
            logger.warning("Monitoring system is already running")
            return

        self._running = True

        # Start monitoring threads
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._alert_thread = threading.Thread(target=self._alert_loop, daemon=True)

        self._monitor_thread.start()
        self._alert_thread.start()

        logger.info("Comprehensive monitoring system started")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        if self._alert_thread:
            self._alert_thread.join(timeout=5)

        logger.info("Comprehensive monitoring system stopped")

    def track_question_processing(
        self,
        question_id: str,
        model: str,
        task_type: str,
        prompt: str,
        response: str,
        success: bool = True,
        forecast_value: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Comprehensive tracking for question processing."""
        tracking_results = {}

        # Track budget and cost
        budget_result = self.budget_dashboard.track_question_cost(
            question_id, model, task_type, prompt, response, success
        )
        tracking_results["budget"] = budget_result

        # Track forecast performance if this is a forecast
        if (
            task_type == "forecast"
            and forecast_value is not None
            and confidence is not None
        ):
            forecast_record = self.performance_tracker.record_forecast(
                question_id, forecast_value, confidence
            )
            tracking_results["forecast"] = {
                "recorded": True,
                "forecast_value": forecast_value,
                "confidence": confidence,
            }

        # Track API performance
        api_record = self.performance_tracker.record_api_performance(
            question_id, task_type, success, response_time=1.0  # Placeholder
        )
        tracking_results["api"] = api_record

        return tracking_results

    def update_forecast_outcome(self, question_id: str, actual_outcome: float) -> bool:
        """Update forecast with actual outcome."""
        return self.performance_tracker.update_forecast_outcome(
            question_id, actual_outcome
        )

    def get_comprehensive_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        budget_status = self.budget_dashboard.get_real_time_status()
        performance_metrics = self.performance_tracker.get_performance_metrics()
        api_metrics = self.performance_tracker.get_api_success_metrics()

        # Detect any performance issues
        performance_alerts = self.performance_tracker.detect_performance_degradation()

        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "budget_utilization": budget_status["budget"]["utilization_percent"],
                "budget_status": budget_status["budget"]["status_level"],
                "questions_processed": budget_status["budget"]["questions_processed"],
                "estimated_remaining": budget_status["budget"][
                    "estimated_questions_remaining"
                ],
                "overall_brier_score": performance_metrics.overall_brier_score,
                "api_success_rate": api_metrics["success_rate"],
                "active_alerts": len(budget_status["alerts"]) + len(performance_alerts),
            },
            "budget": budget_status,
            "performance": {
                "metrics": performance_metrics.to_dict(),
                "api_performance": api_metrics,
                "alerts": performance_alerts,
            },
            "recommendations": self._get_comprehensive_recommendations(
                budget_status, performance_metrics, api_metrics
            ),
        }

        return dashboard

    def _get_comprehensive_recommendations(
        self,
        budget_status: Dict[str, Any],
        performance_metrics: Any,
        api_metrics: Dict[str, Any],
    ) -> List[str]:
        """Get comprehensive optimization recommendations."""
        recommendations = []

        # Budget recommendations
        budget_recs = budget_status.get("recommendations", [])
        recommendations.extend(budget_recs)

        # Performance recommendations
        if performance_metrics.overall_brier_score > 0.3:
            recommendations.append(
                "HIGH: Brier score is elevated - review forecasting methodology"
            )

        if performance_metrics.calibration_error > 0.15:
            recommendations.append(
                "MEDIUM: Calibration error is high - adjust confidence estimation"
            )

        if performance_metrics.performance_trend == "declining":
            recommendations.append(
                "WARNING: Performance trend is declining - investigate recent changes"
            )

        # API recommendations
        if api_metrics["success_rate"] < 0.9:
            recommendations.append(
                "CRITICAL: API success rate is low - check API connectivity"
            )

        if api_metrics["fallback_rate"] > 0.2:
            recommendations.append(
                "MEDIUM: High fallback usage - primary APIs may be unreliable"
            )

        return recommendations

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Update dashboard data
                dashboard_data = self.get_comprehensive_dashboard()
                self._save_dashboard_data(dashboard_data)

                # Log periodic status
                self._log_periodic_status()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _alert_loop(self):
        """Alert checking loop."""
        while self._running:
            try:
                # Check for performance degradation
                performance_alerts = (
                    self.performance_tracker.detect_performance_degradation()
                )

                for alert in performance_alerts:
                    logger.warning(
                        f"PERFORMANCE ALERT [{alert['severity'].upper()}]: {alert['message']}"
                    )

                time.sleep(self.alert_check_interval)

            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                time.sleep(60)

    def _save_dashboard_data(self, dashboard_data: Dict[str, Any]):
        """Save dashboard data to file."""
        try:
            with open(self.dashboard_file, "w") as f:
                json.dump(dashboard_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save dashboard data: {e}")

    def _log_periodic_status(self):
        """Log periodic status summary."""
        dashboard = self.get_comprehensive_dashboard()
        summary = dashboard["summary"]

        logger.info("=== Monitoring Status ===")
        logger.info(
            f"Budget: {summary['budget_utilization']:.1f}% used ({summary['budget_status']})"
        )
        logger.info(
            f"Questions: {summary['questions_processed']} processed, "
            f"~{summary['estimated_remaining']} remaining"
        )
        logger.info(
            f"Performance: Brier={summary['overall_brier_score']:.4f}, "
            f"API Success={summary['api_success_rate']:.1%}"
        )

        if summary["active_alerts"] > 0:
            logger.warning(f"Active Alerts: {summary['active_alerts']}")

    def get_health_check(self) -> Dict[str, Any]:
        """Get system health check status."""
        dashboard = self.get_comprehensive_dashboard()
        summary = dashboard["summary"]

        # Determine overall health
        health_status = "healthy"
        issues = []

        if summary["budget_utilization"] > 95:
            health_status = "critical"
            issues.append("Budget critically low")
        elif summary["budget_utilization"] > 85:
            health_status = "warning"
            issues.append("Budget usage high")

        if summary["api_success_rate"] < 0.8:
            health_status = "critical"
            issues.append("API success rate critical")
        elif summary["api_success_rate"] < 0.9:
            if health_status == "healthy":
                health_status = "warning"
            issues.append("API success rate low")

        if summary["overall_brier_score"] > 0.4:
            if health_status == "healthy":
                health_status = "warning"
            issues.append("Forecast performance degraded")

        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "summary": summary,
            "monitoring_active": self._running,
        }

    def export_monitoring_data(self, days: int = 7) -> Dict[str, Any]:
        """Export comprehensive monitoring data for analysis."""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Get performance data
        performance_metrics = self.performance_tracker.get_performance_metrics(
            days=days
        )

        # Get budget data
        budget_status = self.budget_dashboard.get_real_time_status()

        # Get recent alerts
        recent_alerts = self.budget_dashboard._get_recent_alerts(hours=days * 24)

        return {
            "export_timestamp": datetime.now().isoformat(),
            "period_days": days,
            "performance": performance_metrics.to_dict(),
            "budget": budget_status,
            "alerts": recent_alerts,
            "summary": {
                "total_questions": budget_status["budget"]["questions_processed"],
                "total_cost": budget_status["budget"]["spent"],
                "avg_cost_per_question": budget_status["budget"][
                    "avg_cost_per_question"
                ],
                "performance_trend": performance_metrics.performance_trend,
            },
        }


# Global comprehensive monitor instance
comprehensive_monitor = ComprehensiveMonitor()
