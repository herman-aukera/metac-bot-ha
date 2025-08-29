"""
Real-time budget utilization dashboard for tournament API optimization.
Provides comprehensive cost and usage monitoring with alert system.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.budget_manager import BudgetStatus, budget_manager
from ..config.cost_monitor import CostAlert, cost_monitor
from ..config.token_tracker import token_tracker

logger = logging.getLogger(__name__)


@dataclass
class BudgetAlert:
    """Budget-specific alert for threshold breaches."""

    timestamp: datetime
    alert_type: str  # "budget_threshold", "cost_spike", "efficiency_drop"
    severity: str  # "info", "warning", "critical"
    message: str
    current_value: float
    threshold_value: float
    recommendation: str
    question_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class BudgetDashboard:
    """Real-time budget utilization dashboard with comprehensive monitoring."""

    def __init__(self):
        """Initialize budget dashboard with monitoring components."""
        self.budget_manager = budget_manager
        self.cost_monitor = cost_monitor
        self.token_tracker = token_tracker

        # Alert thresholds for budget monitoring
        self.budget_thresholds = [0.5, 0.75, 0.85, 0.95]  # 50%, 75%, 85%, 95%
        self.cost_spike_multiplier = 2.0  # Alert if cost is 2x average

        # Dashboard data storage
        self.dashboard_file = Path("logs/budget_dashboard.json")
        self.dashboard_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Budget dashboard initialized")

    def get_real_time_status(self) -> Dict[str, Any]:
        """Get comprehensive real-time budget and usage status."""
        budget_status = self.budget_manager.get_budget_status()
        usage_summary = self.token_tracker.get_usage_summary()
        cost_breakdown = self.budget_manager.get_cost_breakdown()

        return {
            "timestamp": datetime.now().isoformat(),
            "budget": {
                "total": budget_status.total_budget,
                "spent": budget_status.spent,
                "remaining": budget_status.remaining,
                "utilization_percent": budget_status.utilization_percentage,
                "status_level": budget_status.status_level,
                "questions_processed": budget_status.questions_processed,
                "avg_cost_per_question": budget_status.average_cost_per_question,
                "estimated_questions_remaining": budget_status.estimated_questions_remaining,
                "last_updated": budget_status.last_updated.isoformat(),
            },
            "usage": {
                "total_calls": usage_summary["total_calls"],
                "success_rate": usage_summary["success_rate"],
                "total_tokens": usage_summary["total_tokens"],
                "total_cost": usage_summary["total_cost"],
                "by_model": usage_summary["by_model"],
                "by_task_type": usage_summary["by_task_type"],
            },
            "breakdown": cost_breakdown,
            "alerts": self._get_active_alerts(),
            "recommendations": self._get_optimization_recommendations(),
        }

    def track_question_cost(
        self,
        question_id: str,
        model: str,
        task_type: str,
        prompt: str,
        response: str,
        success: bool = True,
    ) -> Dict[str, Any]:
        """Track cost for a specific question with real-time monitoring."""
        # Use cost monitor for comprehensive tracking
        tracking_result = self.cost_monitor.track_api_call_with_monitoring(
            question_id, model, task_type, prompt, response, success
        )

        # Check for budget alerts after each question
        self._check_budget_alerts(question_id, tracking_result["budget_cost"])

        # Update dashboard data
        self._update_dashboard_data()

        return {
            "question_id": question_id,
            "cost": tracking_result["budget_cost"],
            "tokens": {
                "input": tracking_result["input_tokens"],
                "output": tracking_result["output_tokens"],
                "total": tracking_result["total_tokens"],
            },
            "budget_remaining": self.budget_manager.get_budget_status().remaining,
            "utilization_percent": self.budget_manager.get_budget_status().utilization_percentage,
            "alerts_triggered": len(self._get_recent_alerts()),
        }

    def _check_budget_alerts(self, question_id: str, cost: float):
        """Check for budget threshold breaches and trigger alerts."""
        budget_status = self.budget_manager.get_budget_status()
        utilization = budget_status.utilization_percentage / 100

        # Check budget thresholds
        for threshold in self.budget_thresholds:
            if utilization >= threshold and not self._has_recent_threshold_alert(
                threshold
            ):
                severity = self._get_alert_severity(threshold)
                alert = BudgetAlert(
                    timestamp=datetime.now(),
                    alert_type="budget_threshold",
                    severity=severity,
                    message=f"Budget utilization reached {threshold:.0%}",
                    current_value=utilization,
                    threshold_value=threshold,
                    recommendation=self._get_threshold_recommendation(threshold),
                    question_id=question_id,
                )
                self._trigger_alert(alert)

        # Check for cost spikes
        self._check_cost_spike(question_id, cost)

    def _check_cost_spike(self, question_id: str, current_cost: float):
        """Check for unusual cost spikes."""
        recent_costs = [
            record.estimated_cost
            for record in self.budget_manager.cost_records[-10:]
            if record.success and record.question_id != question_id
        ]

        if len(recent_costs) >= 5:
            avg_cost = sum(recent_costs) / len(recent_costs)
            if current_cost > avg_cost * self.cost_spike_multiplier:
                alert = BudgetAlert(
                    timestamp=datetime.now(),
                    alert_type="cost_spike",
                    severity="warning",
                    message=f"Cost spike detected: ${current_cost:.4f} vs avg ${avg_cost:.4f}",
                    current_value=current_cost,
                    threshold_value=avg_cost * self.cost_spike_multiplier,
                    recommendation="Review prompt length and model selection for this question",
                    question_id=question_id,
                )
                self._trigger_alert(alert)

    def _get_alert_severity(self, threshold: float) -> str:
        """Get alert severity based on threshold."""
        if threshold >= 0.95:
            return "critical"
        elif threshold >= 0.85:
            return "warning"
        else:
            return "info"

    def _get_threshold_recommendation(self, threshold: float) -> str:
        """Get recommendation based on threshold level."""
        recommendations = {
            0.5: "Monitor usage closely and consider optimizing model selection",
            0.75: "Switch to more cost-efficient models for non-critical tasks",
            0.85: "Enable conservative mode and reduce forecast frequency",
            0.95: "URGENT: Enable emergency mode - critical budget threshold reached",
        }
        return recommendations.get(
            threshold, "Review budget allocation and usage patterns"
        )

    def _trigger_alert(self, alert: BudgetAlert):
        """Trigger a budget alert with logging and persistence."""
        logger.warning(f"BUDGET ALERT [{alert.severity.upper()}]: {alert.message}")
        logger.warning(f"Recommendation: {alert.recommendation}")

        # Save alert to dashboard data
        self._save_alert(alert)

        # Log detailed context
        if alert.question_id:
            logger.warning(f"Alert triggered by question: {alert.question_id}")

    def _save_alert(self, alert: BudgetAlert):
        """Save alert to persistent storage."""
        try:
            alerts_file = Path("logs/budget_alerts.json")
            alerts_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing alerts
            alerts = []
            if alerts_file.exists():
                with open(alerts_file, "r") as f:
                    data = json.load(f)
                    alerts = data.get("alerts", [])

            # Add new alert
            alerts.append(alert.to_dict())

            # Keep only last 100 alerts
            alerts = alerts[-100:]

            # Save back
            with open(alerts_file, "w") as f:
                json.dump(
                    {"alerts": alerts, "last_updated": datetime.now().isoformat()},
                    f,
                    indent=2,
                )

        except Exception as e:
            logger.error(f"Failed to save budget alert: {e}")

    def _has_recent_threshold_alert(self, threshold: float, hours: int = 1) -> bool:
        """Check if we already have a recent alert for this threshold."""
        try:
            alerts_file = Path("logs/budget_alerts.json")
            if not alerts_file.exists():
                return False

            with open(alerts_file, "r") as f:
                data = json.load(f)
                alerts = data.get("alerts", [])

            cutoff_time = datetime.now() - timedelta(hours=hours)

            for alert_data in alerts:
                alert_time = datetime.fromisoformat(alert_data["timestamp"])
                if (
                    alert_time >= cutoff_time
                    and alert_data["alert_type"] == "budget_threshold"
                    and abs(alert_data["threshold_value"] - threshold) < 0.01
                ):
                    return True

        except Exception as e:
            logger.error(f"Error checking recent alerts: {e}")

        return False

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        return self._get_recent_alerts(hours=24)

    def _get_recent_alerts(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get recent alerts within specified time window."""
        try:
            alerts_file = Path("logs/budget_alerts.json")
            if not alerts_file.exists():
                return []

            with open(alerts_file, "r") as f:
                data = json.load(f)
                alerts = data.get("alerts", [])

            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = []

            for alert_data in alerts:
                alert_time = datetime.fromisoformat(alert_data["timestamp"])
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert_data)

            return recent_alerts

        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []

    def _get_optimization_recommendations(self) -> List[str]:
        """Get actionable optimization recommendations."""
        recommendations = []
        budget_status = self.budget_manager.get_budget_status()
        usage_summary = self.token_tracker.get_usage_summary()

        # Budget-based recommendations
        if budget_status.utilization_percentage > 90:
            recommendations.append(
                "CRITICAL: Switch to GPT-4o-mini immediately for all tasks"
            )
            recommendations.append("Reduce forecast frequency to absolute minimum")
        elif budget_status.utilization_percentage > 80:
            recommendations.append(
                "HIGH: Use GPT-4o-mini for research, GPT-4o only for final forecasts"
            )
            recommendations.append("Implement aggressive prompt optimization")
        elif budget_status.utilization_percentage > 60:
            recommendations.append(
                "MEDIUM: Consider model optimization for non-critical tasks"
            )

        # Usage pattern recommendations
        if usage_summary["success_rate"] < 0.9:
            recommendations.append("Investigate API failures - success rate is low")

        # Cost efficiency recommendations
        avg_cost = budget_status.spent / max(budget_status.questions_processed, 1)
        if avg_cost > 0.5:  # $0.50 per question
            recommendations.append(
                "High cost per question - optimize prompts and model selection"
            )

        return recommendations

    def _update_dashboard_data(self):
        """Update persistent dashboard data."""
        try:
            dashboard_data = {
                "last_updated": datetime.now().isoformat(),
                "status": self.get_real_time_status(),
                "summary": {
                    "total_questions": self.budget_manager.questions_processed,
                    "total_spent": self.budget_manager.current_spend,
                    "utilization": self.budget_manager.get_budget_status().utilization_percentage,
                    "active_alerts": len(self._get_active_alerts()),
                },
            }

            with open(self.dashboard_file, "w") as f:
                json.dump(dashboard_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update dashboard data: {e}")


# Global dashboard instance
budget_dashboard = BudgetDashboard()
