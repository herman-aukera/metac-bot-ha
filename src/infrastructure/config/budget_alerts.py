"""
Budget alerting and monitoring system for tournament API usage.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .budget_manager import budget_manager

logger = logging.getLogger(__name__)


@dataclass
class BudgetAlert:
    """Budget alert record."""

    timestamp: datetime
    alert_type: str  # "warning", "high", "critical"
    message: str
    budget_utilization: float
    remaining_budget: float
    questions_processed: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type,
            "message": self.message,
            "budget_utilization": self.budget_utilization,
            "remaining_budget": self.remaining_budget,
            "questions_processed": self.questions_processed,
        }


class BudgetAlertSystem:
    """Manages budget alerts and notifications."""

    def __init__(self):
        """Initialize budget alert system."""
        self.budget_manager = budget_manager
        self.alert_thresholds = {
            "warning": 0.8,  # 80%
            "high": 0.9,  # 90%
            "critical": 0.95,  # 95%
        }

        # Alert history
        self.alerts_file = Path("logs/budget_alerts.json")
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)
        self.alert_history: List[BudgetAlert] = []
        self._load_alert_history()

        # Tracking for alert frequency
        self.last_alert_time = {}
        self.alert_cooldown = timedelta(
            minutes=30
        )  # Minimum time between similar alerts

        logger.info("Budget alert system initialized")

    def check_and_alert(self) -> Optional[BudgetAlert]:
        """Check budget status and generate alerts if needed."""
        status = self.budget_manager.get_budget_status()
        utilization = status.utilization_percentage / 100

        # Determine alert level
        alert_type = None
        if utilization >= self.alert_thresholds["critical"]:
            alert_type = "critical"
        elif utilization >= self.alert_thresholds["high"]:
            alert_type = "high"
        elif utilization >= self.alert_thresholds["warning"]:
            alert_type = "warning"

        if not alert_type:
            return None

        # Check if we should send this alert (cooldown)
        if self._should_suppress_alert(alert_type):
            return None

        # Create alert
        alert = self._create_alert(alert_type, status)

        # Log alert
        self._log_alert(alert)

        # Save alert to history
        self.alert_history.append(alert)
        self._save_alert_history()

        # Update last alert time
        self.last_alert_time[alert_type] = datetime.now()

        return alert

    def _should_suppress_alert(self, alert_type: str) -> bool:
        """Check if alert should be suppressed due to cooldown."""
        last_alert = self.last_alert_time.get(alert_type)
        if not last_alert:
            return False

        return datetime.now() - last_alert < self.alert_cooldown

    def _create_alert(self, alert_type: str, status) -> BudgetAlert:
        """Create budget alert based on status."""
        messages = {
            "warning": f"Budget usage at {status.utilization_percentage:.1f}% - Monitor spending closely",
            "high": f"HIGH budget usage at {status.utilization_percentage:.1f}% - Consider conservative mode",
            "critical": f"CRITICAL budget usage at {status.utilization_percentage:.1f}% - Emergency mode recommended",
        }

        return BudgetAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            message=messages[alert_type],
            budget_utilization=status.utilization_percentage,
            remaining_budget=status.remaining,
            questions_processed=status.questions_processed,
        )

    def _log_alert(self, alert: BudgetAlert):
        """Log budget alert with appropriate level."""
        log_levels = {
            "warning": logger.warning,
            "high": logger.error,
            "critical": logger.critical,
        }

        log_func = log_levels.get(alert.alert_type, logger.info)
        log_func(f"BUDGET ALERT [{alert.alert_type.upper()}]: {alert.message}")

        # Additional context
        logger.info(f"Remaining budget: ${alert.remaining_budget:.4f}")
        logger.info(f"Questions processed: {alert.questions_processed}")

        # Recommendations based on alert type
        if alert.alert_type == "warning":
            logger.info(
                "Recommendation: Monitor spending and consider using cheaper models"
            )
        elif alert.alert_type == "high":
            logger.warning(
                "Recommendation: Switch to conservative mode (GPT-4o-mini only)"
            )
        elif alert.alert_type == "critical":
            logger.critical("Recommendation: Enable emergency mode or stop processing")

    def get_budget_recommendations(self) -> List[str]:
        """Get budget optimization recommendations based on current status."""
        status = self.budget_manager.get_budget_status()
        utilization = status.utilization_percentage / 100

        recommendations = []

        if utilization >= 0.95:
            recommendations.extend(
                [
                    "EMERGENCY: Stop all non-essential processing",
                    "Use only GPT-4o-mini for all tasks",
                    "Consider pausing bot until budget resets",
                    "Review cost breakdown to identify optimization opportunities",
                ]
            )
        elif utilization >= 0.9:
            recommendations.extend(
                [
                    "Switch to conservative mode immediately",
                    "Use GPT-4o only for final forecasts on complex questions",
                    "Reduce research depth and prompt length",
                    "Monitor every API call closely",
                ]
            )
        elif utilization >= 0.8:
            recommendations.extend(
                [
                    "Monitor budget usage closely",
                    "Consider using GPT-4o-mini for research tasks",
                    "Optimize prompt efficiency",
                    "Review question complexity assessment",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Budget usage is healthy",
                    "Continue with current model selection strategy",
                    "Regular monitoring recommended",
                ]
            )

        return recommendations

    def get_cost_optimization_suggestions(self) -> List[str]:
        """Get specific cost optimization suggestions."""
        breakdown = self.budget_manager.get_cost_breakdown()
        suggestions = []

        # Analyze model usage
        if "by_model" in breakdown:
            total_cost = sum(
                model_data["cost"] for model_data in breakdown["by_model"].values()
            )

            for model, data in breakdown["by_model"].items():
                model_percentage = (
                    (data["cost"] / total_cost) * 100 if total_cost > 0 else 0
                )

                if "gpt-4o" in model and model_percentage > 60:
                    suggestions.append(
                        f"High GPT-4o usage ({model_percentage:.1f}%) - consider GPT-4o-mini for research"
                    )

                if data["cost"] > 20:  # High cost model
                    suggestions.append(
                        f"Consider reducing {model} usage (${data['cost']:.4f} spent)"
                    )

        # Analyze task types
        if "by_task_type" in breakdown:
            for task, data in breakdown["by_task_type"].items():
                if task == "research" and data["cost"] > 10:
                    suggestions.append(
                        "Research costs are high - consider shorter prompts or cheaper models"
                    )
                elif task == "forecast" and data["cost"] > 30:
                    suggestions.append(
                        "Forecast costs are high - ensure complex questions justify GPT-4o usage"
                    )

        if not suggestions:
            suggestions.append("Cost distribution looks reasonable")

        return suggestions

    def generate_budget_report(self) -> Dict[str, Any]:
        """Generate comprehensive budget report."""
        status = self.budget_manager.get_budget_status()
        breakdown = self.budget_manager.get_cost_breakdown()
        recommendations = self.get_budget_recommendations()
        optimizations = self.get_cost_optimization_suggestions()

        # Recent alerts
        recent_alerts = [
            alert
            for alert in self.alert_history
            if alert.timestamp > datetime.now() - timedelta(hours=24)
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "budget_status": status.to_dict(),
            "cost_breakdown": breakdown,
            "recommendations": recommendations,
            "optimization_suggestions": optimizations,
            "recent_alerts": [alert.to_dict() for alert in recent_alerts],
            "alert_summary": {
                "total_alerts": len(self.alert_history),
                "recent_alerts": len(recent_alerts),
                "alert_types": {
                    alert_type: len(
                        [a for a in recent_alerts if a.alert_type == alert_type]
                    )
                    for alert_type in ["warning", "high", "critical"]
                },
            },
        }

    def _load_alert_history(self):
        """Load alert history from file."""
        try:
            if self.alerts_file.exists():
                with open(self.alerts_file, "r") as f:
                    data = json.load(f)

                self.alert_history = []
                for alert_data in data.get("alerts", []):
                    alert = BudgetAlert(
                        timestamp=datetime.fromisoformat(alert_data["timestamp"]),
                        alert_type=alert_data["alert_type"],
                        message=alert_data["message"],
                        budget_utilization=alert_data["budget_utilization"],
                        remaining_budget=alert_data["remaining_budget"],
                        questions_processed=alert_data["questions_processed"],
                    )
                    self.alert_history.append(alert)

                logger.debug(
                    f"Loaded {len(self.alert_history)} budget alerts from history"
                )

        except Exception as e:
            logger.warning(f"Failed to load alert history: {e}")

    def _save_alert_history(self):
        """Save alert history to file."""
        try:
            # Keep only last 100 alerts to prevent file from growing too large
            recent_alerts = (
                self.alert_history[-100:]
                if len(self.alert_history) > 100
                else self.alert_history
            )

            data = {
                "alerts": [alert.to_dict() for alert in recent_alerts],
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.alerts_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alert history: {e}")

    def log_budget_summary(self):
        """Log comprehensive budget summary."""
        report = self.generate_budget_report()

        logger.info("=== Budget Summary Report ===")
        status = report["budget_status"]
        logger.info(
            f"Budget: ${status['spent']:.4f} / ${status['total_budget']:.2f} ({status['utilization_percentage']:.1f}%)"
        )
        logger.info(f"Questions Processed: {status['questions_processed']}")
        logger.info(
            f"Average Cost/Question: ${status['average_cost_per_question']:.4f}"
        )
        logger.info(
            f"Estimated Questions Remaining: {status['estimated_questions_remaining']}"
        )

        # Recent alerts
        recent_count = report["alert_summary"]["recent_alerts"]
        if recent_count > 0:
            logger.warning(f"Recent Alerts (24h): {recent_count}")

        # Top recommendations
        logger.info("Top Recommendations:")
        for i, rec in enumerate(report["recommendations"][:3], 1):
            logger.info(f"  {i}. {rec}")


# Global instance
budget_alert_system = BudgetAlertSystem()
