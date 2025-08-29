"""
Comprehensive cost monitoring and budget utilization tracking.
Integrates TokenTracker and BudgetManager for real-time cost analysis.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .budget_manager import BudgetManager, budget_manager
from .token_tracker import TokenTracker, token_tracker

logger = logging.getLogger(__name__)


@dataclass
class CostAlert:
    """Alert for budget or cost threshold breaches."""

    timestamp: datetime
    alert_type: str  # "budget_threshold", "cost_spike", "efficiency_drop"
    severity: str  # "info", "warning", "critical"
    message: str
    current_value: float
    threshold_value: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class CostMonitor:
    """Comprehensive cost monitoring and budget utilization tracking."""

    def __init__(
        self, token_tracker: TokenTracker = None, budget_manager: BudgetManager = None
    ):
        """Initialize cost monitor with tracker and budget manager."""
        self.token_tracker = token_tracker or token_tracker
        self.budget_manager = budget_manager or budget_manager
        self.alerts: List[CostAlert] = []

        # Alert thresholds
        self.budget_thresholds = [0.5, 0.75, 0.85, 0.95]  # 50%, 75%, 85%, 95%
        self.cost_spike_threshold = 2.0  # 2x average cost per call
        self.efficiency_drop_threshold = 0.5  # 50% drop in tokens per dollar

        # Data persistence
        self.alerts_file = Path("logs/cost_alerts.json")
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)

        self._load_existing_alerts()

    def track_api_call_with_monitoring(
        self,
        question_id: str,
        model: str,
        task_type: str,
        prompt: str,
        response: str,
        success: bool = True,
        actual_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Track API call with comprehensive monitoring and alerting."""
        # Count tokens
        input_tokens = self.token_tracker.count_tokens(prompt, model)
        output_tokens = self.token_tracker.count_tokens(response, model)

        # Track in token tracker
        token_record = self.token_tracker.track_api_call(
            question_id,
            model,
            task_type,
            input_tokens,
            output_tokens,
            success,
            actual_cost,
        )

        # Record in budget manager
        budget_cost = self.budget_manager.record_cost(
            question_id, model, input_tokens, output_tokens, task_type, success
        )

        # Check for alerts
        self._check_for_alerts()

        return {
            "token_record": token_record,
            "budget_cost": budget_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost": token_record.estimated_cost,
            "success": success,
        }

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive cost and budget status."""
        budget_status = self.budget_manager.get_budget_status()
        usage_summary = self.token_tracker.get_usage_summary()
        efficiency_metrics = self.token_tracker.get_cost_efficiency_metrics()

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
            },
            "tokens": {
                "total_calls": usage_summary["total_calls"],
                "total_tokens": usage_summary["total_tokens"],
                "total_cost": usage_summary["total_cost"],
                "success_rate": usage_summary["success_rate"],
                "by_model": usage_summary["by_model"],
                "by_task_type": usage_summary["by_task_type"],
            },
            "efficiency": efficiency_metrics,
            "alerts": {
                "active_alerts": len(
                    [a for a in self.alerts if self._is_alert_recent(a)]
                ),
                "recent_alerts": [
                    a.to_dict() for a in self.alerts[-5:]
                ],  # Last 5 alerts
            },
        }

    def _check_for_alerts(self):
        """Check for various alert conditions."""
        self._check_budget_thresholds()
        self._check_cost_spikes()
        self._check_efficiency_drops()

    def _check_budget_thresholds(self):
        """Check if budget utilization crosses threshold levels."""
        budget_status = self.budget_manager.get_budget_status()
        utilization = budget_status.utilization_percentage / 100

        for threshold in self.budget_thresholds:
            if utilization >= threshold:
                # Check if we already alerted for this threshold recently
                recent_threshold_alerts = [
                    a
                    for a in self.alerts
                    if a.alert_type == "budget_threshold"
                    and abs(a.threshold_value - threshold) < 0.01
                    and self._is_alert_recent(a, hours=24)
                ]

                if not recent_threshold_alerts:
                    severity = (
                        "critical"
                        if threshold >= 0.95
                        else "warning" if threshold >= 0.85 else "info"
                    )

                    recommendations = {
                        0.5: "Monitor usage closely and consider optimizing model selection",
                        0.75: "Switch to more cost-efficient models for non-critical tasks",
                        0.85: "Enable conservative mode and reduce forecast frequency",
                        0.95: "Enable emergency mode - critical budget threshold reached",
                    }

                    alert = CostAlert(
                        timestamp=datetime.now(),
                        alert_type="budget_threshold",
                        severity=severity,
                        message=f"Budget utilization reached {threshold:.0%}",
                        current_value=utilization,
                        threshold_value=threshold,
                        recommendation=recommendations.get(
                            threshold, "Review budget allocation"
                        ),
                    )

                    self.alerts.append(alert)
                    logger.warning(
                        f"BUDGET ALERT: {alert.message} - {alert.recommendation}"
                    )

    def _check_cost_spikes(self):
        """Check for unusual cost spikes in recent API calls."""
        if len(self.token_tracker.usage_records) < 10:
            return  # Need enough data for comparison

        recent_records = self.token_tracker.usage_records[-10:]
        recent_costs = [r.estimated_cost for r in recent_records if r.success]

        if len(recent_costs) < 5:
            return

        avg_cost = sum(recent_costs) / len(recent_costs)
        latest_cost = recent_costs[-1]

        if latest_cost > avg_cost * self.cost_spike_threshold:
            # Check if we already alerted for cost spikes recently
            recent_spike_alerts = [
                a
                for a in self.alerts
                if a.alert_type == "cost_spike" and self._is_alert_recent(a, hours=1)
            ]

            if not recent_spike_alerts:
                alert = CostAlert(
                    timestamp=datetime.now(),
                    alert_type="cost_spike",
                    severity="warning",
                    message=f"Cost spike detected: ${latest_cost:.4f} vs avg ${avg_cost:.4f}",
                    current_value=latest_cost,
                    threshold_value=avg_cost * self.cost_spike_threshold,
                    recommendation="Review recent API calls for unusually long prompts or responses",
                )

                self.alerts.append(alert)
                logger.warning(f"COST SPIKE ALERT: {alert.message}")

    def _check_efficiency_drops(self):
        """Check for drops in cost efficiency (tokens per dollar)."""
        if len(self.token_tracker.usage_records) < 20:
            return  # Need enough data for trend analysis

        # Compare recent efficiency to historical average
        all_records = [
            r
            for r in self.token_tracker.usage_records
            if r.success and r.estimated_cost > 0
        ]
        if len(all_records) < 20:
            return

        # Historical efficiency (tokens per dollar)
        historical_efficiency = sum(
            r.total_tokens / r.estimated_cost for r in all_records[:-10]
        ) / len(all_records[:-10])

        # Recent efficiency
        recent_records = all_records[-10:]
        recent_efficiency = sum(
            r.total_tokens / r.estimated_cost for r in recent_records
        ) / len(recent_records)

        efficiency_ratio = recent_efficiency / historical_efficiency

        if efficiency_ratio < self.efficiency_drop_threshold:
            # Check if we already alerted for efficiency drops recently
            recent_efficiency_alerts = [
                a
                for a in self.alerts
                if a.alert_type == "efficiency_drop"
                and self._is_alert_recent(a, hours=6)
            ]

            if not recent_efficiency_alerts:
                alert = CostAlert(
                    timestamp=datetime.now(),
                    alert_type="efficiency_drop",
                    severity="warning",
                    message=f"Cost efficiency dropped {(1-efficiency_ratio):.1%}",
                    current_value=recent_efficiency,
                    threshold_value=historical_efficiency
                    * self.efficiency_drop_threshold,
                    recommendation="Review model selection and prompt optimization strategies",
                )

                self.alerts.append(alert)
                logger.warning(f"EFFICIENCY DROP ALERT: {alert.message}")

    def _is_alert_recent(self, alert: CostAlert, hours: int = 24) -> bool:
        """Check if alert is within the specified time window."""
        time_threshold = datetime.now() - timedelta(hours=hours)
        return alert.timestamp > time_threshold

    def get_optimization_recommendations(self) -> List[str]:
        """Get actionable optimization recommendations based on current usage patterns."""
        recommendations = []

        budget_status = self.budget_manager.get_budget_status()
        usage_summary = self.token_tracker.get_usage_summary()
        efficiency_metrics = self.token_tracker.get_cost_efficiency_metrics()

        # Budget-based recommendations
        if budget_status.utilization_percentage > 85:
            recommendations.append(
                "URGENT: Switch to GPT-4o-mini for all non-critical tasks"
            )
            recommendations.append("Reduce forecast frequency to conserve budget")
        elif budget_status.utilization_percentage > 75:
            recommendations.append(
                "Use GPT-4o-mini for research tasks, GPT-4o only for final forecasts"
            )
            recommendations.append("Implement more aggressive prompt optimization")

        # Model efficiency recommendations
        if "model_efficiency" in efficiency_metrics:
            most_expensive = max(
                efficiency_metrics["model_efficiency"].items(),
                key=lambda x: x[1]["cost_per_token"],
                default=(None, None),
            )

            if most_expensive[0] and most_expensive[1]["cost_per_token"] > 0.0001:
                recommendations.append(
                    f"Consider reducing usage of {most_expensive[0]} - highest cost per token"
                )

        # Success rate recommendations
        if usage_summary["success_rate"] < 0.9:
            recommendations.append(
                "Investigate API call failures - low success rate detected"
            )

        # Token efficiency recommendations
        avg_tokens_per_call = usage_summary["total_tokens"]["total"] / max(
            usage_summary["total_calls"], 1
        )
        if avg_tokens_per_call > 3000:
            recommendations.append(
                "Optimize prompts to reduce token usage - current average is high"
            )

        return recommendations

    def log_comprehensive_status(self):
        """Log comprehensive cost monitoring status."""
        status = self.get_comprehensive_status()

        logger.info("=== Comprehensive Cost Status ===")

        # Budget status
        budget = status["budget"]
        logger.info(
            f"Budget: ${budget['spent']:.4f} / ${budget['total']:.2f} "
            f"({budget['utilization_percent']:.1f}%) - {budget['status_level'].upper()}"
        )
        logger.info(
            f"Questions: {budget['questions_processed']} processed, "
            f"~{budget['estimated_questions_remaining']} remaining"
        )

        # Token usage
        tokens = status["tokens"]
        logger.info(
            f"API Calls: {tokens['total_calls']} ({tokens['success_rate']:.1%} success)"
        )
        logger.info(
            f"Tokens: {tokens['total_tokens']['total']:,} total "
            f"(${tokens['total_cost']:.4f} estimated)"
        )

        # Recent alerts
        if status["alerts"]["active_alerts"] > 0:
            logger.warning(f"Active Alerts: {status['alerts']['active_alerts']}")
            for alert_data in status["alerts"]["recent_alerts"]:
                logger.warning(
                    f"  - {alert_data['severity'].upper()}: {alert_data['message']}"
                )

        # Optimization recommendations
        recommendations = self.get_optimization_recommendations()
        if recommendations:
            logger.info("--- Optimization Recommendations ---")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")

    def _save_alerts(self):
        """Save alerts to file."""
        try:
            data = {
                "alerts": [alert.to_dict() for alert in self.alerts],
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.alerts_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def _load_existing_alerts(self):
        """Load existing alerts if available."""
        try:
            if self.alerts_file.exists():
                with open(self.alerts_file, "r") as f:
                    data = json.load(f)

                alerts_data = data.get("alerts", [])
                self.alerts = [
                    CostAlert(
                        timestamp=datetime.fromisoformat(alert["timestamp"]),
                        alert_type=alert["alert_type"],
                        severity=alert["severity"],
                        message=alert["message"],
                        current_value=alert["current_value"],
                        threshold_value=alert["threshold_value"],
                        recommendation=alert["recommendation"],
                    )
                    for alert in alerts_data
                ]

                logger.info(f"Loaded {len(self.alerts)} existing alerts")

        except Exception as e:
            logger.warning(f"Failed to load existing alerts: {e}")


# Global instance
cost_monitor = CostMonitor()
