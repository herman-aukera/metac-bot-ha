"""
Integration service for monitoring system with tournament forecasting components.
Provides seamless integration with existing services.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict

from .budget_dashboard import budget_dashboard
from .comprehensive_monitor import comprehensive_monitor
from .performance_tracker import performance_tracker

logger = logging.getLogger(__name__)


class MonitoringIntegration:
    """Integration service for monitoring system."""

    def __init__(self) -> None:
        """Initialize monitoring integration."""
        self.comprehensive_monitor = comprehensive_monitor
        self.budget_dashboard = budget_dashboard
        self.performance_tracker = performance_tracker
        self._monitoring_started = False

        logger.info("Monitoring integration service initialized")

    def start_monitoring_if_needed(self) -> None:
        """Start monitoring only when explicitly requested."""
        if not self._monitoring_started:
            self.comprehensive_monitor.start_monitoring()
            self._monitoring_started = True
            logger.info("Background monitoring started")

    def track_api_call(
        self,
        question_id: str,
        model: str,
        task_type: str,
        prompt: str,
        response: str,
        success: bool = True,
    ) -> Dict[str, Any]:
        """Track API call with comprehensive monitoring."""
        return self.comprehensive_monitor.track_question_processing(
            question_id, model, task_type, prompt, response, success
        )

    def track_forecast(
        self,
        question_id: str,
        forecast_value: float,
        confidence: float,
        model: str = "ensemble",
    ) -> Dict[str, Any]:
        """Track forecast submission."""
        return self.comprehensive_monitor.track_question_processing(
            question_id, model, "forecast", "", "", True, forecast_value, confidence
        )

    def update_forecast_outcome(self, question_id: str, actual_outcome: float) -> bool:
        """Update forecast with actual outcome."""
        return self.comprehensive_monitor.update_forecast_outcome(
            question_id, actual_outcome
        )

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return self.budget_dashboard.get_real_time_status()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_tracker.get_performance_metrics()
        return metrics.to_dict()

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return self.comprehensive_monitor.get_comprehensive_dashboard()

    def check_budget_availability(self, estimated_cost: float) -> bool:
        """Check if budget is available for estimated cost."""
        return self.budget_dashboard.budget_manager.can_afford(estimated_cost)

    def get_cost_estimate(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Get cost estimate for API call."""
        return self.budget_dashboard.budget_manager.estimate_cost(
            model, input_tokens, output_tokens
        )


def monitor_api_call(question_id: str | None = None, task_type: str = "general") -> Callable:
    """Decorator for monitoring API calls."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            success = True

            try:
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    pass
                elif isinstance(result, dict) and "response" in result:
                    result["response"]
                return result
            except Exception as e:
                success = False
                logger.error(f"API call failed: {e}")
                raise
            finally:
                # Track the API call
                response_time = time.time() - start_time

                # Extract question_id from args/kwargs if not provided
                actual_question_id = question_id or "unknown"
                if actual_question_id == "unknown":
                    if args and hasattr(args[0], "question_id"):
                        actual_question_id = str(args[0].question_id)
                    elif "question_id" in kwargs:
                        actual_question_id = str(kwargs["question_id"])

                monitoring_integration.performance_tracker.record_api_performance(
                    actual_question_id, task_type, success, response_time
                )

        return wrapper

    return decorator


# Global monitoring integration instance
monitoring_integration = MonitoringIntegration()
