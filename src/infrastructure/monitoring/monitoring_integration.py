"""
Integration service for monitoring system with tournament forecasting components.
Provides seamless integration with existing services.
"""
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
import time

from .comprehensive_monitor import comprehensive_monitor
from .budget_dashboard import budget_dashboard
from .performance_tracker import performance_tracker

logger = logging.getLogger(__name__)


class MonitoringIntegration:
    """Integration service for monitoring system."""

    def __init__(self):
        """Initialize monitoring integration."""
        self.comprehensive_monitor = comprehensive_monitor
        self.budget_dashboard = budget_dashboard
        self.performance_tracker = performance_tracker

        # Start monitoring automatically
        self.comprehensive_monitor.start_monitoring()

        logger.info("Monitoring integration service initialized")

    def track_api_call(self, question_id: str, model: str, task_type: str,
                      prompt: str, response: str, success: bool = True) -> Dict[str, Any]:
        """Track API call with comprehensive monitoring."""
        return self.comprehensive_monitor.track_question_processing(
            question_id, model, task_type, prompt, response, success
        )

    def track_forecast(self, question_id: str, forecast_value: float,
                      confidence: float, model: str = "ensemble") -> Dict[str, Any]:
        """Track forecast submission."""
        return self.comprehensive_monitor.track_question_processing(
            question_id, model, "forecast", "", "", True, forecast_value, confidence
        )

    def update_forecast_outcome(self, question_id: str, actual_outcome: float) -> bool:
        """Update forecast with actual outcome."""
        return self.comprehensive_monitor.update_forecast_outcome(question_id, actual_outcome)

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

    def get_cost_estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Get cost estimate for API call."""
        return self.budget_dashboard.budget_manager.estimate_cost(model, input_tokens, output_tokens)


def monitor_api_call(question_id: str = None, task_type: str = "general"):
    """Decorator for monitoring API calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            response = ""

            try:
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    response = result
                elif isinstance(result, dict) and 'response' in result:
                    response = result['response']
                return result
            except Exception as e:
                success = False
                logger.error(f"API call failed: {e}")
                raise
            finally:
                # Track the API call
                response_time = time.time() - start_time

                # Extract question_id from args/kwargs if not provided
                actual_question_id = question_id
                if not actual_question_id:
                    if args and hasattr(args[0], 'question_id'):
                        actual_question_id = args[0].question_id
                    elif 'question_id' in kwargs:
                        actual_question_id = kwargs['question_id']
                    else:
                        actual_question_id = "unknown"

                monitoring_integration.performance_tracker.record_api_performance(
                    actual_question_id, task_type, success, response_time
                )

        return wrapper
    return decorator


# Global monitoring integration instance
monitoring_integration = MonitoringIntegration()
