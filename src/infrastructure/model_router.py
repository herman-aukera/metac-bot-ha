"""
Budget-Aware Model Router

Intelligent model selection based on budget utilization, task requirements,
and testing environment. Implements automatic fallback to free models when
budget is running low or during CI/testing.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BudgetAwareModelRouter:
    """
    Routes model selection based on budget constraints and task requirements.

    Budget Thresholds:
    - >95%: Free models only
    - >80%: Budget-conscious models (gpt-5-nano/mini)
    - <50%: Premium models allowed (gpt-5)

    Free Models (OpenRouter-routed):
    - "openai/gpt-oss-120b:free" (tools support varies; verify per model meta)
    - "moonshotai/kimi-k2:free" (no tools)
    """

    def __init__(self, budget_limit: float = 100.0):
        self.budget_limit = budget_limit
        self.current_spend = 0.0
        self.load_budget_status()

    def load_budget_status(self) -> None:
        """Load current budget status from environment variables."""
        try:
            self.current_spend = float(os.getenv("CURRENT_SPEND", "0.0"))
            self.budget_limit = float(os.getenv("BUDGET_LIMIT", str(self.budget_limit)))
        except (ValueError, TypeError):
            logger.warning(
                "Could not load budget status from environment, using defaults"
            )
            self.current_spend = 0.0

    def get_budget_utilization(self) -> float:
        """Get current budget utilization as percentage (0-100)."""
        if self.budget_limit <= 0:
            return 0.0
        return (self.current_spend / self.budget_limit) * 100

    def is_test_environment(self) -> bool:
        """Check if running in test/CI environment."""
        return (
            os.getenv("CI") == "true"
            or os.getenv("PYTEST_CURRENT_TEST") is not None
            or os.getenv("GITHUB_ACTIONS") == "true"
            or "test" in os.getenv("ENVIRONMENT", "").lower()
        )

    def select_model(
        self,
        task_type: str,
        requires_tools: bool = False,
        is_test: Optional[bool] = None,
    ) -> str:
        """
        Select appropriate model based on budget and requirements.

        Args:
            task_type: Type of task (research, forecasting, validation, etc.)
            requires_tools: Whether the task requires function calling/tools
            is_test: Override test environment detection

        Returns:
            Model identifier string
        """
        # Force free models in test environments
        if is_test is None:
            is_test = self.is_test_environment()

        if is_test:
            logger.info("Test environment detected, using free model")
            return "openai/gpt-oss-120b:free"

        # Get current budget utilization
        utilization = self.get_budget_utilization()

        # Critical budget situation (>95%) - free models only
        if utilization > 95:
            logger.warning(
                f"Critical budget utilization ({utilization:.1f}%), using free models only"
            )
            if requires_tools:
                return "openai/gpt-oss-120b:free"  # Supports tools
            else:
                return (
                    "moonshotai/kimi-k2:free"  # No tools but potentially better quality
                )

        # High budget utilization (>80%) - budget-conscious models
        elif utilization > 80:
            logger.info(
                f"High budget utilization ({utilization:.1f}%), using budget-conscious models"
            )
            return "openai/gpt-5-nano"

        # Medium budget utilization (>50%) - balanced approach
        elif utilization > 50:
            logger.info(
                f"Medium budget utilization ({utilization:.1f}%), using balanced model selection"
            )
            if task_type in ["research", "validation"]:
                return "openai/gpt-5-nano"  # Use cheaper model for non-critical tasks
            else:
                return "openai/gpt-5"  # Use premium for forecasting

        # Low budget utilization (<50%) - premium models allowed
        else:
            logger.info(
                f"Low budget utilization ({utilization:.1f}%), using premium models"
            )
            if task_type == "forecasting":
                return "openai/gpt-5"  # Best model for critical forecasting
            else:
                return "openai/gpt-5-mini"  # Still efficient for other tasks

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for selected model."""
        model_configs = {
            "openai/gpt-5": {
                "provider": "openrouter",
                "model": "openai/gpt-5",
                "max_tokens": 6000,
                "temperature": 0.1,
                "supports_tools": True,
                "cost_per_1k_tokens": 0.0015,  # See tri-model guidance (verify: PRICES)
            },
            "openai/gpt-5-mini": {
                "provider": "openrouter",
                "model": "openai/gpt-5-mini",
                "max_tokens": 4000,
                "temperature": 0.1,
                "supports_tools": True,
                "cost_per_1k_tokens": 0.00025,
            },
            "openai/gpt-5-nano": {
                "provider": "openrouter",
                "model": "openai/gpt-5-nano",
                "max_tokens": 4000,
                "temperature": 0.1,
                "supports_tools": True,
                "cost_per_1k_tokens": 0.00005,
            },
            "openai/gpt-oss-120b:free": {
                "provider": "openrouter",
                "model": "openai/gpt-oss-120b:free",
                "max_tokens": 2000,
                "temperature": 0.1,
                "supports_tools": True,
                "cost_per_1k_tokens": 0.0,  # Free
            },
            "moonshotai/kimi-k2:free": {
                "provider": "openrouter",
                "model": "moonshotai/kimi-k2:free",
                "max_tokens": 2000,
                "temperature": 0.1,
                "supports_tools": False,
                "cost_per_1k_tokens": 0.0,  # Free
            },
        }

        return model_configs.get(model_name, model_configs["openai/gpt-5-mini"])

    def estimate_cost(self, model_name: str, estimated_tokens: int) -> float:
        """Estimate cost for using a model with given token count."""
        config = self.get_model_config(model_name)
        cost_per_1k = config.get("cost_per_1k_tokens", 0.0015)
        return (estimated_tokens / 1000) * cost_per_1k

    def can_afford_model(self, model_name: str, estimated_tokens: int) -> bool:
        """Check if we can afford to use a model for estimated tokens."""
        estimated_cost = self.estimate_cost(model_name, estimated_tokens)
        remaining_budget = self.budget_limit - self.current_spend
        return estimated_cost <= remaining_budget

    def get_fallback_model(
        self, original_model: str, requires_tools: bool = False
    ) -> str:
        """Get fallback model if original model is too expensive."""
        if requires_tools:
            return "openai/gpt-oss-120b:free"
        else:
            return "moonshotai/kimi-k2:free"

    def select_model_with_fallback(
        self,
        task_type: str,
        requires_tools: bool = False,
        estimated_tokens: int = 1000,
        is_test: Optional[bool] = None,
    ) -> str:
        """
        Select model with automatic fallback if budget insufficient.

        Args:
            task_type: Type of task
            requires_tools: Whether tools are required
            estimated_tokens: Estimated token usage
            is_test: Override test environment detection

        Returns:
            Model identifier with fallback applied if needed
        """
        # Get primary model selection
        primary_model = self.select_model(task_type, requires_tools, is_test)

        # Check if we can afford it
        if self.can_afford_model(primary_model, estimated_tokens):
            return primary_model

        # Fall back to free model
        fallback_model = self.get_fallback_model(primary_model, requires_tools)
        logger.warning(
            f"Cannot afford {primary_model} (${self.estimate_cost(primary_model, estimated_tokens):.4f}), "
            f"falling back to {fallback_model}"
        )
        return fallback_model

    def log_model_usage(
        self, model_name: str, tokens_used: int, actual_cost: Optional[float] = None
    ) -> None:
        """Log model usage for budget tracking."""
        if actual_cost is None:
            actual_cost = self.estimate_cost(model_name, tokens_used)

        self.current_spend += actual_cost

        logger.info(
            f"Model usage: {model_name}, tokens: {tokens_used}, "
            f"cost: ${actual_cost:.4f}, total spend: ${self.current_spend:.2f}"
        )

        # Update environment variable for other processes
        os.environ["CURRENT_SPEND"] = str(self.current_spend)

    def get_budget_status(self) -> Dict[str, Any]:
        """Get comprehensive budget status."""
        utilization = self.get_budget_utilization()
        remaining = self.budget_limit - self.current_spend

        return {
            "budget_limit": self.budget_limit,
            "current_spend": self.current_spend,
            "remaining_budget": remaining,
            "utilization_percent": utilization,
            "operation_mode": self._get_operation_mode(utilization),
            "recommended_models": self._get_recommended_models(utilization),
            "is_test_environment": self.is_test_environment(),
        }

    def _get_operation_mode(self, utilization: float) -> str:
        """Get current operation mode based on utilization."""
        if utilization > 95:
            return "critical"
        elif utilization > 80:
            return "emergency"
        elif utilization > 50:
            return "conservative"
        else:
            return "normal"

    def _get_recommended_models(self, utilization: float) -> Dict[str, str]:
        """Get recommended models for different tasks based on utilization."""
        if utilization > 95:
            return {
                "research": "openai/gpt-oss-120b:free",
                "forecasting": "openai/gpt-oss-120b:free",
                "validation": "moonshotai/kimi-k2:free",
            }
        elif utilization > 80:
            return {
                "research": "openai/gpt-5-nano",
                "forecasting": "openai/gpt-5-nano",
                "validation": "openai/gpt-5-nano",
            }
        else:
            return {
                "research": "openai/gpt-5-mini",
                "forecasting": "openai/gpt-5",
                "validation": "openai/gpt-5-nano",
            }


# Global instance for easy access
_global_router = None


def get_model_router() -> BudgetAwareModelRouter:
    """Get global model router instance."""
    global _global_router
    if _global_router is None:
        _global_router = BudgetAwareModelRouter()
    return _global_router


def select_model_for_task(task_type: str, requires_tools: bool = False) -> str:
    """Convenience function to select model for a task."""
    router = get_model_router()
    return router.select_model_with_fallback(task_type, requires_tools)
