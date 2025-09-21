"""
Simple interface for the comprehensive error handling and recovery system.
Provides easy-to-use functions for integrating error handling into the forecasting system.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from .comprehensive_error_recovery import (
    ComprehensiveErrorRecoveryManager,
    RecoveryConfiguration,
)
from .error_classification import ErrorContext

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Simple interface for comprehensive error handling and recovery.

    This class provides a clean, easy-to-use interface for integrating
    the comprehensive error handling system into the forecasting pipeline.
    """

    def __init__(
        self,
        tri_model_router=None,
        budget_manager=None,
        config: Optional[RecoveryConfiguration] = None,
    ):
        """
        Initialize error handler with recovery system.

        Args:
            tri_model_router: The tri-model router instance
            budget_manager: The budget manager instance
            config: Recovery configuration (uses defaults if None)
        """
        self.recovery_manager = ComprehensiveErrorRecoveryManager(
            tri_model_router, budget_manager, config
        )
        self._initialized = True

        logger.info("Error handler initialized with comprehensive recovery system")

    async def handle_error(
        self,
        error: Exception,
        task_type: str = "forecast",
        model_tier: str = "mini",
        operation_mode: str = "normal",
        budget_remaining: float = 50.0,
        attempt_number: int = 1,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        original_prompt: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle an error with comprehensive recovery strategies.

        Args:
            error: The exception that occurred
            task_type: Type of task being performed
            model_tier: Model tier being used
            operation_mode: Current operation mode
            budget_remaining: Remaining budget percentage
            attempt_number: Current attempt number
            model_name: Name of the model that failed
            provider: Provider that failed
            original_prompt: Original prompt (for revision strategies)

        Returns:
            Tuple of (success, recovery_info) where recovery_info contains:
            - strategy: Recovery strategy used
            - recovery_time: Time taken for recovery
            - performance_impact: Performance impact (0.0-1.0)
            - cost_impact: Cost impact (-1.0 to 1.0)
            - message: Human-readable message
            - metadata: Additional recovery metadata
        """
        if not self._initialized:
            raise RuntimeError("Error handler not initialized")

        # Create error context
        context = ErrorContext(
            task_type=task_type,
            model_tier=model_tier,
            operation_mode=operation_mode,
            budget_remaining=budget_remaining,
            attempt_number=attempt_number,
            model_name=model_name,
            provider=provider,
            original_prompt=original_prompt,
        )

        try:
            # Execute comprehensive recovery
            result = await self.recovery_manager.recover_from_error(error, context)

            # Format recovery information
            recovery_info = {
                "strategy": result.recovery_strategy.value,
                "recovery_time": result.recovery_time,
                "performance_impact": result.performance_impact,
                "cost_impact": result.cost_impact,
                "message": result.message,
                "attempts_made": result.attempts_made,
                "metadata": result.metadata,
            }

            if result.fallback_result:
                recovery_info["fallback_details"] = {
                    "fallback_used": (
                        result.fallback_result.fallback_used.name
                        if result.fallback_result.fallback_used
                        else None
                    ),
                    "fallback_tier": (
                        result.fallback_result.fallback_used.tier.value
                        if result.fallback_result.fallback_used
                        else None
                    ),
                    "performance_level": (
                        result.fallback_result.fallback_used.performance_level.value
                        if result.fallback_result.fallback_used
                        else None
                    ),
                }

            logger.info(
                f"Error recovery {'successful' if result.success else 'failed'}: "
                f"{result.recovery_strategy.value} in {result.recovery_time:.2f}s"
            )

            return result.success, recovery_info

        except Exception as recovery_error:
            logger.error(f"Error recovery failed with exception: {recovery_error}")

            return False, {
                "strategy": "abort",
                "recovery_time": 0.0,
                "performance_impact": 1.0,
                "cost_impact": 0.0,
                "message": f"Recovery failed: {recovery_error}",
                "attempts_made": 0,
                "metadata": {"recovery_error": str(recovery_error)},
            }

    async def handle_model_error(
        self,
        error: Exception,
        model_name: str,
        model_tier: str,
        budget_remaining: float = 50.0,
        attempt_number: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle model-specific errors with optimized recovery.

        Args:
            error: The model error that occurred
            model_name: Name of the failed model
            model_tier: Tier of the failed model
            budget_remaining: Remaining budget percentage
            attempt_number: Current attempt number

        Returns:
            Tuple of (success, recovery_info)
        """
        return await self.handle_error(
            error=error,
            task_type="forecast",
            model_tier=model_tier,
            operation_mode="normal",
            budget_remaining=budget_remaining,
            attempt_number=attempt_number,
            model_name=model_name,
            provider="openrouter",
        )

    async def handle_budget_error(
        self,
        error: Exception,
        budget_remaining: float,
        operation_mode: str = "critical",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle budget-related errors with emergency protocols.

        Args:
            error: The budget error that occurred
            budget_remaining: Remaining budget percentage
            operation_mode: Current operation mode

        Returns:
            Tuple of (success, recovery_info)
        """
        return await self.handle_error(
            error=error,
            task_type="forecast",
            model_tier="nano",  # Use cheapest tier for budget errors
            operation_mode=operation_mode,
            budget_remaining=budget_remaining,
            attempt_number=1,
        )

    async def handle_api_error(
        self,
        error: Exception,
        provider: str,
        status_code: Optional[int] = None,
        budget_remaining: float = 50.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle API-related errors with provider fallbacks.

        Args:
            error: The API error that occurred
            provider: Provider that failed
            status_code: HTTP status code (if applicable)
            budget_remaining: Remaining budget percentage

        Returns:
            Tuple of (success, recovery_info)
        """
        return await self.handle_error(
            error=error,
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=budget_remaining,
            attempt_number=1,
            provider=provider,
        )

    async def handle_quality_error(
        self,
        error: Exception,
        original_prompt: str,
        quality_issues: Optional[list] = None,
        budget_remaining: float = 50.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle quality validation errors with prompt revision.

        Args:
            error: The quality error that occurred
            original_prompt: Original prompt that failed quality checks
            quality_issues: List of specific quality issues
            budget_remaining: Remaining budget percentage

        Returns:
            Tuple of (success, recovery_info)
        """
        return await self.handle_error(
            error=error,
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=budget_remaining,
            attempt_number=1,
            original_prompt=original_prompt,
        )

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current error handling system status.

        Returns:
            Dictionary with system status information
        """
        if not self._initialized:
            return {"status": "not_initialized"}

        return self.recovery_manager.get_recovery_status()

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get error statistics for the specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with error statistics
        """
        if not self._initialized:
            return {"error": "not_initialized"}

        # Get statistics from logging system
        logging_system = self.recovery_manager.fallback_orchestrator.logging_system
        return logging_system.get_error_summary(hours=hours)

    def is_emergency_mode_active(self) -> bool:
        """
        Check if emergency mode is currently active.

        Returns:
            True if emergency mode is active, False otherwise
        """
        if not self._initialized:
            return False

        return (
            self.recovery_manager.fallback_orchestrator.emergency_manager.is_emergency_active()
        )

    async def deactivate_emergency_mode(self) -> bool:
        """
        Attempt to deactivate emergency mode.

        Returns:
            True if successfully deactivated, False otherwise
        """
        if not self._initialized:
            return False

        return (
            await self.recovery_manager.fallback_orchestrator.emergency_manager.deactivate_emergency_mode()
        )

    async def test_system(self) -> Dict[str, Any]:
        """
        Test the error handling system functionality.

        Returns:
            Dictionary with test results
        """
        if not self._initialized:
            return {"error": "not_initialized"}

        return await self.recovery_manager.test_recovery_system()

    def get_recovery_recommendations(
        self, error_type: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get recovery recommendations for a specific error type and context.

        Args:
            error_type: Type of error (e.g., "model_error", "budget_error")
            context: Context information

        Returns:
            Dictionary with recovery recommendations
        """
        recommendations = {
            "model_error": {
                "primary_strategy": "fallback_model",
                "secondary_strategy": "fallback_provider",
                "considerations": [
                    "Check model availability",
                    "Consider tier downgrade",
                    "Monitor performance impact",
                ],
            },
            "budget_error": {
                "primary_strategy": "emergency_mode",
                "secondary_strategy": "budget_conservation",
                "considerations": [
                    "Activate free models only",
                    "Reduce functionality",
                    "Monitor remaining budget",
                ],
            },
            "api_error": {
                "primary_strategy": "fallback_provider",
                "secondary_strategy": "retry",
                "considerations": [
                    "Check provider status",
                    "Use exponential backoff",
                    "Consider alternative APIs",
                ],
            },
            "quality_error": {
                "primary_strategy": "prompt_revision",
                "secondary_strategy": "fallback_model",
                "considerations": [
                    "Enhance quality directives",
                    "Add citation requirements",
                    "Consider model upgrade",
                ],
            },
        }

        return recommendations.get(
            error_type,
            {
                "primary_strategy": "retry",
                "secondary_strategy": "fallback_model",
                "considerations": [
                    "Analyze error pattern",
                    "Consider context",
                    "Monitor recovery success",
                ],
            },
        )


# Convenience functions for common error handling scenarios


async def handle_forecasting_error(
    error: Exception, tri_model_router=None, budget_manager=None, **kwargs
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to handle forecasting errors.

    Args:
        error: The error that occurred
        tri_model_router: Tri-model router instance
        budget_manager: Budget manager instance
        **kwargs: Additional context parameters

    Returns:
        Tuple of (success, recovery_info)
    """
    handler = ErrorHandler(tri_model_router, budget_manager)
    return await handler.handle_error(error, **kwargs)


async def handle_model_failure(
    model_name: str,
    error: Exception,
    tri_model_router=None,
    budget_remaining: float = 50.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to handle model failures.

    Args:
        model_name: Name of the failed model
        error: The error that occurred
        tri_model_router: Tri-model router instance
        budget_remaining: Remaining budget percentage

    Returns:
        Tuple of (success, recovery_info)
    """
    handler = ErrorHandler(tri_model_router)

    # Determine model tier from name
    if "gpt-5" in model_name and "mini" not in model_name and "nano" not in model_name:
        tier = "full"
    elif "mini" in model_name:
        tier = "mini"
    elif "nano" in model_name:
        tier = "nano"
    else:
        tier = "mini"  # Default

    return await handler.handle_model_error(error, model_name, tier, budget_remaining)


async def handle_budget_exhaustion(
    budget_remaining: float, tri_model_router=None, budget_manager=None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to handle budget exhaustion.

    Args:
        budget_remaining: Remaining budget percentage
        tri_model_router: Tri-model router instance
        budget_manager: Budget manager instance

    Returns:
        Tuple of (success, recovery_info)
    """
    handler = ErrorHandler(tri_model_router, budget_manager)

    from .error_classification import BudgetError

    error = BudgetError(
        f"Budget critically low: {budget_remaining}%", budget_remaining, 0.0
    )

    return await handler.handle_budget_error(error, budget_remaining, "critical")


def create_error_handler(
    tri_model_router=None,
    budget_manager=None,
    max_recovery_attempts: int = 3,
    max_recovery_time: float = 120.0,
    enable_emergency_mode: bool = True,
) -> ErrorHandler:
    """
    Factory function to create a configured error handler.

    Args:
        tri_model_router: Tri-model router instance
        budget_manager: Budget manager instance
        max_recovery_attempts: Maximum recovery attempts
        max_recovery_time: Maximum recovery time in seconds
        enable_emergency_mode: Whether to enable emergency mode

    Returns:
        Configured ErrorHandler instance
    """
    config = RecoveryConfiguration(
        max_recovery_attempts=max_recovery_attempts,
        max_recovery_time=max_recovery_time,
        enable_emergency_mode=enable_emergency_mode,
        enable_circuit_breakers=True,
        enable_quality_recovery=True,
        budget_threshold_for_emergency=5.0,
    )

    return ErrorHandler(tri_model_router, budget_manager, config)
