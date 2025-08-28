"""
Comprehensive error recovery manager that integrates error classification,
intelligent fallback strategies, and recovery orchestration for the OpenRouter
tri-model optimization system.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from .error_classification import (
    ErrorClassifier, ErrorRecoveryManager, ErrorContext, ErrorCategory,
    ErrorSeverity, RecoveryStrategy, RecoveryAction, ForecastingError,
    ModelError, BudgetError, APIError, QualityError
)
from .fallback_strategies import (
    IntelligentFallbackOrchestrator, FallbackResult, FallbackTier,
    PerformanceLevel, AlertConfig
)

logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    """Complete result of error recovery operation."""
    success: bool
    recovery_strategy: RecoveryStrategy
    fallback_result: Optional[FallbackResult]
    recovery_time: float
    attempts_made: int
    final_error: Optional[Exception]
    performance_impact: float
    cost_impact: float
    message: str
    metadata: Dict[str, Any]


@dataclass
class RecoveryConfiguration:
    """Configuration for error recovery behavior."""
    max_recovery_attempts: int = 3
    max_recovery_time: float = 300.0  # 5 minutes
    enable_circuit_breakers: bool = True
    enable_emergency_mode: bool = True
    enable_quality_recovery: bool = True
    budget_threshold_for_emergency: float = 5.0  # 5% remaining
    alert_config: Optional[AlertConfig] = None


class ComprehensiveErrorRecoveryManager:
    """
    Comprehensive error recovery manager that orchestrates all recovery strategies
    for the OpenRouter tri-model optimization system.
    """

    def __init__(self, tri_model_router=None, budget_manager=None,
                 config: Optional[RecoveryConfiguration] = None):
        """
        Initialize comprehensive error recovery manager.

        Args:
            tri_model_router: The tri-model router instance
            budget_manager: The budget manager instance
            config: Recovery configuration
        """
        self.tri_model_router = tri_model_router
        self.budget_manager = budget_manager
        self.config = config or RecoveryConfiguration()

        # Initialize recovery components
        self.error_classifier = ErrorClassifier()
        self.error_recovery_manager = ErrorRecoveryManager(tri_model_router, budget_manager)
        self.fallback_orchestrator = IntelligentFallbackOrchestrator(tri_model_router, budget_manager)

        # Recovery tracking
        self.recovery_history = []
        self.active_recoveries = {}
        self.recovery_statistics = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "average_recovery_time": 0.0,
            "strategy_effectiveness": {}
        }

        logger.info("Comprehensive error recovery manager initialized")

    async def recover_from_error(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """
        Main entry point for error recovery with comprehensive strategy orchestration.

        Args:
            error: The exception that occurred
            context: Error context information

        Returns:
            RecoveryResult with complete recovery details
        """
        recovery_start_time = time.time()
        recovery_id = f"recovery_{int(recovery_start_time)}_{id(error)}"

        logger.info(f"Starting error recovery {recovery_id} for {type(error).__name__}: {error}")

        # Track active recovery
        self.active_recoveries[recovery_id] = {
            "start_time": recovery_start_time,
            "error": error,
            "context": context,
            "attempts": 0
        }

        try:
            # Execute recovery with comprehensive strategy
            recovery_result = await self._execute_comprehensive_recovery(error, context, recovery_id)

            # Update statistics
            self._update_recovery_statistics(recovery_result)

            # Record recovery in history
            self._record_recovery_attempt(recovery_result, recovery_id)

            return recovery_result

        except Exception as recovery_error:
            logger.error(f"Recovery {recovery_id} failed with exception: {recovery_error}")

            # Create failure result
            recovery_time = time.time() - recovery_start_time
            failure_result = RecoveryResult(
                success=False,
                recovery_strategy=RecoveryStrategy.ABORT,
                fallback_result=None,
                recovery_time=recovery_time,
                attempts_made=self.active_recoveries[recovery_id]["attempts"],
                final_error=recovery_error,
                performance_impact=1.0,
                cost_impact=0.0,
                message=f"Recovery failed with exception: {recovery_error}",
                metadata={"recovery_id": recovery_id, "original_error": str(error)}
            )

            self._update_recovery_statistics(failure_result)
            return failure_result

        finally:
            # Clean up active recovery tracking
            if recovery_id in self.active_recoveries:
                del self.active_recoveries[recovery_id]

    async def _execute_comprehensive_recovery(self, error: Exception, context: ErrorContext,
                                           recovery_id: str) -> RecoveryResult:
        """Execute comprehensive recovery strategy with multiple approaches."""
        recovery_start_time = time.time()
        attempts_made = 0
        last_error = error

        # Classify the error
        error_classification = self.error_classifier.classify_error(error, context)
        logger.info(f"Error classified as: {error_classification.error_code} - {error_classification.description}")

        # Check if we should attempt recovery
        if not self._should_attempt_recovery(error_classification, context):
            return RecoveryResult(
                success=False,
                recovery_strategy=RecoveryStrategy.ABORT,
                fallback_result=None,
                recovery_time=time.time() - recovery_start_time,
                attempts_made=0,
                final_error=error,
                performance_impact=1.0,
                cost_impact=0.0,
                message="Recovery not attempted due to conditions",
                metadata={"error_classification": error_classification.error_code}
            )

        # Try recovery strategies in order of preference
        for strategy in error_classification.recovery_strategies:
            if attempts_made >= self.config.max_recovery_attempts:
                logger.warning(f"Max recovery attempts ({self.config.max_recovery_attempts}) reached")
                break

            if time.time() - recovery_start_time > self.config.max_recovery_time:
                logger.warning(f"Max recovery time ({self.config.max_recovery_time}s) exceeded")
                break

            attempts_made += 1
            self.active_recoveries[recovery_id]["attempts"] = attempts_made

            logger.info(f"Attempting recovery strategy: {strategy.value} (attempt {attempts_made})")

            try:
                # Execute specific recovery strategy
                strategy_result = await self._execute_recovery_strategy(
                    strategy, error_classification, context, last_error
                )

                if strategy_result.success:
                    recovery_time = time.time() - recovery_start_time

                    logger.info(f"Recovery successful using strategy: {strategy.value} "
                              f"in {recovery_time:.2f}s after {attempts_made} attempts")

                    return RecoveryResult(
                        success=True,
                        recovery_strategy=strategy,
                        fallback_result=strategy_result.fallback_result,
                        recovery_time=recovery_time,
                        attempts_made=attempts_made,
                        final_error=None,
                        performance_impact=strategy_result.performance_impact,
                        cost_impact=strategy_result.cost_impact,
                        message=f"Recovery successful using {strategy.value}",
                        metadata={
                            "recovery_id": recovery_id,
                            "error_classification": error_classification.error_code,
                            "strategy_details": strategy_result.metadata
                        }
                    )
                else:
                    logger.warning(f"Recovery strategy {strategy.value} failed: {strategy_result.message}")
                    last_error = strategy_result.final_error or last_error

                    # Add delay before next attempt if specified
                    if error_classification.retry_delay > 0:
                        delay = self.error_classifier.calculate_retry_delay(error_classification, attempts_made)
                        logger.info(f"Waiting {delay:.1f}s before next recovery attempt")
                        await asyncio.sleep(delay)

            except Exception as strategy_error:
                logger.error(f"Recovery strategy {strategy.value} raised exception: {strategy_error}")
                last_error = strategy_error
                continue

        # All recovery strategies failed
        recovery_time = time.time() - recovery_start_time

        logger.error(f"All recovery strategies failed after {attempts_made} attempts in {recovery_time:.2f}s")

        return RecoveryResult(
            success=False,
            recovery_strategy=RecoveryStrategy.ABORT,
            fallback_result=None,
            recovery_time=recovery_time,
            attempts_made=attempts_made,
            final_error=last_error,
            performance_impact=1.0,
            cost_impact=0.0,
            message="All recovery strategies failed",
            metadata={
                "recovery_id": recovery_id,
                "error_classification": error_classification.error_code,
                "strategies_attempted": [s.value for s in error_classification.recovery_strategies[:attempts_made]]
            }
        )

    async def _execute_recovery_strategy(self, strategy: RecoveryStrategy,
                                       error_classification, context: ErrorContext,
                                       original_error: Exception) -> RecoveryResult:
        """Execute a specific recovery strategy."""

        if strategy == RecoveryStrategy.RETRY:
            return await self._execute_retry_strategy(error_classification, context, original_error)

        elif strategy == RecoveryStrategy.FALLBACK_MODEL:
            return await self._execute_model_fallback_strategy(context, original_error)

        elif strategy == RecoveryStrategy.FALLBACK_PROVIDER:
            return await self._execute_provider_fallback_strategy(context, original_error)

        elif strategy == RecoveryStrategy.PROMPT_REVISION:
            return await self._execute_prompt_revision_strategy(error_classification, context, original_error)

        elif strategy == RecoveryStrategy.EMERGENCY_MODE:
            return await self._execute_emergency_mode_strategy(context, original_error)

        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._execute_graceful_degradation_strategy(context, original_error)

        elif strategy == RecoveryStrategy.BUDGET_CONSERVATION:
            return await self._execute_budget_conservation_strategy(context, original_error)

        else:
            return RecoveryResult(
                success=False,
                recovery_strategy=strategy,
                fallback_result=None,
                recovery_time=0.0,
                attempts_made=1,
                final_error=original_error,
                performance_impact=1.0,
                cost_impact=0.0,
                message=f"Unknown recovery strategy: {strategy.value}",
                metadata={}
            )

    async def _execute_retry_strategy(self, error_classification, context: ErrorContext,
                                    original_error: Exception) -> RecoveryResult:
        """Execute retry recovery strategy."""
        if not self.error_classifier.should_retry(error_classification, context.attempt_number):
            return RecoveryResult(
                success=False,
                recovery_strategy=RecoveryStrategy.RETRY,
                fallback_result=None,
                recovery_time=0.0,
                attempts_made=1,
                final_error=original_error,
                performance_impact=1.0,
                cost_impact=0.0,
                message="Retry not recommended for this error type",
                metadata={}
            )

        # Calculate retry delay
        retry_delay = self.error_classifier.calculate_retry_delay(error_classification, context.attempt_number)

        # Wait before retry
        await asyncio.sleep(retry_delay)

        # For retry strategy, we indicate success but let the caller handle the actual retry
        return RecoveryResult(
            success=True,
            recovery_strategy=RecoveryStrategy.RETRY,
            fallback_result=None,
            recovery_time=retry_delay,
            attempts_made=1,
            final_error=None,
            performance_impact=1.0,  # No performance impact for retry
            cost_impact=0.0,  # No additional cost for retry
            message=f"Retry recommended after {retry_delay:.1f}s delay",
            metadata={"retry_delay": retry_delay, "attempt_number": context.attempt_number + 1}
        )

    async def _execute_model_fallback_strategy(self, context: ErrorContext,
                                             original_error: Exception) -> RecoveryResult:
        """Execute model fallback recovery strategy."""
        fallback_result = await self.fallback_orchestrator.model_fallback_manager.execute_fallback(
            context.model_tier, context, context.budget_remaining
        )

        return RecoveryResult(
            success=fallback_result.success,
            recovery_strategy=RecoveryStrategy.FALLBACK_MODEL,
            fallback_result=fallback_result,
            recovery_time=fallback_result.recovery_time,
            attempts_made=1,
            final_error=original_error if not fallback_result.success else None,
            performance_impact=fallback_result.performance_impact,
            cost_impact=fallback_result.cost_impact,
            message=fallback_result.message,
            metadata=fallback_result.metadata
        )

    async def _execute_provider_fallback_strategy(self, context: ErrorContext,
                                                original_error: Exception) -> RecoveryResult:
        """Execute provider fallback recovery strategy."""
        fallback_result = await self.fallback_orchestrator.provider_fallback_manager.execute_provider_fallback(
            context.provider or "openrouter", context
        )

        return RecoveryResult(
            success=fallback_result.success,
            recovery_strategy=RecoveryStrategy.FALLBACK_PROVIDER,
            fallback_result=fallback_result,
            recovery_time=fallback_result.recovery_time,
            attempts_made=1,
            final_error=original_error if not fallback_result.success else None,
            performance_impact=fallback_result.performance_impact,
            cost_impact=fallback_result.cost_impact,
            message=fallback_result.message,
            metadata=fallback_result.metadata
        )

    async def _execute_prompt_revision_strategy(self, error_classification, context: ErrorContext,
                                              original_error: Exception) -> RecoveryResult:
        """Execute prompt revision recovery strategy."""
        # Use the error recovery manager's prompt revision capability
        recovery_action = await self.error_recovery_manager._revise_prompt(context, error_classification)

        if recovery_action:
            return RecoveryResult(
                success=True,
                recovery_strategy=RecoveryStrategy.PROMPT_REVISION,
                fallback_result=None,
                recovery_time=2.0,  # Estimated time for prompt revision
                attempts_made=1,
                final_error=None,
                performance_impact=0.9,  # Slight performance impact from revision
                cost_impact=0.0,  # No additional cost
                message="Prompt successfully revised",
                metadata={"revised_prompt": recovery_action}
            )
        else:
            return RecoveryResult(
                success=False,
                recovery_strategy=RecoveryStrategy.PROMPT_REVISION,
                fallback_result=None,
                recovery_time=1.0,
                attempts_made=1,
                final_error=original_error,
                performance_impact=1.0,
                cost_impact=0.0,
                message="Prompt revision not possible",
                metadata={}
            )

    async def _execute_emergency_mode_strategy(self, context: ErrorContext,
                                             original_error: Exception) -> RecoveryResult:
        """Execute emergency mode recovery strategy."""
        recovery_action = await self.fallback_orchestrator.emergency_manager.activate_emergency_mode(
            original_error, context
        )

        return RecoveryResult(
            success=True,
            recovery_strategy=RecoveryStrategy.EMERGENCY_MODE,
            fallback_result=None,
            recovery_time=recovery_action.expected_delay,
            attempts_made=1,
            final_error=None,
            performance_impact=0.3,  # Significant performance reduction
            cost_impact=-0.8,  # Significant cost savings
            message="Emergency mode activated",
            metadata=recovery_action.parameters
        )

    async def _execute_graceful_degradation_strategy(self, context: ErrorContext,
                                                   original_error: Exception) -> RecoveryResult:
        """Execute graceful degradation recovery strategy."""
        # Implement graceful degradation by reducing functionality
        degradation_config = {
            "reduced_functionality": True,
            "essential_features_only": True,
            "simplified_prompts": True,
            "shorter_responses": True
        }

        return RecoveryResult(
            success=True,
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            fallback_result=None,
            recovery_time=1.0,
            attempts_made=1,
            final_error=None,
            performance_impact=0.6,  # Moderate performance reduction
            cost_impact=-0.3,  # Some cost savings
            message="Graceful degradation activated",
            metadata=degradation_config
        )

    async def _execute_budget_conservation_strategy(self, context: ErrorContext,
                                                  original_error: Exception) -> RecoveryResult:
        """Execute budget conservation recovery strategy."""
        # Implement budget conservation measures
        conservation_config = {
            "use_cheaper_models": True,
            "reduce_token_limits": True,
            "prioritize_essential_tasks": True,
            "disable_expensive_features": True
        }

        return RecoveryResult(
            success=True,
            recovery_strategy=RecoveryStrategy.BUDGET_CONSERVATION,
            fallback_result=None,
            recovery_time=1.0,
            attempts_made=1,
            final_error=None,
            performance_impact=0.7,  # Moderate performance impact
            cost_impact=-0.5,  # Significant cost savings
            message="Budget conservation measures activated",
            metadata=conservation_config
        )

    def _should_attempt_recovery(self, error_classification, context: ErrorContext) -> bool:
        """Determine if recovery should be attempted based on conditions."""

        # Don't attempt recovery if circuit breaker is open
        if (self.config.enable_circuit_breakers and
            self.error_recovery_manager._is_circuit_breaker_open(error_classification.error_code)):
            logger.warning(f"Circuit breaker open for {error_classification.error_code}, skipping recovery")
            return False

        # Don't attempt recovery for critical budget errors if emergency mode is disabled
        if (error_classification.category == ErrorCategory.BUDGET_ERROR and
            error_classification.severity == ErrorSeverity.CRITICAL and
            not self.config.enable_emergency_mode):
            logger.warning("Budget emergency detected but emergency mode disabled")
            return False

        # Don't attempt recovery if too many recent attempts
        if context.attempt_number > self.config.max_recovery_attempts:
            logger.warning(f"Too many recovery attempts ({context.attempt_number})")
            return False

        return True

    def _update_recovery_statistics(self, recovery_result: RecoveryResult):
        """Update recovery statistics based on result."""
        self.recovery_statistics["total_recoveries"] += 1

        if recovery_result.success:
            self.recovery_statistics["successful_recoveries"] += 1
        else:
            self.recovery_statistics["failed_recoveries"] += 1

        # Update average recovery time
        total_recoveries = self.recovery_statistics["total_recoveries"]
        current_avg = self.recovery_statistics["average_recovery_time"]
        new_avg = ((current_avg * (total_recoveries - 1)) + recovery_result.recovery_time) / total_recoveries
        self.recovery_statistics["average_recovery_time"] = new_avg

        # Update strategy effectiveness
        strategy = recovery_result.recovery_strategy.value
        if strategy not in self.recovery_statistics["strategy_effectiveness"]:
            self.recovery_statistics["strategy_effectiveness"][strategy] = {
                "attempts": 0,
                "successes": 0,
                "success_rate": 0.0,
                "avg_recovery_time": 0.0
            }

        strategy_stats = self.recovery_statistics["strategy_effectiveness"][strategy]
        strategy_stats["attempts"] += 1

        if recovery_result.success:
            strategy_stats["successes"] += 1

        strategy_stats["success_rate"] = strategy_stats["successes"] / strategy_stats["attempts"]

        # Update average recovery time for strategy
        strategy_attempts = strategy_stats["attempts"]
        current_strategy_avg = strategy_stats["avg_recovery_time"]
        new_strategy_avg = ((current_strategy_avg * (strategy_attempts - 1)) + recovery_result.recovery_time) / strategy_attempts
        strategy_stats["avg_recovery_time"] = new_strategy_avg

    def _record_recovery_attempt(self, recovery_result: RecoveryResult, recovery_id: str):
        """Record recovery attempt in history."""
        self.recovery_history.append({
            "recovery_id": recovery_id,
            "timestamp": datetime.utcnow(),
            "success": recovery_result.success,
            "strategy": recovery_result.recovery_strategy.value,
            "recovery_time": recovery_result.recovery_time,
            "attempts_made": recovery_result.attempts_made,
            "performance_impact": recovery_result.performance_impact,
            "cost_impact": recovery_result.cost_impact,
            "message": recovery_result.message
        })

        # Keep only recent history (last 1000 recoveries)
        if len(self.recovery_history) > 1000:
            self.recovery_history = self.recovery_history[-1000:]

    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery system status."""
        return {
            "active_recoveries": len(self.active_recoveries),
            "recovery_statistics": self.recovery_statistics,
            "emergency_mode_active": self.fallback_orchestrator.emergency_manager.is_emergency_active(),
            "circuit_breaker_states": self.error_recovery_manager.circuit_breakers,
            "recent_recovery_count": len([
                r for r in self.recovery_history
                if r["timestamp"] > datetime.utcnow() - timedelta(hours=1)
            ]),
            "system_health": self._assess_system_health()
        }

    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health based on recovery patterns."""
        recent_recoveries = [
            r for r in self.recovery_history
            if r["timestamp"] > datetime.utcnow() - timedelta(hours=1)
        ]

        if not recent_recoveries:
            return {"status": "healthy", "score": 1.0, "issues": []}

        success_rate = sum(1 for r in recent_recoveries if r["success"]) / len(recent_recoveries)
        avg_recovery_time = sum(r["recovery_time"] for r in recent_recoveries) / len(recent_recoveries)

        issues = []
        health_score = 1.0

        # Check success rate
        if success_rate < 0.8:
            issues.append(f"Low recovery success rate: {success_rate:.2f}")
            health_score *= 0.7

        # Check recovery time
        if avg_recovery_time > 60:  # More than 1 minute average
            issues.append(f"High average recovery time: {avg_recovery_time:.1f}s")
            health_score *= 0.8

        # Check emergency mode
        if self.fallback_orchestrator.emergency_manager.is_emergency_active():
            issues.append("Emergency mode is active")
            health_score *= 0.5

        # Check active recoveries
        if len(self.active_recoveries) > 5:
            issues.append(f"High number of active recoveries: {len(self.active_recoveries)}")
            health_score *= 0.6

        # Determine status
        if health_score > 0.8:
            status = "healthy"
        elif health_score > 0.6:
            status = "degraded"
        elif health_score > 0.4:
            status = "unhealthy"
        else:
            status = "critical"

        return {
            "status": status,
            "score": health_score,
            "issues": issues,
            "recent_success_rate": success_rate,
            "avg_recovery_time": avg_recovery_time
        }

    async def test_recovery_system(self) -> Dict[str, Any]:
        """Test the recovery system with simulated errors."""
        test_results = {}

        # Test error classification
        test_context = ErrorContext(
            task_type="test",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1
        )

        test_errors = [
            Exception("Test generic error"),
            ModelError("Test model error", "test-model", "mini", test_context),
            BudgetError("Test budget error", 5.0, 10.0, test_context),
            APIError("Test API error", "test-provider", 500, test_context),
            QualityError("Test quality error", ["missing_citations"], 0.3, test_context)
        ]

        for i, test_error in enumerate(test_errors):
            try:
                classification = self.error_classifier.classify_error(test_error, test_context)
                test_results[f"classification_test_{i}"] = {
                    "success": True,
                    "error_type": type(test_error).__name__,
                    "classification": classification.error_code,
                    "strategies": [s.value for s in classification.recovery_strategies]
                }
            except Exception as e:
                test_results[f"classification_test_{i}"] = {
                    "success": False,
                    "error": str(e)
                }

        # Test fallback availability
        try:
            availability = await self.tri_model_router.detect_model_availability() if self.tri_model_router else {}
            test_results["model_availability"] = {
                "success": True,
                "available_models": availability
            }
        except Exception as e:
            test_results["model_availability"] = {
                "success": False,
                "error": str(e)
            }

        return {
            "test_timestamp": datetime.utcnow().isoformat(),
            "test_results": test_results,
            "system_status": self.get_recovery_status()
        }
