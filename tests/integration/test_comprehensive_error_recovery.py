"""
Integration tests for the comprehensive error handling and recovery system.
Tests the complete error recovery workflow with all fallback strategies.
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.infrastructure.reliability import (
    APIError,
    BudgetError,
    ComprehensiveErrorRecoveryManager,
    ErrorContext,
    ModelError,
    QualityError,
    RecoveryConfiguration,
    RecoveryStrategy,
)


class TestComprehensiveErrorRecoveryIntegration:
    """Integration tests for comprehensive error recovery system."""

    @pytest.fixture
    async def recovery_system(self):
        """Set up a complete recovery system for testing."""
        # Create mock components
        mock_router = Mock()
        mock_router.models = {
            "full": "openai/gpt-5",
            "mini": "openai/gpt-5-mini",
            "nano": "openai/gpt-5-nano",
        }
        mock_router.model_configs = {
            "full": Mock(model_name="openai/gpt-5"),
            "mini": Mock(model_name="openai/gpt-5-mini"),
            "nano": Mock(model_name="openai/gpt-5-nano"),
        }
        mock_router.detect_model_availability = AsyncMock(
            return_value={
                "openai/gpt-5": True,
                "openai/gpt-5-mini": True,
                "openai/gpt-5-nano": True,
                "openai/gpt-oss-20b:free": True,
                "moonshotai/kimi-k2:free": True,
            }
        )
        mock_router.check_model_health = AsyncMock()

        mock_budget_manager = Mock()
        mock_budget_manager.get_budget_status = AsyncMock(
            return_value={
                "remaining_percentage": 50.0,
                "total_budget": 100.0,
                "used_budget": 50.0,
            }
        )

        # Create recovery configuration
        config = RecoveryConfiguration(
            max_recovery_attempts=3,
            max_recovery_time=60.0,
            enable_circuit_breakers=True,
            enable_emergency_mode=True,
            enable_quality_recovery=True,
            budget_threshold_for_emergency=5.0,
        )

        # Initialize recovery manager
        recovery_manager = ComprehensiveErrorRecoveryManager(
            mock_router, mock_budget_manager, config
        )

        return recovery_manager, mock_router, mock_budget_manager

    @pytest.mark.asyncio
    async def test_model_error_recovery_workflow(self, recovery_system):
        """Test complete model error recovery workflow."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Create model error context
        context = ErrorContext(
            task_type="forecast",
            model_tier="full",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
            model_name="openai/gpt-5",
            provider="openrouter",
        )

        # Create model error
        error = ModelError("GPT-5 model failed", "openai/gpt-5", "full", context)

        # Mock successful fallback
        with patch("forecasting_tools.GeneralLlm") as mock_llm:
            mock_llm.return_value.invoke = AsyncMock(return_value="Test response")

            # Execute recovery
            result = await recovery_manager.recover_from_error(error, context)

            # Verify recovery success
            assert result.success
            assert result.recovery_strategy == RecoveryStrategy.FALLBACK_MODEL
            assert result.performance_impact < 1.0  # Some performance impact expected
            assert result.recovery_time > 0
            assert result.attempts_made >= 1

    @pytest.mark.asyncio
    async def test_budget_error_emergency_mode_workflow(self, recovery_system):
        """Test budget error triggering emergency mode workflow."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Create budget exhaustion context
        context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="critical",
            budget_remaining=2.0,
            attempt_number=1,
            model_name="openai/gpt-5-mini",
            provider="openrouter",
        )

        # Create budget error
        error = BudgetError("Budget exhausted", 2.0, 10.0, context)

        # Mock emergency alert sending
        with patch.object(
            recovery_manager.fallback_orchestrator.emergency_manager,
            "_send_emergency_alert",
            new_callable=AsyncMock,
        ):
            # Execute recovery
            result = await recovery_manager.recover_from_error(error, context)

            # Verify emergency mode activation
            assert result.success
            assert result.recovery_strategy == RecoveryStrategy.EMERGENCY_MODE
            assert result.cost_impact < 0  # Should save costs
            assert recovery_manager.fallback_orchestrator.emergency_manager.is_emergency_active()

    @pytest.mark.asyncio
    async def test_api_error_provider_fallback_workflow(self, recovery_system):
        """Test API error triggering provider fallback workflow."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Create API error context
        context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
            model_name="openai/gpt-5-mini",
            provider="openrouter",
        )

        # Create API error
        error = APIError("OpenRouter API failed", "openrouter", 500, context)

        # Mock provider availability
        with patch.dict(os.environ, {"ENABLE_PROXY_CREDITS": "true"}):
            # Execute recovery
            result = await recovery_manager.recover_from_error(error, context)

            # Verify provider fallback (may succeed or fail depending on mocks)
            assert result.recovery_strategy in [
                RecoveryStrategy.FALLBACK_PROVIDER,
                RecoveryStrategy.FALLBACK_MODEL,
                RecoveryStrategy.EMERGENCY_MODE,
            ]
            assert result.recovery_time > 0

    @pytest.mark.asyncio
    async def test_quality_error_prompt_revision_workflow(self, recovery_system):
        """Test quality error triggering prompt revision workflow."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Create quality error context with original prompt
        context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
            model_name="openai/gpt-5-mini",
            provider="openrouter",
            original_prompt="Make a forecast without citations",
        )

        # Create quality error
        error = QualityError("Missing citations", ["missing_citations"], 0.3, context)

        # Execute recovery
        result = await recovery_manager.recover_from_error(error, context)

        # Verify prompt revision or fallback
        assert result.recovery_strategy in [
            RecoveryStrategy.PROMPT_REVISION,
            RecoveryStrategy.FALLBACK_MODEL,
        ]
        assert result.recovery_time > 0

    @pytest.mark.asyncio
    async def test_cascading_fallback_workflow(self, recovery_system):
        """Test cascading fallback when multiple strategies fail."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Create error context
        context = ErrorContext(
            task_type="forecast",
            model_tier="full",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
            model_name="openai/gpt-5",
            provider="openrouter",
        )

        # Create generic error
        error = Exception("Multiple system failures")

        # Mock all fallbacks to fail initially, then succeed on emergency mode
        with (
            patch.object(
                recovery_manager.fallback_orchestrator.model_fallback_manager,
                "execute_fallback",
            ) as mock_model_fallback,
            patch.object(
                recovery_manager.fallback_orchestrator.provider_fallback_manager,
                "execute_provider_fallback",
            ) as mock_provider_fallback,
        ):
            # Mock failures for first attempts
            mock_model_fallback.return_value = Mock(
                success=False, message="Model fallback failed"
            )
            mock_provider_fallback.return_value = Mock(
                success=False, message="Provider fallback failed"
            )

            # Execute recovery
            result = await recovery_manager.recover_from_error(error, context)

            # Should eventually succeed with emergency mode
            assert result.recovery_strategy == RecoveryStrategy.EMERGENCY_MODE
            assert result.attempts_made > 1  # Multiple attempts made

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, recovery_system):
        """Test circuit breaker preventing repeated failed recoveries."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Create error context
        context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
            model_name="openai/gpt-5-mini",
            provider="openrouter",
        )

        # Create repeated error
        error = Exception("Persistent system error")

        # Simulate multiple failures to trigger circuit breaker
        for i in range(6):  # Exceed circuit breaker threshold
            try:
                await recovery_manager.recover_from_error(error, context)
            except:
                pass  # Ignore failures for this test

        # Check circuit breaker state
        circuit_breakers = recovery_manager.error_recovery_manager.circuit_breakers
        assert len(circuit_breakers) > 0  # Circuit breakers should be tracking failures

    @pytest.mark.asyncio
    async def test_recovery_statistics_tracking(self, recovery_system):
        """Test recovery statistics tracking and reporting."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Create various error scenarios
        scenarios = [
            (ModelError("Model failed", "test-model", "mini"), "normal", 50.0),
            (BudgetError("Budget low", 5.0, 10.0), "critical", 5.0),
            (APIError("API failed", "openrouter", 500), "normal", 50.0),
        ]

        for error, operation_mode, budget_remaining in scenarios:
            context = ErrorContext(
                task_type="forecast",
                model_tier="mini",
                operation_mode=operation_mode,
                budget_remaining=budget_remaining,
                attempt_number=1,
            )

            try:
                await recovery_manager.recover_from_error(error, context)
            except:
                pass  # Ignore failures for statistics test

        # Check statistics
        status = recovery_manager.get_recovery_status()
        stats = status["recovery_statistics"]

        assert stats["total_recoveries"] >= len(scenarios)
        assert "strategy_effectiveness" in stats
        assert "average_recovery_time" in stats

    @pytest.mark.asyncio
    async def test_system_health_assessment(self, recovery_system):
        """Test system health assessment functionality."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Get initial system health
        status = recovery_manager.get_recovery_status()
        health = status["system_health"]

        assert "status" in health
        assert "score" in health
        assert "issues" in health
        assert 0.0 <= health["score"] <= 1.0

        # Health status should be one of the expected values
        assert health["status"] in ["healthy", "degraded", "unhealthy", "critical"]

    @pytest.mark.asyncio
    async def test_recovery_system_testing(self, recovery_system):
        """Test the recovery system's self-testing functionality."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Run system test
        test_results = await recovery_manager.test_recovery_system()

        assert "test_timestamp" in test_results
        assert "test_results" in test_results
        assert "system_status" in test_results

        # Check that classification tests were run
        test_results_dict = test_results["test_results"]
        classification_tests = [
            k for k in test_results_dict.keys() if k.startswith("classification_test")
        ]
        assert len(classification_tests) > 0

        # Check model availability test
        assert "model_availability" in test_results_dict

    @pytest.mark.asyncio
    async def test_error_logging_and_alerting(self, recovery_system):
        """Test error logging and alerting functionality."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Create error context
        context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
        )

        # Create error
        error = Exception("Test error for logging")

        # Mock file writing to avoid actual file operations
        with patch("builtins.open"), patch("os.makedirs"):
            # Execute recovery (which should log the error)
            await recovery_manager.recover_from_error(error, context)

            # Check that error was logged
            logging_system = recovery_manager.fallback_orchestrator.logging_system
            assert len(logging_system.error_log) > 0

            # Get error summary
            summary = logging_system.get_error_summary(hours=1)
            assert summary["total_errors"] > 0

    @pytest.mark.asyncio
    async def test_budget_aware_recovery_decisions(self, recovery_system):
        """Test that recovery decisions are budget-aware."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Test scenarios with different budget levels
        budget_scenarios = [
            (80.0, "normal"),  # High budget - should use optimal strategies
            (
                30.0,
                "conservative",
            ),  # Medium budget - should use cost-conscious strategies
            (10.0, "emergency"),  # Low budget - should use cheap/free strategies
            (2.0, "critical"),  # Critical budget - should use emergency mode
        ]

        for budget_remaining, expected_mode in budget_scenarios:
            context = ErrorContext(
                task_type="forecast",
                model_tier="mini",
                operation_mode=expected_mode,
                budget_remaining=budget_remaining,
                attempt_number=1,
            )

            # Update mock budget manager
            mock_budget_manager.budget_remaining = budget_remaining
            mock_budget_manager.get_budget_status = AsyncMock(
                return_value={"remaining_percentage": budget_remaining}
            )

            # Create error
            error = Exception(f"Test error with {budget_remaining}% budget")

            # Execute recovery
            result = await recovery_manager.recover_from_error(error, context)

            # Verify budget-appropriate recovery strategy
            if budget_remaining < 5.0:
                # Should activate emergency mode for critical budget
                assert result.recovery_strategy == RecoveryStrategy.EMERGENCY_MODE
            else:
                # Should use appropriate fallback strategies
                assert result.recovery_strategy in [
                    RecoveryStrategy.FALLBACK_MODEL,
                    RecoveryStrategy.FALLBACK_PROVIDER,
                    RecoveryStrategy.GRACEFUL_DEGRADATION,
                    RecoveryStrategy.BUDGET_CONSERVATION,
                ]

    @pytest.mark.asyncio
    async def test_performance_impact_tracking(self, recovery_system):
        """Test tracking of performance impact from recovery strategies."""
        recovery_manager, mock_router, mock_budget_manager = recovery_system

        # Create error context
        context = ErrorContext(
            task_type="forecast",
            model_tier="full",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
        )

        # Create model error that should trigger tier downgrade
        error = ModelError("GPT-5 failed", "openai/gpt-5", "full", context)

        # Mock successful fallback with performance impact
        with patch("forecasting_tools.GeneralLlm") as mock_llm:
            mock_llm.return_value.invoke = AsyncMock(return_value="Test response")

            # Execute recovery
            result = await recovery_manager.recover_from_error(error, context)

            # Verify performance impact tracking
            assert result.success
            assert 0.0 <= result.performance_impact <= 1.0
            assert result.cost_impact is not None

            # Performance impact should be less than 1.0 for tier downgrade
            if result.recovery_strategy == RecoveryStrategy.FALLBACK_MODEL:
                assert result.performance_impact < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
