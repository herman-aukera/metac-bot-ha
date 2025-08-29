"""
Unit tests for the comprehensive error handling and recovery system.
Tests error classification, fallback strategies, and recovery orchestration.
"""

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.infrastructure.reliability.comprehensive_error_recovery import (
    ComprehensiveErrorRecoveryManager,
    RecoveryConfiguration,
    RecoveryResult,
)
from src.infrastructure.reliability.error_classification import (
    APIError,
    BudgetError,
    ErrorCategory,
    ErrorClassifier,
    ErrorContext,
    ErrorRecoveryManager,
    ErrorSeverity,
    ForecastingError,
    ModelError,
    QualityError,
    RecoveryStrategy,
)
from src.infrastructure.reliability.fallback_strategies import (
    CrossProviderFallbackManager,
    EmergencyModeManager,
    ErrorLoggingAndAlertingSystem,
    FallbackOption,
    FallbackTier,
    IntelligentFallbackOrchestrator,
    ModelTierFallbackManager,
    PerformanceLevel,
)


class TestErrorClassifier:
    """Test error classification functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = ErrorClassifier()
        self.test_context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
            model_name="openai/gpt-5-mini",
            provider="openrouter",
        )

    def test_classify_rate_limit_error(self):
        """Test classification of rate limit errors."""
        error = Exception("Rate limit exceeded")
        classification = self.classifier.classify_error(error, self.test_context)

        assert classification.category == ErrorCategory.RATE_LIMIT_ERROR
        assert classification.severity == ErrorSeverity.HIGH
        assert classification.error_code == "OPENAI_RATE_LIMIT"
        assert RecoveryStrategy.RETRY in classification.recovery_strategies
        assert RecoveryStrategy.FALLBACK_MODEL in classification.recovery_strategies

    def test_classify_model_unavailable_error(self):
        """Test classification of model unavailable errors."""
        error = Exception("Model not found")
        classification = self.classifier.classify_error(error, self.test_context)

        assert classification.category == ErrorCategory.MODEL_ERROR
        assert classification.error_code == "MODEL_UNAVAILABLE"
        assert RecoveryStrategy.FALLBACK_MODEL in classification.recovery_strategies

    def test_classify_budget_exhausted_error(self):
        """Test classification of budget exhaustion."""
        budget_context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="critical",
            budget_remaining=0.0,
            attempt_number=1,
        )

        error = Exception("Budget exhausted")
        classification = self.classifier.classify_error(error, budget_context)

        assert classification.category == ErrorCategory.BUDGET_ERROR
        assert classification.error_code == "BUDGET_EXHAUSTED"
        assert RecoveryStrategy.EMERGENCY_MODE in classification.recovery_strategies

    def test_classify_context_length_error(self):
        """Test classification of context length errors."""
        error = Exception("Context length exceeded")
        classification = self.classifier.classify_error(error, self.test_context)

        assert classification.category == ErrorCategory.MODEL_ERROR
        assert classification.error_code == "CONTEXT_TOO_LONG"
        assert RecoveryStrategy.PROMPT_REVISION in classification.recovery_strategies

    def test_classify_quality_error(self):
        """Test classification of quality validation errors."""
        error = Exception("Quality validation failed")
        classification = self.classifier.classify_error(error, self.test_context)

        assert classification.category == ErrorCategory.QUALITY_ERROR
        assert classification.error_code == "QUALITY_FAILED"
        assert RecoveryStrategy.PROMPT_REVISION in classification.recovery_strategies

    def test_should_retry_logic(self):
        """Test retry decision logic."""
        classification = self.classifier.error_patterns["openai_rate_limit"]

        # Should retry within max attempts
        assert self.classifier.should_retry(classification, 1)
        assert self.classifier.should_retry(classification, 2)

        # Should not retry beyond max attempts
        assert not self.classifier.should_retry(classification, 5)

    def test_calculate_retry_delay(self):
        """Test retry delay calculation."""
        classification = self.classifier.error_patterns["openai_rate_limit"]

        # Test exponential backoff
        delay1 = self.classifier.calculate_retry_delay(classification, 1)
        delay2 = self.classifier.calculate_retry_delay(classification, 2)
        delay3 = self.classifier.calculate_retry_delay(classification, 3)

        assert delay1 == classification.retry_delay  # Base delay
        assert delay2 > delay1  # Exponential increase
        assert delay3 > delay2  # Continued increase

    def test_error_statistics(self):
        """Test error statistics tracking."""
        # Record some errors
        for i in range(5):
            error = Exception(f"Test error {i}")
            classification = self.classifier.classify_error(error, self.test_context)

        stats = self.classifier.get_error_statistics()
        assert stats["total_errors"] == 5
        assert "error_categories" in stats
        assert "most_common" in stats


class TestModelTierFallbackManager:
    """Test model tier fallback functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_router = Mock()
        self.fallback_manager = ModelTierFallbackManager(self.mock_router)
        self.test_context = ErrorContext(
            task_type="forecast",
            model_tier="full",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
        )

    @pytest.mark.asyncio
    async def test_execute_fallback_success(self):
        """Test successful model tier fallback."""
        # Mock availability check
        with (
            patch.object(
                self.fallback_manager, "_check_availability", return_value=True
            ),
            patch.object(
                self.fallback_manager, "_test_fallback_option", return_value=True
            ),
        ):

            result = await self.fallback_manager.execute_fallback(
                "full", self.test_context, 50.0
            )

            assert result.success
            assert result.fallback_used is not None
            assert result.performance_impact < 1.0  # Some performance impact expected

    @pytest.mark.asyncio
    async def test_execute_fallback_no_viable_options(self):
        """Test fallback when no viable options available."""
        # Mock no availability
        with patch.object(
            self.fallback_manager, "_check_availability", return_value=False
        ):

            result = await self.fallback_manager.execute_fallback(
                "full", self.test_context, 50.0
            )

            assert not result.success
            assert "No viable fallback options available" in result.message

    @pytest.mark.asyncio
    async def test_filter_viable_options_budget_constraints(self):
        """Test filtering options based on budget constraints."""
        fallback_chain = self.fallback_manager.fallback_chains["full"]

        # Test with low budget - should only return free models
        with patch.object(
            self.fallback_manager, "_check_availability", return_value=True
        ):
            viable_options = await self.fallback_manager._filter_viable_options(
                fallback_chain, 10.0, self.test_context  # 10% budget remaining
            )

            # Should only include free models
            for option in viable_options:
                assert option.cost_per_million == 0.0

    def test_calculate_performance_impact(self):
        """Test performance impact calculation."""
        fallback_option = FallbackOption(
            name="test-model",
            tier=FallbackTier.SECONDARY,
            performance_level=PerformanceLevel.GOOD,
            cost_per_million=0.25,
            availability_check="test",
            configuration={},
        )

        impact = self.fallback_manager._calculate_performance_impact(
            "full", fallback_option
        )
        assert 0.0 < impact < 1.0  # Should be reduced performance

    def test_calculate_cost_impact(self):
        """Test cost impact calculation."""
        fallback_option = FallbackOption(
            name="test-model",
            tier=FallbackTier.EMERGENCY,
            performance_level=PerformanceLevel.ACCEPTABLE,
            cost_per_million=0.0,  # Free model
            availability_check="test",
            configuration={},
        )

        impact = self.fallback_manager._calculate_cost_impact("full", fallback_option)
        assert impact < 0  # Should be cost savings


class TestCrossProviderFallbackManager:
    """Test cross-provider fallback functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fallback_manager = CrossProviderFallbackManager()
        self.test_context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
            provider="openrouter",
        )

    @pytest.mark.asyncio
    async def test_execute_provider_fallback_success(self):
        """Test successful provider fallback."""
        with (
            patch.object(
                self.fallback_manager, "_check_provider_availability", return_value=True
            ),
            patch.object(self.fallback_manager, "_test_provider", return_value=True),
        ):

            result = await self.fallback_manager.execute_provider_fallback(
                "openrouter", self.test_context
            )

            assert result.success
            assert result.fallback_used is not None

    @pytest.mark.asyncio
    async def test_execute_provider_fallback_no_fallbacks(self):
        """Test provider fallback when no fallbacks available."""
        # Test with provider that has no fallbacks
        result = await self.fallback_manager.execute_provider_fallback(
            "unknown_provider", self.test_context
        )

        assert not result.success
        assert "No fallback providers defined" in result.message

    @pytest.mark.asyncio
    async def test_check_openrouter_availability(self):
        """Test OpenRouter availability check."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            available = await self.fallback_manager._check_openrouter_availability()
            assert available

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "dummy_key"}):
            available = await self.fallback_manager._check_openrouter_availability()
            assert not available

    @pytest.mark.asyncio
    async def test_check_metaculus_proxy_availability(self):
        """Test Metaculus proxy availability check."""
        with patch.dict(os.environ, {"ENABLE_PROXY_CREDITS": "true"}):
            available = (
                await self.fallback_manager._check_metaculus_proxy_availability()
            )
            assert available

        with patch.dict(os.environ, {"ENABLE_PROXY_CREDITS": "false"}):
            available = (
                await self.fallback_manager._check_metaculus_proxy_availability()
            )
            assert not available


class TestEmergencyModeManager:
    """Test emergency mode functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_budget_manager = Mock()
        self.emergency_manager = EmergencyModeManager(self.mock_budget_manager)
        self.test_context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="critical",
            budget_remaining=2.0,
            attempt_number=1,
        )

    @pytest.mark.asyncio
    async def test_activate_emergency_mode(self):
        """Test emergency mode activation."""
        error = BudgetError("Budget exhausted", 0.0, 10.0, self.test_context)

        with patch.object(
            self.emergency_manager, "_send_emergency_alert", new_callable=AsyncMock
        ):
            recovery_action = await self.emergency_manager.activate_emergency_mode(
                error, self.test_context
            )

            assert self.emergency_manager.is_emergency_active()
            assert recovery_action.strategy == RecoveryStrategy.EMERGENCY_MODE
            assert recovery_action.parameters["free_models_only"]

    @pytest.mark.asyncio
    async def test_deactivate_emergency_mode(self):
        """Test emergency mode deactivation."""
        # First activate emergency mode
        error = Exception("Test error")
        await self.emergency_manager.activate_emergency_mode(error, self.test_context)

        # Mock conditions for deactivation
        with patch.object(
            self.emergency_manager,
            "_check_normal_operation_conditions",
            return_value=True,
        ):
            success = await self.emergency_manager.deactivate_emergency_mode()

            assert success
            assert not self.emergency_manager.is_emergency_active()

    @pytest.mark.asyncio
    async def test_check_normal_operation_conditions(self):
        """Test normal operation conditions check."""
        # Mock budget manager
        self.mock_budget_manager.get_budget_status = AsyncMock(
            return_value={"remaining_percentage": 20.0}
        )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            conditions_met = (
                await self.emergency_manager._check_normal_operation_conditions()
            )
            assert conditions_met

        # Test with low budget
        self.mock_budget_manager.get_budget_status = AsyncMock(
            return_value={"remaining_percentage": 2.0}
        )
        conditions_met = (
            await self.emergency_manager._check_normal_operation_conditions()
        )
        assert not conditions_met

    def test_get_emergency_status(self):
        """Test emergency status reporting."""
        # Test inactive status
        status = self.emergency_manager.get_emergency_status()
        assert not status["active"]

        # Activate emergency mode
        self.emergency_manager.emergency_active = True
        self.emergency_manager.emergency_start_time = datetime.utcnow()

        status = self.emergency_manager.get_emergency_status()
        assert status["active"]
        assert "start_time" in status
        assert "duration_seconds" in status


class TestErrorLoggingAndAlertingSystem:
    """Test error logging and alerting functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logging_system = ErrorLoggingAndAlertingSystem()
        self.test_context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
        )

    @pytest.mark.asyncio
    async def test_log_error(self):
        """Test error logging functionality."""
        error = Exception("Test error")
        recovery_action = Mock()
        recovery_action.strategy.value = "retry"
        recovery_action.parameters = {"delay": 5.0}
        recovery_action.success_probability = 0.8

        with (
            patch.object(
                self.logging_system, "_check_and_trigger_alerts", new_callable=AsyncMock
            ),
            patch.object(
                self.logging_system, "_write_to_log_file", new_callable=AsyncMock
            ),
        ):

            await self.logging_system.log_error(
                error, self.test_context, recovery_action
            )

            assert len(self.logging_system.error_log) == 1
            log_entry = self.logging_system.error_log[0]
            assert log_entry["error_type"] == "Exception"
            assert log_entry["error_message"] == "Test error"

    @pytest.mark.asyncio
    async def test_alert_triggering(self):
        """Test alert triggering based on error thresholds."""
        error = Exception("Repeated error")

        # Log multiple errors of the same type
        with (
            patch.object(
                self.logging_system, "_send_alert", new_callable=AsyncMock
            ) as mock_send_alert,
            patch.object(
                self.logging_system, "_write_to_log_file", new_callable=AsyncMock
            ),
        ):

            # Log errors up to threshold
            for i in range(self.logging_system.alert_config.error_threshold):
                await self.logging_system.log_error(error, self.test_context)

            # Should trigger alert
            mock_send_alert.assert_called_once()

    def test_get_error_summary(self):
        """Test error summary generation."""
        # Add some test errors
        for i in range(3):
            self.logging_system.error_log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "error_type": "TestError",
                    "error_message": f"Test error {i}",
                    "context": {
                        "task_type": "forecast",
                        "model_tier": "mini",
                        "operation_mode": "normal",
                    },
                }
            )

        summary = self.logging_system.get_error_summary(hours=24)
        assert summary["total_errors"] == 3
        assert "TestError" in summary["error_types"]
        assert summary["error_types"]["TestError"] == 3


class TestComprehensiveErrorRecoveryManager:
    """Test comprehensive error recovery manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_router = Mock()
        self.mock_budget_manager = Mock()
        self.config = RecoveryConfiguration(
            max_recovery_attempts=2, max_recovery_time=60.0
        )

        self.recovery_manager = ComprehensiveErrorRecoveryManager(
            self.mock_router, self.mock_budget_manager, self.config
        )

        self.test_context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
        )

    @pytest.mark.asyncio
    async def test_recover_from_model_error(self):
        """Test recovery from model errors."""
        error = ModelError("Model failed", "test-model", "mini", self.test_context)

        # Mock successful model fallback
        mock_fallback_result = Mock()
        mock_fallback_result.success = True
        mock_fallback_result.recovery_time = 5.0
        mock_fallback_result.performance_impact = 0.8
        mock_fallback_result.cost_impact = -0.2
        mock_fallback_result.message = "Fallback successful"
        mock_fallback_result.metadata = {}

        with patch.object(
            self.recovery_manager.fallback_orchestrator.model_fallback_manager,
            "execute_fallback",
            return_value=mock_fallback_result,
        ):
            result = await self.recovery_manager.recover_from_error(
                error, self.test_context
            )

            assert result.success
            assert result.recovery_strategy == RecoveryStrategy.FALLBACK_MODEL
            assert result.performance_impact == 0.8

    @pytest.mark.asyncio
    async def test_recover_from_budget_error(self):
        """Test recovery from budget errors."""
        error = BudgetError("Budget exhausted", 0.0, 10.0, self.test_context)

        # Mock emergency mode activation
        mock_recovery_action = Mock()
        mock_recovery_action.strategy = RecoveryStrategy.EMERGENCY_MODE
        mock_recovery_action.expected_delay = 1.0
        mock_recovery_action.parameters = {"free_models_only": True}

        with patch.object(
            self.recovery_manager.fallback_orchestrator.emergency_manager,
            "activate_emergency_mode",
            return_value=mock_recovery_action,
        ):
            result = await self.recovery_manager.recover_from_error(
                error, self.test_context
            )

            assert result.success
            assert result.recovery_strategy == RecoveryStrategy.EMERGENCY_MODE
            assert result.cost_impact < 0  # Should save costs

    @pytest.mark.asyncio
    async def test_recover_from_api_error(self):
        """Test recovery from API errors."""
        error = APIError("API failed", "openrouter", 500, self.test_context)

        # Mock successful provider fallback
        mock_fallback_result = Mock()
        mock_fallback_result.success = True
        mock_fallback_result.recovery_time = 10.0
        mock_fallback_result.performance_impact = 0.9
        mock_fallback_result.cost_impact = 0.0
        mock_fallback_result.message = "Provider fallback successful"
        mock_fallback_result.metadata = {}

        with patch.object(
            self.recovery_manager.fallback_orchestrator.provider_fallback_manager,
            "execute_provider_fallback",
            return_value=mock_fallback_result,
        ):
            result = await self.recovery_manager.recover_from_error(
                error, self.test_context
            )

            assert result.success
            assert result.recovery_strategy == RecoveryStrategy.FALLBACK_PROVIDER

    @pytest.mark.asyncio
    async def test_recovery_attempt_limits(self):
        """Test recovery attempt limits."""
        error = Exception("Persistent error")

        # Set context with high attempt number
        high_attempt_context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=10,  # Exceeds max attempts
        )

        result = await self.recovery_manager.recover_from_error(
            error, high_attempt_context
        )

        assert not result.success
        assert result.recovery_strategy == RecoveryStrategy.ABORT

    def test_update_recovery_statistics(self):
        """Test recovery statistics updates."""
        # Create a successful recovery result
        recovery_result = RecoveryResult(
            success=True,
            recovery_strategy=RecoveryStrategy.FALLBACK_MODEL,
            fallback_result=None,
            recovery_time=5.0,
            attempts_made=1,
            final_error=None,
            performance_impact=0.8,
            cost_impact=-0.1,
            message="Test recovery",
            metadata={},
        )

        initial_total = self.recovery_manager.recovery_statistics["total_recoveries"]
        initial_successful = self.recovery_manager.recovery_statistics[
            "successful_recoveries"
        ]

        self.recovery_manager._update_recovery_statistics(recovery_result)

        assert (
            self.recovery_manager.recovery_statistics["total_recoveries"]
            == initial_total + 1
        )
        assert (
            self.recovery_manager.recovery_statistics["successful_recoveries"]
            == initial_successful + 1
        )

        # Check strategy effectiveness tracking
        strategy_stats = self.recovery_manager.recovery_statistics[
            "strategy_effectiveness"
        ]["fallback_model"]
        assert strategy_stats["attempts"] == 1
        assert strategy_stats["successes"] == 1
        assert strategy_stats["success_rate"] == 1.0

    def test_assess_system_health(self):
        """Test system health assessment."""
        # Add some recovery history
        self.recovery_manager.recovery_history = [
            {"timestamp": datetime.utcnow(), "success": True, "recovery_time": 5.0},
            {"timestamp": datetime.utcnow(), "success": False, "recovery_time": 30.0},
        ]

        health = self.recovery_manager._assess_system_health()

        assert "status" in health
        assert "score" in health
        assert "issues" in health
        assert 0.0 <= health["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_test_recovery_system(self):
        """Test recovery system testing functionality."""
        # Mock router availability detection
        if self.recovery_manager.tri_model_router:
            self.recovery_manager.tri_model_router.detect_model_availability = (
                AsyncMock(
                    return_value={"openai/gpt-5": True, "openai/gpt-5-mini": True}
                )
            )

        test_results = await self.recovery_manager.test_recovery_system()

        assert "test_timestamp" in test_results
        assert "test_results" in test_results
        assert "system_status" in test_results

        # Check that classification tests were run
        classification_tests = [
            k
            for k in test_results["test_results"].keys()
            if k.startswith("classification_test")
        ]
        assert len(classification_tests) > 0


if __name__ == "__main__":
    pytest.main([__file__])
