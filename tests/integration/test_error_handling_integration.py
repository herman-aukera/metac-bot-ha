"""
Integration tests for error handling and resilience components.

Tests the complete error handling system working together including
circuit breakers, retry strategies, graceful degradation, and logging.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock

from src.infrastructure.resilience.circuit_breaker import CircuitBreakerManager, CircuitBreakerConfig
from src.infrastructure.resilience.retry_strategy import RetryStrategy, RetryConfig, BackoffStrategy
from src.infrastructure.resilience.graceful_degradation import GracefulDegradationManager, ServiceHealth, FallbackConfig
from src.infrastructure.logging.structured_logger import StructuredLogger
from src.infrastructure.logging.error_logger import ErrorLogger, ErrorSeverity, ErrorCategory
from src.infrastructure.logging.correlation_context import correlation_id
from src.domain.exceptions import InfrastructureError, NetworkError


class TestErrorHandlingIntegration:
    """Integration tests for complete error handling system."""

    @pytest.fixture
    async def integrated_system(self):
        """Setup integrated error handling system."""
        # Circuit breaker manager
        cb_manager = CircuitBreakerManager()

        # Retry strategy
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            backoff_strategy=BackoffStrategy.FIXED
        )
        retry_strategy = RetryStrategy(retry_config)

        # Graceful degradation manager
        degradation_manager = GracefulDegradationManager()

        # Setup services
        services = ["primary_service", "fallback_service"]
        for service in services:
            await degradation_manager.register_service(service)
            await degradation_manager.update_service_health(service, ServiceHealth.HEALTHY)

        # Configure fallback
        degradation_manager.fallback_configs["test_service"] = FallbackConfig(
            primary_services=["primary_service"],
            fallback_services=["fallback_service"],
            enable_caching=False
        )

        # Structured logger
        logger = StructuredLogger("integration_test")

        # Error logger
        error_logger = ErrorLogger(logger)

        return {
            "cb_manager": cb_manager,
            "retry_strategy": retry_strategy,
            "degradation_manager": degradation_manager,
            "logger": logger,
            "error_logger": error_logger,
        }

    @pytest.mark.asyncio
    async def test_complete_error_handling_flow(self, integrated_system):
        """Test complete error handling flow with all components."""
        system = await integrated_system
        cb_manager = system["cb_manager"]
        retry_strategy = system["retry_strategy"]
        degradation_manager = system["degradation_manager"]
        error_logger = system["error_logger"]

        # Track calls
        primary_calls = 0
        fallback_calls = 0

        # Mock service operations
        async def primary_service_operation(service_name):
            nonlocal primary_calls
            if service_name == "primary_service":
                primary_calls += 1
                if primary_calls <= 2:  # Fail first 2 calls
                    raise NetworkError("Primary service network error", host="primary.com")
                return "primary_success"
            else:
                return "unexpected_service"

        async def fallback_service_operation(service_name):
            nonlocal fallback_calls
            if service_name == "fallback_service":
                fallback_calls += 1
                return "fallback_success"
            else:
                return "unexpected_service"

        # Integrated operation with all error handling
        async def integrated_operation():
            with correlation_id(operation="test_operation", component="integration_test"):
                # Use graceful degradation
                async def fallback_operation(service_name):
                    try:
                        # Use circuit breaker
                        circuit_breaker = await cb_manager.get_circuit_breaker(f"{service_name}_cb")

                        # Use retry strategy
                        async def retryable_operation():
                            if service_name == "primary_service":
                                return await circuit_breaker.call(primary_service_operation, service_name)
                            else:
                                return await circuit_breaker.call(fallback_service_operation, service_name)

                        return await retry_strategy.execute(retryable_operation)

                    except Exception as e:
                        # Log error with comprehensive context
                        error_id = error_logger.log_error(
                            e,
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.INTEGRATION,
                            component="integration_test",
                            operation="integrated_operation",
                            context={"service_name": service_name}
                        )
                        raise

                return await degradation_manager.execute_with_fallback("test_service", fallback_operation)

        # Execute integrated operation
        result = await integrated_operation()

        # Verify results
        assert result == "primary_success"  # Should succeed after retries
        assert primary_calls == 3  # Should have retried primary service
        assert fallback_calls == 0  # Should not have needed fallback

        # Verify circuit breaker metrics
        cb_metrics = cb_manager.get_all_metrics()
        assert "primary_service_cb" in cb_metrics
        assert cb_metrics["primary_service_cb"]["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_fallback_with_error_logging(self, integrated_system):
        """Test fallback mechanism with comprehensive error logging."""
        system = await integrated_system
        cb_manager = system["cb_manager"]
        retry_strategy = system["retry_strategy"]
        degradation_manager = system["degradation_manager"]
        error_logger = system["error_logger"]

        # Mock service operations - primary always fails
        async def failing_primary_operation(service_name):
            if service_name == "primary_service":
                raise InfrastructureError("Primary service is down", recoverable=False)
            return "unexpected"

        async def working_fallback_operation(service_name):
            if service_name == "fallback_service":
                return "fallback_success"
            return "unexpected"

        # Integrated operation
        async def integrated_operation():
            with correlation_id(operation="fallback_test", component="integration_test"):
                async def fallback_operation(service_name):
                    try:
                        circuit_breaker = await cb_manager.get_circuit_breaker(f"{service_name}_cb")

                        if service_name == "primary_service":
                            return await circuit_breaker.call(failing_primary_operation, service_name)
                        else:
                            return await circuit_breaker.call(working_fallback_operation, service_name)

                    except Exception as e:
                        error_id = error_logger.log_error(
                            e,
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.SYSTEM,
                            component="integration_test",
                            operation="fallback_test",
                            context={"service_name": service_name, "fallback_attempt": True}
                        )
                        raise

                return await degradation_manager.execute_with_fallback("test_service", fallback_operation)

        # Execute operation
        result = await integrated_operation()

        # Should succeed with fallback
        assert result == "fallback_success"

        # Verify error logging
        error_summary = error_logger.get_error_summary(hours=1)
        assert error_summary["total_errors"] > 0
        assert error_summary["category_breakdown"]["system"] > 0

        # Verify circuit breaker opened for primary service
        cb_metrics = cb_manager.get_all_metrics()
        primary_cb_metrics = cb_metrics.get("primary_service_cb", {})
        assert primary_cb_metrics.get("failed_requests", 0) > 0

    @pytest.mark.asyncio
    async def test_system_degradation_monitoring(self, integrated_system):
        """Test system degradation monitoring and alerting."""
        system = await integrated_system
        degradation_manager = system["degradation_manager"]
        error_logger = system["error_logger"]

        # Add more services for degradation testing
        additional_services = ["service_a", "service_b", "service_c"]
        for service in additional_services:
            await degradation_manager.register_service(service)
            await degradation_manager.update_service_health(service, ServiceHealth.HEALTHY)

        # Gradually degrade services
        await degradation_manager.update_service_health("service_a", ServiceHealth.UNHEALTHY)

        # Check degradation level
        system_status = degradation_manager.get_system_status()
        assert system_status["degradation_level"] in ["minimal", "none"]

        # Degrade more services
        await degradation_manager.update_service_health("service_b", ServiceHealth.UNHEALTHY)
        await degradation_manager.update_service_health("service_c", ServiceHealth.UNHEALTHY)

        # Check increased degradation
        system_status = degradation_manager.get_system_status()
        degradation_level = system_status["degradation_level"]

        # Should show some level of degradation
        assert degradation_level != "none"

        # Verify service health tracking
        services_status = system_status["services"]
        unhealthy_count = sum(
            1 for service_status in services_status.values()
            if service_status["health"] == "unhealthy"
        )
        assert unhealthy_count >= 3

    @pytest.mark.asyncio
    async def test_correlation_tracking(self, integrated_system):
        """Test correlation ID tracking across all components."""
        system = await integrated_system
        cb_manager = system["cb_manager"]
        retry_strategy = system["retry_strategy"]
        error_logger = system["error_logger"]
        logger = system["logger"]

        # Mock operation that logs and fails
        async def logging_operation():
            logger.info("Operation started")
            raise NetworkError("Test network error", host="test.com")

        # Execute with correlation context
        with correlation_id(
            operation="correlation_test",
            component="integration_test",
            user_id="test_user_123"
        ) as ctx:
            try:
                circuit_breaker = await cb_manager.get_circuit_breaker("test_cb")
                await retry_strategy.execute(
                    lambda: circuit_breaker.call(logging_operation)
                )
            except Exception as e:
                error_id = error_logger.log_error(
                    e,
                    severity=ErrorSeverity.LOW,
                    category=ErrorCategory.SYSTEM,
                    component="integration_test",
                    operation="correlation_test"
                )

        # Verify correlation ID was tracked
        assert ctx.correlation_id is not None
        assert ctx.user_id == "test_user_123"
        assert ctx.operation == "correlation_test"

        # Verify error was logged with correlation context
        recent_errors = error_logger.get_recent_errors(limit=1)
        assert len(recent_errors) > 0

        latest_error = recent_errors[0]
        assert latest_error.correlation_id == ctx.correlation_id
        assert latest_error.operation == "correlation_test"
        assert latest_error.component == "integration_test"

    @pytest.mark.asyncio
    async def test_recovery_workflows(self, integrated_system):
        """Test error recovery workflows."""
        system = await integrated_system
        error_logger = system["error_logger"]

        # Register recovery handler
        recovery_attempts = 0

        def test_recovery_handler(exception, error_entry):
            nonlocal recovery_attempts
            recovery_attempts += 1

            # Simulate recovery logic
            if isinstance(exception, NetworkError):
                # Simulate successful recovery
                return recovery_attempts <= 2

            return False

        error_logger.register_recovery_handler("NetworkError", test_recovery_handler)

        # Trigger error that should attempt recovery
        test_error = NetworkError("Recoverable network error", host="test.com")

        error_id = error_logger.log_error(
            test_error,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SYSTEM,
            component="integration_test",
            operation="recovery_test",
            attempt_recovery=True
        )

        # Verify recovery was attempted
        assert recovery_attempts > 0

        # Check error entry was updated
        recent_errors = error_logger.get_recent_errors(limit=1)
        assert len(recent_errors) > 0

        error_entry = recent_errors[0]
        assert error_entry.error_id == error_id
        assert error_entry.recovery_attempted is True

    @pytest.mark.asyncio
    async def test_performance_under_load(self, integrated_system):
        """Test error handling performance under concurrent load."""
        system = await integrated_system
        cb_manager = system["cb_manager"]
        retry_strategy = system["retry_strategy"]
        degradation_manager = system["degradation_manager"]

        # Mock operation with random failures
        import random

        async def load_test_operation(service_name, task_id):
            # 30% failure rate
            if random.random() < 0.3:
                raise InfrastructureError(f"Load test failure {task_id}", recoverable=True)

            # Small delay to simulate work
            await asyncio.sleep(0.001)
            return f"success_{task_id}"

        # Integrated operation
        async def integrated_load_operation(task_id):
            async def fallback_operation(service_name):
                circuit_breaker = await cb_manager.get_circuit_breaker(f"load_test_cb_{service_name}")

                async def retryable_op():
                    return await circuit_breaker.call(load_test_operation, service_name, task_id)

                return await retry_strategy.execute(retryable_op)

            return await degradation_manager.execute_with_fallback("test_service", fallback_operation)

        # Launch concurrent operations
        tasks = []
        for i in range(20):  # Moderate load for testing
            task = asyncio.create_task(integrated_load_operation(i))
            tasks.append(task)

        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successes = [r for r in results if isinstance(r, str)]
        failures = [r for r in results if isinstance(r, Exception)]

        # Should have high success rate due to error handling
        success_rate = len(successes) / len(results)
        assert success_rate > 0.5, f"Success rate {success_rate} should be > 0.5"

        # Verify circuit breaker metrics
        cb_metrics = cb_manager.get_all_metrics()
        assert len(cb_metrics) > 0

        # Verify system handled load gracefully
        system_status = degradation_manager.get_system_status()
        assert system_status is not None
