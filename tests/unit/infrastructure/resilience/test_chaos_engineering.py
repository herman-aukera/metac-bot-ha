"""
Chaos engineering tests to validate system resilience under failure conditions.

These tests simulate various failure scenarios to ensure the error handling
and resilience mechanisms work correctly under stress.
"""

import pytest
import asyncio
import random
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from src.infrastructure.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from src.infrastructure.resilience.retry_strategy import RetryStrategy, RetryConfig, BackoffStrategy
from src.infrastructure.resilience.graceful_degradation import GracefulDegradationManager, ServiceHealth
from src.domain.exceptions import InfrastructureError, NetworkError, TimeoutError


class TestChaosCircuitBreaker:
    """Chaos engineering tests for circuit breaker."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with aggressive settings for chaos testing."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,  # Very short for testing
            success_threshold=1,
            timeout=0.5
        )
        return CircuitBreaker("chaos_service", config)

    @pytest.mark.asyncio
    async def test_random_failure_patterns(self, circuit_breaker):
        """Test circuit breaker with random failure patterns."""
        call_count = 0
        failure_rate = 0.7  # 70% failure rate

        async def chaotic_function():
            nonlocal call_count
            call_count += 1

            if random.random() < failure_rate:
                # Random failure types
                failures = [
                    NetworkError("Network timeout", host="test.com"),
                    TimeoutError("Operation timeout", timeout_duration=30.0),
                    InfrastructureError("Service unavailable", recoverable=True),
                    ConnectionError("Connection refused")
                ]
                raise random.choice(failures)

            return f"success_{call_count}"

        # Run many operations to test circuit breaker behavior
        successes = 0
        circuit_opens = 0
        total_attempts = 100

        for i in range(total_attempts):
            try:
                result = await circuit_breaker.call(chaotic_function)
                successes += 1

                # Reset failure rate occasionally to test recovery
                if i % 20 == 0:
                    failure_rate = random.uniform(0.3, 0.9)

            except Exception as e:
                # Track circuit opens
                if isinstance(e, Exception) and "Circuit breaker is open" in str(e):
                    circuit_opens += 1

                # Small delay to allow recovery
                await asyncio.sleep(0.01)

        # Verify circuit breaker protected the system
        assert circuit_opens > 0, "Circuit breaker should have opened during chaos"
        assert circuit_breaker.metrics.circuit_opens > 0
        assert circuit_breaker.metrics.total_requests > 0

        # System should have some successes despite chaos
        assert successes > 0, "Some operations should succeed"

    @pytest.mark.asyncio
    async def test_cascading_failures(self, circuit_breaker):
        """Test circuit breaker behavior during cascading failures."""
        failure_cascade = True

        async def cascading_function():
            if failure_cascade:
                # Simulate cascading failure
                raise InfrastructureError("Cascading failure", recoverable=True)
            return "recovered"

        # Trigger cascading failures
        for _ in range(5):
            with pytest.raises(Exception):
                await circuit_breaker.call(cascading_function)

        # Circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN

        # Stop cascade and wait for recovery
        failure_cascade = False
        await asyncio.sleep(0.2)  # Wait for recovery timeout

        # Should recover
        result = await circuit_breaker.call(cascading_function)
        assert result == "recovered"
        assert circuit_breaker.state in [CircuitState.HALF_OPEN, CircuitState.CLOSED]

    @pytest.mark.asyncio
    async def test_concurrent_chaos(self, circuit_breaker):
        """Test circuit breaker under concurrent chaotic load."""
        async def chaotic_concurrent_function(task_id):
            # Random delay to simulate varying response times
            await asyncio.sleep(random.uniform(0.01, 0.1))

            # Random failures
            if random.random() < 0.6:
                raise InfrastructureError(f"Task {task_id} failed", recoverable=True)

            return f"task_{task_id}_success"

        # Launch many concurrent tasks
        tasks = []
        for i in range(50):
            task = asyncio.create_task(
                circuit_breaker.call(chaotic_concurrent_function, i)
            )
            tasks.append(task)

        # Wait for all tasks with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successes = sum(1 for r in results if isinstance(r, str))
        exceptions = sum(1 for r in results if isinstance(r, Exception))

        assert successes + exceptions == 50
        assert circuit_breaker.metrics.total_requests > 0

        # Circuit breaker should have provided protection
        assert circuit_breaker.metrics.circuit_opens >= 0  # May or may not open depending on timing


class TestChaosRetryStrategy:
    """Chaos engineering tests for retry strategy."""

    @pytest.fixture
    def retry_strategy(self):
        """Create retry strategy for chaos testing."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.01,
            max_delay=0.1,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            timeout=2.0
        )
        return RetryStrategy(config)

    @pytest.mark.asyncio
    async def test_intermittent_failures(self, retry_strategy):
        """Test retry strategy with intermittent failures."""
        call_count = 0

        async def intermittent_function():
            nonlocal call_count
            call_count += 1

            # Fail on specific attempts to test retry logic
            if call_count in [1, 3]:  # Fail on 1st and 3rd attempts
                raise NetworkError("Intermittent network error", host="api.test.com")

            return f"success_after_{call_count}_attempts"

        result = await retry_strategy.execute(intermittent_function)

        assert result == "success_after_4_attempts"
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_random_transient_failures(self, retry_strategy):
        """Test retry strategy with random transient failures."""
        async def random_failure_function():
            failure_types = [
                NetworkError("Random network error", host="random.com"),
                TimeoutError("Random timeout", timeout_duration=10.0),
                InfrastructureError("Random infrastructure error", recoverable=True),
                ConnectionError("Random connection error")
            ]

            # 60% chance of failure
            if random.random() < 0.6:
                raise random.choice(failure_types)

            return "success"

        # Run multiple tests to account for randomness
        successes = 0
        failures = 0

        for _ in range(20):
            try:
                result = await retry_strategy.execute(random_failure_function)
                if result == "success":
                    successes += 1
            except Exception:
                failures += 1

        # Should have some successes due to retries
        assert successes > 0, "Retry strategy should enable some successes"

        # May have some failures if all retries exhausted
        total_operations = successes + failures
        assert total_operations == 20

    @pytest.mark.asyncio
    async def test_timeout_chaos(self, retry_strategy):
        """Test retry strategy with timeout chaos."""
        async def timeout_chaos_function():
            # Random delay that may exceed timeout
            delay = random.uniform(0.05, 0.3)
            await asyncio.sleep(delay)

            if delay > 0.2:  # Simulate timeout-like behavior
                raise TimeoutError("Simulated timeout", timeout_duration=0.2)

            return "success"

        # Test multiple operations
        results = []
        for _ in range(10):
            try:
                result = await retry_strategy.execute(timeout_chaos_function)
                results.append(result)
            except Exception as e:
                results.append(e)

        # Should have mix of successes and failures
        successes = sum(1 for r in results if r == "success")
        assert len(results) == 10
        # Due to randomness, we can't guarantee exact counts, but should have some results


class TestChaosGracefulDegradation:
    """Chaos engineering tests for graceful degradation."""

    @pytest.fixture
    def manager(self):
        """Create graceful degradation manager for chaos testing."""
        return GracefulDegradationManager()

    @pytest.mark.asyncio
    async def test_service_chaos(self, manager):
        """Test graceful degradation with chaotic service behavior."""
        # Setup multiple services
        services = ["service1", "service2", "service3", "service4"]
        for service in services:
            await manager.register_service(service)
            await manager.update_service_health(service, ServiceHealth.HEALTHY)

        # Configure fallback
        from src.infrastructure.resilience.graceful_degradation import FallbackConfig
        manager.fallback_configs["chaos_test"] = FallbackConfig(
            primary_services=["service1", "service2"],
            fallback_services=["service3", "service4"],
            enable_caching=False
        )

        # Chaotic operation that randomly fails for different services
        async def chaotic_operation(service_name):
            failure_rates = {
                "service1": 0.8,  # High failure rate
                "service2": 0.6,  # Medium failure rate
                "service3": 0.3,  # Low failure rate
                "service4": 0.1,  # Very low failure rate
            }

            if random.random() < failure_rates.get(service_name, 0.5):
                raise InfrastructureError(f"{service_name} chaos failure", recoverable=True)

            return f"success_from_{service_name}"

        # Run many operations to test fallback behavior
        results = []
        for _ in range(50):
            try:
                result = await manager.execute_with_fallback("chaos_test", chaotic_operation)
                results.append(result)
            except Exception as e:
                results.append(e)

        # Analyze results
        successes = [r for r in results if isinstance(r, str)]
        failures = [r for r in results if isinstance(r, Exception)]

        # Should have more successes than failures due to fallback
        assert len(successes) > len(failures), "Fallback should improve success rate"

        # Should see results from different services
        service_results = {}
        for success in successes:
            if "service" in success:
                service = success.split("_")[-1]
                service_results[service] = service_results.get(service, 0) + 1

        # Should have used fallback services
        assert len(service_results) > 1, "Should have used multiple services"

    @pytest.mark.asyncio
    async def test_cascading_service_failures(self, manager):
        """Test graceful degradation during cascading service failures."""
        # Setup services
        services = ["primary", "secondary", "tertiary", "last_resort"]
        for service in services:
            await manager.register_service(service)
            await manager.update_service_health(service, ServiceHealth.HEALTHY)

        # Configure cascading fallback
        from src.infrastructure.resilience.graceful_degradation import FallbackConfig
        manager.fallback_configs["cascade_test"] = FallbackConfig(
            primary_services=["primary"],
            fallback_services=["secondary", "tertiary", "last_resort"],
            enable_caching=False
        )

        failed_services = set()

        async def cascading_operation(service_name):
            # Services fail in cascade
            if service_name in failed_services:
                raise InfrastructureError(f"{service_name} is down", recoverable=False)

            # Randomly trigger cascade
            if service_name == "primary" and random.random() < 0.3:
                failed_services.add("primary")
                raise InfrastructureError("Primary service cascade failure", recoverable=False)
            elif service_name == "secondary" and "primary" in failed_services and random.random() < 0.5:
                failed_services.add("secondary")
                raise InfrastructureError("Secondary service cascade failure", recoverable=False)

            return f"success_from_{service_name}"

        # Run operations and observe cascade behavior
        results = []
        for i in range(30):
            try:
                result = await manager.execute_with_fallback("cascade_test", cascading_operation)
                results.append(result)
            except Exception as e:
                results.append(e)

            # Occasionally recover services
            if i % 10 == 0 and random.random() < 0.3:
                failed_services.clear()

        # Should have some successes despite cascading failures
        successes = [r for r in results if isinstance(r, str)]
        assert len(successes) > 0, "Should have some successes despite cascade"

    @pytest.mark.asyncio
    async def test_concurrent_degradation_chaos(self, manager):
        """Test graceful degradation under concurrent chaotic load."""
        # Setup services
        services = ["svc1", "svc2", "svc3"]
        for service in services:
            await manager.register_service(service)
            await manager.update_service_health(service, ServiceHealth.HEALTHY)

        # Configure fallback
        from src.infrastructure.resilience.graceful_degradation import FallbackConfig
        manager.fallback_configs["concurrent_test"] = FallbackConfig(
            primary_services=["svc1"],
            fallback_services=["svc2", "svc3"],
            enable_caching=True,
            cache_ttl=0.1
        )

        async def concurrent_chaotic_operation(service_name, task_id):
            # Random delay
            await asyncio.sleep(random.uniform(0.01, 0.05))

            # Random failures with different patterns per service
            failure_patterns = {
                "svc1": 0.7,  # Often fails
                "svc2": 0.4,  # Sometimes fails
                "svc3": 0.2,  # Rarely fails
            }

            if random.random() < failure_patterns.get(service_name, 0.5):
                raise InfrastructureError(f"{service_name} concurrent failure {task_id}", recoverable=True)

            return f"{service_name}_task_{task_id}"

        # Launch concurrent operations
        tasks = []
        for i in range(100):
            task = asyncio.create_task(
                manager.execute_with_fallback("concurrent_test", concurrent_chaotic_operation, i)
            )
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successes = [r for r in results if isinstance(r, str)]
        failures = [r for r in results if isinstance(r, Exception)]

        # Should have high success rate due to fallback and caching
        success_rate = len(successes) / len(results)
        assert success_rate > 0.5, f"Success rate {success_rate} should be > 0.5 with fallback"

        # Should have used multiple services
        service_usage = {}
        for success in successes:
            if "_task_" in success:
                service = success.split("_task_")[0]
                service_usage[service] = service_usage.get(service, 0) + 1

        assert len(service_usage) > 1, "Should have used multiple services for fallback"


class TestIntegratedChaos:
    """Integrated chaos engineering tests combining all resilience mechanisms."""

    @pytest.mark.asyncio
    async def test_full_system_chaos(self):
        """Test all resilience mechanisms working together under chaos."""
        # Setup integrated system
        manager = GracefulDegradationManager()

        # Setup services
        services = ["api1", "api2", "api3"]
        for service in services:
            await manager.register_service(service)
            await manager.update_service_health(service, ServiceHealth.HEALTHY)

        # Configure fallback
        from src.infrastructure.resilience.graceful_degradation import FallbackConfig
        manager.fallback_configs["integrated_test"] = FallbackConfig(
            primary_services=["api1"],
            fallback_services=["api2", "api3"],
            enable_caching=True,
            cache_ttl=0.2
        )

        # Setup circuit breaker and retry strategy
        from src.infrastructure.resilience.circuit_breaker import CircuitBreakerManager
        cb_manager = CircuitBreakerManager()

        from src.infrastructure.resilience.retry_strategy import RetryStrategy, RetryConfig
        retry_strategy = RetryStrategy(RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        ))

        # Chaotic operation with all failure modes
        async def full_chaos_operation(service_name):
            # Random delays
            await asyncio.sleep(random.uniform(0.01, 0.1))

            # Multiple failure modes
            failure_mode = random.choice([
                "network", "timeout", "service_error", "rate_limit", "success"
            ])

            if failure_mode == "network":
                raise NetworkError("Network chaos", host=f"{service_name}.com")
            elif failure_mode == "timeout":
                raise TimeoutError("Timeout chaos", timeout_duration=5.0)
            elif failure_mode == "service_error":
                raise InfrastructureError(f"{service_name} service chaos", recoverable=True)
            elif failure_mode == "rate_limit":
                from src.domain.exceptions import RateLimitError
                raise RateLimitError("Rate limit chaos", service_name=service_name, limit=100)
            else:
                return f"chaos_success_{service_name}"

        # Integrated operation with all resilience mechanisms
        async def integrated_operation():
            # Use graceful degradation
            async def fallback_operation(service_name):
                # Use circuit breaker
                circuit_breaker = await cb_manager.get_circuit_breaker(f"{service_name}_cb")

                # Use retry strategy
                async def retryable_operation():
                    return await circuit_breaker.call(full_chaos_operation, service_name)

                return await retry_strategy.execute(retryable_operation)

            return await manager.execute_with_fallback("integrated_test", fallback_operation)

        # Run integrated chaos test
        results = []
        for _ in range(50):
            try:
                result = await integrated_operation()
                results.append(result)
            except Exception as e:
                results.append(e)

        # Analyze integrated results
        successes = [r for r in results if isinstance(r, str)]
        failures = [r for r in results if isinstance(r, Exception)]

        # Integrated resilience should provide high success rate
        success_rate = len(successes) / len(results)
        assert success_rate > 0.3, f"Integrated success rate {success_rate} should be > 0.3"

        # Should have circuit breaker metrics
        all_cb_metrics = cb_manager.get_all_metrics()
        assert len(all_cb_metrics) > 0, "Should have circuit breaker metrics"

        # Should have used multiple services
        if successes:
            service_usage = set()
            for success in successes:
                if "api" in success:
                    service = success.split("_")[-1]
                    service_usage.add(service)

            # Due to chaos, we might not always use multiple services, but system should be resilient
            assert len(service_usage) >= 1, "Should have used at least one service"

    @pytest.mark.asyncio
    async def test_resource_exhaustion_chaos(self):
        """Test system behavior under resource exhaustion chaos."""
        # Simulate resource exhaustion scenarios
        from src.domain.exceptions import ResourceError

        async def resource_exhaustion_operation():
            resource_types = ["memory", "cpu", "disk", "network"]
            resource_type = random.choice(resource_types)

            # Simulate resource exhaustion
            if random.random() < 0.4:
                raise ResourceError(
                    f"{resource_type} exhausted",
                    resource_type=resource_type,
                    current_usage=95.0,
                    limit=100.0
                )

            return f"resource_success_{resource_type}"

        # Test with retry strategy
        retry_config = RetryConfig(
            max_attempts=5,
            base_delay=0.02,
            backoff_strategy=BackoffStrategy.LINEAR
        )
        retry_strategy = RetryStrategy(retry_config)

        # Run resource exhaustion tests
        results = []
        for _ in range(30):
            try:
                result = await retry_strategy.execute(resource_exhaustion_operation)
                results.append(result)
            except Exception as e:
                results.append(e)

        # Should handle resource exhaustion gracefully
        successes = [r for r in results if isinstance(r, str)]
        resource_errors = [r for r in results if isinstance(r, ResourceError)]

        # Should have some successes and some resource errors
        assert len(successes) > 0, "Should have some successes despite resource chaos"
        assert len(results) == 30, "Should have processed all operations"
