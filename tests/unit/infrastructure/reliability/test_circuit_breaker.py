"""Tests for circuit breaker implementation."""

import asyncio

import pytest

from src.infrastructure.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    circuit_breaker_manager,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def config(self):
        """Circuit breaker configuration for testing."""
        return CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=1.0, success_threshold=2, timeout=0.5
        )

    @pytest.fixture
    def circuit_breaker(self, config):
        """Circuit breaker instance for testing."""
        return CircuitBreaker("test_circuit", config)

    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call."""

        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.successful_requests == 1
        assert circuit_breaker.failed_requests == 0

    @pytest.mark.asyncio
    async def test_failed_call(self, circuit_breaker):
        """Test failed function call."""

        async def fail_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await circuit_breaker.call(fail_func)

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.successful_requests == 0
        assert circuit_breaker.failed_requests == 1
        assert circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, circuit_breaker):
        """Test circuit opens after threshold failures."""

        async def fail_func():
            raise ValueError("test error")

        # Fail enough times to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(fail_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self, circuit_breaker):
        """Test circuit rejects calls when open."""

        async def fail_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(fail_func)

        # Next call should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(fail_func)

        assert circuit_breaker.rejected_requests == 1

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, circuit_breaker):
        """Test circuit transitions to half-open after timeout."""

        async def fail_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(fail_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Check state transition (happens during next call attempt)
        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_after_half_open_successes(self, circuit_breaker):
        """Test circuit closes after successful calls in half-open state."""

        async def fail_func():
            raise ValueError("test error")

        async def success_func():
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(fail_func)

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Make successful calls to close circuit
        for _ in range(2):
            result = await circuit_breaker.call(success_func)
            assert result == "success"

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_reopens_on_half_open_failure(self, circuit_breaker):
        """Test circuit reopens on failure in half-open state."""

        async def fail_func():
            raise ValueError("test error")

        async def success_func():
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(fail_func)

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Make one successful call (transitions to half-open)
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Fail in half-open state
        with pytest.raises(ValueError):
            await circuit_breaker.call(fail_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_timeout_handling(self, circuit_breaker):
        """Test function timeout handling."""

        async def slow_func():
            await asyncio.sleep(1.0)  # Longer than timeout
            return "success"

        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_func)

        assert circuit_breaker.failed_requests == 1

    @pytest.mark.asyncio
    async def test_sync_function_support(self, circuit_breaker):
        """Test support for synchronous functions."""

        def sync_func():
            return "sync_success"

        result = await circuit_breaker.call(sync_func)
        assert result == "sync_success"
        assert circuit_breaker.successful_requests == 1

    @pytest.mark.asyncio
    async def test_metrics(self, circuit_breaker):
        """Test circuit breaker metrics."""

        async def success_func():
            return "success"

        async def fail_func():
            raise ValueError("test error")

        # Make some calls
        await circuit_breaker.call(success_func)

        with pytest.raises(ValueError):
            await circuit_breaker.call(fail_func)

        metrics = circuit_breaker.get_metrics()

        assert metrics["name"] == "test_circuit"
        assert metrics["state"] == CircuitBreakerState.CLOSED.value
        assert metrics["total_requests"] == 2
        assert metrics["successful_requests"] == 1
        assert metrics["failed_requests"] == 1
        assert metrics["failure_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_reset(self, circuit_breaker):
        """Test circuit breaker reset."""

        async def fail_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(fail_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Reset circuit
        await circuit_breaker.reset()

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_force_open(self, circuit_breaker):
        """Test forcing circuit open."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        await circuit_breaker.force_open()

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Should reject calls
        async def success_func():
            return "success"

        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(success_func)


class TestCircuitBreakerManager:
    """Test circuit breaker manager."""

    def test_get_circuit_breaker(self):
        """Test getting circuit breaker from manager."""
        manager = CircuitBreakerManager()

        cb1 = manager.get_circuit_breaker("test1")
        cb2 = manager.get_circuit_breaker("test1")  # Same instance
        cb3 = manager.get_circuit_breaker("test2")  # Different instance

        assert cb1 is cb2
        assert cb1 is not cb3
        assert cb1.name == "test1"
        assert cb3.name == "test2"

    @pytest.mark.asyncio
    async def test_get_all_metrics(self):
        """Test getting metrics for all circuit breakers."""
        manager = CircuitBreakerManager()

        cb1 = manager.get_circuit_breaker("test1")
        cb2 = manager.get_circuit_breaker("test2")

        # Make some calls to generate metrics
        async def success_func():
            return "success"

        await cb1.call(success_func)
        await cb2.call(success_func)

        metrics = manager.get_all_metrics()

        assert "test1" in metrics
        assert "test2" in metrics
        assert metrics["test1"]["successful_requests"] == 1
        assert metrics["test2"]["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all circuit breakers."""
        manager = CircuitBreakerManager()

        cb1 = manager.get_circuit_breaker("test1")
        cb2 = manager.get_circuit_breaker("test2")

        # Force open both circuits
        await cb1.force_open()
        await cb2.force_open()

        assert cb1.state == CircuitBreakerState.OPEN
        assert cb2.state == CircuitBreakerState.OPEN

        # Reset all
        await manager.reset_all()

        assert cb1.state == CircuitBreakerState.CLOSED
        assert cb2.state == CircuitBreakerState.CLOSED

    def test_get_unhealthy_circuits(self):
        """Test getting unhealthy circuits."""
        manager = CircuitBreakerManager()

        cb1 = manager.get_circuit_breaker("test1")
        manager.get_circuit_breaker("test2")

        # Initially all healthy
        unhealthy = manager.get_unhealthy_circuits()
        assert len(unhealthy) == 0

        # Force one open
        asyncio.run(cb1.force_open())

        unhealthy = manager.get_unhealthy_circuits()
        assert len(unhealthy) == 1
        assert "test1" in unhealthy

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check for all circuits."""
        manager = CircuitBreakerManager()

        cb1 = manager.get_circuit_breaker("test1")
        manager.get_circuit_breaker("test2")

        health = await manager.health_check()

        assert health["test1"] is True
        assert health["test2"] is True

        # Force one open
        await cb1.force_open()

        health = await manager.health_check()

        assert health["test1"] is False
        assert health["test2"] is True


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0
        assert config.expected_exception == Exception

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120.0,
            success_threshold=5,
            timeout=60.0,
            expected_exception=ValueError,
        )

        assert config.failure_threshold == 10
        assert config.recovery_timeout == 120.0
        assert config.success_threshold == 5
        assert config.timeout == 60.0
        assert config.expected_exception == ValueError


@pytest.mark.asyncio
async def test_global_circuit_breaker_manager():
    """Test global circuit breaker manager instance."""
    cb = circuit_breaker_manager.get_circuit_breaker("global_test")

    assert cb.name == "global_test"

    async def success_func():
        return "success"

    result = await cb.call(success_func)
    assert result == "success"
