"""
Unit tests for circuit breaker implementation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerOpenError,
    CircuitBreakerManager,
)


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60
        assert config.success_threshold == 3
        assert config.timeout == 30.0
        assert config.expected_exception_types == (Exception,)
        assert config.monitor_window == 300


class TestCircuitBreakerMetrics:
    """Test CircuitBreakerMetrics functionality."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = CircuitBreakerMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.timeouts == 0
        assert metrics.circuit_opens == 0
        assert metrics.circuit_closes == 0
        assert metrics.last_failure_time is None
        assert metrics.last_success_time is None
        assert metrics.failure_history == []

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        metrics = CircuitBreakerMetrics()

        # No requests
        assert metrics.failure_rate == 0.0

        # Some requests
        metrics.total_requests = 10
        metrics.failed_requests = 3
        assert metrics.failure_rate == 0.3

        # All failed
        metrics.failed_requests = 10
        assert metrics.failure_rate == 1.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = CircuitBreakerMetrics()

        # No requests
        assert metrics.success_rate == 0.0

        # Some requests
        metrics.total_requests = 10
        metrics.successful_requests = 7
        assert metrics.success_rate == 0.7

        # All successful
        metrics.successful_requests = 10
        assert metrics.success_rate == 1.0


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,  # Short timeout for testing
            success_threshold=2,
            timeout=1.0
        )
        return CircuitBreaker("test_service", config)

    @pytest.fixture
    def mock_function(self):
        """Create mock async function."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker, mock_function):
        """Test successful function call."""
        mock_function.return_value = "success"

        result = await circuit_breaker.call(mock_function, "arg1", key="value")

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.total_requests == 1
        assert circuit_breaker.metrics.successful_requests == 1
        assert circuit_breaker.metrics.failed_requests == 0
        assert circuit_breaker.consecutive_failures == 0
        assert circuit_breaker.consecutive_successes == 1

    @pytest.mark.asyncio
    async def test_failed_call(self, circuit_breaker, mock_function):
        """Test failed function call."""
        mock_function.side_effect = ValueError("Test error")

        with pytest.raises(ValueError):
            await circuit_breaker.call(mock_function)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.total_requests == 1
        assert circuit_breaker.metrics.successful_requests == 0
        assert circuit_breaker.metrics.failed_requests == 1
        assert circuit_breaker.consecutive_failures == 1
        assert circuit_breaker.consecutive_successes == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, circuit_breaker, mock_function):
        """Test circuit opens after threshold failures."""
        mock_function.side_effect = ValueError("Test error")

        # Fail enough times to open circuit
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(mock_function)

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.metrics.circuit_opens == 1
        assert circuit_breaker.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_error(self, circuit_breaker, mock_function):
        """Test CircuitBreakerOpenError when circuit is open."""
        mock_function.side_effect = ValueError("Test error")

        # Open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(mock_function)

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await circuit_breaker.call(mock_function)

        assert "Circuit breaker is open for service: test_service" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_recovery_to_half_open(self, circuit_breaker, mock_function):
        """Test circuit recovery to half-open state."""
        mock_function.side_effect = ValueError("Test error")

        # Open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(mock_function)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Next call should transition to half-open
        mock_function.side_effect = None
        mock_function.return_value = "success"

        result = await circuit_breaker.call(mock_function)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_after_successes(self, circuit_breaker, mock_function):
        """Test circuit closes after successful calls in half-open state."""
        mock_function.side_effect = ValueError("Test error")

        # Open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(mock_function)

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Succeed enough times to close circuit
        mock_function.side_effect = None
        mock_function.return_value = "success"

        for i in range(2):  # success_threshold = 2
            await circuit_breaker.call(mock_function)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.circuit_closes == 1

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker, mock_function):
        """Test failure in half-open state reopens circuit."""
        mock_function.side_effect = ValueError("Test error")

        # Open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(mock_function)

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Fail in half-open state
        with pytest.raises(ValueError):
            await circuit_breaker.call(mock_function)

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.metrics.circuit_opens == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self, circuit_breaker):
        """Test timeout handling."""
        async def slow_function():
            await asyncio.sleep(2)  # Longer than timeout
            return "success"

        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_function)

        assert circuit_breaker.metrics.timeouts == 1
        assert circuit_breaker.metrics.failed_requests == 1

    def test_get_metrics(self, circuit_breaker):
        """Test metrics retrieval."""
        metrics = circuit_breaker.get_metrics()

        assert metrics["name"] == "test_service"
        assert metrics["state"] == CircuitState.CLOSED.value
        assert metrics["consecutive_failures"] == 0
        assert metrics["consecutive_successes"] == 0
        assert metrics["total_requests"] == 0
        assert "failure_rate" in metrics
        assert "success_rate" in metrics

    def test_reset(self, circuit_breaker):
        """Test circuit breaker reset."""
        # Modify state
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.consecutive_failures = 5
        circuit_breaker.metrics.total_requests = 10

        circuit_breaker.reset()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.consecutive_failures == 0
        assert circuit_breaker.consecutive_successes == 0
        assert circuit_breaker.metrics.total_requests == 0


class TestCircuitBreakerManager:
    """Test CircuitBreakerManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create circuit breaker manager for testing."""
        return CircuitBreakerManager()

    @pytest.mark.asyncio
    async def test_get_circuit_breaker(self, manager):
        """Test getting circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)

        cb1 = await manager.get_circuit_breaker("service1", config)
        cb2 = await manager.get_circuit_breaker("service1")  # Should return same instance
        cb3 = await manager.get_circuit_breaker("service2")  # Should create new instance

        assert cb1 is cb2
        assert cb1 is not cb3
        assert cb1.name == "service1"
        assert cb3.name == "service2"

    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker(self, manager):
        """Test calling function with circuit breaker protection."""
        mock_function = AsyncMock(return_value="success")

        result = await manager.call_with_circuit_breaker(
            "test_service",
            mock_function,
            "arg1",
            key="value"
        )

        assert result == "success"
        mock_function.assert_called_once_with("arg1", key="value")

    def test_get_all_metrics(self, manager):
        """Test getting all circuit breaker metrics."""
        # Create some circuit breakers
        asyncio.run(manager.get_circuit_breaker("service1"))
        asyncio.run(manager.get_circuit_breaker("service2"))

        metrics = manager.get_all_metrics()

        assert "service1" in metrics
        assert "service2" in metrics
        assert metrics["service1"]["name"] == "service1"
        assert metrics["service2"]["name"] == "service2"

    def test_reset_all(self, manager):
        """Test resetting all circuit breakers."""
        # Create and modify circuit breakers
        cb1 = asyncio.run(manager.get_circuit_breaker("service1"))
        cb2 = asyncio.run(manager.get_circuit_breaker("service2"))

        cb1.state = CircuitState.OPEN
        cb2.consecutive_failures = 5

        manager.reset_all()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.consecutive_failures == 0

    def test_get_unhealthy_services(self, manager):
        """Test getting unhealthy services."""
        # Create circuit breakers
        cb1 = asyncio.run(manager.get_circuit_breaker("service1"))
        cb2 = asyncio.run(manager.get_circuit_breaker("service2"))
        cb3 = asyncio.run(manager.get_circuit_breaker("service3"))

        # Open some circuits
        cb1.state = CircuitState.OPEN
        cb3.state = CircuitState.OPEN

        unhealthy = manager.get_unhealthy_services()

        assert set(unhealthy) == {"service1", "service3"}
        assert "service2" not in unhealthy
