"""
Unit tests for retry strategy implementation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import time

from src.infrastructure.resilience.retry_strategy import (
    RetryStrategy,
    RetryConfig,
    BackoffStrategy,
    RetryExhaustedException,
    RetryableOperation,
    AdaptiveRetryStrategy,
)
from src.domain.exceptions import InfrastructureError, TimeoutError


class TestRetryConfig:
    """Test RetryConfig functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER
        assert config.jitter_factor == 0.1
        assert config.multiplier == 2.0
        assert InfrastructureError in config.retryable_exceptions
        assert ValueError in config.non_retryable_exceptions


class TestRetryStrategy:
    """Test RetryStrategy functionality."""

    @pytest.fixture
    def retry_strategy(self):
        """Create retry strategy for testing."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Short delay for testing
            max_delay=1.0,
            backoff_strategy=BackoffStrategy.FIXED
        )
        return RetryStrategy(config)

    @pytest.fixture
    def mock_function(self):
        """Create mock async function."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_successful_execution(self, retry_strategy, mock_function):
        """Test successful function execution on first try."""
        mock_function.return_value = "success"

        result = await retry_strategy.execute(mock_function, "arg1", key="value")

        assert result == "success"
        assert mock_function.call_count == 1
        mock_function.assert_called_with("arg1", key="value")

    @pytest.mark.asyncio
    async def test_retry_on_retryable_exception(self, retry_strategy, mock_function):
        """Test retry on retryable exception."""
        # Fail twice, then succeed
        mock_function.side_effect = [
            InfrastructureError("Transient error", recoverable=True),
            InfrastructureError("Another error", recoverable=True),
            "success"
        ]

        result = await retry_strategy.execute(mock_function)

        assert result == "success"
        assert mock_function.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_exception(self, retry_strategy, mock_function):
        """Test no retry on non-retryable exception."""
        mock_function.side_effect = ValueError("Non-retryable error")

        with pytest.raises(ValueError):
            await retry_strategy.execute(mock_function)

        assert mock_function.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, retry_strategy, mock_function):
        """Test retry exhausted exception."""
        mock_function.side_effect = InfrastructureError("Persistent error", recoverable=True)

        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_strategy.execute(mock_function)

        assert mock_function.call_count == 3
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, InfrastructureError)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test overall timeout handling."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.1,
            timeout=0.2  # Very short timeout
        )
        retry_strategy = RetryStrategy(config)

        async def slow_function():
            await asyncio.sleep(0.1)
            raise InfrastructureError("Error", recoverable=True)

        with pytest.raises(TimeoutError):
            await retry_strategy.execute(slow_function)

    def test_is_retryable_exception(self, retry_strategy):
        """Test exception retryability classification."""
        # Retryable exceptions
        assert retry_strategy._is_retryable_exception(
            InfrastructureError("Error", recoverable=True)
        )
        assert retry_strategy._is_retryable_exception(ConnectionError("Connection failed"))

        # Non-retryable exceptions
        assert not retry_strategy._is_retryable_exception(ValueError("Invalid value"))
        assert not retry_strategy._is_retryable_exception(
            InfrastructureError("Error", recoverable=False)
        )

    def test_calculate_delay_fixed(self):
        """Test fixed backoff delay calculation."""
        config = RetryConfig(
            base_delay=2.0,
            backoff_strategy=BackoffStrategy.FIXED
        )
        retry_strategy = RetryStrategy(config)

        assert retry_strategy._calculate_delay(0) == 2.0
        assert retry_strategy._calculate_delay(1) == 2.0
        assert retry_strategy._calculate_delay(5) == 2.0

    def test_calculate_delay_linear(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_strategy=BackoffStrategy.LINEAR
        )
        retry_strategy = RetryStrategy(config)

        assert retry_strategy._calculate_delay(0) == 1.0
        assert retry_strategy._calculate_delay(1) == 2.0
        assert retry_strategy._calculate_delay(2) == 3.0

    def test_calculate_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            multiplier=2.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        retry_strategy = RetryStrategy(config)

        assert retry_strategy._calculate_delay(0) == 1.0
        assert retry_strategy._calculate_delay(1) == 2.0
        assert retry_strategy._calculate_delay(2) == 4.0

    def test_calculate_delay_max_limit(self):
        """Test delay maximum limit."""
        config = RetryConfig(
            base_delay=10.0,
            max_delay=5.0,
            multiplier=2.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        retry_strategy = RetryStrategy(config)

        assert retry_strategy._calculate_delay(0) == 5.0  # Limited by max_delay
        assert retry_strategy._calculate_delay(1) == 5.0  # Limited by max_delay

    @patch('random.random')
    def test_calculate_delay_exponential_jitter(self, mock_random):
        """Test exponential backoff with jitter."""
        mock_random.return_value = 0.5  # Fixed random value

        config = RetryConfig(
            base_delay=2.0,
            multiplier=2.0,
            jitter_factor=0.2,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER
        )
        retry_strategy = RetryStrategy(config)

        # base_delay * multiplier^attempt + jitter
        # 2.0 * 2^0 + (2.0 * 0.2 * 0.5) = 2.0 + 0.2 = 2.2
        assert retry_strategy._calculate_delay(0) == 2.2


class TestRetryableOperation:
    """Test RetryableOperation decorator and context manager."""

    @pytest.fixture
    def retryable_op(self):
        """Create retryable operation for testing."""
        config = RetryConfig(
            max_attempts=2,
            base_delay=0.01,
            backoff_strategy=BackoffStrategy.FIXED
        )
        return RetryableOperation(config)

    @pytest.mark.asyncio
    async def test_decorator_usage(self, retryable_op):
        """Test using RetryableOperation as decorator."""
        call_count = 0

        @retryable_op
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise InfrastructureError("Transient error", recoverable=True)
            return "success"

        result = await test_function()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, retryable_op):
        """Test using RetryableOperation as context manager."""
        async def test_function():
            raise InfrastructureError("Transient error", recoverable=True)

        async with retryable_op as retry_op:
            with pytest.raises(RetryExhaustedException):
                await retry_op.execute(test_function)


class TestAdaptiveRetryStrategy:
    """Test AdaptiveRetryStrategy functionality."""

    @pytest.fixture
    def adaptive_strategy(self):
        """Create adaptive retry strategy for testing."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.01,
            backoff_strategy=BackoffStrategy.FIXED
        )
        return AdaptiveRetryStrategy(config)

    @pytest.mark.asyncio
    async def test_success_tracking(self, adaptive_strategy):
        """Test success tracking and adaptation."""
        mock_function = AsyncMock(return_value="success")

        # Execute multiple successful operations
        for _ in range(10):
            await adaptive_strategy.execute(mock_function)

        assert len(adaptive_strategy.success_history) == 10
        assert all(attempts == 1 for attempts in adaptive_strategy.success_history)

    @pytest.mark.asyncio
    async def test_failure_tracking(self, adaptive_strategy):
        """Test failure tracking."""
        mock_function = AsyncMock(side_effect=ValueError("Non-retryable"))

        # Execute multiple failed operations
        for _ in range(3):
            with pytest.raises(ValueError):
                await adaptive_strategy.execute(mock_function)

        assert "ValueError" in adaptive_strategy.failure_patterns
        assert len(adaptive_strategy.failure_patterns["ValueError"]) == 3

    def test_adaptation_metrics(self, adaptive_strategy):
        """Test adaptation metrics."""
        # Add some test data
        adaptive_strategy.success_history = [1, 1, 2, 1, 1]
        adaptive_strategy.failure_patterns = {
            "ValueError": [5, 5, 5],
            "ConnectionError": [3, 4]
        }

        metrics = adaptive_strategy.get_adaptation_metrics()

        assert metrics["success_history_size"] == 5
        assert metrics["avg_attempts_for_success"] == 1.2
        assert "ValueError" in metrics["failure_patterns"]
        assert metrics["failure_patterns"]["ValueError"]["count"] == 3
        assert metrics["failure_patterns"]["ValueError"]["avg_attempts"] == 5.0

    def test_configuration_adaptation(self, adaptive_strategy):
        """Test configuration adaptation based on history."""
        # Simulate mostly successful operations with few attempts
        adaptive_strategy.success_history = [1] * 20  # All succeed on first try
        adaptive_strategy.adaptation_enabled = True

        original_max_attempts = adaptive_strategy.config.max_attempts

        # Trigger adaptation
        adaptive_strategy._adapt_configuration()

        # Should reduce max_attempts since operations succeed quickly
        assert adaptive_strategy.config.max_attempts <= original_max_attempts

    def test_history_size_limits(self, adaptive_strategy):
        """Test history size limits."""
        # Add more than the limit
        adaptive_strategy.success_history = list(range(150))
        adaptive_strategy._record_success(1)

        # Should be trimmed to 50
        assert len(adaptive_strategy.success_history) == 50

        # Test failure history limits
        adaptive_strategy.failure_patterns["TestError"] = list(range(60))
        adaptive_strategy._record_failure("TestError", 3)

        # Should be trimmed to 25
        assert len(adaptive_strategy.failure_patterns["TestError"]) == 25
