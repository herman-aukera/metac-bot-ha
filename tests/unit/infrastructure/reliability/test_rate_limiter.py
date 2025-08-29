"""Tests for rate limiter implementation."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from src.infrastructure.reliability.rate_limiter import (
    RateLimitConfig,
    RateLimitedClient,
    RateLimiterManager,
    RateLimitStrategy,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    rate_limiter_manager,
)


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiter."""

    @pytest.fixture
    def config(self):
        """Rate limit configuration for testing."""
        return RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            backoff_factor=2.0,
            max_backoff=60.0,
        )

    @pytest.fixture
    def rate_limiter(self, config):
        """Rate limiter instance for testing."""
        return TokenBucketRateLimiter("test_limiter", config)

    @pytest.mark.asyncio
    async def test_initial_tokens(self, rate_limiter):
        """Test initial token availability."""
        # Should have full bucket initially
        assert rate_limiter.tokens == 20.0

        # Should be able to acquire tokens
        result = await rate_limiter.acquire(5)
        assert result is True
        assert rate_limiter.tokens == 15.0

    @pytest.mark.asyncio
    async def test_token_refill(self, rate_limiter):
        """Test token refill over time."""
        # Consume all tokens
        await rate_limiter.acquire(20)
        assert rate_limiter.tokens == 0.0

        # Wait for refill (0.1 seconds = 1 token at 10 tokens/second)
        await asyncio.sleep(0.1)

        # Should be able to acquire 1 token
        result = await rate_limiter.acquire(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiting(self, rate_limiter):
        """Test rate limiting behavior."""
        # Consume all tokens
        result = await rate_limiter.acquire(20)
        assert result is True

        # Next request should be rate limited
        result = await rate_limiter.acquire(1)
        assert result is False
        assert rate_limiter.rejected_requests == 1

    @pytest.mark.asyncio
    async def test_wait_for_tokens(self, rate_limiter):
        """Test waiting for tokens."""
        # Consume all tokens
        await rate_limiter.acquire(20)

        start_time = time.time()

        # This should wait for tokens to be available
        await rate_limiter.wait_for_tokens(1)

        elapsed = time.time() - start_time
        # Should have waited at least some time for refill
        assert elapsed > 0.05  # At least 50ms

    @pytest.mark.asyncio
    async def test_backoff_calculation(self, rate_limiter):
        """Test backoff time calculation."""
        # Consume all tokens
        await rate_limiter.acquire(20)

        # First rejection
        await rate_limiter.acquire(1)
        first_backoff = rate_limiter.current_backoff
        assert first_backoff > 0

        # Second rejection should increase backoff
        await rate_limiter.acquire(1)
        second_backoff = rate_limiter.current_backoff
        assert second_backoff > first_backoff

    @pytest.mark.asyncio
    async def test_disabled_rate_limiter(self, rate_limiter):
        """Test disabled rate limiter."""
        rate_limiter.config.enabled = False

        # Should always allow requests when disabled
        for _ in range(100):
            result = await rate_limiter.acquire(1)
            assert result is True

    def test_metrics(self, rate_limiter):
        """Test rate limiter metrics."""
        metrics = rate_limiter.get_metrics()

        assert metrics["name"] == "test_limiter"
        assert metrics["strategy"] == RateLimitStrategy.TOKEN_BUCKET.value
        assert metrics["requests_per_second"] == 10.0
        assert metrics["burst_size"] == 20
        assert metrics["current_tokens"] == 20.0
        assert metrics["enabled"] is True

    def test_reset(self, rate_limiter):
        """Test rate limiter reset."""
        # Consume some tokens and set backoff
        asyncio.run(rate_limiter.acquire(10))
        rate_limiter.current_backoff = 5.0

        # Reset
        rate_limiter.reset()

        assert rate_limiter.tokens == 20.0
        assert rate_limiter.current_backoff == 0.0


class TestSlidingWindowRateLimiter:
    """Test sliding window rate limiter."""

    @pytest.fixture
    def config(self):
        """Rate limit configuration for testing."""
        return RateLimitConfig(
            requests_per_second=5.0,
            window_size=1.0,  # 1 second window
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )

    @pytest.fixture
    def rate_limiter(self, config):
        """Rate limiter instance for testing."""
        return SlidingWindowRateLimiter("test_sliding", config)

    @pytest.mark.asyncio
    async def test_initial_requests(self, rate_limiter):
        """Test initial request allowance."""
        # Should allow up to 5 requests in 1 second window
        for i in range(5):
            result = await rate_limiter.acquire(1)
            assert result is True

        # 6th request should be rejected
        result = await rate_limiter.acquire(1)
        assert result is False

    @pytest.mark.asyncio
    async def test_window_sliding(self, rate_limiter):
        """Test sliding window behavior."""
        # Make 5 requests
        for _ in range(5):
            await rate_limiter.acquire(1)

        # Wait for window to slide
        await asyncio.sleep(1.1)

        # Should be able to make requests again
        result = await rate_limiter.acquire(1)
        assert result is True

    def test_metrics(self, rate_limiter):
        """Test sliding window metrics."""
        metrics = rate_limiter.get_metrics()

        assert metrics["name"] == "test_sliding"
        assert metrics["strategy"] == RateLimitStrategy.SLIDING_WINDOW.value
        assert metrics["requests_per_second"] == 5.0
        assert metrics["window_size"] == 1.0


class TestRateLimiterManager:
    """Test rate limiter manager."""

    def test_create_rate_limiter(self):
        """Test creating rate limiters."""
        manager = RateLimiterManager()

        config = RateLimitConfig(
            requests_per_second=5.0, strategy=RateLimitStrategy.TOKEN_BUCKET
        )

        limiter = manager.create_rate_limiter("test", config)

        assert isinstance(limiter, TokenBucketRateLimiter)
        assert limiter.name == "test"
        assert limiter.config.requests_per_second == 5.0

    def test_get_rate_limiter(self):
        """Test getting rate limiter."""
        manager = RateLimiterManager()

        config = RateLimitConfig()
        limiter1 = manager.create_rate_limiter("test", config)
        limiter2 = manager.get_rate_limiter("test")

        assert limiter1 is limiter2

        # Non-existent limiter
        limiter3 = manager.get_rate_limiter("nonexistent")
        assert limiter3 is None

    @pytest.mark.asyncio
    async def test_get_all_metrics(self):
        """Test getting all metrics."""
        manager = RateLimiterManager()

        config = RateLimitConfig()
        limiter1 = manager.create_rate_limiter("test1", config)
        limiter2 = manager.create_rate_limiter("test2", config)

        # Make some requests
        await limiter1.acquire(1)
        await limiter2.acquire(1)

        metrics = manager.get_all_metrics()

        assert "test1" in metrics
        assert "test2" in metrics
        assert metrics["test1"]["allowed_requests"] == 1
        assert metrics["test2"]["allowed_requests"] == 1

    def test_reset_all(self):
        """Test resetting all rate limiters."""
        manager = RateLimiterManager()

        config = RateLimitConfig()
        limiter1 = manager.create_rate_limiter("test1", config)
        limiter2 = manager.create_rate_limiter("test2", config)

        # Consume some tokens
        asyncio.run(limiter1.acquire(10))
        asyncio.run(limiter2.acquire(10))

        # Reset all
        manager.reset_all()

        # Should have full tokens again
        assert limiter1.tokens == limiter1.config.burst_size
        assert limiter2.tokens == limiter2.config.burst_size

    def test_enable_disable_all(self):
        """Test enabling/disabling all rate limiters."""
        manager = RateLimiterManager()

        config = RateLimitConfig()
        limiter1 = manager.create_rate_limiter("test1", config)
        limiter2 = manager.create_rate_limiter("test2", config)

        # Disable all
        manager.disable_all()
        assert limiter1.config.enabled is False
        assert limiter2.config.enabled is False

        # Enable all
        manager.enable_all()
        assert limiter1.config.enabled is True
        assert limiter2.config.enabled is True


class TestRateLimitedClient:
    """Test rate-limited client wrapper."""

    @pytest.fixture
    def mock_client(self):
        """Mock client for testing."""
        client = Mock()
        client.test_method = AsyncMock(return_value="success")
        return client

    @pytest.fixture
    def rate_limiter(self):
        """Rate limiter for testing."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=20)
        return TokenBucketRateLimiter("test", config)

    @pytest.fixture
    def rate_limited_client(self, mock_client, rate_limiter):
        """Rate-limited client for testing."""
        return RateLimitedClient("test_client", mock_client, rate_limiter)

    @pytest.mark.asyncio
    async def test_successful_call(self, rate_limited_client, mock_client):
        """Test successful rate-limited call."""
        result = await rate_limited_client.call("test_method")

        assert result == "success"
        assert rate_limited_client.successful_calls == 1
        assert rate_limited_client.total_calls == 1
        mock_client.test_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_with_args(self, rate_limited_client, mock_client):
        """Test call with arguments."""
        mock_client.test_method.return_value = "success_with_args"

        result = await rate_limited_client.call("test_method", "arg1", kwarg1="value1")

        assert result == "success_with_args"
        mock_client.test_method.assert_called_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_rate_limiting(self, rate_limited_client, mock_client):
        """Test rate limiting behavior."""
        # Consume all tokens from rate limiter
        await rate_limited_client.rate_limiter.acquire(20)

        # This call should wait for tokens
        start_time = time.time()
        result = await rate_limited_client.call("test_method")
        elapsed = time.time() - start_time

        assert result == "success"
        assert elapsed > 0.05  # Should have waited

    @pytest.mark.asyncio
    async def test_error_handling(self, rate_limited_client, mock_client):
        """Test error handling."""
        mock_client.test_method.side_effect = ValueError("test error")

        with pytest.raises(ValueError):
            await rate_limited_client.call("test_method")

        assert rate_limited_client.failed_calls == 1

    def test_metrics(self, rate_limited_client):
        """Test client metrics."""
        metrics = rate_limited_client.get_metrics()

        assert metrics["name"] == "test_client"
        assert metrics["total_calls"] == 0
        assert metrics["successful_calls"] == 0
        assert metrics["failed_calls"] == 0
        assert metrics["success_rate"] == 0.0


class TestConvenienceFunctions:
    """Test convenience functions for common rate limiters."""

    def test_create_openai_rate_limiter(self):
        """Test OpenAI rate limiter creation."""
        from src.infrastructure.reliability.rate_limiter import (
            create_openai_rate_limiter,
        )

        limiter = create_openai_rate_limiter()

        assert limiter.name == "openai"
        assert limiter.config.requests_per_second == 3.0
        assert limiter.config.burst_size == 10

    def test_create_metaculus_rate_limiter(self):
        """Test Metaculus rate limiter creation."""
        from src.infrastructure.reliability.rate_limiter import (
            create_metaculus_rate_limiter,
        )

        limiter = create_metaculus_rate_limiter()

        assert limiter.name == "metaculus"
        assert limiter.config.requests_per_second == 2.0
        assert limiter.config.burst_size == 5

    def test_create_tournament_rate_limiter(self):
        """Test tournament rate limiter creation."""
        from src.infrastructure.reliability.rate_limiter import (
            create_tournament_rate_limiter,
        )

        limiter = create_tournament_rate_limiter()

        assert limiter.name == "tournament"
        assert limiter.config.requests_per_second == 1.0
        assert limiter.config.window_size == 300.0


@pytest.mark.asyncio
async def test_global_rate_limiter_manager():
    """Test global rate limiter manager."""
    config = RateLimitConfig(requests_per_second=5.0)
    limiter = rate_limiter_manager.create_rate_limiter("global_test", config)

    assert limiter.name == "global_test"

    result = await limiter.acquire(1)
    assert result is True
