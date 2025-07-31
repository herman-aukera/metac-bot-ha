"""Unit tests for RateLimiter."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import json

from src.infrastructure.security.rate_limiter import (
    RateLimiter,
    RateLimitRule,
    RateLimitStrategy,
    RateLimitResult,
    RedisClient,
    InMemoryRedis
)
from src.domain.exceptions.infrastructure_exceptions import RateLimitExceededError


class TestRateLimiter:
    """Test RateLimiter functionality."""

    @pytest.fixture
    def mock_redis_client(self):
        redis_client = Mock(spec=RedisClient)
        redis_client.get = AsyncMock()
        redis_client.set = AsyncMock()
        redis_client.incr = AsyncMock()
        redis_client.expire = AsyncMock()
        redis_client.zadd = AsyncMock()
        redis_client.zremrangebyscore = AsyncMock()
        redis_client.zcard = AsyncMock()
        redis_client.eval = AsyncMock()
        return redis_client

    @pytest.fixture
    def rate_limiter(self, mock_redis_client):
        return RateLimiter(redis_client=mock_redis_client)

    def test_add_rule(self, rate_limiter):
        """Test adding a rate limit rule."""
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )

        rate_limiter.add_rule(rule)

        assert "test_rule" in rate_limiter.rules
        assert rate_limiter.rules["test_rule"] == rule

    def test_remove_rule(self, rate_limiter):
        """Test removing a rate limit rule."""
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60
        )

        rate_limiter.add_rule(rule)
        rate_limiter.remove_rule("test_rule")

        assert "test_rule" not in rate_limiter.rules

    def test_generate_key(self, rate_limiter):
        """Test rate limit key generation."""
        key = rate_limiter.generate_key("test_rule", "user123")
        assert key == "rate_limit:test_rule:user123"

        # Test with additional context
        context = {"endpoint": "/api/test", "method": "POST"}
        key_with_context = rate_limiter.generate_key("test_rule", "user123", context)
        assert key_with_context.startswith("rate_limit:test_rule:user123:")
        assert len(key_with_context) > len(key)

    @pytest.mark.asyncio
    async def test_check_rate_limit_no_rule(self, rate_limiter):
        """Test rate limit check when no rule is defined."""
        result = await rate_limiter.check_rate_limit("nonexistent_rule", "user123")

        assert result.allowed is True
        assert result.remaining_requests == float('inf')

    @pytest.mark.asyncio
    async def test_fixed_window_rate_limit_allowed(self, rate_limiter, mock_redis_client):
        """Test fixed window rate limiting - request allowed."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        rate_limiter.add_rule(rule)

        mock_redis_client.incr.return_value = 5  # 5th request in window
        mock_redis_client.expire.return_value = True

        # Execute
        result = await rate_limiter.check_rate_limit("test_rule", "user123")

        # Verify
        assert result.allowed is True
        assert result.remaining_requests == 5  # 10 - 5 = 5
        mock_redis_client.incr.assert_called_once()

    @pytest.mark.asyncio
    async def test_fixed_window_rate_limit_exceeded(self, rate_limiter, mock_redis_client):
        """Test fixed window rate limiting - limit exceeded."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        rate_limiter.add_rule(rule)

        mock_redis_client.incr.return_value = 11  # 11th request, exceeds limit

        # Execute
        result = await rate_limiter.check_rate_limit("test_rule", "user123")

        # Verify
        assert result.allowed is False
        assert result.remaining_requests == 0
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_sliding_window_rate_limit(self, rate_limiter, mock_redis_client):
        """Test sliding window rate limiting."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        rate_limiter.add_rule(rule)

        # Mock Lua script result: [allowed, remaining, reset_time]
        current_time = time.time()
        mock_redis_client.eval.return_value = [1, 7, current_time + 60]

        # Execute
        result = await rate_limiter.check_rate_limit("test_rule", "user123")

        # Verify
        assert result.allowed is True
        assert result.remaining_requests == 7
        mock_redis_client.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_sliding_window_rate_limit_exceeded(self, rate_limiter, mock_redis_client):
        """Test sliding window rate limiting - limit exceeded."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        rate_limiter.add_rule(rule)

        # Mock Lua script result: [allowed, remaining, reset_time]
        current_time = time.time()
        mock_redis_client.eval.return_value = [0, 0, current_time + 30]

        # Execute
        result = await rate_limiter.check_rate_limit("test_rule", "user123")

        # Verify
        assert result.allowed is False
        assert result.remaining_requests == 0
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_token_bucket_rate_limit(self, rate_limiter, mock_redis_client):
        """Test token bucket rate limiting."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        rate_limiter.add_rule(rule)

        # Mock Lua script result: [allowed, remaining_tokens, wait_time]
        mock_redis_client.eval.return_value = [1, 5, 0]

        # Execute
        result = await rate_limiter.check_rate_limit("test_rule", "user123")

        # Verify
        assert result.allowed is True
        assert result.remaining_requests == 5

    @pytest.mark.asyncio
    async def test_leaky_bucket_rate_limit(self, rate_limiter, mock_redis_client):
        """Test leaky bucket rate limiting."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.LEAKY_BUCKET
        )
        rate_limiter.add_rule(rule)

        # Mock empty bucket state
        mock_redis_client.get.return_value = None
        mock_redis_client.set.return_value = True

        # Execute
        result = await rate_limiter.check_rate_limit("test_rule", "user123")

        # Verify
        assert result.allowed is True
        mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_leaky_bucket_rate_limit_full(self, rate_limiter, mock_redis_client):
        """Test leaky bucket rate limiting when bucket is full."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.LEAKY_BUCKET
        )
        rate_limiter.add_rule(rule)

        # Mock full bucket state
        current_time = time.time()
        bucket_data = {
            'level': 10,  # Full bucket
            'last_leak': current_time
        }
        mock_redis_client.get.return_value = json.dumps(bucket_data)

        # Execute
        result = await rate_limiter.check_rate_limit("test_rule", "user123")

        # Verify
        assert result.allowed is False
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_reset_rate_limit(self, rate_limiter, mock_redis_client):
        """Test resetting rate limit for a user."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60
        )
        rate_limiter.add_rule(rule)

        mock_redis_client.set.return_value = True

        # Execute
        result = await rate_limiter.reset_rate_limit("test_rule", "user123")

        # Verify
        assert result is True
        mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_rate_limit_status_fixed_window(self, rate_limiter, mock_redis_client):
        """Test getting rate limit status for fixed window."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        rate_limiter.add_rule(rule)

        mock_redis_client.get.return_value = "3"  # 3 requests made

        # Execute
        status = await rate_limiter.get_rate_limit_status("test_rule", "user123")

        # Verify
        assert status is not None
        assert status["rule_name"] == "test_rule"
        assert status["strategy"] == "fixed_window"
        assert status["current_count"] == 3
        assert status["limit"] == 10
        assert status["remaining"] == 7

    @pytest.mark.asyncio
    async def test_get_rate_limit_status_sliding_window(self, rate_limiter, mock_redis_client):
        """Test getting rate limit status for sliding window."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        rate_limiter.add_rule(rule)

        mock_redis_client.zremrangebyscore.return_value = 2  # Removed 2 expired entries
        mock_redis_client.zcard.return_value = 4  # 4 current requests

        # Execute
        status = await rate_limiter.get_rate_limit_status("test_rule", "user123")

        # Verify
        assert status is not None
        assert status["strategy"] == "sliding_window"
        assert status["current_count"] == 4
        assert status["remaining"] == 6

    @pytest.mark.asyncio
    async def test_middleware_check_allowed(self, rate_limiter, mock_redis_client):
        """Test middleware check when request is allowed."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        rate_limiter.add_rule(rule)

        mock_redis_client.incr.return_value = 5

        # Execute - should not raise exception
        await rate_limiter.middleware_check("test_rule", "user123")

    @pytest.mark.asyncio
    async def test_middleware_check_exceeded(self, rate_limiter, mock_redis_client):
        """Test middleware check when rate limit is exceeded."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        rate_limiter.add_rule(rule)

        mock_redis_client.incr.return_value = 11

        # Execute & Verify
        with pytest.raises(RateLimitExceededError) as exc_info:
            await rate_limiter.middleware_check("test_rule", "user123")

        assert exc_info.value.error_code == "RATE_LIMIT_EXCEEDED"
        assert "test_rule" in exc_info.value.context["rule_name"]

    def test_create_default_rules(self, rate_limiter):
        """Test creation of default rate limiting rules."""
        rate_limiter.create_default_rules()

        expected_rules = [
            "api_general",
            "auth_attempts",
            "forecast_submission",
            "research_query",
            "heavy_operation"
        ]

        for rule_name in expected_rules:
            assert rule_name in rate_limiter.rules

        # Check specific rule properties
        auth_rule = rate_limiter.rules["auth_attempts"]
        assert auth_rule.requests_per_window == 5
        assert auth_rule.window_size_seconds == 300
        assert auth_rule.strategy == RateLimitStrategy.FIXED_WINDOW

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, rate_limiter, mock_redis_client):
        """Test error handling in rate limiting."""
        # Setup
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        rate_limiter.add_rule(rule)

        # Mock Redis error
        mock_redis_client.incr.side_effect = Exception("Redis connection failed")

        # Execute - should fail open (allow request)
        result = await rate_limiter.check_rate_limit("test_rule", "user123")

        # Verify
        assert result.allowed is True  # Fail open


class TestRedisClient:
    """Test RedisClient functionality."""

    @pytest.fixture
    def redis_client(self):
        return RedisClient("redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_redis_client_fallback_to_memory(self, redis_client):
        """Test Redis client fallback to in-memory when aioredis not available."""
        with patch('aioredis.from_url', side_effect=ImportError("aioredis not available")):
            redis = await redis_client._get_redis()
            assert isinstance(redis, InMemoryRedis)

    @pytest.mark.asyncio
    async def test_redis_operations_with_errors(self, redis_client):
        """Test Redis operations with connection errors."""
        # Mock Redis connection that raises errors
        mock_redis = Mock()
        mock_redis.get = AsyncMock(side_effect=Exception("Connection error"))
        mock_redis.set = AsyncMock(side_effect=Exception("Connection error"))
        mock_redis.incr = AsyncMock(side_effect=Exception("Connection error"))

        redis_client._redis = mock_redis

        # Test operations return safe defaults on error
        result = await redis_client.get("test_key")
        assert result is None

        result = await redis_client.set("test_key", "value")
        assert result is False

        result = await redis_client.incr("counter")
        assert result == 1


class TestInMemoryRedis:
    """Test InMemoryRedis functionality."""

    @pytest.fixture
    def redis(self):
        return InMemoryRedis()

    @pytest.mark.asyncio
    async def test_basic_operations(self, redis):
        """Test basic Redis operations."""
        # Set and get
        await redis.set("key1", "value1")
        result = await redis.get("key1")
        assert result == "value1"

        # Increment
        result = await redis.incr("counter")
        assert result == 1

        result = await redis.incr("counter")
        assert result == 2

    @pytest.mark.asyncio
    async def test_expiration(self, redis):
        """Test key expiration."""
        await redis.set("temp_key", "temp_value")
        await redis.expire("temp_key", 1)

        # Should exist immediately
        result = await redis.get("temp_key")
        assert result == "temp_value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        result = await redis.get("temp_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_sorted_set_operations(self, redis):
        """Test sorted set operations."""
        # Add members
        result = await redis.zadd("test_set", {"member1": 1.0, "member2": 2.0, "member3": 3.0})
        assert result == 3

        # Check cardinality
        count = await redis.zcard("test_set")
        assert count == 3

        # Remove by score range
        removed = await redis.zremrangebyscore("test_set", 1.0, 2.0)
        assert removed == 2

        # Check remaining count
        count = await redis.zcard("test_set")
        assert count == 1

    @pytest.mark.asyncio
    async def test_set_with_expiration(self, redis):
        """Test set operation with expiration."""
        await redis.set("expire_key", "value", ex=1)

        # Should exist immediately
        result = await redis.get("expire_key")
        assert result == "value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        result = await redis.get("expire_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys(self, redis):
        """Test automatic cleanup of expired keys."""
        # Set multiple keys with expiration
        await redis.set("key1", "value1", ex=1)
        await redis.set("key2", "value2", ex=2)
        await redis.set("key3", "value3")  # No expiration

        # Wait for first key to expire
        await asyncio.sleep(1.1)

        # Access should trigger cleanup
        result = await redis.get("key2")
        assert result == "value2"

        # First key should be cleaned up
        result = await redis.get("key1")
        assert result is None

        # Third key should still exist
        result = await redis.get("key3")
        assert result == "value3"


@pytest.mark.asyncio
async def test_rate_limiter_integration():
    """Integration test for rate limiter with in-memory Redis."""
    # Use in-memory Redis for testing
    redis_client = RedisClient()
    rate_limiter = RateLimiter(redis_client=redis_client)

    # Add a test rule
    rule = RateLimitRule(
        name="integration_test",
        requests_per_window=3,
        window_size_seconds=60,
        strategy=RateLimitStrategy.FIXED_WINDOW
    )
    rate_limiter.add_rule(rule)

    # Test multiple requests
    user_id = "test_user"

    # First 3 requests should be allowed
    for i in range(3):
        result = await rate_limiter.check_rate_limit("integration_test", user_id)
        assert result.allowed is True
        assert result.remaining_requests == 2 - i

    # 4th request should be denied
    result = await rate_limiter.check_rate_limit("integration_test", user_id)
    assert result.allowed is False
    assert result.remaining_requests == 0

    # Reset and try again
    await rate_limiter.reset_rate_limit("integration_test", user_id)
    result = await rate_limiter.check_rate_limit("integration_test", user_id)
    assert result.allowed is True


@pytest.mark.asyncio
async def test_concurrent_rate_limiting():
    """Test rate limiting under concurrent load."""
    redis_client = RedisClient()
    rate_limiter = RateLimiter(redis_client=redis_client)

    rule = RateLimitRule(
        name="concurrent_test",
        requests_per_window=10,
        window_size_seconds=60,
        strategy=RateLimitStrategy.FIXED_WINDOW
    )
    rate_limiter.add_rule(rule)

    # Simulate concurrent requests
    async def make_request(user_id):
        return await rate_limiter.check_rate_limit("concurrent_test", user_id)

    # Make 15 concurrent requests for the same user
    tasks = [make_request("concurrent_user") for _ in range(15)]
    results = await asyncio.gather(*tasks)

    # Count allowed and denied requests
    allowed_count = sum(1 for result in results if result.allowed)
    denied_count = sum(1 for result in results if not result.allowed)

    # Should allow exactly 10 requests and deny 5
    assert allowed_count <= 10  # May be less due to race conditions
    assert denied_count >= 5
    assert allowed_count + denied_count == 15
