"""Rate limiter with Redis backend for API protection and abuse prevention."""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

from ...domain.exceptions.infrastructure_exceptions import RateLimitExceededError


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    name: str
    requests_per_window: int
    window_size_seconds: int
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_allowance: Optional[int] = None
    key_generator: Optional[str] = None  # Custom key generation logic


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining_requests: int
    reset_time: float
    retry_after: Optional[int] = None


class RedisClient:
    """Redis client abstraction for rate limiting."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
        self.logger = logging.getLogger(__name__)

    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import aioredis
                self._redis = aioredis.from_url(self.redis_url)
            except ImportError:
                self.logger.warning("aioredis not available, using in-memory fallback")
                self._redis = InMemoryRedis()
        return self._redis

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        redis = await self._get_redis()
        try:
            return await redis.get(key)
        except Exception as e:
            self.logger.error(f"Redis GET error: {e}")
            return None

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration."""
        redis = await self._get_redis()
        try:
            return await redis.set(key, value, ex=ex)
        except Exception as e:
            self.logger.error(f"Redis SET error: {e}")
            return False

    async def incr(self, key: str) -> int:
        """Increment value in Redis."""
        redis = await self._get_redis()
        try:
            return await redis.incr(key)
        except Exception as e:
            self.logger.error(f"Redis INCR error: {e}")
            return 1

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key."""
        redis = await self._get_redis()
        try:
            return await redis.expire(key, seconds)
        except Exception as e:
            self.logger.error(f"Redis EXPIRE error: {e}")
            return False

    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add to sorted set."""
        redis = await self._get_redis()
        try:
            return await redis.zadd(key, mapping)
        except Exception as e:
            self.logger.error(f"Redis ZADD error: {e}")
            return 0

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """Remove from sorted set by score range."""
        redis = await self._get_redis()
        try:
            return await redis.zremrangebyscore(key, min_score, max_score)
        except Exception as e:
            self.logger.error(f"Redis ZREMRANGEBYSCORE error: {e}")
            return 0

    async def zcard(self, key: str) -> int:
        """Get sorted set cardinality."""
        redis = await self._get_redis()
        try:
            return await redis.zcard(key)
        except Exception as e:
            self.logger.error(f"Redis ZCARD error: {e}")
            return 0

    async def eval(self, script: str, keys: List[str], args: List[str]) -> any:
        """Execute Lua script."""
        redis = await self._get_redis()
        try:
            return await redis.eval(script, len(keys), *keys, *args)
        except Exception as e:
            self.logger.error(f"Redis EVAL error: {e}")
            return None


class InMemoryRedis:
    """In-memory Redis fallback for testing/development."""

    def __init__(self):
        self._data: Dict[str, any] = {}
        self._expiry: Dict[str, float] = {}
        self._sorted_sets: Dict[str, Dict[str, float]] = {}

    def _cleanup_expired(self):
        """Remove expired keys."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self._expiry.items()
            if expiry <= current_time
        ]
        for key in expired_keys:
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            self._sorted_sets.pop(key, None)

    async def get(self, key: str) -> Optional[str]:
        self._cleanup_expired()
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._data[key] = value
        if ex:
            self._expiry[key] = time.time() + ex
        return True

    async def incr(self, key: str) -> int:
        self._cleanup_expired()
        current = int(self._data.get(key, 0))
        self._data[key] = str(current + 1)
        return current + 1

    async def expire(self, key: str, seconds: int) -> bool:
        if key in self._data:
            self._expiry[key] = time.time() + seconds
            return True
        return False

    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        if key not in self._sorted_sets:
            self._sorted_sets[key] = {}

        added = 0
        for member, score in mapping.items():
            if member not in self._sorted_sets[key]:
                added += 1
            self._sorted_sets[key][member] = score

        return added

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        if key not in self._sorted_sets:
            return 0

        to_remove = [
            member for member, score in self._sorted_sets[key].items()
            if min_score <= score <= max_score
        ]

        for member in to_remove:
            del self._sorted_sets[key][member]

        return len(to_remove)

    async def zcard(self, key: str) -> int:
        return len(self._sorted_sets.get(key, {}))

    async def eval(self, script: str, keys: List[str], args: List[str]) -> any:
        # Simple implementation for basic scripts
        return None


class RateLimiter:
    """Rate limiter with Redis backend for API protection and abuse prevention."""

    def __init__(self, redis_client: Optional[RedisClient] = None):
        self.redis_client = redis_client or RedisClient()
        self.logger = logging.getLogger(__name__)

        # Default rate limit rules
        self.rules: Dict[str, RateLimitRule] = {}

        # Lua scripts for atomic operations
        self.sliding_window_script = """
            local key = KEYS[1]
            local window = tonumber(ARGV[1])
            local limit = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])

            -- Remove expired entries
            redis.call('ZREMRANGEBYSCORE', key, 0, current_time - window)

            -- Count current requests
            local current_count = redis.call('ZCARD', key)

            if current_count < limit then
                -- Add current request
                redis.call('ZADD', key, current_time, current_time)
                redis.call('EXPIRE', key, window)
                return {1, limit - current_count - 1, current_time + window}
            else
                return {0, 0, current_time + window}
            end
        """

        self.token_bucket_script = """
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local refill_rate = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])
            local requested_tokens = tonumber(ARGV[4])

            local bucket_data = redis.call('HMGET', key, 'tokens', 'last_refill')
            local tokens = tonumber(bucket_data[1]) or capacity
            local last_refill = tonumber(bucket_data[2]) or current_time

            -- Calculate tokens to add based on time elapsed
            local time_elapsed = current_time - last_refill
            local tokens_to_add = math.floor(time_elapsed * refill_rate)
            tokens = math.min(capacity, tokens + tokens_to_add)

            if tokens >= requested_tokens then
                tokens = tokens - requested_tokens
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
                redis.call('EXPIRE', key, 3600)  -- 1 hour expiry
                return {1, tokens, 0}
            else
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
                redis.call('EXPIRE', key, 3600)
                local wait_time = math.ceil((requested_tokens - tokens) / refill_rate)
                return {0, tokens, wait_time}
            end
        """

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limiting rule."""
        self.rules[rule.name] = rule
        self.logger.info(f"Added rate limit rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> None:
        """Remove a rate limiting rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.logger.info(f"Removed rate limit rule: {rule_name}")

    def generate_key(self, rule_name: str, identifier: str, additional_context: Optional[Dict[str, str]] = None) -> str:
        """Generate Redis key for rate limiting."""
        base_key = f"rate_limit:{rule_name}:{identifier}"

        if additional_context:
            # Include additional context in key generation
            context_str = json.dumps(additional_context, sort_keys=True)
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
            base_key += f":{context_hash}"

        return base_key

    async def check_rate_limit(self,
                             rule_name: str,
                             identifier: str,
                             additional_context: Optional[Dict[str, str]] = None) -> RateLimitResult:
        """Check if request is within rate limits."""
        if rule_name not in self.rules:
            # No rule defined, allow request
            return RateLimitResult(
                allowed=True,
                remaining_requests=float('inf'),
                reset_time=0
            )

        rule = self.rules[rule_name]
        key = self.generate_key(rule_name, identifier, additional_context)
        current_time = time.time()

        try:
            if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._check_fixed_window(key, rule, current_time)
            elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(key, rule, current_time)
            elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(key, rule, current_time)
            elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return await self._check_leaky_bucket(key, rule, current_time)
            else:
                self.logger.error(f"Unknown rate limit strategy: {rule.strategy}")
                return RateLimitResult(allowed=True, remaining_requests=0, reset_time=0)

        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiting fails
            return RateLimitResult(allowed=True, remaining_requests=0, reset_time=0)

    async def _check_fixed_window(self, key: str, rule: RateLimitRule, current_time: float) -> RateLimitResult:
        """Check rate limit using fixed window strategy."""
        window_start = int(current_time // rule.window_size_seconds) * rule.window_size_seconds
        window_key = f"{key}:{window_start}"

        current_count = await self.redis_client.incr(window_key)

        if current_count == 1:
            # First request in window, set expiration
            await self.redis_client.expire(window_key, rule.window_size_seconds)

        if current_count <= rule.requests_per_window:
            return RateLimitResult(
                allowed=True,
                remaining_requests=rule.requests_per_window - current_count,
                reset_time=window_start + rule.window_size_seconds
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining_requests=0,
                reset_time=window_start + rule.window_size_seconds,
                retry_after=int(window_start + rule.window_size_seconds - current_time)
            )

    async def _check_sliding_window(self, key: str, rule: RateLimitRule, current_time: float) -> RateLimitResult:
        """Check rate limit using sliding window strategy."""
        result = await self.redis_client.eval(
            self.sliding_window_script,
            [key],
            [str(rule.window_size_seconds), str(rule.requests_per_window), str(current_time)]
        )

        if result:
            allowed, remaining, reset_time = result
            return RateLimitResult(
                allowed=bool(allowed),
                remaining_requests=remaining,
                reset_time=reset_time,
                retry_after=int(reset_time - current_time) if not allowed else None
            )
        else:
            # Fallback to fixed window if Lua script fails
            return await self._check_fixed_window(key, rule, current_time)

    async def _check_token_bucket(self, key: str, rule: RateLimitRule, current_time: float) -> RateLimitResult:
        """Check rate limit using token bucket strategy."""
        capacity = rule.requests_per_window
        refill_rate = capacity / rule.window_size_seconds  # tokens per second

        result = await self.redis_client.eval(
            self.token_bucket_script,
            [key],
            [str(capacity), str(refill_rate), str(current_time), "1"]
        )

        if result:
            allowed, remaining_tokens, wait_time = result
            return RateLimitResult(
                allowed=bool(allowed),
                remaining_requests=remaining_tokens,
                reset_time=current_time + wait_time,
                retry_after=wait_time if not allowed else None
            )
        else:
            # Fallback
            return await self._check_fixed_window(key, rule, current_time)

    async def _check_leaky_bucket(self, key: str, rule: RateLimitRule, current_time: float) -> RateLimitResult:
        """Check rate limit using leaky bucket strategy."""
        # Simplified leaky bucket implementation
        leak_rate = rule.requests_per_window / rule.window_size_seconds
        bucket_size = rule.requests_per_window

        bucket_data = await self.redis_client.get(f"{key}:bucket")
        if bucket_data:
            data = json.loads(bucket_data)
            last_leak = data['last_leak']
            current_level = data['level']
        else:
            last_leak = current_time
            current_level = 0

        # Calculate leakage
        time_elapsed = current_time - last_leak
        leaked = time_elapsed * leak_rate
        current_level = max(0, current_level - leaked)

        if current_level < bucket_size:
            # Add request to bucket
            current_level += 1

            # Store updated bucket state
            bucket_data = {
                'level': current_level,
                'last_leak': current_time
            }
            await self.redis_client.set(
                f"{key}:bucket",
                json.dumps(bucket_data),
                ex=rule.window_size_seconds * 2
            )

            return RateLimitResult(
                allowed=True,
                remaining_requests=int(bucket_size - current_level),
                reset_time=current_time + (current_level / leak_rate)
            )
        else:
            # Bucket is full
            wait_time = (current_level - bucket_size + 1) / leak_rate
            return RateLimitResult(
                allowed=False,
                remaining_requests=0,
                reset_time=current_time + wait_time,
                retry_after=int(wait_time)
            )

    async def reset_rate_limit(self, rule_name: str, identifier: str) -> bool:
        """Reset rate limit for a specific identifier."""
        if rule_name not in self.rules:
            return False

        key = self.generate_key(rule_name, identifier)

        try:
            # Delete all keys with this prefix
            # This is a simplified implementation
            await self.redis_client.set(key, "0", ex=1)  # Set to expire quickly
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset rate limit: {e}")
            return False

    async def get_rate_limit_status(self, rule_name: str, identifier: str) -> Optional[Dict[str, any]]:
        """Get current rate limit status without incrementing."""
        if rule_name not in self.rules:
            return None

        rule = self.rules[rule_name]
        key = self.generate_key(rule_name, identifier)
        current_time = time.time()

        try:
            if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                window_start = int(current_time // rule.window_size_seconds) * rule.window_size_seconds
                window_key = f"{key}:{window_start}"
                current_count_str = await self.redis_client.get(window_key)
                current_count = int(current_count_str) if current_count_str else 0

                return {
                    'rule_name': rule_name,
                    'strategy': rule.strategy.value,
                    'current_count': current_count,
                    'limit': rule.requests_per_window,
                    'remaining': max(0, rule.requests_per_window - current_count),
                    'reset_time': window_start + rule.window_size_seconds,
                    'window_size': rule.window_size_seconds
                }

            elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                # Remove expired entries and count
                await self.redis_client.zremrangebyscore(
                    key, 0, current_time - rule.window_size_seconds
                )
                current_count = await self.redis_client.zcard(key)

                return {
                    'rule_name': rule_name,
                    'strategy': rule.strategy.value,
                    'current_count': current_count,
                    'limit': rule.requests_per_window,
                    'remaining': max(0, rule.requests_per_window - current_count),
                    'window_size': rule.window_size_seconds
                }

            # Add other strategies as needed

        except Exception as e:
            self.logger.error(f"Failed to get rate limit status: {e}")

        return None

    def create_default_rules(self) -> None:
        """Create default rate limiting rules."""
        # API endpoint protection
        self.add_rule(RateLimitRule(
            name="api_general",
            requests_per_window=100,
            window_size_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        ))

        # Authentication attempts
        self.add_rule(RateLimitRule(
            name="auth_attempts",
            requests_per_window=5,
            window_size_seconds=300,  # 5 minutes
            strategy=RateLimitStrategy.FIXED_WINDOW
        ))

        # Forecasting submissions
        self.add_rule(RateLimitRule(
            name="forecast_submission",
            requests_per_window=10,
            window_size_seconds=60,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        ))

        # Research queries
        self.add_rule(RateLimitRule(
            name="research_query",
            requests_per_window=50,
            window_size_seconds=60,
            strategy=RateLimitStrategy.LEAKY_BUCKET
        ))

        # Heavy operations
        self.add_rule(RateLimitRule(
            name="heavy_operation",
            requests_per_window=5,
            window_size_seconds=300,
            strategy=RateLimitStrategy.FIXED_WINDOW
        ))

    async def middleware_check(self, rule_name: str, identifier: str) -> None:
        """Middleware function that raises exception if rate limit exceeded."""
        result = await self.check_rate_limit(rule_name, identifier)

        if not result.allowed:
            raise RateLimitExceededError(
                f"Rate limit exceeded for {rule_name}",
                error_code="RATE_LIMIT_EXCEEDED",
                context={
                    "rule_name": rule_name,
                    "identifier": identifier,
                    "retry_after": result.retry_after,
                    "reset_time": result.reset_time
                }
            )
