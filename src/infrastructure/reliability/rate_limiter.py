"""Advanced rate limiting for API management."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10.0
    burst_size: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    window_size: float = 60.0  # For sliding window
    backoff_factor: float = 2.0
    max_backoff: float = 300.0  # 5 minutes
    enabled: bool = True


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter implementation.

    Allows burst traffic up to bucket capacity while maintaining
    average rate over time.
    """

    def __init__(self, name: str, config: RateLimitConfig):
        self.name = name
        self.config = config
        self.tokens = float(config.burst_size)
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        self.logger = logger.bind(rate_limiter=name)

        # Metrics
        self.total_requests = 0
        self.allowed_requests = 0
        self.rejected_requests = 0
        self.current_backoff = 0.0

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limited
        """
        if not self.config.enabled:
            return True

        async with self.lock:
            self.total_requests += 1

            # Refill tokens based on elapsed time
            current_time = time.time()
            elapsed = current_time - self.last_refill
            tokens_to_add = elapsed * self.config.requests_per_second

            self.tokens = min(self.config.burst_size, self.tokens + tokens_to_add)
            self.last_refill = current_time

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.allowed_requests += 1
                self.current_backoff = 0.0  # Reset backoff on success

                self.logger.debug(
                    "Tokens acquired",
                    tokens_requested=tokens,
                    tokens_remaining=self.tokens,
                )
                return True
            else:
                self.rejected_requests += 1

                # Calculate backoff time
                self.current_backoff = min(
                    self.config.max_backoff,
                    max(1.0, self.current_backoff * self.config.backoff_factor),
                )

                self.logger.warning(
                    "Rate limit exceeded",
                    tokens_requested=tokens,
                    tokens_available=self.tokens,
                    backoff_time=self.current_backoff,
                )
                return False

    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens to wait for
        """
        while not await self.acquire(tokens):
            await asyncio.sleep(self.current_backoff)

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        return {
            "name": self.name,
            "strategy": self.config.strategy.value,
            "requests_per_second": self.config.requests_per_second,
            "burst_size": self.config.burst_size,
            "current_tokens": self.tokens,
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": self.allowed_requests / max(1, self.total_requests),
            "current_backoff": self.current_backoff,
            "enabled": self.config.enabled,
        }

    def reset(self):
        """Reset rate limiter state."""
        self.tokens = float(self.config.burst_size)
        self.last_refill = time.time()
        self.current_backoff = 0.0
        self.logger.info("Rate limiter reset")


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter implementation.

    Maintains a sliding window of requests to enforce
    rate limits more precisely than fixed windows.
    """

    def __init__(self, name: str, config: RateLimitConfig):
        self.name = name
        self.config = config
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
        self.logger = logger.bind(rate_limiter=name)

        # Metrics
        self.total_requests = 0
        self.allowed_requests = 0
        self.rejected_requests = 0

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire permission for requests.

        Args:
            tokens: Number of requests to acquire

        Returns:
            True if requests allowed, False if rate limited
        """
        if not self.config.enabled:
            return True

        async with self.lock:
            self.total_requests += 1
            current_time = time.time()

            # Remove old requests outside the window
            cutoff_time = current_time - self.config.window_size
            self.requests = [
                req_time for req_time in self.requests if req_time > cutoff_time
            ]

            # Check if we can add new requests
            max_requests = int(
                self.config.requests_per_second * self.config.window_size
            )

            if len(self.requests) + tokens <= max_requests:
                # Add new request timestamps
                for _ in range(tokens):
                    self.requests.append(current_time)

                self.allowed_requests += 1

                self.logger.debug(
                    "Requests allowed",
                    tokens_requested=tokens,
                    current_window_requests=len(self.requests),
                    max_requests=max_requests,
                )
                return True
            else:
                self.rejected_requests += 1

                self.logger.warning(
                    "Rate limit exceeded",
                    tokens_requested=tokens,
                    current_window_requests=len(self.requests),
                    max_requests=max_requests,
                )
                return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        current_time = time.time()
        cutoff_time = current_time - self.config.window_size
        current_window_requests = len(
            [req for req in self.requests if req > cutoff_time]
        )

        return {
            "name": self.name,
            "strategy": self.config.strategy.value,
            "requests_per_second": self.config.requests_per_second,
            "window_size": self.config.window_size,
            "current_window_requests": current_window_requests,
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": self.allowed_requests / max(1, self.total_requests),
            "enabled": self.config.enabled,
        }


class RateLimiterManager:
    """
    Manager for multiple rate limiters.

    Provides centralized management of rate limiters for different
    APIs and services with hierarchical rate limiting support.
    """

    def __init__(self):
        self.rate_limiters: Dict[str, Any] = {}
        self.logger = logger.bind(component="rate_limiter_manager")

    def create_rate_limiter(self, name: str, config: RateLimitConfig) -> Any:
        """
        Create a rate limiter with specified configuration.

        Args:
            name: Rate limiter name
            config: Rate limiting configuration

        Returns:
            Rate limiter instance
        """
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            limiter = TokenBucketRateLimiter(name, config)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            limiter = SlidingWindowRateLimiter(name, config)
        else:
            # Default to token bucket
            limiter = TokenBucketRateLimiter(name, config)

        self.rate_limiters[name] = limiter
        self.logger.info(
            "Created rate limiter",
            name=name,
            strategy=config.strategy.value,
            requests_per_second=config.requests_per_second,
        )

        return limiter

    def get_rate_limiter(self, name: str) -> Optional[Any]:
        """
        Get rate limiter by name.

        Args:
            name: Rate limiter name

        Returns:
            Rate limiter instance or None
        """
        return self.rate_limiters.get(name)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all rate limiters."""
        return {
            name: limiter.get_metrics() for name, limiter in self.rate_limiters.items()
        }

    def reset_all(self):
        """Reset all rate limiters."""
        for limiter in self.rate_limiters.values():
            if hasattr(limiter, "reset"):
                limiter.reset()
        self.logger.info("Reset all rate limiters")

    def enable_all(self):
        """Enable all rate limiters."""
        for limiter in self.rate_limiters.values():
            limiter.config.enabled = True
        self.logger.info("Enabled all rate limiters")

    def disable_all(self):
        """Disable all rate limiters."""
        for limiter in self.rate_limiters.values():
            limiter.config.enabled = False
        self.logger.info("Disabled all rate limiters")


class RateLimitedClient:
    """
    Rate-limited client wrapper for API calls.

    Wraps API clients with rate limiting and automatic backoff
    to prevent API quota exhaustion and handle rate limit responses.
    """

    def __init__(
        self, name: str, client: Any, rate_limiter: Any, backoff_on_429: bool = True
    ):
        self.name = name
        self.client = client
        self.rate_limiter = rate_limiter
        self.backoff_on_429 = backoff_on_429
        self.logger = logger.bind(rate_limited_client=name)

        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.rate_limited_calls = 0
        self.failed_calls = 0

    async def call(self, method: str, *args, tokens: int = 1, **kwargs) -> Any:
        """
        Make rate-limited API call.

        Args:
            method: Method name to call on client
            *args: Method arguments
            tokens: Number of rate limit tokens to consume
            **kwargs: Method keyword arguments

        Returns:
            Method result
        """
        self.total_calls += 1

        # Wait for rate limit tokens
        await self.rate_limiter.wait_for_tokens(tokens)

        try:
            # Get method from client
            client_method = getattr(self.client, method)

            # Execute method
            if asyncio.iscoroutinefunction(client_method):
                result = await client_method(*args, **kwargs)
            else:
                # Run sync method in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: client_method(*args, **kwargs)
                )

            self.successful_calls += 1
            return result

        except Exception as e:
            # Check if it's a rate limit error (HTTP 429)
            if self.backoff_on_429 and self._is_rate_limit_error(e):
                self.rate_limited_calls += 1

                # Extract retry-after header if available
                retry_after = self._extract_retry_after(e)
                if retry_after:
                    self.logger.warning(
                        "API rate limit hit, backing off", retry_after=retry_after
                    )
                    await asyncio.sleep(retry_after)
                else:
                    # Use exponential backoff
                    backoff_time = min(60.0, 2.0 ** (self.rate_limited_calls % 6))
                    self.logger.warning(
                        "API rate limit hit, backing off", backoff_time=backoff_time
                    )
                    await asyncio.sleep(backoff_time)

                # Retry the call
                return await self.call(method, *args, tokens=tokens, **kwargs)
            else:
                self.failed_calls += 1
                self.logger.error("API call failed", method=method, error=str(e))
                raise

    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """Check if exception indicates rate limiting."""
        # This should be customized based on your HTTP client
        error_str = str(exception).lower()
        return (
            "429" in error_str
            or "rate limit" in error_str
            or "too many requests" in error_str
            or "quota exceeded" in error_str
        )

    def _extract_retry_after(self, exception: Exception) -> Optional[float]:
        """Extract retry-after value from exception."""
        # This should be customized based on your HTTP client
        # For now, return None to use exponential backoff
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "rate_limited_calls": self.rate_limited_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / max(1, self.total_calls),
            "rate_limit_rate": self.rate_limited_calls / max(1, self.total_calls),
        }


# Global rate limiter manager
rate_limiter_manager = RateLimiterManager()


# Convenience functions for common rate limiting scenarios
def create_openai_rate_limiter(name: str = "openai") -> TokenBucketRateLimiter:
    """Create rate limiter for OpenAI API."""
    config = RateLimitConfig(
        requests_per_second=3.0,  # Conservative for GPT-4
        burst_size=10,
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        backoff_factor=2.0,
        max_backoff=300.0,
    )
    return rate_limiter_manager.create_rate_limiter(name, config)


def create_anthropic_rate_limiter(name: str = "anthropic") -> TokenBucketRateLimiter:
    """Create rate limiter for Anthropic API."""
    config = RateLimitConfig(
        requests_per_second=5.0,
        burst_size=15,
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        backoff_factor=2.0,
        max_backoff=300.0,
    )
    return rate_limiter_manager.create_rate_limiter(name, config)


def create_metaculus_rate_limiter(name: str = "metaculus") -> TokenBucketRateLimiter:
    """Create rate limiter for Metaculus API."""
    config = RateLimitConfig(
        requests_per_second=2.0,  # Conservative for Metaculus
        burst_size=5,
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        backoff_factor=1.5,
        max_backoff=600.0,  # Longer backoff for tournament API
    )
    return rate_limiter_manager.create_rate_limiter(name, config)


def create_search_rate_limiter(name: str = "search") -> TokenBucketRateLimiter:
    """Create rate limiter for search APIs."""
    config = RateLimitConfig(
        requests_per_second=10.0,
        burst_size=20,
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        backoff_factor=2.0,
        max_backoff=120.0,
    )
    return rate_limiter_manager.create_rate_limiter(name, config)


def create_tournament_rate_limiter(
    name: str = "tournament",
) -> SlidingWindowRateLimiter:
    """Create rate limiter optimized for tournament conditions."""
    config = RateLimitConfig(
        requests_per_second=1.0,  # Very conservative for tournament
        window_size=300.0,  # 5-minute window
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        backoff_factor=1.2,  # Gentle backoff
        max_backoff=900.0,  # 15 minutes max
    )
    return rate_limiter_manager.create_rate_limiter(name, config)
