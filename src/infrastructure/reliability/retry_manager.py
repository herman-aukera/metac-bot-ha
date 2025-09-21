"""Intelligent retry logic with exponential backoff and jitter."""

import asyncio
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Type

import structlog

logger = structlog.get_logger(__name__)


class RetryStrategy(Enum):
    """Retry strategies."""

    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    jitter_range: float = 0.1  # Jitter as fraction of delay
    retryable_exceptions: List[Type[Exception]] = None
    non_retryable_exceptions: List[Type[Exception]] = None

    def __post_init__(self):
        if self.retryable_exceptions is None:
            self.retryable_exceptions = [Exception]
        if self.non_retryable_exceptions is None:
            self.non_retryable_exceptions = []


class RetryManager:
    """
    Intelligent retry manager with various backoff strategies.

    Provides sophisticated retry logic with exponential backoff,
    jitter, and configurable exception handling for tournament-grade reliability.
    """

    def __init__(self, name: str, policy: Optional[RetryPolicy] = None):
        self.name = name
        self.policy = policy or RetryPolicy()
        self.logger = logger.bind(retry_manager=name)

        # Metrics
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.retry_attempts = 0

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(1, self.policy.max_attempts + 1):
            self.total_attempts += 1

            try:
                self.logger.debug(
                    "Executing function",
                    attempt=attempt,
                    max_attempts=self.policy.max_attempts,
                )

                result = await self._execute_function(func, *args, **kwargs)

                if attempt > 1:
                    self.logger.info(
                        "Function succeeded after retries",
                        attempt=attempt,
                        total_attempts=self.total_attempts,
                    )

                self.successful_attempts += 1
                return result

            except Exception as e:
                last_exception = e
                self.failed_attempts += 1

                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    self.logger.error(
                        "Non-retryable exception, not retrying",
                        error=str(e),
                        exception_type=type(e).__name__,
                    )
                    raise

                # Don't retry on last attempt
                if attempt == self.policy.max_attempts:
                    self.logger.error(
                        "All retry attempts exhausted", attempts=attempt, error=str(e)
                    )
                    break

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                self.retry_attempts += 1

                self.logger.warning(
                    "Function failed, retrying",
                    attempt=attempt,
                    delay=delay,
                    error=str(e),
                    exception_type=type(e).__name__,
                )

                await asyncio.sleep(delay)

        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Unexpected retry loop exit")

    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        # Check non-retryable exceptions first
        for exc_type in self.policy.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False

        # Check retryable exceptions
        for exc_type in self.policy.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True

        return False

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.policy.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.policy.base_delay

        elif self.policy.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.policy.base_delay * (
                self.policy.backoff_multiplier ** (attempt - 1)
            )

        elif self.policy.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.policy.base_delay * attempt

        elif self.policy.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.policy.base_delay * self._fibonacci(attempt)

        else:
            delay = self.policy.base_delay

        # Apply maximum delay limit
        delay = min(delay, self.policy.max_delay)

        # Add jitter if enabled
        if self.policy.jitter:
            jitter_amount = delay * self.policy.jitter_range
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)

        return delay

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n

        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b

        return b

    def get_metrics(self) -> dict:
        """Get retry manager metrics."""
        return {
            "name": self.name,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "retry_attempts": self.retry_attempts,
            "success_rate": (self.successful_attempts / max(1, self.total_attempts)),
            "retry_rate": (self.retry_attempts / max(1, self.total_attempts)),
            "policy": {
                "max_attempts": self.policy.max_attempts,
                "base_delay": self.policy.base_delay,
                "max_delay": self.policy.max_delay,
                "strategy": self.policy.strategy.value,
                "backoff_multiplier": self.policy.backoff_multiplier,
                "jitter": self.policy.jitter,
            },
        }

    def reset_metrics(self):
        """Reset retry metrics."""
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.retry_attempts = 0
        self.logger.info("Retry metrics reset")


class RetryableOperation:
    """
    Decorator for making operations retryable.

    Provides a convenient way to add retry logic to functions
    without modifying their implementation.
    """

    def __init__(
        self, policy: Optional[RetryPolicy] = None, name: Optional[str] = None
    ):
        self.policy = policy or RetryPolicy()
        self.name = name

    def __call__(self, func: Callable) -> Callable:
        """Decorate function with retry logic."""
        retry_manager = RetryManager(
            name=self.name or f"{func.__module__}.{func.__name__}", policy=self.policy
        )

        async def wrapper(*args, **kwargs):
            return await retry_manager.execute(func, *args, **kwargs)

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        wrapper._retry_manager = retry_manager

        return wrapper


class RetryManagerRegistry:
    """
    Registry for managing multiple retry managers.

    Provides centralized management and monitoring of retry managers
    across different operations and services.
    """

    def __init__(self):
        self.retry_managers: dict[str, RetryManager] = {}
        self.logger = logger.bind(component="retry_manager_registry")

    def get_retry_manager(
        self, name: str, policy: Optional[RetryPolicy] = None
    ) -> RetryManager:
        """
        Get or create a retry manager.

        Args:
            name: Retry manager name
            policy: Retry policy (uses default if not provided)

        Returns:
            RetryManager instance
        """
        if name not in self.retry_managers:
            self.retry_managers[name] = RetryManager(name, policy)
            self.logger.info("Created retry manager", name=name)

        return self.retry_managers[name]

    def get_all_metrics(self) -> dict[str, dict]:
        """Get metrics for all retry managers."""
        return {name: rm.get_metrics() for name, rm in self.retry_managers.items()}

    def reset_all_metrics(self):
        """Reset metrics for all retry managers."""
        for rm in self.retry_managers.values():
            rm.reset_metrics()
        self.logger.info("Reset all retry manager metrics")

    def get_total_metrics(self) -> dict:
        """Get aggregated metrics across all retry managers."""
        total_attempts = sum(rm.total_attempts for rm in self.retry_managers.values())
        successful_attempts = sum(
            rm.successful_attempts for rm in self.retry_managers.values()
        )
        failed_attempts = sum(rm.failed_attempts for rm in self.retry_managers.values())
        retry_attempts = sum(rm.retry_attempts for rm in self.retry_managers.values())

        return {
            "total_retry_managers": len(self.retry_managers),
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "failed_attempts": failed_attempts,
            "retry_attempts": retry_attempts,
            "overall_success_rate": successful_attempts / max(1, total_attempts),
            "overall_retry_rate": retry_attempts / max(1, total_attempts),
        }


# Global retry manager registry
retry_manager_registry = RetryManagerRegistry()


# Convenience functions for common retry policies
def create_api_retry_policy() -> RetryPolicy:
    """Create retry policy optimized for API calls."""
    return RetryPolicy(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_multiplier=2.0,
        jitter=True,
        retryable_exceptions=[
            ConnectionError,
            TimeoutError,
            OSError,
            # Add HTTP-specific exceptions as needed
        ],
        non_retryable_exceptions=[
            ValueError,
            TypeError,
            KeyError,
            # Add authentication/authorization errors as needed
        ],
    )


def create_database_retry_policy() -> RetryPolicy:
    """Create retry policy optimized for database operations."""
    return RetryPolicy(
        max_attempts=3,
        base_delay=0.5,
        max_delay=10.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_multiplier=2.0,
        jitter=True,
        retryable_exceptions=[
            ConnectionError,
            TimeoutError,
            # Add database-specific exceptions as needed
        ],
    )


def create_tournament_retry_policy() -> RetryPolicy:
    """Create retry policy optimized for tournament operations."""
    return RetryPolicy(
        max_attempts=7,  # More attempts for critical tournament operations
        base_delay=2.0,
        max_delay=60.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_multiplier=1.5,  # Gentler backoff for tournament conditions
        jitter=True,
        jitter_range=0.2,  # More jitter to avoid thundering herd
        retryable_exceptions=[Exception],  # Retry most exceptions in tournament
        non_retryable_exceptions=[
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ],
    )
