"""
Configurable retry strategy with exponential backoff and jitter.

Provides intelligent retry mechanisms for transient failures with
configurable backoff strategies and failure classification.
"""

import asyncio
import random
import time
from enum import Enum
from typing import Callable, Any, Optional, List, Type, Union
from dataclasses import dataclass
import logging

from src.domain.exceptions import InfrastructureError, TimeoutError


logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry attempts."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    """Maximum number of retry attempts"""

    base_delay: float = 1.0
    """Base delay between retries in seconds"""

    max_delay: float = 60.0
    """Maximum delay between retries in seconds"""

    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    """Backoff strategy to use"""

    jitter_factor: float = 0.1
    """Jitter factor for randomization (0.0 to 1.0)"""

    multiplier: float = 2.0
    """Multiplier for exponential backoff"""

    retryable_exceptions: tuple = (
        InfrastructureError,
        TimeoutError,
        ConnectionError,
        OSError,
    )
    """Exception types that should trigger retries"""

    non_retryable_exceptions: tuple = (
        ValueError,
        TypeError,
        KeyError,
    )
    """Exception types that should not trigger retries"""

    timeout: Optional[float] = None
    """Overall timeout for all retry attempts"""


class RetryExhaustedException(InfrastructureError):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Exception,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.attempts = attempts
        self.last_exception = last_exception
        self.context.metadata.update({
            "attempts": attempts,
            "last_exception": str(last_exception),
            "last_exception_type": type(last_exception).__name__,
        })


class RetryStrategy:
    """
    Configurable retry strategy with exponential backoff and jitter.

    Provides intelligent retry mechanisms for transient failures with
    support for different backoff strategies and failure classification.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

        logger.debug(
            f"Retry strategy initialized with config: {self.config}",
            extra={"retry_config": self.config}
        )

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            RetryExhaustedException: When all retry attempts are exhausted
            Exception: Non-retryable exceptions are re-raised immediately
        """
        start_time = time.time()
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                # Check overall timeout
                if self.config.timeout:
                    elapsed = time.time() - start_time
                    if elapsed >= self.config.timeout:
                        raise TimeoutError(
                            f"Retry timeout exceeded: {elapsed:.2f}s >= {self.config.timeout}s",
                            timeout_duration=self.config.timeout,
                            operation_type="retry_execution"
                        )

                logger.debug(
                    f"Executing function attempt {attempt + 1}/{self.config.max_attempts}",
                    extra={
                        "attempt": attempt + 1,
                        "max_attempts": self.config.max_attempts,
                        "function": func.__name__ if hasattr(func, '__name__') else str(func),
                    }
                )

                result = await func(*args, **kwargs)

                if attempt > 0:
                    logger.info(
                        f"Function succeeded after {attempt + 1} attempts",
                        extra={
                            "attempts": attempt + 1,
                            "function": func.__name__ if hasattr(func, '__name__') else str(func),
                        }
                    )

                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.debug(
                        f"Non-retryable exception encountered: {type(e).__name__}",
                        extra={
                            "exception": str(e),
                            "exception_type": type(e).__name__,
                        }
                    )
                    raise

                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break

                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)

                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                    extra={
                        "attempt": attempt + 1,
                        "max_attempts": self.config.max_attempts,
                        "delay": delay,
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                    }
                )

                await asyncio.sleep(delay)

        # All attempts exhausted
        raise RetryExhaustedException(
            f"All {self.config.max_attempts} retry attempts exhausted",
            attempts=self.config.max_attempts,
            last_exception=last_exception,
            context={
                "function": func.__name__ if hasattr(func, '__name__') else str(func),
                "config": self.config,
            }
        )

    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        # Check non-retryable exceptions first
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False

        # Check retryable exceptions
        if isinstance(exception, self.config.retryable_exceptions):
            return True

        # Check if it's a recoverable infrastructure error
        if isinstance(exception, InfrastructureError):
            return exception.recoverable

        # Default to non-retryable for unknown exceptions
        return False

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt based on backoff strategy."""
        if self.config.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.config.base_delay

        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)

        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.multiplier ** attempt)

        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            base_delay = self.config.base_delay * (self.config.multiplier ** attempt)
            jitter = base_delay * self.config.jitter_factor * random.random()
            delay = base_delay + jitter

        else:
            delay = self.config.base_delay

        # Apply maximum delay limit
        return min(delay, self.config.max_delay)


class RetryableOperation:
    """
    Decorator and context manager for retryable operations.

    Provides convenient ways to apply retry logic to functions
    and code blocks.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.retry_strategy = RetryStrategy(config)

    def __call__(self, func: Callable) -> Callable:
        """Decorator for retryable functions."""
        async def wrapper(*args, **kwargs):
            return await self.retry_strategy.execute(func, *args, **kwargs)

        wrapper.__name__ = f"retryable_{func.__name__}"
        wrapper.__doc__ = f"Retryable version of {func.__name__}"
        return wrapper

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        return await self.retry_strategy.execute(func, *args, **kwargs)


class AdaptiveRetryStrategy(RetryStrategy):
    """
    Adaptive retry strategy that adjusts behavior based on success/failure patterns.

    Learns from previous attempts to optimize retry parameters
    for different types of operations and failures.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        super().__init__(config)
        self.success_history: List[int] = []  # Track attempts needed for success
        self.failure_patterns: dict = {}  # Track failure patterns
        self.adaptation_enabled = True

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with adaptive retry logic."""
        # Use base implementation but track results
        start_time = time.time()
        attempts = 0

        try:
            result = await super().execute(func, *args, **kwargs)
            attempts = self._estimate_attempts_used(start_time)
            self._record_success(attempts)
            return result

        except Exception as e:
            attempts = self.config.max_attempts
            self._record_failure(type(e).__name__, attempts)
            raise

    def _estimate_attempts_used(self, start_time: float) -> int:
        """Estimate number of attempts based on execution time."""
        elapsed = time.time() - start_time

        # Simple heuristic: if execution was quick, likely succeeded on first try
        if elapsed < self.config.base_delay:
            return 1

        # Otherwise estimate based on expected delay patterns
        total_delay = 0
        for attempt in range(self.config.max_attempts):
            total_delay += self._calculate_delay(attempt)
            if total_delay >= elapsed:
                return attempt + 1

        return self.config.max_attempts

    def _record_success(self, attempts: int):
        """Record successful execution for adaptation."""
        self.success_history.append(attempts)

        # Keep only recent history
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-50:]

        # Adapt configuration if enabled
        if self.adaptation_enabled:
            self._adapt_configuration()

    def _record_failure(self, exception_type: str, attempts: int):
        """Record failure for pattern analysis."""
        if exception_type not in self.failure_patterns:
            self.failure_patterns[exception_type] = []

        self.failure_patterns[exception_type].append(attempts)

        # Keep only recent history
        if len(self.failure_patterns[exception_type]) > 50:
            self.failure_patterns[exception_type] = self.failure_patterns[exception_type][-25:]

    def _adapt_configuration(self):
        """Adapt retry configuration based on historical data."""
        if not self.success_history:
            return

        # Calculate average attempts needed for success
        avg_attempts = sum(self.success_history) / len(self.success_history)

        # Adjust max_attempts if most operations succeed with fewer attempts
        if avg_attempts < self.config.max_attempts * 0.5:
            # Most operations succeed quickly, can reduce max attempts
            new_max = max(2, int(avg_attempts * 1.5))
            if new_max != self.config.max_attempts:
                logger.info(
                    f"Adapting max_attempts from {self.config.max_attempts} to {new_max}",
                    extra={
                        "old_max_attempts": self.config.max_attempts,
                        "new_max_attempts": new_max,
                        "avg_attempts": avg_attempts,
                    }
                )
                self.config.max_attempts = new_max

        elif avg_attempts > self.config.max_attempts * 0.8:
            # Most operations need many attempts, increase max attempts
            new_max = min(10, int(avg_attempts * 1.2))
            if new_max != self.config.max_attempts:
                logger.info(
                    f"Adapting max_attempts from {self.config.max_attempts} to {new_max}",
                    extra={
                        "old_max_attempts": self.config.max_attempts,
                        "new_max_attempts": new_max,
                        "avg_attempts": avg_attempts,
                    }
                )
                self.config.max_attempts = new_max

    def get_adaptation_metrics(self) -> dict:
        """Get metrics about adaptation behavior."""
        return {
            "success_history_size": len(self.success_history),
            "avg_attempts_for_success": sum(self.success_history) / len(self.success_history) if self.success_history else 0,
            "failure_patterns": {
                exc_type: {
                    "count": len(attempts),
                    "avg_attempts": sum(attempts) / len(attempts) if attempts else 0
                }
                for exc_type, attempts in self.failure_patterns.items()
            },
            "current_config": {
                "max_attempts": self.config.max_attempts,
                "base_delay": self.config.base_delay,
                "max_delay": self.config.max_delay,
            }
        }
