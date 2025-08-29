"""Circuit breaker implementation for fault tolerance."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout
    expected_exception: type = Exception  # Exception type to count as failure


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Implements the circuit breaker pattern to prevent repeated calls
    to a failing service, allowing it time to recover.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.lock = asyncio.Lock()
        self.logger = logger.bind(circuit_breaker=name)

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rejected_requests = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: When circuit is open
            Original exception: When function fails
        """
        async with self.lock:
            self.total_requests += 1

            # Check if circuit should transition states
            await self._check_state_transition()

            # Reject request if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                self.rejected_requests += 1
                self.logger.warning("Circuit breaker open, rejecting request")
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")

        # Execute the function
        try:
            # Apply timeout
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self.config.timeout,
            )

            # Record success
            await self._record_success()
            return result

        except asyncio.TimeoutError as e:
            await self._record_failure()
            self.logger.error("Function call timed out", timeout=self.config.timeout)
            raise

        except self.config.expected_exception as e:
            await self._record_failure()
            self.logger.error("Function call failed", error=str(e))
            raise

        except Exception as e:
            # Unexpected exceptions don't count as circuit breaker failures
            self.logger.error("Unexpected error in circuit breaker", error=str(e))
            raise

    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _check_state_transition(self):
        """Check if circuit breaker should change state."""
        current_time = time.time()

        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to half-open")

        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Check if enough successes to close
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info("Circuit breaker closed after recovery")

    async def _record_success(self):
        """Record a successful operation."""
        async with self.lock:
            self.successful_requests += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                self.logger.debug(
                    "Success recorded in half-open state",
                    success_count=self.success_count,
                )

                # Check if enough successes to close
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info("Circuit breaker closed after recovery")

            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    async def _record_failure(self):
        """Record a failed operation."""
        async with self.lock:
            self.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = time.time()

            self.logger.warning(
                "Failure recorded",
                failure_count=self.failure_count,
                state=self.state.value,
            )

            # Check if should open circuit
            if (
                self.state == CircuitBreakerState.CLOSED
                and self.failure_count >= self.config.failure_threshold
            ):

                self.state = CircuitBreakerState.OPEN
                self.logger.error(
                    "Circuit breaker opened due to failures",
                    failure_count=self.failure_count,
                )

            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                self.logger.error("Circuit breaker reopened after half-open failure")

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_rate": (
                self.failed_requests
                / max(1, self.total_requests - self.rejected_requests)
            ),
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }

    async def reset(self):
        """Reset circuit breaker to closed state."""
        async with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0.0
            self.logger.info("Circuit breaker manually reset")

    async def force_open(self):
        """Force circuit breaker to open state."""
        async with self.lock:
            self.state = CircuitBreakerState.OPEN
            self.last_failure_time = time.time()
            self.logger.warning("Circuit breaker manually opened")

    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == CircuitBreakerState.CLOSED

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitBreakerState.OPEN

    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == CircuitBreakerState.HALF_OPEN


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.

    Provides centralized management and monitoring of circuit breakers
    across different services and operations.
    """

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logger.bind(component="circuit_breaker_manager")

    def get_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration (uses default if not provided)

        Returns:
            CircuitBreaker instance
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
            self.logger.info("Created circuit breaker", name=name)

        return self.circuit_breakers[name]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: cb.get_metrics() for name, cb in self.circuit_breakers.items()}

    async def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            await cb.reset()
        self.logger.info("Reset all circuit breakers")

    def get_unhealthy_circuits(self) -> List[str]:
        """Get list of circuit breakers that are not closed."""
        return [
            name for name, cb in self.circuit_breakers.items() if not cb.is_closed()
        ]

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all circuit breakers."""
        return {name: cb.is_closed() for name, cb in self.circuit_breakers.items()}


# Global circuit breaker manager instance
circuit_breaker_manager = CircuitBreakerManager()
