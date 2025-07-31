"""
Circuit breaker implementation for external service protection.

Provides automatic failure detection, circuit opening/closing, and
health monitoring to prevent cascading failures.
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from src.domain.exceptions import InfrastructureError


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    """Number of failures before opening circuit"""

    recovery_timeout: int = 60
    """Seconds to wait before attempting recovery"""

    success_threshold: int = 3
    """Number of successes needed to close circuit from half-open"""

    timeout: float = 30.0
    """Request timeout in seconds"""

    expected_exception_types: tuple = (Exception,)
    """Exception types that count as failures"""

    monitor_window: int = 300
    """Time window for monitoring failures (seconds)"""


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by the circuit breaker."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    failure_history: List[datetime] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class CircuitBreakerOpenError(InfrastructureError):
    """Raised when circuit breaker is open."""

    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            f"Circuit breaker is open for service: {service_name}",
            service_name=service_name,
            **kwargs
        )


class CircuitBreaker:
    """
    Circuit breaker for external service protection.

    Implements the circuit breaker pattern to prevent cascading failures
    by monitoring service health and automatically opening/closing the circuit.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.last_failure_time: Optional[float] = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self._lock = asyncio.Lock()

        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")

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
            Exception: Original exception from function
        """
        async with self._lock:
            # Check if circuit should be opened
            if self.state == CircuitState.OPEN:
                if not self._should_attempt_reset():
                    self.metrics.total_requests += 1
                    raise CircuitBreakerOpenError(
                        service_name=self.name,
                        context={
                            "state": self.state.value,
                            "consecutive_failures": self.consecutive_failures,
                            "last_failure_time": self.last_failure_time,
                        }
                    )
                else:
                    # Transition to half-open for testing
                    self._transition_to_half_open()

        # Execute the function
        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            # Apply timeout if configured
            if self.config.timeout > 0:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await func(*args, **kwargs)

            # Record success
            await self._on_success()
            return result

        except asyncio.TimeoutError as e:
            self.metrics.timeouts += 1
            await self._on_failure(e)
            raise

        except self.config.expected_exception_types as e:
            await self._on_failure(e)
            raise

        except Exception as e:
            # Unexpected exception - log but don't count as circuit failure
            logger.warning(
                f"Unexpected exception in circuit breaker '{self.name}': {e}",
                extra={"circuit_breaker": self.name, "exception": str(e)}
            )
            raise

    async def _on_success(self):
        """Handle successful request."""
        async with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.utcnow()
            self.consecutive_failures = 0
            self.consecutive_successes += 1

            # Close circuit if in half-open state and enough successes
            if (self.state == CircuitState.HALF_OPEN and
                self.consecutive_successes >= self.config.success_threshold):
                self._close_circuit()

            logger.debug(
                f"Circuit breaker '{self.name}' recorded success",
                extra={
                    "circuit_breaker": self.name,
                    "state": self.state.value,
                    "consecutive_successes": self.consecutive_successes,
                }
            )

    async def _on_failure(self, exception: Exception):
        """Handle failed request."""
        async with self._lock:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.utcnow()
            self.last_failure_time = time.time()
            self.consecutive_failures += 1
            self.consecutive_successes = 0

            # Add to failure history for monitoring
            now = datetime.utcnow()
            self.metrics.failure_history.append(now)

            # Clean old failures outside monitoring window
            cutoff = now - timedelta(seconds=self.config.monitor_window)
            self.metrics.failure_history = [
                f for f in self.metrics.failure_history if f > cutoff
            ]

            # Open circuit if failure threshold exceeded
            if (self.state == CircuitState.CLOSED and
                self.consecutive_failures >= self.config.failure_threshold):
                self._open_circuit()
            elif self.state == CircuitState.HALF_OPEN:
                # Return to open state on any failure during half-open
                self._open_circuit()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure",
                extra={
                    "circuit_breaker": self.name,
                    "state": self.state.value,
                    "consecutive_failures": self.consecutive_failures,
                    "exception": str(exception),
                }
            )

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset to half-open."""
        if self.last_failure_time is None:
            return True

        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout

    def _open_circuit(self):
        """Open the circuit."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.metrics.circuit_opens += 1
            logger.warning(
                f"Circuit breaker '{self.name}' opened",
                extra={
                    "circuit_breaker": self.name,
                    "consecutive_failures": self.consecutive_failures,
                    "failure_threshold": self.config.failure_threshold,
                }
            )

    def _close_circuit(self):
        """Close the circuit."""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.metrics.circuit_closes += 1
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            logger.info(
                f"Circuit breaker '{self.name}' closed",
                extra={
                    "circuit_breaker": self.name,
                    "consecutive_successes": self.consecutive_successes,
                }
            )

    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.consecutive_successes = 0
        logger.info(
            f"Circuit breaker '{self.name}' transitioned to half-open",
            extra={"circuit_breaker": self.name}
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "timeouts": self.metrics.timeouts,
            "circuit_opens": self.metrics.circuit_opens,
            "circuit_closes": self.metrics.circuit_closes,
            "failure_rate": self.metrics.failure_rate,
            "success_rate": self.metrics.success_rate,
            "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success_time": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            "recent_failures": len(self.metrics.failure_history),
        }

    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.last_failure_time = None
            self.metrics = CircuitBreakerMetrics()
            logger.info(f"Circuit breaker '{self.name}' reset")


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services.

    Provides centralized management, monitoring, and configuration
    of circuit breakers across the system.
    """

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        async with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, config)
            return self.circuit_breakers[name]

    async def call_with_circuit_breaker(
        self,
        service_name: str,
        func: Callable,
        *args,
        config: Optional[CircuitBreakerConfig] = None,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection."""
        circuit_breaker = await self.get_circuit_breaker(service_name, config)
        return await circuit_breaker.call(func, *args, **kwargs)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {
            name: cb.get_metrics()
            for name, cb in self.circuit_breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers."""
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.reset()
        logger.info("All circuit breakers reset")

    def get_unhealthy_services(self) -> List[str]:
        """Get list of services with open circuit breakers."""
        return [
            name for name, cb in self.circuit_breakers.items()
            if cb.state == CircuitState.OPEN
        ]
