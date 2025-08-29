"""Advanced API management with failover and redundancy."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import structlog

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .rate_limiter import RateLimitConfig, RateLimitedClient
from .retry_manager import RetryManager, RetryPolicy

logger = structlog.get_logger(__name__)


class APIStatus(Enum):
    """API endpoint status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class FailoverStrategy(Enum):
    """Failover strategies."""

    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"


@dataclass
class APIEndpoint:
    """API endpoint configuration."""

    name: str
    client: Any
    priority: int = 1  # Lower number = higher priority
    weight: float = 1.0
    enabled: bool = True
    health_check_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3

    # Runtime state
    status: APIStatus = APIStatus.HEALTHY
    last_health_check: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0


@dataclass
class APIManagerConfig:
    """Configuration for API manager."""

    failover_strategy: FailoverStrategy = FailoverStrategy.PRIORITY
    health_check_interval: float = 60.0
    max_response_time_samples: int = 100
    unhealthy_threshold: int = 5  # Consecutive failures
    recovery_threshold: int = 3  # Consecutive successes
    enable_circuit_breaker: bool = True
    enable_rate_limiting: bool = True
    enable_health_monitoring: bool = True


class APIManager:
    """
    Advanced API management with failover and redundancy.

    Manages multiple API endpoints with automatic failover,
    health monitoring, rate limiting, and circuit breaking
    for tournament-grade reliability.
    """

    def __init__(self, name: str, config: Optional[APIManagerConfig] = None):
        self.name = name
        self.config = config or APIManagerConfig()
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimitedClient] = {}
        self.retry_managers: Dict[str, RetryManager] = {}

        self.current_endpoint_index = 0
        self.running = False
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.logger = logger.bind(api_manager=name)

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.failover_count = 0

    def add_endpoint(
        self,
        endpoint: APIEndpoint,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        """
        Add API endpoint with optional reliability components.

        Args:
            endpoint: APIEndpoint configuration
            circuit_breaker_config: Circuit breaker configuration
            rate_limit_config: Rate limiting configuration
            retry_policy: Retry policy configuration
        """
        self.endpoints[endpoint.name] = endpoint

        # Create circuit breaker if enabled
        if self.config.enable_circuit_breaker:
            cb_config = circuit_breaker_config or CircuitBreakerConfig()
            self.circuit_breakers[endpoint.name] = CircuitBreaker(
                f"{self.name}_{endpoint.name}", cb_config
            )

        # Create rate limiter if enabled
        if self.config.enable_rate_limiting and rate_limit_config:
            from .rate_limiter import rate_limiter_manager

            rate_limiter = rate_limiter_manager.create_rate_limiter(
                f"{self.name}_{endpoint.name}", rate_limit_config
            )
            self.rate_limiters[endpoint.name] = RateLimitedClient(
                f"{self.name}_{endpoint.name}", endpoint.client, rate_limiter
            )

        # Create retry manager
        if retry_policy:
            self.retry_managers[endpoint.name] = RetryManager(
                f"{self.name}_{endpoint.name}", retry_policy
            )

        self.logger.info(
            "Added API endpoint",
            endpoint_name=endpoint.name,
            priority=endpoint.priority,
            circuit_breaker=self.config.enable_circuit_breaker,
            rate_limiting=self.config.enable_rate_limiting,
        )

    async def start_monitoring(self):
        """Start health monitoring for endpoints."""
        if not self.config.enable_health_monitoring:
            return

        if self.running:
            self.logger.warning("API manager already monitoring")
            return

        self.running = True
        self.health_monitor_task = asyncio.create_task(self._health_monitoring_loop())
        self.logger.info(
            "Started API health monitoring", interval=self.config.health_check_interval
        )

    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.running:
            return

        self.running = False
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped API health monitoring")

    async def _health_monitoring_loop(self):
        """Health monitoring loop."""
        while self.running:
            try:
                await self._check_endpoint_health()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(self.config.health_check_interval)

    async def _check_endpoint_health(self):
        """Check health of all endpoints."""
        for endpoint_name, endpoint in self.endpoints.items():
            if not endpoint.enabled:
                continue

            try:
                await self._check_single_endpoint_health(endpoint)
            except Exception as e:
                self.logger.error(
                    "Error checking endpoint health",
                    endpoint_name=endpoint_name,
                    error=str(e),
                )

    async def _check_single_endpoint_health(self, endpoint: APIEndpoint):
        """Check health of a single endpoint."""
        if not endpoint.health_check_url:
            return  # No health check configured

        start_time = time.time()

        try:
            # Perform health check (this should be customized based on your HTTP client)
            # For now, we'll simulate a health check
            await asyncio.sleep(0.1)  # Simulate network call

            duration = time.time() - start_time

            # Record response time
            endpoint.response_times.append(duration)
            if len(endpoint.response_times) > self.config.max_response_time_samples:
                endpoint.response_times.pop(0)

            # Update status based on response time and error count
            if duration > endpoint.timeout:
                endpoint.error_count += 1
                endpoint.status = APIStatus.DEGRADED
            else:
                endpoint.success_count += 1
                if endpoint.success_count >= self.config.recovery_threshold:
                    endpoint.status = APIStatus.HEALTHY
                    endpoint.error_count = 0

            endpoint.last_health_check = time.time()

            self.logger.debug(
                "Health check completed",
                endpoint_name=endpoint.name,
                status=endpoint.status.value,
                duration=duration,
            )

        except Exception as e:
            endpoint.error_count += 1
            endpoint.last_health_check = time.time()

            # Update status based on consecutive errors
            if endpoint.error_count >= self.config.unhealthy_threshold:
                endpoint.status = APIStatus.UNHEALTHY
            elif endpoint.error_count > 1:
                endpoint.status = APIStatus.DEGRADED

            self.logger.warning(
                "Health check failed",
                endpoint_name=endpoint.name,
                error=str(e),
                error_count=endpoint.error_count,
            )

    async def call(
        self, method: str, *args, endpoint_name: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Make API call with automatic failover.

        Args:
            method: Method name to call
            *args: Method arguments
            endpoint_name: Specific endpoint to use (optional)
            **kwargs: Method keyword arguments

        Returns:
            Method result
        """
        self.total_requests += 1

        # Get endpoint to use
        if endpoint_name:
            endpoint = self.endpoints.get(endpoint_name)
            if not endpoint or not endpoint.enabled:
                raise ValueError(f"Endpoint {endpoint_name} not available")
            endpoints_to_try = [endpoint]
        else:
            endpoints_to_try = self._get_endpoints_by_strategy()

        last_exception = None

        for endpoint in endpoints_to_try:
            try:
                result = await self._call_endpoint(endpoint, method, *args, **kwargs)
                self.successful_requests += 1
                return result

            except Exception as e:
                last_exception = e
                self.logger.warning(
                    "Endpoint call failed, trying next",
                    endpoint_name=endpoint.name,
                    error=str(e),
                )

                # Update endpoint error count
                endpoint.error_count += 1

                # Check if we should mark endpoint as unhealthy
                if endpoint.error_count >= self.config.unhealthy_threshold:
                    endpoint.status = APIStatus.UNHEALTHY
                    self.logger.error(
                        "Endpoint marked as unhealthy", endpoint_name=endpoint.name
                    )

                continue

        # All endpoints failed
        self.failed_requests += 1
        self.failover_count += 1

        if last_exception:
            self.logger.error(
                "All endpoints failed",
                endpoints_tried=len(endpoints_to_try),
                error=str(last_exception),
            )
            raise last_exception
        else:
            raise RuntimeError("No available endpoints")

    async def _call_endpoint(
        self, endpoint: APIEndpoint, method: str, *args, **kwargs
    ) -> Any:
        """
        Call specific endpoint with reliability components.

        Args:
            endpoint: APIEndpoint to call
            method: Method name
            *args: Method arguments
            **kwargs: Method keyword arguments

        Returns:
            Method result
        """
        start_time = time.time()

        try:
            # Use circuit breaker if available
            if endpoint.name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[endpoint.name]

                if endpoint.name in self.rate_limiters:
                    # Use rate-limited client with circuit breaker
                    rate_limited_client = self.rate_limiters[endpoint.name]
                    result = await circuit_breaker.call(
                        rate_limited_client.call, method, *args, **kwargs
                    )
                else:
                    # Use circuit breaker directly
                    client_method = getattr(endpoint.client, method)
                    result = await circuit_breaker.call(client_method, *args, **kwargs)

            elif endpoint.name in self.rate_limiters:
                # Use rate-limited client only
                rate_limited_client = self.rate_limiters[endpoint.name]
                result = await rate_limited_client.call(method, *args, **kwargs)

            else:
                # Direct client call
                client_method = getattr(endpoint.client, method)
                if asyncio.iscoroutinefunction(client_method):
                    result = await client_method(*args, **kwargs)
                else:
                    # Run sync method in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: client_method(*args, **kwargs)
                    )

            # Record successful call
            duration = time.time() - start_time
            endpoint.response_times.append(duration)
            if len(endpoint.response_times) > self.config.max_response_time_samples:
                endpoint.response_times.pop(0)

            endpoint.success_count += 1
            endpoint.error_count = max(
                0, endpoint.error_count - 1
            )  # Reduce error count

            # Update status if recovering
            if (
                endpoint.status != APIStatus.HEALTHY
                and endpoint.success_count >= self.config.recovery_threshold
            ):
                endpoint.status = APIStatus.HEALTHY
                self.logger.info("Endpoint recovered", endpoint_name=endpoint.name)

            return result

        except Exception as e:
            duration = time.time() - start_time
            endpoint.error_count += 1

            self.logger.error(
                "Endpoint call failed",
                endpoint_name=endpoint.name,
                method=method,
                duration=duration,
                error=str(e),
            )
            raise

    def _get_endpoints_by_strategy(self) -> List[APIEndpoint]:
        """
        Get endpoints ordered by failover strategy.

        Returns:
            List of endpoints in order to try
        """
        available_endpoints = [
            ep
            for ep in self.endpoints.values()
            if ep.enabled and ep.status != APIStatus.OFFLINE
        ]

        if not available_endpoints:
            return []

        if self.config.failover_strategy == FailoverStrategy.PRIORITY:
            return sorted(available_endpoints, key=lambda ep: ep.priority)

        elif self.config.failover_strategy == FailoverStrategy.ROUND_ROBIN:
            # Simple round-robin
            if self.current_endpoint_index >= len(available_endpoints):
                self.current_endpoint_index = 0

            # Rotate the list
            rotated = (
                available_endpoints[self.current_endpoint_index :]
                + available_endpoints[: self.current_endpoint_index]
            )
            self.current_endpoint_index = (self.current_endpoint_index + 1) % len(
                available_endpoints
            )
            return rotated

        elif self.config.failover_strategy == FailoverStrategy.LEAST_LOADED:
            # Sort by error count (ascending)
            return sorted(available_endpoints, key=lambda ep: ep.error_count)

        elif self.config.failover_strategy == FailoverStrategy.FASTEST_RESPONSE:
            # Sort by average response time
            def avg_response_time(ep):
                if not ep.response_times:
                    return float("inf")
                return sum(ep.response_times) / len(ep.response_times)

            return sorted(available_endpoints, key=avg_response_time)

        else:
            return available_endpoints

    def get_endpoint_status(self, endpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of specific endpoint.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Endpoint status dictionary
        """
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            return None

        avg_response_time = 0.0
        if endpoint.response_times:
            avg_response_time = sum(endpoint.response_times) / len(
                endpoint.response_times
            )

        return {
            "name": endpoint.name,
            "status": endpoint.status.value,
            "enabled": endpoint.enabled,
            "priority": endpoint.priority,
            "error_count": endpoint.error_count,
            "success_count": endpoint.success_count,
            "avg_response_time": avg_response_time,
            "last_health_check": endpoint.last_health_check,
            "circuit_breaker": endpoint.name in self.circuit_breakers,
            "rate_limited": endpoint.name in self.rate_limiters,
        }

    def get_all_endpoint_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all endpoints."""
        return {name: self.get_endpoint_status(name) for name in self.endpoints.keys()}

    def get_metrics(self) -> Dict[str, Any]:
        """Get API manager metrics."""
        healthy_endpoints = sum(
            1 for ep in self.endpoints.values() if ep.status == APIStatus.HEALTHY
        )

        return {
            "name": self.name,
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": healthy_endpoints,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "failover_count": self.failover_count,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "failover_strategy": self.config.failover_strategy.value,
            "monitoring_active": self.running,
        }

    def enable_endpoint(self, endpoint_name: str):
        """Enable specific endpoint."""
        if endpoint_name in self.endpoints:
            self.endpoints[endpoint_name].enabled = True
            self.logger.info("Enabled endpoint", endpoint_name=endpoint_name)

    def disable_endpoint(self, endpoint_name: str):
        """Disable specific endpoint."""
        if endpoint_name in self.endpoints:
            self.endpoints[endpoint_name].enabled = False
            self.logger.info("Disabled endpoint", endpoint_name=endpoint_name)

    async def health_check(self) -> bool:
        """
        Check if API manager has healthy endpoints.

        Returns:
            True if at least one endpoint is healthy
        """
        healthy_count = sum(
            1
            for ep in self.endpoints.values()
            if ep.enabled and ep.status == APIStatus.HEALTHY
        )

        return healthy_count > 0
