"""
Base async client with connection pooling and resource management.
"""

import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from contextlib import asynccontextmanager

from ..resilience.circuit_breaker import CircuitBreaker
from ..resilience.retry_strategy import RetryStrategy

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for async client."""
    base_url: str
    timeout: int = 30
    max_connections: int = 100
    max_connections_per_host: int = 30
    keepalive_timeout: int = 30
    enable_keepalive: bool = True
    headers: Optional[Dict[str, str]] = None
    auth_token: Optional[str] = None
    rate_limit: Optional[int] = None  # requests per second
    circuit_breaker_enabled: bool = True
    retry_enabled: bool = True


@dataclass
class RequestMetrics:
    """Metrics for request tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class AsyncClientBase(ABC):
    """Base class for async HTTP clients with connection pooling."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.metrics = RequestMetrics()

        # Setup circuit breaker
        if config.circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=aiohttp.ClientError
            )
        else:
            self.circuit_breaker = None

        # Setup retry strategy
        if config.retry_enabled:
            self.retry_strategy = RetryStrategy(
                max_attempts=3,
                base_delay=1.0,
                max_delay=10.0,
                exponential_base=2.0
            )
        else:
            self.retry_strategy = None

        self._rate_limiter: Optional[asyncio.Semaphore] = None
        if config.rate_limit:
            self._rate_limiter = asyncio.Semaphore(config.rate_limit)

        self._last_request_times: List[float] = []

    async def initialize(self) -> None:
        """Initialize the client session."""
        if self.session is not None:
            return

        # Create connector with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections_per_host,
            keepalive_timeout=self.config.keepalive_timeout,
            enable_cleanup_closed=True,
            use_dns_cache=True
        )

        # Setup timeout
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        # Setup headers
        headers = self.config.headers or {}
        if self.config.auth_token:
            headers['Authorization'] = f'Bearer {self.config.auth_token}'

        # Create session
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers,
            raise_for_status=False  # Handle status codes manually
        )

        logger.info(f"Initialized async client for {self.config.base_url}")

    async def close(self) -> None:
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info(f"Closed async client for {self.config.base_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with connection pooling and resilience."""
        if not self.session:
            await self.initialize()

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Apply rate limiting
        if self._rate_limiter:
            await self._apply_rate_limit()

        # Prepare request
        request_kwargs = {
            'method': method,
            'url': url,
            'json': data,
            'params': params,
            'headers': headers
        }

        if timeout:
            request_kwargs['timeout'] = aiohttp.ClientTimeout(total=timeout)

        # Execute with circuit breaker and retry
        if self.circuit_breaker:
            response = await self.circuit_breaker.call(self._execute_request, **request_kwargs)
        else:
            response = await self._execute_request(**request_kwargs)

        return response

    async def _execute_request(self, **kwargs) -> Dict[str, Any]:
        """Execute HTTP request with retry logic."""
        start_time = time.time()

        try:
            if self.retry_strategy:
                response = await self.retry_strategy.execute(self._make_request, **kwargs)
            else:
                response = await self._make_request(**kwargs)

            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_request_time = datetime.now()

            # Update average response time
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)

            return response

        except Exception as e:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_request_time = datetime.now()

            logger.error(f"Request failed: {e}")
            raise

    async def _make_request(self, **kwargs) -> Dict[str, Any]:
        """Make the actual HTTP request."""
        async with self.session.request(**kwargs) as response:
            # Handle different response types
            content_type = response.headers.get('content-type', '').lower()

            if response.status >= 400:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=error_text
                )

            if 'application/json' in content_type:
                return await response.json()
            else:
                text = await response.text()
                return {'content': text, 'status': response.status}

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting."""
        current_time = time.time()

        # Clean old timestamps
        cutoff_time = current_time - 1.0  # 1 second window
        self._last_request_times = [
            t for t in self._last_request_times if t > cutoff_time
        ]

        # Check if we need to wait
        if len(self._last_request_times) >= self.config.rate_limit:
            sleep_time = 1.0 - (current_time - self._last_request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self._last_request_times.append(current_time)

    def _update_avg_response_time(self, response_time: float) -> None:
        """Update average response time."""
        if self.metrics.avg_response_time == 0:
            self.metrics.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_response_time = (
                alpha * response_time +
                (1 - alpha) * self.metrics.avg_response_time
            )

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        pass

    async def get_metrics(self) -> RequestMetrics:
        """Get client metrics."""
        return self.metrics


class ClientPool:
    """Pool of async clients for load balancing and failover."""

    def __init__(self, clients: List[AsyncClientBase]):
        self.clients = clients
        self.current_index = 0
        self._health_status: Dict[int, bool] = {}
        self._health_check_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize all clients in the pool."""
        for client in self.clients:
            await client.initialize()

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Initialized client pool with {len(self.clients)} clients")

    async def close(self) -> None:
        """Close all clients in the pool."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        for client in self.clients:
            await client.close()

        logger.info("Closed client pool")

    async def get_healthy_client(self) -> Optional[AsyncClientBase]:
        """Get a healthy client from the pool."""
        healthy_clients = [
            client for i, client in enumerate(self.clients)
            if self._health_status.get(i, True)
        ]

        if not healthy_clients:
            logger.warning("No healthy clients available")
            return None

        # Round-robin selection
        client = healthy_clients[self.current_index % len(healthy_clients)]
        self.current_index += 1

        return client

    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make request using healthy client from pool."""
        client = await self.get_healthy_client()
        if not client:
            raise RuntimeError("No healthy clients available")

        return await client.request(method, endpoint, **kwargs)

    async def _health_check_loop(self) -> None:
        """Periodic health check for all clients."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._check_all_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_all_health(self) -> None:
        """Check health of all clients."""
        tasks = []
        for i, client in enumerate(self.clients):
            task = asyncio.create_task(self._check_client_health(i, client))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_client_health(self, index: int, client: AsyncClientBase) -> None:
        """Check health of individual client."""
        try:
            is_healthy = await asyncio.wait_for(client.health_check(), timeout=10)

            if self._health_status.get(index) != is_healthy:
                status = "healthy" if is_healthy else "unhealthy"
                logger.info(f"Client {index} is now {status}")

            self._health_status[index] = is_healthy

        except Exception as e:
            logger.warning(f"Health check failed for client {index}: {e}")
            self._health_status[index] = False

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for the client pool."""
        healthy_count = sum(1 for status in self._health_status.values() if status)

        client_metrics = []
        for i, client in enumerate(self.clients):
            metrics = await client.get_metrics()
            client_metrics.append({
                'index': i,
                'healthy': self._health_status.get(i, True),
                'metrics': {
                    'total_requests': metrics.total_requests,
                    'success_rate': metrics.success_rate,
                    'avg_response_time': metrics.avg_response_time
                }
            })

        return {
            'total_clients': len(self.clients),
            'healthy_clients': healthy_count,
            'client_metrics': client_metrics
        }
