"""
Connection manager for external API clients with pooling and resource management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from enum import Enum

from .async_client_base import AsyncClientBase, ClientPool, ClientConfig

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class ConnectionStats:
    """Statistics for a connection."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def uptime(self) -> timedelta:
        """Calculate connection uptime."""
        return datetime.now() - self.created_at


class ConnectionManager:
    """Manages connections to external APIs with pooling and health monitoring."""

    def __init__(self, health_check_interval: int = 60):
        self.health_check_interval = health_check_interval

        # Connection pools by service name
        self.client_pools: Dict[str, ClientPool] = {}
        self.connection_configs: Dict[str, List[ClientConfig]] = {}
        self.connection_stats: Dict[str, ConnectionStats] = {}
        self.connection_states: Dict[str, ConnectionState] = {}

        # Health monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

        # Circuit breaker states
        self.circuit_breakers: Dict[str, bool] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_times: Dict[str, datetime] = {}

    async def initialize(self) -> None:
        """Initialize the connection manager."""
        logger.info("Initializing connection manager")

        # Initialize all client pools
        for service_name, pool in self.client_pools.items():
            try:
                await pool.initialize()
                self.connection_states[service_name] = ConnectionState.CONNECTED
                logger.info(f"Initialized connection pool for service: {service_name}")
            except Exception as e:
                self.connection_states[service_name] = ConnectionState.ERROR
                logger.error(f"Failed to initialize pool for {service_name}: {e}")

        # Start health monitoring
        await self.start_health_monitoring()

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        logger.info("Closing connection manager")

        # Stop health monitoring
        await self.stop_health_monitoring()

        # Close all client pools
        for service_name, pool in self.client_pools.items():
            try:
                await pool.close()
                self.connection_states[service_name] = ConnectionState.DISCONNECTED
                logger.info(f"Closed connection pool for service: {service_name}")
            except Exception as e:
                logger.error(f"Error closing pool for {service_name}: {e}")

    def register_service(
        self,
        service_name: str,
        client_configs: List[ClientConfig],
        client_class: Type[AsyncClientBase]
    ) -> None:
        """Register a service with multiple client configurations for load balancing."""

        # Create clients from configurations
        clients = []
        for config in client_configs:
            client = client_class(config)
            clients.append(client)

        # Create client pool
        pool = ClientPool(clients)

        # Store configurations and pool
        self.connection_configs[service_name] = client_configs
        self.client_pools[service_name] = pool
        self.connection_stats[service_name] = ConnectionStats()
        self.connection_states[service_name] = ConnectionState.DISCONNECTED
        self.circuit_breakers[service_name] = False
        self.failure_counts[service_name] = 0

        logger.info(f"Registered service '{service_name}' with {len(clients)} clients")

    async def get_client_pool(self, service_name: str) -> Optional[ClientPool]:
        """Get client pool for a service."""
        if service_name not in self.client_pools:
            logger.error(f"Service '{service_name}' not registered")
            return None

        # Check circuit breaker
        if self.circuit_breakers.get(service_name, False):
            logger.warning(f"Circuit breaker open for service: {service_name}")
            return None

        # Check connection state
        state = self.connection_states.get(service_name)
        if state not in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            logger.warning(f"Service '{service_name}' not available (state: {state})")
            return None

        return self.client_pools[service_name]

    @asynccontextmanager
    async def get_client(self, service_name: str):
        """Context manager to get a client for a service."""
        pool = await self.get_client_pool(service_name)
        if not pool:
            raise RuntimeError(f"No available clients for service: {service_name}")

        client = await pool.get_healthy_client()
        if not client:
            raise RuntimeError(f"No healthy clients available for service: {service_name}")

        try:
            yield client
        finally:
            # Client is automatically returned to pool
            pass

    async def make_request(
        self,
        service_name: str,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a request using the connection manager."""

        start_time = datetime.now()
        stats = self.connection_stats.get(service_name)

        try:
            async with self.get_client(service_name) as client:
                response = await client.request(method, endpoint, **kwargs)

                # Update success statistics
                if stats:
                    stats.total_requests += 1
                    stats.successful_requests += 1
                    stats.last_used = datetime.now()

                    # Update average response time
                    response_time = (datetime.now() - start_time).total_seconds()
                    if stats.avg_response_time == 0:
                        stats.avg_response_time = response_time
                    else:
                        # Exponential moving average
                        alpha = 0.1
                        stats.avg_response_time = (
                            alpha * response_time +
                            (1 - alpha) * stats.avg_response_time
                        )

                # Reset failure count on success
                self.failure_counts[service_name] = 0

                return response

        except Exception as e:
            # Update failure statistics
            if stats:
                stats.total_requests += 1
                stats.failed_requests += 1

            # Update failure tracking
            self.failure_counts[service_name] = self.failure_counts.get(service_name, 0) + 1
            self.last_failure_times[service_name] = datetime.now()

            # Check if circuit breaker should be triggered
            await self._check_circuit_breaker(service_name)

            logger.error(f"Request failed for service '{service_name}': {e}")
            raise

    async def _check_circuit_breaker(self, service_name: str) -> None:
        """Check if circuit breaker should be triggered for a service."""
        failure_count = self.failure_counts.get(service_name, 0)
        failure_threshold = 5  # Configurable threshold

        if failure_count >= failure_threshold:
            self.circuit_breakers[service_name] = True
            self.connection_states[service_name] = ConnectionState.ERROR
            logger.warning(f"Circuit breaker triggered for service: {service_name}")

            # Schedule circuit breaker reset
            asyncio.create_task(self._reset_circuit_breaker(service_name, 60))  # 60 seconds

    async def _reset_circuit_breaker(self, service_name: str, delay: int) -> None:
        """Reset circuit breaker after delay."""
        await asyncio.sleep(delay)

        # Try to reconnect
        try:
            pool = self.client_pools.get(service_name)
            if pool:
                # Test connection health
                healthy_clients = 0
                for client in pool.clients:
                    if await client.health_check():
                        healthy_clients += 1

                if healthy_clients > 0:
                    self.circuit_breakers[service_name] = False
                    self.connection_states[service_name] = ConnectionState.CONNECTED
                    self.failure_counts[service_name] = 0
                    logger.info(f"Circuit breaker reset for service: {service_name}")
                else:
                    # Schedule another reset attempt
                    asyncio.create_task(self._reset_circuit_breaker(service_name, delay * 2))

        except Exception as e:
            logger.error(f"Failed to reset circuit breaker for {service_name}: {e}")
            # Schedule another reset attempt
            asyncio.create_task(self._reset_circuit_breaker(service_name, delay * 2))

    async def start_health_monitoring(self) -> None:
        """Start health monitoring for all services."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("Started connection health monitoring")

    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped connection health monitoring")

    async def _health_monitoring_loop(self) -> None:
        """Health monitoring loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all services."""
        for service_name, pool in self.client_pools.items():
            try:
                # Skip if circuit breaker is open
                if self.circuit_breakers.get(service_name, False):
                    continue

                # Check pool health
                pool_stats = await pool.get_pool_stats()
                healthy_clients = pool_stats.get('healthy_clients', 0)
                total_clients = pool_stats.get('total_clients', 0)

                if healthy_clients == 0:
                    self.connection_states[service_name] = ConnectionState.ERROR
                    logger.warning(f"No healthy clients for service: {service_name}")
                elif healthy_clients < total_clients:
                    logger.warning(
                        f"Service '{service_name}' has {healthy_clients}/{total_clients} healthy clients"
                    )
                else:
                    self.connection_states[service_name] = ConnectionState.CONNECTED

            except Exception as e:
                logger.error(f"Health check failed for service '{service_name}': {e}")
                self.connection_states[service_name] = ConnectionState.ERROR

    def get_service_stats(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a service."""
        if service_name not in self.connection_stats:
            return None

        stats = self.connection_stats[service_name]
        state = self.connection_states.get(service_name, ConnectionState.DISCONNECTED)

        return {
            'service_name': service_name,
            'state': state.value,
            'total_requests': stats.total_requests,
            'successful_requests': stats.successful_requests,
            'failed_requests': stats.failed_requests,
            'success_rate': stats.success_rate,
            'avg_response_time': stats.avg_response_time,
            'uptime_seconds': stats.uptime.total_seconds(),
            'last_used': stats.last_used.isoformat() if stats.last_used else None,
            'circuit_breaker_open': self.circuit_breakers.get(service_name, False),
            'failure_count': self.failure_counts.get(service_name, 0)
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all services."""
        all_stats = {}
        for service_name in self.connection_stats.keys():
            all_stats[service_name] = self.get_service_stats(service_name)
        return all_stats

    async def force_reconnect(self, service_name: str) -> bool:
        """Force reconnection for a service."""
        if service_name not in self.client_pools:
            return False

        try:
            self.connection_states[service_name] = ConnectionState.CONNECTING

            # Close and reinitialize the pool
            pool = self.client_pools[service_name]
            await pool.close()
            await pool.initialize()

            # Reset circuit breaker and failure counts
            self.circuit_breakers[service_name] = False
            self.failure_counts[service_name] = 0

            self.connection_states[service_name] = ConnectionState.CONNECTED
            logger.info(f"Successfully reconnected service: {service_name}")
            return True

        except Exception as e:
            self.connection_states[service_name] = ConnectionState.ERROR
            logger.error(f"Failed to reconnect service '{service_name}': {e}")
            return False

    async def maintenance_mode(self, service_name: str, enable: bool) -> None:
        """Enable or disable maintenance mode for a service."""
        if service_name not in self.client_pools:
            return

        if enable:
            self.connection_states[service_name] = ConnectionState.MAINTENANCE
            logger.info(f"Enabled maintenance mode for service: {service_name}")
        else:
            self.connection_states[service_name] = ConnectionState.CONNECTED
            logger.info(f"Disabled maintenance mode for service: {service_name}")


# Global connection manager instance
connection_manager = ConnectionManager()
