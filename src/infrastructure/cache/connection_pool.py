"""
Redis connection pool management for high-performance caching.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)


class RedisConnectionPool:
    """Manages Redis connection pool with health monitoring and failover."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        connection_timeout: int = 5,
        socket_timeout: int = 5
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        self.connection_timeout = connection_timeout
        self.socket_timeout = socket_timeout

        self._pool: Optional[ConnectionPool] = None
        self._redis_client: Optional[redis.Redis] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = True

    async def initialize(self) -> None:
        """Initialize connection pool."""
        try:
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                retry_on_timeout=self.retry_on_timeout,
                socket_connect_timeout=self.connection_timeout,
                socket_timeout=self.socket_timeout,
                decode_responses=False  # We handle serialization ourselves
            )

            self._redis_client = redis.Redis(connection_pool=self._pool)

            # Test connection
            await self._redis_client.ping()
            logger.info(f"Redis connection pool initialized: {self.host}:{self.port}")

            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            raise

    async def get_connection(self) -> redis.Redis:
        """Get Redis client from pool."""
        if not self._redis_client:
            await self.initialize()

        if not self._is_healthy:
            raise ConnectionError("Redis connection pool is unhealthy")

        return self._redis_client

    async def close(self) -> None:
        """Close connection pool and cleanup resources."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._redis_client:
            await self._redis_client.close()

        if self._pool:
            await self._pool.disconnect()

        logger.info("Redis connection pool closed")

    async def _health_check_loop(self) -> None:
        """Periodic health check for Redis connection."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _perform_health_check(self) -> None:
        """Perform health check on Redis connection."""
        try:
            if self._redis_client:
                await asyncio.wait_for(
                    self._redis_client.ping(),
                    timeout=self.connection_timeout
                )

                if not self._is_healthy:
                    logger.info("Redis connection restored")
                    self._is_healthy = True

        except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
            if self._is_healthy:
                logger.warning(f"Redis connection unhealthy: {e}")
                self._is_healthy = False
        except Exception as e:
            logger.error(f"Unexpected error in health check: {e}")
            self._is_healthy = False

    @property
    def is_healthy(self) -> bool:
        """Check if connection pool is healthy."""
        return self._is_healthy

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self._pool:
            return {}

        try:
            return {
                'max_connections': self.max_connections,
                'created_connections': self._pool.created_connections,
                'available_connections': len(self._pool._available_connections),
                'in_use_connections': len(self._pool._in_use_connections),
                'is_healthy': self._is_healthy
            }
        except Exception as e:
            logger.error(f"Error getting pool stats: {e}")
            return {}


class RedisConnectionManager:
    """Manages multiple Redis connection pools for different use cases."""

    def __init__(self):
        self._pools: Dict[str, RedisConnectionPool] = {}

    def add_pool(self, name: str, pool: RedisConnectionPool) -> None:
        """Add a named connection pool."""
        self._pools[name] = pool

    async def get_pool(self, name: str) -> RedisConnectionPool:
        """Get connection pool by name."""
        if name not in self._pools:
            raise ValueError(f"Connection pool '{name}' not found")

        pool = self._pools[name]
        if not pool.is_healthy:
            # Try to reinitialize unhealthy pool
            try:
                await pool.initialize()
            except Exception as e:
                logger.error(f"Failed to reinitialize pool '{name}': {e}")
                raise

        return pool

    async def close_all(self) -> None:
        """Close all connection pools."""
        for name, pool in self._pools.items():
            try:
                await pool.close()
                logger.info(f"Closed connection pool: {name}")
            except Exception as e:
                logger.error(f"Error closing pool '{name}': {e}")

        self._pools.clear()

    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all connection pools."""
        stats = {}
        for name, pool in self._pools.items():
            stats[name] = await pool.get_pool_stats()
        return stats


# Global connection manager instance
connection_manager = RedisConnectionManager()


@asynccontextmanager
async def redis_connection_context(pool_name: str = "default"):
    """Context manager for Redis connections."""
    pool = await connection_manager.get_pool(pool_name)
    connection = await pool.get_connection()
    try:
        yield connection
    finally:
        # Connection is returned to pool automatically
        pass
