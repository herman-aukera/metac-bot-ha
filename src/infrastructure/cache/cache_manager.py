"""
High-level cache manager for coordinating multiple cache instances and strategies.
"""

import asyncio
import logging
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from .redis_cache import RedisCache, CacheNamespace
from .connection_pool import RedisConnectionPool, connection_manager
from .cache_strategies import CacheStrategy, CacheStrategyFactory

logger = logging.getLogger(__name__)


class CacheManager:
    """High-level cache manager with multiple cache instances and intelligent routing."""

    def __init__(self):
        self._caches: Dict[str, RedisCache] = {}
        self._namespaces: Dict[str, CacheNamespace] = {}
        self._strategies: Dict[str, CacheStrategy] = {}
        self._default_cache: Optional[str] = None
        self._warming_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(
        self,
        cache_configs: Dict[str, Dict[str, Any]],
        default_cache: str = "default"
    ) -> None:
        """Initialize cache manager with multiple cache configurations."""
        try:
            for cache_name, config in cache_configs.items():
                await self._setup_cache(cache_name, config)

            self._default_cache = default_cache
            logger.info(f"Cache manager initialized with {len(self._caches)} caches")

        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise

    async def _setup_cache(self, cache_name: str, config: Dict[str, Any]) -> None:
        """Setup individual cache instance."""
        # Create connection pool
        pool_config = config.get("connection", {})
        pool = RedisConnectionPool(
            host=pool_config.get("host", "localhost"),
            port=pool_config.get("port", 6379),
            db=pool_config.get("db", 0),
            password=pool_config.get("password"),
            max_connections=pool_config.get("max_connections", 50)
        )

        await pool.initialize()
        connection_manager.add_pool(cache_name, pool)

        # Create cache instance
        cache_config = config.get("cache", {})
        cache = RedisCache(
            connection_pool=pool,
            default_ttl=cache_config.get("default_ttl", 3600),
            key_prefix=cache_config.get("key_prefix", f"{cache_name}:"),
            serialization_method=cache_config.get("serialization", "json")
        )

        self._caches[cache_name] = cache

        # Setup cache strategy
        strategy_config = config.get("strategy", {"type": "adaptive"})
        strategy = CacheStrategyFactory.create_strategy(
            strategy_config["type"],
            **strategy_config.get("params", {})
        )
        self._strategies[cache_name] = strategy

        # Setup namespaces
        for namespace in config.get("namespaces", []):
            namespace_key = f"{cache_name}:{namespace}"
            self._namespaces[namespace_key] = CacheNamespace(cache, namespace)

        logger.info(f"Setup cache '{cache_name}' with strategy '{strategy_config['type']}'")

    def get_cache(self, cache_name: Optional[str] = None) -> RedisCache:
        """Get cache instance by name."""
        name = cache_name or self._default_cache
        if name not in self._caches:
            raise ValueError(f"Cache '{name}' not found")
        return self._caches[name]

    def get_namespace(self, cache_name: str, namespace: str) -> CacheNamespace:
        """Get cache namespace."""
        namespace_key = f"{cache_name}:{namespace}"
        if namespace_key not in self._namespaces:
            cache = self.get_cache(cache_name)
            self._namespaces[namespace_key] = CacheNamespace(cache, namespace)
        return self._namespaces[namespace_key]

    async def get(
        self,
        key: str,
        cache_name: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from cache with strategy consultation."""
        try:
            if namespace:
                cache_ns = self.get_namespace(cache_name or self._default_cache, namespace)
                result = await cache_ns.get(key)
            else:
                cache = self.get_cache(cache_name)
                result = await cache.get(key)

            # Update strategy metrics
            strategy = self._strategies.get(cache_name or self._default_cache)
            if strategy:
                if result is not None:
                    strategy.update_metrics("hit")
                    if hasattr(strategy, 'record_access'):
                        strategy.record_access(key)
                else:
                    strategy.update_metrics("miss")

            return result

        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_name: Optional[str] = None,
        namespace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set value in cache with strategy consultation."""
        try:
            cache_name = cache_name or self._default_cache
            strategy = self._strategies.get(cache_name)
            context = context or {}

            # Consult strategy
            if strategy:
                should_cache = await strategy.should_cache(key, value, context)
                if not should_cache:
                    logger.debug(f"Strategy declined to cache key: {key}")
                    return False

                if ttl is None:
                    ttl = await strategy.get_ttl(key, value, context)

            # Set in cache
            if namespace:
                cache_ns = self.get_namespace(cache_name, namespace)
                result = await cache_ns.set(key, value, ttl)
            else:
                cache = self.get_cache(cache_name)
                result = await cache.set(key, value, ttl)

            return result

        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    async def delete(
        self,
        key: str,
        cache_name: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """Delete key from cache."""
        try:
            if namespace:
                cache_ns = self.get_namespace(cache_name or self._default_cache, namespace)
                return await cache_ns.delete(key)
            else:
                cache = self.get_cache(cache_name)
                return await cache.delete(key)

        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    async def mget(
        self,
        keys: List[str],
        cache_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get multiple values from cache."""
        try:
            cache = self.get_cache(cache_name)
            return await cache.mget(keys)
        except Exception as e:
            logger.error(f"Error getting multiple cache keys: {e}")
            return {}

    async def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        cache_name: Optional[str] = None
    ) -> bool:
        """Set multiple values in cache."""
        try:
            cache = self.get_cache(cache_name)
            return await cache.mset(mapping, ttl)
        except Exception as e:
            logger.error(f"Error setting multiple cache keys: {e}")
            return False

    async def warm_cache(
        self,
        cache_name: str,
        warm_function: Callable[[], Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> None:
        """Warm cache with data from function."""
        try:
            logger.info(f"Starting cache warming for '{cache_name}'")

            # Cancel existing warming task
            if cache_name in self._warming_tasks:
                self._warming_tasks[cache_name].cancel()

            # Start new warming task
            task = asyncio.create_task(
                self._warm_cache_task(cache_name, warm_function, ttl)
            )
            self._warming_tasks[cache_name] = task

        except Exception as e:
            logger.error(f"Error starting cache warming for '{cache_name}': {e}")

    async def _warm_cache_task(
        self,
        cache_name: str,
        warm_function: Callable[[], Dict[str, Any]],
        ttl: Optional[int]
    ) -> None:
        """Background task for cache warming."""
        try:
            data = await asyncio.get_event_loop().run_in_executor(None, warm_function)

            if data:
                await self.mset(data, ttl, cache_name)
                logger.info(f"Cache warming completed for '{cache_name}': {len(data)} keys")

        except asyncio.CancelledError:
            logger.info(f"Cache warming cancelled for '{cache_name}'")
        except Exception as e:
            logger.error(f"Error in cache warming for '{cache_name}': {e}")
        finally:
            if cache_name in self._warming_tasks:
                del self._warming_tasks[cache_name]

    async def invalidate_pattern(
        self,
        pattern: str,
        cache_name: Optional[str] = None
    ) -> int:
        """Invalidate all keys matching pattern."""
        try:
            cache = self.get_cache(cache_name)
            return await cache.clear_pattern(pattern)
        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "caches": {},
            "strategies": {},
            "namespaces": len(self._namespaces),
            "warming_tasks": len(self._warming_tasks)
        }

        # Cache stats
        for name, cache in self._caches.items():
            cache_stats = await cache.get_stats()
            stats["caches"][name] = cache_stats

        # Strategy stats
        for name, strategy in self._strategies.items():
            stats["strategies"][name] = {
                "type": strategy.__class__.__name__,
                "metrics": {
                    "hits": strategy.metrics.hits,
                    "misses": strategy.metrics.misses,
                    "hit_rate": strategy.metrics.hit_rate,
                    "evictions": strategy.metrics.evictions
                }
            }

        return stats

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all cache instances."""
        health = {}

        for name, cache in self._caches.items():
            try:
                # Try a simple operation
                test_key = f"health_check_{int(datetime.now().timestamp())}"
                await cache.set(test_key, "ok", ttl=1)
                result = await cache.get(test_key)
                await cache.delete(test_key)

                health[name] = result == "ok"

            except Exception as e:
                logger.error(f"Health check failed for cache '{name}': {e}")
                health[name] = False

        return health

    async def close(self) -> None:
        """Close all cache connections and cleanup resources."""
        # Cancel warming tasks
        for task in self._warming_tasks.values():
            task.cancel()

        if self._warming_tasks:
            await asyncio.gather(*self._warming_tasks.values(), return_exceptions=True)

        # Close connection manager
        await connection_manager.close_all()

        logger.info("Cache manager closed")


# Global cache manager instance
cache_manager = CacheManager()


@asynccontextmanager
async def cache_context(cache_name: Optional[str] = None):
    """Context manager for cache operations."""
    cache = cache_manager.get_cache(cache_name)
    try:
        yield cache
    finally:
        # Cleanup if needed
        pass


# Convenience functions
async def get_cached(key: str, cache_name: Optional[str] = None) -> Optional[Any]:
    """Get value from cache."""
    return await cache_manager.get(key, cache_name)


async def set_cached(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
    cache_name: Optional[str] = None
) -> bool:
    """Set value in cache."""
    return await cache_manager.set(key, value, ttl, cache_name)


async def delete_cached(key: str, cache_name: Optional[str] = None) -> bool:
    """Delete key from cache."""
    return await cache_manager.delete(key, cache_name)
