"""
Redis-based caching implementation with TTL management and connection pooling.
"""

import json
import pickle
import asyncio
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
import logging

from .connection_pool import RedisConnectionPool
from .cache_strategies import CacheStrategy, TTLCacheStrategy

logger = logging.getLogger(__name__)


class RedisCache:
    """High-performance Redis cache with TTL management and connection pooling."""

    def __init__(
        self,
        connection_pool: RedisConnectionPool,
        default_ttl: int = 3600,
        key_prefix: str = "tournament_opt:",
        serialization_method: str = "json"
    ):
        self.connection_pool = connection_pool
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.serialization_method = serialization_method
        self.strategy = TTLCacheStrategy(default_ttl)

    async def get_connection(self) -> redis.Redis:
        """Get Redis connection from pool."""
        return await self.connection_pool.get_connection()

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.serialization_method == "json":
            return json.dumps(value, default=str).encode('utf-8')
        elif self.serialization_method == "pickle":
            return pickle.dumps(value)
        else:
            raise ValueError(f"Unsupported serialization method: {self.serialization_method}")

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.serialization_method == "json":
            return json.loads(data.decode('utf-8'))
        elif self.serialization_method == "pickle":
            return pickle.loads(data)
        else:
            raise ValueError(f"Unsupported serialization method: {self.serialization_method}")

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            redis_client = await self.get_connection()
            cache_key = self._make_key(key)

            data = await redis_client.get(cache_key)
            if data is None:
                logger.debug(f"Cache miss for key: {key}")
                return None

            logger.debug(f"Cache hit for key: {key}")
            return self._deserialize(data)

        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False
    ) -> bool:
        """Set value in cache with TTL."""
        try:
            redis_client = await self.get_connection()
            cache_key = self._make_key(key)
            serialized_value = self._serialize(value)

            ttl_seconds = ttl or self.default_ttl

            if nx:
                # Set only if key doesn't exist
                result = await redis_client.set(cache_key, serialized_value, ex=ttl_seconds, nx=True)
            else:
                result = await redis_client.set(cache_key, serialized_value, ex=ttl_seconds)

            logger.debug(f"Cache set for key: {key}, TTL: {ttl_seconds}s")
            return bool(result)

        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            redis_client = await self.get_connection()
            cache_key = self._make_key(key)

            result = await redis_client.delete(cache_key)
            logger.debug(f"Cache delete for key: {key}")
            return bool(result)

        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            redis_client = await self.get_connection()
            cache_key = self._make_key(key)

            result = await redis_client.exists(cache_key)
            return bool(result)

        except Exception as e:
            logger.error(f"Error checking cache key existence {key}: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        try:
            redis_client = await self.get_connection()
            cache_key = self._make_key(key)

            result = await redis_client.expire(cache_key, ttl)
            return bool(result)

        except Exception as e:
            logger.error(f"Error setting TTL for cache key {key}: {e}")
            return False

    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key."""
        try:
            redis_client = await self.get_connection()
            cache_key = self._make_key(key)

            ttl = await redis_client.ttl(cache_key)
            return ttl if ttl > 0 else None

        except Exception as e:
            logger.error(f"Error getting TTL for cache key {key}: {e}")
            return None

    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        try:
            redis_client = await self.get_connection()
            cache_keys = [self._make_key(key) for key in keys]

            values = await redis_client.mget(cache_keys)
            result = {}

            for i, (key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    result[key] = self._deserialize(value)
                    logger.debug(f"Cache hit for key: {key}")
                else:
                    logger.debug(f"Cache miss for key: {key}")

            return result

        except Exception as e:
            logger.error(f"Error getting multiple cache keys: {e}")
            return {}

    async def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        try:
            redis_client = await self.get_connection()
            ttl_seconds = ttl or self.default_ttl

            # Use pipeline for atomic operations
            async with redis_client.pipeline() as pipe:
                for key, value in mapping.items():
                    cache_key = self._make_key(key)
                    serialized_value = self._serialize(value)
                    pipe.set(cache_key, serialized_value, ex=ttl_seconds)

                await pipe.execute()

            logger.debug(f"Cache mset for {len(mapping)} keys, TTL: {ttl_seconds}s")
            return True

        except Exception as e:
            logger.error(f"Error setting multiple cache keys: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value in cache."""
        try:
            redis_client = await self.get_connection()
            cache_key = self._make_key(key)

            result = await redis_client.incrby(cache_key, amount)
            return result

        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            return None

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            redis_client = await self.get_connection()
            cache_pattern = self._make_key(pattern)

            keys = await redis_client.keys(cache_pattern)
            if keys:
                deleted = await redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache keys matching pattern: {pattern}")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            redis_client = await self.get_connection()
            info = await redis_client.info()

            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                )
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


class CacheNamespace:
    """Namespace wrapper for cache operations."""

    def __init__(self, cache: RedisCache, namespace: str):
        self.cache = cache
        self.namespace = namespace

    def _namespaced_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.namespace}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from namespaced cache."""
        return await self.cache.get(self._namespaced_key(key))

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in namespaced cache."""
        return await self.cache.set(self._namespaced_key(key), value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete key from namespaced cache."""
        return await self.cache.delete(self._namespaced_key(key))

    async def clear_namespace(self) -> int:
        """Clear all keys in namespace."""
        return await self.cache.clear_pattern(f"{self.namespace}:*")
