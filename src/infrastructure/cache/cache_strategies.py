"""
Cache strategies for intelligent caching behavior and optimization.
"""

import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: int = 0
    avg_access_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        self.metrics = CacheMetrics()

    @abstractmethod
    async def should_cache(self, key: str, value: Any, context: Dict[str, Any]) -> bool:
        """Determine if item should be cached."""
        pass

    @abstractmethod
    async def get_ttl(self, key: str, value: Any, context: Dict[str, Any]) -> Optional[int]:
        """Get TTL for cache item."""
        pass

    @abstractmethod
    async def should_evict(self, key: str, access_info: Dict[str, Any]) -> bool:
        """Determine if item should be evicted."""
        pass

    def update_metrics(self, operation: str, **kwargs) -> None:
        """Update cache metrics."""
        if operation == "hit":
            self.metrics.hits += 1
        elif operation == "miss":
            self.metrics.misses += 1
        elif operation == "eviction":
            self.metrics.evictions += 1


class TTLCacheStrategy(CacheStrategy):
    """Time-to-live based cache strategy."""

    def __init__(self, default_ttl: int = 3600, max_size: Optional[int] = None):
        super().__init__(max_size)
        self.default_ttl = default_ttl
        self.ttl_rules: Dict[str, int] = {}

    def add_ttl_rule(self, key_pattern: str, ttl: int) -> None:
        """Add TTL rule for key pattern."""
        self.ttl_rules[key_pattern] = ttl

    async def should_cache(self, key: str, value: Any, context: Dict[str, Any]) -> bool:
        """Always cache with TTL strategy."""
        return True

    async def get_ttl(self, key: str, value: Any, context: Dict[str, Any]) -> Optional[int]:
        """Get TTL based on key pattern rules."""
        # Check for specific TTL rules
        for pattern, ttl in self.ttl_rules.items():
            if pattern in key:
                return ttl

        # Check context for custom TTL
        if "ttl" in context:
            return context["ttl"]

        # Use default TTL
        return self.default_ttl

    async def should_evict(self, key: str, access_info: Dict[str, Any]) -> bool:
        """TTL-based eviction is handled by Redis automatically."""
        return False


class LRUCacheStrategy(CacheStrategy):
    """Least Recently Used cache strategy."""

    def __init__(self, max_size: int = 10000):
        super().__init__(max_size)
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}

    async def should_cache(self, key: str, value: Any, context: Dict[str, Any]) -> bool:
        """Cache if under size limit or can evict LRU item."""
        if self.max_size is None:
            return True

        current_size = len(self.access_times)
        return current_size < self.max_size or await self._can_evict_lru()

    async def get_ttl(self, key: str, value: Any, context: Dict[str, Any]) -> Optional[int]:
        """No TTL for LRU strategy."""
        return context.get("ttl")

    async def should_evict(self, key: str, access_info: Dict[str, Any]) -> bool:
        """Evict if over size limit and this is LRU item."""
        if self.max_size is None:
            return False

        current_size = len(self.access_times)
        if current_size <= self.max_size:
            return False

        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        return key == lru_key

    def record_access(self, key: str) -> None:
        """Record access for LRU tracking."""
        self.access_times[key] = time.time()
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

    async def _can_evict_lru(self) -> bool:
        """Check if we can evict LRU item."""
        return len(self.access_times) > 0


class AdaptiveCacheStrategy(CacheStrategy):
    """Adaptive cache strategy that adjusts based on access patterns."""

    def __init__(self, max_size: int = 10000, adaptation_interval: int = 300):
        super().__init__(max_size)
        self.adaptation_interval = adaptation_interval
        self.access_patterns: Dict[str, List[float]] = {}
        self.last_adaptation = time.time()
        self.dynamic_ttls: Dict[str, int] = {}
        self.base_ttl = 3600

    async def should_cache(self, key: str, value: Any, context: Dict[str, Any]) -> bool:
        """Adaptive caching based on access patterns."""
        # Always cache initially
        if key not in self.access_patterns:
            return True

        # Check access frequency
        recent_accesses = self._get_recent_accesses(key)
        if len(recent_accesses) >= 2:  # Frequently accessed
            return True

        # Cache based on value size and importance
        importance = context.get("importance", 0.5)
        value_size = context.get("size", 0)

        # Cache important or small items
        return importance > 0.7 or value_size < 1024

    async def get_ttl(self, key: str, value: Any, context: Dict[str, Any]) -> Optional[int]:
        """Adaptive TTL based on access patterns."""
        await self._adapt_if_needed()

        # Use dynamic TTL if available
        if key in self.dynamic_ttls:
            return self.dynamic_ttls[key]

        # Calculate TTL based on access frequency
        recent_accesses = self._get_recent_accesses(key)
        if len(recent_accesses) >= 3:
            # Frequently accessed - longer TTL
            return self.base_ttl * 2
        elif len(recent_accesses) == 0:
            # Never accessed - shorter TTL
            return self.base_ttl // 2

        return self.base_ttl

    async def should_evict(self, key: str, access_info: Dict[str, Any]) -> bool:
        """Evict based on adaptive criteria."""
        if self.max_size is None:
            return False

        # Don't evict frequently accessed items
        recent_accesses = self._get_recent_accesses(key)
        if len(recent_accesses) >= 3:
            return False

        # Evict old, infrequently accessed items
        last_access = access_info.get("last_access", 0)
        return time.time() - last_access > self.base_ttl

    def record_access(self, key: str) -> None:
        """Record access for adaptive learning."""
        current_time = time.time()

        if key not in self.access_patterns:
            self.access_patterns[key] = []

        self.access_patterns[key].append(current_time)

        # Keep only recent accesses
        cutoff_time = current_time - self.adaptation_interval * 2
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff_time
        ]

    def _get_recent_accesses(self, key: str) -> List[float]:
        """Get recent accesses for key."""
        if key not in self.access_patterns:
            return []

        cutoff_time = time.time() - self.adaptation_interval
        return [t for t in self.access_patterns[key] if t > cutoff_time]

    async def _adapt_if_needed(self) -> None:
        """Adapt cache strategy based on patterns."""
        current_time = time.time()
        if current_time - self.last_adaptation < self.adaptation_interval:
            return

        self.last_adaptation = current_time

        # Analyze access patterns and adjust TTLs
        for key, accesses in self.access_patterns.items():
            recent_accesses = [a for a in accesses if a > current_time - self.adaptation_interval]

            if len(recent_accesses) >= 5:
                # Very frequent - increase TTL
                self.dynamic_ttls[key] = self.base_ttl * 3
            elif len(recent_accesses) >= 2:
                # Moderate frequency - normal TTL
                self.dynamic_ttls[key] = self.base_ttl
            else:
                # Infrequent - decrease TTL
                self.dynamic_ttls[key] = self.base_ttl // 3

        logger.debug(f"Adapted cache strategy for {len(self.dynamic_ttls)} keys")


class TournamentCacheStrategy(AdaptiveCacheStrategy):
    """Specialized cache strategy for tournament data."""

    def __init__(self, max_size: int = 50000):
        super().__init__(max_size, adaptation_interval=180)  # 3 minutes

        # Tournament-specific TTL rules
        self.add_ttl_rule("question:", 1800)      # 30 minutes
        self.add_ttl_rule("research:", 3600)      # 1 hour
        self.add_ttl_rule("prediction:", 7200)    # 2 hours
        self.add_ttl_rule("tournament:", 300)     # 5 minutes
        self.add_ttl_rule("standings:", 60)       # 1 minute

    def add_ttl_rule(self, key_pattern: str, ttl: int) -> None:
        """Add TTL rule for key pattern."""
        if not hasattr(self, 'ttl_rules'):
            self.ttl_rules = {}
        self.ttl_rules[key_pattern] = ttl

    async def get_ttl(self, key: str, value: Any, context: Dict[str, Any]) -> Optional[int]:
        """Get TTL with tournament-specific rules."""
        # Check tournament-specific rules first
        for pattern, ttl in getattr(self, 'ttl_rules', {}).items():
            if pattern in key:
                return ttl

        # Fall back to adaptive TTL
        return await super().get_ttl(key, value, context)

    async def should_cache(self, key: str, value: Any, context: Dict[str, Any]) -> bool:
        """Tournament-specific caching decisions."""
        # Always cache tournament-critical data
        critical_patterns = ["tournament:", "standings:", "question:"]
        if any(pattern in key for pattern in critical_patterns):
            return True

        # Use adaptive strategy for other data
        return await super().should_cache(key, value, context)


class CacheStrategyFactory:
    """Factory for creating cache strategies."""

    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> CacheStrategy:
        """Create cache strategy by type."""
        if strategy_type == "ttl":
            return TTLCacheStrategy(**kwargs)
        elif strategy_type == "lru":
            return LRUCacheStrategy(**kwargs)
        elif strategy_type == "adaptive":
            return AdaptiveCacheStrategy(**kwargs)
        elif strategy_type == "tournament":
            return TournamentCacheStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown cache strategy type: {strategy_type}")

    @staticmethod
    def get_recommended_strategy(use_case: str) -> str:
        """Get recommended strategy for use case."""
        recommendations = {
            "tournament": "tournament",
            "research": "adaptive",
            "predictions": "ttl",
            "general": "adaptive"
        }
        return recommendations.get(use_case, "adaptive")
