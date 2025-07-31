"""
Graceful degradation manager for maintaining service during partial failures.

Provides fallback mechanisms, service health monitoring, and automatic
failover to ensure system continues operating during component failures.
"""

import asyncio
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from src.domain.exceptions import InfrastructureError


logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DegradationLevel(Enum):
    """System degradation levels."""
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class ServiceStatus:
    """Status information for a service."""

    name: str
    health: ServiceHealth = ServiceHealth.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    error_rate: float = 0.0
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Check if service is available for use."""
        return self.health in [ServiceHealth.HEALTHY, ServiceHealth.DEGRADED]

    @property
    def uptime_percentage(self) -> float:
        """Calculate uptime percentage based on recent history."""
        if self.consecutive_failures == 0:
            return 100.0

        total_checks = self.consecutive_failures + self.consecutive_successes
        if total_checks == 0:
            return 0.0

        return (self.consecutive_successes / total_checks) * 100.0


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""

    primary_services: List[str]
    fallback_services: List[str]
    fallback_strategy: str = "round_robin"  # round_robin, priority, random
    health_check_interval: int = 30  # seconds
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    enable_partial_results: bool = True


class GracefulDegradationManager:
    """
    Manages graceful degradation during service failures.

    Provides fallback mechanisms, health monitoring, and automatic
    failover to maintain system functionality during partial outages.
    """

    def __init__(self):
        self.service_status: Dict[str, ServiceStatus] = {}
        self.fallback_configs: Dict[str, FallbackConfig] = {}
        self.cached_results: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.degradation_level = DegradationLevel.NONE
        self._lock = asyncio.Lock()
        self._health_check_tasks: Dict[str, asyncio.Task] = {}

        # Default fallback configurations
        self._setup_default_configs()

        logger.info("Graceful degradation manager initialized")

    def _setup_default_configs(self):
        """Setup default fallback configurations."""
        # Search providers fallback
        self.fallback_configs["search"] = FallbackConfig(
            primary_services=["asknews", "perplexity"],
            fallback_services=["exa", "serpapi", "duckduckgo"],
            fallback_strategy="priority",
            health_check_interval=60,
            failure_threshold=2,
            recovery_threshold=3,
        )

        # LLM providers fallback
        self.fallback_configs["llm"] = FallbackConfig(
            primary_services=["openai", "anthropic"],
            fallback_services=["google", "local"],
            fallback_strategy="round_robin",
            health_check_interval=30,
            failure_threshold=3,
            recovery_threshold=2,
        )

        # Agent ensemble fallback
        self.fallback_configs["agents"] = FallbackConfig(
            primary_services=["ensemble"],
            fallback_services=["cot", "tot", "react"],
            fallback_strategy="priority",
            health_check_interval=120,
            failure_threshold=2,
            recovery_threshold=1,
            enable_partial_results=True,
        )

    async def register_service(
        self,
        name: str,
        health_check_func: Optional[Callable] = None,
        initial_health: ServiceHealth = ServiceHealth.UNKNOWN
    ):
        """Register a service for monitoring."""
        async with self._lock:
            self.service_status[name] = ServiceStatus(
                name=name,
                health=initial_health,
                last_check=datetime.utcnow()
            )

            # Start health check task if function provided
            if health_check_func:
                self._health_check_tasks[name] = asyncio.create_task(
                    self._health_check_loop(name, health_check_func)
                )

        logger.info(f"Service '{name}' registered for monitoring")

    async def update_service_health(
        self,
        name: str,
        health: ServiceHealth,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update service health status."""
        async with self._lock:
            if name not in self.service_status:
                self.service_status[name] = ServiceStatus(name=name)

            status = self.service_status[name]
            old_health = status.health
            status.health = health
            status.last_check = datetime.utcnow()

            if metadata:
                status.metadata.update(metadata)

            # Update success/failure counters
            if health == ServiceHealth.HEALTHY:
                status.last_success = datetime.utcnow()
                status.consecutive_successes += 1
                status.consecutive_failures = 0
            elif health == ServiceHealth.UNHEALTHY:
                status.last_failure = datetime.utcnow()
                status.consecutive_failures += 1
                status.consecutive_successes = 0

            # Log health changes
            if old_health != health:
                logger.info(
                    f"Service '{name}' health changed: {old_health.value} -> {health.value}",
                    extra={
                        "service": name,
                        "old_health": old_health.value,
                        "new_health": health.value,
                        "consecutive_failures": status.consecutive_failures,
                        "consecutive_successes": status.consecutive_successes,
                    }
                )

        # Update overall degradation level
        await self._update_degradation_level()

    async def get_available_services(
        self,
        service_type: str,
        include_degraded: bool = True
    ) -> List[str]:
        """Get list of available services for a given type."""
        config = self.fallback_configs.get(service_type)
        if not config:
            return []

        all_services = config.primary_services + config.fallback_services
        available = []

        async with self._lock:
            for service in all_services:
                status = self.service_status.get(service)
                if not status:
                    continue

                if status.health == ServiceHealth.HEALTHY:
                    available.append(service)
                elif include_degraded and status.health == ServiceHealth.DEGRADED:
                    available.append(service)

        return available

    async def get_primary_service(self, service_type: str) -> Optional[str]:
        """Get the primary service to use for a given type."""
        config = self.fallback_configs.get(service_type)
        if not config:
            return None

        # Try primary services first
        for service in config.primary_services:
            status = self.service_status.get(service)
            if status and status.is_available:
                return service

        # Fall back to fallback services
        return await self._select_fallback_service(service_type)

    async def _select_fallback_service(self, service_type: str) -> Optional[str]:
        """Select a fallback service based on strategy."""
        config = self.fallback_configs.get(service_type)
        if not config:
            return None

        available_fallbacks = []
        async with self._lock:
            for service in config.fallback_services:
                status = self.service_status.get(service)
                if status and status.is_available:
                    available_fallbacks.append((service, status))

        if not available_fallbacks:
            return None

        # Apply fallback strategy
        if config.fallback_strategy == "priority":
            # Return first available in order
            return available_fallbacks[0][0]

        elif config.fallback_strategy == "round_robin":
            # Simple round-robin based on current time
            import time
            index = int(time.time()) % len(available_fallbacks)
            return available_fallbacks[index][0]

        elif config.fallback_strategy == "random":
            import random
            return random.choice(available_fallbacks)[0]

        else:
            # Default to first available
            return available_fallbacks[0][0]

    async def execute_with_fallback(
        self,
        service_type: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with automatic fallback on failure."""
        config = self.fallback_configs.get(service_type)
        if not config:
            raise InfrastructureError(
                f"No fallback configuration for service type: {service_type}",
                service_name=service_type
            )

        # Try to get cached result first
        if config.enable_caching:
            cache_key = self._generate_cache_key(service_type, operation, args, kwargs)
            cached_result = await self._get_cached_result(cache_key, config.cache_ttl)
            if cached_result is not None:
                logger.debug(f"Returning cached result for {service_type}")
                return cached_result

        # Get available services in priority order
        all_services = config.primary_services + config.fallback_services
        last_exception = None

        for service in all_services:
            status = self.service_status.get(service)
            if not status or not status.is_available:
                continue

            try:
                logger.debug(f"Attempting operation with service: {service}")
                result = await operation(service, *args, **kwargs)

                # Update service health on success
                await self.update_service_health(service, ServiceHealth.HEALTHY)

                # Cache successful result
                if config.enable_caching:
                    await self._cache_result(cache_key, result)

                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Operation failed with service '{service}': {e}",
                    extra={
                        "service": service,
                        "service_type": service_type,
                        "exception": str(e),
                    }
                )

                # Update service health on failure
                await self.update_service_health(service, ServiceHealth.UNHEALTHY)

        # All services failed
        if config.enable_partial_results:
            # Try to return partial results from cache
            cache_key = self._generate_cache_key(service_type, operation, args, kwargs)
            stale_result = await self._get_cached_result(cache_key, config.cache_ttl * 2)
            if stale_result is not None:
                logger.warning(
                    f"Returning stale cached result for {service_type} due to all services failing"
                )
                return stale_result

        # No fallback available
        raise InfrastructureError(
            f"All services failed for {service_type}",
            service_name=service_type,
            cause=last_exception
        )

    async def _health_check_loop(self, service_name: str, health_check_func: Callable):
        """Continuous health check loop for a service."""
        config = None
        for cfg in self.fallback_configs.values():
            if service_name in cfg.primary_services + cfg.fallback_services:
                config = cfg
                break

        if not config:
            return

        while True:
            try:
                await asyncio.sleep(config.health_check_interval)

                # Perform health check
                is_healthy = await health_check_func()
                health = ServiceHealth.HEALTHY if is_healthy else ServiceHealth.UNHEALTHY

                await self.update_service_health(service_name, health)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Health check failed for service '{service_name}': {e}",
                    extra={"service": service_name, "exception": str(e)}
                )
                await self.update_service_health(service_name, ServiceHealth.UNHEALTHY)

    async def _update_degradation_level(self):
        """Update overall system degradation level."""
        async with self._lock:
            healthy_count = 0
            degraded_count = 0
            unhealthy_count = 0
            total_count = len(self.service_status)

            for status in self.service_status.values():
                if status.health == ServiceHealth.HEALTHY:
                    healthy_count += 1
                elif status.health == ServiceHealth.DEGRADED:
                    degraded_count += 1
                elif status.health == ServiceHealth.UNHEALTHY:
                    unhealthy_count += 1

            if total_count == 0:
                new_level = DegradationLevel.UNKNOWN
            elif unhealthy_count == 0:
                new_level = DegradationLevel.NONE
            elif unhealthy_count / total_count < 0.2:
                new_level = DegradationLevel.MINIMAL
            elif unhealthy_count / total_count < 0.4:
                new_level = DegradationLevel.MODERATE
            elif unhealthy_count / total_count < 0.7:
                new_level = DegradationLevel.SEVERE
            else:
                new_level = DegradationLevel.CRITICAL

            if new_level != self.degradation_level:
                old_level = self.degradation_level
                self.degradation_level = new_level

                logger.warning(
                    f"System degradation level changed: {old_level.value} -> {new_level.value}",
                    extra={
                        "old_level": old_level.value,
                        "new_level": new_level.value,
                        "healthy_services": healthy_count,
                        "degraded_services": degraded_count,
                        "unhealthy_services": unhealthy_count,
                        "total_services": total_count,
                    }
                )

    def _generate_cache_key(
        self,
        service_type: str,
        operation: Callable,
        args: tuple,
        kwargs: dict
    ) -> str:
        """Generate cache key for operation."""
        import hashlib

        key_parts = [
            service_type,
            operation.__name__ if hasattr(operation, '__name__') else str(operation),
            str(args),
            str(sorted(kwargs.items()))
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _get_cached_result(self, cache_key: str, ttl: int) -> Optional[Any]:
        """Get cached result if still valid."""
        if cache_key not in self.cached_results:
            return None

        timestamp = self.cache_timestamps.get(cache_key)
        if not timestamp:
            return None

        if datetime.utcnow() - timestamp > timedelta(seconds=ttl):
            # Cache expired
            del self.cached_results[cache_key]
            del self.cache_timestamps[cache_key]
            return None

        return self.cached_results[cache_key]

    async def _cache_result(self, cache_key: str, result: Any):
        """Cache operation result."""
        self.cached_results[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.utcnow()

        # Simple cache cleanup - remove oldest entries if cache gets too large
        if len(self.cached_results) > 1000:
            # Remove oldest 100 entries
            sorted_keys = sorted(
                self.cache_timestamps.keys(),
                key=lambda k: self.cache_timestamps[k]
            )

            for key in sorted_keys[:100]:
                del self.cached_results[key]
                del self.cache_timestamps[key]

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and metrics."""
        service_summary = {}
        for name, status in self.service_status.items():
            service_summary[name] = {
                "health": status.health.value,
                "consecutive_failures": status.consecutive_failures,
                "consecutive_successes": status.consecutive_successes,
                "uptime_percentage": status.uptime_percentage,
                "last_check": status.last_check.isoformat() if status.last_check else None,
                "response_time": status.response_time,
            }

        return {
            "degradation_level": self.degradation_level.value,
            "services": service_summary,
            "cache_size": len(self.cached_results),
            "active_health_checks": len(self._health_check_tasks),
        }

    async def cleanup(self):
        """Cleanup resources and stop health check tasks."""
        # Cancel all health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._health_check_tasks:
            await asyncio.gather(
                *self._health_check_tasks.values(),
                return_exceptions=True
            )

        self._health_check_tasks.clear()
        logger.info("Graceful degradation manager cleaned up")
