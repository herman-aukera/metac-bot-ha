"""
Comprehensive health check manager with component-level monitoring.

Provides health monitoring for all system components with detailed
status reporting and automated health assessment.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from abc import ABC, abstractmethod

from src.infrastructure.logging.structured_logger import get_logger
from src.infrastructure.monitoring.metrics_collector import get_metrics_collector


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms
        }


@dataclass
class SystemHealthSummary:
    """Overall system health summary."""
    overall_status: HealthStatus
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    unknown_components: int
    component_results: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_status": self.overall_status.value,
            "healthy_components": self.healthy_components,
            "degraded_components": self.degraded_components,
            "unhealthy_components": self.unhealthy_components,
            "unknown_components": self.unknown_components,
            "total_components": len(self.component_results),
            "component_results": [result.to_dict() for result in self.component_results],
            "timestamp": self.timestamp.isoformat()
        }


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, component_name: str, timeout: float = 5.0):
        self.component_name = component_name
        self.timeout = timeout

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform the health check."""
        pass


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""

    def __init__(self, db_client, component_name: str = "database"):
        super().__init__(component_name)
        self.db_client = db_client

    async def check_health(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        start_time = time.time()

        try:
            # Simple connectivity test
            await asyncio.wait_for(
                self._test_connection(),
                timeout=self.timeout
            )

            duration_ms = (time.time() - start_time) * 1000

            if duration_ms > 2000:  # 2 seconds
                return HealthCheckResult(
                    component=self.component_name,
                    status=HealthStatus.DEGRADED,
                    message="Database responding slowly",
                    details={"response_time_ms": duration_ms},
                    duration_ms=duration_ms
                )

            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.HEALTHY,
                message="Database connection healthy",
                details={"response_time_ms": duration_ms},
                duration_ms=duration_ms
            )

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message="Database connection timeout",
                details={"timeout_ms": self.timeout * 1000},
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms
            )

    async def _test_connection(self):
        """Test database connection - implement based on your DB client."""
        # This is a placeholder - implement based on your actual database client
        await asyncio.sleep(0.1)  # Simulate DB query


class ExternalAPIHealthCheck(HealthCheck):
    """Health check for external API services."""

    def __init__(self, api_client, service_name: str, endpoint: str = "/health"):
        super().__init__(f"api_{service_name}")
        self.api_client = api_client
        self.service_name = service_name
        self.endpoint = endpoint

    async def check_health(self) -> HealthCheckResult:
        """Check external API health."""
        start_time = time.time()

        try:
            # Attempt to call health endpoint or make a simple request
            response = await asyncio.wait_for(
                self._make_health_request(),
                timeout=self.timeout
            )

            duration_ms = (time.time() - start_time) * 1000

            if response.get("status") == "ok" or response.get("healthy", True):
                status = HealthStatus.HEALTHY
                message = f"{self.service_name} API healthy"
            else:
                status = HealthStatus.DEGRADED
                message = f"{self.service_name} API reporting issues"

            return HealthCheckResult(
                component=self.component_name,
                status=status,
                message=message,
                details={
                    "response_time_ms": duration_ms,
                    "response": response
                },
                duration_ms=duration_ms
            )

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.service_name} API timeout",
                details={"timeout_ms": self.timeout * 1000},
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.service_name} API failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms
            )

    async def _make_health_request(self) -> Dict[str, Any]:
        """Make health request to API - implement based on your API client."""
        # This is a placeholder - implement based on your actual API client
        await asyncio.sleep(0.1)  # Simulate API call
        return {"status": "ok", "healthy": True}


class MemoryHealthCheck(HealthCheck):
    """Health check for memory usage."""

    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        super().__init__("memory")
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check_health(self) -> HealthCheckResult:
        """Check memory usage."""
        start_time = time.time()

        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100.0

            duration_ms = (time.time() - start_time) * 1000

            if usage_percent >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {usage_percent:.1%}"
            elif usage_percent >= self.warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {usage_percent:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {usage_percent:.1%}"

            return HealthCheckResult(
                component=self.component_name,
                status=status,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used
                },
                duration_ms=duration_ms
            )

        except ImportError:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNKNOWN,
                message="psutil not available for memory monitoring",
                details={"error": "psutil not installed"},
                duration_ms=0.0
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms
            )


class DiskHealthCheck(HealthCheck):
    """Health check for disk usage."""

    def __init__(self, path: str = "/", warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        super().__init__("disk")
        self.path = path
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check_health(self) -> HealthCheckResult:
        """Check disk usage."""
        start_time = time.time()

        try:
            import psutil
            disk = psutil.disk_usage(self.path)
            usage_percent = disk.used / disk.total

            duration_ms = (time.time() - start_time) * 1000

            if usage_percent >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {usage_percent:.1%}"
            elif usage_percent >= self.warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {usage_percent:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1%}"

            return HealthCheckResult(
                component=self.component_name,
                status=status,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "total_bytes": disk.total,
                    "free_bytes": disk.free,
                    "used_bytes": disk.used,
                    "path": self.path
                },
                duration_ms=duration_ms
            )

        except ImportError:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNKNOWN,
                message="psutil not available for disk monitoring",
                details={"error": "psutil not installed"},
                duration_ms=0.0
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNKNOWN,
                message=f"Disk check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms
            )


class CustomHealthCheck(HealthCheck):
    """Custom health check with user-defined logic."""

    def __init__(
        self,
        component_name: str,
        check_function: Callable[[], Awaitable[HealthCheckResult]],
        timeout: float = 5.0
    ):
        super().__init__(component_name, timeout)
        self.check_function = check_function

    async def check_health(self) -> HealthCheckResult:
        """Execute custom health check function."""
        try:
            return await asyncio.wait_for(
                self.check_function(),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message="Custom health check timeout",
                details={"timeout_ms": self.timeout * 1000}
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Custom health check failed: {str(e)}",
                details={"error": str(e)}
            )


class HealthCheckManager:
    """
    Comprehensive health check manager.

    Manages health checks for all system components and provides
    aggregated health status with detailed reporting.
    """

    def __init__(self, check_interval: float = 30.0):
        self.logger = get_logger("health_check_manager")
        self.metrics_collector = get_metrics_collector()
        self.check_interval = check_interval

        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}

        # Latest results cache
        self.latest_results: Dict[str, HealthCheckResult] = {}

        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False
        self._lock = threading.Lock()

        # Initialize default health checks
        self._initialize_default_checks()

    def _initialize_default_checks(self):
        """Initialize default system health checks."""
        # Memory check
        self.register_health_check(MemoryHealthCheck())

        # Disk check
        self.register_health_check(DiskHealthCheck())

        # Add more default checks as needed

    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        with self._lock:
            self.health_checks[health_check.component_name] = health_check
            self.logger.info(f"Registered health check for component: {health_check.component_name}")

    def unregister_health_check(self, component_name: str):
        """Unregister a health check."""
        with self._lock:
            if component_name in self.health_checks:
                del self.health_checks[component_name]
                if component_name in self.latest_results:
                    del self.latest_results[component_name]
                self.logger.info(f"Unregistered health check for component: {component_name}")

    async def check_component_health(self, component_name: str) -> Optional[HealthCheckResult]:
        """Check health of a specific component."""
        if component_name not in self.health_checks:
            return None

        health_check = self.health_checks[component_name]

        try:
            result = await health_check.check_health()

            # Update cache
            with self._lock:
                self.latest_results[component_name] = result

            # Record metrics
            health_score = self._status_to_score(result.status)
            self.metrics_collector.set_gauge(
                "system_health_score",
                health_score,
                labels={"component": component_name}
            )

            # Log result
            if result.status == HealthStatus.HEALTHY:
                self.logger.debug(f"Health check passed for {component_name}: {result.message}")
            else:
                self.logger.warning(
                    f"Health check failed for {component_name}: {result.message}",
                    extra={"health_result": result.to_dict()}
                )

            return result

        except Exception as e:
            self.logger.error(f"Health check error for {component_name}", exception=e)

            error_result = HealthCheckResult(
                component=component_name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check error: {str(e)}",
                details={"error": str(e)}
            )

            with self._lock:
                self.latest_results[component_name] = error_result

            return error_result

    async def check_all_health(self) -> SystemHealthSummary:
        """Check health of all registered components."""
        results = []

        # Run all health checks concurrently
        tasks = []
        for component_name in list(self.health_checks.keys()):
            task = asyncio.create_task(self.check_component_health(component_name))
            tasks.append(task)

        if tasks:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in completed_results:
                if isinstance(result, HealthCheckResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    self.logger.error("Health check task failed", exception=result)

        # Calculate summary
        summary = self._calculate_health_summary(results)

        # Log overall status
        self.logger.info(
            f"System health check completed: {summary.overall_status.value}",
            extra={"health_summary": summary.to_dict()}
        )

        return summary

    def get_latest_health_status(self) -> SystemHealthSummary:
        """Get latest health status from cache."""
        with self._lock:
            results = list(self.latest_results.values())

        return self._calculate_health_summary(results)

    def get_component_status(self, component_name: str) -> Optional[HealthCheckResult]:
        """Get latest status for a specific component."""
        with self._lock:
            return self.latest_results.get(component_name)

    async def start_monitoring(self):
        """Start background health monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self.logger.warning("Health monitoring already running")
            return

        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(f"Started health monitoring with {self.check_interval}s interval")

    async def stop_monitoring(self):
        """Stop background health monitoring."""
        self._stop_monitoring = True

        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped health monitoring")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                await self.check_all_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in health monitoring loop", exception=e)
                await asyncio.sleep(min(self.check_interval, 10.0))  # Shorter retry interval

    def _calculate_health_summary(self, results: List[HealthCheckResult]) -> SystemHealthSummary:
        """Calculate overall health summary from component results."""
        if not results:
            return SystemHealthSummary(
                overall_status=HealthStatus.UNKNOWN,
                healthy_components=0,
                degraded_components=0,
                unhealthy_components=0,
                unknown_components=0,
                component_results=[]
            )

        # Count components by status
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }

        for result in results:
            status_counts[result.status] += 1

        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            overall_status = HealthStatus.DEGRADED  # Treat unknown as degraded
        else:
            overall_status = HealthStatus.HEALTHY

        return SystemHealthSummary(
            overall_status=overall_status,
            healthy_components=status_counts[HealthStatus.HEALTHY],
            degraded_components=status_counts[HealthStatus.DEGRADED],
            unhealthy_components=status_counts[HealthStatus.UNHEALTHY],
            unknown_components=status_counts[HealthStatus.UNKNOWN],
            component_results=results
        )

    def _status_to_score(self, status: HealthStatus) -> float:
        """Convert health status to numeric score for metrics."""
        return {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.0,
            HealthStatus.UNKNOWN: -1.0
        }.get(status, -1.0)


# Global health check manager instance
health_check_manager = HealthCheckManager()


def get_health_check_manager() -> HealthCheckManager:
    """Get the global health check manager instance."""
    return health_check_manager


def configure_health_monitoring(
    check_interval: float = 30.0,
    auto_start: bool = True
) -> HealthCheckManager:
    """Configure and optionally start health monitoring."""
    global health_check_manager
    health_check_manager = HealthCheckManager(check_interval=check_interval)

    if auto_start:
        # Start monitoring in background
        asyncio.create_task(health_check_manager.start_monitoring())

    return health_check_manager
