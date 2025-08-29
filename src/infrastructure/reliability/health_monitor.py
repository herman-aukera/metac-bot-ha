"""Health monitoring system for tournament-grade reliability."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check configuration."""

    name: str
    check_function: Callable[[], Any]
    timeout: float = 10.0
    interval: float = 30.0
    critical: bool = False  # If True, failure marks entire system as critical
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)


@dataclass
class HealthResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    duration: float
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class HealthMonitor:
    """
    Comprehensive health monitoring system.

    Monitors various system components and provides real-time health status
    for tournament-grade reliability and observability.
    """

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_results: Dict[str, HealthResult] = {}
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.logger = logger.bind(component="health_monitor")

        # Metrics
        self.total_checks = 0
        self.successful_checks = 0
        self.failed_checks = 0
        self.last_check_time = 0.0

        # Callbacks for status changes
        self.status_change_callbacks: List[
            Callable[[str, HealthStatus, HealthStatus], None]
        ] = []

    def register_health_check(self, health_check: HealthCheck):
        """
        Register a health check.

        Args:
            health_check: HealthCheck configuration
        """
        self.health_checks[health_check.name] = health_check
        self.logger.info(
            "Registered health check",
            name=health_check.name,
            critical=health_check.critical,
            interval=health_check.interval,
        )

    def register_callback(
        self, callback: Callable[[str, HealthStatus, HealthStatus], None]
    ):
        """
        Register callback for health status changes.

        Args:
            callback: Function called when health status changes
                     (check_name, old_status, new_status)
        """
        self.status_change_callbacks.append(callback)

    async def start_monitoring(self):
        """Start the health monitoring loop."""
        if self.running:
            self.logger.warning("Health monitor already running")
            return

        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(
            "Started health monitoring",
            check_interval=self.check_interval,
            registered_checks=len(self.health_checks),
        )

    async def stop_monitoring(self):
        """Stop the health monitoring loop."""
        if not self.running:
            return

        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped health monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._run_health_checks()
                self.last_check_time = time.time()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(self.check_interval)

    async def _run_health_checks(self):
        """Run all enabled health checks."""
        if not self.health_checks:
            return

        # Run checks concurrently
        tasks = []
        for check in self.health_checks.values():
            if check.enabled:
                task = asyncio.create_task(self._run_single_check(check))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_single_check(self, check: HealthCheck):
        """Run a single health check."""
        start_time = time.time()
        self.total_checks += 1

        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self._execute_check_function(check.check_function),
                timeout=check.timeout,
            )

            duration = time.time() - start_time

            # Determine status based on result
            if result is True or (
                isinstance(result, dict) and result.get("healthy", True)
            ):
                status = HealthStatus.HEALTHY
                message = "Check passed"
                details = result if isinstance(result, dict) else {}
                error = None
                self.successful_checks += 1
            else:
                status = HealthStatus.UNHEALTHY
                message = str(result) if result else "Check failed"
                details = result if isinstance(result, dict) else {}
                error = None
                self.failed_checks += 1

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            status = HealthStatus.UNHEALTHY
            message = f"Check timed out after {check.timeout}s"
            details = {}
            error = "timeout"
            self.failed_checks += 1

        except Exception as e:
            duration = time.time() - start_time
            status = (
                HealthStatus.UNHEALTHY if not check.critical else HealthStatus.CRITICAL
            )
            message = f"Check failed: {str(e)}"
            details = {}
            error = str(e)
            self.failed_checks += 1

        # Create health result
        health_result = HealthResult(
            name=check.name,
            status=status,
            message=message,
            duration=duration,
            timestamp=time.time(),
            details=details,
            error=error,
        )

        # Check for status change
        old_result = self.health_results.get(check.name)
        old_status = old_result.status if old_result else None

        # Store result
        self.health_results[check.name] = health_result

        # Log result
        if status == HealthStatus.HEALTHY:
            self.logger.debug("Health check passed", name=check.name, duration=duration)
        else:
            self.logger.warning(
                "Health check failed",
                name=check.name,
                status=status.value,
                message=message,
                duration=duration,
            )

        # Notify callbacks of status change
        if old_status and old_status != status:
            for callback in self.status_change_callbacks:
                try:
                    callback(check.name, old_status, status)
                except Exception as e:
                    self.logger.error(
                        "Error in status change callback",
                        callback=callback.__name__,
                        error=str(e),
                    )

    async def _execute_check_function(self, func: Callable) -> Any:
        """Execute health check function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func)

    async def run_check_now(self, check_name: str) -> Optional[HealthResult]:
        """
        Run a specific health check immediately.

        Args:
            check_name: Name of the health check to run

        Returns:
            HealthResult if check exists, None otherwise
        """
        check = self.health_checks.get(check_name)
        if not check:
            self.logger.warning("Health check not found", name=check_name)
            return None

        await self._run_single_check(check)
        return self.health_results.get(check_name)

    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns:
            Overall health status based on all checks
        """
        if not self.health_results:
            return HealthStatus.UNHEALTHY

        # Check for critical failures
        for result in self.health_results.values():
            if result.status == HealthStatus.CRITICAL:
                return HealthStatus.CRITICAL

        # Count unhealthy checks
        unhealthy_count = sum(
            1
            for result in self.health_results.values()
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        )

        total_checks = len(self.health_results)
        unhealthy_ratio = unhealthy_count / total_checks

        if unhealthy_ratio == 0:
            return HealthStatus.HEALTHY
        elif unhealthy_ratio < 0.3:  # Less than 30% unhealthy
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive health summary.

        Returns:
            Dictionary with health status and metrics
        """
        overall_status = self.get_overall_status()

        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = sum(
                1 for result in self.health_results.values() if result.status == status
            )

        return {
            "overall_status": overall_status.value,
            "total_checks": len(self.health_checks),
            "enabled_checks": sum(
                1 for check in self.health_checks.values() if check.enabled
            ),
            "status_counts": status_counts,
            "last_check_time": self.last_check_time,
            "monitoring_active": self.running,
            "metrics": {
                "total_checks_run": self.total_checks,
                "successful_checks": self.successful_checks,
                "failed_checks": self.failed_checks,
                "success_rate": self.successful_checks / max(1, self.total_checks),
            },
        }

    def get_check_results(
        self, tags: Optional[Set[str]] = None
    ) -> Dict[str, HealthResult]:
        """
        Get health check results, optionally filtered by tags.

        Args:
            tags: Optional set of tags to filter by

        Returns:
            Dictionary of health results
        """
        if not tags:
            return self.health_results.copy()

        filtered_results = {}
        for name, result in self.health_results.items():
            check = self.health_checks.get(name)
            if check and tags.intersection(check.tags):
                filtered_results[name] = result

        return filtered_results

    def get_unhealthy_checks(self) -> List[HealthResult]:
        """Get list of unhealthy checks."""
        return [
            result
            for result in self.health_results.values()
            if result.status != HealthStatus.HEALTHY
        ]

    def enable_check(self, check_name: str):
        """Enable a health check."""
        if check_name in self.health_checks:
            self.health_checks[check_name].enabled = True
            self.logger.info("Enabled health check", name=check_name)

    def disable_check(self, check_name: str):
        """Disable a health check."""
        if check_name in self.health_checks:
            self.health_checks[check_name].enabled = False
            self.logger.info("Disabled health check", name=check_name)

    def remove_check(self, check_name: str):
        """Remove a health check."""
        if check_name in self.health_checks:
            del self.health_checks[check_name]
            if check_name in self.health_results:
                del self.health_results[check_name]
            self.logger.info("Removed health check", name=check_name)


# Convenience functions for common health checks
def create_api_health_check(name: str, url: str, timeout: float = 5.0) -> HealthCheck:
    """Create health check for API endpoint."""
    import httpx

    async def check_api():
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout)
            return {
                "healthy": response.status_code < 400,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
            }

    return HealthCheck(
        name=name,
        check_function=check_api,
        timeout=timeout + 2.0,
        tags={"api", "external"},
    )


def create_database_health_check(name: str, connection_func: Callable) -> HealthCheck:
    """Create health check for database connection."""

    async def check_database():
        try:
            # This should be customized based on your database client
            result = await connection_func()
            return {"healthy": True, "connection_time": time.time()}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    return HealthCheck(
        name=name,
        check_function=check_database,
        timeout=10.0,
        critical=True,
        tags={"database", "critical"},
    )


def create_memory_health_check(
    name: str = "memory", threshold_mb: int = 1000
) -> HealthCheck:
    """Create health check for memory usage."""
    import psutil

    def check_memory():
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        return {
            "healthy": memory_mb < threshold_mb,
            "memory_mb": memory_mb,
            "threshold_mb": threshold_mb,
            "memory_percent": process.memory_percent(),
        }

    return HealthCheck(
        name=name, check_function=check_memory, timeout=5.0, tags={"system", "memory"}
    )


def create_disk_health_check(
    name: str = "disk", threshold_percent: float = 90.0
) -> HealthCheck:
    """Create health check for disk usage."""
    import psutil

    def check_disk():
        disk_usage = psutil.disk_usage("/")
        used_percent = (disk_usage.used / disk_usage.total) * 100

        return {
            "healthy": used_percent < threshold_percent,
            "used_percent": used_percent,
            "threshold_percent": threshold_percent,
            "free_gb": disk_usage.free / 1024 / 1024 / 1024,
        }

    return HealthCheck(
        name=name, check_function=check_disk, timeout=5.0, tags={"system", "disk"}
    )
