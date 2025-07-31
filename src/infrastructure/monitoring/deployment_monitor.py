"""
Deployment monitoring system with automated rollback triggers.
Monitors deployment health and performance metrics to detect issues.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status states"""
    DEPLOYING = "deploying"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


class MetricType(Enum):
    """Types of metrics to monitor"""
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    SUCCESS_RATE = "success_rate"


@dataclass
class MetricThreshold:
    """Threshold configuration for a metric"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    duration_seconds: int = 60
    comparison: str = "greater_than"  # greater_than, less_than, equals


@dataclass
class DeploymentMetrics:
    """Current deployment metrics"""
    timestamp: datetime
    error_rate: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    throughput: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    success_rate: float = 100.0
    active_connections: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    endpoint: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DeploymentMonitor:
    """
    Monitors deployment health and triggers rollbacks when necessary.
    Integrates with Kubernetes and monitoring systems.
    """

    def __init__(self,
                 namespace: str = "tournament-optimization",
                 deployment_name: str = "tournament-optimization",
                 monitoring_interval: int = 30,
                 rollback_callback: Optional[Callable] = None):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.monitoring_interval = monitoring_interval
        self.rollback_callback = rollback_callback

        # Monitoring state
        self.current_status = DeploymentStatus.HEALTHY
        self.metrics_history: List[DeploymentMetrics] = []
        self.health_check_history: List[HealthCheckResult] = []
        self.alert_history: List[Dict[str, Any]] = []

        # Configuration
        self.thresholds = self._get_default_thresholds()
        self.health_check_endpoints = [
            "/health",
            "/ready",
            "/metrics"
        ]

        # Rollback configuration
        self.rollback_enabled = True
        self.rollback_threshold_breaches = 3
        self.rollback_cooldown_minutes = 15
        self.last_rollback_time: Optional[datetime] = None

        # Monitoring tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

    def _get_default_thresholds(self) -> List[MetricThreshold]:
        """Get default metric thresholds"""
        return [
            MetricThreshold(
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=5.0,
                critical_threshold=10.0,
                duration_seconds=120,
                comparison="greater_than"
            ),
            MetricThreshold(
                metric_type=MetricType.RESPONSE_TIME,
                warning_threshold=2000.0,  # 2 seconds
                critical_threshold=5000.0,  # 5 seconds
                duration_seconds=180,
                comparison="greater_than"
            ),
            MetricThreshold(
                metric_type=MetricType.SUCCESS_RATE,
                warning_threshold=95.0,
                critical_threshold=90.0,
                duration_seconds=120,
                comparison="less_than"
            ),
            MetricThreshold(
                metric_type=MetricType.CPU_USAGE,
                warning_threshold=80.0,
                critical_threshold=95.0,
                duration_seconds=300,
                comparison="greater_than"
            ),
            MetricThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                warning_threshold=85.0,
                critical_threshold=95.0,
                duration_seconds=300,
                comparison="greater_than"
            )
        ]

    async def start_monitoring(self) -> None:
        """Start the deployment monitoring process"""
        if self._running:
            logger.warning("Deployment monitoring is already running")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started deployment monitoring for {self.deployment_name}")

    async def stop_monitoring(self) -> None:
        """Stop the deployment monitoring process"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped deployment monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)

                # Perform health checks
                health_results = await self._perform_health_checks()
                self.health_check_history.extend(health_results)

                # Evaluate thresholds
                alerts = self._evaluate_thresholds(metrics)

                # Check for rollback conditions
                if alerts and self._should_trigger_rollback(alerts):
                    await self._trigger_rollback(alerts)

                # Clean up old data
                self._cleanup_history()

                # Log status
                self._log_monitoring_status(metrics, health_results, alerts)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            await asyncio.sleep(self.monitoring_interval)

    async def _collect_metrics(self) -> DeploymentMetrics:
        """Collect current deployment metrics"""
        try:
            # In a real implementation, this would integrate with Prometheus,
            # Kubernetes metrics API, or other monitoring systems

            # Simulate metric collection
            current_time = datetime.utcnow()

            # Mock metrics - replace with actual metric collection
            metrics = DeploymentMetrics(
                timestamp=current_time,
                error_rate=self._get_mock_error_rate(),
                response_time_p95=self._get_mock_response_time(),
                response_time_p99=self._get_mock_response_time() * 1.5,
                throughput=self._get_mock_throughput(),
                cpu_usage=self._get_mock_cpu_usage(),
                memory_usage=self._get_mock_memory_usage(),
                success_rate=100.0 - self._get_mock_error_rate(),
                active_connections=self._get_mock_active_connections()
            )

            return metrics

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return DeploymentMetrics(timestamp=datetime.utcnow())

    def _get_mock_error_rate(self) -> float:
        """Mock error rate - replace with actual implementation"""
        import random
        return random.uniform(0, 2)  # 0-2% error rate

    def _get_mock_response_time(self) -> float:
        """Mock response time - replace with actual implementation"""
        import random
        return random.uniform(100, 1000)  # 100-1000ms

    def _get_mock_throughput(self) -> float:
        """Mock throughput - replace with actual implementation"""
        import random
        return random.uniform(50, 200)  # 50-200 requests/second

    def _get_mock_cpu_usage(self) -> float:
        """Mock CPU usage - replace with actual implementation"""
        import random
        return random.uniform(20, 60)  # 20-60% CPU

    def _get_mock_memory_usage(self) -> float:
        """Mock memory usage - replace with actual implementation"""
        import random
        return random.uniform(30, 70)  # 30-70% memory

    def _get_mock_active_connections(self) -> int:
        """Mock active connections - replace with actual implementation"""
        import random
        return random.randint(10, 100)

    async def _perform_health_checks(self) -> List[HealthCheckResult]:
        """Perform health checks on deployment endpoints"""
        results = []

        for endpoint in self.health_check_endpoints:
            try:
                # In a real implementation, this would make HTTP requests
                # to the actual service endpoints

                # Mock health check
                import random
                success = random.random() > 0.05  # 95% success rate
                status_code = 200 if success else random.choice([500, 503, 504])
                response_time = random.uniform(10, 100)

                result = HealthCheckResult(
                    endpoint=endpoint,
                    status_code=status_code,
                    response_time=response_time,
                    success=success,
                    error_message=None if success else f"HTTP {status_code}"
                )
                results.append(result)

            except Exception as e:
                result = HealthCheckResult(
                    endpoint=endpoint,
                    status_code=0,
                    response_time=0,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)

        return results

    def _evaluate_thresholds(self, metrics: DeploymentMetrics) -> List[Dict[str, Any]]:
        """Evaluate metric thresholds and generate alerts"""
        alerts = []

        for threshold in self.thresholds:
            metric_value = self._get_metric_value(metrics, threshold.metric_type)
            if metric_value is None:
                continue

            # Check if threshold is breached
            breached = self._is_threshold_breached(metric_value, threshold)

            if breached:
                # Check if breach has persisted for required duration
                if self._is_sustained_breach(threshold):
                    severity = "critical" if self._is_critical_breach(metric_value, threshold) else "warning"

                    alert = {
                        "timestamp": datetime.utcnow(),
                        "metric_type": threshold.metric_type.value,
                        "metric_value": metric_value,
                        "threshold": threshold.critical_threshold if severity == "critical" else threshold.warning_threshold,
                        "severity": severity,
                        "duration_seconds": threshold.duration_seconds,
                        "deployment": self.deployment_name
                    }
                    alerts.append(alert)

        return alerts

    def _get_metric_value(self, metrics: DeploymentMetrics, metric_type: MetricType) -> Optional[float]:
        """Get metric value by type"""
        metric_map = {
            MetricType.ERROR_RATE: metrics.error_rate,
            MetricType.RESPONSE_TIME: metrics.response_time_p95,
            MetricType.THROUGHPUT: metrics.throughput,
            MetricType.CPU_USAGE: metrics.cpu_usage,
            MetricType.MEMORY_USAGE: metrics.memory_usage,
            MetricType.SUCCESS_RATE: metrics.success_rate
        }
        return metric_map.get(metric_type)

    def _is_threshold_breached(self, value: float, threshold: MetricThreshold) -> bool:
        """Check if a threshold is breached"""
        if threshold.comparison == "greater_than":
            return value > threshold.warning_threshold
        elif threshold.comparison == "less_than":
            return value < threshold.warning_threshold
        elif threshold.comparison == "equals":
            return abs(value - threshold.warning_threshold) < 0.001
        return False

    def _is_critical_breach(self, value: float, threshold: MetricThreshold) -> bool:
        """Check if a critical threshold is breached"""
        if threshold.comparison == "greater_than":
            return value > threshold.critical_threshold
        elif threshold.comparison == "less_than":
            return value < threshold.critical_threshold
        elif threshold.comparison == "equals":
            return abs(value - threshold.critical_threshold) < 0.001
        return False

    def _is_sustained_breach(self, threshold: MetricThreshold) -> bool:
        """Check if threshold breach has been sustained for required duration"""
        # In a real implementation, this would check historical data
        # For now, we'll assume breaches are sustained
        return True

    def _should_trigger_rollback(self, alerts: List[Dict[str, Any]]) -> bool:
        """Determine if rollback should be triggered based on alerts"""
        if not self.rollback_enabled:
            return False

        # Check rollback cooldown
        if self.last_rollback_time:
            cooldown_end = self.last_rollback_time + timedelta(minutes=self.rollback_cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                logger.info("Rollback in cooldown period, skipping")
                return False

        # Count critical alerts
        critical_alerts = [alert for alert in alerts if alert["severity"] == "critical"]

        # Trigger rollback if we have enough critical alerts
        return len(critical_alerts) >= self.rollback_threshold_breaches

    async def _trigger_rollback(self, alerts: List[Dict[str, Any]]) -> None:
        """Trigger deployment rollback"""
        logger.critical(f"Triggering rollback for deployment {self.deployment_name}")
        logger.critical(f"Rollback triggered by alerts: {json.dumps(alerts, indent=2, default=str)}")

        self.current_status = DeploymentStatus.ROLLING_BACK
        self.last_rollback_time = datetime.utcnow()

        try:
            if self.rollback_callback:
                await self.rollback_callback(self.deployment_name, alerts)
            else:
                # Default rollback implementation
                await self._perform_default_rollback()

            logger.info("Rollback completed successfully")
            self.current_status = DeploymentStatus.HEALTHY

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self.current_status = DeploymentStatus.FAILED

    async def _perform_default_rollback(self) -> None:
        """Perform default rollback using kubectl"""
        try:
            import subprocess

            # Rollback deployment
            cmd = [
                "kubectl", "rollout", "undo",
                f"deployment/{self.deployment_name}",
                f"--namespace={self.namespace}"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                raise Exception(f"kubectl rollback failed: {result.stderr}")

            # Wait for rollback to complete
            cmd = [
                "kubectl", "rollout", "status",
                f"deployment/{self.deployment_name}",
                f"--namespace={self.namespace}",
                "--timeout=300s"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise Exception(f"Rollback status check failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise Exception("Rollback operation timed out")
        except Exception as e:
            raise Exception(f"Default rollback failed: {e}")

    def _cleanup_history(self) -> None:
        """Clean up old monitoring data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)

        # Keep only last 24 hours of data
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]

        self.health_check_history = [
            h for h in self.health_check_history
            if h.timestamp > cutoff_time
        ]

        self.alert_history = [
            a for a in self.alert_history
            if a["timestamp"] > cutoff_time
        ]

    def _log_monitoring_status(self,
                              metrics: DeploymentMetrics,
                              health_results: List[HealthCheckResult],
                              alerts: List[Dict[str, Any]]) -> None:
        """Log current monitoring status"""
        healthy_checks = sum(1 for h in health_results if h.success)
        total_checks = len(health_results)

        logger.info(
            f"Deployment {self.deployment_name} status: {self.current_status.value} | "
            f"Error rate: {metrics.error_rate:.2f}% | "
            f"Response time P95: {metrics.response_time_p95:.0f}ms | "
            f"CPU: {metrics.cpu_usage:.1f}% | "
            f"Memory: {metrics.memory_usage:.1f}% | "
            f"Health checks: {healthy_checks}/{total_checks} | "
            f"Alerts: {len(alerts)}"
        )

        if alerts:
            for alert in alerts:
                logger.warning(f"Alert: {alert}")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        recent_health_checks = [
            h for h in self.health_check_history
            if h.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]

        return {
            "deployment_name": self.deployment_name,
            "status": self.current_status.value,
            "last_updated": datetime.utcnow(),
            "metrics": {
                "error_rate": latest_metrics.error_rate if latest_metrics else 0,
                "response_time_p95": latest_metrics.response_time_p95 if latest_metrics else 0,
                "cpu_usage": latest_metrics.cpu_usage if latest_metrics else 0,
                "memory_usage": latest_metrics.memory_usage if latest_metrics else 0,
                "success_rate": latest_metrics.success_rate if latest_metrics else 100
            } if latest_metrics else {},
            "health_checks": {
                "total": len(recent_health_checks),
                "successful": sum(1 for h in recent_health_checks if h.success),
                "failed": sum(1 for h in recent_health_checks if not h.success)
            },
            "rollback_info": {
                "enabled": self.rollback_enabled,
                "last_rollback": self.last_rollback_time,
                "cooldown_until": (
                    self.last_rollback_time + timedelta(minutes=self.rollback_cooldown_minutes)
                    if self.last_rollback_time else None
                )
            }
        }
