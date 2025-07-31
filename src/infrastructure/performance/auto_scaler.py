"""
Auto-scaling mechanisms based on load and performance metrics.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import psutil
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ResourceType(Enum):
    """Resource types for monitoring."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    CUSTOM = "custom"


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    network_io: Tuple[int, int]  # bytes_sent, bytes_recv
    disk_io: Tuple[int, int]     # read_bytes, write_bytes
    custom_metrics: Dict[str, float]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'network_bytes_sent': self.network_io[0],
            'network_bytes_recv': self.network_io[1],
            'disk_read_bytes': self.disk_io[0],
            'disk_write_bytes': self.disk_io[1],
            'custom_metrics': self.custom_metrics,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ScalingPolicy:
    """Scaling policy configuration."""
    name: str
    resource_type: ResourceType
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int = 1
    max_instances: int = 10
    cooldown_period: int = 300  # seconds
    evaluation_period: int = 60  # seconds
    evaluation_points: int = 3   # number of data points to evaluate
    scale_up_adjustment: int = 1
    scale_down_adjustment: int = 1
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: datetime
    policy_name: str
    direction: ScalingDirection
    old_instances: int
    new_instances: int
    trigger_value: float
    reason: str


class ResourceMonitor:
    """Monitors system resources for auto-scaling decisions."""

    def __init__(self, monitoring_interval: int = 10):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Metrics history
        self.metrics_history: deque[ResourceMetrics] = deque(maxlen=1000)
        self.custom_metric_collectors: Dict[str, Callable[[], float]] = {}

        # Network and disk baseline
        self._last_network_io: Optional[Tuple[int, int]] = None
        self._last_disk_io: Optional[Tuple[int, int]] = None

    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started resource monitoring")

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped resource monitoring")

    async def _monitoring_loop(self) -> None:
        """Resource monitoring loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.monitoring_interval)

                # Collect metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)

                # Log metrics periodically
                if len(self.metrics_history) % 30 == 0:  # Every 5 minutes at 10s intervals
                    logger.info(
                        f"Resource usage - CPU: {metrics.cpu_percent:.1f}%, "
                        f"Memory: {metrics.memory_percent:.1f}%"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")

    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Network I/O
        network_io = psutil.net_io_counters()
        if self._last_network_io:
            network_delta = (
                network_io.bytes_sent - self._last_network_io[0],
                network_io.bytes_recv - self._last_network_io[1]
            )
        else:
            network_delta = (0, 0)
        self._last_network_io = (network_io.bytes_sent, network_io.bytes_recv)

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io and self._last_disk_io:
            disk_delta = (
                disk_io.read_bytes - self._last_disk_io[0],
                disk_io.write_bytes - self._last_disk_io[1]
            )
        else:
            disk_delta = (0, 0)
        if disk_io:
            self._last_disk_io = (disk_io.read_bytes, disk_io.write_bytes)

        # Custom metrics
        custom_metrics = {}
        for name, collector in self.custom_metric_collectors.items():
            try:
                custom_metrics[name] = collector()
            except Exception as e:
                logger.error(f"Error collecting custom metric '{name}': {e}")
                custom_metrics[name] = 0.0

        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            network_io=network_delta,
            disk_io=disk_delta,
            custom_metrics=custom_metrics,
            timestamp=datetime.now()
        )

    def add_custom_metric_collector(self, name: str, collector: Callable[[], float]) -> None:
        """Add custom metric collector."""
        self.custom_metric_collectors[name] = collector
        logger.info(f"Added custom metric collector: {name}")

    def get_recent_metrics(self, duration_seconds: int = 300) -> List[ResourceMetrics]:
        """Get recent metrics within duration."""
        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]

    def get_average_metrics(self, duration_seconds: int = 300) -> Optional[ResourceMetrics]:
        """Get average metrics over duration."""
        recent_metrics = self.get_recent_metrics(duration_seconds)
        if not recent_metrics:
            return None

        # Calculate averages
        avg_cpu = statistics.mean(m.cpu_percent for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_percent for m in recent_metrics)

        # Network and disk are cumulative, so we take the latest values
        latest_network = recent_metrics[-1].network_io
        latest_disk = recent_metrics[-1].disk_io

        # Average custom metrics
        avg_custom = {}
        for name in self.custom_metric_collectors.keys():
            values = [m.custom_metrics.get(name, 0.0) for m in recent_metrics]
            avg_custom[name] = statistics.mean(values)

        return ResourceMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            network_io=latest_network,
            disk_io=latest_disk,
            custom_metrics=avg_custom,
            timestamp=datetime.now()
        )


class AutoScaler:
    """Auto-scaling system based on resource metrics and policies."""

    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.policies: Dict[str, ScalingPolicy] = {}
        self.scaling_events: deque[ScalingEvent] = deque(maxlen=1000)

        # Current instance counts by policy
        self.current_instances: Dict[str, int] = {}

        # Scaling callbacks
        self.scale_up_callbacks: Dict[str, Callable[[int], None]] = {}
        self.scale_down_callbacks: Dict[str, Callable[[int], None]] = {}

        # Auto-scaling state
        self.is_running = False
        self.scaling_task: Optional[asyncio.Task] = None
        self.evaluation_interval = 30  # seconds

    def add_policy(self, policy: ScalingPolicy) -> None:
        """Add scaling policy."""
        self.policies[policy.name] = policy
        self.current_instances[policy.name] = policy.min_instances
        logger.info(f"Added scaling policy: {policy.name}")

    def set_scale_callbacks(
        self,
        policy_name: str,
        scale_up_callback: Callable[[int], None],
        scale_down_callback: Callable[[int], None]
    ) -> None:
        """Set scaling callbacks for policy."""
        self.scale_up_callbacks[policy_name] = scale_up_callback
        self.scale_down_callbacks[policy_name] = scale_down_callback
        logger.info(f"Set scaling callbacks for policy: {policy_name}")

    async def start_auto_scaling(self) -> None:
        """Start auto-scaling system."""
        if self.is_running:
            return

        # Start resource monitoring if not already running
        if not self.resource_monitor.is_monitoring:
            await self.resource_monitor.start_monitoring()

        self.is_running = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Started auto-scaling system")

    async def stop_auto_scaling(self) -> None:
        """Stop auto-scaling system."""
        if not self.is_running:
            return

        self.is_running = False

        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped auto-scaling system")

    async def _scaling_loop(self) -> None:
        """Main scaling evaluation loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.evaluation_interval)

                # Evaluate all policies
                for policy_name, policy in self.policies.items():
                    if policy.enabled:
                        await self._evaluate_policy(policy)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")

    async def _evaluate_policy(self, policy: ScalingPolicy) -> None:
        """Evaluate scaling policy and take action if needed."""
        # Check cooldown period
        if not self._is_cooldown_expired(policy):
            return

        # Get metrics for evaluation
        metrics = self.resource_monitor.get_average_metrics(policy.evaluation_period)
        if not metrics:
            logger.warning(f"No metrics available for policy: {policy.name}")
            return

        # Get metric value based on resource type
        metric_value = self._get_metric_value(metrics, policy.resource_type, policy.name)
        if metric_value is None:
            return

        # Determine scaling direction
        scaling_direction = self._determine_scaling_direction(policy, metric_value)

        if scaling_direction != ScalingDirection.NONE:
            await self._execute_scaling(policy, scaling_direction, metric_value)

    def _is_cooldown_expired(self, policy: ScalingPolicy) -> bool:
        """Check if cooldown period has expired for policy."""
        # Find last scaling event for this policy
        last_event = None
        for event in reversed(self.scaling_events):
            if event.policy_name == policy.name:
                last_event = event
                break

        if not last_event:
            return True

        time_since_last = (datetime.now() - last_event.timestamp).total_seconds()
        return time_since_last >= policy.cooldown_period

    def _get_metric_value(
        self,
        metrics: ResourceMetrics,
        resource_type: ResourceType,
        policy_name: str
    ) -> Optional[float]:
        """Get metric value for resource type."""
        if resource_type == ResourceType.CPU:
            return metrics.cpu_percent
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_percent
        elif resource_type == ResourceType.NETWORK:
            # Use total network I/O rate
            return sum(metrics.network_io) / 1024 / 1024  # MB/s
        elif resource_type == ResourceType.DISK:
            # Use total disk I/O rate
            return sum(metrics.disk_io) / 1024 / 1024  # MB/s
        elif resource_type == ResourceType.CUSTOM:
            # Use custom metric with same name as policy
            return metrics.custom_metrics.get(policy_name)

        return None

    def _determine_scaling_direction(
        self,
        policy: ScalingPolicy,
        metric_value: float
    ) -> ScalingDirection:
        """Determine if scaling is needed."""
        current_instances = self.current_instances[policy.name]

        # Check scale up conditions
        if (metric_value > policy.scale_up_threshold and
            current_instances < policy.max_instances):
            return ScalingDirection.UP

        # Check scale down conditions
        if (metric_value < policy.scale_down_threshold and
            current_instances > policy.min_instances):
            return ScalingDirection.DOWN

        return ScalingDirection.NONE

    async def _execute_scaling(
        self,
        policy: ScalingPolicy,
        direction: ScalingDirection,
        trigger_value: float
    ) -> None:
        """Execute scaling action."""
        current_instances = self.current_instances[policy.name]

        if direction == ScalingDirection.UP:
            new_instances = min(
                current_instances + policy.scale_up_adjustment,
                policy.max_instances
            )
            callback = self.scale_up_callbacks.get(policy.name)
        else:  # ScalingDirection.DOWN
            new_instances = max(
                current_instances - policy.scale_down_adjustment,
                policy.min_instances
            )
            callback = self.scale_down_callbacks.get(policy.name)

        if new_instances == current_instances:
            return  # No change needed

        # Execute scaling callback
        if callback:
            try:
                callback(new_instances)

                # Update instance count
                self.current_instances[policy.name] = new_instances

                # Record scaling event
                event = ScalingEvent(
                    timestamp=datetime.now(),
                    policy_name=policy.name,
                    direction=direction,
                    old_instances=current_instances,
                    new_instances=new_instances,
                    trigger_value=trigger_value,
                    reason=f"{policy.resource_type.value} {direction.value} threshold"
                )
                self.scaling_events.append(event)

                logger.info(
                    f"Scaled {direction.value} policy '{policy.name}': "
                    f"{current_instances} -> {new_instances} instances "
                    f"(trigger: {trigger_value:.1f})"
                )

            except Exception as e:
                logger.error(f"Error executing scaling for policy '{policy.name}': {e}")
        else:
            logger.warning(f"No scaling callback configured for policy: {policy.name}")

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        stats = {
            'policies': {},
            'current_instances': dict(self.current_instances),
            'recent_events': [],
            'is_running': self.is_running
        }

        # Policy stats
        for name, policy in self.policies.items():
            stats['policies'][name] = {
                'enabled': policy.enabled,
                'resource_type': policy.resource_type.value,
                'scale_up_threshold': policy.scale_up_threshold,
                'scale_down_threshold': policy.scale_down_threshold,
                'min_instances': policy.min_instances,
                'max_instances': policy.max_instances,
                'current_instances': self.current_instances.get(name, 0)
            }

        # Recent events (last 10)
        recent_events = list(self.scaling_events)[-10:]
        for event in recent_events:
            stats['recent_events'].append({
                'timestamp': event.timestamp.isoformat(),
                'policy_name': event.policy_name,
                'direction': event.direction.value,
                'old_instances': event.old_instances,
                'new_instances': event.new_instances,
                'trigger_value': event.trigger_value,
                'reason': event.reason
            })

        return stats

    def manually_scale(self, policy_name: str, instances: int) -> bool:
        """Manually scale a policy to specific instance count."""
        if policy_name not in self.policies:
            logger.error(f"Policy not found: {policy_name}")
            return False

        policy = self.policies[policy_name]

        # Validate instance count
        if instances < policy.min_instances or instances > policy.max_instances:
            logger.error(
                f"Instance count {instances} outside bounds "
                f"[{policy.min_instances}, {policy.max_instances}]"
            )
            return False

        current_instances = self.current_instances[policy_name]
        if instances == current_instances:
            return True  # No change needed

        # Determine direction and callback
        if instances > current_instances:
            callback = self.scale_up_callbacks.get(policy_name)
            direction = ScalingDirection.UP
        else:
            callback = self.scale_down_callbacks.get(policy_name)
            direction = ScalingDirection.DOWN

        if not callback:
            logger.error(f"No scaling callback for policy: {policy_name}")
            return False

        try:
            callback(instances)

            # Update instance count
            self.current_instances[policy_name] = instances

            # Record manual scaling event
            event = ScalingEvent(
                timestamp=datetime.now(),
                policy_name=policy_name,
                direction=direction,
                old_instances=current_instances,
                new_instances=instances,
                trigger_value=0.0,
                reason="Manual scaling"
            )
            self.scaling_events.append(event)

            logger.info(f"Manually scaled policy '{policy_name}': {current_instances} -> {instances}")
            return True

        except Exception as e:
            logger.error(f"Error in manual scaling for policy '{policy_name}': {e}")
            return False
