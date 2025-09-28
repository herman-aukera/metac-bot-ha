"""Auto-scaling infrastructure for tournament load management."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""

    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingTrigger(Enum):
    """Scaling trigger types."""

    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    REQUEST_RATE = "request_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling behavior."""

    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int = 1
    max_instances: int = 10
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    scale_up_step: int = 1
    scale_down_step: int = 1
    evaluation_periods: int = 2  # Number of periods before scaling
    enabled: bool = True


@dataclass
class ScalingMetric:
    """Metric data for scaling decisions."""

    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingEvent:
    """Record of a scaling event."""

    timestamp: float
    direction: ScalingDirection
    trigger: str
    old_instances: int
    new_instances: int
    metric_value: float
    reason: str


class AutoScaler:
    """
    Auto-scaling system for tournament load management.

    Automatically scales resources based on various metrics to handle
    tournament load spikes and optimize resource utilization.
    """

    def __init__(self, name: str):
        self.name = name
        self.policies: Dict[str, ScalingPolicy] = {}
        self.current_instances = 1
        self.target_instances = 1
        self.last_scale_time = 0.0
        self.last_scale_direction = ScalingDirection.NONE
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.logger = logger.bind(auto_scaler=name)

        # Metrics storage
        self.metrics_history: Dict[str, List[ScalingMetric]] = {}
        self.scaling_events: List[ScalingEvent] = []

        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], Any]] = None
        self.scale_down_callback: Optional[Callable[[int], Any]] = None
        self.metric_collectors: Dict[str, Callable[[], float]] = {}

        # Configuration
        self.evaluation_interval = 60.0  # Check every minute
        self.metrics_retention_hours = 24

    def add_scaling_policy(self, policy: ScalingPolicy):
        """
        Add a scaling policy.

        Args:
            policy: ScalingPolicy configuration
        """
        self.policies[policy.name] = policy
        self.logger.info(
            "Added scaling policy",
            policy_name=policy.name,
            trigger=policy.trigger.value,
            scale_up_threshold=policy.scale_up_threshold,
            scale_down_threshold=policy.scale_down_threshold,
        )

    def set_scale_callbacks(
        self,
        scale_up_callback: Callable[[int], Any],
        scale_down_callback: Callable[[int], Any],
    ):
        """
        Set callbacks for scaling operations.

        Args:
            scale_up_callback: Function called when scaling up (new_instance_count)
            scale_down_callback: Function called when scaling down (new_instance_count)
        """
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback

    def add_metric_collector(self, metric_name: str, collector: Callable[[], float]):
        """
        Add a metric collector function.

        Args:
            metric_name: Name of the metric
            collector: Function that returns current metric value
        """
        self.metric_collectors[metric_name] = collector
        self.logger.info("Added metric collector", metric_name=metric_name)

    async def start_monitoring(self):
        """Start the auto-scaling monitoring loop."""
        if self.running:
            self.logger.warning("Auto-scaler already running")
            return

        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(
            "Started auto-scaling monitoring",
            evaluation_interval=self.evaluation_interval,
            policies=len(self.policies),
        )

    async def stop_monitoring(self):
        """Stop the auto-scaling monitoring loop."""
        if not self.running:
            return

        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped auto-scaling monitoring")

    async def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while self.running:
            try:
                await self._collect_metrics()
                await self._evaluate_scaling_policies()
                await self._cleanup_old_metrics()
                await asyncio.sleep(self.evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in auto-scaling loop", error=str(e))
                await asyncio.sleep(self.evaluation_interval)

    async def _collect_metrics(self):
        """Collect metrics from registered collectors."""
        current_time = time.time()

        for metric_name, collector in self.metric_collectors.items():
            try:
                value = await self._execute_collector(collector)
                metric = ScalingMetric(
                    name=metric_name, value=value, timestamp=current_time
                )

                if metric_name not in self.metrics_history:
                    self.metrics_history[metric_name] = []

                self.metrics_history[metric_name].append(metric)

                self.logger.debug(
                    "Collected metric", metric_name=metric_name, value=value
                )

            except Exception as e:
                self.logger.error(
                    "Error collecting metric", metric_name=metric_name, error=str(e)
                )

    async def _execute_collector(self, collector: Callable) -> float:
        """Execute metric collector, handling both sync and async."""
        if asyncio.iscoroutinefunction(collector):
            return await collector()
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, collector)

    async def _evaluate_scaling_policies(self):
        """Evaluate all scaling policies and make scaling decisions."""
        current_time = time.time()

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            try:
                scaling_decision = await self._evaluate_policy(policy, current_time)
                if scaling_decision != ScalingDirection.NONE:
                    await self._execute_scaling(policy, scaling_decision, current_time)
            except Exception as e:
                self.logger.error(
                    "Error evaluating scaling policy",
                    policy_name=policy.name,
                    error=str(e),
                )

    async def _evaluate_policy(
        self, policy: ScalingPolicy, current_time: float
    ) -> ScalingDirection:
        """
        Evaluate a single scaling policy.

        Args:
            policy: ScalingPolicy to evaluate
            current_time: Current timestamp

        Returns:
            ScalingDirection decision
        """
        # Get recent metrics for this trigger
        metric_name = policy.trigger.value
        if metric_name not in self.metrics_history:
            return ScalingDirection.NONE

        recent_metrics = [
            m
            for m in self.metrics_history[metric_name]
            if current_time - m.timestamp
            <= self.evaluation_interval * policy.evaluation_periods
        ]

        if len(recent_metrics) < policy.evaluation_periods:
            return ScalingDirection.NONE

        # Calculate average metric value
        avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)

        # Check cooldown periods
        time_since_last_scale = current_time - self.last_scale_time

        # Evaluate scaling up
        if (
            avg_value > policy.scale_up_threshold
            and self.current_instances < policy.max_instances
            and (
                self.last_scale_direction != ScalingDirection.UP
                or time_since_last_scale >= policy.scale_up_cooldown
            )
        ):
            self.logger.info(
                "Scale up condition met",
                policy_name=policy.name,
                avg_value=avg_value,
                threshold=policy.scale_up_threshold,
                current_instances=self.current_instances,
            )
            return ScalingDirection.UP

        # Evaluate scaling down
        elif (
            avg_value < policy.scale_down_threshold
            and self.current_instances > policy.min_instances
            and (
                self.last_scale_direction != ScalingDirection.DOWN
                or time_since_last_scale >= policy.scale_down_cooldown
            )
        ):
            self.logger.info(
                "Scale down condition met",
                policy_name=policy.name,
                avg_value=avg_value,
                threshold=policy.scale_down_threshold,
                current_instances=self.current_instances,
            )
            return ScalingDirection.DOWN

        return ScalingDirection.NONE

    async def _execute_scaling(
        self, policy: ScalingPolicy, direction: ScalingDirection, current_time: float
    ):
        """
        Execute scaling operation.

        Args:
            policy: ScalingPolicy that triggered scaling
            direction: ScalingDirection to scale
            current_time: Current timestamp
        """
        old_instances = self.current_instances

        if direction == ScalingDirection.UP:
            new_instances = min(
                self.current_instances + policy.scale_up_step, policy.max_instances
            )
            callback = self.scale_up_callback
        else:  # ScalingDirection.DOWN
            new_instances = max(
                self.current_instances - policy.scale_down_step, policy.min_instances
            )
            callback = self.scale_down_callback

        if new_instances == old_instances:
            return  # No change needed

        # Get current metric value for logging
        metric_name = policy.trigger.value
        current_metric_value = 0.0
        if metric_name in self.metrics_history and self.metrics_history[metric_name]:
            current_metric_value = self.metrics_history[metric_name][-1].value

        # Execute scaling callback
        try:
            if callback:
                await self._execute_callback(callback, new_instances)

            # Update state
            self.current_instances = new_instances
            self.target_instances = new_instances
            self.last_scale_time = current_time
            self.last_scale_direction = direction

            # Record scaling event
            event = ScalingEvent(
                timestamp=current_time,
                direction=direction,
                trigger=policy.name,
                old_instances=old_instances,
                new_instances=new_instances,
                metric_value=current_metric_value,
                reason=f"{policy.trigger.value} threshold exceeded",
            )
            self.scaling_events.append(event)

            self.logger.info(
                "Executed scaling operation",
                direction=direction.value,
                old_instances=old_instances,
                new_instances=new_instances,
                trigger=policy.name,
                metric_value=current_metric_value,
            )

        except Exception as e:
            self.logger.error(
                "Failed to execute scaling operation",
                direction=direction.value,
                new_instances=new_instances,
                error=str(e),
            )

    async def _execute_callback(self, callback: Callable, instances: int):
        """Execute scaling callback, handling both sync and async."""
        if asyncio.iscoroutinefunction(callback):
            await callback(instances)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: callback(instances))

    async def _cleanup_old_metrics(self):
        """Remove old metrics to prevent memory growth."""
        current_time = time.time()
        retention_seconds = self.metrics_retention_hours * 3600

        for metric_name in self.metrics_history:
            self.metrics_history[metric_name] = [
                m
                for m in self.metrics_history[metric_name]
                if current_time - m.timestamp <= retention_seconds
            ]

        # Also cleanup old scaling events
        self.scaling_events = [
            e
            for e in self.scaling_events
            if current_time - e.timestamp <= retention_seconds
        ]

    async def manual_scale(self, target_instances: int, reason: str = "manual"):
        """
        Manually scale to target instance count.

        Args:
            target_instances: Target number of instances
            reason: Reason for manual scaling
        """
        if target_instances == self.current_instances:
            self.logger.info(
                "Manual scale requested but already at target",
                target_instances=target_instances,
            )
            return

        old_instances = self.current_instances
        direction = (
            ScalingDirection.UP
            if target_instances > old_instances
            else ScalingDirection.DOWN
        )
        callback = (
            self.scale_up_callback
            if direction == ScalingDirection.UP
            else self.scale_down_callback
        )

        try:
            if callback:
                await self._execute_callback(callback, target_instances)

            # Update state
            self.current_instances = target_instances
            self.target_instances = target_instances
            self.last_scale_time = time.time()
            self.last_scale_direction = direction

            # Record scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                direction=direction,
                trigger="manual",
                old_instances=old_instances,
                new_instances=target_instances,
                metric_value=0.0,
                reason=reason,
            )
            self.scaling_events.append(event)

            self.logger.info(
                "Manual scaling completed",
                old_instances=old_instances,
                new_instances=target_instances,
                reason=reason,
            )

        except Exception as e:
            self.logger.error(
                "Manual scaling failed", target_instances=target_instances, error=str(e)
            )
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get auto-scaler metrics and status."""
        current_time = time.time()

        # Calculate recent scaling events
        recent_events = [
            e
            for e in self.scaling_events
            if current_time - e.timestamp <= 3600  # Last hour
        ]

        return {
            "name": self.name,
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "last_scale_time": self.last_scale_time,
            "last_scale_direction": self.last_scale_direction.value,
            "monitoring_active": self.running,
            "policies": {
                name: {
                    "enabled": policy.enabled,
                    "trigger": policy.trigger.value,
                    "scale_up_threshold": policy.scale_up_threshold,
                    "scale_down_threshold": policy.scale_down_threshold,
                    "min_instances": policy.min_instances,
                    "max_instances": policy.max_instances,
                }
                for name, policy in self.policies.items()
            },
            "recent_scaling_events": len(recent_events),
            "total_scaling_events": len(self.scaling_events),
            "metrics_collected": {
                name: len(metrics) for name, metrics in self.metrics_history.items()
            },
        }

    def get_recent_metrics(
        self, metric_name: str, hours: int = 1
    ) -> List[ScalingMetric]:
        """
        Get recent metrics for a specific metric.

        Args:
            metric_name: Name of the metric
            hours: Number of hours of history to return

        Returns:
            List of recent ScalingMetric objects
        """
        if metric_name not in self.metrics_history:
            return []

        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)

        return [
            m for m in self.metrics_history[metric_name] if m.timestamp >= cutoff_time
        ]

    def get_scaling_events(self, hours: int = 24) -> List[ScalingEvent]:
        """
        Get recent scaling events.

        Args:
            hours: Number of hours of history to return

        Returns:
            List of recent ScalingEvent objects
        """
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)

        return [e for e in self.scaling_events if e.timestamp >= cutoff_time]


# Convenience functions for common scaling policies
def create_cpu_scaling_policy(
    name: str = "cpu_scaling",
    scale_up_threshold: float = 70.0,
    scale_down_threshold: float = 30.0,
    min_instances: int = 1,
    max_instances: int = 10,
) -> ScalingPolicy:
    """Create CPU-based scaling policy."""
    return ScalingPolicy(
        name=name,
        trigger=ScalingTrigger.CPU_USAGE,
        scale_up_threshold=scale_up_threshold,
        scale_down_threshold=scale_down_threshold,
        min_instances=min_instances,
        max_instances=max_instances,
        scale_up_cooldown=300.0,  # 5 minutes
        scale_down_cooldown=600.0,  # 10 minutes
        evaluation_periods=2,
    )


def create_memory_scaling_policy(
    name: str = "memory_scaling",
    scale_up_threshold: float = 80.0,
    scale_down_threshold: float = 40.0,
    min_instances: int = 1,
    max_instances: int = 10,
) -> ScalingPolicy:
    """Create memory-based scaling policy."""
    return ScalingPolicy(
        name=name,
        trigger=ScalingTrigger.MEMORY_USAGE,
        scale_up_threshold=scale_up_threshold,
        scale_down_threshold=scale_down_threshold,
        min_instances=min_instances,
        max_instances=max_instances,
        scale_up_cooldown=300.0,
        scale_down_cooldown=600.0,
        evaluation_periods=2,
    )


def create_request_rate_scaling_policy(
    name: str = "request_rate_scaling",
    scale_up_threshold: float = 100.0,  # requests per minute
    scale_down_threshold: float = 20.0,
    min_instances: int = 1,
    max_instances: int = 20,
) -> ScalingPolicy:
    """Create request rate-based scaling policy."""
    return ScalingPolicy(
        name=name,
        trigger=ScalingTrigger.REQUEST_RATE,
        scale_up_threshold=scale_up_threshold,
        scale_down_threshold=scale_down_threshold,
        min_instances=min_instances,
        max_instances=max_instances,
        scale_up_cooldown=180.0,  # 3 minutes - faster for request rate
        scale_down_cooldown=600.0,  # 10 minutes
        evaluation_periods=2,
    )
