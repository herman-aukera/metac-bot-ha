"""Graceful degradation manager for maintaining functionality under stress."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


class DegradationLevel(Enum):
    """Levels of system degradation."""

    NORMAL = "normal"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    CRITICAL = "critical"


class FeaturePriority(Enum):
    """Priority levels for features during degradation."""

    CRITICAL = "critical"  # Never disable
    HIGH = "high"  # Disable only in critical situations
    MEDIUM = "medium"  # Disable in heavy degradation
    LOW = "low"  # Disable in moderate degradation
    OPTIONAL = "optional"  # Disable in light degradation


@dataclass
class DegradationRule:
    """Rule for degrading system functionality."""

    name: str
    trigger_condition: Callable[[], bool]
    degradation_level: DegradationLevel
    affected_features: Set[str]
    recovery_condition: Optional[Callable[[], bool]] = None
    enabled: bool = True
    description: str = ""


@dataclass
class Feature:
    """System feature that can be degraded."""

    name: str
    priority: FeaturePriority
    enabled: bool = True
    degraded: bool = False
    fallback_function: Optional[Callable] = None
    description: str = ""
    tags: Set[str] = field(default_factory=set)


@dataclass
class DegradationEvent:
    """Record of a degradation event."""

    timestamp: float
    level: DegradationLevel
    triggered_by: str
    affected_features: List[str]
    reason: str
    recovered: bool = False
    recovery_timestamp: Optional[float] = None


class GracefulDegradationManager:
    """
    Graceful degradation manager for maintaining core functionality under stress.

    Automatically disables non-critical features and provides fallback mechanisms
    to maintain tournament performance during high load or system stress.
    """

    def __init__(self, name: str = "degradation_manager"):
        self.name = name
        self.current_level = DegradationLevel.NORMAL
        self.features: Dict[str, Feature] = {}
        self.degradation_rules: Dict[str, DegradationRule] = {}
        self.active_degradations: Dict[str, DegradationEvent] = {}
        self.degradation_history: List[DegradationEvent] = []
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.logger = logger.bind(component="graceful_degradation")

        # Configuration
        self.evaluation_interval = 30.0  # Check every 30 seconds
        self.history_retention_hours = 24

        # Callbacks
        self.degradation_callbacks: List[
            Callable[[DegradationLevel, DegradationLevel], None]
        ] = []
        self.feature_callbacks: Dict[str, List[Callable[[bool], None]]] = {}

    def register_feature(self, feature: Feature):
        """
        Register a feature that can be degraded.

        Args:
            feature: Feature configuration
        """
        self.features[feature.name] = feature
        self.logger.info(
            "Registered feature",
            name=feature.name,
            priority=feature.priority.value,
            enabled=feature.enabled,
        )

    def register_degradation_rule(self, rule: DegradationRule):
        """
        Register a degradation rule.

        Args:
            rule: DegradationRule configuration
        """
        self.degradation_rules[rule.name] = rule
        self.logger.info(
            "Registered degradation rule",
            name=rule.name,
            level=rule.degradation_level.value,
            affected_features=len(rule.affected_features),
        )

    def register_degradation_callback(
        self, callback: Callable[[DegradationLevel, DegradationLevel], None]
    ):
        """
        Register callback for degradation level changes.

        Args:
            callback: Function called when degradation level changes
                     (old_level, new_level)
        """
        self.degradation_callbacks.append(callback)

    def register_feature_callback(
        self, feature_name: str, callback: Callable[[bool], None]
    ):
        """
        Register callback for feature state changes.

        Args:
            feature_name: Name of the feature
            callback: Function called when feature state changes (enabled)
        """
        if feature_name not in self.feature_callbacks:
            self.feature_callbacks[feature_name] = []
        self.feature_callbacks[feature_name].append(callback)

    async def start_monitoring(self):
        """Start the degradation monitoring loop."""
        if self.running:
            self.logger.warning("Degradation manager already running")
            return

        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(
            "Started degradation monitoring",
            evaluation_interval=self.evaluation_interval,
            rules=len(self.degradation_rules),
            features=len(self.features),
        )

    async def stop_monitoring(self):
        """Stop the degradation monitoring loop."""
        if not self.running:
            return

        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped degradation monitoring")

    async def _monitoring_loop(self):
        """Main monitoring and degradation loop."""
        while self.running:
            try:
                await self._evaluate_degradation_rules()
                await self._update_degradation_level()
                await self._cleanup_old_events()
                await asyncio.sleep(self.evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in degradation monitoring loop", error=str(e))
                await asyncio.sleep(self.evaluation_interval)

    async def _evaluate_degradation_rules(self):
        """Evaluate all degradation rules."""
        current_time = time.time()

        for rule_name, rule in self.degradation_rules.items():
            if not rule.enabled:
                continue

            try:
                # Check if rule should trigger
                should_trigger = await self._execute_condition(rule.trigger_condition)
                is_active = rule_name in self.active_degradations

                if should_trigger and not is_active:
                    # Trigger degradation
                    await self._trigger_degradation(rule, current_time)

                elif not should_trigger and is_active:
                    # Check recovery condition
                    should_recover = True
                    if rule.recovery_condition:
                        should_recover = await self._execute_condition(
                            rule.recovery_condition
                        )

                    if should_recover:
                        await self._recover_degradation(rule_name, current_time)

            except Exception as e:
                self.logger.error(
                    "Error evaluating degradation rule",
                    rule_name=rule_name,
                    error=str(e),
                )

    async def _execute_condition(self, condition: Callable) -> bool:
        """Execute condition function, handling both sync and async."""
        if asyncio.iscoroutinefunction(condition):
            return await condition()
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, condition)

    async def _trigger_degradation(self, rule: DegradationRule, current_time: float):
        """
        Trigger degradation based on rule.

        Args:
            rule: DegradationRule that triggered
            current_time: Current timestamp
        """
        # Create degradation event
        event = DegradationEvent(
            timestamp=current_time,
            level=rule.degradation_level,
            triggered_by=rule.name,
            affected_features=list(rule.affected_features),
            reason=rule.description or f"Rule {rule.name} triggered",
        )

        # Store active degradation
        self.active_degradations[rule.name] = event
        self.degradation_history.append(event)

        # Degrade affected features
        for feature_name in rule.affected_features:
            await self._degrade_feature(feature_name, rule.degradation_level)

        self.logger.warning(
            "Degradation triggered",
            rule_name=rule.name,
            level=rule.degradation_level.value,
            affected_features=len(rule.affected_features),
        )

    async def _recover_degradation(self, rule_name: str, current_time: float):
        """
        Recover from degradation.

        Args:
            rule_name: Name of the rule to recover from
            current_time: Current timestamp
        """
        if rule_name not in self.active_degradations:
            return

        event = self.active_degradations[rule_name]
        rule = self.degradation_rules[rule_name]

        # Mark event as recovered
        event.recovered = True
        event.recovery_timestamp = current_time

        # Remove from active degradations
        del self.active_degradations[rule_name]

        # Recover affected features
        for feature_name in rule.affected_features:
            await self._recover_feature(feature_name)

        self.logger.info(
            "Degradation recovered",
            rule_name=rule_name,
            duration=current_time - event.timestamp,
            affected_features=len(rule.affected_features),
        )

    async def _degrade_feature(
        self, feature_name: str, degradation_level: DegradationLevel
    ):
        """
        Degrade a specific feature.

        Args:
            feature_name: Name of the feature to degrade
            degradation_level: Level of degradation
        """
        if feature_name not in self.features:
            self.logger.warning(
                "Attempted to degrade unknown feature", feature_name=feature_name
            )
            return

        feature = self.features[feature_name]

        # Check if feature should be degraded based on priority
        if not self._should_degrade_feature(feature, degradation_level):
            return

        # Degrade feature
        old_enabled = feature.enabled
        feature.enabled = False
        feature.degraded = True

        # Notify callbacks
        await self._notify_feature_callbacks(feature_name, False)

        self.logger.info(
            "Feature degraded",
            feature_name=feature_name,
            priority=feature.priority.value,
            degradation_level=degradation_level.value,
        )

    async def _recover_feature(self, feature_name: str):
        """
        Recover a degraded feature.

        Args:
            feature_name: Name of the feature to recover
        """
        if feature_name not in self.features:
            return

        feature = self.features[feature_name]

        if not feature.degraded:
            return  # Feature wasn't degraded

        # Check if feature is still affected by other active degradations
        for active_event in self.active_degradations.values():
            if feature_name in active_event.affected_features:
                return  # Still affected by another degradation

        # Recover feature
        feature.enabled = True
        feature.degraded = False

        # Notify callbacks
        await self._notify_feature_callbacks(feature_name, True)

        self.logger.info("Feature recovered", feature_name=feature_name)

    def _should_degrade_feature(
        self, feature: Feature, degradation_level: DegradationLevel
    ) -> bool:
        """
        Check if feature should be degraded at given level.

        Args:
            feature: Feature to check
            degradation_level: Current degradation level

        Returns:
            True if feature should be degraded
        """
        if feature.priority == FeaturePriority.CRITICAL:
            return False  # Never degrade critical features

        degradation_thresholds = {
            DegradationLevel.LIGHT: [FeaturePriority.OPTIONAL],
            DegradationLevel.MODERATE: [FeaturePriority.OPTIONAL, FeaturePriority.LOW],
            DegradationLevel.HEAVY: [
                FeaturePriority.OPTIONAL,
                FeaturePriority.LOW,
                FeaturePriority.MEDIUM,
            ],
            DegradationLevel.CRITICAL: [
                FeaturePriority.OPTIONAL,
                FeaturePriority.LOW,
                FeaturePriority.MEDIUM,
                FeaturePriority.HIGH,
            ],
        }

        return feature.priority in degradation_thresholds.get(degradation_level, [])

    async def _update_degradation_level(self):
        """Update overall system degradation level."""
        if not self.active_degradations:
            new_level = DegradationLevel.NORMAL
        else:
            # Use highest degradation level from active degradations
            levels = [event.level for event in self.active_degradations.values()]
            level_order = [
                DegradationLevel.NORMAL,
                DegradationLevel.LIGHT,
                DegradationLevel.MODERATE,
                DegradationLevel.HEAVY,
                DegradationLevel.CRITICAL,
            ]

            max_level_index = max(level_order.index(level) for level in levels)
            new_level = level_order[max_level_index]

        if new_level != self.current_level:
            old_level = self.current_level
            self.current_level = new_level

            # Notify callbacks
            await self._notify_degradation_callbacks(old_level, new_level)

            self.logger.info(
                "Degradation level changed",
                old_level=old_level.value,
                new_level=new_level.value,
            )

    async def _notify_feature_callbacks(self, feature_name: str, enabled: bool):
        """Notify feature state change callbacks."""
        if feature_name in self.feature_callbacks:
            for callback in self.feature_callbacks[feature_name]:
                try:
                    await self._execute_callback(callback, enabled)
                except Exception as e:
                    self.logger.error(
                        "Error in feature callback",
                        feature_name=feature_name,
                        callback=callback.__name__,
                        error=str(e),
                    )

    async def _notify_degradation_callbacks(
        self, old_level: DegradationLevel, new_level: DegradationLevel
    ):
        """Notify degradation level change callbacks."""
        for callback in self.degradation_callbacks:
            try:
                await self._execute_callback(callback, old_level, new_level)
            except Exception as e:
                self.logger.error(
                    "Error in degradation callback",
                    callback=callback.__name__,
                    error=str(e),
                )

    async def _execute_callback(self, callback: Callable, *args):
        """Execute callback, handling both sync and async."""
        if asyncio.iscoroutinefunction(callback):
            await callback(*args)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: callback(*args))

    async def _cleanup_old_events(self):
        """Remove old degradation events to prevent memory growth."""
        current_time = time.time()
        retention_seconds = self.history_retention_hours * 3600

        self.degradation_history = [
            event
            for event in self.degradation_history
            if current_time - event.timestamp <= retention_seconds
        ]

    async def manual_degrade(self, level: DegradationLevel, reason: str = "manual"):
        """
        Manually trigger degradation to specified level.

        Args:
            level: Target degradation level
            reason: Reason for manual degradation
        """
        current_time = time.time()

        # Create manual degradation event
        event = DegradationEvent(
            timestamp=current_time,
            level=level,
            triggered_by="manual",
            affected_features=[],
            reason=reason,
        )

        # Determine features to degrade
        features_to_degrade = []
        for feature_name, feature in self.features.items():
            if self._should_degrade_feature(feature, level):
                features_to_degrade.append(feature_name)
                await self._degrade_feature(feature_name, level)

        event.affected_features = features_to_degrade

        # Store degradation
        self.active_degradations["manual"] = event
        self.degradation_history.append(event)

        self.logger.warning(
            "Manual degradation triggered",
            level=level.value,
            reason=reason,
            affected_features=len(features_to_degrade),
        )

    async def manual_recover(self):
        """Manually recover from all degradations."""
        current_time = time.time()

        # Recover all active degradations
        for rule_name in list(self.active_degradations.keys()):
            await self._recover_degradation(rule_name, current_time)

        self.logger.info("Manual recovery completed")

    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is currently enabled.

        Args:
            feature_name: Name of the feature

        Returns:
            True if feature is enabled
        """
        if feature_name not in self.features:
            return False

        return self.features[feature_name].enabled

    def get_fallback_function(self, feature_name: str) -> Optional[Callable]:
        """
        Get fallback function for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Fallback function if available
        """
        if feature_name not in self.features:
            return None

        return self.features[feature_name].fallback_function

    def get_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        return {
            "current_level": self.current_level.value,
            "active_degradations": len(self.active_degradations),
            "degraded_features": sum(1 for f in self.features.values() if f.degraded),
            "total_features": len(self.features),
            "monitoring_active": self.running,
            "active_degradation_details": {
                name: {
                    "level": event.level.value,
                    "triggered_by": event.triggered_by,
                    "duration": time.time() - event.timestamp,
                    "affected_features": len(event.affected_features),
                }
                for name, event in self.active_degradations.items()
            },
            "feature_status": {
                name: {
                    "enabled": feature.enabled,
                    "degraded": feature.degraded,
                    "priority": feature.priority.value,
                }
                for name, feature in self.features.items()
            },
        }

    def get_degradation_history(self, hours: int = 24) -> List[DegradationEvent]:
        """
        Get degradation history.

        Args:
            hours: Number of hours of history to return

        Returns:
            List of DegradationEvent objects
        """
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)

        return [
            event
            for event in self.degradation_history
            if event.timestamp >= cutoff_time
        ]


# Convenience functions for common degradation scenarios
def create_cpu_degradation_rule(
    name: str = "high_cpu",
    cpu_threshold: float = 90.0,
    affected_features: Set[str] = None,
) -> DegradationRule:
    """Create CPU-based degradation rule."""
    import psutil

    def check_cpu():
        return psutil.cpu_percent(interval=1) > cpu_threshold

    return DegradationRule(
        name=name,
        trigger_condition=check_cpu,
        degradation_level=DegradationLevel.MODERATE,
        affected_features=affected_features or {"analytics", "detailed_logging"},
        description=f"CPU usage above {cpu_threshold}%",
    )


def create_memory_degradation_rule(
    name: str = "high_memory",
    memory_threshold: float = 85.0,
    affected_features: Set[str] = None,
) -> DegradationRule:
    """Create memory-based degradation rule."""
    import psutil

    def check_memory():
        return psutil.virtual_memory().percent > memory_threshold

    return DegradationRule(
        name=name,
        trigger_condition=check_memory,
        degradation_level=DegradationLevel.HEAVY,
        affected_features=affected_features
        or {"caching", "background_tasks", "analytics"},
        description=f"Memory usage above {memory_threshold}%",
    )
