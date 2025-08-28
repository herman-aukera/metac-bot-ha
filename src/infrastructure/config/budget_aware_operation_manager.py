"""
Budget-Aware Operation Manager with dynamic mode detection and switching.
Implements comprehensive budget monitoring, operation mode transitions, and cost optimization strategies.
"""
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .budget_manager import budget_manager, BudgetStatus
from .operation_modes import operation_mode_manager, OperationMode, ModeTransition
from .tri_model_router import OpenRouterTriModelRouter

logger = logging.getLogger(__name__)


class EmergencyProtocol(Enum):
    """Emergency protocol activation levels."""
    NONE = "none"
    BUDGET_WARNING = "budget_warning"
    BUDGET_CRITICAL = "budget_critical"
    SYSTEM_FAILURE = "system_failure"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class BudgetThreshold:
    """Budget utilization threshold configuration."""
    name: str
    percentage: float
    operation_mode: OperationMode
    emergency_protocol: EmergencyProtocol
    description: str
    actions: List[str]


@dataclass
class OperationModeTransitionLog:
    """Detailed log entry for operation mode transitions."""
    timestamp: datetime
    from_mode: OperationMode
    to_mode: OperationMode
    budget_utilization: float
    remaining_budget: float
    trigger_reason: str
    threshold_crossed: str
    emergency_protocol: EmergencyProtocol
    performance_impact: Dict[str, Any]
    cost_savings_estimate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['from_mode'] = self.from_mode.value
        data['to_mode'] = self.to_mode.value
        data['emergency_protocol'] = self.emergency_protocol.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperationModeTransitionLog':
        """Create from dictionary for JSON deserialization."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['from_mode'] = OperationMode(data['from_mode'])
        data['to_mode'] = OperationMode(data['to_mode'])
        data['emergency_protocol'] = EmergencyProtocol(data['emergency_protocol'])
        return cls(**data)


@dataclass
class CostOptimizationStrategy:
    """Cost optimization strategy for each operation mode."""
    mode: OperationMode
    model_selection_adjustments: Dict[str, str]
    task_prioritization_rules: List[str]
    research_depth_limits: Dict[str, int]
    feature_degradation_config: Dict[str, Any]
    estimated_cost_reduction: float
    performance_impact_score: float


class BudgetAwareOperationManager:
    """
    Comprehensive budget-aware operation manager with dynamic mode detection and switching.

    Implements:
    - Budget utilization monitoring and threshold detection
    - Automatic operation mode transitions
    - Operation mode logging and performance impact tracking
    - Emergency protocol activation and management
    - Cost optimization strategies for each operation mode
    """

    def __init__(self):
        """Initialize the budget-aware operation manager."""
        self.budget_manager = budget_manager
        self.operation_mode_manager = operation_mode_manager
        self.tri_model_router = OpenRouterTriModelRouter()

        # Initialize logging and data persistence
        self.log_file = Path("logs/operation_mode_transitions.json")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Operation mode transition history
        self.transition_history: List[OperationModeTransitionLog] = []

        # Budget thresholds configuration
        self.budget_thresholds = self._setup_budget_thresholds()

        # Cost optimization strategies
        self.cost_optimization_strategies = self._setup_cost_optimization_strategies()

        # Emergency protocol state
        self.current_emergency_protocol = EmergencyProtocol.NONE
        self.emergency_activation_time: Optional[datetime] = None

        # Performance tracking
        self.performance_metrics = {
            "mode_switches_count": 0,
            "emergency_activations": 0,
            "cost_savings_achieved": 0.0,
            "questions_processed_by_mode": {mode.value: 0 for mode in OperationMode},
            "average_cost_by_mode": {mode.value: 0.0 for mode in OperationMode}
        }

        # Load existing data
        self._load_existing_data()

        logger.info("Budget-aware operation manager initialized")
        self._log_initialization_status()

    def _setup_budget_thresholds(self) -> List[BudgetThreshold]:
        """Setup budget utilization thresholds with corresponding actions."""
        return [
            BudgetThreshold(
                name="normal_operation",
                percentage=0.0,
                operation_mode=OperationMode.NORMAL,
                emergency_protocol=EmergencyProtocol.NONE,
                description="Normal operation with full functionality",
                actions=[
                    "Use optimal model selection",
                    "Enable all features",
                    "Process all question priorities",
                    "Full complexity analysis enabled"
                ]
            ),
            BudgetThreshold(
                name="conservative_threshold",
                percentage=80.0,
                operation_mode=OperationMode.CONSERVATIVE,
                emergency_protocol=EmergencyProtocol.BUDGET_WARNING,
                description="Conservative mode to preserve budget",
                actions=[
                    "Switch to cost-efficient models",
                    "Skip low-priority questions",
                    "Reduce batch sizes",
                    "Limit retries to 2 attempts"
                ]
            ),
            BudgetThreshold(
                name="emergency_threshold",
                percentage=95.0,
                operation_mode=OperationMode.EMERGENCY,
                emergency_protocol=EmergencyProtocol.BUDGET_CRITICAL,
                description="Emergency mode with minimal functionality",
                actions=[
                    "Use cheapest models only",
                    "Process critical questions only",
                    "Disable complexity analysis",
                    "Single retry attempt",
                    "Minimal batch sizes"
                ]
            )
        ]

    def _setup_cost_optimization_strategies(self) -> Dict[OperationMode, CostOptimizationStrategy]:
        """Setup cost optimization strategies for each operation mode."""
        return {
            OperationMode.NORMAL: CostOptimizationStrategy(
                mode=OperationMode.NORMAL,
                model_selection_adjustments={
                    "research": "openai/gpt-4o-mini",
                    "forecast": "openai/gpt-4o",
                    "validation": "openai/gpt-4o-mini"
                },
                task_prioritization_rules=[
                    "Process all priorities",
                    "Optimize for quality",
                    "Use complexity analysis"
                ],
                research_depth_limits={
                    "max_sources": 10,
                    "max_depth": 3,
                    "max_iterations": 5
                },
                feature_degradation_config={
                    "complexity_analysis": True,
                    "multi_stage_validation": True,
                    "detailed_logging": True
                },
                estimated_cost_reduction=0.0,
                performance_impact_score=1.0
            ),
            OperationMode.CONSERVATIVE: CostOptimizationStrategy(
                mode=OperationMode.CONSERVATIVE,
                model_selection_adjustments={
                    "research": "openai/gpt-4o-mini",
                    "forecast": "openai/gpt-4o-mini",
                    "validation": "openai/gpt-4o-mini"
                },
                task_prioritization_rules=[
                    "Skip low priority questions",
                    "Prioritize high-value tasks",
                    "Reduce research depth"
                ],
                research_depth_limits={
                    "max_sources": 5,
                    "max_depth": 2,
                    "max_iterations": 3
                },
                feature_degradation_config={
                    "complexity_analysis": True,
                    "multi_stage_validation": False,
                    "detailed_logging": False
                },
                estimated_cost_reduction=0.4,
                performance_impact_score=0.8
            ),
            OperationMode.EMERGENCY: CostOptimizationStrategy(
                mode=OperationMode.EMERGENCY,
                model_selection_adjustments={
                    "research": "openai/gpt-4o-mini",
                    "forecast": "openai/gpt-4o-mini",
                    "validation": "openai/gpt-4o-mini"
                },
                task_prioritization_rules=[
                    "Critical questions only",
                    "Minimal processing",
                    "Emergency protocols active"
                ],
                research_depth_limits={
                    "max_sources": 2,
                    "max_depth": 1,
                    "max_iterations": 1
                },
                feature_degradation_config={
                    "complexity_analysis": False,
                    "multi_stage_validation": False,
                    "detailed_logging": False
                },
                estimated_cost_reduction=0.7,
                performance_impact_score=0.5
            )
        }
    def monitor_budget_utilization(self) -> Dict[str, Any]:
        """
        Monitor budget utilization and detect threshold crossings.

        Returns:
            Dict containing current status and any threshold alerts
        """
        budget_status = self.budget_manager.get_budget_status()
        current_utilization = budget_status.utilization_percentage

        # Check for threshold crossings
        threshold_alerts = []
        crossed_threshold = None

        for threshold in sorted(self.budget_thresholds, key=lambda x: x.percentage, reverse=True):
            if current_utilization >= threshold.percentage:
                crossed_threshold = threshold
                break

        # Detect if we've crossed a new threshold
        current_mode = self.operation_mode_manager.get_current_mode()
        if crossed_threshold and crossed_threshold.operation_mode != current_mode:
            threshold_alerts.append({
                "threshold_name": crossed_threshold.name,
                "percentage": crossed_threshold.percentage,
                "current_utilization": current_utilization,
                "recommended_mode": crossed_threshold.operation_mode.value,
                "emergency_protocol": crossed_threshold.emergency_protocol.value,
                "actions": crossed_threshold.actions
            })

        return {
            "budget_status": budget_status,
            "current_utilization": current_utilization,
            "crossed_threshold": crossed_threshold.name if crossed_threshold else None,
            "threshold_alerts": threshold_alerts,
            "emergency_protocol": self.current_emergency_protocol.value,
            "monitoring_timestamp": datetime.now().isoformat()
        }

    def detect_and_switch_operation_mode(self) -> Tuple[bool, Optional[OperationModeTransitionLog]]:
        """
        Detect budget threshold crossings and automatically switch operation modes.

        Returns:
            Tuple of (mode_changed, transition_log)
        """
        monitoring_result = self.monitor_budget_utilization()
        budget_status = monitoring_result["budget_status"]
        current_utilization = monitoring_result["current_utilization"]

        # Check if mode transition is needed
        threshold_alerts = monitoring_result["threshold_alerts"]
        if not threshold_alerts:
            return False, None

        # Get the most restrictive threshold that was crossed
        alert = threshold_alerts[0]
        target_mode = OperationMode(alert["recommended_mode"])
        emergency_protocol = EmergencyProtocol(alert["emergency_protocol"])

        # Perform the mode transition
        current_mode = self.operation_mode_manager.get_current_mode()

        # Calculate performance impact and cost savings
        performance_impact = self._calculate_performance_impact(current_mode, target_mode)
        cost_savings_estimate = self._estimate_cost_savings(current_mode, target_mode, budget_status)

        # Create detailed transition log
        transition_log = OperationModeTransitionLog(
            timestamp=datetime.now(),
            from_mode=current_mode,
            to_mode=target_mode,
            budget_utilization=current_utilization,
            remaining_budget=budget_status.remaining,
            trigger_reason=f"Budget threshold crossed: {alert['threshold_name']}",
            threshold_crossed=alert["threshold_name"],
            emergency_protocol=emergency_protocol,
            performance_impact=performance_impact,
            cost_savings_estimate=cost_savings_estimate
        )

        # Execute the transition
        self.operation_mode_manager.force_mode_transition(
            target_mode,
            f"Budget threshold: {alert['threshold_name']}"
        )

        # Activate emergency protocol if needed
        if emergency_protocol != EmergencyProtocol.NONE:
            self._activate_emergency_protocol(emergency_protocol, transition_log)

        # Record the transition
        self.transition_history.append(transition_log)
        self.performance_metrics["mode_switches_count"] += 1

        # Save transition data
        self._save_transition_data()

        logger.warning(f"Operation mode switched: {current_mode.value} → {target_mode.value} "
                      f"(Budget: {current_utilization:.1f}%, Reason: {transition_log.trigger_reason})")

        return True, transition_log
    def _calculate_performance_impact(self, from_mode: OperationMode, to_mode: OperationMode) -> Dict[str, Any]:
        """Calculate the performance impact of mode transition."""
        from_strategy = self.cost_optimization_strategies[from_mode]
        to_strategy = self.cost_optimization_strategies[to_mode]

        return {
            "performance_score_change": to_strategy.performance_impact_score - from_strategy.performance_impact_score,
            "feature_changes": {
                "complexity_analysis": {
                    "from": from_strategy.feature_degradation_config["complexity_analysis"],
                    "to": to_strategy.feature_degradation_config["complexity_analysis"]
                },
                "multi_stage_validation": {
                    "from": from_strategy.feature_degradation_config["multi_stage_validation"],
                    "to": to_strategy.feature_degradation_config["multi_stage_validation"]
                },
                "detailed_logging": {
                    "from": from_strategy.feature_degradation_config["detailed_logging"],
                    "to": to_strategy.feature_degradation_config["detailed_logging"]
                }
            },
            "research_depth_changes": {
                "max_sources": {
                    "from": from_strategy.research_depth_limits["max_sources"],
                    "to": to_strategy.research_depth_limits["max_sources"]
                },
                "max_depth": {
                    "from": from_strategy.research_depth_limits["max_depth"],
                    "to": to_strategy.research_depth_limits["max_depth"]
                }
            },
            "model_changes": {
                "research_model": {
                    "from": from_strategy.model_selection_adjustments["research"],
                    "to": to_strategy.model_selection_adjustments["research"]
                },
                "forecast_model": {
                    "from": from_strategy.model_selection_adjustments["forecast"],
                    "to": to_strategy.model_selection_adjustments["forecast"]
                }
            }
        }

    def _estimate_cost_savings(self, from_mode: OperationMode, to_mode: OperationMode,
                              budget_status: BudgetStatus) -> float:
        """Estimate cost savings from mode transition."""
        from_strategy = self.cost_optimization_strategies[from_mode]
        to_strategy = self.cost_optimization_strategies[to_mode]

        # Calculate relative cost reduction
        cost_reduction_factor = to_strategy.estimated_cost_reduction - from_strategy.estimated_cost_reduction

        # Estimate savings based on remaining questions
        estimated_savings = (budget_status.remaining * cost_reduction_factor *
                           budget_status.estimated_questions_remaining /
                           max(budget_status.questions_processed, 1))

        return max(0.0, estimated_savings)

    def _activate_emergency_protocol(self, protocol: EmergencyProtocol,
                                   transition_log: OperationModeTransitionLog):
        """Activate emergency protocol with appropriate actions."""
        self.current_emergency_protocol = protocol
        self.emergency_activation_time = datetime.now()
        self.performance_metrics["emergency_activations"] += 1

        logger.critical(f"Emergency protocol activated: {protocol.value}")

        if protocol == EmergencyProtocol.BUDGET_WARNING:
            self._execute_budget_warning_protocol()
        elif protocol == EmergencyProtocol.BUDGET_CRITICAL:
            self._execute_budget_critical_protocol()
        elif protocol == EmergencyProtocol.SYSTEM_FAILURE:
            self._execute_system_failure_protocol()

    def _execute_budget_warning_protocol(self):
        """Execute budget warning emergency protocol."""
        actions = [
            "Switch to conservative operation mode",
            "Reduce batch processing sizes",
            "Skip low-priority questions",
            "Enable cost monitoring alerts"
        ]

        logger.warning("Budget warning protocol active - implementing cost conservation measures")
        for action in actions:
            logger.info(f"Emergency action: {action}")

    def _execute_budget_critical_protocol(self):
        """Execute budget critical emergency protocol."""
        actions = [
            "Switch to emergency operation mode",
            "Process critical questions only",
            "Use cheapest models exclusively",
            "Disable non-essential features",
            "Implement strict cost limits"
        ]

        logger.critical("Budget critical protocol active - implementing emergency measures")
        for action in actions:
            logger.critical(f"Emergency action: {action}")

    def _execute_system_failure_protocol(self):
        """Execute system failure emergency protocol."""
        actions = [
            "Halt all non-critical processing",
            "Save current state",
            "Switch to minimal functionality mode",
            "Alert system administrators"
        ]

        logger.critical("System failure protocol active - implementing emergency shutdown")
        for action in actions:
            logger.critical(f"Emergency action: {action}")
    def get_cost_optimization_strategy(self, mode: Optional[OperationMode] = None) -> CostOptimizationStrategy:
        """Get cost optimization strategy for specified or current mode."""
        target_mode = mode or self.operation_mode_manager.get_current_mode()
        return self.cost_optimization_strategies[target_mode]

    def apply_model_selection_adjustments(self, task_type: str,
                                        mode: Optional[OperationMode] = None) -> str:
        """Apply model selection adjustments based on operation mode."""
        strategy = self.get_cost_optimization_strategy(mode)

        # Get base model from strategy
        base_model = strategy.model_selection_adjustments.get(task_type, "openai/gpt-4o-mini")

        # Apply additional optimizations based on current emergency protocol
        if self.current_emergency_protocol == EmergencyProtocol.BUDGET_CRITICAL:
            # Force cheapest model in critical situations
            return "openai/gpt-4o-mini"

        return base_model

    def should_skip_question(self, question_priority: str = "normal",
                           question_complexity: str = "medium") -> Tuple[bool, str]:
        """Determine if a question should be skipped based on current operation mode."""
        current_mode = self.operation_mode_manager.get_current_mode()
        strategy = self.get_cost_optimization_strategy(current_mode)

        # Check emergency protocol first
        if self.current_emergency_protocol == EmergencyProtocol.BUDGET_CRITICAL:
            if question_priority.lower() not in ["critical", "high"]:
                return True, f"Emergency protocol active: skipping {question_priority} priority question"

        # Apply mode-specific rules
        if current_mode == OperationMode.EMERGENCY:
            if question_priority.lower() not in ["critical", "high"]:
                return True, f"Emergency mode: processing critical/high priority only"
        elif current_mode == OperationMode.CONSERVATIVE:
            if question_priority.lower() == "low":
                return True, f"Conservative mode: skipping low priority questions"

        return False, "Question can be processed"

    def get_research_depth_limits(self, mode: Optional[OperationMode] = None) -> Dict[str, int]:
        """Get research depth limits for specified or current mode."""
        strategy = self.get_cost_optimization_strategy(mode)
        return strategy.research_depth_limits.copy()

    def get_graceful_degradation_config(self, mode: Optional[OperationMode] = None) -> Dict[str, Any]:
        """Get graceful feature degradation configuration."""
        strategy = self.get_cost_optimization_strategy(mode)

        config = strategy.feature_degradation_config.copy()

        # Apply emergency protocol overrides
        if self.current_emergency_protocol == EmergencyProtocol.BUDGET_CRITICAL:
            config.update({
                "complexity_analysis": False,
                "multi_stage_validation": False,
                "detailed_logging": False
            })

        return config

    def log_operation_mode_performance(self):
        """Log comprehensive operation mode and performance tracking information."""
        current_mode = self.operation_mode_manager.get_current_mode()
        budget_status = self.budget_manager.get_budget_status()
        strategy = self.get_cost_optimization_strategy()

        logger.info("=== Budget-Aware Operation Manager Status ===")
        logger.info(f"Current Operation Mode: {current_mode.value.upper()}")
        logger.info(f"Emergency Protocol: {self.current_emergency_protocol.value.upper()}")
        logger.info(f"Budget Utilization: {budget_status.utilization_percentage:.1f}%")
        logger.info(f"Remaining Budget: ${budget_status.remaining:.4f}")

        logger.info("=== Cost Optimization Strategy ===")
        logger.info(f"Research Model: {strategy.model_selection_adjustments['research']}")
        logger.info(f"Forecast Model: {strategy.model_selection_adjustments['forecast']}")
        logger.info(f"Estimated Cost Reduction: {strategy.estimated_cost_reduction:.1%}")
        logger.info(f"Performance Impact Score: {strategy.performance_impact_score:.2f}")

        logger.info("=== Research Depth Limits ===")
        for limit_type, value in strategy.research_depth_limits.items():
            logger.info(f"{limit_type}: {value}")

        logger.info("=== Feature Degradation Status ===")
        for feature, enabled in strategy.feature_degradation_config.items():
            status = "ENABLED" if enabled else "DISABLED"
            logger.info(f"{feature}: {status}")

        logger.info("=== Performance Metrics ===")
        logger.info(f"Mode Switches: {self.performance_metrics['mode_switches_count']}")
        logger.info(f"Emergency Activations: {self.performance_metrics['emergency_activations']}")
        logger.info(f"Cost Savings Achieved: ${self.performance_metrics['cost_savings_achieved']:.4f}")

        # Log recent transitions
        if self.transition_history:
            recent_transitions = self.transition_history[-3:]
            logger.info("=== Recent Mode Transitions ===")
            for transition in recent_transitions:
                logger.info(f"{transition.timestamp.strftime('%H:%M:%S')}: "
                           f"{transition.from_mode.value} → {transition.to_mode.value} "
                           f"(Budget: {transition.budget_utilization:.1f}%, "
                           f"Savings: ${transition.cost_savings_estimate:.4f})")
    def _save_transition_data(self):
        """Save transition history and performance metrics to file."""
        try:
            data = {
                "transition_history": [t.to_dict() for t in self.transition_history],
                "performance_metrics": self.performance_metrics,
                "current_emergency_protocol": self.current_emergency_protocol.value,
                "emergency_activation_time": (
                    self.emergency_activation_time.isoformat()
                    if self.emergency_activation_time else None
                ),
                "last_updated": datetime.now().isoformat()
            }

            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save transition data: {e}")

    def _load_existing_data(self):
        """Load existing transition history and performance metrics."""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    data = json.load(f)

                # Load transition history
                history_data = data.get("transition_history", [])
                self.transition_history = [
                    OperationModeTransitionLog.from_dict(t) for t in history_data
                ]

                # Load performance metrics
                self.performance_metrics.update(data.get("performance_metrics", {}))

                # Load emergency protocol state
                protocol_value = data.get("current_emergency_protocol", "none")
                self.current_emergency_protocol = EmergencyProtocol(protocol_value)

                activation_time = data.get("emergency_activation_time")
                if activation_time:
                    self.emergency_activation_time = datetime.fromisoformat(activation_time)

                logger.info(f"Loaded {len(self.transition_history)} transition records")

        except Exception as e:
            logger.warning(f"Failed to load existing transition data: {e}")

    def _log_initialization_status(self):
        """Log initialization status and current configuration."""
        current_mode = self.operation_mode_manager.get_current_mode()
        budget_status = self.budget_manager.get_budget_status()

        logger.info("=== Budget-Aware Operation Manager Initialized ===")
        logger.info(f"Current Mode: {current_mode.value}")
        logger.info(f"Budget Utilization: {budget_status.utilization_percentage:.1f}%")
        logger.info(f"Emergency Protocol: {self.current_emergency_protocol.value}")
        logger.info(f"Loaded Transitions: {len(self.transition_history)}")
        logger.info(f"Total Mode Switches: {self.performance_metrics['mode_switches_count']}")

    def get_transition_history(self) -> List[OperationModeTransitionLog]:
        """Get complete transition history."""
        return self.transition_history.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def reset_emergency_protocol(self):
        """Reset emergency protocol to normal state."""
        if self.current_emergency_protocol != EmergencyProtocol.NONE:
            logger.info(f"Resetting emergency protocol from {self.current_emergency_protocol.value}")
            self.current_emergency_protocol = EmergencyProtocol.NONE
            self.emergency_activation_time = None
            self._save_transition_data()

    def force_emergency_protocol(self, protocol: EmergencyProtocol, reason: str = "manual_override"):
        """Force activation of emergency protocol."""
        transition_log = OperationModeTransitionLog(
            timestamp=datetime.now(),
            from_mode=self.operation_mode_manager.get_current_mode(),
            to_mode=self.operation_mode_manager.get_current_mode(),
            budget_utilization=self.budget_manager.get_budget_status().utilization_percentage,
            remaining_budget=self.budget_manager.get_budget_status().remaining,
            trigger_reason=reason,
            threshold_crossed="manual_override",
            emergency_protocol=protocol,
            performance_impact={},
            cost_savings_estimate=0.0
        )

        self._activate_emergency_protocol(protocol, transition_log)
        self.transition_history.append(transition_log)
        self._save_transition_data()

    def get_operation_mode_for_budget(self, budget_utilization_percentage: float) -> str:
        """Get the expected operation mode for a given budget utilization percentage."""
        for threshold in sorted(self.budget_thresholds, key=lambda x: x.percentage, reverse=True):
            if budget_utilization_percentage >= threshold.percentage:
                return threshold.operation_mode.value

        # Default to normal if below all thresholds
        return "normal"


# Global instance
budget_aware_operation_manager = BudgetAwareOperationManager()
