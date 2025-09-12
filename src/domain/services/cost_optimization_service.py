"""
Cost Optimization Service for Budget-Aware Operation Modes.
Implements model selection adjustments, task prioritization algorithms,
research depth adaptation, and graceful feature degradation.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ...infrastructure.config.budget_manager import budget_manager
from ...infrastructure.config.operation_modes import OperationMode

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for cost optimization."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskComplexity(Enum):
    """Task complexity levels for resource allocation."""

    MINIMAL = "minimal"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TaskPrioritizationResult:
    """Result of task prioritization algorithm."""

    should_process: bool
    priority_score: float
    reason: str
    estimated_cost: float
    resource_allocation: Dict[str, Any]


@dataclass
class ModelSelectionResult:
    """Result of model selection optimization."""

    selected_model: str
    original_model: str
    cost_reduction: float
    performance_impact: float
    rationale: str


@dataclass
class ResearchDepthConfig:
    """Configuration for research depth adaptation."""

    max_sources: int
    max_depth: int
    max_iterations: int
    enable_deep_analysis: bool
    complexity_threshold: float
    time_limit_seconds: int


class CostOptimizationService:
    """
    Service for implementing cost optimization strategies across operation modes.

    Provides:
    - Model selection adjustments per operation mode
    - Task prioritization algorithms for budget conservation
    - Research depth adaptation based on budget constraints
    - Graceful feature degradation for emergency modes
    """

    def __init__(self) -> None:
        """Initialize cost optimization service."""
        self.budget_manager = budget_manager

        # Model cost hierarchy (cost per 1M tokens) - GPT-4o family removed
        # Pricing placeholders for GPT-5 tiers (verify against provider docs)
        self.model_costs = {
            "openai/gpt-5-mini": {"input": 2.2, "output": 8.8},
            "openai/gpt-5-nano": {"input": 0.12, "output": 0.5},
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "perplexity/sonar-reasoning": {"input": 5.0, "output": 5.0},
            "perplexity/sonar-pro": {"input": 1.0, "output": 1.0},
            "openai/gpt-oss-20b:free": {"input": 0.0, "output": 0.0},
            "moonshotai/kimi-k2:free": {"input": 0.0, "output": 0.0},
        }

        # Model performance scores (0.0 to 1.0) - approximate; verify
        self.model_performance = {
            "openai/gpt-5-mini": 0.96,
            "openai/gpt-5-nano": 0.87,
            "claude-3-5-sonnet": 0.92,
            "claude-3-haiku": 0.78,
            "perplexity/sonar-reasoning": 0.88,
            "perplexity/sonar-pro": 0.82,
            "openai/gpt-oss-20b:free": 0.71,
            "moonshotai/kimi-k2:free": 0.69,
        }

        # Task priority weights for scoring
        self.priority_weights = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 0.8,
            TaskPriority.NORMAL: 0.6,
            TaskPriority.LOW: 0.3,
        }

        # Complexity cost multipliers
        self.complexity_multipliers = {
            TaskComplexity.MINIMAL: 0.7,
            TaskComplexity.MEDIUM: 1.0,
            TaskComplexity.HIGH: 1.5,
        }

        logger.info("Cost optimization service initialized")

    def optimize_model_selection(
        self,
        task_type: str,
        original_model: str,
        operation_mode: OperationMode,
        task_complexity: TaskComplexity = TaskComplexity.MEDIUM,
    ) -> ModelSelectionResult:
        """
        Optimize model selection based on operation mode and budget constraints.

        Args:
            task_type: Type of task (research, forecast, validation)
            original_model: Originally requested model
            operation_mode: Current operation mode
            task_complexity: Complexity level of the task

        Returns:
            ModelSelectionResult with optimized model selection
        """
        budget_status = self.budget_manager.get_budget_status()
        utilization = budget_status.utilization_percentage / 100.0

        # Get mode-specific model preferences
        mode_preferences = self._get_model_preferences_for_mode(
            operation_mode, task_type
        )

        # Calculate cost-performance scores for available models
        model_scores = {}
        for model in mode_preferences:
            if model in self.model_costs and model in self.model_performance:
                cost_score = self._calculate_cost_score(model, task_complexity)
                performance_score = self.model_performance[model]

                # Weight based on operation mode
                if operation_mode == OperationMode.NORMAL:
                    # Prioritize performance
                    combined_score = (performance_score * 0.7) + (
                        (1.0 - cost_score) * 0.3
                    )
                elif operation_mode == OperationMode.CONSERVATIVE:
                    # Balance cost and performance
                    combined_score = (performance_score * 0.5) + (
                        (1.0 - cost_score) * 0.5
                    )
                else:  # EMERGENCY
                    # Prioritize cost
                    combined_score = (performance_score * 0.3) + (
                        (1.0 - cost_score) * 0.7
                    )

                model_scores[model] = combined_score

        # Select best model
        if not model_scores:
            selected_model = original_model
            cost_reduction = 0.0
            performance_impact = 0.0
            rationale = "No alternative models available"
        else:
            selected_model = max(model_scores.keys(), key=lambda x: model_scores[x])

            # Calculate cost reduction and performance impact
            original_cost = self._get_model_cost(original_model)
            selected_cost = self._get_model_cost(selected_model)
            cost_reduction = max(0.0, (original_cost - selected_cost) / original_cost)

            original_perf = self.model_performance.get(original_model, 0.8)
            selected_perf = self.model_performance.get(selected_model, 0.8)
            performance_impact = (selected_perf - original_perf) / original_perf

            rationale = f"Selected based on {operation_mode.value} mode optimization (score: {model_scores[selected_model]:.3f})"

        return ModelSelectionResult(
            selected_model=selected_model,
            original_model=original_model,
            cost_reduction=cost_reduction,
            performance_impact=performance_impact,
            rationale=rationale,
        )

    def prioritize_task(
        self,
        task_description: str,
        task_priority: TaskPriority,
        task_complexity: TaskComplexity,
        operation_mode: OperationMode,
        estimated_tokens: int = 1000,
    ) -> TaskPrioritizationResult:
        """
        Apply task prioritization algorithm for budget conservation.

        Args:
            task_description: Description of the task
            task_priority: Priority level of the task
            task_complexity: Complexity level of the task
            operation_mode: Current operation mode
            estimated_tokens: Estimated token usage

        Returns:
            TaskPrioritizationResult with processing decision
        """
        budget_status = self.budget_manager.get_budget_status()

        # Calculate priority score
        base_priority = self.priority_weights[task_priority]
        complexity_factor = self.complexity_multipliers[task_complexity]

        # Adjust priority based on operation mode
        mode_adjustments = {
            OperationMode.NORMAL: 1.0,
            OperationMode.CONSERVATIVE: 0.8,
            OperationMode.EMERGENCY: 0.5,
        }

        priority_score = base_priority * mode_adjustments[operation_mode]

        # Estimate cost
        estimated_cost = self._estimate_task_cost(
            estimated_tokens, task_complexity, operation_mode
        )

        # Determine if task should be processed
        should_process = self._should_process_task(
            priority_score, estimated_cost, operation_mode, budget_status
        )

        # Determine resource allocation
        resource_allocation = self._calculate_resource_allocation(
            task_complexity, operation_mode, priority_score
        )

        # Generate reason
        if should_process:
            reason = f"Task approved: priority={task_priority.value}, mode={operation_mode.value}, cost=${estimated_cost:.4f}"
        else:
            reason = self._get_rejection_reason(
                task_priority, operation_mode, estimated_cost, budget_status
            )

        return TaskPrioritizationResult(
            should_process=should_process,
            priority_score=priority_score,
            reason=reason,
            estimated_cost=estimated_cost,
            resource_allocation=resource_allocation,
        )

    def adapt_research_depth(
        self,
        base_config: Dict[str, Any],
        operation_mode: OperationMode,
        task_complexity: TaskComplexity,
        budget_remaining: float,
    ) -> ResearchDepthConfig:
        """
        Adapt research depth based on budget constraints and operation mode.

        Args:
            base_config: Base research configuration
            operation_mode: Current operation mode
            task_complexity: Complexity of the research task
            budget_remaining: Remaining budget percentage

        Returns:
            ResearchDepthConfig with adapted limits
        """
        # Base configuration from input
        base_sources = base_config.get("max_sources", 10)
        base_depth = base_config.get("max_depth", 3)
        base_iterations = base_config.get("max_iterations", 5)

        # Mode-specific reduction factors
        mode_factors = {
            OperationMode.NORMAL: {
                "sources": 1.0,
                "depth": 1.0,
                "iterations": 1.0,
                "enable_deep": True,
                "time_limit": 300,
            },
            OperationMode.CONSERVATIVE: {
                "sources": 0.6,
                "depth": 0.7,
                "iterations": 0.6,
                "enable_deep": True,
                "time_limit": 180,
            },
            OperationMode.EMERGENCY: {
                "sources": 0.3,
                "depth": 0.5,
                "iterations": 0.4,
                "enable_deep": False,
                "time_limit": 90,
            },
        }

        factors = mode_factors[operation_mode]

        # Apply complexity adjustments
        complexity_adjustments = {
            TaskComplexity.MINIMAL: 0.7,
            TaskComplexity.MEDIUM: 1.0,
            TaskComplexity.HIGH: 1.2,
        }

        complexity_factor = complexity_adjustments[task_complexity]

        # Calculate adapted limits
        max_sources = max(1, int(base_sources * factors["sources"] * complexity_factor))
        max_depth = max(1, int(base_depth * factors["depth"]))
        max_iterations = max(1, int(base_iterations * factors["iterations"]))

        # Budget-based further restrictions
        if budget_remaining < 0.1:  # Less than 10% budget remaining
            max_sources = min(max_sources, 2)
            max_depth = min(max_depth, 1)
            max_iterations = min(max_iterations, 1)
            factors["enable_deep"] = False
        elif budget_remaining < 0.2:  # Less than 20% budget remaining
            max_sources = min(max_sources, 3)
            max_depth = min(max_depth, 2)
            max_iterations = min(max_iterations, 2)

        return ResearchDepthConfig(
            max_sources=max_sources,
            max_depth=max_depth,
            max_iterations=max_iterations,
            enable_deep_analysis=bool(factors["enable_deep"]),
            complexity_threshold=(
                0.7 if operation_mode == OperationMode.EMERGENCY else 0.5
            ),
            time_limit_seconds=int(factors["time_limit"]),
        )

    def get_graceful_degradation_strategy(
        self, operation_mode: OperationMode, budget_remaining: float
    ) -> Dict[str, Any]:
        """
        Get graceful feature degradation strategy for emergency modes.

        Args:
            operation_mode: Current operation mode
            budget_remaining: Remaining budget percentage

        Returns:
            Dictionary with degradation configuration
        """
        base_strategy = {
            "complexity_analysis": True,
            "multi_stage_validation": True,
            "detailed_logging": True,
            "parallel_processing": True,
            "caching_enabled": True,
            "retry_attempts": 3,
            "timeout_seconds": 90,
            "batch_size": 10,
        }

        if operation_mode == OperationMode.NORMAL:
            # No degradation in normal mode
            return base_strategy

        elif operation_mode == OperationMode.CONSERVATIVE:
            # Moderate degradation
            return {
                **base_strategy,
                "multi_stage_validation": False,
                "detailed_logging": False,
                "retry_attempts": 2,
                "timeout_seconds": 60,
                "batch_size": 5,
            }

        else:  # EMERGENCY mode
            # Aggressive degradation
            degraded_strategy = {
                "complexity_analysis": False,
                "multi_stage_validation": False,
                "detailed_logging": False,
                "parallel_processing": False,
                "caching_enabled": True,  # Keep caching for efficiency
                "retry_attempts": 1,
                "timeout_seconds": 30,
                "batch_size": 2,
            }

            # Further degradation based on remaining budget
            if budget_remaining < 0.05:  # Less than 5% remaining
                degraded_strategy.update(
                    {"caching_enabled": False, "timeout_seconds": 15, "batch_size": 1}
                )

            return degraded_strategy

    def _get_model_preferences_for_mode(
        self, operation_mode: OperationMode, task_type: str
    ) -> List[str]:
        """Get model preferences based on operation mode and task type."""
        if operation_mode == OperationMode.NORMAL:
            if task_type == "forecast":
                return [
                    "openai/gpt-5-mini",
                    "claude-3-5-sonnet",
                    "openai/gpt-5-nano",
                ]
            else:  # research, validation
                return [
                    "openai/gpt-5-nano",
                    "claude-3-haiku",
                    "openai/gpt-5-mini",
                ]

        elif operation_mode == OperationMode.CONSERVATIVE:
            # Prefer cost-efficient models
            return [
                "openai/gpt-5-nano",
                "claude-3-haiku",
                "perplexity/sonar-pro",
                "openai/gpt-oss-20b:free",
            ]

        else:  # EMERGENCY
            # Only cheapest / free models
            return [
                "openai/gpt-5-nano",
                "claude-3-haiku",
                "openai/gpt-oss-20b:free",
                "moonshotai/kimi-k2:free",
            ]

    def _calculate_cost_score(self, model: str, complexity: TaskComplexity) -> float:
        """Calculate normalized cost score (0.0 = cheapest, 1.0 = most expensive)."""
        if model not in self.model_costs:
            return 0.5  # Default middle score

        # Use average of input and output costs
        model_cost = (
            self.model_costs[model]["input"] + self.model_costs[model]["output"]
        ) / 2

        # Apply complexity multiplier
        adjusted_cost = model_cost * self.complexity_multipliers[complexity]

        # Normalize to 0-1 scale (assuming max cost of 15.0)
        return min(1.0, adjusted_cost / 15.0)

    def _get_model_cost(self, model: str) -> float:
        """Get average cost per token for a model."""
        if model not in self.model_costs:
            return 1.0  # Default cost

        return (
            self.model_costs[model]["input"] + self.model_costs[model]["output"]
        ) / 2

    def _estimate_task_cost(
        self,
        estimated_tokens: int,
        complexity: TaskComplexity,
        operation_mode: OperationMode,
    ) -> float:
        """Estimate cost for a task based on tokens and complexity."""
        # Base cost per 1000 tokens (using gpt-4o-mini as baseline)
        base_cost_per_1k = 0.375  # Average of input/output costs

        # Apply complexity multiplier
        complexity_factor = self.complexity_multipliers[complexity]

        # Apply mode efficiency factor
        mode_efficiency = {
            OperationMode.NORMAL: 1.0,
            OperationMode.CONSERVATIVE: 0.8,
            OperationMode.EMERGENCY: 0.6,
        }

        efficiency_factor = mode_efficiency[operation_mode]

        return (
            (estimated_tokens / 1000.0)
            * base_cost_per_1k
            * complexity_factor
            * efficiency_factor
        )

    def _should_process_task(
        self,
        priority_score: float,
        estimated_cost: float,
        operation_mode: OperationMode,
        budget_status: Any,
    ) -> bool:
        """Determine if a task should be processed based on priority and budget."""
        # Check if we have enough budget
        if estimated_cost > budget_status.remaining:
            return False

        # Mode-specific thresholds
        mode_thresholds = {
            OperationMode.NORMAL: 0.3,  # Process tasks with priority >= 0.3
            OperationMode.CONSERVATIVE: 0.5,  # Process tasks with priority >= 0.5
            OperationMode.EMERGENCY: 0.7,  # Process tasks with priority >= 0.7
        }

        return priority_score >= mode_thresholds[operation_mode]

    def _calculate_resource_allocation(
        self,
        complexity: TaskComplexity,
        operation_mode: OperationMode,
        priority_score: float,
    ) -> Dict[str, Any]:
        """Calculate resource allocation for a task."""
        base_allocation = {
            "cpu_priority": "normal",
            "memory_limit_mb": 512,
            "timeout_seconds": 90,
            "max_retries": 3,
        }

        # Adjust based on complexity
        if complexity == TaskComplexity.HIGH:
            base_allocation.update(
                {
                    "cpu_priority": "high",
                    "memory_limit_mb": 1024,
                    "timeout_seconds": 180,
                }
            )
        elif complexity == TaskComplexity.MINIMAL:
            base_allocation.update(
                {"cpu_priority": "low", "memory_limit_mb": 256, "timeout_seconds": 30}
            )

        # Adjust based on operation mode
        if operation_mode == OperationMode.EMERGENCY:
            def _to_int(v: Any, default: int) -> int:
                if isinstance(v, int):
                    return v
                if isinstance(v, float):
                    return int(v)
                try:
                    return int(str(v))
                except Exception:
                    return default

            mem_int = _to_int(base_allocation.get("memory_limit_mb", 256), 256)
            timeout_int = _to_int(base_allocation.get("timeout_seconds", 45), 45)
            base_allocation.update(
                {
                    "cpu_priority": "low",
                    "memory_limit_mb": min(mem_int, 256),
                    "timeout_seconds": min(timeout_int, 45),
                    "max_retries": 1,
                }
            )
        elif operation_mode == OperationMode.CONSERVATIVE:
            def _to_int2(v: Any, default: int) -> int:
                if isinstance(v, int):
                    return v
                if isinstance(v, float):
                    return int(v)
                try:
                    return int(str(v))
                except Exception:
                    return default
            timeout_int2 = _to_int2(base_allocation.get("timeout_seconds", 60), 60)
            base_allocation.update(
                {
                    "timeout_seconds": min(timeout_int2, 60),
                    "max_retries": 2,
                }
            )

        return base_allocation

    def _get_rejection_reason(
        self, priority: TaskPriority, mode: OperationMode, cost: float, budget_status: Any
    ) -> str:
        """Generate reason for task rejection."""
        if cost > budget_status.remaining:
            return f"Insufficient budget: task costs ${cost:.4f}, only ${budget_status.remaining:.4f} remaining"

        mode_reasons = {
            OperationMode.CONSERVATIVE: f"Conservative mode: {priority.value} priority tasks not processed",
            OperationMode.EMERGENCY: f"Emergency mode: only critical/high priority tasks processed",
        }

        return mode_reasons.get(
            mode,
            f"Task priority {priority.value} below threshold for {mode.value} mode",
        )


# Global instance
cost_optimization_service = CostOptimizationService()
