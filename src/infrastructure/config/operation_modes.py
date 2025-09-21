"""
Budget-aware operation modes for tournament API optimization.
Implements normal, conservative, and emergency operation modes with automatic switching.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .budget_manager import budget_manager
from .task_complexity_analyzer import task_complexity_analyzer

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    """Available operation modes based on budget utilization."""

    NORMAL = "normal"
    CONSERVATIVE = "conservative"
    EMERGENCY = "emergency"


@dataclass
class OperationModeConfig:
    """Configuration for each operation mode."""

    mode: OperationMode
    budget_threshold: float  # Budget utilization threshold to trigger this mode
    max_questions_per_batch: int
    research_model: str
    forecast_model: str
    max_retries: int
    timeout_seconds: int
    enable_complexity_analysis: bool
    skip_low_priority_questions: bool
    description: str


@dataclass
class ModeTransition:
    """Record of operation mode transitions."""

    timestamp: datetime
    from_mode: OperationMode
    to_mode: OperationMode
    budget_utilization: float
    trigger_reason: str


class OperationModeManager:
    """Manages budget-aware operation modes with automatic switching."""

    def __init__(self):
        """Initialize operation mode manager."""
        self.budget_manager = budget_manager
        self.complexity_analyzer = task_complexity_analyzer
        self.current_mode = OperationMode.NORMAL
        self.mode_transitions = []

        # Setup mode configurations
        self.mode_configs = self._setup_mode_configurations()

        # Initialize with current budget status
        self._update_mode_based_on_budget()

        logger.info(
            f"Operation mode manager initialized in {self.current_mode.value} mode"
        )

    def _setup_mode_configurations(self) -> Dict[OperationMode, OperationModeConfig]:
        """Setup configurations for each operation mode."""
        return {
            OperationMode.NORMAL: OperationModeConfig(
                mode=OperationMode.NORMAL,
                budget_threshold=0.0,  # No threshold - default mode
                max_questions_per_batch=10,
                # Cost policy: GPT-5 tiers only (simple→nano, research→mini, complex→full)
                research_model="openai/gpt-5-mini",
                forecast_model="openai/gpt-5",
                max_retries=3,
                timeout_seconds=90,
                enable_complexity_analysis=True,
                skip_low_priority_questions=False,
                description="Full functionality with optimal model selection",
            ),
            OperationMode.CONSERVATIVE: OperationModeConfig(
                mode=OperationMode.CONSERVATIVE,
                budget_threshold=0.80,  # Trigger at 80% budget utilization
                max_questions_per_batch=5,
                research_model="openai/gpt-5-mini",
                forecast_model="openai/gpt-5-nano",  # Downgrade forecast model (nano)
                max_retries=2,
                timeout_seconds=60,
                enable_complexity_analysis=True,
                skip_low_priority_questions=True,
                description="Reduced functionality to conserve budget",
            ),
            OperationMode.EMERGENCY: OperationModeConfig(
                mode=OperationMode.EMERGENCY,
                budget_threshold=0.95,  # Trigger at 95% budget utilization
                max_questions_per_batch=2,
                research_model="openai/gpt-5-nano",
                forecast_model="openai/gpt-5-nano",
                max_retries=1,
                timeout_seconds=45,
                enable_complexity_analysis=False,  # Disable to save processing
                skip_low_priority_questions=True,
                description="Minimal functionality to preserve remaining budget",
            ),
        }

    def get_current_mode(self) -> OperationMode:
        """Get the current operation mode."""
        return self.current_mode

    def get_mode_config(
        self, mode: Optional[OperationMode] = None
    ) -> OperationModeConfig:
        """Get configuration for specified mode or current mode."""
        target_mode = mode or self.current_mode
        return self.mode_configs[target_mode]

    def check_and_update_mode(self) -> Tuple[bool, Optional[ModeTransition]]:
        """Check budget status and update operation mode if needed."""
        budget_status = self.budget_manager.get_budget_status()
        utilization = budget_status.utilization_percentage / 100.0

        # Determine appropriate mode based on budget utilization
        new_mode = self._determine_mode_from_utilization(utilization)

        if new_mode != self.current_mode:
            transition = self._transition_to_mode(
                new_mode, utilization, "budget_threshold"
            )
            return True, transition

        return False, None

    def _determine_mode_from_utilization(self, utilization: float) -> OperationMode:
        """Determine appropriate operation mode based on budget utilization."""
        if utilization >= self.mode_configs[OperationMode.EMERGENCY].budget_threshold:
            return OperationMode.EMERGENCY
        elif (
            utilization
            >= self.mode_configs[OperationMode.CONSERVATIVE].budget_threshold
        ):
            return OperationMode.CONSERVATIVE
        else:
            return OperationMode.NORMAL

    def _transition_to_mode(
        self, new_mode: OperationMode, utilization: float, reason: str
    ) -> ModeTransition:
        """Transition to a new operation mode."""
        old_mode = self.current_mode

        transition = ModeTransition(
            timestamp=datetime.now(),
            from_mode=old_mode,
            to_mode=new_mode,
            budget_utilization=utilization,
            trigger_reason=reason,
        )

        self.mode_transitions.append(transition)
        self.current_mode = new_mode

        logger.warning(
            f"Operation mode changed: {old_mode.value} → {new_mode.value} "
            f"(utilization: {utilization:.1%}, reason: {reason})"
        )

        return transition

    def _update_mode_based_on_budget(self):
        """Update mode based on current budget status."""
        budget_status = self.budget_manager.get_budget_status()
        utilization = budget_status.utilization_percentage / 100.0

        appropriate_mode = self._determine_mode_from_utilization(utilization)

        if appropriate_mode != self.current_mode:
            self._transition_to_mode(appropriate_mode, utilization, "initialization")

    def force_mode_transition(
        self, target_mode: OperationMode, reason: str = "manual"
    ) -> ModeTransition:
        """Force transition to a specific mode (for testing or manual override)."""
        budget_status = self.budget_manager.get_budget_status()
        utilization = budget_status.utilization_percentage / 100.0

        return self._transition_to_mode(target_mode, utilization, reason)

    def can_process_question(
        self, question_priority: str = "normal"
    ) -> Tuple[bool, str]:
        """Check if a question can be processed in current mode."""
        config = self.get_mode_config()

        # Check budget availability
        budget_status = self.budget_manager.get_budget_status()
        if budget_status.remaining <= 0:
            return False, "No budget remaining"

        # In emergency mode, only process high priority questions
        if (
            self.current_mode == OperationMode.EMERGENCY
            and question_priority.lower() not in ["high", "critical"]
        ):
            return (
                False,
                f"Emergency mode: skipping {question_priority} priority question",
            )

        # In conservative mode, skip low priority questions
        if (
            self.current_mode == OperationMode.CONSERVATIVE
            and config.skip_low_priority_questions
            and question_priority.lower() == "low"
        ):
            return False, "Conservative mode: skipping low priority question"

        return True, "Question can be processed"

    def get_model_for_task(self, task_type: str, complexity_assessment=None) -> str:
        """Get appropriate model for task based on current operation mode."""
        config = self.get_mode_config()

        # If complexity analysis is disabled in current mode, use mode defaults
        if not config.enable_complexity_analysis or complexity_assessment is None:
            if task_type == "research":
                return config.research_model
            elif task_type == "forecast":
                return config.forecast_model
            else:
                return config.research_model  # Default to research model

        # Use complexity-based selection with mode constraints
        recommended_model = self.complexity_analyzer.get_model_for_task(
            task_type, complexity_assessment, self.current_mode.value
        )

        # Override with mode constraints if needed
        if self.current_mode == OperationMode.EMERGENCY:
            # Always use cheapest model in emergency per new GPT-5 policy
            return "openai/gpt-5-nano"
        elif self.current_mode == OperationMode.CONSERVATIVE:
            # Limit expensive models in conservative mode – downgrade GPT-5 full to mini
            if recommended_model == "openai/gpt-5" and task_type == "research":
                return "openai/gpt-5-mini"

        return recommended_model

    def get_processing_limits(self) -> Dict[str, Any]:
        """Get processing limits for current operation mode."""
        config = self.get_mode_config()

        return {
            "max_questions_per_batch": config.max_questions_per_batch,
            "max_retries": config.max_retries,
            "timeout_seconds": config.timeout_seconds,
            "enable_complexity_analysis": config.enable_complexity_analysis,
            "skip_low_priority_questions": config.skip_low_priority_questions,
        }

    def estimate_question_cost(
        self, question_text: str, task_type: str
    ) -> Dict[str, Any]:
        """Estimate cost for processing a question in current mode."""
        config = self.get_mode_config()

        # Get complexity assessment if enabled
        complexity_assessment = None
        if config.enable_complexity_analysis:
            complexity_assessment = self.complexity_analyzer.assess_question_complexity(
                question_text
            )

        # Get model for task
        model = self.get_model_for_task(task_type, complexity_assessment)

        # Estimate tokens (simplified)
        base_tokens = {
            "research": {"input": 1200, "output": 800},
            "forecast": {"input": 1000, "output": 500},
        }

        tokens = base_tokens.get(task_type, base_tokens["research"])

        # Estimate cost using budget manager
        estimated_cost = self.budget_manager.estimate_cost(
            model, tokens["input"], tokens["output"]
        )

        return {
            "model": model,
            "estimated_cost": estimated_cost,
            "input_tokens": tokens["input"],
            "output_tokens": tokens["output"],
            "complexity": (
                complexity_assessment.level.value
                if complexity_assessment
                else "unknown"
            ),
            "operation_mode": self.current_mode.value,
        }

    def get_graceful_degradation_strategy(self) -> Dict[str, Any]:
        """Get strategy for graceful degradation when approaching budget limits."""
        budget_status = self.budget_manager.get_budget_status()
        utilization = budget_status.utilization_percentage / 100.0

        strategy = {
            "current_mode": self.current_mode.value,
            "budget_utilization": utilization,
            "actions": [],
        }

        if utilization >= 0.95:
            strategy["actions"].extend(
                [
                    "Process only critical priority questions",
                    "Use minimal model (gpt-5-nano) for all tasks",
                    "Disable complexity analysis",
                    "Reduce batch size to 2 questions",
                    "Single retry attempt only",
                ]
            )
        elif utilization >= 0.80:
            strategy["actions"].extend(
                [
                    "Skip low priority questions",
                    "Use cost-efficient models",
                    "Reduce batch size to 5 questions",
                    "Limit retries to 2 attempts",
                ]
            )
        else:
            strategy["actions"].append("Normal operation - no degradation needed")

        return strategy

    def log_mode_status(self):
        """Log current operation mode status."""
        config = self.get_mode_config()
        budget_status = self.budget_manager.get_budget_status()

        logger.info("=== Operation Mode Status ===")
        logger.info(f"Current Mode: {self.current_mode.value.upper()}")
        logger.info(f"Description: {config.description}")
        logger.info(f"Budget Utilization: {budget_status.utilization_percentage:.1f}%")
        logger.info(f"Max Questions/Batch: {config.max_questions_per_batch}")
        logger.info(f"Research Model: {config.research_model}")
        logger.info(f"Forecast Model: {config.forecast_model}")
        logger.info(f"Max Retries: {config.max_retries}")
        logger.info(f"Timeout: {config.timeout_seconds}s")
        logger.info(
            f"Complexity Analysis: {'Enabled' if config.enable_complexity_analysis else 'Disabled'}"
        )
        logger.info(
            f"Skip Low Priority: {'Yes' if config.skip_low_priority_questions else 'No'}"
        )

        # Log recent transitions
        if self.mode_transitions:
            recent_transitions = self.mode_transitions[-3:]  # Last 3 transitions
            logger.info("Recent Mode Transitions:")
            for transition in recent_transitions:
                logger.info(
                    f"  {transition.timestamp.strftime('%H:%M:%S')}: "
                    f"{transition.from_mode.value} → {transition.to_mode.value} "
                    f"({transition.budget_utilization:.1%}, {transition.trigger_reason})"
                )

    def get_mode_history(self) -> list[ModeTransition]:
        """Get history of mode transitions."""
        return self.mode_transitions.copy()

    def reset_mode_history(self):
        """Reset mode transition history (for testing)."""
        self.mode_transitions.clear()
        logger.info("Operation mode history reset")


# Global instance
operation_mode_manager = OperationModeManager()
