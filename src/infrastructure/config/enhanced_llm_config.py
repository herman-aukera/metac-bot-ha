from typing import Any, Dict, Tuple

# Default model constants to avoid duplicated literals
DEFAULT_GPT5 = "openai/gpt-5"
DEFAULT_GPT5_MINI = "openai/gpt-5-mini"
DEFAULT_GPT5_NANO = "openai/gpt-5-nano"

"""
Enhanced LLM configuration with budget management and smart model selection.
"""

import logging
import asyncio
import os

from .api_keys import api_key_manager
from .budget_manager import budget_manager
from .operation_modes import operation_mode_manager
from .task_complexity_analyzer import ComplexityLevel, task_complexity_analyzer
from .token_tracker import token_tracker

logger = logging.getLogger(__name__)


class EnhancedLLMConfig:
    """Enhanced LLM configuration with budget-aware model selection."""

    def __init__(self):
        """Initialize enhanced LLM configuration."""
        self.budget_manager = budget_manager
        self.token_tracker = token_tracker
        self.complexity_analyzer = task_complexity_analyzer
        self.operation_mode_manager = operation_mode_manager

        # Get OpenRouter API key
        self.mock_mode = False
        self.openrouter_key = api_key_manager.get_api_key("OPENROUTER_API_KEY")

        # Allow operation in DRY_RUN or local CI without a real key by switching to mock mode
        is_dry_run = os.getenv("DRY_RUN", "false").lower() == "true"
        local_network_enabled = os.getenv("LOCAL_CI_NETWORK", "0") in ("1", "true")

        if not self.openrouter_key:
            if is_dry_run and not local_network_enabled:
                # In local dry-run without network, don't hard-fail – use mock LLMs
                self.mock_mode = True
                self.openrouter_key = "dummy_local_dry_run"
                logger.warning(
                    "OPENROUTER_API_KEY missing, entering mock LLM mode for DRY_RUN without network"
                )
            else:
                logger.error(
                    "OpenRouter API key not found! This is required for tournament operation."
                )
                raise ValueError("OpenRouter API key is required")

        # Model configuration based on budget status
        self.model_configs = self._setup_model_configs()

        logger.info(
            "Enhanced LLM configuration initialized with task complexity analyzer"
        )

    def _setup_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Setup model configurations with cost optimization."""
        return {
            "research": {
                "model": os.getenv("PRIMARY_RESEARCH_MODEL", DEFAULT_GPT5_MINI),
                "temperature": 0.3,
                "max_tokens": 1500,
                "timeout": 60,
                "allowed_tries": 3,
                "cost_tier": "low",
            },
            "forecast": {
                "model": os.getenv("PRIMARY_FORECAST_MODEL", DEFAULT_GPT5),
                "max_tokens": 2000,
                "timeout": 90,
                "allowed_tries": 2,
                "cost_tier": "high",
            },
            "simple": {
                "model": os.getenv("SIMPLE_TASK_MODEL", DEFAULT_GPT5_NANO),
                "temperature": 0.1,
                "max_tokens": 1000,
                "timeout": 45,
                "cost_tier": "low",
            },
            "summarizer": {
                "model": DEFAULT_GPT5_NANO,
                "temperature": 0.0,
                "max_tokens": 800,
                "timeout": 45,
                "allowed_tries": 3,
                "cost_tier": "low",
            },
        }

    def get_llm_for_task(
        self,
        task_type: str,
        _question_complexity: str = "medium",
        complexity_assessment=None,
    ):
        """Get appropriate LLM based on task type, complexity assessment, and operation mode."""
        # If running in mock mode (e.g., local DRY_RUN without network), return an async mock LLM
        if getattr(self, "mock_mode", False):
            class AsyncMockLLM:
                def __init__(self, **kwargs):
                    self.model = kwargs.get("model", "mock-model")
                    self.temperature = kwargs.get("temperature", 0.1)
                    self.max_tokens = kwargs.get("max_tokens", 1000)

                async def invoke(self, prompt: str) -> str:
                    # Deterministic short response for tests/dry-runs
                    await asyncio.sleep(0)
                    return (
                        "[MOCK-LLM] Deterministic response for "
                        f"{task_type} (temp={self.temperature}, max_tokens={self.max_tokens})\n"
                        + (prompt[:300] if isinstance(prompt, str) else "")
                    )

            # Use the configured model for the task, but serve through the mock
            model_config = self.model_configs.get(task_type, self.model_configs["simple"]).copy()
            return AsyncMockLLM(**model_config)

        # Check and update operation mode based on current budget
        mode_changed, transition = self.operation_mode_manager.check_and_update_mode()
        if mode_changed and transition is not None:
            try:
                logger.warning(
                    f"Operation mode automatically changed: {transition.from_mode.value} → {transition.to_mode.value}"
                )
            except Exception:
                logger.warning("Operation mode automatically changed")

        # Get model from operation mode manager (integrates complexity analysis)
        recommended_model = self.operation_mode_manager.get_model_for_task(
            task_type, complexity_assessment
        )
        model_config = self._get_model_config_for_model(recommended_model)

        # Apply operation mode limits
        processing_limits = self.operation_mode_manager.get_processing_limits()
        model_config["max_tokens"] = min(
            model_config.get("max_tokens", 2000),
            2000 if processing_limits["enable_complexity_analysis"] else 1500,
        )
        model_config["timeout"] = processing_limits["timeout_seconds"]
        model_config["allowed_tries"] = processing_limits["max_retries"]

        logger.info(
            f"Using operation mode {self.operation_mode_manager.current_mode.value}: "
            f"{recommended_model} for {task_type}"
        )

        # Create LLM via centralized factory to ensure normalization and OpenRouter routing
        try:
            from .llm_factory import create_llm

            model_name = model_config.get("model")
            if not isinstance(model_name, str) or not model_name:
                raise ValueError("Model name missing from configuration")

            return create_llm(
                model=model_name,
                temperature=model_config.get("temperature", 0.1),
                timeout=model_config.get("timeout", 60),
                allowed_tries=model_config.get("allowed_tries", 2),
            )
        except Exception as e:
            logger.error(f"Failed to create LLM via factory: {e}")

            # Return an async mock for testing to match call sites (await llm.invoke)
            class AsyncMockLLMImportFallback:
                def __init__(self, **kwargs):
                    self.model = kwargs.get("model", "mock-model")
                    self.temperature = kwargs.get("temperature", 0.1)
                    self.max_tokens = kwargs.get("max_tokens", 1000)

                async def invoke(self, prompt: str) -> str:
                    await asyncio.sleep(0)
                    return (
                        "[MOCK-LLM] Deterministic response (factory fallback)\n"
                        + (prompt[:300] if isinstance(prompt, str) else "")
                    )

            return AsyncMockLLMImportFallback(**model_config)

    def estimate_task_cost(
        self,
        prompt: str,
        task_type: str,
        question_complexity: str = "medium",
        complexity_assessment=None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Estimate cost for a task before execution using complexity analysis."""
        budget_status = self.budget_manager.get_budget_status()

        # Use complexity assessment if provided
        if complexity_assessment is not None:
            cost_estimate = self.complexity_analyzer.estimate_cost_per_task(
                complexity_assessment, task_type, budget_status.status_level
            )

            # Get actual cost using budget manager
            estimated_cost = self.budget_manager.estimate_cost(
                cost_estimate["model"],
                cost_estimate["input_tokens"],
                cost_estimate["output_tokens"],
            )

            return estimated_cost, {
                "model": cost_estimate["model"],
                "input_tokens": cost_estimate["input_tokens"],
                "estimated_output_tokens": cost_estimate["output_tokens"],
                "budget_status": budget_status.status_level,
                "complexity": cost_estimate["complexity"],
                "complexity_score": complexity_assessment.score,
            }
        else:
            # Fallback to original logic
            if budget_status.status_level == "emergency":
                model_name = DEFAULT_GPT5_NANO
            elif budget_status.status_level == "conservative":
                if task_type == "forecast" and question_complexity == "complex":
                    model_name = self.model_configs["forecast"]["model"]
                else:
                    model_name = self.model_configs["simple"]["model"]
            else:
                model_name = self.model_configs.get(
                    task_type, self.model_configs["simple"]
                )["model"]

            # Estimate tokens
            token_estimate = self.token_tracker.estimate_tokens_for_prompt(
                prompt, model_name
            )

            # Estimate cost
            estimated_cost = self.budget_manager.estimate_cost(
                model_name,
                token_estimate["input_tokens"],
                token_estimate["estimated_output_tokens"],
            )

            return estimated_cost, {
                "model": model_name,
                "input_tokens": token_estimate["input_tokens"],
                "estimated_output_tokens": token_estimate["estimated_output_tokens"],
                "budget_status": budget_status.status_level,
            }

    def can_afford_task(
        self,
        prompt: str,
        task_type: str,
        question_complexity: str = "medium",
        complexity_assessment=None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if we can afford to execute a task."""
        estimated_cost, details = self.estimate_task_cost(
            prompt, task_type, question_complexity, complexity_assessment
        )
        can_afford = self.budget_manager.can_afford(estimated_cost)

        details["estimated_cost"] = estimated_cost
        details["can_afford"] = can_afford

        return can_afford, details

    def record_task_completion(
        self,
        question_id: str,
        prompt: str,
        response: str,
        task_type: str,
        model_used: str,
        success: bool = True,
    ) -> float:
        """Record completion of a task for budget tracking."""
        # Track actual token usage
        actual_usage = self.token_tracker.track_actual_usage(
            prompt, response, model_used
        )

        # Record cost
        actual_cost = self.budget_manager.record_cost(
            question_id=question_id,
            model=model_used,
            input_tokens=actual_usage["input_tokens"],
            output_tokens=actual_usage["output_tokens"],
            task_type=task_type,
            success=success,
        )

        return actual_cost

    def get_fallback_models(self) -> Dict[str, str]:
        """Get fallback models for different scenarios."""
        return {
            # Prefer GPT-5 tiers + free fallbacks; purge GPT-4o(mini) usage per policy
            "emergency": DEFAULT_GPT5_NANO,
            "conservative": DEFAULT_GPT5_NANO,
            # Use nano as proxy fallback (cheapest) until explicit proxy GPT-5 mapping is available
            "proxy_fallback": DEFAULT_GPT5_NANO,
            "last_resort": "openai/gpt-5-mini",
        }

    def _get_model_config_for_model(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration for a specific model name."""
        # Find matching config or create default
        for config_name, config in self.model_configs.items():
            if config["model"] == model_name:
                return config.copy()

        # Create default config for unknown models
        return {
            "model": model_name,
            "temperature": 0.1,
            "max_tokens": 1500,
            "timeout": 60,
            "allowed_tries": 2,
            "cost_tier": "medium",
        }

    def assess_question_complexity(
        self,
        question_text: str,
        background_info: str = "",
        resolution_criteria: str = "",
        fine_print: str = "",
    ):
        """Assess question complexity using the advanced complexity analyzer."""
        return self.complexity_analyzer.assess_question_complexity(
            question_text, background_info, resolution_criteria, fine_print
        )

    def assess_question_complexity_simple(
        self, question_text: str, background_info: str = ""
    ) -> str:
        """Simple complexity assessment for backward compatibility."""
        assessment = self.assess_question_complexity(question_text, background_info)
        return assessment.level.value

    def analyze_question_for_forecasting(
        self,
        question_id: str,
        question_text: str,
        background_info: str = "",
        resolution_criteria: str = "",
        fine_print: str = "",
    ) -> Dict[str, Any]:
        """Perform comprehensive complexity analysis for a forecasting question."""
        # Get complexity assessment
        complexity_assessment = self.assess_question_complexity(
            question_text, background_info, resolution_criteria, fine_print
        )

        # Log the assessment
        self.complexity_analyzer.log_complexity_assessment(
            question_id, complexity_assessment
        )

        # Get cost estimates for different task types
        budget_status = self.budget_manager.get_budget_status()

        research_cost = self.complexity_analyzer.estimate_cost_per_task(
            complexity_assessment, "research", budget_status.status_level
        )

        forecast_cost = self.complexity_analyzer.estimate_cost_per_task(
            complexity_assessment, "forecast", budget_status.status_level
        )

        return {
            "question_id": question_id,
            "complexity_assessment": complexity_assessment,
            "research_cost_estimate": research_cost,
            "forecast_cost_estimate": forecast_cost,
            "total_estimated_cost": research_cost["estimated_cost"]
            + forecast_cost["estimated_cost"],
            "budget_status": budget_status.status_level,
            "can_afford": self.budget_manager.can_afford(
                research_cost["estimated_cost"] + forecast_cost["estimated_cost"]
            ),
        }

    def can_process_question(
        self, question_priority: str = "normal"
    ) -> Tuple[bool, str]:
        """Check if a question can be processed based on current operation mode."""
        return self.operation_mode_manager.can_process_question(question_priority)

    def log_configuration_status(self):
        """Log current configuration status."""
        budget_status = self.budget_manager.get_budget_status()

        logger.info("=== Enhanced LLM Configuration Status ===")
        logger.info(
            f"OpenRouter API Key: {'✓ Configured' if self.openrouter_key else '✗ Missing'}"
        )
        logger.info(f"Budget Status: {budget_status.status_level.upper()}")
        logger.info(f"Budget Utilization: {budget_status.utilization_percentage:.1f}%")
        logger.info(
            f"Task Complexity Analyzer: {'✓ Active' if self.complexity_analyzer else '✗ Missing'}"
        )

        # Log operation mode status
        self.operation_mode_manager.log_mode_status()

        # Log current model selection for different complexity levels
        for complexity in [
            ComplexityLevel.SIMPLE,
            ComplexityLevel.MEDIUM,
            ComplexityLevel.COMPLEX,
        ]:
            for task_type in ["research", "forecast"]:
                mock_assessment = type("MockAssessment", (), {"level": complexity})()
                model = self.operation_mode_manager.get_model_for_task(
                    task_type, mock_assessment
                )
                logger.info(f"{complexity.value.capitalize()} {task_type}: {model}")


# Global instance
enhanced_llm_config = EnhancedLLMConfig()
