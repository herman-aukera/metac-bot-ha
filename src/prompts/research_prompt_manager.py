"""Research prompt management for budget-optimized forecasting.

This module provides a unified interface for selecting and using optimized
research prompts based on question characteristics and budget constraints.
"""

from typing import Dict, Any, Optional
from ..domain.entities.question import Question
from .optimized_research_prompts import OptimizedResearchPrompts, QuestionComplexityAnalyzer


class ResearchPromptManager:
    """
    Manages selection and usage of optimized research prompts.

    This class provides intelligent prompt selection based on:
    - Question complexity
    - Budget constraints
    - Time sensitivity
    - Research focus requirements
    """

    def __init__(self, budget_aware: bool = True):
        """
        Initialize the research prompt manager.

        Args:
            budget_aware: Whether to consider budget constraints in prompt selection
        """
        self.optimized_prompts = OptimizedResearchPrompts()
        self.complexity_analyzer = QuestionComplexityAnalyzer()
        self.budget_aware = budget_aware

        # Token cost estimates for different models (per 1K tokens)
        self.model_costs = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
        }

    def get_optimal_research_prompt(self, question: Question,
                                  budget_remaining: Optional[float] = None,
                                  force_complexity: Optional[str] = None,
                                  force_focus: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the optimal research prompt for a given question and constraints.

        Args:
            question: The forecasting question
            budget_remaining: Remaining budget in dollars (optional)
            force_complexity: Force specific complexity level (optional)
            force_focus: Force specific focus type (optional)

        Returns:
            Dictionary containing prompt, metadata, and cost estimates
        """
        # Analyze question characteristics
        complexity = force_complexity or self.complexity_analyzer.analyze_complexity(question)
        focus_type = force_focus or self.complexity_analyzer.determine_focus_type(question)

        # Apply budget constraints if enabled
        if self.budget_aware and budget_remaining is not None:
            complexity = self._apply_budget_constraints(complexity, budget_remaining)

        # Get the appropriate prompt
        prompt = self.optimized_prompts.get_research_prompt(
            question=question,
            complexity_level=complexity,
            focus_type=focus_type
        )

        # Get token estimates
        token_estimates = self.optimized_prompts.estimate_token_usage(complexity)

        # Calculate cost estimates for different models
        cost_estimates = self._calculate_cost_estimates(token_estimates)

        return {
            "prompt": prompt,
            "complexity_level": complexity,
            "focus_type": focus_type,
            "token_estimates": token_estimates,
            "cost_estimates": cost_estimates,
            "recommended_model": self._recommend_model(complexity, budget_remaining),
            "metadata": {
                "question_id": getattr(question, 'id', None),
                "question_type": question.question_type.value,
                "categories": question.categories,
                "close_date": question.close_time.isoformat() if question.close_time else None
            }
        }

    def get_research_prompt_by_type(self, question: Question,
                                  prompt_type: str) -> Dict[str, Any]:
        """
        Get a specific type of research prompt.

        Args:
            question: The forecasting question
            prompt_type: Type of prompt ("simple", "standard", "comprehensive", "news", "base_rate")

        Returns:
            Dictionary containing prompt and metadata
        """
        if prompt_type == "simple":
            prompt = self.optimized_prompts.get_simple_research_prompt(question)
        elif prompt_type == "standard":
            prompt = self.optimized_prompts.get_standard_research_prompt(question)
        elif prompt_type == "comprehensive":
            prompt = self.optimized_prompts.get_comprehensive_research_prompt(question)
        elif prompt_type == "news":
            prompt = self.optimized_prompts.get_news_focused_prompt(question)
        elif prompt_type == "base_rate":
            prompt = self.optimized_prompts.get_base_rate_prompt(question)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        token_estimates = self.optimized_prompts.estimate_token_usage(prompt_type)
        cost_estimates = self._calculate_cost_estimates(token_estimates)

        return {
            "prompt": prompt,
            "prompt_type": prompt_type,
            "token_estimates": token_estimates,
            "cost_estimates": cost_estimates,
            "metadata": {
                "question_id": getattr(question, 'id', None),
                "question_type": question.question_type.value
            }
        }

    def _apply_budget_constraints(self, complexity: str, budget_remaining: float) -> str:
        """
        Apply budget constraints to complexity selection.

        Args:
            complexity: Original complexity level
            budget_remaining: Remaining budget in dollars

        Returns:
            Adjusted complexity level based on budget constraints
        """
        # Conservative budget thresholds
        if budget_remaining < 10:  # Less than $10 remaining
            return "simple"
        elif budget_remaining < 25:  # Less than $25 remaining
            if complexity == "comprehensive":
                return "standard"

        return complexity

    def _calculate_cost_estimates(self, token_estimates: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Calculate cost estimates for different models.

        Args:
            token_estimates: Dictionary with input_tokens and expected_output estimates

        Returns:
            Dictionary with cost estimates per model
        """
        input_tokens = token_estimates["input_tokens"]
        output_tokens = token_estimates["expected_output"]

        cost_estimates = {}
        for model, rates in self.model_costs.items():
            input_cost = (input_tokens / 1000) * rates["input"]
            output_cost = (output_tokens / 1000) * rates["output"]
            total_cost = input_cost + output_cost

            cost_estimates[model] = {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost
            }

        return cost_estimates

    def _recommend_model(self, complexity: str, budget_remaining: Optional[float]) -> str:
        """
        Recommend the appropriate model based on complexity and budget.

        Args:
            complexity: Question complexity level
            budget_remaining: Remaining budget in dollars

        Returns:
            Recommended model name
        """
        # If budget is very low, always use mini
        if budget_remaining is not None and budget_remaining < 15:
            return "gpt-4o-mini"

        # For simple questions, use mini
        if complexity == "simple":
            return "gpt-4o-mini"

        # For standard questions, use mini for research (this is research phase)
        if complexity == "standard":
            return "gpt-4o-mini"

        # For comprehensive questions, consider using gpt-4o if budget allows
        if complexity == "comprehensive":
            if budget_remaining is None or budget_remaining > 30:
                return "gpt-4o"
            else:
                return "gpt-4o-mini"

        return "gpt-4o-mini"  # Default to mini for cost efficiency

    def get_prompt_efficiency_metrics(self) -> Dict[str, Any]:
        """
        Get efficiency metrics for different prompt types.

        Returns:
            Dictionary with efficiency metrics and recommendations
        """
        prompt_types = ["simple", "standard", "comprehensive", "news", "base_rate"]
        metrics = {}

        for prompt_type in prompt_types:
            token_est = self.optimized_prompts.estimate_token_usage(prompt_type)
            cost_est = self._calculate_cost_estimates(token_est)

            metrics[prompt_type] = {
                "tokens_per_dollar": {
                    "gpt-4o": (token_est["input_tokens"] + token_est["expected_output"]) / cost_est["gpt-4o"]["total_cost"],
                    "gpt-4o-mini": (token_est["input_tokens"] + token_est["expected_output"]) / cost_est["gpt-4o-mini"]["total_cost"]
                },
                "cost_per_question": cost_est,
                "token_estimates": token_est
            }

        return {
            "prompt_metrics": metrics,
            "recommendations": {
                "most_efficient": "simple",
                "best_balance": "standard",
                "highest_quality": "comprehensive",
                "time_sensitive": "news",
                "historical_context": "base_rate"
            }
        }
