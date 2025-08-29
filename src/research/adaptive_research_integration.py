"""Integration layer for adaptive research depth with existing prompt systems.

This module provides a unified interface that combines adaptive research depth
determination with optimized prompt selection.
"""

from typing import Any, Dict, List, Optional

from ..domain.entities.question import Question
from ..domain.entities.research_report import ResearchSource
from ..prompts.research_prompt_manager import ResearchPromptManager
from .adaptive_research_manager import AdaptiveResearchManager


class AdaptiveResearchIntegration:
    """
    Integrates adaptive research depth with prompt management for optimal budget efficiency.
    """

    def __init__(self, budget_aware: bool = True):
        self.research_manager = AdaptiveResearchManager(budget_aware=budget_aware)
        self.prompt_manager = ResearchPromptManager(budget_aware=budget_aware)

    def get_adaptive_research_plan(
        self, question: Question, budget_remaining: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive adaptive research plan combining depth analysis and prompt optimization.

        Args:
            question: The forecasting question
            budget_remaining: Remaining budget in dollars

        Returns:
            Dictionary with complete research plan and cost estimates
        """
        # Determine research strategy
        research_strategy = self.research_manager.determine_research_strategy(
            question, budget_remaining
        )

        # Get optimal research prompt
        prompt_info = self.prompt_manager.get_optimal_research_prompt(
            question, budget_remaining
        )

        # Combine strategies
        research_plan = {
            "question_analysis": {
                "complexity_score": research_strategy["complexity_analysis"][
                    "total_complexity_score"
                ],
                "complexity_factors": research_strategy["complexity_analysis"][
                    "complexity_factors"
                ],
                "priority_areas": research_strategy["complexity_analysis"][
                    "research_priority_areas"
                ],
            },
            "research_strategy": {
                "depth_level": research_strategy["research_depth"],
                "max_sources": research_strategy["research_config"]["max_sources"],
                "news_window_hours": research_strategy["research_config"][
                    "news_window_hours"
                ],
                "use_asknews_48h": research_strategy["research_config"][
                    "use_asknews_48h"
                ],
                "token_budget": research_strategy["research_config"]["token_budget"],
            },
            "prompt_strategy": {
                "prompt_type": prompt_info["complexity_level"],
                "focus_type": prompt_info["focus_type"],
                "prompt": prompt_info["prompt"],
                "token_estimates": prompt_info["token_estimates"],
            },
            "cost_estimates": {
                "research_cost": research_strategy["research_config"]["estimated_cost"],
                "prompt_cost": prompt_info["cost_estimates"]["gpt-4o-mini"][
                    "total_cost"
                ],
                "total_estimated_cost": research_strategy["research_config"][
                    "estimated_cost"
                ]
                + prompt_info["cost_estimates"]["gpt-4o-mini"]["total_cost"],
            },
            "recommendations": {
                "model": prompt_info["recommended_model"],
                "budget_adjusted": research_strategy["budget_adjusted"],
            },
        }

        return research_plan

    def validate_and_improve_research(
        self,
        question: Question,
        research_sources: List[ResearchSource],
        original_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate research quality and suggest improvements.

        Args:
            question: The forecasting question
            research_sources: Research sources that were gathered
            original_plan: Original research plan

        Returns:
            Dictionary with quality assessment and improvement suggestions
        """
        # Validate research quality
        quality_assessment = self.research_manager.validate_research_quality(
            research_sources, original_plan["research_strategy"]
        )

        # Generate improvement plan if quality is low
        improvement_plan = None
        if quality_assessment["overall_quality"] in ["low", "medium"]:
            improvement_plan = self._generate_improvement_plan(
                question, quality_assessment, original_plan
            )

        return {
            "quality_assessment": quality_assessment,
            "needs_improvement": quality_assessment["overall_quality"] != "high",
            "improvement_plan": improvement_plan,
        }

    def _generate_improvement_plan(
        self,
        question: Question,
        quality_assessment: Dict[str, Any],
        original_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate plan to improve research quality."""
        gaps = quality_assessment["identified_gaps"]
        recommendations = quality_assessment["recommendations"]

        # Determine if we should upgrade research depth
        current_depth = original_plan["research_strategy"]["depth_level"]
        depth_hierarchy = ["shallow", "standard", "deep", "comprehensive"]
        current_index = depth_hierarchy.index(current_depth)

        suggested_actions = []

        if "insufficient_sources" in gaps and current_index < len(depth_hierarchy) - 1:
            suggested_actions.append(
                {
                    "action": "upgrade_research_depth",
                    "from": current_depth,
                    "to": depth_hierarchy[current_index + 1],
                    "reason": "Insufficient sources found at current depth level",
                }
            )

        if "outdated_information" in gaps:
            suggested_actions.append(
                {
                    "action": "enable_asknews_48h",
                    "reason": "Need more recent information",
                }
            )

        if "limited_perspective_diversity" in gaps:
            suggested_actions.append(
                {
                    "action": "expand_source_types",
                    "reason": "Need more diverse perspectives",
                }
            )

        return {
            "suggested_actions": suggested_actions,
            "recommendations": recommendations,
            "estimated_additional_cost": self._estimate_improvement_cost(
                suggested_actions
            ),
        }

    def _estimate_improvement_cost(
        self, suggested_actions: List[Dict[str, Any]]
    ) -> float:
        """Estimate additional cost for improvement actions."""
        additional_cost = 0.0

        for action in suggested_actions:
            if action["action"] == "upgrade_research_depth":
                # Cost difference between depth levels
                depth_costs = {
                    "shallow": 0.05,
                    "standard": 0.12,
                    "deep": 0.25,
                    "comprehensive": 0.45,
                }
                from_cost = depth_costs.get(action["from"], 0.05)
                to_cost = depth_costs.get(action["to"], 0.12)
                additional_cost += max(0, to_cost - from_cost)

            elif action["action"] == "enable_asknews_48h":
                additional_cost += 0.03  # Estimated cost for AskNews API calls

            elif action["action"] == "expand_source_types":
                additional_cost += 0.05  # Additional research cost

        return additional_cost
