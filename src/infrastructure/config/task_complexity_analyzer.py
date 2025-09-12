"""
Task complexity analyzer for intelligent model selection in tournament forecasting.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Complexity levels for task assessment."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class ComplexityAssessment:
    """Result of complexity assessment."""

    level: ComplexityLevel
    score: float
    factors: Dict[str, float]
    recommended_model: str
    reasoning: str


class TaskComplexityAnalyzer:
    """Analyzes task complexity for intelligent model selection."""

    def __init__(self):
        """Initialize the complexity analyzer."""
        self.complexity_indicators = self._setup_complexity_indicators()
        self.model_recommendations = self._setup_model_recommendations()

        logger.info("Task complexity analyzer initialized")

    def _setup_complexity_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Setup indicators for complexity assessment."""
        return {
            "simple_indicators": {
                "keywords": [
                    "yes/no",
                    "binary",
                    "specific date",
                    "specific number",
                    "announced",
                    "scheduled",
                    "confirmed",
                    "official",
                    "released",
                    "published",
                    "launched",
                    "completed",
                    "will be",
                    "has been",
                    "is scheduled",
                    "is planned",
                ],
                "patterns": [
                    r"\b\d{4}\b",  # Years
                    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",  # Months
                    r"\b(before|after|by)\s+\d{4}\b",  # Time constraints
                    r"\b(more|less|at least|at most)\s+\d+\b",  # Numeric comparisons
                ],
                "weight": 1.0,
            },
            "medium_indicators": {
                "keywords": [
                    "market",
                    "price",
                    "stock",
                    "election",
                    "vote",
                    "poll",
                    "technology",
                    "development",
                    "research",
                    "study",
                    "approval",
                    "regulation",
                    "law",
                    "policy",
                    "trend",
                    "growth",
                    "decline",
                    "change",
                ],
                "patterns": [
                    r"\b(will|might|could|may)\s+\w+\b",  # Uncertainty language
                    r"\b(increase|decrease|rise|fall)\s+by\b",  # Change indicators
                    r"\b(above|below|exceed|reach)\s+\d+\b",  # Threshold language
                ],
                "weight": 1.5,
            },
            "complex_indicators": {
                "keywords": [
                    "geopolitical",
                    "economic",
                    "financial",
                    "systemic",
                    "multiple factors",
                    "complex",
                    "uncertain",
                    "unprecedented",
                    "international",
                    "global",
                    "interdependent",
                    "cascading",
                    "macroeconomic",
                    "geopolitics",
                    "diplomatic",
                    "strategic",
                    "multifaceted",
                    "interconnected",
                    "compound",
                    "aggregate",
                ],
                "patterns": [
                    r"\b(depends on|contingent on|subject to)\b",  # Conditional language
                    r"\b(various|multiple|several|many)\s+factors\b",  # Multiple factors
                    r"\b(complex|complicated|intricate|sophisticated)\b",  # Complexity language
                    r"\b(if and only if|provided that|assuming)\b",  # Conditional logic
                ],
                "weight": 2.0,
            },
        }

    def _setup_model_recommendations(self) -> Dict[ComplexityLevel, Dict[str, str]]:
        """Setup model recommendations based on complexity and budget status."""
        return {
            ComplexityLevel.SIMPLE: {
                "normal": "openai/gpt-5-mini",
                "conservative": "openai/gpt-5-nano",
                "emergency": "openai/gpt-5-nano",
            },
            ComplexityLevel.MEDIUM: {
                "normal": "openai/gpt-5-mini",  # Research tasks
                "conservative": "openai/gpt-5-nano",
                "emergency": "openai/gpt-5-nano",
            },
            ComplexityLevel.COMPLEX: {
                "normal": "openai/gpt-5",  # Use premium model for complex tasks
                "conservative": "openai/gpt-5-mini",  # Downgrade in conservative mode
                "emergency": "openai/gpt-5-nano",  # Always use cheapest model in emergency
            },
        }

    def assess_question_complexity(
        self,
        question_text: str,
        background_info: str = "",
        resolution_criteria: str = "",
        fine_print: str = "",
    ) -> ComplexityAssessment:
        """Assess the complexity of a forecasting question."""
        # Combine all text for analysis
        combined_text = f"{question_text} {background_info} {resolution_criteria} {fine_print}".lower()

        # Calculate complexity scores
        scores = self._calculate_complexity_scores(combined_text)

        # Determine overall complexity level
        complexity_level = self._determine_complexity_level(scores)

        # Generate reasoning
        reasoning = self._generate_reasoning(scores, complexity_level, combined_text)

        return ComplexityAssessment(
            level=complexity_level,
            score=scores["total_score"],
            factors=scores,
            recommended_model=self.model_recommendations[complexity_level]["normal"],
            reasoning=reasoning,
        )

    def _calculate_complexity_scores(self, text: str) -> Dict[str, float]:
        """Calculate complexity scores based on various factors."""
        scores = {
            "simple_score": 0.0,
            "medium_score": 0.0,
            "complex_score": 0.0,
            "length_score": 0.0,
            "uncertainty_score": 0.0,
            "temporal_score": 0.0,
        }

        # Keyword-based scoring
        for category, indicators in self.complexity_indicators.items():
            category_score = 0.0

            # Check keywords
            for keyword in indicators["keywords"]:
                if keyword in text:
                    category_score += indicators["weight"]

            # Check patterns
            for pattern in indicators["patterns"]:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                category_score += matches * indicators["weight"]

            # Map to score categories
            if category == "simple_indicators":
                scores["simple_score"] = category_score
            elif category == "medium_indicators":
                scores["medium_score"] = category_score
            elif category == "complex_indicators":
                scores["complex_score"] = category_score

        # Length-based complexity
        text_length = len(text)
        if text_length > 3000:
            scores["length_score"] = 3.0
        elif text_length > 2000:
            scores["length_score"] = 2.0
        elif text_length > 1000:
            scores["length_score"] = 1.0
        else:
            scores["length_score"] = 0.0

        # Uncertainty language scoring
        uncertainty_patterns = [
            r"\b(uncertain|unclear|unknown|ambiguous|vague)\b",
            r"\b(might|could|may|possibly|potentially)\b",
            r"\b(depends|contingent|subject to|conditional)\b",
        ]
        uncertainty_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in uncertainty_patterns
        )
        scores["uncertainty_score"] = min(uncertainty_count * 0.5, 3.0)

        # Temporal complexity (multiple time references)
        temporal_patterns = [
            r"\b\d{4}\b",  # Years
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
            r"\b(before|after|by|until|during)\s+\d{4}\b",
            r"\b(next|this|last)\s+(year|month|quarter|decade)\b",
        ]
        temporal_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in temporal_patterns
        )
        scores["temporal_score"] = min(temporal_count * 0.3, 2.0)

        # Calculate total weighted score
        scores["total_score"] = (
            scores["simple_score"] * -0.5  # Simple factors reduce complexity
            + scores["medium_score"] * 1.0
            + scores["complex_score"] * 2.0
            + scores["length_score"] * 0.5
            + scores["uncertainty_score"] * 1.5
            + scores["temporal_score"] * 0.3
        )

        return scores

    def _determine_complexity_level(self, scores: Dict[str, float]) -> ComplexityLevel:
        """Determine complexity level based on scores."""
        total_score = scores["total_score"]
        complex_score = scores["complex_score"]
        simple_score = scores["simple_score"]

        # Strong indicators for complex
        if complex_score >= 4.0 or total_score >= 8.0:
            return ComplexityLevel.COMPLEX

        # Strong indicators for simple
        if simple_score >= 3.0 and complex_score == 0.0 and total_score <= 2.0:
            return ComplexityLevel.SIMPLE

        # Complex if high uncertainty and complexity indicators
        if scores["uncertainty_score"] >= 2.0 and complex_score >= 2.0:
            return ComplexityLevel.COMPLEX

        # Simple if clear temporal constraints and simple language
        if (
            scores["temporal_score"] >= 1.0
            and simple_score >= 2.0
            and complex_score <= 1.0
        ):
            return ComplexityLevel.SIMPLE

        # Default to medium for everything else
        return ComplexityLevel.MEDIUM

    def _generate_reasoning(
        self, scores: Dict[str, float], complexity_level: ComplexityLevel, text: str
    ) -> str:
        """Generate human-readable reasoning for the complexity assessment."""
        reasoning_parts = []

        # Overall assessment
        reasoning_parts.append(
            f"Assessed as {complexity_level.value.upper()} complexity"
        )

        # Key factors
        if scores["complex_score"] >= 2.0:
            reasoning_parts.append(
                f"High complexity indicators (score: {scores['complex_score']:.1f})"
            )

        if scores["simple_score"] >= 2.0:
            reasoning_parts.append(
                f"Simple task indicators (score: {scores['simple_score']:.1f})"
            )

        if scores["uncertainty_score"] >= 1.5:
            reasoning_parts.append(
                f"High uncertainty language (score: {scores['uncertainty_score']:.1f})"
            )

        if scores["length_score"] >= 2.0:
            reasoning_parts.append(
                f"Long text requiring detailed analysis (score: {scores['length_score']:.1f})"
            )

        # Text length context
        text_length = len(text)
        reasoning_parts.append(f"Text length: {text_length} characters")

        return "; ".join(reasoning_parts)

    def get_model_for_task(
        self,
        task_type: str,
        complexity_assessment: ComplexityAssessment,
        budget_status: str = "normal",
    ) -> str:
        """Get recommended model for a specific task based on complexity and budget."""
        # For research tasks, generally use cheaper models unless very complex
        if task_type == "research":
            if (
                complexity_assessment.level == ComplexityLevel.COMPLEX
                and budget_status == "normal"
            ):
                return "openai/gpt-5-mini"  # Use mini for research; upgrade if budget allows
            else:
                return "openai/gpt-5-nano"

        # For forecast tasks, use complexity-based selection
        elif task_type == "forecast":
            return self.model_recommendations[complexity_assessment.level][
                budget_status
            ]

        # For other tasks, use simple model
        else:
            return self.model_recommendations[ComplexityLevel.SIMPLE][budget_status]

    def estimate_cost_per_task(
        self,
        complexity_assessment: ComplexityAssessment,
        task_type: str,
        budget_status: str = "normal",
    ) -> Dict[str, Any]:
        """Estimate cost per task based on complexity assessment."""
        model = self.get_model_for_task(task_type, complexity_assessment, budget_status)

        # Base token estimates by complexity
        token_estimates = {
            ComplexityLevel.SIMPLE: {"input": 800, "output": 400},
            ComplexityLevel.MEDIUM: {"input": 1200, "output": 600},
            ComplexityLevel.COMPLEX: {"input": 1800, "output": 900},
        }

        base_tokens = token_estimates[complexity_assessment.level]

        # Adjust for task type
        if task_type == "research":
            input_tokens = int(base_tokens["input"] * 1.2)  # More context for research
            output_tokens = int(base_tokens["output"] * 1.5)  # Longer research outputs
        elif task_type == "forecast":
            input_tokens = int(base_tokens["input"] * 1.0)  # Standard context
            output_tokens = int(base_tokens["output"] * 0.8)  # Shorter forecast outputs
        else:
            input_tokens = base_tokens["input"]
            output_tokens = base_tokens["output"]

        # Cost calculation (simplified - would use actual BudgetManager in practice)
        cost_per_1k = {
            "openai/gpt-5": {"input": 0.0025, "output": 0.01},  # placeholder cost parity assumption
            "openai/gpt-5-mini": {"input": 0.0005, "output": 0.0015},  # adjust when official pricing known (verify)
            "openai/gpt-5-nano": {"input": 0.00015, "output": 0.0006},
        }

        model_costs = cost_per_1k.get(model, cost_per_1k["openai/gpt-5-nano"])
        estimated_cost = (input_tokens * model_costs["input"] / 1000) + (
            output_tokens * model_costs["output"] / 1000
        )

        return {
            "model": model,
            "complexity": complexity_assessment.level.value,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": estimated_cost,
            "task_type": task_type,
            "budget_status": budget_status,
        }

    def log_complexity_assessment(
        self, question_id: str, assessment: ComplexityAssessment
    ):
        """Log complexity assessment for debugging and monitoring."""
        logger.info(f"Question {question_id} complexity assessment:")
        logger.info(f"  Level: {assessment.level.value}")
        logger.info(f"  Score: {assessment.score:.2f}")
        logger.info(f"  Recommended Model: {assessment.recommended_model}")
        logger.info(f"  Reasoning: {assessment.reasoning}")

        # Log detailed scores for debugging
        logger.debug(f"  Detailed scores: {assessment.factors}")


# Global instance
task_complexity_analyzer = TaskComplexityAnalyzer()
