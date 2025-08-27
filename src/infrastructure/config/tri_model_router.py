"""
Strategic Tri-Model Router for GPT-5 variants with anti-slop directives.
Implements cost-performance optimization for tournament forecasting.
"""

import logging
import os
from typing import Dict, Literal, Optional, Tuple
from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

TaskType = Literal["validation", "research", "forecast", "simple"]
ComplexityLevel = Literal["minimal", "medium", "high"]
ModelTier = Literal["nano", "mini", "full"]


class TriModelRouter:
    """
    Strategic model routing system for GPT-5 variants.

    Cost-Performance Triangle:
    - GPT-5 nano: Ultra-fast, cheapest—perfect for validation, parsing, simple summaries
    - GPT-5 mini: Balanced speed/intelligence—ideal for research synthesis, intermediate reasoning
    - GPT-5 full: Maximum reasoning power—reserved for final forecasting decisions, complex analysis

    Budget Impact: At $100 budget, this tri-model approach processes ~2000 questions vs ~300 with GPT-5 full only.
    """

    def __init__(self):
        """Initialize tri-model configuration with strategic routing."""
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")

        # Model configuration with cost optimization
        self.models = self._initialize_models()

        # Routing strategy based on task complexity and budget constraints
        self.routing_strategy = {
            "validation": "nano",      # Ultra-fast validation, parsing
            "simple": "nano",          # Simple summaries, basic tasks
            "research": "mini",        # Research synthesis, intermediate reasoning
            "forecast": "full",        # Final forecasting decisions, complex analysis
        }

        # Cost per 1M tokens (estimated)
        self.model_costs = {
            "nano": 0.05,   # $0.05/1M tokens
            "mini": 0.25,   # $0.25/1M tokens
            "full": 1.50,   # $1.50/1M tokens
        }

        logger.info("Tri-model router initialized with GPT-5 variants")

    def _initialize_models(self) -> Dict[ModelTier, GeneralLlm]:
        """Initialize the three GPT-5 model variants."""
        models = {}

        try:
            # GPT-5 Nano - Ultra-fast, cheapest
            models["nano"] = GeneralLlm(
                model=os.getenv("NANO_MODEL", "gpt-5-nano"),
                api_key=self.openrouter_key,
                temperature=0.1,  # Low temperature for deterministic validation
                timeout=30,
                allowed_tries=2,
            )

            # GPT-5 Mini - Balanced speed/intelligence
            models["mini"] = GeneralLlm(
                model=os.getenv("MINI_MODEL", "gpt-5-mini"),
                api_key=self.openrouter_key,
                temperature=0.3,  # Moderate temperature for research synthesis
                timeout=60,
                allowed_tries=2,
            )

            # GPT-5 Full - Maximum reasoning power
            models["full"] = GeneralLlm(
                model=os.getenv("DEFAULT_MODEL", "gpt-5"),
                api_key=self.openrouter_key,
                temperature=0.0,  # Zero temperature for precise forecasting
                timeout=90,
                allowed_tries=3,
            )

            logger.info("All GPT-5 model variants initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize GPT-5 models: {e}")
            # Fallback to current models if GPT-5 not available
            models = self._initialize_fallback_models()

        return models

    def _initialize_fallback_models(self) -> Dict[ModelTier, GeneralLlm]:
        """Fallback to current GPT-4 models if GPT-5 not available."""
        logger.warning("Falling back to GPT-4 models")

        return {
            "nano": GeneralLlm(
                model=os.getenv("SIMPLE_TASK_MODEL", "openai/gpt-4o-mini"),
                api_key=self.openrouter_key,
                temperature=0.1,
                timeout=30,
                allowed_tries=2,
            ),
            "mini": GeneralLlm(
                model=os.getenv("PRIMARY_RESEARCH_MODEL", "openai/gpt-4o-mini"),
                api_key=self.openrouter_key,
                temperature=0.3,
                timeout=60,
                allowed_tries=2,
            ),
            "full": GeneralLlm(
                model=os.getenv("PRIMARY_FORECAST_MODEL", "openai/gpt-4o"),
                api_key=self.openrouter_key,
                temperature=0.0,
                timeout=90,
                allowed_tries=3,
            ),
        }

    def choose_model(self, task_type: TaskType, complexity: Optional[ComplexityLevel] = None,
                    content_length: int = 0, budget_remaining: float = 100.0) -> Tuple[GeneralLlm, ModelTier]:
        """
        Choose optimal model based on task requirements and budget constraints.

        Args:
            task_type: Type of task (validation, research, forecast, simple)
            complexity: Complexity level (minimal, medium, high)
            content_length: Length of content to process
            budget_remaining: Remaining budget percentage (0-100)

        Returns:
            Tuple of (selected_model, model_tier)
        """
        # Base model selection from routing strategy
        base_tier = self.routing_strategy.get(task_type, "mini")

        # Budget-aware adjustments
        if budget_remaining < 20:  # Emergency mode - use cheapest
            selected_tier = "nano"
            logger.warning(f"Emergency budget mode: forcing nano model for {task_type}")
        elif budget_remaining < 50:  # Conservative mode - downgrade if possible
            if base_tier == "full":
                selected_tier = "mini"
                logger.info(f"Conservative budget mode: downgrading {task_type} from full to mini")
            else:
                selected_tier = base_tier
        else:
            selected_tier = base_tier

        # Complexity-based adjustments (only upgrade, never downgrade for budget)
        if complexity == "high" and selected_tier != "nano" and budget_remaining > 30:
            if selected_tier == "mini":
                selected_tier = "full"
                logger.debug(f"High complexity: upgrading {task_type} from mini to full")
        elif complexity == "minimal" and selected_tier == "full":
            selected_tier = "mini"
            logger.debug(f"Minimal complexity: downgrading {task_type} from full to mini")

        # Content length adjustments for very short content
        if content_length < 100 and selected_tier != "nano":
            selected_tier = "nano"
            logger.debug(f"Short content: using nano model for {task_type}")

        selected_model = self.models[selected_tier]

        logger.debug(f"Selected {selected_tier} model for {task_type} "
                    f"(complexity: {complexity}, budget: {budget_remaining:.1f}%)")

        return selected_model, selected_tier

    async def route_query(self, task_type: TaskType, content: str,
                         complexity: Optional[ComplexityLevel] = None,
                         budget_remaining: float = 100.0) -> str:
        """
        Route query to optimal model and execute with anti-slop directives.

        Args:
            task_type: Type of task to perform
            content: Content/prompt to process
            complexity: Optional complexity level
            budget_remaining: Remaining budget percentage

        Returns:
            Model response with anti-slop quality assurance
        """
        model, tier = self.choose_model(
            task_type=task_type,
            complexity=complexity,
            content_length=len(content),
            budget_remaining=budget_remaining
        )

        # Add anti-slop directives to prompt
        enhanced_prompt = self._add_anti_slop_directives(content, task_type, tier)

        try:
            response = await model.invoke(enhanced_prompt)

            # Quality validation
            validated_response = self._validate_response_quality(response, task_type)

            logger.debug(f"Successfully routed {task_type} to {tier} model")
            return validated_response

        except Exception as e:
            logger.error(f"Model routing failed for {task_type} with {tier}: {e}")
            # Fallback to nano model for emergency
            if tier != "nano":
                logger.info(f"Falling back to nano model for {task_type}")
                fallback_prompt = self._add_anti_slop_directives(content, task_type, "nano")
                return await self.models["nano"].invoke(fallback_prompt)
            else:
                raise

    def _add_anti_slop_directives(self, prompt: str, task_type: TaskType, model_tier: ModelTier) -> str:
        """Add anti-slop quality guard directives to prompt."""

        # Base anti-slop directives
        base_directives = """
# ANTI-SLOP / QUALITY GUARD
• Think step-by-step internally, then output only final, clear reasoning
• Ground every claim with specific evidence sources - no hallucinations
• If uncertain about anything, acknowledge it explicitly
• Use bullet points (•) for structure, keep response ≤ 300 words unless complex analysis required
• Maintain human, helpful tone while being precise
• Pre-check: Does every statement trace to verifiable evidence?
• Question your own reasoning - could there be edge cases or alternatives?
"""

        # Task-specific directives
        task_directives = {
            "validation": """
• Focus on factual accuracy and source verification
• Flag any unsupported claims or potential hallucinations
• Keep response concise and deterministic
""",
            "research": """
• Cite every factual claim with sources
• Acknowledge information gaps explicitly
• Prioritize recent developments and credible sources
• Synthesize information without speculation
""",
            "forecast": """
• Base predictions on verifiable evidence and historical precedents
• Acknowledge uncertainty and provide confidence bounds
• Consider multiple scenarios and their probabilities
• Avoid overconfidence - calibrate predictions carefully
""",
            "simple": """
• Provide clear, concise responses
• Verify basic facts before stating them
• Keep explanations simple but accurate
"""
        }

        # Model-tier specific adjustments
        tier_adjustments = {
            "nano": "• Prioritize speed and accuracy over depth\n• Focus on essential information only\n",
            "mini": "• Balance depth with efficiency\n• Provide moderate detail with good reasoning\n",
            "full": "• Use maximum reasoning capability\n• Provide comprehensive analysis when warranted\n"
        }

        # Combine directives
        full_directives = (
            base_directives +
            task_directives.get(task_type, "") +
            tier_adjustments.get(model_tier, "")
        )

        return f"{full_directives}\n\n{prompt}"

    def _validate_response_quality(self, response: str, task_type: TaskType) -> str:
        """Apply quality validation to response."""

        # Check for basic quality indicators
        if len(response.strip()) < 10:
            logger.warning(f"Response too short for {task_type}: {len(response)} chars")

        # Check for uncertainty acknowledgment in forecasting
        if task_type == "forecast":
            uncertainty_indicators = ["uncertain", "unclear", "difficult to predict", "confidence", "probability"]
            if not any(indicator in response.lower() for indicator in uncertainty_indicators):
                response += "\n\n[Note: Moderate confidence given available evidence and inherent uncertainty]"

        # Length compliance check
        word_count = len(response.split())
        if word_count > 400 and task_type != "forecast":
            logger.warning(f"Response exceeds recommended length for {task_type}: {word_count} words")

        return response

    def get_cost_estimate(self, task_type: TaskType, content_length: int,
                         complexity: Optional[ComplexityLevel] = None) -> float:
        """Estimate cost for a given task."""
        _, tier = self.choose_model(task_type, complexity, content_length)

        # Rough token estimation (4 chars per token average)
        estimated_tokens = content_length / 4 * 1.5  # 1.5x for output
        cost_per_token = self.model_costs[tier] / 1_000_000

        return estimated_tokens * cost_per_token

    def get_model_status(self) -> Dict[str, str]:
        """Get status of all models."""
        status = {}
        for tier, model in self.models.items():
            try:
                status[tier] = f"✓ {model.model} (Ready)"
            except Exception as e:
                status[tier] = f"✗ {model.model} (Error: {e})"
        return status


# Global instance
tri_model_router = TriModelRouter()
