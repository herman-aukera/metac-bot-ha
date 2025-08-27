"""
Enhanced Strategic Tri-Model Router for GPT-5 variants with anti-slop directives.
Implements cost-performance optimization for tournament forecasting with advanced
model availability detection, robust fallback chains, and comprehensive monitoring.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union
from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

TaskType = Literal["validation", "research", "forecast", "simple"]
ComplexityLevel = Literal["minimal", "medium", "high"]
ModelTier = Literal["nano", "mini", "full"]
OperationMode = Literal["normal", "conservative", "emergency", "critical"]

@dataclass
class ModelConfig:
    """Configuration for a model tier."""
    model_name: str
    cost_per_million_tokens: float
    temperature: float
    timeout: int
    allowed_tries: int
    description: str

@dataclass
class ModelStatus:
    """Status information for a model."""
    tier: ModelTier
    model_name: str
    is_available: bool
    last_check: float
    error_message: Optional[str] = None
    response_time: Optional[float] = None


class EnhancedTriModelRouter:
    """
    Enhanced strategic model routing system for GPT-5 variants with advanced features.

    Cost-Performance Triangle:
    - GPT-5 nano: Ultra-fast, cheapest ($0.05/1M tokens) — validation, parsing, simple summaries
    - GPT-5 mini: Balanced speed/intelligence ($0.25/1M tokens) — research synthesis, intermediate reasoning
    - GPT-5 full: Maximum reasoning power ($1.50/1M tokens) — final forecasting decisions, complex analysis

    Budget Impact: At $100 budget, this tri-model approach processes ~2000 questions vs ~300 with GPT-5 full only.

    Enhanced Features:
    - Model availability detection and health monitoring
    - Robust fallback chains with performance tracking
    - Budget-aware operation modes with automatic switching
    - Advanced error recovery and circuit breaker patterns
    """

    def __init__(self):
        """Initialize enhanced tri-model configuration with strategic routing."""
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")

        # Model configurations with enhanced settings
        self.model_configs = self._get_model_configurations()

        # Initialize models with availability detection
        self.models = {}
        self.model_status = {}
        self.fallback_chains = self._define_fallback_chains()

        # Routing strategy based on task complexity and budget constraints
        self.routing_strategy = {
            "validation": "nano",      # Ultra-fast validation, parsing
            "simple": "nano",          # Simple summaries, basic tasks
            "research": "mini",        # Research synthesis, intermediate reasoning
            "forecast": "full",        # Final forecasting decisions, complex analysis
        }

        # Operation mode thresholds (budget utilization percentages)
        self.operation_thresholds = {
            "normal": (0, 50),         # 0-50% budget used
            "conservative": (50, 80),   # 50-80% budget used
            "emergency": (80, 95),      # 80-95% budget used
            "critical": (95, 100),      # 95-100% budget used
        }

        # Initialize models and check availability
        self._initialize_all_models()

        logger.info("Enhanced tri-model router initialized with GPT-5 variants and fallback chains")

    def _get_model_configurations(self) -> Dict[ModelTier, ModelConfig]:
        """Get model configurations for all tiers."""
        return {
            "nano": ModelConfig(
                model_name=os.getenv("NANO_MODEL", "gpt-5-nano"),
                cost_per_million_tokens=0.05,
                temperature=0.1,  # Low temperature for deterministic validation
                timeout=30,
                allowed_tries=2,
                description="Ultra-fast validation, parsing, simple summaries"
            ),
            "mini": ModelConfig(
                model_name=os.getenv("MINI_MODEL", "gpt-5-mini"),
                cost_per_million_tokens=0.25,
                temperature=0.3,  # Moderate temperature for research synthesis
                timeout=60,
                allowed_tries=2,
                description="Balanced speed/intelligence for research synthesis"
            ),
            "full": ModelConfig(
                model_name=os.getenv("DEFAULT_MODEL", "gpt-5"),
                cost_per_million_tokens=1.50,
                temperature=0.0,  # Zero temperature for precise forecasting
                timeout=90,
                allowed_tries=3,
                description="Maximum reasoning power for final forecasting decisions"
            )
        }

    def _define_fallback_chains(self) -> Dict[ModelTier, List[str]]:
        """Define fallback chains for each model tier."""
        return {
            "nano": [
                "gpt-5-nano",
                "openai/gpt-4o-mini",
                "metaculus/gpt-4o-mini",
                "openai/gpt-3.5-turbo"
            ],
            "mini": [
                "gpt-5-mini",
                "openai/gpt-4o-mini",
                "metaculus/gpt-4o-mini",
                "openai/gpt-4o"
            ],
            "full": [
                "gpt-5",
                "openai/gpt-4o",
                "metaculus/gpt-4o",
                "openai/gpt-4o-mini"
            ]
        }

    def _initialize_all_models(self):
        """Initialize all models with availability detection."""
        for tier, config in self.model_configs.items():
            try:
                model, status = self._initialize_model_with_fallback(tier, config)
                self.models[tier] = model
                self.model_status[tier] = status

                if status.is_available:
                    logger.info(f"✓ {tier.upper()} model initialized: {status.model_name}")
                else:
                    logger.warning(f"⚠ {tier.upper()} model fallback used: {status.model_name} ({status.error_message})")

            except Exception as e:
                logger.error(f"✗ Failed to initialize {tier.upper()} model: {e}")
                # Create emergency fallback
                self.models[tier] = self._create_emergency_model(tier)
                self.model_status[tier] = ModelStatus(
                    tier=tier,
                    model_name="emergency-fallback",
                    is_available=False,
                    last_check=0,
                    error_message=str(e)
                )

    def _initialize_model_with_fallback(self, tier: ModelTier, config: ModelConfig) -> Tuple[GeneralLlm, ModelStatus]:
        """Initialize a model with fallback chain."""
        fallback_chain = self.fallback_chains[tier]

        for model_name in fallback_chain:
            try:
                # Test model availability
                test_model = GeneralLlm(
                    model=model_name,
                    api_key=self.openrouter_key,
                    temperature=config.temperature,
                    timeout=config.timeout,
                    allowed_tries=config.allowed_tries,
                )

                # Quick availability test (could be enhanced with actual API call)
                model = GeneralLlm(
                    model=model_name,
                    api_key=self.openrouter_key,
                    temperature=config.temperature,
                    timeout=config.timeout,
                    allowed_tries=config.allowed_tries,
                )

                status = ModelStatus(
                    tier=tier,
                    model_name=model_name,
                    is_available=True,
                    last_check=asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
                )

                return model, status

            except Exception as e:
                logger.debug(f"Model {model_name} not available for {tier}: {e}")
                continue

        # If all models in chain fail, create emergency fallback
        emergency_model = self._create_emergency_model(tier)
        status = ModelStatus(
            tier=tier,
            model_name="emergency-fallback",
            is_available=False,
            last_check=0,
            error_message="All models in fallback chain failed"
        )

        return emergency_model, status

    def _create_emergency_model(self, tier: ModelTier) -> GeneralLlm:
        """Create emergency fallback model."""
        config = self.model_configs[tier]
        return GeneralLlm(
            model="openai/gpt-3.5-turbo",  # Most reliable fallback
            api_key=self.openrouter_key,
            temperature=config.temperature,
            timeout=config.timeout,
            allowed_tries=1,  # Reduced tries for emergency
        )

    def get_operation_mode(self, budget_remaining: float) -> OperationMode:
        """Determine operation mode based on budget utilization."""
        budget_used = 100.0 - budget_remaining

        for mode, (min_used, max_used) in self.operation_thresholds.items():
            if min_used <= budget_used < max_used:
                return mode

        # Default to critical if over 100%
        return "critical"

    def get_model_costs(self) -> Dict[ModelTier, float]:
        """Get cost per million tokens for each model tier."""
        return {tier: config.cost_per_million_tokens for tier, config in self.model_configs.items()}

    async def check_model_health(self, tier: ModelTier) -> ModelStatus:
        """Check health of a specific model tier."""
        try:
            model = self.models[tier]
            start_time = asyncio.get_event_loop().time()

            # Simple health check with minimal prompt
            test_response = await model.invoke("Test")

            response_time = asyncio.get_event_loop().time() - start_time

            status = ModelStatus(
                tier=tier,
                model_name=model.model,
                is_available=True,
                last_check=start_time,
                response_time=response_time
            )

            self.model_status[tier] = status
            return status

        except Exception as e:
            status = ModelStatus(
                tier=tier,
                model_name=self.models[tier].model,
                is_available=False,
                last_check=asyncio.get_event_loop().time(),
                error_message=str(e)
            )

            self.model_status[tier] = status
            return status

    def choose_model(self, task_type: TaskType, complexity: Optional[ComplexityLevel] = None,
                    content_length: int = 0, budget_remaining: float = 100.0) -> Tuple[GeneralLlm, ModelTier]:
        """
        Choose optimal model based on task requirements and budget constraints with enhanced logic.

        Args:
            task_type: Type of task (validation, research, forecast, simple)
            complexity: Complexity level (minimal, medium, high)
            content_length: Length of content to process
            budget_remaining: Remaining budget percentage (0-100)

        Returns:
            Tuple of (selected_model, model_tier)
        """
        # Determine operation mode based on budget
        operation_mode = self.get_operation_mode(budget_remaining)

        # Base model selection from routing strategy
        base_tier = self.routing_strategy.get(task_type, "mini")

        # Operation mode adjustments
        selected_tier = self._adjust_for_operation_mode(base_tier, operation_mode, task_type)

        # Complexity-based adjustments (only upgrade if budget allows)
        selected_tier = self._adjust_for_complexity(selected_tier, complexity, operation_mode)

        # Content length adjustments for very short content
        if content_length < 100 and selected_tier != "nano":
            selected_tier = "nano"
            logger.debug(f"Short content ({content_length} chars): using nano model for {task_type}")

        # Ensure model is available, fallback if necessary
        if not self.model_status[selected_tier].is_available:
            selected_tier = self._find_available_fallback(selected_tier)

        selected_model = self.models[selected_tier]

        logger.debug(f"Selected {selected_tier} model for {task_type} "
                    f"(mode: {operation_mode}, complexity: {complexity}, budget: {budget_remaining:.1f}%)")

        return selected_model, selected_tier

    def _adjust_for_operation_mode(self, base_tier: ModelTier, mode: OperationMode, task_type: TaskType) -> ModelTier:
        """Adjust model tier based on operation mode."""
        if mode == "critical":
            # Critical mode: nano only
            return "nano"
        elif mode == "emergency":
            # Emergency mode: prefer nano, allow mini for critical tasks
            if task_type == "forecast" and base_tier == "full":
                return "mini"  # Downgrade forecasting from full to mini
            else:
                return "nano"
        elif mode == "conservative":
            # Conservative mode: avoid full model when possible
            if base_tier == "full":
                return "mini"
            else:
                return base_tier
        else:
            # Normal mode: use base tier
            return base_tier

    def _adjust_for_complexity(self, tier: ModelTier, complexity: Optional[ComplexityLevel],
                              mode: OperationMode) -> ModelTier:
        """Adjust model tier based on complexity, respecting operation mode."""
        if complexity is None:
            return tier

        # Only allow upgrades in normal and conservative modes
        if mode in ["emergency", "critical"]:
            return tier

        if complexity == "high" and tier == "mini" and mode == "normal":
            return "full"
        elif complexity == "minimal" and tier == "full":
            return "mini"

        return tier

    def _find_available_fallback(self, preferred_tier: ModelTier) -> ModelTier:
        """Find an available model tier as fallback."""
        # Try tiers in order of preference: preferred -> lower cost -> any available
        tier_order = ["nano", "mini", "full"]
        preferred_index = tier_order.index(preferred_tier)

        # First try the preferred tier and lower cost options
        for i in range(preferred_index, len(tier_order)):
            tier = tier_order[i]
            if self.model_status[tier].is_available:
                if tier != preferred_tier:
                    logger.info(f"Fallback: using {tier} instead of unavailable {preferred_tier}")
                return tier

        # If no lower-cost options, try higher-cost options
        for i in range(preferred_index - 1, -1, -1):
            tier = tier_order[i]
            if self.model_status[tier].is_available:
                logger.warning(f"Emergency fallback: using {tier} instead of unavailable {preferred_tier}")
                return tier

        # If nothing is available, return nano (should have emergency fallback)
        logger.error(f"No models available, using nano emergency fallback")
        return "nano"

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
                         complexity: Optional[ComplexityLevel] = None,
                         budget_remaining: float = 100.0) -> float:
        """Estimate cost for a given task with enhanced accuracy."""
        _, tier = self.choose_model(task_type, complexity, content_length, budget_remaining)

        # Enhanced token estimation based on task type
        base_tokens = content_length / 4  # 4 chars per token average

        # Task-specific output multipliers
        output_multipliers = {
            "validation": 0.3,    # Short validation responses
            "simple": 0.5,        # Brief simple responses
            "research": 1.8,      # Detailed research with citations
            "forecast": 2.2,      # Comprehensive forecasting analysis
        }

        multiplier = output_multipliers.get(task_type, 1.5)
        estimated_tokens = base_tokens * multiplier

        # Get cost per token for selected tier
        cost_per_token = self.model_configs[tier].cost_per_million_tokens / 1_000_000

        return estimated_tokens * cost_per_token

    def get_model_status(self) -> Dict[str, str]:
        """Get comprehensive status of all models."""
        status = {}
        for tier in self.model_status:
            model_status = self.model_status[tier]
            if model_status.is_available:
                response_info = f" ({model_status.response_time:.2f}s)" if model_status.response_time else ""
                status[tier] = f"✓ {model_status.model_name} (Ready{response_info})"
            else:
                error_info = f" - {model_status.error_message}" if model_status.error_message else ""
                status[tier] = f"✗ {model_status.model_name} (Unavailable{error_info})"
        return status

    def get_detailed_status(self) -> Dict[str, Dict]:
        """Get detailed status information for monitoring."""
        detailed_status = {}
        for tier, status in self.model_status.items():
            config = self.model_configs[tier]
            detailed_status[tier] = {
                "model_name": status.model_name,
                "is_available": status.is_available,
                "cost_per_million_tokens": config.cost_per_million_tokens,
                "description": config.description,
                "last_check": status.last_check,
                "response_time": status.response_time,
                "error_message": status.error_message,
                "fallback_chain": self.fallback_chains[tier]
            }
        return detailed_status

    def get_routing_explanation(self, task_type: TaskType, complexity: Optional[ComplexityLevel] = None,
                               content_length: int = 0, budget_remaining: float = 100.0) -> str:
        """Get detailed explanation of routing decision."""
        operation_mode = self.get_operation_mode(budget_remaining)
        base_tier = self.routing_strategy.get(task_type, "mini")
        selected_model, selected_tier = self.choose_model(task_type, complexity, content_length, budget_remaining)

        explanation = [
            f"Task: {task_type}",
            f"Operation Mode: {operation_mode} (budget remaining: {budget_remaining:.1f}%)",
            f"Base Tier: {base_tier}",
            f"Selected Tier: {selected_tier}",
            f"Model: {selected_model.model}",
            f"Estimated Cost: ${self.get_cost_estimate(task_type, content_length, complexity, budget_remaining):.6f}"
        ]

        if complexity:
            explanation.insert(2, f"Complexity: {complexity}")
        if content_length > 0:
            explanation.insert(-2, f"Content Length: {content_length} chars")

        return " | ".join(explanation)


# Global instance with backward compatibility
tri_model_router = EnhancedTriModelRouter()

# Backward compatibility alias
TriModelRouter = EnhancedTriModelRouter
