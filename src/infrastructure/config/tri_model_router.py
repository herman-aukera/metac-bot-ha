"""
OpenRouter Tri-Model Router with Anti-Slop Directives.
Implements strategic cost-performance optimization through OpenRouter's unified gateway
with actual available models and correct pricing for tournament forecasting.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

from forecasting_tools import GeneralLlm
from ..external_apis.llm_client import LLMClient
from ..external_apis.llm_client_adapter import HardenedOpenRouterModel
from .settings import LLMConfig

logger = logging.getLogger(__name__)

TaskType = Literal["validation", "research", "forecast", "simple"]
ComplexityLevel = Literal["minimal", "medium", "high"]
ModelTier = Literal["nano", "mini", "full"]
OperationMode = Literal["normal", "conservative", "emergency", "critical"]
TaskPriority = Literal["low", "normal", "high", "critical"]


@dataclass
class ModelConfig:
    """Configuration for a model tier with OpenRouter pricing."""

    model_name: str
    cost_per_million_input: float
    cost_per_million_output: float
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


@dataclass
class ContentAnalysis:
    """Analysis of content for optimal model selection."""

    length: int
    complexity_score: float
    domain: str
    urgency: float
    estimated_tokens: int
    word_count: int
    complexity_indicators: List[str]


@dataclass
class BudgetContext:
    """Budget context for routing decisions."""

    remaining_percentage: float
    estimated_questions_remaining: int
    current_burn_rate: float
    operation_mode: OperationMode
    budget_used_percentage: float


@dataclass
class OpenRouterModelSelection:
    """Result of OpenRouter model selection process."""

    selected_model: str
    selected_tier: ModelTier
    rationale: str
    estimated_cost: float
    confidence_score: float
    fallback_models: List[str]
    operation_mode: OperationMode
    provider_preferences: Optional[Dict[str, Any]] = None


@dataclass
class RoutingResult:
    """Complete result of query routing and execution."""

    response: str
    model_used: ModelTier
    actual_model_name: str
    actual_cost: float
    performance_metrics: Dict[str, Any]
    quality_score: float
    execution_time: float
    fallback_used: bool
    routing_rationale: str


class OpenRouterTriModelRouter:
    """
    OpenRouter strategic model routing system with GPT-5 cost optimization.

    Cost-Performance Triangle via OpenRouter:
    - Tier 3 (nano): openai/gpt-5-nano ($0.05/1M tokens) — ultra-fast validation, parsing
    - Tier 2 (mini): openai/gpt-5-mini ($0.25/1M tokens) — research synthesis, intermediate reasoning
    - Tier 1 (full): openai/gpt-5 ($1.50/1M tokens) — final forecasting, complex analysis
    - Free Fallbacks: openai/gpt-oss-20b:free, moonshotai/kimi-k2:free — budget exhaustion operation

    Budget Impact: At $100 budget, processes 5000+ questions vs ~300 with single GPT-4o.

    OpenRouter Features:
    - Unified API gateway with provider routing
    - Attribution headers for ranking optimization
    - Automatic provider fallbacks and load balancing
    - Price-based model selection with :floor shortcuts
    """

    def __init__(self) -> None:
        """Initialize OpenRouter tri-model configuration with strategic routing."""
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        # Allow overriding base URL via env; default to official OpenRouter API
        self.openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.openrouter_headers = self._get_attribution_headers()
        # OpenRouter model configurations with actual pricing
        self.model_configs = self._get_openrouter_model_configurations()
        # Initialize containers for models & status (after configs defined)
        self.models: Dict[ModelTier, GeneralLlm] = {}
        self.model_status: Dict[ModelTier, ModelStatus] = {}
        self.fallback_chains: Dict[ModelTier, List[str]] = self._define_openrouter_fallback_chains()

        # Routing strategy based on task complexity and budget constraints
        self.routing_strategy = {
            "validation": "nano",
            "simple": "nano",
            "research": "mini",
            "forecast": "full",
        }

        # Operation mode thresholds (budget utilization percentages) - optimized for free fallbacks
        self.operation_thresholds = {
            "normal": (0, 70),  # 0-70% budget used: Use GPT-5 models
            "conservative": (70, 85),  # 70-85% budget used: Use GPT-5 mini/nano only
            "emergency": (85, 95),  # 85-95% budget used: Use free models
            "critical": (95, 100),  # 95-100% budget used: Free models only
        }

        # Initialize models and check availability
        self._initialize_all_models()

        logger.info(
            "OpenRouter tri-model router initialized with actual available models and pricing"
        )
        # Expose underlying hardened client for metrics (optional use by main run summary)
        try:
            self.llm_client = LLMClient(
                LLMConfig(
                    provider="openrouter",
                    model="openai/gpt-5-mini",
                    temperature=0.1,
                    max_tokens=None,
                    api_key=self.openrouter_key or "",
                    openrouter_api_key=self.openrouter_key or "",
                    timeout=60.0,
                )
            )
        except Exception:
            self.llm_client = None  # type: ignore[attr-defined]

    @classmethod
    async def create_with_auto_configuration(cls) -> "OpenRouterTriModelRouter":
        """Create router instance with automatic configuration and health monitoring."""
        logger.info("Creating OpenRouter tri-model router with auto-configuration...")

        # Create instance
        router = cls()

        # Perform startup health monitoring and auto-configuration
        startup_success = await router.health_monitor_startup()

        if not startup_success:
            logger.warning(
                "OpenRouter startup health check failed - router may have limited functionality"
            )
        else:
            logger.info("OpenRouter tri-model router successfully configured and ready")

        return router

    def _get_attribution_headers(self) -> Dict[str, str]:
        """Get OpenRouter attribution headers for ranking optimization."""
        headers = {}
        if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = referer
        if title := os.getenv("OPENROUTER_APP_TITLE"):
            headers["X-Title"] = title
        return headers

    def _get_openrouter_model_configurations(self) -> Dict[ModelTier, ModelConfig]:
        """Cost-optimized model configurations with free fallbacks."""
        return {
            "nano": ModelConfig(
                model_name=self._normalize_model_id(
                    os.getenv("NANO_MODEL", "openai/gpt-5-nano")
                ),
                cost_per_million_input=0.05,  # GPT-5 Nano
                cost_per_million_output=0.05,
                temperature=0.1,
                timeout=30,
                allowed_tries=3,  # More tries since we have free fallbacks
                description="GPT-5 Nano with free OSS/Kimi fallbacks",
            ),
            "mini": ModelConfig(
                model_name=self._normalize_model_id(
                    os.getenv("MINI_MODEL", "openai/gpt-5-mini")
                ),
                cost_per_million_input=0.25,  # GPT-5 Mini
                cost_per_million_output=0.25,
                temperature=0.3,
                timeout=60,
                allowed_tries=3,
                description="GPT-5 Mini with free fallbacks",
            ),
            "full": ModelConfig(
                model_name=self._normalize_model_id(
                    os.getenv("DEFAULT_MODEL", "openai/gpt-5")
                ),
                cost_per_million_input=1.50,  # GPT-5 Full
                cost_per_million_output=1.50,
                temperature=0.0,
                timeout=90,
                allowed_tries=3,
                description="GPT-5 Full with cost-optimized fallbacks",
            ),
        }

    def _define_openrouter_fallback_chains(self) -> Dict[ModelTier, List[str]]:
        """Cost-optimized fallback chains: GPT-5 → Free models (skip expensive GPT-4o)."""
        # Check which API keys are available
        has_openrouter = self.openrouter_key and not self.openrouter_key.startswith(
            "dummy_"
        )
        has_metaculus_proxy = (
            os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true"
        )

        # Free models for budget-conscious operation
        # Prefer OSS-20B over Kimi due to availability issues observed via OpenRouter (404 NotFound)
        free_models = [
            "openai/gpt-oss-20b:free",  # Free OSS - reliable fallback
            "moonshotai/kimi-k2:free",  # Free Kimi - alternate
        ]

        if has_openrouter and has_metaculus_proxy:
            # Metaculus proxy present but GPT-4o family purged; keep free models only
            return {
                "nano": [
                    "openai/gpt-5-nano",
                    "openai/gpt-oss-20b:free",
                    "moonshotai/kimi-k2:free",
                ],
                "mini": [
                    "openai/gpt-5-mini",
                    "openai/gpt-5-nano",
                    "openai/gpt-oss-20b:free",
                    "moonshotai/kimi-k2:free",
                ],
                "full": [
                    "openai/gpt-5",
                    "openai/gpt-5-mini",
                    "openai/gpt-oss-20b:free",
                    "openai/gpt-5-nano",
                    "moonshotai/kimi-k2:free",
                ],
            }
        elif has_openrouter:
            # OpenRouter only configuration with cost optimization
            logger.info(
                "Using cost-optimized OpenRouter fallback chains: GPT-5 → Free models"
            )
            return {
                "nano": [
                    "openai/gpt-5-nano",  # Primary: GPT-5 Nano ($0.05)
                    "openai/gpt-oss-20b:free",  # Free fallback (skip expensive models)
                    "moonshotai/kimi-k2:free",  # Free alternative
                ],
                "mini": [
                    "openai/gpt-5-mini",  # Primary: GPT-5 Mini ($0.25)
                    "openai/gpt-5-nano",  # Downgrade to nano first
                    "openai/gpt-oss-20b:free",  # Free research model (prefer over Kimi)
                    "moonshotai/kimi-k2:free",  # Free alternative
                ],
                "full": [
                    "openai/gpt-5",  # Primary: GPT-5 Full ($1.50)
                    "openai/gpt-5-mini",  # Downgrade to mini first
                    "openai/gpt-oss-20b:free",  # Free research model (prefer over Kimi)
                    "openai/gpt-5-nano",  # Further downgrade
                    "moonshotai/kimi-k2:free",  # Final free fallback
                ],
            }
        elif has_metaculus_proxy:
            # Metaculus proxy present; GPT-4o banned → rely on free chains
            logger.info("Metaculus proxy detected; using free model chains (GPT-4o removed)")
            return {
                "nano": free_models,
                "mini": free_models,
                "full": free_models,
            }
        else:
            # Emergency configuration - use free models only
            logger.warning("No primary API keys available - using free models only")
            return {"nano": free_models, "mini": free_models, "full": free_models}

    def _initialize_all_models(self) -> None:
        """Initialize all models with availability detection."""
        for tier, config in self.model_configs.items():
            try:
                model, status = self._initialize_model_with_fallback(tier, config)
                self.models[tier] = model
                self.model_status[tier] = status

                if status.is_available:
                    logger.info(
                        f"✓ {tier.upper()} model initialized: {status.model_name}"
                    )
                else:
                    logger.warning(
                        f"⚠ {tier.upper()} model fallback used: {status.model_name} ({status.error_message})"
                    )

            except Exception as e:
                logger.error(f"✗ Failed to initialize {tier.upper()} model: {e}")
                # Create emergency fallback
                self.models[tier] = self._create_emergency_model(tier)
                self.model_status[tier] = ModelStatus(
                    tier=tier,
                    model_name="emergency-fallback",
                    is_available=False,
                    last_check=0,
                    error_message=str(e),
                )

    def _initialize_model_with_fallback(
        self, tier: ModelTier, config: ModelConfig
    ) -> Tuple[GeneralLlm, ModelStatus]:
        """Initialize a model with OpenRouter fallback chain."""
        fallback_chain = self.fallback_chains[tier]
        for model_name in fallback_chain:
            try:
                model = self._create_openrouter_model(model_name, config, "normal")
                if model is None:
                    logger.debug(f"Could not create model for {model_name}, skipping")
                    continue

                status = ModelStatus(
                    tier=tier,
                    model_name=model_name,
                    is_available=True,
                    last_check=(
                        asyncio.get_event_loop().time()
                        if asyncio.get_event_loop().is_running()
                        else 0
                    ),
                    error_message=None,
                )
                logger.debug(
                    f"Successfully initialized {model_name} for {tier} tier via OpenRouter"
                )
                return model, status
            except Exception as e:
                logger.debug(f"Model {model_name} not available for {tier}: {e}")
                continue

        emergency_model = self._create_emergency_model(tier)
        status = ModelStatus(
            tier=tier,
            model_name="emergency-fallback",
            is_available=False,
            last_check=0,
            error_message="All models in fallback chain failed",
        )
        return emergency_model, status

    def _create_openrouter_model(
        self,
        model_name: str,
        config: ModelConfig,
        operation_mode: Optional[OperationMode] = None,
    ) -> Optional[GeneralLlm]:
        """Create a model configured for OpenRouter with proper headers and provider routing."""
        # Defensive normalization to ensure provider prefix before any routing logic
        model_name = self._normalize_model_id(model_name)

        # Determine API key and base URL based on model
        if model_name.startswith("metaculus/"):
            # Metaculus proxy models don't use OpenRouter
            if not os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true":
                return None
            return GeneralLlm(
                model=model_name,
                api_key=None,  # Proxy doesn't need API key
                temperature=config.temperature,
                timeout=config.timeout,
                allowed_tries=config.allowed_tries,
            )

        # OpenRouter models
        if not self.openrouter_key or self.openrouter_key.startswith("dummy_"):
            logger.debug(f"OpenRouter API key not available for {model_name}")
            return None

        # Create model with OpenRouter configuration and provider preferences
        extra_headers = self.openrouter_headers.copy()

        # Apply model shortcuts for optimization
        optimized_model_name = self._apply_model_shortcuts(
            model_name, operation_mode or "normal"
        )

        # Defensive: ensure provider prefix remains after shortcut application
        if (
            optimized_model_name
            and "/" not in optimized_model_name
            and not optimized_model_name.startswith("metaculus/")
        ):
            optimized_model_name = self._normalize_model_id(optimized_model_name)

        # Prefer hardened OpenRouter client when available to unify backoff/diagnostics
        try:
            # Lazily create client if not present (e.g., during tests)
            if not hasattr(self, "llm_client") or self.llm_client is None:
                self.llm_client = LLMClient(
                    LLMConfig(
                        provider="openrouter",
                        model=optimized_model_name,
                        temperature=config.temperature,
                        max_tokens=None,
                        api_key=self.openrouter_key or "",
                        openrouter_api_key=self.openrouter_key or "",
                        timeout=float(config.timeout),
                    )
                )
            return HardenedOpenRouterModel(
                llm_client=self.llm_client,
                model=optimized_model_name,
                temperature=config.temperature,
                timeout=config.timeout,
                allowed_tries=config.allowed_tries,
            )
        except Exception:
            # Fallback to GeneralLlm if hardened client unavailable for any reason
            return GeneralLlm(
                model=optimized_model_name,
                api_key=self.openrouter_key,
                base_url=self.openrouter_base_url,
                extra_headers=extra_headers,
                temperature=config.temperature,
                timeout=config.timeout,
                allowed_tries=config.allowed_tries,
                custom_llm_provider="openrouter",
            )

    def _get_provider_preferences_for_operation_mode(
        self, operation_mode: OperationMode
    ) -> Dict[str, Any]:
        """Get OpenRouter provider preferences based on operation mode with enhanced routing."""
        base_preferences = {
            "allow_fallbacks": True,
            "data_collection": "deny",  # Always prefer privacy-respecting providers
        }

        if operation_mode == "critical":
            # Critical mode: only free models, price-sorted with strict limits
            return {
                **base_preferences,
                "sort": "price",
                "max_price": {"prompt": 0, "completion": 0},  # Free only
                "order": ["free", "cheapest"],  # Prioritize free providers
                "require_parameters": True,  # Ensure we get exactly what we ask for
            }
        elif operation_mode == "emergency":
            # Emergency mode: prefer cheapest options with strict cost controls
            return {
                **base_preferences,
                "sort": "price",
                "max_price": {"prompt": 0.1, "completion": 0.1},  # Very low price limit
                "order": ["cheapest", "fastest"],  # Price first, then speed
                "timeout": 30,  # Shorter timeout for emergency mode
            }
        elif operation_mode == "conservative":
            # Conservative mode: balance price, reliability, and speed
            return {
                **base_preferences,
                "sort": "price",
                "max_price": {"prompt": 1.0, "completion": 2.0},  # Moderate price limit
                "order": ["cheapest", "most_reliable", "fastest"],
                "min_success_rate": 0.95,  # Require high reliability
                "timeout": 60,
            }
        else:
            # Normal mode: optimal performance with cost awareness
            return {
                **base_preferences,
                "sort": "throughput",  # Optimize for speed when budget allows
                "order": ["fastest", "most_reliable", "cheapest"],
                "min_success_rate": 0.98,  # Highest reliability standards
                "timeout": 90,
                "prefer_streaming": True,  # Enable streaming for better UX
            }

    def get_operation_mode_details(self, budget_remaining: float) -> Dict[str, Any]:
        """Get detailed information about current operation mode and routing strategy."""
        operation_mode = self.get_operation_mode(budget_remaining)
        budget_used = 100.0 - budget_remaining

        mode_details = {
            "normal": {
                "description": "Optimal GPT-5 model selection with performance priority",
                "model_preference": "GPT-5 models preferred, full tier available",
                "cost_strategy": "Quality-first routing with cost awareness",
                "provider_routing": "Throughput-optimized with reliability focus",
            },
            "conservative": {
                "description": "Cost-conscious routing with GPT-5 mini/nano preference",
                "model_preference": "GPT-5 mini/nano preferred, full tier limited",
                "cost_strategy": "Price-balanced routing with quality preservation",
                "provider_routing": "Price-first with reliability requirements",
            },
            "emergency": {
                "description": "Budget preservation mode with free model preference",
                "model_preference": "Free models preferred, GPT-5 nano for critical tasks",
                "cost_strategy": "Aggressive cost minimization",
                "provider_routing": "Cheapest available with speed priority",
            },
            "critical": {
                "description": "Budget exhaustion mode - free models only",
                "model_preference": "Free models only (Kimi-K2, OSS-20B)",
                "cost_strategy": "Zero-cost operation only",
                "provider_routing": "Free providers only with strict limits",
            },
        }

        current_details = mode_details[operation_mode]
        provider_prefs = self._get_provider_preferences_for_operation_mode(
            operation_mode
        )

        return {
            "operation_mode": operation_mode,
            "budget_used_percentage": budget_used,
            "budget_remaining_percentage": budget_remaining,
            "mode_description": current_details["description"],
            "model_preference": current_details["model_preference"],
            "cost_strategy": current_details["cost_strategy"],
            "provider_routing": current_details["provider_routing"],
            "openrouter_preferences": provider_prefs,
            "threshold_ranges": self.operation_thresholds,
        }

    def _apply_model_shortcuts(
        self, model_name: str, operation_mode: Optional[OperationMode]
    ) -> str:
        """Apply OpenRouter model shortcuts based on operation mode."""
        # Ensure model has provider prefix before adding shortcuts
        model_name = self._normalize_model_id(model_name)
        if operation_mode is None:
            # No shortcuts for testing or special cases
            return model_name
        elif operation_mode in ["emergency", "critical", "conservative"]:
            # Use :floor shortcut for price optimization
            if not model_name.endswith(":floor") and not model_name.endswith(":free"):
                return f"{model_name}:floor"
        elif operation_mode == "normal":
            # Do NOT auto-append :nitro. Many provider/model combos don't support it and it
            # has produced malformed IDs (e.g., gpt-5-nano:nitro) leading to provider errors.
            # If explicitly requested via env, allow opting in.
            try:
                import os  # local import to avoid top-level cycles
                enable_nitro = os.getenv("OPENROUTER_ENABLE_NITRO", "0") == "1"
            except Exception:
                enable_nitro = False

            if enable_nitro and (not model_name.endswith(":nitro")) and (not model_name.endswith(":free")):
                return f"{model_name}:nitro"

        return model_name

    def _normalize_model_id(self, model_name: str) -> str:
        """Normalize model identifiers to include provider prefix for OpenRouter/litellm.

        Examples:
        - gpt-5-nano -> openai/gpt-5-nano
        - gpt-5-mini:floor -> openai/gpt-5-mini:floor
        - claude-3-5-sonnet -> anthropic/claude-3-5-sonnet
        - gpt-oss-20b:free -> openai/gpt-oss-20b:free
        Leaves already-qualified IDs ("openai/...", "anthropic/...", "moonshotai/...",
        "metaculus/...") untouched.
        """
        if not model_name:
            return model_name
        if "/" in model_name or model_name.startswith("metaculus/"):
            return model_name

        # Preserve suffix like :nitro/:floor/:free while normalizing the base name
        parts = model_name.split(":", 1)
        base = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        normalized_base = base
        lower = base.lower()
        if lower.startswith("gpt-5") or lower.startswith("gpt-4o") or lower.startswith("gpt-oss"):
            normalized_base = f"openai/{base}"
        elif lower.startswith("claude"):
            normalized_base = f"anthropic/{base}"
        elif lower.startswith("kimi") or lower.startswith("k2"):
            normalized_base = f"moonshotai/{base}"

        return f"{normalized_base}{(':' + suffix) if suffix else ''}"

    def _get_api_key_for_model(self, model_name: str) -> Optional[str]:
        """Get the appropriate API key for a given model."""
        if model_name.startswith("metaculus/"):
            # Metaculus proxy models don't need API key
            return None
        elif model_name.startswith("openai/") or model_name.startswith("gpt-"):
            # OpenRouter models
            if self.openrouter_key and not self.openrouter_key.startswith("dummy_"):
                return self.openrouter_key
            else:
                logger.debug(f"OpenRouter API key not available for {model_name}")
                return None
        else:
            # Default to OpenRouter for unknown models
            return (
                self.openrouter_key
                if self.openrouter_key and not self.openrouter_key.startswith("dummy_")
                else None
            )

    def _create_emergency_model(self, tier: ModelTier) -> GeneralLlm:
        """Create emergency fallback model using available API keys."""
        config = self.model_configs[tier]

        # Try to use the best available emergency model
        if self.openrouter_key and not self.openrouter_key.startswith("dummy_"):
            # Use OpenRouter with free model as emergency
            return GeneralLlm(
                model="openai/gpt-oss-20b:free",  # Free model for emergency
                api_key=self.openrouter_key,
                base_url=self.openrouter_base_url,
                extra_headers=self.openrouter_headers,
                temperature=config.temperature,
                timeout=config.timeout,
                allowed_tries=1,
                # Ensure LiteLLM routes via OpenRouter; prevents provider ambiguity
                custom_llm_provider="openrouter",
            )
        elif os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true":
            # Use alternative free model despite proxy (GPT-4o removed)
            return GeneralLlm(
                model="moonshotai/kimi-k2:free",
                api_key=None,
                temperature=config.temperature,
                timeout=config.timeout,
                allowed_tries=1,
            )
        else:
            # Last resort - create a dummy model that will fail gracefully
            logger.error("No API keys available for emergency fallback")
            return GeneralLlm(
                model="dummy-model",
                api_key=None,
                temperature=config.temperature,
                timeout=config.timeout,
                allowed_tries=1,
            )

    async def detect_model_availability(self) -> Dict[str, bool]:
        """Detect availability of OpenRouter models with comprehensive testing."""
        availability = {}

        # Test cost-optimized models: GPT-5 primary + free fallbacks
        test_models = [
            "openai/gpt-5",  # Tier 1 (full) - GPT-5 primary
            "openai/gpt-5-mini",  # Tier 2 (mini) - GPT-5 mini
            "openai/gpt-5-nano",  # Tier 3 (nano) - GPT-5 nano
            "moonshotai/kimi-k2:free",  # Free fallback 1 - Kimi reasoning
            "openai/gpt-oss-20b:free",  # Free fallback 2 - OSS reliable
        ]

        logger.info("Starting OpenRouter model availability detection...")

        for model_name in test_models:
            try:
                # Create a test model instance without shortcuts for availability testing
                test_model = self._create_openrouter_model(
                    model_name, self.model_configs["mini"], None
                )
                if test_model is None:
                    availability[model_name] = False
                    logger.debug(f"✗ {model_name} could not be created")
                    continue

                # Try a simple test call with very short timeout
                # Short timeout to avoid blocking startup; health monitor will refine later
                await asyncio.wait_for(test_model.invoke("Test"), timeout=3.0)
                availability[model_name] = True
                logger.info(f"✓ {model_name} is available via OpenRouter")

            except asyncio.TimeoutError:
                availability[model_name] = False
                logger.debug(f"✗ {model_name} timeout during availability check")
            except Exception as e:
                availability[model_name] = False
                logger.debug(f"✗ {model_name} unavailable: {e}")

        # Log summary
        available_count = sum(1 for available in availability.values() if available)
        total_count = len(availability)
        logger.info(
            f"Model availability check complete: {available_count}/{total_count} models available"
        )

        return availability

    async def auto_configure_fallback_chains(self) -> Dict[ModelTier, List[str]]:
        """Automatically configure fallback chains based on model availability."""
        logger.info("Auto-configuring fallback chains based on model availability...")
        # Detect current model availability
        availability = await self.detect_model_availability()

        # If none available, return emergency chains
        if not any(availability.values()):
            logger.error("No OpenRouter models available - using emergency configuration")
            return self._get_emergency_fallback_chains()

        optimized_chains: Dict[ModelTier, List[str]] = {}
        for tier_name in ["nano", "mini", "full"]:
            tier = cast(ModelTier, tier_name)
            chain: List[str] = []
            primary_model = self.model_configs[tier].model_name
            if availability.get(primary_model, False):
                chain.append(primary_model)
                logger.info(f"✓ Primary model {primary_model} available for {tier} tier")
            else:
                logger.warning(f"⚠ Primary model {primary_model} unavailable for {tier} tier")

            if tier == "full":
                if availability.get("openai/gpt-5-mini", False):
                    chain.append("openai/gpt-5-mini")
                if availability.get("openai/gpt-oss-20b:free", False):
                    chain.append("openai/gpt-oss-20b:free")
                if availability.get("openai/gpt-5-nano", False):
                    chain.append("openai/gpt-5-nano")
                if availability.get("moonshotai/kimi-k2:free", False):
                    chain.append("moonshotai/kimi-k2:free")
            elif tier == "mini":
                if availability.get("openai/gpt-5-nano", False):
                    chain.append("openai/gpt-5-nano")
                if availability.get("openai/gpt-oss-20b:free", False):
                    chain.append("openai/gpt-oss-20b:free")
                if availability.get("moonshotai/kimi-k2:free", False):
                    chain.append("moonshotai/kimi-k2:free")
            else:  # nano
                if availability.get("openai/gpt-oss-20b:free", False):
                    chain.append("openai/gpt-oss-20b:free")
                if availability.get("moonshotai/kimi-k2:free", False):
                    chain.append("moonshotai/kimi-k2:free")

            optimized_chains[tier] = chain
            logger.info(f"Auto-configured {tier} tier chain: {' → '.join(chain)}")

        self.fallback_chains = optimized_chains
        return optimized_chains

    def _get_emergency_fallback_chains(self) -> Dict[ModelTier, List[str]]:
        """Get emergency fallback chains when no OpenRouter primary models are available.

        Uses only free community models (GPT-4o family removed).
        """
        logger.warning("Using emergency fallback chains - free models only (GPT-4o purged)")
        free_models = ["openai/gpt-oss-20b:free", "moonshotai/kimi-k2:free"]
        return {"nano": free_models, "mini": free_models, "full": free_models}

    async def validate_openrouter_configuration(self) -> Dict[str, Any]:
        """Validate OpenRouter configuration and report any issues."""
        validation_report: Dict[str, Any] = {
            "api_key_status": "missing",
            "base_url_status": "ok",
            "attribution_headers": {},
            "model_availability": {},
            "fallback_chains": {},
            "configuration_errors": cast(List[str], []),
            "recommendations": cast(List[str], []),
        }

        # Check API key
        if not self.openrouter_key:
            validation_report["configuration_errors"].append(
                "OPENROUTER_API_KEY not set"
            )
            validation_report["recommendations"].append(
                "Set OPENROUTER_API_KEY environment variable"
            )
        elif self.openrouter_key.startswith("dummy_"):
            validation_report["api_key_status"] = "dummy"
            validation_report["configuration_errors"].append(
                "Using dummy OpenRouter API key"
            )
            validation_report["recommendations"].append(
                "Replace dummy API key with real OpenRouter API key"
            )
        else:
            validation_report["api_key_status"] = "configured"

        # Check base URL
        if self.openrouter_base_url != "https://openrouter.ai/api/v1":
            validation_report["base_url_status"] = "incorrect"
            validation_report["configuration_errors"].append(
                f"Incorrect base URL: {self.openrouter_base_url}"
            )
            validation_report["recommendations"].append(
                "Set OPENROUTER_BASE_URL to https://openrouter.ai/api/v1"
            )

        # Check attribution headers
        validation_report["attribution_headers"] = self.openrouter_headers.copy()
        if not self.openrouter_headers.get("HTTP-Referer"):
            validation_report["recommendations"].append(
                "Set OPENROUTER_HTTP_REFERER for better ranking"
            )
        if not self.openrouter_headers.get("X-Title"):
            validation_report["recommendations"].append(
                "Set OPENROUTER_APP_TITLE for attribution"
            )

        # Test model availability if API key is valid
        if validation_report["api_key_status"] == "configured":
            try:
                validation_report["model_availability"] = (
                    await self.detect_model_availability()
                )

                # Auto-configure fallback chains based on availability
                validation_report["fallback_chains"] = (
                    await self.auto_configure_fallback_chains()
                )

                # Check if any models are available
                available_models = [
                    model
                    for model, available in validation_report[
                        "model_availability"
                    ].items()
                    if available
                ]
                if not available_models:
                    validation_report["configuration_errors"].append(
                        "No OpenRouter models are available"
                    )
                    validation_report["recommendations"].append(
                        "Check OpenRouter API key and account status"
                    )
                else:
                    logger.info(
                        f"OpenRouter validation successful: {len(available_models)} models available"
                    )

            except Exception as e:
                validation_report["configuration_errors"].append(
                    f"Model availability check failed: {e}"
                )
                validation_report["recommendations"].append(
                    "Check network connectivity and API key validity"
                )

        # Environment variable recommendations
        missing_env_vars = []
        recommended_env_vars = [
            "OPENROUTER_API_KEY",
            "OPENROUTER_BASE_URL",
            "OPENROUTER_HTTP_REFERER",
            "OPENROUTER_APP_TITLE",
            "DEFAULT_MODEL",
            "MINI_MODEL",
            "NANO_MODEL",
        ]

        for var in recommended_env_vars:
            if not os.getenv(var):
                missing_env_vars.append(var)

        if missing_env_vars:
            validation_report["recommendations"].append(
                f"Consider setting environment variables: {', '.join(missing_env_vars)}"
            )

        return validation_report

    async def health_monitor_startup(self) -> bool:
        """Perform comprehensive health monitoring on startup."""
        logger.info("Starting OpenRouter health monitoring...")

        try:
            # Validate configuration
            validation_report = await self.validate_openrouter_configuration()

            # Log configuration status
            if validation_report["configuration_errors"]:
                logger.error("OpenRouter configuration errors found:")
                for error in validation_report["configuration_errors"]:
                    logger.error(f"  - {error}")

            if validation_report["recommendations"]:
                logger.info("OpenRouter configuration recommendations:")
                for rec in validation_report["recommendations"]:
                    logger.info(f"  - {rec}")

            # Check if we have at least one working model per tier
            working_tiers = 0
            for tier in ["nano", "mini", "full"]:
                tier_literal = cast(ModelTier, tier)
                if self.model_status[tier_literal].is_available:
                    working_tiers += 1

            if working_tiers == 0:
                logger.error(
                    "No model tiers are available - system will have limited functionality"
                )
                return False
            elif working_tiers < 3:
                logger.warning(
                    f"Only {working_tiers}/3 model tiers available - some functionality may be degraded"
                )
            else:
                logger.info("All model tiers available - system fully operational")

            # Test each tier's health
            for tier in ["nano", "mini", "full"]:
                tier_literal = cast(ModelTier, tier)
                health_status = await self.check_model_health(tier_literal)
                if health_status.is_available:
                    logger.info(
                        f"✓ {tier.upper()} tier healthy: {health_status.model_name} ({health_status.response_time:.2f}s)"
                    )
                else:
                    logger.warning(
                        f"⚠ {tier.upper()} tier unhealthy: {health_status.error_message}"
                    )

            return working_tiers > 0

        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
            return False

    def get_openrouter_provider_routing_info(self) -> Dict[str, Any]:
        """Get OpenRouter provider routing configuration information."""
        return {
            "base_url": self.openrouter_base_url,
            "attribution_headers": self.openrouter_headers,
            "operation_modes": {
                "normal": "GPT-5 models with optimal routing (0-70% budget)",
                "conservative": "GPT-5 mini/nano only (70-85% budget)",
                "emergency": "Free models preferred (85-95% budget)",
                "critical": "Free models only (95-100% budget)",
            },
            "tier_models": {
                "full": self.model_configs["full"].model_name,  # openai/gpt-5
                "mini": self.model_configs["mini"].model_name,  # openai/gpt-5-mini
                "nano": self.model_configs["nano"].model_name,  # openai/gpt-5-nano
            },
            "current_fallback_chains": self.fallback_chains,
            "cost_optimized_fallbacks": {
                "full_fallbacks": [
                    "openai/gpt-5-mini",
                    "openai/gpt-oss-20b:free",
                    "moonshotai/kimi-k2:free",
                ],
                "mini_fallbacks": [
                    "openai/gpt-5-nano",
                    "openai/gpt-oss-20b:free",
                    "moonshotai/kimi-k2:free",
                ],
                "nano_fallbacks": [
                    "openai/gpt-oss-20b:free",
                    "moonshotai/kimi-k2:free",
                ],
            },
            "free_fallbacks": [
                "openai/gpt-oss-20b:free",  # GPT-OSS free model
                "moonshotai/kimi-k2:free",  # Kimi free model
            ],
            "model_status": {
                tier: status.__dict__ for tier, status in self.model_status.items()
            },
        }

    async def continuous_health_monitoring(self, interval_seconds: int = 300) -> None:
        """Continuously monitor OpenRouter model health and auto-reconfigure as needed."""
        logger.info(
            f"Starting continuous health monitoring (interval: {interval_seconds}s)"
        )

        while True:
            try:
                await asyncio.sleep(interval_seconds)

                logger.debug("Performing periodic health check...")

                # Check health of all tiers
                unhealthy_tiers = []
                for tier in ["nano", "mini", "full"]:
                    tier_literal = cast(ModelTier, tier)
                    health_status = await self.check_model_health(tier_literal)
                    if not health_status.is_available:
                        unhealthy_tiers.append(tier_literal)
                        logger.warning(
                            f"Tier {tier} unhealthy: {health_status.error_message}"
                        )

                # If any tiers are unhealthy, try to reconfigure
                if unhealthy_tiers:
                    logger.info(f"Reconfiguring unhealthy tiers: {unhealthy_tiers}")
                    await self.auto_configure_fallback_chains()

                    # Re-initialize unhealthy models
                    for tier_literal in unhealthy_tiers:
                        config = self.model_configs[tier_literal]
                        try:
                            model, status = self._initialize_model_with_fallback(tier_literal, config)
                            self.models[tier_literal] = model
                            self.model_status[tier_literal] = status

                            if status.is_available:
                                logger.info(
                                    f"✓ Successfully reconfigured {tier_literal} tier: {status.model_name}"
                                )
                            else:
                                logger.warning(
                                    f"⚠ {tier_literal} tier still unhealthy after reconfiguration"
                                )
                        except Exception as e:
                            logger.error(f"Failed to reconfigure {tier_literal} tier: {e}")

                # Log periodic status
                healthy_tiers = sum(
                    1 for status in self.model_status.values() if status.is_available
                )
                logger.debug(f"Health check complete: {healthy_tiers}/3 tiers healthy")

            except Exception as e:
                logger.error(f"Error in continuous health monitoring: {e}")
                # Continue monitoring despite errors
                continue

    def get_configuration_status_report(self) -> Dict[str, Any]:
        """Get comprehensive configuration and status report."""
        return {
            "router_info": {
                "base_url": self.openrouter_base_url,
                "api_key_configured": bool(
                    self.openrouter_key and not self.openrouter_key.startswith("dummy_")
                ),
                "attribution_headers": self.openrouter_headers,
                "operation_thresholds": self.operation_thresholds,
            },
            "model_configurations": {
                tier: {
                    "model_name": config.model_name,
                    "cost_per_million_input": config.cost_per_million_input,
                    "cost_per_million_output": config.cost_per_million_output,
                    "temperature": config.temperature,
                    "timeout": config.timeout,
                    "description": config.description,
                }
                for tier, config in self.model_configs.items()
            },
            "model_status": {
                tier: {
                    "tier": status.tier,
                    "model_name": status.model_name,
                    "is_available": status.is_available,
                    "last_check": status.last_check,
                    "response_time": status.response_time,
                    "error_message": status.error_message,
                }
                for tier, status in self.model_status.items()
            },
            "fallback_chains": self.fallback_chains,
            "routing_strategy": self.routing_strategy,
            "environment_variables": {
                "OPENROUTER_API_KEY": (
                    "configured" if os.getenv("OPENROUTER_API_KEY") else "missing"
                ),
                "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL", "default"),
                "OPENROUTER_HTTP_REFERER": os.getenv(
                    "OPENROUTER_HTTP_REFERER", "not_set"
                ),
                "OPENROUTER_APP_TITLE": os.getenv("OPENROUTER_APP_TITLE", "not_set"),
                "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "default"),
                "MINI_MODEL": os.getenv("MINI_MODEL", "default"),
                "NANO_MODEL": os.getenv("NANO_MODEL", "default"),
                "ENABLE_PROXY_CREDITS": os.getenv("ENABLE_PROXY_CREDITS", "true"),
            },
        }

    def get_operation_mode(self, budget_remaining: float) -> OperationMode:
        """Determine operation mode based on budget utilization."""
        budget_used = 100.0 - budget_remaining

        for mode, (min_used, max_used) in self.operation_thresholds.items():
            if min_used <= budget_used < max_used:
                return cast(OperationMode, mode)

        # Default to critical if over 100%
        return cast(OperationMode, "critical")

    def get_model_costs(self) -> Dict[ModelTier, Dict[str, Union[float, str]]]:
        """Get OpenRouter pricing for each model tier (separate input/output costs)."""
        return {
            tier: {
                "input_cost_per_million": config.cost_per_million_input,
                "output_cost_per_million": config.cost_per_million_output,
                "model_name": config.model_name,
            }
            for tier, config in self.model_configs.items()
        }

    async def check_model_health(self, tier: ModelTier) -> ModelStatus:
        """Check health of a specific model tier."""
        try:
            model = self.models[tier]
            start_time = asyncio.get_event_loop().time()

            # Simple health check with minimal prompt
            await model.invoke("Test")

            response_time = asyncio.get_event_loop().time() - start_time

            status = ModelStatus(
                tier=tier,
                model_name=model.model,
                is_available=True,
                last_check=start_time,
                response_time=response_time,
            )

            self.model_status[tier] = status
            return status

        except Exception as e:
            status = ModelStatus(
                tier=tier,
                model_name=self.models[tier].model,
                is_available=False,
                last_check=asyncio.get_event_loop().time(),
                error_message=str(e),
            )

            self.model_status[tier] = status
            return status

    def analyze_content_complexity(
        self, content: str, task_type: TaskType
    ) -> ComplexityLevel:
        """
        Analyze content complexity for optimal model selection.

        Args:
            content: The content to analyze
            task_type: Type of task being performed

        Returns:
            ComplexityLevel: minimal, medium, or high
        """
        content_length = len(content)
    # word_count intentionally omitted (unused)

        # Content length factors
        length_score = 0
        if content_length > 2000:
            length_score += 2
        elif content_length > 500:
            length_score += 1

        # Word complexity factors
        complexity_indicators = [
            "analysis",
            "complex",
            "detailed",
            "comprehensive",
            "intricate",
            "sophisticated",
            "nuanced",
            "multifaceted",
            "elaborate",
            "thorough",
            "probability",
            "forecast",
            "prediction",
            "uncertainty",
            "scenario",
            "research",
            "evidence",
            "citation",
            "source",
            "study",
        ]

        complexity_score = sum(
            1 for indicator in complexity_indicators if indicator in content.lower()
        )

        # Task-specific adjustments
        task_multipliers = {
            "forecast": 1.5,  # Forecasting is inherently complex
            "research": 1.2,  # Research requires synthesis
            "validation": 0.8,  # Validation is typically simpler
            "simple": 0.5,  # Simple tasks are straightforward
        }

        total_score = (length_score + complexity_score) * task_multipliers.get(
            task_type, 1.0
        )

        # Determine complexity level
        if total_score >= 4:
            return "high"
        elif total_score >= 2:
            return "medium"
        else:
            return "minimal"

    def estimate_token_usage(
        self, content: str, task_type: TaskType
    ) -> Tuple[int, int]:
        """
        Estimate input and output token usage for cost calculation.

        Args:
            content: Input content
            task_type: Type of task

        Returns:
            Tuple of (input_tokens, estimated_output_tokens)
        """
        # Basic token estimation (4 chars per token average)
        input_tokens = len(content) // 4

        # Task-specific output multipliers based on typical response patterns
        output_multipliers = {
            "validation": 0.2,  # Short validation responses
            "simple": 0.3,  # Brief simple responses
            "research": 1.8,  # Detailed research with citations
            "forecast": 2.5,  # Comprehensive forecasting analysis
        }

        multiplier = output_multipliers.get(task_type, 1.0)
        estimated_output_tokens = int(input_tokens * multiplier)

        # Add base response overhead
        estimated_output_tokens += 50

        return input_tokens, estimated_output_tokens

    def assess_urgency_priority(self, content: str) -> float:
        """
        Assess urgency and priority for task routing.

        Args:
            content: Content to analyze

        Returns:
            Priority score (0.0 to 1.0, higher = more urgent)
        """
        urgency_indicators = [
            "urgent",
            "immediate",
            "asap",
            "critical",
            "emergency",
            "deadline",
            "time-sensitive",
            "priority",
            "important",
        ]

        content_lower = content.lower()
        urgency_score = sum(
            1 for indicator in urgency_indicators if indicator in content_lower
        )

        # Normalize to 0-1 scale
        return min(urgency_score / 3.0, 1.0)

    def choose_model(
        self,
        task_type: TaskType,
        complexity: Optional[ComplexityLevel] = None,
        content_length: int = 0,
        budget_remaining: float = 100.0,
        content: Optional[str] = None,
    ) -> Tuple[GeneralLlm, ModelTier]:
        """
        Choose optimal model based on task requirements and budget constraints with enhanced logic.

        Args:
            task_type: Type of task (validation, research, forecast, simple)
            complexity: Complexity level (minimal, medium, high)
            content_length: Length of content to process
            budget_remaining: Remaining budget percentage (0-100)
            content: Optional content for advanced analysis

        Returns:
            Tuple of (selected_model, model_tier)
        """
        # Enhanced complexity analysis if content is provided
        if content and not complexity:
            complexity = self.analyze_content_complexity(content, task_type)
            logger.debug(f"Auto-detected complexity: {complexity} for {task_type}")

        # Domain-specific complexity assessment
        if content:
            domain_complexity = self._assess_domain_complexity(content)
            if domain_complexity == "high" and complexity != "high":
                complexity = "medium"  # Upgrade if domain is complex
                logger.debug(f"Domain complexity upgrade: {complexity}")

        # Priority-based routing adjustments
        if content:
            priority_score = self.assess_urgency_priority(content)
            if priority_score > 0.7 and budget_remaining > 30:
                # High priority tasks get better models if budget allows
                logger.debug(
                    f"High priority task (score: {priority_score:.2f}), considering model upgrade"
                )

        # Determine operation mode based on budget
        operation_mode = self.get_operation_mode(budget_remaining)

        # Base model selection from routing strategy
        base_tier = cast(ModelTier, self.routing_strategy.get(task_type, "mini"))

        # Operation mode adjustments
        selected_tier = self._adjust_for_operation_mode(
            base_tier, operation_mode, task_type
        )

        # Complexity-based adjustments (only upgrade if budget allows)
        selected_tier = self._adjust_for_complexity(
            selected_tier, complexity, operation_mode
        )

        # Content length adjustments for very short content
        if content_length < 100 and selected_tier != "nano":
            selected_tier = "nano"
            logger.debug(
                f"Short content ({content_length} chars): using nano model for {task_type}"
            )

        # Ensure model is available, fallback if necessary
        if not self.model_status[selected_tier].is_available:
            selected_tier = self._find_available_fallback(selected_tier)

        selected_model = self.models[selected_tier]

        logger.debug(
            f"Selected {selected_tier} model for {task_type} "
            f"(mode: {operation_mode}, complexity: {complexity}, budget: {budget_remaining:.1f}%)"
        )

        return selected_model, selected_tier

    def prioritize_tasks_by_budget(
        self, tasks: List[Dict[str, Any]], budget_remaining: float
    ) -> List[Dict[str, Any]]:
        """
        Prioritize tasks based on budget constraints and importance.

        Args:
            tasks: List of task dictionaries with 'type', 'content', 'priority' keys
            budget_remaining: Remaining budget percentage

        Returns:
            Prioritized list of tasks
        """
        operation_mode = self.get_operation_mode(budget_remaining)

        # Calculate cost estimates for all tasks
        for task in tasks:
            task["estimated_cost"] = self.get_cost_estimate(
                task_type=task["type"],
                content_length=len(task.get("content", "")),
                complexity=task.get("complexity"),
                budget_remaining=budget_remaining,
                content=task.get("content"),
            )

            # Calculate priority score based on multiple factors
            base_priority = task.get("priority", 0.5)
            urgency_score = self.assess_urgency_priority(task.get("content", ""))

            # Adjust priority based on operation mode
            if operation_mode == "critical":
                # In critical mode, heavily favor free tasks
                cost_factor = 0.0 if task["estimated_cost"] == 0 else -2.0
            elif operation_mode == "emergency":
                # In emergency mode, penalize expensive tasks
                cost_factor = -task["estimated_cost"] * 10
            elif operation_mode == "conservative":
                # In conservative mode, moderate cost consideration
                cost_factor = -task["estimated_cost"] * 2
            else:
                # In normal mode, slight cost consideration
                cost_factor = -task["estimated_cost"] * 0.5

            task["final_priority"] = base_priority + urgency_score + cost_factor

        # Sort by final priority (highest first)
        return sorted(tasks, key=lambda x: x["final_priority"], reverse=True)

    def get_budget_optimization_suggestions(
        self, budget_remaining: float, recent_costs: List[float]
    ) -> List[str]:
        """Generate budget optimization suggestions based on current state."""
        suggestions = []
        operation_mode = self.get_operation_mode(budget_remaining)

        if recent_costs:
            avg_cost = sum(recent_costs) / len(recent_costs)

            if operation_mode == "normal" and avg_cost > 0.02:
                suggestions.append(
                    f"Consider using GPT-5 mini for research tasks to reduce average cost "
                    f"from ${avg_cost:.4f} to ~${avg_cost * 0.6:.4f} per task"
                )

            if operation_mode == "conservative":
                suggestions.append(
                    "Prioritize validation and simple tasks that can use GPT-5 nano "
                    "to preserve budget for critical forecasting tasks"
                )

            if operation_mode == "emergency":
                suggestions.append(
                    "Switch to free models (Kimi-K2, OSS-20B) for non-critical tasks "
                    "to extend operational capacity"
                )

        if budget_remaining < 20:
            suggestions.append(
                "Consider batching similar tasks to reduce API overhead and "
                "maximize remaining budget efficiency"
            )

        if operation_mode == "critical":
            suggestions.append(
                "Operating in critical mode - only free models available. "
                "Focus on highest-priority tasks only."
            )

        return suggestions

    async def intelligent_fallback_with_recovery(
        self,
        task_type: TaskType,
        content: str,
        complexity: Optional[ComplexityLevel] = None,
        budget_remaining: float = 100.0,
        max_retries: int = 3,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute query with intelligent fallback and comprehensive error recovery.

        Args:
            task_type: Type of task to perform
            content: Content to process
            complexity: Optional complexity level
            budget_remaining: Remaining budget percentage
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (response, execution_metadata)
        """
        execution_metadata: Dict[str, Any] = {
            "attempts": cast(List[Dict[str, Any]], []),
            "final_model": None,
            "final_tier": None,
            "total_cost": 0.0,
            "fallback_used": False,
            "recovery_actions": cast(List[str], []),
        }

        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                # Choose model for this attempt
                model, tier = self.choose_model(
                    task_type=task_type,
                    complexity=complexity,
                    content_length=len(content),
                    budget_remaining=budget_remaining,
                    content=content,
                )

                attempt_info = {
                    "attempt_number": attempt + 1,
                    "model": model.model,
                    "tier": tier,
                    "timestamp": asyncio.get_event_loop().time(),
                }

                # Add anti-slop directives
                enhanced_prompt = self._add_anti_slop_directives(
                    content, task_type, tier
                )

                # Execute with timeout and error handling
                response = await asyncio.wait_for(
                    model.invoke(enhanced_prompt),
                    timeout=self.model_configs[tier].timeout,
                )

                # Validate response quality
                validated_response = self._validate_response_quality(
                    response, task_type
                )

                # Success - update metadata
                attempt_info["status"] = "success"
                attempt_info["response_length"] = len(validated_response)
                execution_metadata["attempts"].append(attempt_info)
                execution_metadata["final_model"] = model.model
                execution_metadata["final_tier"] = tier

                # Estimate cost for successful attempt
                estimated_cost = self.get_cost_estimate(
                    task_type, len(content), complexity, budget_remaining, content
                )
                execution_metadata["total_cost"] = estimated_cost

                logger.info(
                    f"Task completed successfully on attempt {attempt + 1} using {tier} model"
                )
                return validated_response, execution_metadata

            except asyncio.TimeoutError as e:
                last_error = e
                attempt_info["status"] = "timeout"
                attempt_info["error"] = str(e)
                execution_metadata["attempts"].append(attempt_info)
                execution_metadata["recovery_actions"].append(
                    f"Timeout on attempt {attempt + 1}"
                )

                logger.warning(f"Timeout on attempt {attempt + 1} with {tier} model")

                # Timeout recovery: try faster model or reduce complexity
                if complexity == "high":
                    complexity = "medium"
                    execution_metadata["recovery_actions"].append(
                        "Reduced complexity from high to medium"
                    )
                elif tier == "full":
                    # Force downgrade to faster model
                    budget_remaining = min(
                        budget_remaining, 50.0
                    )  # Simulate conservative mode
                    execution_metadata["recovery_actions"].append(
                        "Forced downgrade to faster model"
                    )

            except Exception as e:
                last_error = e
                attempt_info["status"] = "error"
                attempt_info["error"] = str(e)
                execution_metadata["attempts"].append(attempt_info)

                logger.warning(f"Error on attempt {attempt + 1} with {tier} model: {e}")

                # Error-specific recovery strategies
                if "rate limit" in str(e).lower():
                    execution_metadata["recovery_actions"].append(
                        "Rate limit detected - switching provider"
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                elif "context length" in str(e).lower():
                    execution_metadata["recovery_actions"].append(
                        "Context length exceeded - truncating content"
                    )
                    content = content[: len(content) // 2]  # Truncate content
                elif (
                    "insufficient funds" in str(e).lower() or "quota" in str(e).lower()
                ):
                    execution_metadata["recovery_actions"].append(
                        "Budget/quota issue - forcing free models"
                    )
                    budget_remaining = 0.0  # Force critical mode
                else:
                    execution_metadata["recovery_actions"].append(
                        f"Generic error recovery: {type(e).__name__}"
                    )

                # Brief delay before retry
                await asyncio.sleep(min(2**attempt, 10))

        # All attempts failed - create emergency response
        execution_metadata["fallback_used"] = True
        execution_metadata["final_error"] = (
            str(last_error) if last_error else "Unknown error"
        )

        emergency_response = self._create_emergency_response_with_context(
            task_type, content, execution_metadata
        )

        logger.error(
            f"All {max_retries} attempts failed for {task_type}. Using emergency response."
        )
        return emergency_response, execution_metadata

    def _create_emergency_response_with_context(
        self, task_type: TaskType, content: str, metadata: Dict[str, Any]
    ) -> str:
        """Create contextual emergency response when all models fail."""
        failed_models = [
            attempt.get("model", "unknown") for attempt in metadata["attempts"]
        ]
        recovery_actions = metadata.get("recovery_actions", [])

        base_response = f"""
EMERGENCY RESPONSE - API FAILURES

Task Type: {task_type}
Failed Models: {', '.join(failed_models)}
Recovery Actions Attempted: {len(recovery_actions)}

Content Preview: {content[:200]}{'...' if len(content) > 200 else ''}
"""

        if task_type == "research":
            return (
                base_response
                + """
Research Status: Unable to complete due to system failures.
Recommendation: Proceed with forecast based on available question information only.
Quality Note: This response lacks the usual research depth due to technical constraints.
"""
            )
        elif task_type == "forecast":
            return (
                base_response
                + """
Forecast Status: Unable to generate detailed prediction due to system failures.
Fallback Strategy: Publishing withheld due to insufficient analysis capability.
Quality Note: Forecasting paused to avoid low-confidence neutral outputs.
Recommendation: Manual review recommended when systems are restored.
"""
            )
        elif task_type == "validation":
            return (
                base_response
                + """
Validation Status: Unable to complete validation due to system failures.
Fallback: Assuming content requires manual review.
Quality Note: Validation could not be performed - proceed with caution.
"""
            )
        else:
            return (
                base_response
                + """
Task Status: Unable to complete due to system failures.
Recommendation: Retry when systems are restored or use alternative approach.
"""
            )

    async def test_model_chain_health(self, tier: ModelTier) -> Dict[str, Any]:
        """Test the health of an entire model fallback chain."""
        chain = self.fallback_chains[tier]
        health_report: Dict[str, Any] = {
            "tier": tier,
            "chain_length": len(chain),
            "healthy_models": [],
            "failed_models": [],
            "total_response_time": 0.0,
            "fastest_model": None,
            "most_reliable": None,
        }

        for model_name in chain:
            try:
                start_time = asyncio.get_event_loop().time()

                # Create test model
                test_model = self._create_openrouter_model(
                    model_name, self.model_configs[tier], "normal"
                )

                if test_model is None:
                    health_report["failed_models"].append(
                        {
                            "model": model_name,
                            "error": "Could not create model instance",
                        }
                    )
                    continue

                # Simple health check
                response = await asyncio.wait_for(
                    test_model.invoke("Test health check"), timeout=15.0
                )

                response_time = asyncio.get_event_loop().time() - start_time
                health_report["total_response_time"] += response_time

                model_health = {
                    "model": model_name,
                    "response_time": response_time,
                    "response_length": len(response),
                    "status": "healthy",
                }

                health_report["healthy_models"].append(model_health)

                # Track fastest model
                if (
                    health_report["fastest_model"] is None
                    or response_time < health_report["fastest_model"]["response_time"]
                ):
                    health_report["fastest_model"] = model_health

            except Exception as e:
                health_report["failed_models"].append(
                    {"model": model_name, "error": str(e)}
                )

        # Calculate reliability metrics
        total_models = len(chain)
        healthy_count = len(health_report["healthy_models"])
        health_report["chain_reliability"] = (
            healthy_count / total_models if total_models > 0 else 0.0
        )
        health_report["has_working_fallback"] = healthy_count > 0

        return health_report

    def _assess_domain_complexity(self, content: str) -> ComplexityLevel:
        """Assess domain-specific complexity factors."""
        content_lower = content.lower()

        # High complexity domains
        high_complexity_domains = [
            "artificial intelligence",
            "machine learning",
            "quantum",
            "cryptocurrency",
            "geopolitics",
            "economics",
            "climate change",
            "biotechnology",
            "nuclear",
            "financial markets",
            "regulatory",
            "policy",
            "international relations",
        ]

        # Medium complexity domains
        medium_complexity_domains = [
            "technology",
            "business",
            "politics",
            "science",
            "healthcare",
            "energy",
            "transportation",
            "education",
            "social media",
        ]

        for domain in high_complexity_domains:
            if domain in content_lower:
                return "high"

        for domain in medium_complexity_domains:
            if domain in content_lower:
                return "medium"

        return "minimal"

    def _adjust_for_operation_mode(
        self, base_tier: ModelTier, mode: OperationMode, task_type: TaskType
    ) -> ModelTier:
        """Adjust model tier based on operation mode - optimized for free fallbacks."""
        if mode == "critical":
            # Critical mode: free models only (handled by fallback chain)
            return base_tier  # Will use free models from chain
        elif mode == "emergency":
            # Emergency mode: prefer nano, allow mini for forecasts
            if task_type == "forecast" and base_tier == "full":
                return "mini"  # Downgrade forecasting to mini
            else:
                return "nano"  # Everything else to nano
        elif mode == "conservative":
            # Conservative mode: avoid full model, prefer mini/nano
            if base_tier == "full":
                return "mini"
            else:
                return base_tier
        else:
            # Normal mode: use GPT-5 models as configured
            return base_tier

    def _adjust_for_complexity(
        self,
        tier: ModelTier,
        complexity: Optional[ComplexityLevel],
        mode: OperationMode,
    ) -> ModelTier:
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
        tier_order: List[ModelTier] = ["nano", "mini", "full"]
        preferred_index = tier_order.index(preferred_tier)

        # First try the preferred tier and lower cost options
        for i in range(preferred_index, len(tier_order)):
            tier = tier_order[i]
            if self.model_status[tier].is_available:
                if tier != preferred_tier:
                    logger.info(
                        f"Fallback: using {tier} instead of unavailable {preferred_tier}"
                    )
                return tier

        # If no lower-cost options, try higher-cost options
        for i in range(preferred_index - 1, -1, -1):
            tier = tier_order[i]
            if self.model_status[tier].is_available:
                logger.warning(
                    f"Emergency fallback: using {tier} instead of unavailable {preferred_tier}"
                )
                return tier

        # If nothing is available, return nano (should have emergency fallback)
        logger.error("No models available, using nano emergency fallback")
        return "nano"

    async def route_query(
        self,
        task_type: TaskType,
        content: str,
        complexity: Optional[ComplexityLevel] = None,
        budget_remaining: float = 100.0,
    ) -> str:
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
            budget_remaining=budget_remaining,
            content=content,  # Pass content for enhanced analysis
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
                fallback_prompt = self._add_anti_slop_directives(
                    content, task_type, "nano"
                )
                return await self.models["nano"].invoke(fallback_prompt)
            else:
                raise

    def _add_anti_slop_directives(
        self, prompt: str, task_type: TaskType, model_tier: ModelTier
    ) -> str:
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
""",
        }

        # Model-tier specific adjustments
        tier_adjustments = {
            "nano": "• Prioritize speed and accuracy over depth\n• Focus on essential information only\n",
            "mini": "• Balance depth with efficiency\n• Provide moderate detail with good reasoning\n",
            "full": "• Use maximum reasoning capability\n• Provide comprehensive analysis when warranted\n",
        }

        # Combine directives
        full_directives = (
            base_directives
            + task_directives.get(task_type, "")
            + tier_adjustments.get(model_tier, "")
        )

        return f"{full_directives}\n\n{prompt}"

    def _validate_response_quality(self, response: str, task_type: TaskType) -> str:
        """Apply quality validation to response."""

        # Check for basic quality indicators
        if len(response.strip()) < 10:
            logger.warning(f"Response too short for {task_type}: {len(response)} chars")

        # Check for uncertainty acknowledgment in forecasting
        if task_type == "forecast":
            uncertainty_indicators = [
                "uncertain",
                "unclear",
                "difficult to predict",
                "confidence",
                "probability",
            ]
            if not any(
                indicator in response.lower() for indicator in uncertainty_indicators
            ):
                response += "\n\n[Note: Moderate confidence given available evidence and inherent uncertainty]"

        # Length compliance check
        word_count = len(response.split())
        if word_count > 400 and task_type != "forecast":
            logger.warning(
                f"Response exceeds recommended length for {task_type}: {word_count} words"
            )

        return response

    def get_cost_estimate(
        self,
        task_type: TaskType,
        content_length: int,
        complexity: Optional[ComplexityLevel] = None,
        budget_remaining: float = 100.0,
        content: Optional[str] = None,
    ) -> float:
        """Estimate cost for a given task with enhanced GPT-5 pricing analysis."""
        _, tier = self.choose_model(
            task_type, complexity, content_length, budget_remaining, content
        )

        # Use enhanced token estimation if content is available
        if content:
            input_tokens, output_tokens = self.estimate_token_usage(content, task_type)
        else:
            # Fallback to basic estimation
            input_tokens = content_length // 4  # 4 chars per token average
            output_multipliers = {
                "validation": 0.3,
                "simple": 0.5,
                "research": 1.8,
                "forecast": 2.2,
            }
            multiplier = output_multipliers.get(task_type, 1.5)
            output_tokens = int(input_tokens * multiplier)

        # Get cost-optimized pricing for selected tier
        config = self.model_configs[tier]

        # Check if we're likely to use free models based on operation mode
        operation_mode = self.get_operation_mode(budget_remaining)
        if operation_mode in ["emergency", "critical"]:
            # Likely to use free models, so cost is $0
            return 0.0

        # Calculate cost with separate input/output pricing
        input_cost = (input_tokens / 1_000_000) * config.cost_per_million_input
        output_cost = (output_tokens / 1_000_000) * config.cost_per_million_output
        total_cost = input_cost + output_cost

        logger.debug(
            f"Cost estimate for {task_type} ({tier}): "
            f"${total_cost:.6f} ({input_tokens} in + {output_tokens} out tokens)"
        )

        return total_cost

    def get_model_status(self) -> Dict[str, str]:
        """Get comprehensive status of all models."""
        status: Dict[str, str] = {}
        for tier_key, model_status in self.model_status.items():
            if model_status.is_available:
                response_info = (
                    f" ({model_status.response_time:.2f}s)"
                    if model_status.response_time
                    else ""
                )
                status[str(tier_key)] = f"✓ {model_status.model_name} (Ready{response_info})"
            else:
                error_info = (
                    f" - {model_status.error_message}"
                    if model_status.error_message
                    else ""
                )
                status[str(tier_key)] = f"✗ {model_status.model_name} (Unavailable{error_info})"
        return status

    def get_detailed_status(self) -> Dict[str, Dict]:
        """Get detailed status information for monitoring."""
        detailed_status: Dict[str, Dict[str, Any]] = {}
        for tier_key, status in self.model_status.items():
            config = self.model_configs[tier_key]
            detailed_status[str(tier_key)] = {
                "model_name": status.model_name,
                "is_available": status.is_available,
                "cost_per_million_input": config.cost_per_million_input,
                "cost_per_million_output": config.cost_per_million_output,
                "description": config.description,
                "last_check": status.last_check,
                "response_time": status.response_time,
                "error_message": status.error_message,
                "fallback_chain": self.fallback_chains[tier_key],
            }
        return detailed_status

    def get_routing_explanation(
        self,
        task_type: TaskType,
        complexity: Optional[ComplexityLevel] = None,
        content_length: int = 0,
        budget_remaining: float = 100.0,
    ) -> str:
        """Get detailed explanation of routing decision."""
        operation_mode = self.get_operation_mode(budget_remaining)
        base_tier = self.routing_strategy.get(task_type, "mini")
        selected_model, selected_tier = self.choose_model(
            task_type, complexity, content_length, budget_remaining
        )

        explanation = [
            f"Task: {task_type}",
            f"Operation Mode: {operation_mode} (budget remaining: {budget_remaining:.1f}%)",
            f"Base Tier: {base_tier}",
            f"Selected Tier: {selected_tier}",
            f"Model: {selected_model.model}",
            f"Estimated Cost: ${self.get_cost_estimate(task_type, content_length, complexity, budget_remaining):.6f}",
        ]

        if complexity:
            explanation.insert(2, f"Complexity: {complexity}")
        if content_length > 0:
            explanation.insert(-2, f"Content Length: {content_length} chars")

        return " | ".join(explanation)

    def analyze_content_for_routing(
        self, content: str, task_type: TaskType
    ) -> ContentAnalysis:
        """
        Comprehensive content analysis for optimal model selection.

        Args:
            content: Content to analyze
            task_type: Type of task being performed

        Returns:
            ContentAnalysis with detailed metrics
        """
        # Basic metrics
        length = len(content)
        word_count = len(content.split())
        estimated_tokens = max(length // 4, word_count)  # Conservative token estimate

        # Complexity indicators
        complexity_indicators = [
            "analysis",
            "complex",
            "detailed",
            "comprehensive",
            "intricate",
            "sophisticated",
            "nuanced",
            "multifaceted",
            "elaborate",
            "thorough",
            "probability",
            "forecast",
            "prediction",
            "uncertainty",
            "scenario",
            "research",
            "evidence",
            "citation",
            "source",
            "study",
            "correlation",
            "causation",
            "statistical",
            "quantitative",
            "qualitative",
            "methodology",
        ]

        found_indicators = [
            indicator
            for indicator in complexity_indicators
            if indicator in content.lower()
        ]

        # Calculate complexity score
        base_complexity = len(found_indicators) / len(complexity_indicators)
        length_factor = min(length / 2000, 1.0)  # Normalize to 2000 chars
        word_density = word_count / max(length, 1) * 100  # Words per 100 chars

        complexity_score = (
            base_complexity * 0.5
            + length_factor * 0.3
            + min(word_density / 20, 1.0) * 0.2
        )

        # Domain assessment
        domain = self._assess_content_domain(content)

        # Urgency assessment
        urgency = self.assess_urgency_priority(content)

        return ContentAnalysis(
            length=length,
            complexity_score=complexity_score,
            domain=domain,
            urgency=urgency,
            estimated_tokens=estimated_tokens,
            word_count=word_count,
            complexity_indicators=found_indicators,
        )

    def _assess_content_domain(self, content: str) -> str:
        """Assess the domain/topic of content for specialized routing."""
        content_lower = content.lower()

        domain_keywords = {
            "ai_tech": [
                "artificial intelligence",
                "machine learning",
                "ai",
                "neural network",
                "deep learning",
                "algorithm",
                "automation",
                "robotics",
            ],
            "finance": [
                "financial",
                "economic",
                "market",
                "investment",
                "trading",
                "cryptocurrency",
                "bitcoin",
                "stock",
                "bond",
                "inflation",
            ],
            "geopolitics": [
                "geopolitical",
                "international",
                "diplomatic",
                "military",
                "conflict",
                "war",
                "treaty",
                "sanctions",
                "alliance",
            ],
            "science": [
                "scientific",
                "research",
                "study",
                "experiment",
                "hypothesis",
                "theory",
                "discovery",
                "breakthrough",
                "publication",
            ],
            "climate": [
                "climate",
                "environmental",
                "carbon",
                "emission",
                "renewable",
                "sustainability",
                "global warming",
                "green energy",
            ],
            "health": [
                "medical",
                "health",
                "disease",
                "treatment",
                "vaccine",
                "pharmaceutical",
                "clinical",
                "patient",
                "diagnosis",
            ],
            "technology": [
                "technology",
                "software",
                "hardware",
                "digital",
                "cyber",
                "internet",
                "platform",
                "innovation",
                "startup",
            ],
            "politics": [
                "political",
                "election",
                "government",
                "policy",
                "legislation",
                "congress",
                "parliament",
                "vote",
                "campaign",
            ],
        }

        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            # Select the domain with the highest score deterministically
            return max(domain_scores.items(), key=lambda kv: kv[1])[0]
        return "general"

    def choose_optimal_model(
        self,
        task_type: TaskType,
        content: str,
        complexity: Optional[ComplexityLevel] = None,
        budget_context: Optional[BudgetContext] = None,
        priority: TaskPriority = "normal",
    ) -> OpenRouterModelSelection:
        """
        Advanced model selection with comprehensive analysis and OpenRouter optimization.

        Args:
            task_type: Type of task to perform
            content: Content to process
            complexity: Optional complexity override
            budget_context: Budget context for decision making
            priority: Task priority level

        Returns:
            OpenRouterModelSelection with detailed rationale
        """
        # Analyze content comprehensively
        content_analysis = self.analyze_content_for_routing(content, task_type)

        # Use provided complexity or auto-detect
        if complexity is None:
            if content_analysis.complexity_score >= 0.7:
                complexity = "high"
            elif content_analysis.complexity_score >= 0.4:
                complexity = "medium"
            else:
                complexity = "minimal"

        # Create budget context if not provided
        if budget_context is None:
            budget_context = BudgetContext(
                remaining_percentage=100.0,
                estimated_questions_remaining=1000,
                current_burn_rate=0.01,
                operation_mode="normal",
                budget_used_percentage=0.0,
            )

        # Choose model using existing logic
        selected_model, selected_tier = self.choose_model(
            task_type=task_type,
            complexity=complexity,
            content_length=content_analysis.length,
            budget_remaining=budget_context.remaining_percentage,
            content=content,
        )

        # Compute provider preferences for internal decision-making (not passed to LLM)
        _provider_preferences = self._get_provider_preferences_for_operation_mode(
            budget_context.operation_mode
        )

        # Build fallback chain
        fallback_models = self.fallback_chains[selected_tier].copy()

        # Calculate confidence score
        confidence_factors = []

        # Model availability factor
        if self.model_status[selected_tier].is_available:
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.1)

        # Budget appropriateness factor
        estimated_cost = self.get_cost_estimate(
            task_type,
            content_analysis.length,
            complexity,
            budget_context.remaining_percentage,
            content,
        )

        if budget_context.operation_mode == "normal" and estimated_cost < 0.05:
            confidence_factors.append(0.25)
        elif budget_context.operation_mode == "conservative" and estimated_cost < 0.02:
            confidence_factors.append(0.25)
        elif (
            budget_context.operation_mode in ["emergency", "critical"]
            and estimated_cost == 0
        ):
            confidence_factors.append(0.25)
        else:
            confidence_factors.append(0.1)

        # Task-model alignment factor
        optimal_tier = self.routing_strategy.get(task_type, "mini")
        if selected_tier == optimal_tier:
            confidence_factors.append(0.2)
        elif (
            abs(
                ["nano", "mini", "full"].index(selected_tier)
                - ["nano", "mini", "full"].index(optimal_tier)
            )
            == 1
        ):
            confidence_factors.append(0.15)
        else:
            confidence_factors.append(0.1)

        # Complexity alignment factor
        complexity_tier_map = {"minimal": "nano", "medium": "mini", "high": "full"}
        if selected_tier == complexity_tier_map.get(complexity, "mini"):
            confidence_factors.append(0.15)
        else:
            confidence_factors.append(0.08)

        # Priority factor
        if priority == "critical" and selected_tier == "full":
            confidence_factors.append(0.1)
        elif priority == "high" and selected_tier in ["mini", "full"]:
            confidence_factors.append(0.08)
        else:
            confidence_factors.append(0.05)

        confidence_score = sum(confidence_factors)

        # Build rationale
        rationale_parts = [
            f"Task: {task_type} (complexity: {complexity})",
            f"Content: {content_analysis.length} chars, {content_analysis.word_count} words",
            f"Domain: {content_analysis.domain}",
            f"Operation mode: {budget_context.operation_mode}",
            f"Selected tier: {selected_tier}",
            f"Estimated cost: ${estimated_cost:.6f}",
            f"Confidence: {confidence_score:.2f}",
        ]

        if content_analysis.urgency > 0.5:
            rationale_parts.append(
                f"High urgency detected ({content_analysis.urgency:.2f})"
            )

        if len(content_analysis.complexity_indicators) > 0:
            rationale_parts.append(
                f"Complexity indicators: {len(content_analysis.complexity_indicators)}"
            )

        return OpenRouterModelSelection(
            selected_model=selected_model.model,
            selected_tier=selected_tier,
            rationale=" | ".join(rationale_parts),
            estimated_cost=estimated_cost,
            confidence_score=confidence_score,
            # Note: provider preferences are used internally, not passed to LLM
            fallback_models=fallback_models,
            operation_mode=budget_context.operation_mode,
        )

    async def route_query_enhanced(
        self,
        task_type: TaskType,
        content: str,
        complexity: Optional[ComplexityLevel] = None,
        budget_context: Optional[BudgetContext] = None,
        priority: TaskPriority = "normal",
    ) -> RoutingResult:
        """
        Enhanced query routing with comprehensive result tracking.

        Args:
            task_type: Type of task to perform
            content: Content to process
            complexity: Optional complexity level
            budget_context: Budget context for routing
            priority: Task priority level

        Returns:
            RoutingResult with complete execution details
        """
        start_time = asyncio.get_event_loop().time()

        # Get optimal model selection
        model_selection = self.choose_optimal_model(
            task_type, content, complexity, budget_context, priority
        )

        # Execute with fallback handling
        try:
            response, execution_metadata = (
                await self.intelligent_fallback_with_recovery(
                    task_type=task_type,
                    content=content,
                    complexity=complexity,
                    budget_remaining=(
                        budget_context.remaining_percentage if budget_context else 100.0
                    ),
                    max_retries=3,
                )
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            # Calculate quality score based on response characteristics
            quality_score = self._calculate_quality_score(response, task_type, content)

            return RoutingResult(
                response=response,
                model_used=execution_metadata.get(
                    "final_tier", model_selection.selected_tier
                ),
                actual_model_name=execution_metadata.get(
                    "final_model", model_selection.selected_model
                ),
                actual_cost=execution_metadata.get(
                    "total_cost", model_selection.estimated_cost
                ),
                performance_metrics={
                    "execution_time": execution_time,
                    "attempts": len(execution_metadata.get("attempts", [])),
                    "recovery_actions": len(
                        execution_metadata.get("recovery_actions", [])
                    ),
                    "response_length": len(response),
                    "confidence_score": model_selection.confidence_score,
                },
                quality_score=quality_score,
                execution_time=execution_time,
                fallback_used=execution_metadata.get("fallback_used", False),
                routing_rationale=model_selection.rationale,
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Enhanced routing failed completely: {e}")

            # Return error result
            return RoutingResult(
                response=f"ERROR: Unable to process request - {str(e)}",
                model_used="nano",
                actual_model_name="error",
                actual_cost=0.0,
                performance_metrics={
                    "execution_time": execution_time,
                    "attempts": 0,
                    "recovery_actions": 0,
                    "response_length": 0,
                    "confidence_score": 0.0,
                },
                quality_score=0.0,
                execution_time=execution_time,
                fallback_used=True,
                routing_rationale=f"Complete failure: {str(e)}",
            )

    def _calculate_quality_score(
        self, response: str, task_type: TaskType, original_content: str
    ) -> float:
        """Calculate quality score for a response."""
        score_factors = []

        # Length appropriateness (0.2 weight)
        response_length = len(response)
        if task_type == "validation" and 50 <= response_length <= 200:
            score_factors.append(0.2)
        elif task_type == "simple" and 20 <= response_length <= 150:
            score_factors.append(0.2)
        elif task_type == "research" and 200 <= response_length <= 800:
            score_factors.append(0.2)
        elif task_type == "forecast" and 150 <= response_length <= 600:
            score_factors.append(0.2)
        else:
            score_factors.append(0.1)

        # Content quality indicators (0.3 weight)
        quality_indicators = [
            "evidence",
            "source",
            "analysis",
            "reasoning",
            "conclusion",
        ]
        found_indicators = sum(
            1 for indicator in quality_indicators if indicator in response.lower()
        )
        score_factors.append(min(found_indicators / len(quality_indicators), 1.0) * 0.3)

        # Structure and formatting (0.2 weight)
        has_structure = any(
            marker in response for marker in ["•", "-", "1.", "2.", "\n\n"]
        )
        score_factors.append(0.2 if has_structure else 0.1)

        # Task-specific quality (0.3 weight)
        if task_type == "forecast":
            forecast_indicators = [
                "probability",
                "confidence",
                "uncertainty",
                "scenario",
            ]
            forecast_score = sum(
                1 for indicator in forecast_indicators if indicator in response.lower()
            )
            score_factors.append(
                min(forecast_score / len(forecast_indicators), 1.0) * 0.3
            )
        elif task_type == "research":
            research_indicators = ["study", "research", "data", "findings", "report"]
            research_score = sum(
                1 for indicator in research_indicators if indicator in response.lower()
            )
            score_factors.append(
                min(research_score / len(research_indicators), 1.0) * 0.3
            )
        else:
            # General quality for validation/simple tasks
            score_factors.append(0.2)

        return min(sum(score_factors), 1.0)

    def integrate_with_budget_manager(self, budget_manager: Any, budget_aware_manager: Any) -> None:
        """Integrate tri-model router with budget management systems (Task 8.2)."""
        self.budget_manager = budget_manager
        self.budget_aware_manager = budget_aware_manager
        logger.info("Tri-model router integrated with budget management systems")

    def get_budget_aware_routing_context(self) -> Optional[BudgetContext]:
        """Get current budget context for routing decisions."""
        if not hasattr(self, "budget_manager") or not self.budget_manager:
            return None

        try:
            budget_status = self.budget_manager.get_budget_status()
            operation_mode: OperationMode = cast(OperationMode, "normal")

            if hasattr(self, "budget_aware_manager") and self.budget_aware_manager:
                operation_mode = cast(
                    OperationMode,
                    self.budget_aware_manager.operation_mode_manager.get_current_mode().value,
                )

            return BudgetContext(
                remaining_percentage=100.0 - budget_status.utilization_percentage,
                estimated_questions_remaining=budget_status.estimated_questions_remaining,
                current_burn_rate=budget_status.average_cost_per_question,
                operation_mode=operation_mode,
                budget_used_percentage=budget_status.utilization_percentage,
            )
        except Exception as e:
            logger.warning(f"Failed to get budget context: {e}")
            return None

    def apply_budget_aware_model_adjustments(
        self, base_selection: OpenRouterModelSelection
    ) -> OpenRouterModelSelection:
        """Apply budget-aware adjustments to model selection (Task 8.2)."""
        if not hasattr(self, "budget_aware_manager") or not self.budget_aware_manager:
            return base_selection

        try:
            # Get cost optimization strategy for current operation mode
            current_mode = (
                self.budget_aware_manager.operation_mode_manager.get_current_mode()
            )
            # Retrieve strategy (currently unused but call retained for side effects / logging)
            self.budget_aware_manager.get_cost_optimization_strategy(current_mode)

            # Apply model selection adjustments
            task_type_mapping = {
                "validation": "validation",
                "simple": "research",  # Map simple to research for strategy
                "research": "research",
                "forecast": "forecast",
            }

            # Determine task type from rationale or use default
            task_type = "research"  # Default
            for t in task_type_mapping.keys():
                if t in base_selection.rationale.lower():
                    task_type = task_type_mapping[t]
                    break

            # Get adjusted model from strategy
            adjusted_model = (
                self.budget_aware_manager.apply_model_selection_adjustments(
                    task_type, current_mode
                )
            )

            # Update selection if adjustment is needed
            if adjusted_model != base_selection.selected_model:
                # Map adjusted model to tier
                adjusted_tier = base_selection.selected_tier
                if "gpt-5-nano" in adjusted_model or "nano" in adjusted_model:
                    adjusted_tier = "nano"
                elif "gpt-5-mini" in adjusted_model or "mini" in adjusted_model:
                    adjusted_tier = "mini"
                elif (
                    "gpt-5" in adjusted_model
                    and "mini" not in adjusted_model
                    and "nano" not in adjusted_model
                ):
                    adjusted_tier = "full"

                # Create adjusted selection
                adjusted_selection = OpenRouterModelSelection(
                    selected_model=adjusted_model,
                    selected_tier=adjusted_tier,
                    rationale=f"Budget-aware adjustment: {base_selection.rationale} (mode: {current_mode.value})",
                    estimated_cost=self._estimate_cost_for_model(
                        adjusted_model, 1000, 500
                    ),  # Rough estimate
                    confidence_score=base_selection.confidence_score
                    * 0.9,  # Slightly lower confidence for adjustments
                    fallback_models=base_selection.fallback_models,
                    operation_mode=current_mode.value,
                )

                logger.info(
                    f"Budget-aware model adjustment: {base_selection.selected_model} → {adjusted_model} (mode: {current_mode.value})"
                )
                return adjusted_selection

            return base_selection

        except Exception as e:
            logger.warning(f"Failed to apply budget-aware adjustments: {e}")
            return base_selection

    def _estimate_cost_for_model(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost for a specific model."""
        # Use model configs if available
        for tier, config in self.model_configs.items():
            if config.model_name == model_name:
                return (input_tokens * config.cost_per_million_input / 1_000_000) + (
                    output_tokens * config.cost_per_million_output / 1_000_000
                )

        # Default estimation for unknown models
        return 0.001  # $0.001 default


# Global instance with backward compatibility
tri_model_router = OpenRouterTriModelRouter()

# Backward compatibility aliases
TriModelRouter = OpenRouterTriModelRouter
EnhancedTriModelRouter = OpenRouterTriModelRouter
