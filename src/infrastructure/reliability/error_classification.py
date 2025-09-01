"""
Comprehensive error classification and handling system for OpenRouter tri-model optimization.
Implements model-specific error detection, budget exhaustion handling, API failure recovery,
and quality validation failure recovery with intelligent prompt revision.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors in the forecasting system."""

    MODEL_ERROR = "model_error"
    BUDGET_ERROR = "budget_error"
    API_ERROR = "api_error"
    QUALITY_ERROR = "quality_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "auth_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    CONFIGURATION_ERROR = "config_error"
    VALIDATION_ERROR = "validation_error"


class ErrorSeverity(Enum):
    """Severity levels for error classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies for different error types."""

    RETRY = "retry"
    FALLBACK_MODEL = "fallback_model"
    FALLBACK_PROVIDER = "fallback_provider"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_MODE = "emergency_mode"
    PROMPT_REVISION = "prompt_revision"
    BUDGET_CONSERVATION = "budget_conservation"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for error analysis and recovery."""

    task_type: str
    model_tier: str
    operation_mode: str
    budget_remaining: float
    attempt_number: int
    original_prompt: Optional[str] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ErrorClassification:
    """Complete error classification with recovery recommendations."""

    category: ErrorCategory
    severity: ErrorSeverity
    error_code: str
    description: str
    recovery_strategies: List[RecoveryStrategy]
    retry_delay: float
    max_retries: int
    fallback_options: List[str]
    context_requirements: Dict[str, Any]


@dataclass
class RecoveryAction:
    """Specific recovery action to be taken."""

    strategy: RecoveryStrategy
    parameters: Dict[str, Any]
    expected_delay: float
    success_probability: float
    fallback_action: Optional["RecoveryAction"] = None


class ForecastingError(Exception):
    """Base exception for forecasting system errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context
        self.original_error = original_error
        self.timestamp = datetime.utcnow()


class ModelError(ForecastingError):
    """Errors related to model execution and responses."""

    def __init__(
        self,
        message: str,
        model_name: str,
        model_tier: str,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            ErrorCategory.MODEL_ERROR,
            ErrorSeverity.HIGH,
            context,
            original_error,
        )
        self.model_name = model_name
        self.model_tier = model_tier


class BudgetError(ForecastingError):
    """Errors related to budget exhaustion and cost management."""

    def __init__(
        self,
        message: str,
        budget_remaining: float,
        estimated_cost: float,
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(
            message, ErrorCategory.BUDGET_ERROR, ErrorSeverity.CRITICAL, context
        )
        self.budget_remaining = budget_remaining
        self.estimated_cost = estimated_cost


class APIError(ForecastingError):
    """Errors related to API calls and external service failures."""

    def __init__(
        self,
        message: str,
        api_provider: str,
        status_code: Optional[int] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            ErrorCategory.API_ERROR,
            ErrorSeverity.HIGH,
            context,
            original_error,
        )
        self.api_provider = api_provider
        self.status_code = status_code


class QualityError(ForecastingError):
    """Errors related to quality validation failures."""

    def __init__(
        self,
        message: str,
        quality_issues: List[str],
        quality_score: float,
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(
            message, ErrorCategory.QUALITY_ERROR, ErrorSeverity.MEDIUM, context
        )
        self.quality_issues = quality_issues
        self.quality_score = quality_score


class ErrorClassifier:
    """
    Comprehensive error classification system with model-specific detection.
    """

    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.error_history = []
        self.pattern_cache = {}

    def _initialize_error_patterns(self) -> Dict[str, ErrorClassification]:
        """Initialize error pattern recognition and classification rules."""
        return {
            # Model-specific errors
            "openai_rate_limit": ErrorClassification(
                category=ErrorCategory.RATE_LIMIT_ERROR,
                severity=ErrorSeverity.HIGH,
                error_code="OPENAI_RATE_LIMIT",
                description="OpenRouter/OpenAI rate limit exceeded",
                recovery_strategies=[
                    RecoveryStrategy.RETRY,
                    RecoveryStrategy.FALLBACK_MODEL,
                ],
                retry_delay=60.0,
                max_retries=3,
                fallback_options=["free_models", "metaculus_proxy"],
                context_requirements={"exponential_backoff": True},
            ),
            "model_unavailable": ErrorClassification(
                category=ErrorCategory.MODEL_ERROR,
                severity=ErrorSeverity.HIGH,
                error_code="MODEL_UNAVAILABLE",
                description="Requested model is temporarily unavailable",
                recovery_strategies=[
                    RecoveryStrategy.FALLBACK_MODEL,
                    RecoveryStrategy.FALLBACK_PROVIDER,
                ],
                retry_delay=30.0,
                max_retries=2,
                fallback_options=["tier_downgrade", "free_models"],
                context_requirements={"preserve_task_quality": True},
            ),
            "context_length_exceeded": ErrorClassification(
                category=ErrorCategory.MODEL_ERROR,
                severity=ErrorSeverity.MEDIUM,
                error_code="CONTEXT_TOO_LONG",
                description="Input context exceeds model's maximum length",
                recovery_strategies=[
                    RecoveryStrategy.PROMPT_REVISION,
                    RecoveryStrategy.FALLBACK_MODEL,
                ],
                retry_delay=5.0,
                max_retries=2,
                fallback_options=["context_compression", "tier_upgrade"],
                context_requirements={"preserve_key_information": True},
            ),
            # Budget-related errors
            "budget_exhausted": ErrorClassification(
                category=ErrorCategory.BUDGET_ERROR,
                severity=ErrorSeverity.CRITICAL,
                error_code="BUDGET_EXHAUSTED",
                description="Budget limit reached or exceeded",
                recovery_strategies=[
                    RecoveryStrategy.EMERGENCY_MODE,
                    RecoveryStrategy.GRACEFUL_DEGRADATION,
                ],
                retry_delay=0.0,
                max_retries=0,
                fallback_options=["free_models_only", "essential_functions"],
                context_requirements={"preserve_core_functionality": True},
            ),
            "budget_threshold_warning": ErrorClassification(
                category=ErrorCategory.BUDGET_ERROR,
                severity=ErrorSeverity.MEDIUM,
                error_code="BUDGET_WARNING",
                description="Budget utilization approaching critical threshold",
                recovery_strategies=[
                    RecoveryStrategy.BUDGET_CONSERVATION,
                    RecoveryStrategy.FALLBACK_MODEL,
                ],
                retry_delay=0.0,
                max_retries=1,
                fallback_options=["cheaper_models", "reduced_complexity"],
                context_requirements={"cost_optimization": True},
            ),
            # API and network errors
            "network_timeout": ErrorClassification(
                category=ErrorCategory.TIMEOUT_ERROR,
                severity=ErrorSeverity.HIGH,
                error_code="NETWORK_TIMEOUT",
                description="Network request timed out",
                recovery_strategies=[
                    RecoveryStrategy.RETRY,
                    RecoveryStrategy.FALLBACK_PROVIDER,
                ],
                retry_delay=10.0,
                max_retries=3,
                fallback_options=["alternative_provider", "reduced_timeout"],
                context_requirements={"exponential_backoff": True},
            ),
            "api_authentication_failed": ErrorClassification(
                category=ErrorCategory.AUTHENTICATION_ERROR,
                severity=ErrorSeverity.CRITICAL,
                error_code="AUTH_FAILED",
                description="API authentication failed",
                recovery_strategies=[
                    RecoveryStrategy.FALLBACK_PROVIDER,
                    RecoveryStrategy.EMERGENCY_MODE,
                ],
                retry_delay=0.0,
                max_retries=1,
                fallback_options=["alternative_api", "free_models"],
                context_requirements={"verify_credentials": True},
            ),
            "api_server_error": ErrorClassification(
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.HIGH,
                error_code="API_SERVER_ERROR",
                description="API server returned 5xx error",
                recovery_strategies=[
                    RecoveryStrategy.FALLBACK_PROVIDER,
                    RecoveryStrategy.RETRY,
                ],
                retry_delay=30.0,
                max_retries=3,
                fallback_options=["alternative_provider", "cached_response"],
                context_requirements={"exponential_backoff": True},
            ),
            # Quality validation errors
            "quality_validation_failed": ErrorClassification(
                category=ErrorCategory.QUALITY_ERROR,
                severity=ErrorSeverity.MEDIUM,
                error_code="QUALITY_FAILED",
                description="Response failed quality validation checks",
                recovery_strategies=[
                    RecoveryStrategy.PROMPT_REVISION,
                    RecoveryStrategy.FALLBACK_MODEL,
                ],
                retry_delay=5.0,
                max_retries=2,
                fallback_options=["enhanced_prompts", "tier_upgrade"],
                context_requirements={"improve_quality_directives": True},
            ),
            "citation_missing": ErrorClassification(
                category=ErrorCategory.QUALITY_ERROR,
                severity=ErrorSeverity.MEDIUM,
                error_code="MISSING_CITATIONS",
                description="Response lacks required source citations",
                recovery_strategies=[
                    RecoveryStrategy.PROMPT_REVISION,
                    RecoveryStrategy.RETRY,
                ],
                retry_delay=3.0,
                max_retries=2,
                fallback_options=["citation_enforcement", "research_enhancement"],
                context_requirements={"enforce_citations": True},
            ),
            "hallucination_detected": ErrorClassification(
                category=ErrorCategory.QUALITY_ERROR,
                severity=ErrorSeverity.HIGH,
                error_code="HALLUCINATION",
                description="Potential hallucination detected in response",
                recovery_strategies=[
                    RecoveryStrategy.PROMPT_REVISION,
                    RecoveryStrategy.FALLBACK_MODEL,
                ],
                retry_delay=10.0,
                max_retries=2,
                fallback_options=["fact_checking", "conservative_prompts"],
                context_requirements={"enhance_fact_checking": True},
            ),
        }

    def _initialize_recovery_strategies(self) -> Dict[RecoveryStrategy, Dict[str, Any]]:
        """Initialize recovery strategy configurations."""
        return {
            RecoveryStrategy.RETRY: {
                "base_delay": 1.0,
                "max_delay": 300.0,  # Increased to allow exponential backoff
                "exponential_base": 2.0,
                "jitter": True,
            },
            RecoveryStrategy.FALLBACK_MODEL: {
                "preserve_quality": True,
                "cost_awareness": True,
                "tier_downgrade_order": ["full", "mini", "nano", "free"],
            },
            RecoveryStrategy.FALLBACK_PROVIDER: {
                "provider_order": ["openrouter", "metaculus_proxy", "free_models"],
                "preserve_functionality": True,
            },
            RecoveryStrategy.GRACEFUL_DEGRADATION: {
                "essential_functions": ["basic_forecasting", "simple_research"],
                "disable_features": ["advanced_analysis", "detailed_reasoning"],
            },
            RecoveryStrategy.EMERGENCY_MODE: {
                "free_models_only": True,
                "minimal_functionality": True,
                "cost_limit": 0.0,
            },
            RecoveryStrategy.PROMPT_REVISION: {
                "simplification_strategies": [
                    "reduce_complexity",
                    "shorter_prompts",
                    "clearer_instructions",
                ],
                "quality_enhancement": [
                    "add_citations",
                    "fact_checking",
                    "uncertainty_acknowledgment",
                ],
            },
            RecoveryStrategy.BUDGET_CONSERVATION: {
                "cost_reduction_methods": [
                    "cheaper_models",
                    "shorter_responses",
                    "essential_only",
                ],
                "threshold_adjustments": {"emergency": 85, "critical": 95},
            },
        }

    def classify_error(
        self, error: Exception, context: ErrorContext
    ) -> ErrorClassification:
        """
        Classify an error and determine appropriate recovery strategies.

        Args:
            error: The exception that occurred
            context: Context information about the error

        Returns:
            ErrorClassification with recovery recommendations
        """
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check for specific error patterns
        classification = self._match_error_pattern(
            error_message, error_type, context, error
        )

        if classification:
            # Log the classification
            logger.info(
                f"Error classified: {classification.error_code} - {classification.description}"
            )
            self._record_error_occurrence(classification, context)
            return classification

        # Default classification for unknown errors
        default_classification = self._create_default_classification(error, context)
        self._record_error_occurrence(default_classification, context)
        return default_classification

    def _match_error_pattern(
        self,
        error_message: str,
        error_type: str,
        context: ErrorContext,
        error: Exception = None,
    ) -> Optional[ErrorClassification]:
        """Match error against known patterns."""

        # Check for specific error types first
        if error:
            if isinstance(error, ModelError):
                return self.error_patterns["model_unavailable"]
            elif isinstance(error, BudgetError):
                # Use the error's budget_remaining value, not the context's
                if error.budget_remaining <= 0:
                    return self.error_patterns["budget_exhausted"]
                # Treat critically low budgets (<5%) as emergency conditions
                elif error.budget_remaining < 5:
                    return self.error_patterns["budget_exhausted"]
                else:
                    return self.error_patterns["budget_threshold_warning"]
            elif isinstance(error, APIError):
                return self.error_patterns["api_server_error"]
            elif isinstance(error, QualityError):
                return self.error_patterns["quality_validation_failed"]

        # Rate limit patterns
        if any(
            pattern in error_message
            for pattern in ["rate limit", "too many requests", "quota exceeded"]
        ):
            return self.error_patterns["openai_rate_limit"]

        # Model availability patterns
        if any(
            pattern in error_message
            for pattern in ["model not found", "unavailable", "not supported"]
        ):
            return self.error_patterns["model_unavailable"]

        # Context length patterns
        if any(
            pattern in error_message
            for pattern in ["context length", "too long", "maximum length"]
        ):
            return self.error_patterns["context_length_exceeded"]

        # Budget patterns
        if context.budget_remaining <= 0:
            return self.error_patterns["budget_exhausted"]
        # Critically low budget should trigger emergency-mode path
        elif context.budget_remaining < 5:  # Less than 5% remaining
            return self.error_patterns["budget_exhausted"]
        elif context.budget_remaining < 15:  # Less than 15% remaining
            return self.error_patterns["budget_threshold_warning"]

        # Network and timeout patterns
        if any(
            pattern in error_message for pattern in ["timeout", "connection", "network"]
        ):
            return self.error_patterns["network_timeout"]

        # Authentication patterns
        if any(
            pattern in error_message
            for pattern in ["unauthorized", "authentication", "api key"]
        ):
            return self.error_patterns["api_authentication_failed"]

        # Server error patterns
        if any(
            pattern in error_message
            for pattern in ["server error", "internal error", "5"]
        ):
            return self.error_patterns["api_server_error"]

        # Quality validation patterns
        if "quality" in error_message or "validation" in error_message:
            return self.error_patterns["quality_validation_failed"]

        if "citation" in error_message:
            return self.error_patterns["citation_missing"]

        if "hallucination" in error_message or "factual" in error_message:
            return self.error_patterns["hallucination_detected"]

        return None

    def _create_default_classification(
        self, error: Exception, context: ErrorContext
    ) -> ErrorClassification:
        """Create default classification for unknown errors."""
        return ErrorClassification(
            category=ErrorCategory.MODEL_ERROR,
            severity=ErrorSeverity.MEDIUM,
            error_code="UNKNOWN_ERROR",
            description=f"Unknown error: {type(error).__name__}",
            recovery_strategies=[
                RecoveryStrategy.RETRY,
                RecoveryStrategy.FALLBACK_MODEL,
            ],
            retry_delay=10.0,
            max_retries=2,
            fallback_options=["alternative_approach"],
            context_requirements={"investigate_cause": True},
        )

    def _record_error_occurrence(
        self, classification: ErrorClassification, context: ErrorContext
    ):
        """Record error occurrence for pattern analysis."""
        self.error_history.append(
            {
                "timestamp": datetime.utcnow(),
                "classification": classification,
                "context": context,
                "error_code": classification.error_code,
            }
        )

        # Keep only recent history (last 1000 errors)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

    def get_error_statistics(
        self, time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Get error statistics for the specified time window."""
        cutoff_time = datetime.utcnow() - time_window
        recent_errors = [e for e in self.error_history if e["timestamp"] > cutoff_time]

        if not recent_errors:
            return {"total_errors": 0, "error_categories": {}, "most_common": []}

        # Count by category
        category_counts = {}
        error_code_counts = {}

        for error in recent_errors:
            category = error["classification"].category.value
            error_code = error["error_code"]

            category_counts[category] = category_counts.get(category, 0) + 1
            error_code_counts[error_code] = error_code_counts.get(error_code, 0) + 1

        # Sort by frequency
        most_common = sorted(
            error_code_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "total_errors": len(recent_errors),
            "error_categories": category_counts,
            "most_common": most_common,
            "time_window_hours": time_window.total_seconds() / 3600,
        }

    def should_retry(
        self, classification: ErrorClassification, attempt_number: int
    ) -> bool:
        """Determine if an error should trigger a retry."""
        return (
            RecoveryStrategy.RETRY in classification.recovery_strategies
            and attempt_number < classification.max_retries
        )

    def calculate_retry_delay(
        self, classification: ErrorClassification, attempt_number: int
    ) -> float:
        """Calculate appropriate delay before retry."""
        base_delay = classification.retry_delay

        if classification.context_requirements.get("exponential_backoff", False):
            # For first attempt, return base delay
            if attempt_number == 1:
                return base_delay

            # Exponential backoff with jitter for subsequent attempts
            delay = base_delay * (2 ** (attempt_number - 1))
            if self.recovery_strategies[RecoveryStrategy.RETRY].get("jitter", False):
                import random

                delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

            max_delay = self.recovery_strategies[RecoveryStrategy.RETRY]["max_delay"]
            return min(delay, max_delay)

        return base_delay


class ErrorRecoveryManager:
    """
    Comprehensive error recovery manager with intelligent fallback strategies.
    """

    def __init__(self, tri_model_router=None, budget_manager=None):
        self.classifier = ErrorClassifier()
        self.tri_model_router = tri_model_router
        self.budget_manager = budget_manager
        self.recovery_history = []
        self.circuit_breakers = {}

    async def handle_error(
        self, error: Exception, context: ErrorContext
    ) -> RecoveryAction:
        """
        Handle an error with appropriate recovery strategy.

        Args:
            error: The exception that occurred
            context: Context information about the error

        Returns:
            RecoveryAction to be taken
        """
        # Classify the error
        classification = self.classifier.classify_error(error, context)

        # Check circuit breakers
        if self._is_circuit_breaker_open(classification.error_code):
            logger.warning(
                f"Circuit breaker open for {classification.error_code}, using emergency fallback"
            )
            return self._create_emergency_recovery_action(classification, context)

        # Determine best recovery strategy
        recovery_action = await self._determine_recovery_action(classification, context)

        # Record recovery attempt
        self._record_recovery_attempt(classification, context, recovery_action)

        # Update circuit breaker state
        self._update_circuit_breaker(classification.error_code, recovery_action)

        return recovery_action

    async def _determine_recovery_action(
        self, classification: ErrorClassification, context: ErrorContext
    ) -> RecoveryAction:
        """Determine the best recovery action based on classification and context."""

        # Priority order for recovery strategies
        for strategy in classification.recovery_strategies:

            if strategy == RecoveryStrategy.RETRY:
                if self.classifier.should_retry(classification, context.attempt_number):
                    delay = self.classifier.calculate_retry_delay(
                        classification, context.attempt_number
                    )
                    return RecoveryAction(
                        strategy=RecoveryStrategy.RETRY,
                        parameters={
                            "delay": delay,
                            "attempt": context.attempt_number + 1,
                        },
                        expected_delay=delay,
                        success_probability=0.7 - (context.attempt_number * 0.2),
                    )

            elif strategy == RecoveryStrategy.FALLBACK_MODEL:
                fallback_model = await self._get_fallback_model(context)
                if fallback_model:
                    return RecoveryAction(
                        strategy=RecoveryStrategy.FALLBACK_MODEL,
                        parameters={
                            "fallback_model": fallback_model,
                            "preserve_quality": True,
                        },
                        expected_delay=5.0,
                        success_probability=0.8,
                    )

            elif strategy == RecoveryStrategy.FALLBACK_PROVIDER:
                fallback_provider = await self._get_fallback_provider(context)
                if fallback_provider:
                    return RecoveryAction(
                        strategy=RecoveryStrategy.FALLBACK_PROVIDER,
                        parameters={"fallback_provider": fallback_provider},
                        expected_delay=10.0,
                        success_probability=0.6,
                    )

            elif strategy == RecoveryStrategy.PROMPT_REVISION:
                revised_prompt = await self._revise_prompt(context, classification)
                if revised_prompt:
                    return RecoveryAction(
                        strategy=RecoveryStrategy.PROMPT_REVISION,
                        parameters={
                            "revised_prompt": revised_prompt,
                            "revision_type": "quality_enhancement",
                        },
                        expected_delay=2.0,
                        success_probability=0.75,
                    )

            elif strategy == RecoveryStrategy.EMERGENCY_MODE:
                return self._create_emergency_recovery_action(classification, context)

            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return RecoveryAction(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    parameters={"reduced_functionality": True, "essential_only": True},
                    expected_delay=1.0,
                    success_probability=0.9,
                )

        # Default fallback
        return self._create_emergency_recovery_action(classification, context)

    async def _get_fallback_model(self, context: ErrorContext) -> Optional[str]:
        """Get appropriate fallback model based on context."""
        if not self.tri_model_router:
            return None

        current_tier = context.model_tier

        # Define fallback hierarchy
        fallback_hierarchy = {
            "full": ["mini", "nano", "free"],
            "mini": ["nano", "free"],
            "nano": ["free"],
        }

        fallback_tiers = fallback_hierarchy.get(current_tier, ["free"])

        for tier in fallback_tiers:
            if tier == "free":
                # Use free models
                free_models = ["openai/gpt-oss-20b:free", "moonshotai/kimi-k2:free"]
                for model in free_models:
                    if await self._is_model_available(model):
                        return model
            else:
                # Check if tier model is available
                if tier in self.tri_model_router.models:
                    model_status = await self.tri_model_router.check_model_health(tier)
                    if model_status.is_available:
                        return self.tri_model_router.model_configs[tier].model_name

        return None

    async def _get_fallback_provider(self, context: ErrorContext) -> Optional[str]:
        """Get appropriate fallback provider."""
        current_provider = context.provider or "openrouter"

        provider_fallbacks = {
            "openrouter": ["metaculus_proxy", "free_models"],
            "metaculus_proxy": ["free_models"],
            "free_models": [],
        }

        fallbacks = provider_fallbacks.get(current_provider, [])

        for provider in fallbacks:
            if await self._is_provider_available(provider):
                return provider

        return None

    async def _revise_prompt(
        self, context: ErrorContext, classification: ErrorClassification
    ) -> Optional[str]:
        """Revise prompt based on error classification."""
        if not context.original_prompt:
            return None

        original_prompt = context.original_prompt

        # Apply revision strategies based on error type
        if classification.error_code == "CONTEXT_TOO_LONG":
            # Compress the prompt
            return self._compress_prompt(original_prompt)

        elif classification.error_code == "MISSING_CITATIONS":
            # Add citation requirements
            return self._add_citation_requirements(original_prompt)

        elif classification.error_code == "QUALITY_FAILED":
            # Enhance quality directives
            return self._enhance_quality_directives(original_prompt)

        elif classification.error_code == "HALLUCINATION":
            # Add fact-checking directives
            return self._add_fact_checking_directives(original_prompt)

        return None

    def _compress_prompt(self, prompt: str) -> str:
        """Compress prompt to reduce context length."""
        # Simple compression strategies
        lines = prompt.split("\n")

        # Remove empty lines and excessive whitespace
        compressed_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):  # Keep non-empty, non-header lines
                compressed_lines.append(line)

        # Limit to essential content (first 50% of lines)
        if len(compressed_lines) > 20:
            compressed_lines = compressed_lines[: len(compressed_lines) // 2]
            compressed_lines.append(
                "... [content compressed for context length limits]"
            )

        return "\n".join(compressed_lines)

    def _add_citation_requirements(self, prompt: str) -> str:
        """Add stronger citation requirements to prompt."""
        citation_directive = """
CRITICAL: Every factual claim MUST include a specific source citation in [Source: URL/Reference] format.
Do not make any factual statements without proper attribution.
If you cannot find a source for a claim, explicitly state "No source available" rather than omitting the citation.
"""
        return citation_directive + "\n\n" + prompt

    def _enhance_quality_directives(self, prompt: str) -> str:
        """Enhance quality directives in prompt."""
        quality_directive = """
QUALITY REQUIREMENTS:
- Provide evidence for all claims
- Acknowledge uncertainty where appropriate
- Use precise, factual language
- Avoid speculation without clear labeling
- Structure responses clearly with bullet points
"""
        return quality_directive + "\n\n" + prompt

    def _add_fact_checking_directives(self, prompt: str) -> str:
        """Add fact-checking directives to prevent hallucinations."""
        fact_check_directive = """
FACT-CHECKING PROTOCOL:
- Only state facts you can verify
- When uncertain, explicitly say "I'm not certain about..."
- Distinguish between facts and analysis
- Provide confidence levels for uncertain information
- Cross-reference multiple sources when possible
"""
        return fact_check_directive + "\n\n" + prompt

    def _create_emergency_recovery_action(
        self, classification: ErrorClassification, context: ErrorContext
    ) -> RecoveryAction:
        """Create emergency recovery action as last resort."""
        return RecoveryAction(
            strategy=RecoveryStrategy.EMERGENCY_MODE,
            parameters={
                "free_models_only": True,
                "minimal_functionality": True,
                "error_context": context,
                "original_error": classification.error_code,
            },
            expected_delay=1.0,
            success_probability=0.5,
        )

    async def _is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        try:
            if self.tri_model_router:
                # Use router's availability detection
                availability = await self.tri_model_router.detect_model_availability()
                return availability.get(model_name, False)
            return False
        except Exception:
            return False

    async def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available."""
        # Simple availability check based on configuration
        if provider == "metaculus_proxy":
            return os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true"
        elif provider == "free_models":
            return True  # Free models should always be available
        elif provider == "openrouter":
            return bool(os.getenv("OPENROUTER_API_KEY"))
        return False

    def _is_circuit_breaker_open(self, error_code: str) -> bool:
        """Check if circuit breaker is open for specific error type."""
        breaker = self.circuit_breakers.get(error_code)
        if not breaker:
            return False

        # Circuit breaker logic: open if too many failures in recent time
        current_time = time.time()
        failure_window = 300  # 5 minutes
        failure_threshold = 5

        recent_failures = [
            f for f in breaker.get("failures", []) if current_time - f < failure_window
        ]

        return len(recent_failures) >= failure_threshold

    def _update_circuit_breaker(self, error_code: str, recovery_action: RecoveryAction):
        """Update circuit breaker state based on recovery action."""
        if error_code not in self.circuit_breakers:
            self.circuit_breakers[error_code] = {"failures": [], "successes": []}

        current_time = time.time()

        # For now, assume all recovery actions are attempts (success/failure determined later)
        # This would be updated based on actual recovery results
        self.circuit_breakers[error_code]["failures"].append(current_time)

        # Clean old entries (keep only last hour)
        hour_ago = current_time - 3600
        self.circuit_breakers[error_code]["failures"] = [
            f for f in self.circuit_breakers[error_code]["failures"] if f > hour_ago
        ]

    def _record_recovery_attempt(
        self,
        classification: ErrorClassification,
        context: ErrorContext,
        recovery_action: RecoveryAction,
    ):
        """Record recovery attempt for analysis."""
        self.recovery_history.append(
            {
                "timestamp": datetime.utcnow(),
                "error_code": classification.error_code,
                "recovery_strategy": recovery_action.strategy.value,
                "context": context,
                "success_probability": recovery_action.success_probability,
            }
        )

        # Keep only recent history
        if len(self.recovery_history) > 1000:
            self.recovery_history = self.recovery_history[-1000:]

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics and effectiveness metrics."""
        if not self.recovery_history:
            return {"total_recoveries": 0}

        strategy_counts = {}
        error_counts = {}

        for recovery in self.recovery_history:
            strategy = recovery["recovery_strategy"]
            error_code = recovery["error_code"]

            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            error_counts[error_code] = error_counts.get(error_code, 0) + 1

        return {
            "total_recoveries": len(self.recovery_history),
            "strategy_usage": strategy_counts,
            "error_frequency": error_counts,
            "circuit_breaker_states": {
                code: len(breaker.get("failures", []))
                for code, breaker in self.circuit_breakers.items()
            },
        }


# Import os for environment variable access
import os
