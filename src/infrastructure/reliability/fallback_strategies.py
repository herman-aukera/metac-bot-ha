"""
Intelligent fallback strategies for OpenRouter tri-model optimization.
Implements model tier fallback with performance preservation, cross-provider API fallback,
emergency mode activation, and comprehensive error logging and alerting.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .error_classification import (
    APIError,
    BudgetError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    ForecastingError,
    ModelError,
    RecoveryAction,
    RecoveryStrategy,
)

logger = logging.getLogger(__name__)


class FallbackTier(Enum):
    """Tiers of fallback strategies."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    EMERGENCY = "emergency"
    CRITICAL = "critical"


class PerformanceLevel(Enum):
    """Performance levels for fallback strategies."""

    OPTIMAL = "optimal"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MINIMAL = "minimal"


@dataclass
class FallbackOption:
    """Configuration for a fallback option."""

    name: str
    tier: FallbackTier
    performance_level: PerformanceLevel
    cost_per_million: float
    availability_check: str
    configuration: Dict[str, Any]
    success_rate: float = 0.95
    average_response_time: float = 30.0
    last_used: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0


@dataclass
class FallbackResult:
    """Result of a fallback operation."""

    success: bool
    fallback_used: FallbackOption
    original_error: Optional[Exception]
    recovery_time: float
    performance_impact: float
    cost_impact: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Configuration for error alerts."""

    error_threshold: int = 5
    time_window_minutes: int = 15
    severity_levels: List[ErrorSeverity] = field(
        default_factory=lambda: [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
    )
    notification_channels: List[str] = field(default_factory=lambda: ["log", "file"])
    cooldown_minutes: int = 60


class ModelTierFallbackManager:
    """
    Manages model tier fallbacks with performance preservation.
    """

    def __init__(self, tri_model_router=None):
        self.tri_model_router = tri_model_router
        self.fallback_chains = self._initialize_fallback_chains()
        self.performance_tracking = {}
        self.fallback_history = []

    def _initialize_fallback_chains(self) -> Dict[str, List[FallbackOption]]:
        """Initialize model tier fallback chains with cost optimization."""
        return {
            "full": [
                FallbackOption(
                    name="openai/gpt-5",
                    tier=FallbackTier.PRIMARY,
                    performance_level=PerformanceLevel.OPTIMAL,
                    cost_per_million=1.50,
                    availability_check="openrouter_api",
                    configuration={"temperature": 0.0, "timeout": 90},
                ),
                FallbackOption(
                    name="openai/gpt-5-mini",
                    tier=FallbackTier.SECONDARY,
                    performance_level=PerformanceLevel.GOOD,
                    cost_per_million=0.25,
                    availability_check="openrouter_api",
                    configuration={"temperature": 0.1, "timeout": 60},
                ),
                FallbackOption(
                    name="moonshotai/kimi-k2:free",
                    tier=FallbackTier.EMERGENCY,
                    performance_level=PerformanceLevel.ACCEPTABLE,
                    cost_per_million=0.0,
                    availability_check="openrouter_free",
                    configuration={"temperature": 0.2, "timeout": 45},
                ),
                FallbackOption(
                    name="openai/gpt-oss-20b:free",
                    tier=FallbackTier.CRITICAL,
                    performance_level=PerformanceLevel.MINIMAL,
                    cost_per_million=0.0,
                    availability_check="openrouter_free",
                    configuration={"temperature": 0.3, "timeout": 30},
                ),
                FallbackOption(
                    name="metaculus/gpt-4o",
                    tier=FallbackTier.CRITICAL,
                    performance_level=PerformanceLevel.ACCEPTABLE,
                    cost_per_million=0.0,  # Uses proxy credits
                    availability_check="metaculus_proxy",
                    configuration={"temperature": 0.0, "timeout": 60},
                ),
            ],
            "mini": [
                FallbackOption(
                    name="openai/gpt-5-mini",
                    tier=FallbackTier.PRIMARY,
                    performance_level=PerformanceLevel.OPTIMAL,
                    cost_per_million=0.25,
                    availability_check="openrouter_api",
                    configuration={"temperature": 0.3, "timeout": 60},
                ),
                FallbackOption(
                    name="openai/gpt-5-nano",
                    tier=FallbackTier.SECONDARY,
                    performance_level=PerformanceLevel.GOOD,
                    cost_per_million=0.05,
                    availability_check="openrouter_api",
                    configuration={"temperature": 0.1, "timeout": 30},
                ),
                FallbackOption(
                    name="moonshotai/kimi-k2:free",
                    tier=FallbackTier.EMERGENCY,
                    performance_level=PerformanceLevel.ACCEPTABLE,
                    cost_per_million=0.0,
                    availability_check="openrouter_free",
                    configuration={"temperature": 0.2, "timeout": 45},
                ),
                FallbackOption(
                    name="openai/gpt-oss-20b:free",
                    tier=FallbackTier.CRITICAL,
                    performance_level=PerformanceLevel.MINIMAL,
                    cost_per_million=0.0,
                    availability_check="openrouter_free",
                    configuration={"temperature": 0.3, "timeout": 30},
                ),
                FallbackOption(
                    name="metaculus/gpt-4o-mini",
                    tier=FallbackTier.CRITICAL,
                    performance_level=PerformanceLevel.ACCEPTABLE,
                    cost_per_million=0.0,  # Uses proxy credits
                    availability_check="metaculus_proxy",
                    configuration={"temperature": 0.1, "timeout": 45},
                ),
            ],
            "nano": [
                FallbackOption(
                    name="openai/gpt-5-nano",
                    tier=FallbackTier.PRIMARY,
                    performance_level=PerformanceLevel.OPTIMAL,
                    cost_per_million=0.05,
                    availability_check="openrouter_api",
                    configuration={"temperature": 0.1, "timeout": 30},
                ),
                FallbackOption(
                    name="openai/gpt-oss-20b:free",
                    tier=FallbackTier.SECONDARY,
                    performance_level=PerformanceLevel.GOOD,
                    cost_per_million=0.0,
                    availability_check="openrouter_free",
                    configuration={"temperature": 0.2, "timeout": 30},
                ),
                FallbackOption(
                    name="moonshotai/kimi-k2:free",
                    tier=FallbackTier.EMERGENCY,
                    performance_level=PerformanceLevel.ACCEPTABLE,
                    cost_per_million=0.0,
                    availability_check="openrouter_free",
                    configuration={"temperature": 0.3, "timeout": 45},
                ),
                FallbackOption(
                    name="metaculus/gpt-4o-mini",
                    tier=FallbackTier.CRITICAL,
                    performance_level=PerformanceLevel.MINIMAL,
                    cost_per_million=0.0,  # Uses proxy credits
                    availability_check="metaculus_proxy",
                    configuration={"temperature": 0.1, "timeout": 45},
                ),
            ],
        }

    async def execute_fallback(
        self, original_tier: str, context: ErrorContext, budget_remaining: float
    ) -> FallbackResult:
        """
        Execute model tier fallback with performance preservation.

        Args:
            original_tier: The original model tier that failed
            context: Error context information
            budget_remaining: Remaining budget percentage

        Returns:
            FallbackResult with details of the fallback operation
        """
        start_time = time.time()
        fallback_chain = self.fallback_chains.get(original_tier, [])

        if not fallback_chain:
            return FallbackResult(
                success=False,
                fallback_used=None,
                original_error=None,
                recovery_time=0.0,
                performance_impact=1.0,
                cost_impact=0.0,
                message=f"No fallback chain defined for tier: {original_tier}",
            )

        # Filter fallback options based on budget and availability
        viable_options = await self._filter_viable_options(
            fallback_chain, budget_remaining, context
        )

        if not viable_options:
            return FallbackResult(
                success=False,
                fallback_used=None,
                original_error=None,
                recovery_time=time.time() - start_time,
                performance_impact=1.0,
                cost_impact=0.0,
                message="No viable fallback options available",
            )

        # Try each viable option in order
        for option in viable_options:
            try:
                # Test the fallback option
                test_result = await self._test_fallback_option(option, context)

                if test_result:
                    # Update success tracking
                    option.success_count += 1
                    option.last_used = datetime.utcnow()

                    # Calculate performance and cost impact
                    performance_impact = self._calculate_performance_impact(
                        original_tier, option
                    )
                    cost_impact = self._calculate_cost_impact(original_tier, option)

                    recovery_time = time.time() - start_time

                    # Record successful fallback
                    self._record_fallback_success(original_tier, option, recovery_time)

                    logger.info(
                        f"Successful fallback from {original_tier} to {option.name} "
                        f"(performance impact: {performance_impact:.2f}, cost impact: {cost_impact:.2f})"
                    )

                    return FallbackResult(
                        success=True,
                        fallback_used=option,
                        original_error=None,
                        recovery_time=recovery_time,
                        performance_impact=performance_impact,
                        cost_impact=cost_impact,
                        message=f"Successfully fell back to {option.name}",
                        metadata={
                            "original_tier": original_tier,
                            "fallback_tier": option.tier.value,
                            "performance_level": option.performance_level.value,
                        },
                    )

            except Exception as e:
                # Update failure tracking
                option.failure_count += 1
                logger.warning(f"Fallback option {option.name} failed: {e}")
                continue

        # All fallback options failed
        recovery_time = time.time() - start_time
        return FallbackResult(
            success=False,
            fallback_used=None,
            original_error=None,
            recovery_time=recovery_time,
            performance_impact=1.0,
            cost_impact=0.0,
            message="All fallback options failed",
        )

    async def _filter_viable_options(
        self,
        fallback_chain: List[FallbackOption],
        budget_remaining: float,
        context: ErrorContext,
    ) -> List[FallbackOption]:
        """Filter fallback options based on budget and availability constraints."""
        viable_options = []

        for option in fallback_chain:
            # Check budget constraints
            if budget_remaining < 15 and option.cost_per_million > 0:
                # Low budget - only free models
                continue
            elif budget_remaining < 30 and option.cost_per_million > 0.5:
                # Medium budget - avoid expensive models
                continue

            # Check availability
            if await self._check_availability(option):
                viable_options.append(option)

        # Sort by preference (tier, then performance, then cost)
        viable_options.sort(
            key=lambda x: (x.tier.value, -x.success_rate, x.cost_per_million)
        )

        return viable_options

    async def _check_availability(self, option: FallbackOption) -> bool:
        """Check if a fallback option is currently available."""
        check_type = option.availability_check

        if check_type == "openrouter_api":
            return bool(os.getenv("OPENROUTER_API_KEY")) and not os.getenv(
                "OPENROUTER_API_KEY", ""
            ).startswith("dummy_")
        elif check_type == "openrouter_free":
            return bool(
                os.getenv("OPENROUTER_API_KEY")
            )  # Free models still need OpenRouter access
        elif check_type == "metaculus_proxy":
            return os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true"

        return False

    async def _test_fallback_option(
        self, option: FallbackOption, context: ErrorContext
    ) -> bool:
        """Test if a fallback option is working."""
        try:
            if self.tri_model_router:
                # Create a test model instance
                from forecasting_tools import GeneralLlm

                if option.name.startswith("metaculus/"):
                    test_model = GeneralLlm(
                        model=option.name, api_key=None, **option.configuration
                    )
                else:
                    test_model = GeneralLlm(
                        model=option.name,
                        api_key=os.getenv("OPENROUTER_API_KEY"),
                        base_url="https://openrouter.ai/api/v1",
                        **option.configuration,
                    )

                # Quick test with timeout
                test_response = await asyncio.wait_for(
                    test_model.invoke("Test"), timeout=10.0
                )

                return bool(test_response and len(test_response.strip()) > 0)

            return True  # Assume available if no router to test with

        except Exception as e:
            logger.debug(f"Fallback option {option.name} test failed: {e}")
            return False

    def _calculate_performance_impact(
        self, original_tier: str, fallback_option: FallbackOption
    ) -> float:
        """Calculate performance impact of using fallback option."""
        # Performance impact based on tier downgrade
        tier_performance = {"full": 1.0, "mini": 0.8, "nano": 0.6}

        option_performance = {
            PerformanceLevel.OPTIMAL: 1.0,
            PerformanceLevel.GOOD: 0.8,
            PerformanceLevel.ACCEPTABLE: 0.6,
            PerformanceLevel.MINIMAL: 0.4,
        }

        original_performance = tier_performance.get(original_tier, 1.0)
        fallback_performance = option_performance.get(
            fallback_option.performance_level, 0.5
        )

        return fallback_performance / original_performance

    def _calculate_cost_impact(
        self, original_tier: str, fallback_option: FallbackOption
    ) -> float:
        """Calculate cost impact of using fallback option."""
        original_costs = {"full": 1.50, "mini": 0.25, "nano": 0.05}

        original_cost = original_costs.get(original_tier, 1.0)
        fallback_cost = fallback_option.cost_per_million

        if original_cost == 0:
            return 0.0

        return (fallback_cost - original_cost) / original_cost

    def _record_fallback_success(
        self, original_tier: str, option: FallbackOption, recovery_time: float
    ):
        """Record successful fallback for analysis."""
        self.fallback_history.append(
            {
                "timestamp": datetime.utcnow(),
                "original_tier": original_tier,
                "fallback_option": option.name,
                "fallback_tier": option.tier.value,
                "recovery_time": recovery_time,
                "success": True,
            }
        )

        # Keep only recent history
        if len(self.fallback_history) > 1000:
            self.fallback_history = self.fallback_history[-1000:]

    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback usage statistics."""
        if not self.fallback_history:
            return {"total_fallbacks": 0}

        recent_fallbacks = [
            f
            for f in self.fallback_history
            if f["timestamp"] > datetime.utcnow() - timedelta(hours=24)
        ]

        tier_usage = {}
        option_usage = {}
        avg_recovery_time = 0

        for fallback in recent_fallbacks:
            tier = fallback["fallback_tier"]
            option = fallback["fallback_option"]

            tier_usage[tier] = tier_usage.get(tier, 0) + 1
            option_usage[option] = option_usage.get(option, 0) + 1
            avg_recovery_time += fallback["recovery_time"]

        if recent_fallbacks:
            avg_recovery_time /= len(recent_fallbacks)

        return {
            "total_fallbacks": len(self.fallback_history),
            "recent_fallbacks_24h": len(recent_fallbacks),
            "tier_usage": tier_usage,
            "option_usage": option_usage,
            "average_recovery_time": avg_recovery_time,
        }


class CrossProviderFallbackManager:
    """
    Manages cross-provider API fallbacks with intelligent routing.
    """

    def __init__(self):
        self.provider_configs = self._initialize_provider_configs()
        self.provider_health = {}
        self.fallback_history = []

    def _initialize_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize provider configurations with fallback chains."""
        return {
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "priority": 1,
                "cost_multiplier": 1.0,
                "availability_check": self._check_openrouter_availability,
                "fallback_to": ["metaculus_proxy", "free_models"],
            },
            "metaculus_proxy": {
                "base_url": None,  # Uses default
                "api_key_env": None,  # No API key needed
                "priority": 2,
                "cost_multiplier": 0.0,  # Uses proxy credits
                "availability_check": self._check_metaculus_proxy_availability,
                "fallback_to": ["free_models"],
            },
            "free_models": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "priority": 3,
                "cost_multiplier": 0.0,
                "availability_check": self._check_free_models_availability,
                "fallback_to": [],
            },
        }

    async def execute_provider_fallback(
        self, original_provider: str, context: ErrorContext
    ) -> FallbackResult:
        """
        Execute cross-provider fallback with intelligent routing.

        Args:
            original_provider: The original provider that failed
            context: Error context information

        Returns:
            FallbackResult with details of the provider fallback
        """
        start_time = time.time()

        # Get fallback chain for the original provider
        provider_config = self.provider_configs.get(original_provider, {})
        fallback_chain = provider_config.get("fallback_to", [])

        if not fallback_chain:
            return FallbackResult(
                success=False,
                fallback_used=None,
                original_error=None,
                recovery_time=0.0,
                performance_impact=1.0,
                cost_impact=0.0,
                message=f"No fallback providers defined for: {original_provider}",
            )

        # Try each fallback provider
        for provider_name in fallback_chain:
            try:
                # Check provider availability
                if await self._check_provider_availability(provider_name):
                    # Test the provider
                    if await self._test_provider(provider_name, context):
                        recovery_time = time.time() - start_time

                        # Create fallback option for result
                        fallback_option = FallbackOption(
                            name=provider_name,
                            tier=FallbackTier.SECONDARY,
                            performance_level=PerformanceLevel.GOOD,
                            cost_per_million=self.provider_configs[provider_name][
                                "cost_multiplier"
                            ],
                            availability_check=provider_name,
                            configuration={},
                        )

                        # Record successful provider fallback
                        self._record_provider_fallback(
                            original_provider, provider_name, recovery_time
                        )

                        logger.info(
                            f"Successful provider fallback from {original_provider} to {provider_name}"
                        )

                        return FallbackResult(
                            success=True,
                            fallback_used=fallback_option,
                            original_error=None,
                            recovery_time=recovery_time,
                            performance_impact=0.9,  # Slight performance impact
                            cost_impact=self.provider_configs[provider_name][
                                "cost_multiplier"
                            ]
                            - 1.0,
                            message=f"Successfully fell back to provider: {provider_name}",
                            metadata={
                                "original_provider": original_provider,
                                "fallback_provider": provider_name,
                            },
                        )

            except Exception as e:
                logger.warning(f"Provider fallback to {provider_name} failed: {e}")
                continue

        # All provider fallbacks failed
        recovery_time = time.time() - start_time
        return FallbackResult(
            success=False,
            fallback_used=None,
            original_error=None,
            recovery_time=recovery_time,
            performance_impact=1.0,
            cost_impact=0.0,
            message="All provider fallbacks failed",
        )

    async def _check_provider_availability(self, provider_name: str) -> bool:
        """Check if a provider is available."""
        provider_config = self.provider_configs.get(provider_name, {})
        availability_check = provider_config.get("availability_check")

        if availability_check:
            return await availability_check()

        return False

    async def _check_openrouter_availability(self) -> bool:
        """Check OpenRouter availability."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        return bool(api_key and not api_key.startswith("dummy_"))

    async def _check_metaculus_proxy_availability(self) -> bool:
        """Check Metaculus proxy availability."""
        return os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true"

    async def _check_free_models_availability(self) -> bool:
        """Check free models availability."""
        # Free models require OpenRouter access
        return await self._check_openrouter_availability()

    async def _test_provider(self, provider_name: str, context: ErrorContext) -> bool:
        """Test if a provider is working."""
        try:
            # Simple availability test
            provider_config = self.provider_configs[provider_name]

            # For now, just check configuration validity
            # In a real implementation, this would make a test API call
            return bool(provider_config)

        except Exception as e:
            logger.debug(f"Provider {provider_name} test failed: {e}")
            return False

    def _record_provider_fallback(
        self, original_provider: str, fallback_provider: str, recovery_time: float
    ):
        """Record provider fallback for analysis."""
        self.fallback_history.append(
            {
                "timestamp": datetime.utcnow(),
                "original_provider": original_provider,
                "fallback_provider": fallback_provider,
                "recovery_time": recovery_time,
                "success": True,
            }
        )

        # Keep only recent history
        if len(self.fallback_history) > 1000:
            self.fallback_history = self.fallback_history[-1000:]


class EmergencyModeManager:
    """
    Manages emergency mode activation for critical failures.
    """

    def __init__(self, budget_manager=None):
        self.budget_manager = budget_manager
        self.emergency_active = False
        self.emergency_start_time = None
        self.emergency_config = self._initialize_emergency_config()

    def _initialize_emergency_config(self) -> Dict[str, Any]:
        """Initialize emergency mode configuration."""
        return {
            "free_models_only": True,
            "minimal_functionality": True,
            "reduced_timeouts": True,
            "essential_features": [
                "basic_forecasting",
                "simple_research",
                "error_recovery",
            ],
            "disabled_features": [
                "advanced_analysis",
                "detailed_reasoning",
                "comprehensive_research",
            ],
            "model_restrictions": {
                "allowed_models": [
                    "openai/gpt-oss-20b:free",
                    "moonshotai/kimi-k2:free",
                    "metaculus/gpt-4o-mini",
                ],
                "max_tokens": 500,
                "temperature": 0.1,
            },
        }

    async def activate_emergency_mode(
        self, trigger_error: Exception, context: ErrorContext
    ) -> RecoveryAction:
        """
        Activate emergency mode for critical failures.

        Args:
            trigger_error: The error that triggered emergency mode
            context: Error context information

        Returns:
            RecoveryAction for emergency mode operation
        """
        if not self.emergency_active:
            self.emergency_active = True
            self.emergency_start_time = datetime.utcnow()

            logger.critical(f"Emergency mode activated due to: {trigger_error}")

            # Notify about emergency mode activation
            await self._send_emergency_alert(trigger_error, context)

        # Configure emergency operation parameters
        emergency_params = {
            "free_models_only": True,
            "minimal_functionality": True,
            "allowed_models": self.emergency_config["model_restrictions"][
                "allowed_models"
            ],
            "max_tokens": self.emergency_config["model_restrictions"]["max_tokens"],
            "temperature": self.emergency_config["model_restrictions"]["temperature"],
            "essential_features": self.emergency_config["essential_features"],
            "disabled_features": self.emergency_config["disabled_features"],
        }

        return RecoveryAction(
            strategy=RecoveryStrategy.EMERGENCY_MODE,
            parameters=emergency_params,
            expected_delay=1.0,
            success_probability=0.7,
            fallback_action=None,
        )

    async def deactivate_emergency_mode(self) -> bool:
        """
        Deactivate emergency mode and return to normal operation.

        Returns:
            True if successfully deactivated, False otherwise
        """
        if not self.emergency_active:
            return True

        try:
            # Check if conditions allow normal operation
            if await self._check_normal_operation_conditions():
                self.emergency_active = False
                emergency_duration = datetime.utcnow() - self.emergency_start_time

                logger.info(f"Emergency mode deactivated after {emergency_duration}")
                return True
            else:
                logger.warning("Conditions not met for emergency mode deactivation")
                return False

        except Exception as e:
            logger.error(f"Error deactivating emergency mode: {e}")
            return False

    async def _check_normal_operation_conditions(self) -> bool:
        """Check if conditions allow return to normal operation."""
        try:
            # Check budget availability
            if self.budget_manager:
                budget_status = await self.budget_manager.get_budget_status()
                if budget_status.get("remaining_percentage", 0) < 5:
                    return False

            # Check API availability
            openrouter_available = bool(os.getenv("OPENROUTER_API_KEY"))
            if not openrouter_available:
                return False

            # Additional health checks could be added here
            return True

        except Exception:
            return False

    async def _send_emergency_alert(
        self, trigger_error: Exception, context: ErrorContext
    ):
        """Send emergency mode activation alert."""
        alert_message = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "emergency_mode_activated",
            "trigger_error": str(trigger_error),
            "context": {
                "task_type": context.task_type,
                "model_tier": context.model_tier,
                "operation_mode": context.operation_mode,
                "budget_remaining": context.budget_remaining,
                "attempt_number": context.attempt_number,
            },
        }

        # Log the alert
        logger.critical(
            f"EMERGENCY MODE ACTIVATED: {json.dumps(alert_message, indent=2)}"
        )

        # Write to emergency log file
        try:
            emergency_log_path = "logs/emergency_alerts.json"
            os.makedirs(os.path.dirname(emergency_log_path), exist_ok=True)

            with open(emergency_log_path, "a") as f:
                f.write(json.dumps(alert_message) + "\n")

        except Exception as e:
            logger.error(f"Failed to write emergency alert to file: {e}")

    def is_emergency_active(self) -> bool:
        """Check if emergency mode is currently active."""
        return self.emergency_active

    def get_emergency_status(self) -> Dict[str, Any]:
        """Get current emergency mode status."""
        if not self.emergency_active:
            return {"active": False}

        duration = (
            datetime.utcnow() - self.emergency_start_time
            if self.emergency_start_time
            else timedelta(0)
        )

        return {
            "active": True,
            "start_time": (
                self.emergency_start_time.isoformat()
                if self.emergency_start_time
                else None
            ),
            "duration_seconds": duration.total_seconds(),
            "config": self.emergency_config,
        }


class ErrorLoggingAndAlertingSystem:
    """
    Comprehensive error logging and alerting system.
    """

    def __init__(self, alert_config: Optional[AlertConfig] = None):
        self.alert_config = alert_config or AlertConfig()
        self.error_log = []
        self.alert_history = []
        self.last_alert_times = {}

    async def log_error(
        self,
        error: Exception,
        context: ErrorContext,
        recovery_action: Optional[RecoveryAction] = None,
    ):
        """
        Log error with comprehensive details and trigger alerts if needed.

        Args:
            error: The exception that occurred
            context: Error context information
            recovery_action: Recovery action taken (if any)
        """
        # Create error log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": {
                "task_type": context.task_type,
                "model_tier": context.model_tier,
                "operation_mode": context.operation_mode,
                "budget_remaining": context.budget_remaining,
                "attempt_number": context.attempt_number,
                "model_name": context.model_name,
                "provider": context.provider,
            },
            "recovery_action": (
                {
                    "strategy": (
                        recovery_action.strategy.value if recovery_action else None
                    ),
                    "parameters": (
                        recovery_action.parameters if recovery_action else None
                    ),
                    "success_probability": (
                        recovery_action.success_probability if recovery_action else None
                    ),
                }
                if recovery_action
                else None
            ),
        }

        # Add to error log
        self.error_log.append(log_entry)

        # Keep only recent errors (last 10000)
        if len(self.error_log) > 10000:
            self.error_log = self.error_log[-10000:]

        # Log to standard logger
        logger.error(
            f"Error logged: {error} | Context: {context.task_type}/{context.model_tier} | "
            f"Recovery: {recovery_action.strategy.value if recovery_action else 'None'}"
        )

        # Check if alert should be triggered
        await self._check_and_trigger_alerts(error, context)

        # Write to persistent log file
        await self._write_to_log_file(log_entry)

    async def _check_and_trigger_alerts(self, error: Exception, context: ErrorContext):
        """Check if error conditions warrant an alert."""
        error_type = type(error).__name__
        current_time = datetime.utcnow()

        # Check if we're in cooldown period for this error type
        last_alert = self.last_alert_times.get(error_type)
        if last_alert:
            cooldown_period = timedelta(minutes=self.alert_config.cooldown_minutes)
            if current_time - last_alert < cooldown_period:
                return

        # Count recent errors of this type
        time_window = timedelta(minutes=self.alert_config.time_window_minutes)
        cutoff_time = current_time - time_window

        recent_errors = [
            entry
            for entry in self.error_log
            if (
                datetime.fromisoformat(entry["timestamp"]) > cutoff_time
                and entry["error_type"] == error_type
            )
        ]

        # Trigger alert if threshold exceeded
        if len(recent_errors) >= self.alert_config.error_threshold:
            await self._send_alert(error_type, len(recent_errors), context)
            self.last_alert_times[error_type] = current_time

    async def _send_alert(
        self, error_type: str, error_count: int, context: ErrorContext
    ):
        """Send error alert through configured channels."""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": "error_threshold_exceeded",
            "error_type": error_type,
            "error_count": error_count,
            "time_window_minutes": self.alert_config.time_window_minutes,
            "context": {
                "task_type": context.task_type,
                "model_tier": context.model_tier,
                "operation_mode": context.operation_mode,
                "budget_remaining": context.budget_remaining,
            },
        }

        # Add to alert history
        self.alert_history.append(alert)

        # Send through configured channels
        for channel in self.alert_config.notification_channels:
            if channel == "log":
                logger.warning(
                    f"ALERT: {error_type} occurred {error_count} times in {self.alert_config.time_window_minutes} minutes"
                )
            elif channel == "file":
                await self._write_alert_to_file(alert)

    async def _write_to_log_file(self, log_entry: Dict[str, Any]):
        """Write error log entry to persistent file."""
        try:
            log_file_path = "logs/error_log.json"
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            with open(log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Failed to write to error log file: {e}")

    async def _write_alert_to_file(self, alert: Dict[str, Any]):
        """Write alert to persistent file."""
        try:
            alert_file_path = "logs/alerts.json"
            os.makedirs(os.path.dirname(alert_file_path), exist_ok=True)

            with open(alert_file_path, "a") as f:
                f.write(json.dumps(alert) + "\n")

        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        recent_errors = [
            entry
            for entry in self.error_log
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]

        if not recent_errors:
            return {"total_errors": 0, "time_period_hours": hours}

        # Analyze error patterns
        error_types = {}
        model_tiers = {}
        operation_modes = {}

        for error in recent_errors:
            error_type = error["error_type"]
            model_tier = error["context"]["model_tier"]
            operation_mode = error["context"]["operation_mode"]

            error_types[error_type] = error_types.get(error_type, 0) + 1
            model_tiers[model_tier] = model_tiers.get(model_tier, 0) + 1
            operation_modes[operation_mode] = operation_modes.get(operation_mode, 0) + 1

        return {
            "total_errors": len(recent_errors),
            "time_period_hours": hours,
            "error_types": error_types,
            "model_tiers": model_tiers,
            "operation_modes": operation_modes,
            "alerts_sent": len(
                [
                    a
                    for a in self.alert_history
                    if datetime.fromisoformat(a["timestamp"]) > cutoff_time
                ]
            ),
        }


class IntelligentFallbackOrchestrator:
    """
    Orchestrates all fallback strategies with intelligent decision making.
    """

    def __init__(self, tri_model_router=None, budget_manager=None):
        self.model_fallback_manager = ModelTierFallbackManager(tri_model_router)
        self.provider_fallback_manager = CrossProviderFallbackManager()
        self.emergency_manager = EmergencyModeManager(budget_manager)
        self.logging_system = ErrorLoggingAndAlertingSystem()

    async def execute_intelligent_fallback(
        self, error: Exception, context: ErrorContext
    ) -> FallbackResult:
        """
        Execute intelligent fallback strategy based on error type and context.

        Args:
            error: The exception that occurred
            context: Error context information

        Returns:
            FallbackResult with details of the fallback operation
        """
        # Log the error
        await self.logging_system.log_error(error, context)

        # Determine fallback strategy based on error type and severity
        if isinstance(error, BudgetError) or context.budget_remaining < 5:
            # Budget exhaustion - activate emergency mode
            return await self._handle_budget_emergency(error, context)

        elif isinstance(error, ModelError):
            # Model-specific error - try model tier fallback first
            return await self._handle_model_error(error, context)

        elif isinstance(error, APIError):
            # API error - try provider fallback
            return await self._handle_api_error(error, context)

        else:
            # Generic error - try model fallback first, then provider fallback
            return await self._handle_generic_error(error, context)

    async def _handle_budget_emergency(
        self, error: Exception, context: ErrorContext
    ) -> FallbackResult:
        """Handle budget exhaustion emergency."""
        # Activate emergency mode
        recovery_action = await self.emergency_manager.activate_emergency_mode(
            error, context
        )

        # Log the recovery action
        await self.logging_system.log_error(error, context, recovery_action)

        return FallbackResult(
            success=True,
            fallback_used=FallbackOption(
                name="emergency_mode",
                tier=FallbackTier.CRITICAL,
                performance_level=PerformanceLevel.MINIMAL,
                cost_per_million=0.0,
                availability_check="always",
                configuration=recovery_action.parameters,
            ),
            original_error=error,
            recovery_time=recovery_action.expected_delay,
            performance_impact=0.3,  # Significant performance reduction
            cost_impact=-1.0,  # Cost savings
            message="Emergency mode activated due to budget exhaustion",
        )

    async def _handle_model_error(
        self, error: Exception, context: ErrorContext
    ) -> FallbackResult:
        """Handle model-specific errors."""
        # Try model tier fallback first
        model_fallback_result = await self.model_fallback_manager.execute_fallback(
            context.model_tier, context, context.budget_remaining
        )

        if model_fallback_result.success:
            return model_fallback_result

        # If model fallback fails, try provider fallback
        provider_fallback_result = (
            await self.provider_fallback_manager.execute_provider_fallback(
                context.provider or "openrouter", context
            )
        )

        return provider_fallback_result

    async def _handle_api_error(
        self, error: Exception, context: ErrorContext
    ) -> FallbackResult:
        """Handle API-specific errors."""
        # Try provider fallback first for API errors
        provider_fallback_result = (
            await self.provider_fallback_manager.execute_provider_fallback(
                context.provider or "openrouter", context
            )
        )

        if provider_fallback_result.success:
            return provider_fallback_result

        # If provider fallback fails, try model fallback
        model_fallback_result = await self.model_fallback_manager.execute_fallback(
            context.model_tier, context, context.budget_remaining
        )

        return model_fallback_result

    async def _handle_generic_error(
        self, error: Exception, context: ErrorContext
    ) -> FallbackResult:
        """Handle generic errors with comprehensive fallback strategy."""
        # Try model fallback first
        model_fallback_result = await self.model_fallback_manager.execute_fallback(
            context.model_tier, context, context.budget_remaining
        )

        if model_fallback_result.success:
            return model_fallback_result

        # Try provider fallback
        provider_fallback_result = (
            await self.provider_fallback_manager.execute_provider_fallback(
                context.provider or "openrouter", context
            )
        )

        if provider_fallback_result.success:
            return provider_fallback_result

        # Last resort - emergency mode
        return await self._handle_budget_emergency(error, context)

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all fallback systems."""
        return {
            "model_fallback": self.model_fallback_manager.get_fallback_statistics(),
            "provider_fallback": {
                "total_fallbacks": len(self.provider_fallback_manager.fallback_history)
            },
            "emergency_mode": self.emergency_manager.get_emergency_status(),
            "error_summary": self.logging_system.get_error_summary(),
            "timestamp": datetime.utcnow().isoformat(),
        }
