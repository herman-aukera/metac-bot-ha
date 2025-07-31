"""
Feature flags system for gradual rollout and A/B testing of optimization features.
Supports environment-based configuration and runtime flag evaluation.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import hashlib

logger = logging.getLogger(__name__)


class FlagType(Enum):
    """Types of feature flags"""
    BOOLEAN = "boolean"
    STRING = "string"
    NUMBER = "number"
    JSON = "json"


class RolloutStrategy(Enum):
    """Rollout strategies for feature flags"""
    ALL_USERS = "all_users"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    ENVIRONMENT = "environment"
    A_B_TEST = "a_b_test"


@dataclass
class FlagRule:
    """Rule for feature flag evaluation"""
    strategy: RolloutStrategy
    value: Any
    percentage: Optional[float] = None
    user_list: Optional[List[str]] = None
    environments: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureFlag:
    """Feature flag definition"""
    key: str
    name: str
    description: str
    flag_type: FlagType
    default_value: Any
    rules: List[FlagRule] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)


class FeatureFlagManager:
    """
    Feature flag manager with support for various rollout strategies
    and A/B testing capabilities.
    """

    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        self.config_path = config_path or os.getenv("FEATURE_FLAGS_CONFIG", "configs/feature_flags.json")
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.flags: Dict[str, FeatureFlag] = {}
        self.user_context: Dict[str, Any] = {}
        self._load_flags()

    def _load_flags(self) -> None:
        """Load feature flags from configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    self._parse_flags(config_data)
                logger.info(f"Loaded {len(self.flags)} feature flags from {self.config_path}")
            else:
                logger.warning(f"Feature flags config file not found: {self.config_path}")
                self._load_default_flags()
        except Exception as e:
            logger.error(f"Error loading feature flags: {e}")
            self._load_default_flags()

    def _parse_flags(self, config_data: Dict[str, Any]) -> None:
        """Parse feature flags from configuration data"""
        for flag_key, flag_data in config_data.get("flags", {}).items():
            try:
                rules = []
                for rule_data in flag_data.get("rules", []):
                    rule = FlagRule(
                        strategy=RolloutStrategy(rule_data["strategy"]),
                        value=rule_data["value"],
                        percentage=rule_data.get("percentage"),
                        user_list=rule_data.get("user_list"),
                        environments=rule_data.get("environments"),
                        start_date=self._parse_datetime(rule_data.get("start_date")),
                        end_date=self._parse_datetime(rule_data.get("end_date")),
                        metadata=rule_data.get("metadata", {})
                    )
                    rules.append(rule)

                flag = FeatureFlag(
                    key=flag_key,
                    name=flag_data["name"],
                    description=flag_data["description"],
                    flag_type=FlagType(flag_data["type"]),
                    default_value=flag_data["default_value"],
                    rules=rules,
                    enabled=flag_data.get("enabled", True),
                    tags=flag_data.get("tags", [])
                )
                self.flags[flag_key] = flag
            except Exception as e:
                logger.error(f"Error parsing flag {flag_key}: {e}")

    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string to datetime object"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            logger.error(f"Invalid datetime format: {date_str}")
            return None

    def _load_default_flags(self) -> None:
        """Load default feature flags for tournament optimization"""
        default_flags = {
            "ensemble_reasoning": FeatureFlag(
                key="ensemble_reasoning",
                name="Ensemble Reasoning",
                description="Enable ensemble reasoning with multiple agents",
                flag_type=FlagType.BOOLEAN,
                default_value=True,
                rules=[
                    FlagRule(
                        strategy=RolloutStrategy.ENVIRONMENT,
                        value=True,
                        environments=["production", "staging"]
                    )
                ]
            ),
            "advanced_research_pipeline": FeatureFlag(
                key="advanced_research_pipeline",
                name="Advanced Research Pipeline",
                description="Enable multi-provider research pipeline",
                flag_type=FlagType.BOOLEAN,
                default_value=False,
                rules=[
                    FlagRule(
                        strategy=RolloutStrategy.PERCENTAGE,
                        value=True,
                        percentage=50.0
                    )
                ]
            ),
            "tournament_strategy_optimization": FeatureFlag(
                key="tournament_strategy_optimization",
                name="Tournament Strategy Optimization",
                description="Enable advanced tournament strategy optimization",
                flag_type=FlagType.BOOLEAN,
                default_value=False,
                rules=[
                    FlagRule(
                        strategy=RolloutStrategy.A_B_TEST,
                        value=True,
                        percentage=25.0,
                        metadata={"test_group": "optimization_test"}
                    )
                ]
            ),
            "max_ensemble_agents": FeatureFlag(
                key="max_ensemble_agents",
                name="Maximum Ensemble Agents",
                description="Maximum number of agents in ensemble",
                flag_type=FlagType.NUMBER,
                default_value=3,
                rules=[
                    FlagRule(
                        strategy=RolloutStrategy.ENVIRONMENT,
                        value=5,
                        environments=["production"]
                    )
                ]
            ),
            "research_timeout_seconds": FeatureFlag(
                key="research_timeout_seconds",
                name="Research Timeout",
                description="Timeout for research operations in seconds",
                flag_type=FlagType.NUMBER,
                default_value=30,
                rules=[
                    FlagRule(
                        strategy=RolloutStrategy.ALL_USERS,
                        value=45
                    )
                ]
            )
        }

        self.flags.update(default_flags)
        logger.info(f"Loaded {len(default_flags)} default feature flags")

    def set_user_context(self, user_id: str, attributes: Dict[str, Any]) -> None:
        """Set user context for flag evaluation"""
        self.user_context = {
            "user_id": user_id,
            "environment": self.environment,
            **attributes
        }

    def is_enabled(self, flag_key: str, user_id: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature flag is enabled for the given context"""
        try:
            flag = self.flags.get(flag_key)
            if not flag or not flag.enabled:
                return False

            # Merge context
            evaluation_context = {**self.user_context}
            if user_id:
                evaluation_context["user_id"] = user_id
            if context:
                evaluation_context.update(context)

            # Evaluate rules
            for rule in flag.rules:
                if self._evaluate_rule(rule, evaluation_context):
                    result = bool(rule.value) if flag.flag_type == FlagType.BOOLEAN else rule.value
                    logger.debug(f"Flag {flag_key} evaluated to {result} via rule {rule.strategy}")
                    return bool(result)

            # Return default value if no rules match
            result = bool(flag.default_value) if flag.flag_type == FlagType.BOOLEAN else flag.default_value
            logger.debug(f"Flag {flag_key} using default value: {result}")
            return bool(result)

        except Exception as e:
            logger.error(f"Error evaluating flag {flag_key}: {e}")
            return False

    def get_value(self, flag_key: str, user_id: Optional[str] = None,
                  context: Optional[Dict[str, Any]] = None) -> Any:
        """Get the value of a feature flag for the given context"""
        try:
            flag = self.flags.get(flag_key)
            if not flag or not flag.enabled:
                return flag.default_value if flag else None

            # Merge context
            evaluation_context = {**self.user_context}
            if user_id:
                evaluation_context["user_id"] = user_id
            if context:
                evaluation_context.update(context)

            # Evaluate rules
            for rule in flag.rules:
                if self._evaluate_rule(rule, evaluation_context):
                    logger.debug(f"Flag {flag_key} evaluated to {rule.value} via rule {rule.strategy}")
                    return rule.value

            # Return default value if no rules match
            logger.debug(f"Flag {flag_key} using default value: {flag.default_value}")
            return flag.default_value

        except Exception as e:
            logger.error(f"Error getting flag value {flag_key}: {e}")
            flag = self.flags.get(flag_key)
            return flag.default_value if flag else None

    def _evaluate_rule(self, rule: FlagRule, context: Dict[str, Any]) -> bool:
        """Evaluate a single rule against the given context"""
        now = datetime.now(timezone.utc)

        # Check time constraints
        if rule.start_date and now < rule.start_date:
            return False
        if rule.end_date and now > rule.end_date:
            return False

        # Evaluate based on strategy
        if rule.strategy == RolloutStrategy.ALL_USERS:
            return True

        elif rule.strategy == RolloutStrategy.ENVIRONMENT:
            if rule.environments:
                return context.get("environment") in rule.environments
            return True

        elif rule.strategy == RolloutStrategy.USER_LIST:
            if rule.user_list:
                return context.get("user_id") in rule.user_list
            return False

        elif rule.strategy == RolloutStrategy.PERCENTAGE:
            if rule.percentage is not None:
                user_id = context.get("user_id", "anonymous")
                return self._hash_user_percentage(user_id) < rule.percentage
            return False

        elif rule.strategy == RolloutStrategy.A_B_TEST:
            if rule.percentage is not None:
                user_id = context.get("user_id", "anonymous")
                test_group = rule.metadata.get("test_group", "default")
                return self._hash_user_percentage(f"{user_id}:{test_group}") < rule.percentage
            return False

        return False

    def _hash_user_percentage(self, user_key: str) -> float:
        """Generate a consistent percentage (0-100) for a user key"""
        hash_value = hashlib.md5(user_key.encode()).hexdigest()
        # Use first 8 characters of hash to generate percentage
        return (int(hash_value[:8], 16) % 10000) / 100.0

    def get_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all feature flags with their current status"""
        result = {}
        for key, flag in self.flags.items():
            result[key] = {
                "name": flag.name,
                "description": flag.description,
                "type": flag.flag_type.value,
                "enabled": flag.enabled,
                "default_value": flag.default_value,
                "current_value": self.get_value(key),
                "tags": flag.tags
            }
        return result

    def reload_flags(self) -> None:
        """Reload feature flags from configuration file"""
        self.flags.clear()
        self._load_flags()
        logger.info("Feature flags reloaded")


# Global feature flag manager instance
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance"""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager


def is_feature_enabled(flag_key: str, user_id: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to check if a feature is enabled"""
    return get_feature_flag_manager().is_enabled(flag_key, user_id, context)


def get_feature_value(flag_key: str, user_id: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> Any:
    """Convenience function to get a feature flag value"""
    return get_feature_flag_manager().get_value(flag_key, user_id, context)
