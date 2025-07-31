"""Secure configuration management with environment variable validation."""

import os
import re
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from ...domain.exceptions.infrastructure_exceptions import ConfigurationError


class ConfigType(Enum):
    """Configuration value types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    URL = "url"
    EMAIL = "email"
    PATH = "path"
    IP_ADDRESS = "ip_address"
    PORT = "port"
    SECRET = "secret"


@dataclass
class ConfigRule:
    """Configuration validation rule."""
    key: str
    config_type: ConfigType
    required: bool = True
    default_value: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    validator: Optional[Callable[[Any], bool]] = None
    description: Optional[str] = None
    sensitive: bool = False


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    value: Any
    errors: List[str]
    warnings: List[str]


class ConfigValidator:
    """Secure configuration management with environment variable validation."""

    def __init__(self, config_file_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_file_path = config_file_path

        # Configuration rules
        self.rules: Dict[str, ConfigRule] = {}

        # Validated configuration cache
        self._config_cache: Dict[str, Any] = {}

        # Sensitive keys that should be masked in logs
        self.sensitive_patterns = [
            r'.*password.*',
            r'.*secret.*',
            r'.*key.*',
            r'.*token.*',
            r'.*credential.*',
            r'.*auth.*'
        ]

        # Setup default configuration rules
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default configuration validation rules."""
        # Database configuration
        self.add_rule(ConfigRule(
            key="DATABASE_URL",
            config_type=ConfigType.URL,
            required=False,
            description="Database connection URL",
            sensitive=True
        ))

        # Redis configuration
        self.add_rule(ConfigRule(
            key="REDIS_URL",
            config_type=ConfigType.URL,
            required=False,
            default_value="redis://localhost:6379",
            description="Redis connection URL"
        ))

        # API Keys
        self.add_rule(ConfigRule(
            key="OPENAI_API_KEY",
            config_type=ConfigType.SECRET,
            required=False,
            min_length=20,
            description="OpenAI API key",
            sensitive=True
        ))

        self.add_rule(ConfigRule(
            key="ANTHROPIC_API_KEY",
            config_type=ConfigType.SECRET,
            required=False,
            min_length=20,
            description="Anthropic API key",
            sensitive=True
        ))

        self.add_rule(ConfigRule(
            key="METACULUS_API_KEY",
            config_type=ConfigType.SECRET,
            required=False,
            description="Metaculus API key",
            sensitive=True
        ))

        # Search provider keys
        self.add_rule(ConfigRule(
            key="ASKNEWS_API_KEY",
            config_type=ConfigType.SECRET,
            required=False,
            description="AskNews API key",
            sensitive=True
        ))

        self.add_rule(ConfigRule(
            key="PERPLEXITY_API_KEY",
            config_type=ConfigType.SECRET,
            required=False,
            description="Perplexity API key",
            sensitive=True
        ))

        self.add_rule(ConfigRule(
            key="EXA_API_KEY",
            config_type=ConfigType.SECRET,
            required=False,
            description="Exa API key",
            sensitive=True
        ))

        self.add_rule(ConfigRule(
            key="SERPAPI_API_KEY",
            config_type=ConfigType.SECRET,
            required=False,
            description="SerpAPI key",
            sensitive=True
        ))

        # Security configuration
        self.add_rule(ConfigRule(
            key="SECRET_KEY",
            config_type=ConfigType.SECRET,
            required=True,
            min_length=32,
            description="Application secret key",
            sensitive=True
        ))

        self.add_rule(ConfigRule(
            key="CREDENTIAL_ENCRYPTION_KEY",
            config_type=ConfigType.SECRET,
            required=False,
            description="Credential encryption key",
            sensitive=True
        ))

        self.add_rule(ConfigRule(
            key="CREDENTIAL_PASSWORD",
            config_type=ConfigType.SECRET,
            required=False,
            default_value="default-password",
            description="Credential encryption password",
            sensitive=True
        ))

        # Vault configuration
        self.add_rule(ConfigRule(
            key="VAULT_URL",
            config_type=ConfigType.URL,
            required=False,
            description="HashiCorp Vault URL"
        ))

        self.add_rule(ConfigRule(
            key="VAULT_TOKEN",
            config_type=ConfigType.SECRET,
            required=False,
            description="HashiCorp Vault token",
            sensitive=True
        ))

        # Application configuration
        self.add_rule(ConfigRule(
            key="ENVIRONMENT",
            config_type=ConfigType.STRING,
            required=False,
            default_value="development",
            allowed_values=["development", "staging", "production"],
            description="Application environment"
        ))

        self.add_rule(ConfigRule(
            key="DEBUG",
            config_type=ConfigType.BOOLEAN,
            required=False,
            default_value=False,
            description="Enable debug mode"
        ))

        self.add_rule(ConfigRule(
            key="LOG_LEVEL",
            config_type=ConfigType.STRING,
            required=False,
            default_value="INFO",
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            description="Logging level"
        ))

        # Server configuration
        self.add_rule(ConfigRule(
            key="HOST",
            config_type=ConfigType.STRING,
            required=False,
            default_value="localhost",
            description="Server host"
        ))

        self.add_rule(ConfigRule(
            key="PORT",
            config_type=ConfigType.PORT,
            required=False,
            default_value=8000,
            min_value=1,
            max_value=65535,
            description="Server port"
        ))

        # Rate limiting configuration
        self.add_rule(ConfigRule(
            key="RATE_LIMIT_ENABLED",
            config_type=ConfigType.BOOLEAN,
            required=False,
            default_value=True,
            description="Enable rate limiting"
        ))

        self.add_rule(ConfigRule(
            key="RATE_LIMIT_REQUESTS_PER_MINUTE",
            config_type=ConfigType.INTEGER,
            required=False,
            default_value=100,
            min_value=1,
            max_value=10000,
            description="Rate limit requests per minute"
        ))

        # Monitoring configuration
        self.add_rule(ConfigRule(
            key="ENABLE_METRICS",
            config_type=ConfigType.BOOLEAN,
            required=False,
            default_value=True,
            description="Enable metrics collection"
        ))

        self.add_rule(ConfigRule(
            key="METRICS_PORT",
            config_type=ConfigType.PORT,
            required=False,
            default_value=9090,
            description="Metrics server port"
        ))

        # Audit logging configuration
        self.add_rule(ConfigRule(
            key="AUDIT_LOG_ENABLED",
            config_type=ConfigType.BOOLEAN,
            required=False,
            default_value=True,
            description="Enable audit logging"
        ))

        self.add_rule(ConfigRule(
            key="AUDIT_LOG_PATH",
            config_type=ConfigType.PATH,
            required=False,
            default_value="logs/audit.log",
            description="Audit log file path"
        ))

        # Tournament configuration
        self.add_rule(ConfigRule(
            key="MAX_CONCURRENT_QUESTIONS",
            config_type=ConfigType.INTEGER,
            required=False,
            default_value=10,
            min_value=1,
            max_value=100,
            description="Maximum concurrent questions to process"
        ))

        self.add_rule(ConfigRule(
            key="FORECAST_TIMEOUT_SECONDS",
            config_type=ConfigType.INTEGER,
            required=False,
            default_value=300,
            min_value=30,
            max_value=3600,
            description="Forecast timeout in seconds"
        ))

    def add_rule(self, rule: ConfigRule) -> None:
        """Add a configuration validation rule."""
        self.rules[rule.key] = rule
        self.logger.debug(f"Added config rule for {rule.key}")

    def remove_rule(self, key: str) -> None:
        """Remove a configuration validation rule."""
        if key in self.rules:
            del self.rules[key]
            self.logger.debug(f"Removed config rule for {key}")

    def validate_config(self, config_data: Optional[Dict[str, Any]] = None) -> Dict[str, ConfigValidationResult]:
        """Validate all configuration values."""
        results = {}

        # Use provided config data or load from environment/file
        if config_data is None:
            config_data = self._load_config()

        for key, rule in self.rules.items():
            value = config_data.get(key)
            results[key] = self.validate_value(key, value, rule)

        return results

    def validate_value(self, key: str, value: Any, rule: ConfigRule) -> ConfigValidationResult:
        """Validate a single configuration value."""
        errors = []
        warnings = []

        try:
            # Check if required value is present
            if rule.required and (value is None or value == ""):
                errors.append(f"Required configuration '{key}' is missing")
                return ConfigValidationResult(False, rule.default_value, errors, warnings)

            # Use default value if not provided
            if value is None or value == "":
                if rule.default_value is not None:
                    value = rule.default_value
                else:
                    return ConfigValidationResult(True, None, errors, warnings)

            # Type-specific validation
            validated_value = self._validate_by_type(key, value, rule, errors, warnings)

            # Additional validations
            self._validate_constraints(key, validated_value, rule, errors, warnings)

            # Custom validator
            if rule.validator and not rule.validator(validated_value):
                errors.append(f"Configuration '{key}' failed custom validation")

            is_valid = len(errors) == 0
            return ConfigValidationResult(is_valid, validated_value, errors, warnings)

        except Exception as e:
            self.logger.error(f"Validation error for config '{key}': {e}")
            errors.append(f"Validation failed for '{key}': {str(e)}")
            return ConfigValidationResult(False, value, errors, warnings)

    def _validate_by_type(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> Any:
        """Validate value by its type."""
        if rule.config_type == ConfigType.STRING:
            return self._validate_string(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.INTEGER:
            return self._validate_integer(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.FLOAT:
            return self._validate_float(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.BOOLEAN:
            return self._validate_boolean(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.JSON:
            return self._validate_json(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.URL:
            return self._validate_url(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.EMAIL:
            return self._validate_email(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.PATH:
            return self._validate_path(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.IP_ADDRESS:
            return self._validate_ip_address(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.PORT:
            return self._validate_port(key, value, rule, errors, warnings)
        elif rule.config_type == ConfigType.SECRET:
            return self._validate_secret(key, value, rule, errors, warnings)
        else:
            warnings.append(f"Unknown config type for '{key}': {rule.config_type}")
            return value

    def _validate_string(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> str:
        """Validate string value."""
        if not isinstance(value, str):
            try:
                value = str(value)
            except Exception:
                errors.append(f"Cannot convert '{key}' to string")
                return value

        return value

    def _validate_integer(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> int:
        """Validate integer value."""
        try:
            if isinstance(value, str):
                value = int(value)
            elif isinstance(value, float):
                if value.is_integer():
                    value = int(value)
                else:
                    errors.append(f"Configuration '{key}' must be an integer")
                    return value
            elif not isinstance(value, int):
                errors.append(f"Configuration '{key}' must be an integer")
                return value
        except ValueError:
            errors.append(f"Configuration '{key}' must be a valid integer")
            return value

        return value

    def _validate_float(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> float:
        """Validate float value."""
        try:
            value = float(value)
        except (ValueError, TypeError):
            errors.append(f"Configuration '{key}' must be a valid number")
            return value

        return value

    def _validate_boolean(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> bool:
        """Validate boolean value."""
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ['true', '1', 'yes', 'on', 'enabled']:
                return True
            elif value_lower in ['false', '0', 'no', 'off', 'disabled']:
                return False

        if isinstance(value, (int, float)):
            return bool(value)

        errors.append(f"Configuration '{key}' must be a valid boolean")
        return value

    def _validate_json(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> Any:
        """Validate JSON value."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                errors.append(f"Configuration '{key}' must be valid JSON")
                return value

        return value

    def _validate_url(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> str:
        """Validate URL value."""
        if not isinstance(value, str):
            errors.append(f"Configuration '{key}' must be a string")
            return value

        from urllib.parse import urlparse
        try:
            parsed = urlparse(value)
            if not parsed.scheme or not parsed.netloc:
                errors.append(f"Configuration '{key}' must be a valid URL")
        except Exception:
            errors.append(f"Configuration '{key}' must be a valid URL")

        return value

    def _validate_email(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> str:
        """Validate email value."""
        if not isinstance(value, str):
            errors.append(f"Configuration '{key}' must be a string")
            return value

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            errors.append(f"Configuration '{key}' must be a valid email address")

        return value

    def _validate_path(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> str:
        """Validate path value."""
        if not isinstance(value, str):
            errors.append(f"Configuration '{key}' must be a string")
            return value

        # Check for path traversal attempts
        if '..' in value or value.startswith('/'):
            warnings.append(f"Configuration '{key}' contains potentially unsafe path characters")

        return value

    def _validate_ip_address(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> str:
        """Validate IP address value."""
        if not isinstance(value, str):
            errors.append(f"Configuration '{key}' must be a string")
            return value

        try:
            import ipaddress
            ipaddress.ip_address(value)
        except ValueError:
            errors.append(f"Configuration '{key}' must be a valid IP address")

        return value

    def _validate_port(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> int:
        """Validate port value."""
        try:
            port = int(value)
            if not 1 <= port <= 65535:
                errors.append(f"Configuration '{key}' must be a valid port number (1-65535)")
            return port
        except (ValueError, TypeError):
            errors.append(f"Configuration '{key}' must be a valid port number")
            return value

    def _validate_secret(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> str:
        """Validate secret value."""
        if not isinstance(value, str):
            errors.append(f"Configuration '{key}' must be a string")
            return value

        # Check for common weak secrets
        weak_secrets = ['password', '123456', 'secret', 'admin', 'default']
        if value.lower() in weak_secrets:
            warnings.append(f"Configuration '{key}' appears to be a weak secret")

        return value

    def _validate_constraints(self, key: str, value: Any, rule: ConfigRule, errors: List[str], warnings: List[str]) -> None:
        """Validate additional constraints."""
        # Length constraints
        if isinstance(value, str):
            if rule.min_length is not None and len(value) < rule.min_length:
                errors.append(f"Configuration '{key}' must be at least {rule.min_length} characters")

            if rule.max_length is not None and len(value) > rule.max_length:
                errors.append(f"Configuration '{key}' must be at most {rule.max_length} characters")

        # Value constraints
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                errors.append(f"Configuration '{key}' must be at least {rule.min_value}")

            if rule.max_value is not None and value > rule.max_value:
                errors.append(f"Configuration '{key}' must be at most {rule.max_value}")

        # Pattern validation
        if rule.pattern and isinstance(value, str):
            if not re.match(rule.pattern, value):
                errors.append(f"Configuration '{key}' does not match required pattern")

        # Allowed values
        if rule.allowed_values and value not in rule.allowed_values:
            errors.append(f"Configuration '{key}' must be one of: {rule.allowed_values}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables and config file."""
        config = {}

        # Load from environment variables
        for key in self.rules.keys():
            env_value = os.getenv(key)
            if env_value is not None:
                config[key] = env_value

        # Load from config file if specified
        if self.config_file_path and os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    if self.config_file_path.endswith('.json'):
                        file_config = json.load(f)
                    else:
                        # Simple key=value format
                        file_config = {}
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                if '=' in line:
                                    key, value = line.split('=', 1)
                                    file_config[key.strip()] = value.strip()

                # File config overrides environment
                config.update(file_config)

            except Exception as e:
                self.logger.error(f"Failed to load config file {self.config_file_path}: {e}")

        return config

    def get_validated_config(self) -> Dict[str, Any]:
        """Get validated configuration values."""
        if not self._config_cache:
            results = self.validate_config()

            # Check for validation errors
            errors = []
            for key, result in results.items():
                if not result.is_valid:
                    errors.extend([f"{key}: {error}" for error in result.errors])
                else:
                    self._config_cache[key] = result.value

            if errors:
                raise ConfigurationError(
                    f"Configuration validation failed: {'; '.join(errors)}",
                    error_code="CONFIG_VALIDATION_FAILED",
                    context={"validation_errors": errors}
                )

        return self._config_cache.copy()

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a single validated configuration value."""
        config = self.get_validated_config()
        return config.get(key, default)

    def is_sensitive_key(self, key: str) -> bool:
        """Check if a configuration key is sensitive."""
        if key in self.rules and self.rules[key].sensitive:
            return True

        key_lower = key.lower()
        return any(re.match(pattern, key_lower) for pattern in self.sensitive_patterns)

    def mask_sensitive_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive configuration values for logging."""
        masked_config = {}

        for key, value in config.items():
            if self.is_sensitive_key(key):
                if isinstance(value, str) and len(value) > 8:
                    masked_config[key] = f"{value[:4]}***{value[-4:]}"
                else:
                    masked_config[key] = "***MASKED***"
            else:
                masked_config[key] = value

        return masked_config

    def validate_runtime_config_change(self, key: str, new_value: Any) -> ConfigValidationResult:
        """Validate a runtime configuration change."""
        if key not in self.rules:
            return ConfigValidationResult(
                False,
                new_value,
                [f"Unknown configuration key: {key}"],
                []
            )

        rule = self.rules[key]
        result = self.validate_value(key, new_value, rule)

        if result.is_valid:
            # Update cache
            self._config_cache[key] = result.value
            self.logger.info(f"Configuration '{key}' updated successfully")

        return result

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring."""
        config = self.get_validated_config()
        masked_config = self.mask_sensitive_config(config)

        return {
            'total_config_keys': len(config),
            'required_keys_count': sum(1 for rule in self.rules.values() if rule.required),
            'optional_keys_count': sum(1 for rule in self.rules.values() if not rule.required),
            'sensitive_keys_count': sum(1 for key in config.keys() if self.is_sensitive_key(key)),
            'config_values': masked_config
        }
