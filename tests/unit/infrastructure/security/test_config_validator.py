"""Unit tests for ConfigValidator."""

import pytest
import tempfile
import os
import json
from unittest.mock import patch

from src.infrastructure.security.config_validator import (
    ConfigValidator,
    ConfigRule,
    ConfigType,
    ConfigValidationResult
)
from src.domain.exceptions.infrastructure_exceptions import ConfigurationError


class TestConfigRule:
    """Test ConfigRule data structure."""

    def test_config_rule_creation(self):
        """Test creating a config rule."""
        rule = ConfigRule(
            key="TEST_KEY",
            config_type=ConfigType.STRING,
            required=True,
            min_length=5,
            max_length=50,
            description="Test configuration key"
        )

        assert rule.key == "TEST_KEY"
        assert rule.config_type == ConfigType.STRING
        assert rule.required is True
        assert rule.min_length == 5
        assert rule.max_length == 50
        assert rule.description == "Test configuration key"


class TestConfigValidator:
    """Test ConfigValidator functionality."""

    @pytest.fixture
    def validator(self):
        return ConfigValidator()

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            config_data = {
                "TEST_STRING": "test_value",
                "TEST_INTEGER": 42,
                "TEST_BOOLEAN": True
            }
            json.dump(config_data, f)
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_add_rule(self, validator):
        """Test adding a configuration rule."""
        rule = ConfigRule(
            key="TEST_KEY",
            config_type=ConfigType.STRING,
            required=True
        )

        validator.add_rule(rule)

        assert "TEST_KEY" in validator.rules
        assert validator.rules["TEST_KEY"] == rule

    def test_remove_rule(self, validator):
        """Test removing a configuration rule."""
        rule = ConfigRule(
            key="TEST_KEY",
            config_type=ConfigType.STRING
        )

        validator.add_rule(rule)
        validator.remove_rule("TEST_KEY")

        assert "TEST_KEY" not in validator.rules

    def test_validate_string_success(self, validator):
        """Test successful string validation."""
        rule = ConfigRule(
            key="TEST_STRING",
            config_type=ConfigType.STRING,
            min_length=3,
            max_length=10
        )

        result = validator.validate_value("TEST_STRING", "hello", rule)

        assert result.is_valid
        assert result.value == "hello"
        assert len(result.errors) == 0

    def test_validate_string_too_short(self, validator):
        """Test string validation with minimum length violation."""
        rule = ConfigRule(
            key="TEST_STRING",
            config_type=ConfigType.STRING,
            min_length=10
        )

        result = validator.validate_value("TEST_STRING", "short", rule)

        assert not result.is_valid
        assert "must be at least 10 characters" in result.errors[0]

    def test_validate_string_too_long(self, validator):
        """Test string validation with maximum length violation."""
        rule = ConfigRule(
            key="TEST_STRING",
            config_type=ConfigType.STRING,
            max_length=5
        )

        result = validator.validate_value("TEST_STRING", "too long string", rule)

        assert not result.is_valid
        assert "must be at most 5 characters" in result.errors[0]

    def test_validate_integer_success(self, validator):
        """Test successful integer validation."""
        rule = ConfigRule(
            key="TEST_INTEGER",
            config_type=ConfigType.INTEGER,
            min_value=1,
            max_value=100
        )

        result = validator.validate_value("TEST_INTEGER", "42", rule)

        assert result.is_valid
        assert result.value == 42

    def test_validate_integer_from_float(self, validator):
        """Test integer validation from float value."""
        rule = ConfigRule(
            key="TEST_INTEGER",
            config_type=ConfigType.INTEGER
        )

        result = validator.validate_value("TEST_INTEGER", 42.0, rule)

        assert result.is_valid
        assert result.value == 42

    def test_validate_integer_invalid_float(self, validator):
        """Test integer validation with non-integer float."""
        rule = ConfigRule(
            key="TEST_INTEGER",
            config_type=ConfigType.INTEGER
        )

        result = validator.validate_value("TEST_INTEGER", 42.5, rule)

        assert not result.is_valid
        assert "must be an integer" in result.errors[0]

    def test_validate_integer_out_of_range(self, validator):
        """Test integer validation with range violation."""
        rule = ConfigRule(
            key="TEST_INTEGER",
            config_type=ConfigType.INTEGER,
            min_value=1,
            max_value=10
        )

        result = validator.validate_value("TEST_INTEGER", "15", rule)

        assert not result.is_valid
        assert "must be at most 10" in result.errors[0]

    def test_validate_float_success(self, validator):
        """Test successful float validation."""
        rule = ConfigRule(
            key="TEST_FLOAT",
            config_type=ConfigType.FLOAT,
            min_value=0.0,
            max_value=1.0
        )

        result = validator.validate_value("TEST_FLOAT", "0.5", rule)

        assert result.is_valid
        assert result.value == 0.5

    def test_validate_float_invalid(self, validator):
        """Test invalid float validation."""
        rule = ConfigRule(
            key="TEST_FLOAT",
            config_type=ConfigType.FLOAT
        )

        result = validator.validate_value("TEST_FLOAT", "not_a_number", rule)

        assert not result.is_valid
        assert "must be a valid number" in result.errors[0]

    def test_validate_boolean_success(self, validator):
        """Test successful boolean validation."""
        rule = ConfigRule(
            key="TEST_BOOLEAN",
            config_type=ConfigType.BOOLEAN
        )

        test_cases = [
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            ("enabled", True),
            ("disabled", False),
            (True, True),
            (False, False),
            (1, True),
            (0, False)
        ]

        for input_val, expected in test_cases:
            result = validator.validate_value("TEST_BOOLEAN", input_val, rule)
            assert result.is_valid
            assert result.value == expected

    def test_validate_boolean_invalid(self, validator):
        """Test invalid boolean validation."""
        rule = ConfigRule(
            key="TEST_BOOLEAN",
            config_type=ConfigType.BOOLEAN
        )

        result = validator.validate_value("TEST_BOOLEAN", "maybe", rule)

        assert not result.is_valid
        assert "must be a valid boolean" in result.errors[0]

    def test_validate_json_success(self, validator):
        """Test successful JSON validation."""
        rule = ConfigRule(
            key="TEST_JSON",
            config_type=ConfigType.JSON
        )

        json_string = '{"key": "value", "number": 42}'
        result = validator.validate_value("TEST_JSON", json_string, rule)

        assert result.is_valid
        assert result.value == {"key": "value", "number": 42}

    def test_validate_json_invalid(self, validator):
        """Test invalid JSON validation."""
        rule = ConfigRule(
            key="TEST_JSON",
            config_type=ConfigType.JSON
        )

        result = validator.validate_value("TEST_JSON", "invalid json", rule)

        assert not result.is_valid
        assert "must be valid JSON" in result.errors[0]

    def test_validate_url_success(self, validator):
        """Test successful URL validation."""
        rule = ConfigRule(
            key="TEST_URL",
            config_type=ConfigType.URL
        )

        result = validator.validate_value("TEST_URL", "https://example.com", rule)

        assert result.is_valid
        assert result.value == "https://example.com"

    def test_validate_url_invalid(self, validator):
        """Test invalid URL validation."""
        rule = ConfigRule(
            key="TEST_URL",
            config_type=ConfigType.URL
        )

        result = validator.validate_value("TEST_URL", "not-a-url", rule)

        assert not result.is_valid
        assert "must be a valid URL" in result.errors[0]

    def test_validate_email_success(self, validator):
        """Test successful email validation."""
        rule = ConfigRule(
            key="TEST_EMAIL",
            config_type=ConfigType.EMAIL
        )

        result = validator.validate_value("TEST_EMAIL", "test@example.com", rule)

        assert result.is_valid
        assert result.value == "test@example.com"

    def test_validate_email_invalid(self, validator):
        """Test invalid email validation."""
        rule = ConfigRule(
            key="TEST_EMAIL",
            config_type=ConfigType.EMAIL
        )

        result = validator.validate_value("TEST_EMAIL", "invalid-email", rule)

        assert not result.is_valid
        assert "must be a valid email address" in result.errors[0]

    def test_validate_path_success(self, validator):
        """Test successful path validation."""
        rule = ConfigRule(
            key="TEST_PATH",
            config_type=ConfigType.PATH
        )

        result = validator.validate_value("TEST_PATH", "logs/audit.log", rule)

        assert result.is_valid
        assert result.value == "logs/audit.log"

    def test_validate_path_unsafe(self, validator):
        """Test path validation with unsafe characters."""
        rule = ConfigRule(
            key="TEST_PATH",
            config_type=ConfigType.PATH
        )

        result = validator.validate_value("TEST_PATH", "../../../etc/passwd", rule)

        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) > 0
        assert "potentially unsafe path characters" in result.warnings[0]

    def test_validate_ip_address_success(self, validator):
        """Test successful IP address validation."""
        rule = ConfigRule(
            key="TEST_IP",
            config_type=ConfigType.IP_ADDRESS
        )

        valid_ips = ["192.168.1.1", "10.0.0.1", "127.0.0.1", "::1", "2001:db8::1"]
        for ip in valid_ips:
            result = validator.validate_value("TEST_IP", ip, rule)
            assert result.is_valid, f"IP {ip} should be valid"

    def test_validate_ip_address_invalid(self, validator):
        """Test invalid IP address validation."""
        rule = ConfigRule(
            key="TEST_IP",
            config_type=ConfigType.IP_ADDRESS
        )

        invalid_ips = ["256.256.256.256", "not.an.ip", "192.168.1"]
        for ip in invalid_ips:
            result = validator.validate_value("TEST_IP", ip, rule)
            assert not result.is_valid, f"IP {ip} should be invalid"

    def test_validate_port_success(self, validator):
        """Test successful port validation."""
        rule = ConfigRule(
            key="TEST_PORT",
            config_type=ConfigType.PORT
        )

        result = validator.validate_value("TEST_PORT", "8080", rule)

        assert result.is_valid
        assert result.value == 8080

    def test_validate_port_out_of_range(self, validator):
        """Test port validation with out of range value."""
        rule = ConfigRule(
            key="TEST_PORT",
            config_type=ConfigType.PORT
        )

        result = validator.validate_value("TEST_PORT", "70000", rule)

        assert not result.is_valid
        assert "must be a valid port number (1-65535)" in result.errors[0]

    def test_validate_secret_success(self, validator):
        """Test successful secret validation."""
        rule = ConfigRule(
            key="TEST_SECRET",
            config_type=ConfigType.SECRET,
            min_length=8
        )

        result = validator.validate_value("TEST_SECRET", "strong_secret_123", rule)

        assert result.is_valid
        assert result.value == "strong_secret_123"

    def test_validate_secret_weak(self, validator):
        """Test secret validation with weak secret."""
        rule = ConfigRule(
            key="TEST_SECRET",
            config_type=ConfigType.SECRET
        )

        result = validator.validate_value("TEST_SECRET", "password", rule)

        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) > 0
        assert "appears to be a weak secret" in result.warnings[0]

    def test_validate_required_field_missing(self, validator):
        """Test required field validation when missing."""
        rule = ConfigRule(
            key="REQUIRED_FIELD",
            config_type=ConfigType.STRING,
            required=True
        )

        result = validator.validate_value("REQUIRED_FIELD", None, rule)

        assert not result.is_valid
        assert "is missing" in result.errors[0]

    def test_validate_optional_field_missing(self, validator):
        """Test optional field validation when missing."""
        rule = ConfigRule(
            key="OPTIONAL_FIELD",
            config_type=ConfigType.STRING,
            required=False,
            default_value="default"
        )

        result = validator.validate_value("OPTIONAL_FIELD", None, rule)

        assert result.is_valid
        assert result.value == "default"

    def test_validate_allowed_values_success(self, validator):
        """Test validation with allowed values constraint."""
        rule = ConfigRule(
            key="ENVIRONMENT",
            config_type=ConfigType.STRING,
            allowed_values=["development", "staging", "production"]
        )

        result = validator.validate_value("ENVIRONMENT", "production", rule)

        assert result.is_valid
        assert result.value == "production"

    def test_validate_allowed_values_failure(self, validator):
        """Test validation with allowed values constraint violation."""
        rule = ConfigRule(
            key="ENVIRONMENT",
            config_type=ConfigType.STRING,
            allowed_values=["development", "staging", "production"]
        )

        result = validator.validate_value("ENVIRONMENT", "invalid_env", rule)

        assert not result.is_valid
        assert "must be one of" in result.errors[0]

    def test_validate_pattern_success(self, validator):
        """Test pattern validation success."""
        rule = ConfigRule(
            key="VERSION",
            config_type=ConfigType.STRING,
            pattern=r'^\d+\.\d+\.\d+$'
        )

        result = validator.validate_value("VERSION", "1.2.3", rule)

        assert result.is_valid

    def test_validate_pattern_failure(self, validator):
        """Test pattern validation failure."""
        rule = ConfigRule(
            key="VERSION",
            config_type=ConfigType.STRING,
            pattern=r'^\d+\.\d+\.\d+$'
        )

        result = validator.validate_value("VERSION", "invalid-version", rule)

        assert not result.is_valid
        assert "does not match required pattern" in result.errors[0]

    def test_custom_validator_success(self, validator):
        """Test custom validator success."""
        def custom_check(value):
            return len(value) % 2 == 0  # Even length only

        rule = ConfigRule(
            key="EVEN_LENGTH",
            config_type=ConfigType.STRING,
            validator=custom_check
        )

        result = validator.validate_value("EVEN_LENGTH", "test", rule)  # 4 characters

        assert result.is_valid

    def test_custom_validator_failure(self, validator):
        """Test custom validator failure."""
        def custom_check(value):
            return len(value) % 2 == 0  # Even length only

        rule = ConfigRule(
            key="EVEN_LENGTH",
            config_type=ConfigType.STRING,
            validator=custom_check
        )

        result = validator.validate_value("EVEN_LENGTH", "test1", rule)  # 5 characters

        assert not result.is_valid
        assert "failed custom validation" in result.errors[0]

    def test_validate_config_from_environment(self, validator):
        """Test validating configuration from environment variables."""
        # Add test rules
        validator.add_rule(ConfigRule(
            key="TEST_STRING",
            config_type=ConfigType.STRING,
            required=True
        ))
        validator.add_rule(ConfigRule(
            key="TEST_INTEGER",
            config_type=ConfigType.INTEGER,
            required=False,
            default_value=42
        ))

        with patch.dict('os.environ', {'TEST_STRING': 'env_value'}):
            results = validator.validate_config()

            assert results["TEST_STRING"].is_valid
            assert results["TEST_STRING"].value == "env_value"
            assert results["TEST_INTEGER"].is_valid
            assert results["TEST_INTEGER"].value == 42  # Default value

    def test_validate_config_from_file(self, validator, temp_config_file):
        """Test validating configuration from file."""
        # Add test rules
        validator.add_rule(ConfigRule(
            key="TEST_STRING",
            config_type=ConfigType.STRING
        ))
        validator.add_rule(ConfigRule(
            key="TEST_INTEGER",
            config_type=ConfigType.INTEGER
        ))

        # Create validator with config file
        file_validator = ConfigValidator(config_file_path=temp_config_file)
        file_validator.add_rule(ConfigRule(
            key="TEST_STRING",
            config_type=ConfigType.STRING
        ))
        file_validator.add_rule(ConfigRule(
            key="TEST_INTEGER",
            config_type=ConfigType.INTEGER
        ))

        results = file_validator.validate_config()

        assert results["TEST_STRING"].is_valid
        assert results["TEST_STRING"].value == "test_value"
        assert results["TEST_INTEGER"].is_valid
        assert results["TEST_INTEGER"].value == 42

    def test_get_validated_config_success(self, validator):
        """Test getting validated configuration."""
        validator.add_rule(ConfigRule(
            key="TEST_KEY",
            config_type=ConfigType.STRING,
            required=False,
            default_value="default_value"
        ))

        config = validator.get_validated_config()

        assert "TEST_KEY" in config
        assert config["TEST_KEY"] == "default_value"

    def test_get_validated_config_with_errors(self, validator):
        """Test getting validated configuration with validation errors."""
        validator.add_rule(ConfigRule(
            key="REQUIRED_KEY",
            config_type=ConfigType.STRING,
            required=True
        ))

        with pytest.raises(ConfigurationError) as exc_info:
            validator.get_validated_config()

        assert exc_info.value.error_code == "CONFIG_VALIDATION_FAILED"

    def test_get_config_value(self, validator):
        """Test getting a single configuration value."""
        validator.add_rule(ConfigRule(
            key="TEST_KEY",
            config_type=ConfigType.STRING,
            required=False,
            default_value="test_value"
        ))

        value = validator.get_config_value("TEST_KEY")
        assert value == "test_value"

        # Test with default
        value = validator.get_config_value("NONEXISTENT_KEY", "fallback")
        assert value == "fallback"

    def test_is_sensitive_key(self, validator):
        """Test sensitive key detection."""
        # Test rule-based sensitivity
        validator.add_rule(ConfigRule(
            key="API_KEY",
            config_type=ConfigType.SECRET,
            sensitive=True
        ))

        assert validator.is_sensitive_key("API_KEY")

        # Test pattern-based sensitivity
        assert validator.is_sensitive_key("PASSWORD")
        assert validator.is_sensitive_key("SECRET_TOKEN")
        assert validator.is_sensitive_key("AUTH_KEY")

        # Test non-sensitive key
        assert not validator.is_sensitive_key("DEBUG_MODE")

    def test_mask_sensitive_config(self, validator):
        """Test masking sensitive configuration values."""
        config = {
            "DEBUG": True,
            "API_KEY": "sk-1234567890abcdef",
            "PASSWORD": "secret123",
            "DATABASE_URL": "postgres://user:pass@host/db",
            "LOG_LEVEL": "INFO"
        }

        masked = validator.mask_sensitive_config(config)

        assert masked["DEBUG"] is True
        assert masked["API_KEY"] == "sk-1***cdef"
        assert masked["PASSWORD"] == "***MASKED***"
        assert masked["LOG_LEVEL"] == "INFO"

    def test_validate_runtime_config_change(self, validator):
        """Test runtime configuration change validation."""
        validator.add_rule(ConfigRule(
            key="RUNTIME_KEY",
            config_type=ConfigType.INTEGER,
            min_value=1,
            max_value=100
        ))

        # Valid change
        result = validator.validate_runtime_config_change("RUNTIME_KEY", "50")
        assert result.is_valid
        assert result.value == 50

        # Invalid change
        result = validator.validate_runtime_config_change("RUNTIME_KEY", "150")
        assert not result.is_valid

        # Unknown key
        result = validator.validate_runtime_config_change("UNKNOWN_KEY", "value")
        assert not result.is_valid
        assert "Unknown configuration key" in result.errors[0]

    def test_get_config_summary(self, validator):
        """Test getting configuration summary."""
        validator.add_rule(ConfigRule(
            key="PUBLIC_KEY",
            config_type=ConfigType.STRING,
            required=True,
            default_value="public_value"
        ))
        validator.add_rule(ConfigRule(
            key="SECRET_KEY",
            config_type=ConfigType.SECRET,
            required=False,
            sensitive=True
        ))

        summary = validator.get_config_summary()

        assert summary["total_config_keys"] >= 0
        assert summary["required_keys_count"] >= 0
        assert summary["optional_keys_count"] >= 0
        assert summary["sensitive_keys_count"] >= 0
        assert "config_values" in summary

    def test_default_rules_setup(self, validator):
        """Test that default configuration rules are set up."""
        # Check that some expected default rules exist
        expected_keys = [
            "SECRET_KEY",
            "ENVIRONMENT",
            "DEBUG",
            "LOG_LEVEL",
            "HOST",
            "PORT",
            "REDIS_URL"
        ]

        for key in expected_keys:
            assert key in validator.rules

        # Check specific rule properties
        secret_key_rule = validator.rules["SECRET_KEY"]
        assert secret_key_rule.config_type == ConfigType.SECRET
        assert secret_key_rule.required is True
        assert secret_key_rule.sensitive is True

        env_rule = validator.rules["ENVIRONMENT"]
        assert env_rule.config_type == ConfigType.STRING
        assert env_rule.allowed_values == ["development", "staging", "production"]

    def test_validation_error_handling(self, validator):
        """Test validation error handling for edge cases."""
        rule = ConfigRule(
            key="TEST_KEY",
            config_type=ConfigType.INTEGER
        )

        # Test with object that causes validation error
        class ProblematicObject:
            def __str__(self):
                raise Exception("Conversion error")

        result = validator.validate_value("TEST_KEY", ProblematicObject(), rule)

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_load_config_key_value_format(self, validator):
        """Test loading configuration from key=value format file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
            f.write("# Comment line\n")
            f.write("TEST_KEY=test_value\n")
            f.write("ANOTHER_KEY = another_value \n")
            f.write("\n")  # Empty line
            temp_path = f.name

        try:
            file_validator = ConfigValidator(config_file_path=temp_path)
            config_data = file_validator._load_config()

            assert config_data["TEST_KEY"] == "test_value"
            assert config_data["ANOTHER_KEY"] == "another_value"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.asyncio
async def test_config_validator_integration():
    """Integration test for config validator with environment and file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        config_data = {
            "FILE_KEY": "file_value",
            "OVERRIDE_KEY": "file_override"
        }
        json.dump(config_data, f)
        temp_path = f.name

    try:
        # Set environment variables
        with patch.dict('os.environ', {
            'ENV_KEY': 'env_value',
            'OVERRIDE_KEY': 'env_override'  # Should override file value
        }):
            validator = ConfigValidator(config_file_path=temp_path)

            # Add rules
            validator.add_rule(ConfigRule(
                key="ENV_KEY",
                config_type=ConfigType.STRING
            ))
            validator.add_rule(ConfigRule(
                key="FILE_KEY",
                config_type=ConfigType.STRING
            ))
            validator.add_rule(ConfigRule(
                key="OVERRIDE_KEY",
                config_type=ConfigType.STRING
            ))

            # Validate configuration
            config = validator.get_validated_config()

            # Verify values
            assert config["ENV_KEY"] == "env_value"
            assert config["FILE_KEY"] == "file_value"
            assert config["OVERRIDE_KEY"] == "env_override"  # Environment overrides file

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
