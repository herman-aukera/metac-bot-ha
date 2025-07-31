"""Unit tests for InputValidator."""

import pytest
from unittest.mock import Mock, patch
import json

from src.infrastructure.security.input_validator import (
    InputValidator,
    ValidationRule,
    ValidationType,
    ValidationResult
)
from src.domain.exceptions.infrastructure_exceptions import ValidationError


class TestInputValidator:
    """Test InputValidator functionality."""

    @pytest.fixture
    def validator(self):
        return InputValidator()

    def test_validate_string_success(self, validator):
        """Test successful string validation."""
        rule = ValidationRule(
            field_name="test_field",
            validation_type=ValidationType.STRING,
            min_length=5,
            max_length=20
        )

        result = validator.validate_field("test string", rule)

        assert result.is_valid
        assert result.sanitized_value == "test string"
        assert len(result.errors) == 0

    def test_validate_string_too_short(self, validator):
        """Test string validation with minimum length violation."""
        rule = ValidationRule(
            field_name="test_field",
            validation_type=ValidationType.STRING,
            min_length=10
        )

        result = validator.validate_field("short", rule)

        assert not result.is_valid
        assert "must be at least 10 characters" in result.errors[0]

    def test_validate_string_too_long(self, validator):
        """Test string validation with maximum length violation."""
        rule = ValidationRule(
            field_name="test_field",
            validation_type=ValidationType.STRING,
            max_length=5
        )

        result = validator.validate_field("too long string", rule)

        assert not result.is_valid
        assert "must be at most 5 characters" in result.errors[0]

    def test_validate_integer_success(self, validator):
        """Test successful integer validation."""
        rule = ValidationRule(
            field_name="test_field",
            validation_type=ValidationType.INTEGER,
            min_value=1,
            max_value=100
        )

        result = validator.validate_field("42", rule)

        assert result.is_valid
        assert result.sanitized_value == 42

    def test_validate_integer_out_of_range(self, validator):
        """Test integer validation with range violation."""
        rule = ValidationRule(
            field_name="test_field",
            validation_type=ValidationType.INTEGER,
            min_value=1,
            max_value=10
        )

        result = validator.validate_field("15", rule)

        assert not result.is_valid
        assert "must be at most 10" in result.errors[0]

    def test_validate_float_success(self, validator):
        """Test successful float validation."""
        rule = ValidationRule(
            field_name="test_field",
            validation_type=ValidationType.FLOAT,
            min_value=0.0,
            max_value=1.0
        )

        result = validator.validate_field("0.5", rule)

        assert result.is_valid
        assert result.sanitized_value == 0.5

    def test_validate_boolean_success(self, validator):
        """Test successful boolean validation."""
        rule = ValidationRule(
            field_name="test_field",
            validation_type=ValidationType.BOOLEAN
        )

        # Test various boolean representations
        test_cases = [
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            (True, True),
            (False, False)
        ]

        for input_val, expected in test_cases:
            result = validator.validate_field(input_val, rule)
            assert result.is_valid
            assert result.sanitized_value == expected

    def test_validate_email_success(self, validator):
        """Test successful email validation."""
        rule = ValidationRule(
            field_name="email",
            validation_type=ValidationType.EMAIL
        )

        result = validator.validate_field("test@example.com", rule)

        assert result.is_valid
        assert result.sanitized_value == "test@example.com"

    def test_validate_email_invalid(self, validator):
        """Test invalid email validation."""
        rule = ValidationRule(
            field_name="email",
            validation_type=ValidationType.EMAIL
        )

        result = validator.validate_field("invalid-email", rule)

        assert not result.is_valid
        assert "must be a valid email address" in result.errors[0]

    def test_validate_url_success(self, validator):
        """Test successful URL validation."""
        rule = ValidationRule(
            field_name="url",
            validation_type=ValidationType.URL
        )

        result = validator.validate_field("https://example.com", rule)

        assert result.is_valid
        assert result.sanitized_value == "https://example.com"

    def test_validate_url_invalid(self, validator):
        """Test invalid URL validation."""
        rule = ValidationRule(
            field_name="url",
            validation_type=ValidationType.URL
        )

        result = validator.validate_field("not-a-url", rule)

        assert not result.is_valid
        assert "must be a valid URL" in result.errors[0]

    def test_validate_json_success(self, validator):
        """Test successful JSON validation."""
        rule = ValidationRule(
            field_name="json_data",
            validation_type=ValidationType.JSON
        )

        json_string = '{"key": "value", "number": 42}'
        result = validator.validate_field(json_string, rule)

        assert result.is_valid
        assert result.sanitized_value == {"key": "value", "number": 42}

    def test_validate_json_invalid(self, validator):
        """Test invalid JSON validation."""
        rule = ValidationRule(
            field_name="json_data",
            validation_type=ValidationType.JSON
        )

        result = validator.validate_field("invalid json", rule)

        assert not result.is_valid
        assert "must be valid JSON" in result.errors[0]

    def test_validate_prediction_value_binary(self, validator):
        """Test binary prediction value validation."""
        rule = ValidationRule(
            field_name="prediction",
            validation_type=ValidationType.PREDICTION_VALUE
        )

        result = validator.validate_field(0.75, rule)

        assert result.is_valid
        assert result.sanitized_value == 0.75

    def test_validate_prediction_value_binary_out_of_range(self, validator):
        """Test binary prediction value out of range."""
        rule = ValidationRule(
            field_name="prediction",
            validation_type=ValidationType.PREDICTION_VALUE
        )

        result = validator.validate_field(1.5, rule)

        assert not result.is_valid
        assert "must be between 0.0 and 1.0" in result.errors[0]

    def test_validate_prediction_value_multiple_choice(self, validator):
        """Test multiple choice prediction value validation."""
        rule = ValidationRule(
            field_name="prediction",
            validation_type=ValidationType.PREDICTION_VALUE
        )

        prediction = {"option_a": 0.3, "option_b": 0.7}
        result = validator.validate_field(prediction, rule)

        assert result.is_valid
        assert result.sanitized_value == prediction

    def test_validate_prediction_value_multiple_choice_invalid_sum(self, validator):
        """Test multiple choice prediction with invalid probability sum."""
        rule = ValidationRule(
            field_name="prediction",
            validation_type=ValidationType.PREDICTION_VALUE
        )

        prediction = {"option_a": 0.3, "option_b": 0.5}  # Sum = 0.8, not 1.0
        result = validator.validate_field(prediction, rule)

        assert result.is_valid  # Should still be valid but with warning
        assert len(result.warnings) > 0
        assert "should sum to 1.0" in result.warnings[0]

    def test_validate_question_text_success(self, validator):
        """Test successful question text validation."""
        rule = ValidationRule(
            field_name="question",
            validation_type=ValidationType.QUESTION_TEXT
        )

        result = validator.validate_field("Will AI achieve AGI by 2030?", rule)

        assert result.is_valid
        assert "Will AI achieve AGI by 2030?" in result.sanitized_value

    def test_validate_question_text_too_short(self, validator):
        """Test question text too short."""
        rule = ValidationRule(
            field_name="question",
            validation_type=ValidationType.QUESTION_TEXT
        )

        result = validator.validate_field("Short?", rule)

        assert not result.is_valid
        assert "must be at least 10 characters" in result.errors[0]

    def test_validate_question_text_no_question_mark(self, validator):
        """Test question text without question mark."""
        rule = ValidationRule(
            field_name="question",
            validation_type=ValidationType.QUESTION_TEXT
        )

        result = validator.validate_field("This is a statement about AI", rule)

        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) > 0
        assert "should typically end with a question mark" in result.warnings[0]

    def test_validate_sql_safe_success(self, validator):
        """Test SQL-safe validation success."""
        rule = ValidationRule(
            field_name="safe_text",
            validation_type=ValidationType.SQL_SAFE
        )

        result = validator.validate_field("normal text input", rule)

        assert result.is_valid

    def test_validate_sql_safe_injection_attempt(self, validator):
        """Test SQL injection attempt detection."""
        rule = ValidationRule(
            field_name="unsafe_text",
            validation_type=ValidationType.SQL_SAFE
        )

        result = validator.validate_field("'; DROP TABLE users; --", rule)

        assert not result.is_valid
        assert "contains potentially dangerous SQL patterns" in result.errors[0]

    def test_validate_filename_success(self, validator):
        """Test successful filename validation."""
        rule = ValidationRule(
            field_name="filename",
            validation_type=ValidationType.FILENAME
        )

        result = validator.validate_field("document.txt", rule)

        assert result.is_valid
        assert result.sanitized_value == "document.txt"

    def test_validate_filename_dangerous_extension(self, validator):
        """Test filename with dangerous extension."""
        rule = ValidationRule(
            field_name="filename",
            validation_type=ValidationType.FILENAME
        )

        result = validator.validate_field("malware.exe", rule)

        assert not result.is_valid
        assert "potentially dangerous file extension" in result.errors[0]

    def test_validate_filename_path_traversal(self, validator):
        """Test filename with path traversal attempt."""
        rule = ValidationRule(
            field_name="filename",
            validation_type=ValidationType.FILENAME
        )

        result = validator.validate_field("../../../etc/passwd", rule)

        assert not result.is_valid
        assert "contains invalid path characters" in result.errors[0]

    def test_validate_uuid_success(self, validator):
        """Test successful UUID validation."""
        rule = ValidationRule(
            field_name="uuid",
            validation_type=ValidationType.UUID
        )

        result = validator.validate_field("123e4567-e89b-12d3-a456-426614174000", rule)

        assert result.is_valid
        assert result.sanitized_value == "123e4567-e89b-12d3-a456-426614174000"

    def test_validate_uuid_invalid(self, validator):
        """Test invalid UUID validation."""
        rule = ValidationRule(
            field_name="uuid",
            validation_type=ValidationType.UUID
        )

        result = validator.validate_field("not-a-uuid", rule)

        assert not result.is_valid
        assert "must be a valid UUID" in result.errors[0]

    def test_validate_required_field_missing(self, validator):
        """Test required field validation when missing."""
        rule = ValidationRule(
            field_name="required_field",
            validation_type=ValidationType.STRING,
            required=True
        )

        result = validator.validate_field(None, rule)

        assert not result.is_valid
        assert "is required" in result.errors[0]

    def test_validate_optional_field_missing(self, validator):
        """Test optional field validation when missing."""
        rule = ValidationRule(
            field_name="optional_field",
            validation_type=ValidationType.STRING,
            required=False
        )

        result = validator.validate_field(None, rule)

        assert result.is_valid
        assert result.sanitized_value is None

    def test_validate_allowed_values_success(self, validator):
        """Test validation with allowed values constraint."""
        rule = ValidationRule(
            field_name="status",
            validation_type=ValidationType.STRING,
            allowed_values=["active", "inactive", "pending"]
        )

        result = validator.validate_field("active", rule)

        assert result.is_valid
        assert result.sanitized_value == "active"

    def test_validate_allowed_values_failure(self, validator):
        """Test validation with allowed values constraint violation."""
        rule = ValidationRule(
            field_name="status",
            validation_type=ValidationType.STRING,
            allowed_values=["active", "inactive", "pending"]
        )

        result = validator.validate_field("invalid_status", rule)

        assert not result.is_valid
        assert "must be one of" in result.errors[0]

    def test_validate_pattern_success(self, validator):
        """Test pattern validation success."""
        rule = ValidationRule(
            field_name="code",
            validation_type=ValidationType.STRING,
            pattern=r'^[A-Z]{3}-\d{3}$'
        )

        result = validator.validate_field("ABC-123", rule)

        assert result.is_valid

    def test_validate_pattern_failure(self, validator):
        """Test pattern validation failure."""
        rule = ValidationRule(
            field_name="code",
            validation_type=ValidationType.STRING,
            pattern=r'^[A-Z]{3}-\d{3}$'
        )

        result = validator.validate_field("invalid-pattern", rule)

        assert not result.is_valid
        assert "does not match required pattern" in result.errors[0]

    def test_custom_validator_success(self, validator):
        """Test custom validator success."""
        def custom_check(value):
            return len(value) % 2 == 0  # Even length only

        rule = ValidationRule(
            field_name="even_length",
            validation_type=ValidationType.STRING,
            custom_validator=custom_check
        )

        result = validator.validate_field("test", rule)  # 4 characters, even

        assert result.is_valid

    def test_custom_validator_failure(self, validator):
        """Test custom validator failure."""
        def custom_check(value):
            return len(value) % 2 == 0  # Even length only

        rule = ValidationRule(
            field_name="even_length",
            validation_type=ValidationType.STRING,
            custom_validator=custom_check
        )

        result = validator.validate_field("test1", rule)  # 5 characters, odd

        assert not result.is_valid
        assert "failed custom validation" in result.errors[0]

    def test_validate_data_multiple_fields(self, validator):
        """Test validation of multiple fields."""
        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": "30",
            "active": "true"
        }

        rules = [
            ValidationRule("name", ValidationType.STRING, min_length=2),
            ValidationRule("email", ValidationType.EMAIL),
            ValidationRule("age", ValidationType.INTEGER, min_value=0, max_value=120),
            ValidationRule("active", ValidationType.BOOLEAN)
        ]

        results = validator.validate_data(data, rules)

        assert all(result.is_valid for result in results.values())
        assert results["name"].sanitized_value == "John Doe"
        assert results["email"].sanitized_value == "john@example.com"
        assert results["age"].sanitized_value == 30
        assert results["active"].sanitized_value is True

    def test_detect_xss_attempt(self, validator):
        """Test XSS attempt detection."""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(1)'></iframe>"
        ]

        for attempt in xss_attempts:
            assert validator.detect_xss_attempt(attempt)

        # Safe content should not trigger
        safe_content = "This is normal text content"
        assert not validator.detect_xss_attempt(safe_content)

    def test_detect_sql_injection_attempt(self, validator):
        """Test SQL injection attempt detection."""
        sql_attempts = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "admin'--",
            "1; DELETE FROM users"
        ]

        for attempt in sql_attempts:
            assert validator.detect_sql_injection_attempt(attempt)

        # Safe content should not trigger
        safe_content = "normal search query"
        assert not validator.detect_sql_injection_attempt(safe_content)

    def test_sanitize_for_logging(self, validator):
        """Test data sanitization for logging."""
        data = {
            "username": "testuser",
            "password": "secret123",
            "api_key": "sk-1234567890",
            "public_info": "visible data",
            "nested": {
                "token": "bearer_token_123",
                "safe_data": "also visible"
            }
        }

        sanitized = validator.sanitize_for_logging(data)

        assert sanitized["username"] == "testuser"
        assert sanitized["password"] == "***MASKED***"
        assert sanitized["api_key"] == "***MASKED***"
        assert sanitized["public_info"] == "visible data"
        assert sanitized["nested"]["token"] == "***MASKED***"
        assert sanitized["nested"]["safe_data"] == "also visible"

    def test_create_validation_rules_for_question(self, validator):
        """Test creation of validation rules for question data."""
        rules = validator.create_validation_rules_for_question()

        rule_names = [rule.field_name for rule in rules]
        assert "title" in rule_names
        assert "description" in rule_names
        assert "question_type" in rule_names
        assert "close_time" in rule_names

        # Check specific rule properties
        title_rule = next(rule for rule in rules if rule.field_name == "title")
        assert title_rule.validation_type == ValidationType.STRING
        assert title_rule.required is True
        assert title_rule.min_length == 5
        assert title_rule.max_length == 500

    def test_create_validation_rules_for_prediction(self, validator):
        """Test creation of validation rules for prediction data."""
        rules = validator.create_validation_rules_for_prediction()

        rule_names = [rule.field_name for rule in rules]
        assert "question_id" in rule_names
        assert "prediction_value" in rule_names
        assert "confidence" in rule_names

        # Check specific rule properties
        confidence_rule = next(rule for rule in rules if rule.field_name == "confidence")
        assert confidence_rule.validation_type == ValidationType.FLOAT
        assert confidence_rule.min_value == 0.0
        assert confidence_rule.max_value == 1.0

    def test_html_sanitization(self, validator):
        """Test HTML content sanitization."""
        rule = ValidationRule(
            field_name="html_content",
            validation_type=ValidationType.HTML,
            sanitize=True
        )

        html_content = "<p>Safe content</p><script>alert('xss')</script><strong>Bold text</strong>"
        result = validator.validate_field(html_content, rule)

        assert result.is_valid
        # Script tag should be removed, safe tags should remain
        assert "<script>" not in result.sanitized_value
        assert "<p>" in result.sanitized_value
        assert "<strong>" in result.sanitized_value

    def test_ip_address_validation(self, validator):
        """Test IP address validation."""
        rule = ValidationRule(
            field_name="ip",
            validation_type=ValidationType.IP_ADDRESS
        )

        # Valid IP addresses
        valid_ips = ["192.168.1.1", "10.0.0.1", "127.0.0.1", "::1", "2001:db8::1"]
        for ip in valid_ips:
            result = validator.validate_field(ip, rule)
            assert result.is_valid, f"IP {ip} should be valid"

        # Invalid IP addresses
        invalid_ips = ["256.256.256.256", "not.an.ip", "192.168.1", ""]
        for ip in invalid_ips:
            result = validator.validate_field(ip, rule)
            assert not result.is_valid, f"IP {ip} should be invalid"

    def test_validation_error_handling(self, validator):
        """Test validation error handling for edge cases."""
        rule = ValidationRule(
            field_name="test_field",
            validation_type=ValidationType.INTEGER
        )

        # Test with object that can't be converted
        class UnconvertibleObject:
            def __str__(self):
                raise Exception("Cannot convert")

        result = validator.validate_field(UnconvertibleObject(), rule)

        assert not result.is_valid
        assert len(result.errors) > 0
