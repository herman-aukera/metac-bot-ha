"""
Unit tests for base exception classes.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from src.domain.exceptions.base_exceptions import (
    ErrorContext,
    TournamentOptimizationError,
    DomainError,
    ValidationError,
    BusinessRuleViolationError,
    ConfigurationError,
    ResourceError,
)


class TestErrorContext:
    """Test ErrorContext functionality."""

    def test_error_context_creation(self):
        """Test ErrorContext creation with defaults."""
        context = ErrorContext()

        assert context.correlation_id is not None
        assert len(context.correlation_id) == 36  # UUID length
        assert isinstance(context.timestamp, datetime)
        assert context.component is None
        assert context.operation is None
        assert context.metadata == {}

    def test_error_context_with_values(self):
        """Test ErrorContext creation with specific values."""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            user_id="user123",
            metadata={"key": "value"}
        )

        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.user_id == "user123"
        assert context.metadata == {"key": "value"}

    def test_error_context_to_dict(self):
        """Test ErrorContext to_dict conversion."""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            metadata={"key": "value"}
        )

        result = context.to_dict()

        assert result["component"] == "test_component"
        assert result["operation"] == "test_operation"
        assert result["metadata"] == {"key": "value"}
        assert "correlation_id" in result
        assert "timestamp" in result


class TestTournamentOptimizationError:
    """Test TournamentOptimizationError functionality."""

    def test_basic_error_creation(self):
        """Test basic error creation."""
        error = TournamentOptimizationError("Test error message")

        assert str(error) == f"TournamentOptimizationError: Test error message (correlation_id: {error.context.correlation_id})"
        assert error.message == "Test error message"
        assert error.error_code == "TournamentOptimizationError"
        assert error.recoverable is False
        assert error.retry_after is None

    def test_error_with_context(self):
        """Test error creation with custom context."""
        context = ErrorContext(component="test_component")
        error = TournamentOptimizationError(
            "Test error",
            error_code="TEST_ERROR",
            context=context,
            recoverable=True,
            retry_after=60
        )

        assert error.error_code == "TEST_ERROR"
        assert error.context.component == "test_component"
        assert error.recoverable is True
        assert error.retry_after == 60

    def test_error_with_cause(self):
        """Test error creation with cause."""
        cause = ValueError("Original error")
        error = TournamentOptimizationError("Wrapper error", cause=cause)

        assert error.cause == cause

    def test_error_to_dict(self):
        """Test error to_dict conversion."""
        error = TournamentOptimizationError(
            "Test error",
            error_code="TEST_ERROR",
            recoverable=True,
            retry_after=30
        )

        result = error.to_dict()

        assert result["error_type"] == "TournamentOptimizationError"
        assert result["message"] == "Test error"
        assert result["error_code"] == "TEST_ERROR"
        assert result["recoverable"] is True
        assert result["retry_after"] == 30
        assert "context" in result

    def test_error_with_kwargs(self):
        """Test error creation with additional kwargs."""
        error = TournamentOptimizationError(
            "Test error",
            custom_field="custom_value",
            numeric_field=42
        )

        assert error.context.metadata["custom_field"] == "custom_value"
        assert error.context.metadata["numeric_field"] == 42


class TestDomainError:
    """Test DomainError functionality."""

    def test_domain_error_creation(self):
        """Test DomainError sets component correctly."""
        error = DomainError("Domain error message")

        assert error.context.component == "domain"
        assert isinstance(error, TournamentOptimizationError)


class TestValidationError:
    """Test ValidationError functionality."""

    def test_validation_error_creation(self):
        """Test ValidationError creation."""
        field_errors = {
            "field1": ["Error 1", "Error 2"],
            "field2": ["Error 3"]
        }

        error = ValidationError(
            "Validation failed",
            field_errors=field_errors
        )

        assert error.field_errors == field_errors
        assert error.context.metadata["field_errors"] == field_errors
        assert error.context.operation == "validation"

    def test_validation_error_without_field_errors(self):
        """Test ValidationError creation without field errors."""
        error = ValidationError("Validation failed")

        assert error.field_errors == {}
        assert error.context.metadata["field_errors"] == {}


class TestBusinessRuleViolationError:
    """Test BusinessRuleViolationError functionality."""

    def test_business_rule_violation_creation(self):
        """Test BusinessRuleViolationError creation."""
        rule_context = {"entity_id": "123", "current_value": 100}

        error = BusinessRuleViolationError(
            "Business rule violated",
            rule_name="max_value_rule",
            rule_context=rule_context
        )

        assert error.rule_name == "max_value_rule"
        assert error.rule_context == rule_context
        assert error.context.metadata["rule_name"] == "max_value_rule"
        assert error.context.metadata["rule_context"] == rule_context
        assert error.context.operation == "business_rule_validation"

    def test_business_rule_violation_without_context(self):
        """Test BusinessRuleViolationError creation without context."""
        error = BusinessRuleViolationError("Business rule violated")

        assert error.rule_name is None
        assert error.rule_context == {}


class TestConfigurationError:
    """Test ConfigurationError functionality."""

    def test_configuration_error_creation(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError(
            "Invalid configuration",
            config_key="database.host",
            config_value="invalid_host"
        )

        assert error.config_key == "database.host"
        assert error.config_value == "invalid_host"
        assert error.context.component == "configuration"
        assert error.context.metadata["config_key"] == "database.host"
        assert error.context.metadata["config_value"] == "invalid_host"


class TestResourceError:
    """Test ResourceError functionality."""

    def test_resource_error_creation(self):
        """Test ResourceError creation."""
        error = ResourceError(
            "Resource exhausted",
            resource_type="memory",
            current_usage=95.5,
            limit=100.0
        )

        assert error.resource_type == "memory"
        assert error.current_usage == 95.5
        assert error.limit == 100.0
        assert error.context.component == "resource_management"
        assert error.context.metadata["resource_type"] == "memory"
        assert error.context.metadata["current_usage"] == 95.5
        assert error.context.metadata["limit"] == 100.0
