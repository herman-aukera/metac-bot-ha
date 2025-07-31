"""
Base exception classes with structured error context and correlation IDs.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ErrorContext:
    """Structured error context with correlation tracking."""

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    component: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
            "stack_trace": self.stack_trace,
        }


class TournamentOptimizationError(Exception):
    """
    Base exception for all tournament optimization system errors.

    Provides structured error context, correlation IDs, and comprehensive
    error information for debugging and monitoring.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = False,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after

        # Add any additional metadata
        self.context.metadata.update(kwargs)

        # Set component if not already set
        if not self.context.component:
            # Extract component from module path (e.g., src.domain.exceptions -> domain)
            module_parts = self.__class__.__module__.split('.')
            if len(module_parts) >= 2:
                self.context.component = module_parts[1]  # Get 'domain' from 'src.domain.exceptions'
            else:
                self.context.component = self.__class__.__module__

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and monitoring."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "cause": str(self.cause) if self.cause else None,
            "context": self.context.to_dict(),
        }

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message} (correlation_id: {self.context.correlation_id})"


class DomainError(TournamentOptimizationError):
    """Base class for domain-specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        if not self.context.component:
            self.context.component = "domain"


class ValidationError(DomainError):
    """
    Raised when data validation fails.

    Includes detailed information about validation failures
    and the specific fields that failed validation.
    """

    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.field_errors = field_errors or {}
        self.context.metadata["field_errors"] = self.field_errors
        self.context.operation = "validation"


class BusinessRuleViolationError(DomainError):
    """
    Raised when business rules are violated.

    Includes information about the specific rule that was violated
    and the context in which the violation occurred.
    """

    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        rule_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.rule_name = rule_name
        self.rule_context = rule_context or {}
        self.context.metadata.update({
            "rule_name": rule_name,
            "rule_context": rule_context,
        })
        self.context.operation = "business_rule_validation"


class ConfigurationError(TournamentOptimizationError):
    """
    Raised when system configuration is invalid or missing.

    Includes information about the specific configuration
    that is problematic.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_value = config_value
        self.context.metadata.update({
            "config_key": config_key,
            "config_value": str(config_value) if config_value is not None else None,
        })
        self.context.component = "configuration"


class ResourceError(TournamentOptimizationError):
    """
    Raised when system resources are unavailable or exhausted.

    Includes information about the specific resource and
    current utilization levels.
    """

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        self.context.metadata.update({
            "resource_type": resource_type,
            "current_usage": current_usage,
            "limit": limit,
        })
        self.context.component = "resource_management"
