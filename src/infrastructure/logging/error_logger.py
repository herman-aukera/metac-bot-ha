"""
Specialized error logging with comprehensive context and recovery workflows.

Provides detailed error logging, recovery tracking, and integration
with monitoring and alerting systems.
"""

import traceback
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.domain.exceptions import TournamentOptimizationError
from src.domain.exceptions.base_exceptions import ErrorContext
from .structured_logger import StructuredLogger
from .correlation_context import CorrelationContext


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    SYSTEM = "system"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    USER = "user"


@dataclass
class ErrorLogEntry:
    """Comprehensive error log entry."""

    error_id: str
    timestamp: datetime
    correlation_id: str
    exception_type: str
    exception_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    recoverable: bool
    recovery_attempted: bool = False
    recovery_successful: bool = False
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "component": self.component,
            "operation": self.operation,
            "recoverable": self.recoverable,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "metadata": self.metadata,
        }


class ErrorLogger:
    """
    Specialized error logger with comprehensive context and recovery tracking.

    Provides detailed error logging, classification, and integration
    with recovery workflows and monitoring systems.
    """

    def __init__(self, logger: Optional[StructuredLogger] = None):
        self.logger = logger or StructuredLogger("tournament_optimization.errors")
        self.error_history: List[ErrorLogEntry] = []
        self.recovery_handlers: Dict[str, Callable] = {}
        self.error_patterns: Dict[str, int] = {}

    def log_error(
        self,
        exception: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> str:
        """
        Log error with comprehensive context and optional recovery.

        Args:
            exception: The exception to log
            severity: Error severity level
            category: Error category
            component: Component where error occurred
            operation: Operation that failed
            context: Additional context data
            metadata: Additional metadata
            attempt_recovery: Whether to attempt automatic recovery

        Returns:
            Error ID for tracking
        """
        import uuid

        # Generate error ID
        error_id = str(uuid.uuid4())

        # Get correlation context
        correlation_context = CorrelationContext.get_current_context()

        # Extract error information
        exception_type = type(exception).__name__
        exception_message = str(exception)

        # Determine component and operation from context if not provided
        if not component:
            component = correlation_context.component or "unknown"
        if not operation:
            operation = correlation_context.operation or "unknown"

        # Get stack trace
        stack_trace = traceback.format_exc() if exception.__traceback__ else None

        # Determine if error is recoverable
        recoverable = False
        if isinstance(exception, TournamentOptimizationError):
            recoverable = exception.recoverable

        # Create error log entry
        error_entry = ErrorLogEntry(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_context.correlation_id,
            exception_type=exception_type,
            exception_message=exception_message,
            severity=severity,
            category=category,
            component=component,
            operation=operation,
            recoverable=recoverable,
            stack_trace=stack_trace,
            context=context or {},
            metadata=metadata or {}
        )

        # Add to error history
        self.error_history.append(error_entry)

        # Track error patterns
        pattern_key = f"{exception_type}:{component}:{operation}"
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1

        # Log the error
        self.logger.error(
            f"Error occurred: {exception_message}",
            exception=exception,
            extra={
                "error_id": error_id,
                "error_severity": severity.value,
                "error_category": category.value,
                "error_component": component,
                "error_operation": operation,
                "error_recoverable": recoverable,
                "error_context": context or {},
                "error_metadata": metadata or {},
                "error_pattern_count": self.error_patterns[pattern_key],
            }
        )

        # Attempt recovery if enabled and error is recoverable
        if attempt_recovery and recoverable:
            self._attempt_recovery(error_entry, exception)

        # Check for error patterns that might need attention
        self._check_error_patterns(pattern_key)

        return error_id

    def log_recovery_attempt(
        self,
        error_id: str,
        recovery_strategy: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log recovery attempt for an error."""
        # Find error entry
        error_entry = None
        for entry in self.error_history:
            if entry.error_id == error_id:
                error_entry = entry
                break

        if not error_entry:
            self.logger.warning(f"Recovery attempt logged for unknown error ID: {error_id}")
            return

        # Update error entry
        error_entry.recovery_attempted = True
        error_entry.recovery_successful = success

        # Log recovery attempt
        log_level = "info" if success else "warning"
        getattr(self.logger, log_level)(
            f"Recovery {'succeeded' if success else 'failed'} for error {error_id}",
            extra={
                "error_id": error_id,
                "recovery_strategy": recovery_strategy,
                "recovery_success": success,
                "recovery_details": details or {},
                "original_error_type": error_entry.exception_type,
                "original_component": error_entry.component,
                "original_operation": error_entry.operation,
            }
        )

    def register_recovery_handler(
        self,
        exception_type: str,
        handler: Callable[[Exception, ErrorLogEntry], bool]
    ):
        """Register recovery handler for specific exception type."""
        self.recovery_handlers[exception_type] = handler
        self.logger.info(f"Recovery handler registered for {exception_type}")

    def _attempt_recovery(self, error_entry: ErrorLogEntry, exception: Exception):
        """Attempt automatic recovery for an error."""
        handler = self.recovery_handlers.get(error_entry.exception_type)
        if not handler:
            return

        try:
            self.logger.info(
                f"Attempting recovery for error {error_entry.error_id}",
                extra={
                    "error_id": error_entry.error_id,
                    "exception_type": error_entry.exception_type,
                    "recovery_handler": handler.__name__,
                }
            )

            success = handler(exception, error_entry)
            self.log_recovery_attempt(
                error_entry.error_id,
                handler.__name__,
                success
            )

        except Exception as recovery_exception:
            self.logger.error(
                f"Recovery handler failed for error {error_entry.error_id}",
                exception=recovery_exception,
                extra={
                    "error_id": error_entry.error_id,
                    "original_exception_type": error_entry.exception_type,
                    "recovery_handler": handler.__name__,
                }
            )

    def _check_error_patterns(self, pattern_key: str):
        """Check for concerning error patterns."""
        count = self.error_patterns[pattern_key]

        # Alert on repeated errors
        if count in [5, 10, 25, 50, 100]:  # Alert at these thresholds
            self.logger.warning(
                f"Error pattern detected: {pattern_key} occurred {count} times",
                extra={
                    "error_pattern": pattern_key,
                    "occurrence_count": count,
                    "alert_type": "error_pattern",
                }
            )

    def get_error_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get error summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > cutoff_time
        ]

        # Count by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = sum(
                1 for error in recent_errors
                if error.severity == severity
            )

        # Count by category
        category_counts = {}
        for category in ErrorCategory:
            category_counts[category.value] = sum(
                1 for error in recent_errors
                if error.category == category
            )

        # Count by component
        component_counts = {}
        for error in recent_errors:
            component_counts[error.component] = component_counts.get(error.component, 0) + 1

        # Recovery statistics
        recoverable_errors = [error for error in recent_errors if error.recoverable]
        recovery_attempted = sum(1 for error in recoverable_errors if error.recovery_attempted)
        recovery_successful = sum(1 for error in recoverable_errors if error.recovery_successful)

        return {
            "time_period_hours": hours,
            "total_errors": len(recent_errors),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "component_breakdown": component_counts,
            "recovery_stats": {
                "recoverable_errors": len(recoverable_errors),
                "recovery_attempted": recovery_attempted,
                "recovery_successful": recovery_successful,
                "recovery_success_rate": (recovery_successful / recovery_attempted * 100) if recovery_attempted > 0 else 0,
            },
            "top_error_patterns": sorted(
                [(pattern, count) for pattern, count in self.error_patterns.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def get_recent_errors(
        self,
        limit: int = 50,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        component: Optional[str] = None
    ) -> List[ErrorLogEntry]:
        """Get recent errors with optional filtering."""
        errors = self.error_history.copy()

        # Apply filters
        if severity:
            errors = [error for error in errors if error.severity == severity]
        if category:
            errors = [error for error in errors if error.category == category]
        if component:
            errors = [error for error in errors if error.component == component]

        # Sort by timestamp (most recent first) and limit
        errors.sort(key=lambda x: x.timestamp, reverse=True)
        return errors[:limit]

    def clear_old_errors(self, days: int = 7):
        """Clear error history older than specified days."""
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(days=days)
        old_count = len(self.error_history)

        self.error_history = [
            error for error in self.error_history
            if error.timestamp > cutoff_time
        ]

        new_count = len(self.error_history)
        cleared_count = old_count - new_count

        if cleared_count > 0:
            self.logger.info(
                f"Cleared {cleared_count} old error entries (older than {days} days)",
                extra={
                    "cleared_errors": cleared_count,
                    "remaining_errors": new_count,
                    "cutoff_days": days,
                }
            )


# Global error logger instance
error_logger = ErrorLogger()
