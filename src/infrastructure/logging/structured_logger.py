"""
Structured logging with correlation IDs and JSON formatting.

Provides comprehensive logging capabilities with structured data,
correlation tracking, and integration with monitoring systems.
"""

import json
import logging
import sys
from enum import Enum
from typing import Dict, Any, Optional, Union
from datetime import datetime
import traceback

from .correlation_context import CorrelationContext


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        correlation_id = CorrelationContext.get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add request ID if available
        request_id = CorrelationContext.get_request_id()
        if request_id:
            log_entry["request_id"] = request_id

        # Add user ID if available
        user_id = CorrelationContext.get_user_id()
        if user_id:
            log_entry["user_id"] = user_id

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            }:
                # Ensure value is JSON serializable
                try:
                    json.dumps(value)
                    extra_fields[key] = value
                except (TypeError, ValueError):
                    extra_fields[key] = str(value)

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, ensure_ascii=False)


class StructuredLogger:
    """
    Structured logger with correlation IDs and comprehensive context.

    Provides structured logging capabilities with JSON formatting,
    correlation tracking, and integration with monitoring systems.
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = False,
        file_path: Optional[str] = None
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup structured formatter
        formatter = StructuredFormatter()

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if enable_file and file_path:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

    def debug(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log debug message with structured data."""
        self._log(LogLevel.DEBUG, message, extra, **kwargs)

    def info(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log info message with structured data."""
        self._log(LogLevel.INFO, message, extra, **kwargs)

    def warning(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log warning message with structured data."""
        self._log(LogLevel.WARNING, message, extra, **kwargs)

    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log error message with structured data and exception info."""
        if exception:
            kwargs['exc_info'] = (type(exception), exception, exception.__traceback__)
        self._log(LogLevel.ERROR, message, extra, **kwargs)

    def critical(
        self,
        message: str,
        exception: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log critical message with structured data and exception info."""
        if exception:
            kwargs['exc_info'] = (type(exception), exception, exception.__traceback__)
        self._log(LogLevel.CRITICAL, message, extra, **kwargs)

    def _log(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Internal logging method."""
        # Merge extra data with kwargs
        log_extra = {}
        if extra:
            log_extra.update(extra)
        log_extra.update(kwargs)

        # Log with appropriate level
        log_level = getattr(logging, level.value)
        self.logger.log(log_level, message, extra=log_extra)

    def log_operation_start(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Log the start of an operation."""
        self.info(
            f"Operation started: {operation}",
            extra={
                "operation": operation,
                "operation_phase": "start",
                "parameters": parameters or {},
            }
        )

    def log_operation_success(
        self,
        operation: str,
        duration: Optional[float] = None,
        result_summary: Optional[Dict[str, Any]] = None
    ):
        """Log successful completion of an operation."""
        self.info(
            f"Operation completed successfully: {operation}",
            extra={
                "operation": operation,
                "operation_phase": "success",
                "duration_seconds": duration,
                "result_summary": result_summary or {},
            }
        )

    def log_operation_failure(
        self,
        operation: str,
        exception: Exception,
        duration: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log failed operation with exception details."""
        self.error(
            f"Operation failed: {operation}",
            exception=exception,
            extra={
                "operation": operation,
                "operation_phase": "failure",
                "duration_seconds": duration,
                "error_context": context or {},
            }
        )

    def log_performance_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        unit: str = "count",
        tags: Optional[Dict[str, str]] = None
    ):
        """Log performance metric."""
        self.info(
            f"Performance metric: {metric_name}",
            extra={
                "metric_type": "performance",
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                "metric_tags": tags or {},
            }
        )

    def log_business_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Log business event for analytics."""
        self.info(
            f"Business event: {event_type}",
            extra={
                "event_type": "business",
                "business_event_type": event_type,
                "event_data": event_data,
            }
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Log security event."""
        log_level = LogLevel.WARNING if severity.lower() in ["low", "medium"] else LogLevel.ERROR

        self._log(
            log_level,
            f"Security event: {event_type}",
            extra={
                "event_type": "security",
                "security_event_type": event_type,
                "security_severity": severity,
                "security_details": details,
            }
        )

    def create_child_logger(self, suffix: str) -> 'StructuredLogger':
        """Create child logger with extended name."""
        child_name = f"{self.name}.{suffix}"
        child_logger = StructuredLogger(
            name=child_name,
            level=LogLevel(self.logger.level),
            enable_console=False,  # Inherit handlers from parent
            enable_file=False
        )

        # Copy handlers from parent
        for handler in self.logger.handlers:
            child_logger.logger.addHandler(handler)

        return child_logger


# Global logger instances
system_logger = StructuredLogger("tournament_optimization.system")
domain_logger = StructuredLogger("tournament_optimization.domain")
application_logger = StructuredLogger("tournament_optimization.application")
infrastructure_logger = StructuredLogger("tournament_optimization.infrastructure")


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger."""
    return StructuredLogger(f"tournament_optimization.{name}")


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    enable_console: bool = True,
    enable_file: bool = False,
    file_path: Optional[str] = None
):
    """Configure global logging settings."""
    # Configure root logger
    root_logger = logging.getLogger("tournament_optimization")
    root_logger.setLevel(getattr(logging, level.value))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Setup structured formatter
    formatter = StructuredFormatter()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if enable_file and file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False
