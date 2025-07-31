"""
Correlation context management for distributed tracing.

Provides correlation ID tracking across async operations and
request boundaries for comprehensive logging and monitoring.
"""

import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


# Context variables for correlation tracking
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
_user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
_session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
_operation_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('operation_context', default=None)


@dataclass
class CorrelationContext:
    """
    Correlation context for tracking requests across system boundaries.

    Provides correlation ID management and context propagation
    for distributed tracing and comprehensive logging.
    """

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get current correlation ID from context."""
        return _correlation_id.get()

    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID in context."""
        _correlation_id.set(correlation_id)

    @classmethod
    def get_request_id(cls) -> Optional[str]:
        """Get current request ID from context."""
        return _request_id.get()

    @classmethod
    def set_request_id(cls, request_id: str):
        """Set request ID in context."""
        _request_id.set(request_id)

    @classmethod
    def get_user_id(cls) -> Optional[str]:
        """Get current user ID from context."""
        return _user_id.get()

    @classmethod
    def set_user_id(cls, user_id: str):
        """Set user ID in context."""
        _user_id.set(user_id)

    @classmethod
    def get_session_id(cls) -> Optional[str]:
        """Get current session ID from context."""
        return _session_id.get()

    @classmethod
    def set_session_id(cls, session_id: str):
        """Set session ID in context."""
        _session_id.set(session_id)

    @classmethod
    def get_operation_context(cls) -> Optional[Dict[str, Any]]:
        """Get current operation context."""
        return _operation_context.get()

    @classmethod
    def set_operation_context(cls, context: Dict[str, Any]):
        """Set operation context."""
        _operation_context.set(context)

    @classmethod
    def get_current_context(cls) -> 'CorrelationContext':
        """Get current correlation context."""
        return cls(
            correlation_id=cls.get_correlation_id() or str(uuid.uuid4()),
            request_id=cls.get_request_id(),
            user_id=cls.get_user_id(),
            session_id=cls.get_session_id(),
            metadata=cls.get_operation_context() or {}
        )

    def set_context(self):
        """Set this context as current."""
        _correlation_id.set(self.correlation_id)
        if self.request_id:
            _request_id.set(self.request_id)
        if self.user_id:
            _user_id.set(self.user_id)
        if self.session_id:
            _session_id.set(self.session_id)
        if self.metadata:
            _operation_context.set(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "operation": self.operation,
            "component": self.component,
            "start_time": self.start_time.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrelationContext':
        """Create context from dictionary."""
        return cls(
            correlation_id=data.get("correlation_id", str(uuid.uuid4())),
            request_id=data.get("request_id"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            operation=data.get("operation"),
            component=data.get("component"),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else datetime.utcnow(),
            metadata=data.get("metadata", {})
        )


class CorrelationContextManager:
    """
    Context manager for correlation context.

    Provides automatic context setup and cleanup
    for operations that need correlation tracking.
    """

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        **metadata
    ):
        self.context = CorrelationContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            operation=operation,
            component=component,
            metadata=metadata
        )
        self.previous_context = None

    def __enter__(self) -> CorrelationContext:
        """Enter context manager."""
        # Save previous context
        self.previous_context = CorrelationContext.get_current_context()

        # Set new context
        self.context.set_context()

        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        # Restore previous context
        if self.previous_context:
            self.previous_context.set_context()

    async def __aenter__(self) -> CorrelationContext:
        """Async enter context manager."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit context manager."""
        self.__exit__(exc_type, exc_val, exc_tb)


def correlation_id(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    operation: Optional[str] = None,
    component: Optional[str] = None,
    **metadata
) -> CorrelationContextManager:
    """
    Create correlation context manager.

    Usage:
        with correlation_id(operation="forecast_question") as ctx:
            # All logging within this block will include correlation ID
            logger.info("Processing question")
    """
    return CorrelationContextManager(
        correlation_id=correlation_id,
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        operation=operation,
        component=component,
        **metadata
    )


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def generate_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())


def propagate_context_to_headers() -> Dict[str, str]:
    """Get correlation context as HTTP headers for propagation."""
    headers = {}

    correlation_id = CorrelationContext.get_correlation_id()
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    request_id = CorrelationContext.get_request_id()
    if request_id:
        headers["X-Request-ID"] = request_id

    user_id = CorrelationContext.get_user_id()
    if user_id:
        headers["X-User-ID"] = user_id

    session_id = CorrelationContext.get_session_id()
    if session_id:
        headers["X-Session-ID"] = session_id

    return headers


def extract_context_from_headers(headers: Dict[str, str]) -> CorrelationContext:
    """Extract correlation context from HTTP headers."""
    return CorrelationContext(
        correlation_id=headers.get("X-Correlation-ID", str(uuid.uuid4())),
        request_id=headers.get("X-Request-ID"),
        user_id=headers.get("X-User-ID"),
        session_id=headers.get("X-Session-ID")
    )
