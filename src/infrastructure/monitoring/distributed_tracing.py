"""
Distributed tracing for request flow visibility across components.

Provides comprehensive tracing capabilities to track requests
across all system components with detailed span information.
"""

import time
import uuid
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import threading
from collections import defaultdict

from src.infrastructure.logging.structured_logger import get_logger
from src.infrastructure.logging.correlation_context import CorrelationContext


class SpanKind(Enum):
    """Types of spans in distributed tracing."""
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"


class SpanStatus(Enum):
    """Status of a span."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanEvent:
    """Event within a span."""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes
        }


@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    events: List[SpanEvent] = field(default_factory=list)
    component: Optional[str] = None
    service_name: Optional[str] = None

    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status

    def set_tag(self, key: str, value: Any):
        """Set a tag on the span."""
        self.tags[key] = value

    def set_tags(self, tags: Dict[str, Any]):
        """Set multiple tags on the span."""
        self.tags.update(tags)

    def log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        event = SpanEvent(
            name=name,
            timestamp=datetime.utcnow(),
            attributes=attributes or {}
        )
        self.events.append(event)

    def set_error(self, error: Exception):
        """Mark span as error and add error details."""
        self.status = SpanStatus.ERROR
        self.set_tag("error", True)
        self.set_tag("error.type", type(error).__name__)
        self.set_tag("error.message", str(error))
        self.log(f"Error occurred: {str(error)}", level="error")

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "tags": self.tags,
            "logs": self.logs,
            "events": [event.to_dict() for event in self.events],
            "component": self.component,
            "service_name": self.service_name
        }


@dataclass
class Trace:
    """Complete distributed trace."""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    root_span: Optional[Span] = None

    def add_span(self, span: Span):
        """Add a span to the trace."""
        self.spans.append(span)

        # Update trace timing
        if self.start_time is None or span.start_time < self.start_time:
            self.start_time = span.start_time

        if span.end_time:
            if self.end_time is None or span.end_time > self.end_time:
                self.end_time = span.end_time

        # Set root span (span without parent)
        if span.parent_span_id is None:
            self.root_span = span

        # Calculate duration
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """Get span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_child_spans(self, parent_span_id: str) -> List[Span]:
        """Get child spans of a parent span."""
        return [span for span in self.spans if span.parent_span_id == parent_span_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "span_count": len(self.spans),
            "root_span_id": self.root_span.span_id if self.root_span else None,
            "spans": [span.to_dict() for span in self.spans]
        }


class TracingContext:
    """Thread-local tracing context."""

    def __init__(self):
        self._local = threading.local()

    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return getattr(self._local, 'current_span', None)

    def set_current_span(self, span: Optional[Span]):
        """Set current active span."""
        self._local.current_span = span

    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID."""
        span = self.get_current_span()
        return span.trace_id if span else None

    def get_span_id(self) -> Optional[str]:
        """Get current span ID."""
        span = self.get_current_span()
        return span.span_id if span else None


class DistributedTracer:
    """
    Distributed tracer for request flow visibility.

    Provides comprehensive tracing capabilities to track requests
    across all system components with detailed span information.
    """

    def __init__(
        self,
        service_name: str = "tournament_optimization",
        max_traces: int = 1000,
        trace_retention_hours: int = 24
    ):
        self.logger = get_logger("distributed_tracer")
        self.service_name = service_name
        self.max_traces = max_traces
        self.trace_retention_hours = trace_retention_hours

        # Trace storage
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}

        # Thread-local context
        self.context = TracingContext()

        # Thread safety
        self._lock = threading.Lock()

    def start_trace(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new trace with root span."""
        if trace_id is None:
            trace_id = self._generate_trace_id()

        span = self.start_span(
            operation_name=operation_name,
            kind=SpanKind.SERVER,
            trace_id=trace_id,
            parent_span_id=None,
            tags=tags
        )

        # Create trace
        trace = Trace(trace_id=trace_id)
        trace.add_span(span)

        with self._lock:
            self.traces[trace_id] = trace

        # Set correlation context
        CorrelationContext.set_correlation_id(trace_id)

        self.logger.debug(f"Started new trace: {trace_id}")
        return span

    def start_span(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span."""
        # Get trace context
        current_span = self.context.get_current_span()
        if trace_id is None and current_span:
            trace_id = current_span.trace_id
        if parent_span_id is None and current_span:
            parent_span_id = current_span.span_id

        if trace_id is None:
            trace_id = self._generate_trace_id()

        span_id = self._generate_span_id()

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            kind=kind,
            start_time=datetime.utcnow(),
            component=self._extract_component_from_operation(operation_name),
            service_name=self.service_name
        )

        if tags:
            span.set_tags(tags)

        # Store active span
        with self._lock:
            self.active_spans[span_id] = span

            # Add to trace if it exists
            if trace_id in self.traces:
                self.traces[trace_id].add_span(span)

        # Set as current span
        self.context.set_current_span(span)

        self.logger.debug(f"Started span: {operation_name} ({span_id})")
        return span

    def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK):
        """Finish a span."""
        span.finish(status)

        # Remove from active spans
        with self._lock:
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]

        # Clear current span if it's this one
        if self.context.get_current_span() == span:
            # Set parent as current span
            parent_span = self.get_span_by_id(span.parent_span_id) if span.parent_span_id else None
            self.context.set_current_span(parent_span)

        # Log span completion
        self.logger.debug(
            f"Finished span: {span.operation_name} ({span.span_id})",
            extra={
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "duration_ms": span.duration_ms,
                "status": span.status.value
            }
        )

        # Log detailed span info for analysis
        self.logger.info(
            f"Span completed: {span.operation_name}",
            extra={
                "span_data": span.to_dict(),
                "event_type": "span_completed"
            }
        )

    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return self.context.get_current_span()

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        with self._lock:
            return self.traces.get(trace_id)

    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """Get span by ID from active spans."""
        with self._lock:
            return self.active_spans.get(span_id)

    def get_all_traces(self) -> List[Trace]:
        """Get all traces."""
        with self._lock:
            return list(self.traces.values())

    def search_traces(
        self,
        operation_name: Optional[str] = None,
        service_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        status: Optional[SpanStatus] = None,
        limit: int = 100
    ) -> List[Trace]:
        """Search traces by criteria."""
        with self._lock:
            traces = list(self.traces.values())

        filtered_traces = []

        for trace in traces:
            if len(filtered_traces) >= limit:
                break

            # Check if trace matches criteria
            if self._trace_matches_criteria(
                trace, operation_name, service_name, tags,
                min_duration_ms, max_duration_ms, status
            ):
                filtered_traces.append(trace)

        return filtered_traces

    def cleanup_old_traces(self):
        """Clean up old traces to prevent memory leaks."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.trace_retention_hours)

        with self._lock:
            traces_to_remove = []

            for trace_id, trace in self.traces.items():
                if trace.start_time and trace.start_time < cutoff_time:
                    traces_to_remove.append(trace_id)

            for trace_id in traces_to_remove:
                del self.traces[trace_id]

            # Also limit total number of traces
            if len(self.traces) > self.max_traces:
                # Remove oldest traces
                sorted_traces = sorted(
                    self.traces.items(),
                    key=lambda x: x[1].start_time or datetime.min
                )

                traces_to_remove = sorted_traces[:len(self.traces) - self.max_traces]
                for trace_id, _ in traces_to_remove:
                    del self.traces[trace_id]

        if traces_to_remove:
            self.logger.debug(f"Cleaned up {len(traces_to_remove)} old traces")

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing operations."""
        span = self.start_span(operation_name, kind=kind, tags=tags)

        try:
            yield span
            self.finish_span(span, SpanStatus.OK)
        except Exception as e:
            span.set_error(e)
            self.finish_span(span, SpanStatus.ERROR)
            raise

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return str(uuid.uuid4())

    def _extract_component_from_operation(self, operation_name: str) -> str:
        """Extract component name from operation name."""
        # Simple heuristic to extract component
        parts = operation_name.lower().split('.')
        if len(parts) > 1:
            return parts[0]

        # Common patterns
        if 'forecast' in operation_name.lower():
            return 'forecasting'
        elif 'research' in operation_name.lower():
            return 'research'
        elif 'ensemble' in operation_name.lower():
            return 'ensemble'
        elif 'tournament' in operation_name.lower():
            return 'tournament'
        else:
            return 'unknown'

    def _trace_matches_criteria(
        self,
        trace: Trace,
        operation_name: Optional[str],
        service_name: Optional[str],
        tags: Optional[Dict[str, Any]],
        min_duration_ms: Optional[float],
        max_duration_ms: Optional[float],
        status: Optional[SpanStatus]
    ) -> bool:
        """Check if trace matches search criteria."""
        # Check duration
        if min_duration_ms is not None and (trace.duration_ms is None or trace.duration_ms < min_duration_ms):
            return False
        if max_duration_ms is not None and (trace.duration_ms is None or trace.duration_ms > max_duration_ms):
            return False

        # Check spans for other criteria
        for span in trace.spans:
            # Check operation name
            if operation_name and operation_name.lower() not in span.operation_name.lower():
                continue

            # Check service name
            if service_name and span.service_name != service_name:
                continue

            # Check status
            if status and span.status != status:
                continue

            # Check tags
            if tags:
                span_matches_tags = True
                for key, value in tags.items():
                    if key not in span.tags or span.tags[key] != value:
                        span_matches_tags = False
                        break
                if not span_matches_tags:
                    continue

            # If we get here, at least one span matches
            return True

        return False


# Global tracer instance
distributed_tracer = DistributedTracer()


def get_tracer() -> DistributedTracer:
    """Get the global distributed tracer instance."""
    return distributed_tracer


def configure_tracing(
    service_name: str = "tournament_optimization",
    max_traces: int = 1000,
    trace_retention_hours: int = 24
) -> DistributedTracer:
    """Configure and return a distributed tracer."""
    global distributed_tracer
    distributed_tracer = DistributedTracer(
        service_name=service_name,
        max_traces=max_traces,
        trace_retention_hours=trace_retention_hours
    )
    return distributed_tracer


# Convenience functions
def start_trace(operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Span:
    """Start a new trace."""
    return distributed_tracer.start_trace(operation_name, tags=tags)


def start_span(operation_name: str, kind: SpanKind = SpanKind.INTERNAL, tags: Optional[Dict[str, Any]] = None) -> Span:
    """Start a new span."""
    return distributed_tracer.start_span(operation_name, kind=kind, tags=tags)


def finish_span(span: Span, status: SpanStatus = SpanStatus.OK):
    """Finish a span."""
    distributed_tracer.finish_span(span, status)


def get_current_span() -> Optional[Span]:
    """Get current active span."""
    return distributed_tracer.get_current_span()


def trace_operation(operation_name: str, kind: SpanKind = SpanKind.INTERNAL, tags: Optional[Dict[str, Any]] = None):
    """Context manager for tracing operations."""
    return distributed_tracer.trace_operation(operation_name, kind=kind, tags=tags)
