"""
Detailed reasoning trace preservation with searchable logging.

Provides comprehensive reasoning trace storage and search capabilities
for analysis and debugging of forecasting decisions.
"""

import json
import uuid
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import re

from src.infrastructure.logging.structured_logger import get_logger


class ReasoningStepType(Enum):
    """Types of reasoning steps."""
    PROBLEM_ANALYSIS = "problem_analysis"
    EVIDENCE_GATHERING = "evidence_gathering"
    HYPOTHESIS_FORMATION = "hypothesis_formation"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    PREDICTION_GENERATION = "prediction_generation"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    FINAL_DECISION = "final_decision"
    ERROR_CORRECTION = "error_correction"


@dataclass
class ReasoningStep:
    """Individual step in reasoning process."""
    id: str
    step_number: int
    step_type: ReasoningStepType
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence_level: float
    confidence_basis: str
    duration_ms: float
    timestamp: datetime
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "confidence_level": self.confidence_level,
            "confidence_basis": self.confidence_basis,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningStep':
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            step_number=data["step_number"],
            step_type=ReasoningStepType(data["step_type"]),
            description=data["description"],
            input_data=data["input_data"],
            output_data=data["output_data"],
            confidence_level=data["confidence_level"],
            confidence_basis=data["confidence_basis"],
            duration_ms=data["duration_ms"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_id=data.get("agent_id"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a forecasting operation."""
    id: str
    question_id: str
    agent_id: str
    agent_type: str
    operation_type: str  # "forecast", "research", "ensemble"
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    steps: List[ReasoningStep] = field(default_factory=list)
    final_result: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ReasoningStep):
        """Add a reasoning step to the trace."""
        self.steps.append(step)

    def finish(self, success: bool = True, error_message: Optional[str] = None, final_result: Optional[Dict[str, Any]] = None):
        """Finish the reasoning trace."""
        self.end_time = datetime.utcnow()
        self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_message = error_message
        self.final_result = final_result

    def get_step_by_type(self, step_type: ReasoningStepType) -> List[ReasoningStep]:
        """Get steps by type."""
        return [step for step in self.steps if step.step_type == step_type]

    def get_confidence_progression(self) -> List[float]:
        """Get confidence levels throughout the reasoning process."""
        return [step.confidence_level for step in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "question_id": self.question_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "operation_type": self.operation_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "steps": [step.to_dict() for step in self.steps],
            "final_result": self.final_result,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "step_count": len(self.steps),
            "avg_confidence": sum(step.confidence_level for step in self.steps) / len(self.steps) if self.steps else 0.0
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningTrace':
        """Create from dictionary representation."""
        trace = cls(
            id=data["id"],
            question_id=data["question_id"],
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            operation_type=data["operation_type"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            total_duration_ms=data.get("total_duration_ms"),
            final_result=data.get("final_result"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {})
        )

        # Add steps
        for step_data in data.get("steps", []):
            step = ReasoningStep.from_dict(step_data)
            trace.add_step(step)

        return trace


class ReasoningTraceStorage:
    """Storage backend for reasoning traces."""

    def __init__(self, max_traces: int = 10000, retention_days: int = 30):
        self.max_traces = max_traces
        self.retention_days = retention_days
        self.traces: Dict[str, ReasoningTrace] = {}
        self._lock = threading.Lock()

    def store_trace(self, trace: ReasoningTrace):
        """Store a reasoning trace."""
        with self._lock:
            self.traces[trace.id] = trace

            # Cleanup if needed
            if len(self.traces) > self.max_traces:
                self._cleanup_old_traces()

    def get_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Get trace by ID."""
        with self._lock:
            return self.traces.get(trace_id)

    def get_traces_by_question(self, question_id: str) -> List[ReasoningTrace]:
        """Get all traces for a question."""
        with self._lock:
            return [trace for trace in self.traces.values() if trace.question_id == question_id]

    def get_traces_by_agent(self, agent_id: str) -> List[ReasoningTrace]:
        """Get all traces for an agent."""
        with self._lock:
            return [trace for trace in self.traces.values() if trace.agent_id == agent_id]

    def get_all_traces(self) -> List[ReasoningTrace]:
        """Get all traces."""
        with self._lock:
            return list(self.traces.values())

    def _cleanup_old_traces(self):
        """Clean up old traces."""
        cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)

        traces_to_remove = []
        for trace_id, trace in self.traces.items():
            if trace.start_time < cutoff_time:
                traces_to_remove.append(trace_id)

        # Remove oldest traces if still over limit
        if len(self.traces) - len(traces_to_remove) > self.max_traces:
            sorted_traces = sorted(
                self.traces.items(),
                key=lambda x: x[1].start_time
            )

            additional_removals = len(self.traces) - len(traces_to_remove) - self.max_traces
            for i in range(additional_removals):
                traces_to_remove.append(sorted_traces[i][0])

        for trace_id in traces_to_remove:
            del self.traces[trace_id]


class ReasoningTraceSearcher:
    """Search engine for reasoning traces."""

    def __init__(self, storage: ReasoningTraceStorage):
        self.storage = storage
        self.logger = get_logger("reasoning_trace_searcher")

    def search(
        self,
        query: Optional[str] = None,
        question_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        operation_type: Optional[str] = None,
        step_type: Optional[ReasoningStepType] = None,
        success: Optional[bool] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ReasoningTrace]:
        """Search reasoning traces with various criteria."""
        traces = self.storage.get_all_traces()
        filtered_traces = []

        for trace in traces:
            if len(filtered_traces) >= limit:
                break

            # Apply filters
            if question_id and trace.question_id != question_id:
                continue

            if agent_id and trace.agent_id != agent_id:
                continue

            if agent_type and trace.agent_type != agent_type:
                continue

            if operation_type and trace.operation_type != operation_type:
                continue

            if success is not None and trace.success != success:
                continue

            if start_time and trace.start_time < start_time:
                continue

            if end_time and trace.start_time > end_time:
                continue

            # Check step-specific criteria
            if step_type:
                matching_steps = trace.get_step_by_type(step_type)
                if not matching_steps:
                    continue

            # Check confidence range
            if min_confidence is not None or max_confidence is not None:
                confidences = trace.get_confidence_progression()
                if not confidences:
                    continue

                avg_confidence = sum(confidences) / len(confidences)

                if min_confidence is not None and avg_confidence < min_confidence:
                    continue

                if max_confidence is not None and avg_confidence > max_confidence:
                    continue

            # Text search
            if query:
                if not self._matches_text_query(trace, query):
                    continue

            filtered_traces.append(trace)

        return filtered_traces

    def _matches_text_query(self, trace: ReasoningTrace, query: str) -> bool:
        """Check if trace matches text query."""
        query_lower = query.lower()

        # Search in trace metadata
        trace_text = json.dumps(trace.to_dict()).lower()
        if query_lower in trace_text:
            return True

        # Search in individual steps
        for step in trace.steps:
            step_text = f"{step.description} {step.confidence_basis}".lower()
            if query_lower in step_text:
                return True

            # Search in step data
            step_data_text = json.dumps(step.input_data).lower() + json.dumps(step.output_data).lower()
            if query_lower in step_data_text:
                return True

        return False

    def get_reasoning_patterns(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyze reasoning patterns across traces."""
        traces = self.storage.get_all_traces()

        if agent_type:
            traces = [t for t in traces if t.agent_type == agent_type]

        if not traces:
            return {}

        # Analyze patterns
        patterns = {
            "total_traces": len(traces),
            "success_rate": sum(1 for t in traces if t.success) / len(traces),
            "avg_duration_ms": sum(t.total_duration_ms for t in traces if t.total_duration_ms) / len([t for t in traces if t.total_duration_ms]),
            "avg_steps_per_trace": sum(len(t.steps) for t in traces) / len(traces),
            "step_type_distribution": defaultdict(int),
            "confidence_distribution": {"low": 0, "medium": 0, "high": 0},
            "common_error_patterns": defaultdict(int)
        }

        # Analyze step types
        for trace in traces:
            for step in trace.steps:
                patterns["step_type_distribution"][step.step_type.value] += 1

                # Confidence distribution
                if step.confidence_level < 0.3:
                    patterns["confidence_distribution"]["low"] += 1
                elif step.confidence_level < 0.7:
                    patterns["confidence_distribution"]["medium"] += 1
                else:
                    patterns["confidence_distribution"]["high"] += 1

        # Analyze error patterns
        for trace in traces:
            if not trace.success and trace.error_message:
                error_type = trace.error_message.split(':')[0] if ':' in trace.error_message else trace.error_message
                patterns["common_error_patterns"][error_type] += 1

        return patterns


class ReasoningTraceManager:
    """
    Comprehensive reasoning trace management system.

    Provides detailed reasoning trace preservation with searchable
    logging for analysis and debugging of forecasting decisions.
    """

    def __init__(
        self,
        max_traces: int = 10000,
        retention_days: int = 30,
        auto_log: bool = True
    ):
        self.logger = get_logger("reasoning_trace_manager")
        self.storage = ReasoningTraceStorage(max_traces, retention_days)
        self.searcher = ReasoningTraceSearcher(self.storage)
        self.auto_log = auto_log

        # Active traces
        self.active_traces: Dict[str, ReasoningTrace] = {}
        self._lock = threading.Lock()

    def start_trace(
        self,
        question_id: str,
        agent_id: str,
        agent_type: str,
        operation_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new reasoning trace."""
        trace_id = str(uuid.uuid4())

        trace = ReasoningTrace(
            id=trace_id,
            question_id=question_id,
            agent_id=agent_id,
            agent_type=agent_type,
            operation_type=operation_type,
            start_time=datetime.utcnow(),
            metadata=metadata or {}
        )

        with self._lock:
            self.active_traces[trace_id] = trace

        if self.auto_log:
            self.logger.info(
                f"Started reasoning trace: {operation_type}",
                extra={
                    "trace_id": trace_id,
                    "question_id": question_id,
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "operation_type": operation_type
                }
            )

        return trace_id

    def add_reasoning_step(
        self,
        trace_id: str,
        step_type: ReasoningStepType,
        description: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence_level: float,
        confidence_basis: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a reasoning step to an active trace."""
        with self._lock:
            trace = self.active_traces.get(trace_id)
            if not trace:
                self.logger.warning(f"Trace not found: {trace_id}")
                return ""

        step_id = str(uuid.uuid4())
        step = ReasoningStep(
            id=step_id,
            step_number=len(trace.steps) + 1,
            step_type=step_type,
            description=description,
            input_data=input_data,
            output_data=output_data,
            confidence_level=confidence_level,
            confidence_basis=confidence_basis,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            agent_id=trace.agent_id,
            metadata=metadata or {}
        )

        trace.add_step(step)

        if self.auto_log:
            self.logger.info(
                f"Reasoning step: {description}",
                extra={
                    "trace_id": trace_id,
                    "step_id": step_id,
                    "step_type": step_type.value,
                    "step_number": step.step_number,
                    "confidence_level": confidence_level,
                    "duration_ms": duration_ms
                }
            )

        return step_id

    def finish_trace(
        self,
        trace_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        final_result: Optional[Dict[str, Any]] = None
    ):
        """Finish and store a reasoning trace."""
        with self._lock:
            trace = self.active_traces.get(trace_id)
            if not trace:
                self.logger.warning(f"Trace not found: {trace_id}")
                return

            # Remove from active traces
            del self.active_traces[trace_id]

        # Finish the trace
        trace.finish(success, error_message, final_result)

        # Store the trace
        self.storage.store_trace(trace)

        if self.auto_log:
            self.logger.info(
                f"Finished reasoning trace: {trace.operation_type}",
                extra={
                    "trace_id": trace_id,
                    "success": success,
                    "total_duration_ms": trace.total_duration_ms,
                    "step_count": len(trace.steps),
                    "avg_confidence": sum(s.confidence_level for s in trace.steps) / len(trace.steps) if trace.steps else 0.0,
                    "reasoning_trace": trace.to_dict()
                }
            )

    def get_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Get a reasoning trace by ID."""
        # Check active traces first
        with self._lock:
            if trace_id in self.active_traces:
                return self.active_traces[trace_id]

        # Check storage
        return self.storage.get_trace(trace_id)

    def search_traces(self, **kwargs) -> List[ReasoningTrace]:
        """Search reasoning traces."""
        return self.searcher.search(**kwargs)

    def get_reasoning_patterns(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Get reasoning patterns analysis."""
        return self.searcher.get_reasoning_patterns(agent_type)

    def export_traces(
        self,
        traces: List[ReasoningTrace],
        format: str = "json"
    ) -> str:
        """Export traces to various formats."""
        if format == "json":
            return json.dumps([trace.to_dict() for trace in traces], indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get overall trace statistics."""
        all_traces = self.storage.get_all_traces()

        if not all_traces:
            return {"total_traces": 0}

        successful_traces = [t for t in all_traces if t.success]

        return {
            "total_traces": len(all_traces),
            "successful_traces": len(successful_traces),
            "success_rate": len(successful_traces) / len(all_traces),
            "avg_duration_ms": sum(t.total_duration_ms for t in all_traces if t.total_duration_ms) / len([t for t in all_traces if t.total_duration_ms]),
            "avg_steps_per_trace": sum(len(t.steps) for t in all_traces) / len(all_traces),
            "unique_agents": len(set(t.agent_id for t in all_traces)),
            "unique_questions": len(set(t.question_id for t in all_traces)),
            "operation_types": list(set(t.operation_type for t in all_traces))
        }


# Global reasoning trace manager instance
reasoning_trace_manager = ReasoningTraceManager()


def get_reasoning_trace_manager() -> ReasoningTraceManager:
    """Get the global reasoning trace manager instance."""
    return reasoning_trace_manager


def configure_reasoning_traces(
    max_traces: int = 10000,
    retention_days: int = 30,
    auto_log: bool = True
) -> ReasoningTraceManager:
    """Configure and return a reasoning trace manager."""
    global reasoning_trace_manager
    reasoning_trace_manager = ReasoningTraceManager(
        max_traces=max_traces,
        retention_days=retention_days,
        auto_log=auto_log
    )
    return reasoning_trace_manager


# Convenience functions
def start_reasoning_trace(
    question_id: str,
    agent_id: str,
    agent_type: str,
    operation_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Start a new reasoning trace."""
    return reasoning_trace_manager.start_trace(
        question_id, agent_id, agent_type, operation_type, metadata
    )


def add_reasoning_step(
    trace_id: str,
    step_type: ReasoningStepType,
    description: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    confidence_level: float,
    confidence_basis: str,
    duration_ms: float,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Add a reasoning step to a trace."""
    return reasoning_trace_manager.add_reasoning_step(
        trace_id, step_type, description, input_data, output_data,
        confidence_level, confidence_basis, duration_ms, metadata
    )


def finish_reasoning_trace(
    trace_id: str,
    success: bool = True,
    error_message: Optional[str] = None,
    final_result: Optional[Dict[str, Any]] = None
):
    """Finish a reasoning trace."""
    reasoning_trace_manager.finish_trace(trace_id, success, error_message, final_result)
