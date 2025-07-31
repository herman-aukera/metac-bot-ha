"""
Reasoning-specific exceptions for the tournament optimization system.
"""

from typing import Optional, Dict, Any, List
from .base_exceptions import DomainError


class ReasoningError(DomainError):
    """Base class for reasoning-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.context.component = "reasoning"


class ReasoningTimeoutError(ReasoningError):
    """
    Raised when reasoning process exceeds time limits.

    Includes information about the timeout duration and
    the reasoning step that was in progress.
    """

    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        current_step: Optional[str] = None,
        completed_steps: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        self.current_step = current_step
        self.completed_steps = completed_steps or []
        self.context.metadata.update({
            "timeout_duration": timeout_duration,
            "current_step": current_step,
            "completed_steps": completed_steps,
        })
        self.context.operation = "reasoning_timeout"
        self.recoverable = True
        self.retry_after = 60  # Retry after 1 minute


class ReasoningValidationError(ReasoningError):
    """
    Raised when reasoning output fails validation.

    Includes details about validation failures and
    the specific reasoning steps that failed.
    """

    def __init__(
        self,
        message: str,
        validation_errors: Optional[Dict[str, List[str]]] = None,
        failed_step: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or {}
        self.failed_step = failed_step
        self.context.metadata.update({
            "validation_errors": validation_errors,
            "failed_step": failed_step,
        })
        self.context.operation = "reasoning_validation"


class InsufficientEvidenceError(ReasoningError):
    """
    Raised when insufficient evidence is available for reasoning.

    Includes information about evidence requirements and
    what evidence is currently available.
    """

    def __init__(
        self,
        message: str,
        required_evidence_types: Optional[List[str]] = None,
        available_evidence_types: Optional[List[str]] = None,
        evidence_quality_score: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.required_evidence_types = required_evidence_types or []
        self.available_evidence_types = available_evidence_types or []
        self.evidence_quality_score = evidence_quality_score
        self.context.metadata.update({
            "required_evidence_types": required_evidence_types,
            "available_evidence_types": available_evidence_types,
            "evidence_quality_score": evidence_quality_score,
        })
        self.context.operation = "evidence_validation"
        self.recoverable = True


class ReasoningChainError(ReasoningError):
    """
    Raised when reasoning chain is broken or inconsistent.

    Includes information about the chain break point and
    the inconsistency that was detected.
    """

    def __init__(
        self,
        message: str,
        break_point: Optional[str] = None,
        inconsistency_type: Optional[str] = None,
        chain_steps: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.break_point = break_point
        self.inconsistency_type = inconsistency_type
        self.chain_steps = chain_steps or []
        self.context.metadata.update({
            "break_point": break_point,
            "inconsistency_type": inconsistency_type,
            "chain_steps": chain_steps,
        })
        self.context.operation = "reasoning_chain_validation"


class ReasoningResourceError(ReasoningError):
    """
    Raised when reasoning process exhausts available resources.

    Includes information about resource usage and limits.
    """

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        usage_limit: Optional[float] = None,
        current_usage: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.usage_limit = usage_limit
        self.current_usage = current_usage
        self.context.metadata.update({
            "resource_type": resource_type,
            "usage_limit": usage_limit,
            "current_usage": current_usage,
        })
        self.context.operation = "resource_management"
        self.recoverable = True
        self.retry_after = 300  # Retry after 5 minutes
