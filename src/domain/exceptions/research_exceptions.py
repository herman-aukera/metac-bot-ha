"""
Research-specific exceptions for the tournament optimization system.
"""

from typing import Optional, Dict, Any, List
from .base_exceptions import DomainError


class ResearchError(DomainError):
    """Base class for research-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.context.component = "research"


class EvidenceGatheringError(ResearchError):
    """
    Raised when evidence gathering fails.

    Includes information about failed sources and
    the type of evidence that could not be gathered.
    """

    def __init__(
        self,
        message: str,
        failed_sources: Optional[List[str]] = None,
        evidence_type: Optional[str] = None,
        search_query: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.failed_sources = failed_sources or []
        self.evidence_type = evidence_type
        self.search_query = search_query
        self.context.metadata.update({
            "failed_sources": failed_sources,
            "evidence_type": evidence_type,
            "search_query": search_query,
        })
        self.context.operation = "evidence_gathering"
        self.recoverable = True


class SourceValidationError(ResearchError):
    """
    Raised when source validation fails.

    Includes information about validation failures and
    the specific sources that failed validation.
    """

    def __init__(
        self,
        message: str,
        invalid_sources: Optional[List[str]] = None,
        validation_errors: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.invalid_sources = invalid_sources or []
        self.validation_errors = validation_errors or {}
        self.context.metadata.update({
            "invalid_sources": invalid_sources,
            "validation_errors": validation_errors,
        })
        self.context.operation = "source_validation"


class SearchProviderError(ResearchError):
    """
    Raised when search provider operations fail.

    Includes information about the provider, error type,
    and whether fallback providers are available.
    """

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        error_type: Optional[str] = None,
        status_code: Optional[int] = None,
        fallback_available: bool = True,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.provider_name = provider_name
        self.error_type = error_type
        self.status_code = status_code
        self.fallback_available = fallback_available
        self.context.metadata.update({
            "provider_name": provider_name,
            "error_type": error_type,
            "status_code": status_code,
            "fallback_available": fallback_available,
        })
        self.context.operation = "search_provider_request"
        self.recoverable = fallback_available


class CredibilityAnalysisError(ResearchError):
    """
    Raised when credibility analysis fails.

    Includes information about the analysis failure and
    the sources that could not be analyzed.
    """

    def __init__(
        self,
        message: str,
        failed_sources: Optional[List[str]] = None,
        analysis_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.failed_sources = failed_sources or []
        self.analysis_type = analysis_type
        self.context.metadata.update({
            "failed_sources": failed_sources,
            "analysis_type": analysis_type,
        })
        self.context.operation = "credibility_analysis"
        self.recoverable = True


class ResearchTimeoutError(ResearchError):
    """
    Raised when research operations exceed time limits.

    Includes information about timeout duration and
    partial results that may be available.
    """

    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        partial_results_available: bool = False,
        completed_sources: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        self.partial_results_available = partial_results_available
        self.completed_sources = completed_sources or []
        self.context.metadata.update({
            "timeout_duration": timeout_duration,
            "partial_results_available": partial_results_available,
            "completed_sources": completed_sources,
        })
        self.context.operation = "research_timeout"
        self.recoverable = partial_results_available
        self.retry_after = 120  # Retry after 2 minutes


class ResearchQualityError(ResearchError):
    """
    Raised when research quality is below acceptable thresholds.

    Includes information about quality metrics and
    the specific quality issues identified.
    """

    def __init__(
        self,
        message: str,
        quality_score: Optional[float] = None,
        quality_threshold: Optional[float] = None,
        quality_issues: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.quality_score = quality_score
        self.quality_threshold = quality_threshold
        self.quality_issues = quality_issues or []
        self.context.metadata.update({
            "quality_score": quality_score,
            "quality_threshold": quality_threshold,
            "quality_issues": quality_issues,
        })
        self.context.operation = "research_quality_validation"
        self.recoverable = True
