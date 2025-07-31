"""
Domain exceptions for the tournament optimization system.

This module provides a comprehensive exception hierarchy with structured error context,
correlation IDs, and detailed error information for debugging and monitoring.
"""

from .base_exceptions import (
    TournamentOptimizationError,
    DomainError,
    ValidationError,
    BusinessRuleViolationError,
)

from .reasoning_exceptions import (
    ReasoningError,
    ReasoningTimeoutError,
    ReasoningValidationError,
    InsufficientEvidenceError,
    ReasoningChainError,
)

from .research_exceptions import (
    ResearchError,
    EvidenceGatheringError,
    SourceValidationError,
    SearchProviderError,
    CredibilityAnalysisError,
)

from .strategy_exceptions import (
    StrategyError,
    StrategySelectionError,
    StrategyExecutionError,
    TournamentAnalysisError,
    PrioritizationError,
)

from .prediction_exceptions import (
    PredictionError,
    EnsembleError,
    AggregationError,
    CalibrationError,
    ConfidenceError,
)

from .infrastructure_exceptions import (
    InfrastructureError,
    ExternalServiceError,
    NetworkError,
    TimeoutError,
    RateLimitError,
    AuthenticationError,
    ConfigurationError,
)

__all__ = [
    # Base exceptions
    "TournamentOptimizationError",
    "DomainError",
    "ValidationError",
    "BusinessRuleViolationError",

    # Reasoning exceptions
    "ReasoningError",
    "ReasoningTimeoutError",
    "ReasoningValidationError",
    "InsufficientEvidenceError",
    "ReasoningChainError",

    # Research exceptions
    "ResearchError",
    "EvidenceGatheringError",
    "SourceValidationError",
    "SearchProviderError",
    "CredibilityAnalysisError",

    # Strategy exceptions
    "StrategyError",
    "StrategySelectionError",
    "StrategyExecutionError",
    "TournamentAnalysisError",
    "PrioritizationError",

    # Prediction exceptions
    "PredictionError",
    "EnsembleError",
    "AggregationError",
    "CalibrationError",
    "ConfidenceError",

    # Infrastructure exceptions
    "InfrastructureError",
    "ExternalServiceError",
    "NetworkError",
    "TimeoutError",
    "RateLimitError",
    "AuthenticationError",
    "ConfigurationError",
]
