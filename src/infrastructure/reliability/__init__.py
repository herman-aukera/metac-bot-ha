"""
Comprehensive error handling and recovery system for OpenRouter tri-model optimization.

This module provides a complete error handling and recovery infrastructure including:
- Error classification and pattern recognition
- Intelligent fallback strategies (model tier, cross-provider)
- Emergency mode activation and management
- Comprehensive error logging and alerting
- Recovery orchestration and monitoring

Key Components:
- ErrorClassifier: Classifies errors and determines recovery strategies
- ModelTierFallbackManager: Handles model tier fallbacks with performance preservation
- CrossProviderFallbackManager: Manages cross-provider API fallbacks
- EmergencyModeManager: Activates emergency mode for critical failures
- ErrorLoggingAndAlertingSystem: Comprehensive error logging and alerting
- ComprehensiveErrorRecoveryManager: Orchestrates all recovery strategies

Usage:
    from src.infrastructure.reliability import ComprehensiveErrorRecoveryManager

    recovery_manager = ComprehensiveErrorRecoveryManager(
        tri_model_router=router,
        budget_manager=budget_manager
    )

    # Handle an error
    result = await recovery_manager.recover_from_error(error, context)
"""

from .comprehensive_error_recovery import (
    ComprehensiveErrorRecoveryManager,
    RecoveryConfiguration,
    RecoveryResult,
)
from .error_classification import (
    APIError,
    BudgetError,
    ErrorCategory,
    ErrorClassifier,
    ErrorContext,
    ErrorRecoveryManager,
    ErrorSeverity,
    ForecastingError,
    ModelError,
    QualityError,
    RecoveryAction,
    RecoveryStrategy,
)
from .fallback_strategies import (
    AlertConfig,
    CrossProviderFallbackManager,
    EmergencyModeManager,
    ErrorLoggingAndAlertingSystem,
    FallbackOption,
    FallbackResult,
    FallbackTier,
    IntelligentFallbackOrchestrator,
    ModelTierFallbackManager,
    PerformanceLevel,
)

__all__ = [
    # Error Classification
    "ErrorClassifier",
    "ErrorRecoveryManager",
    "ErrorContext",
    "ErrorCategory",
    "ErrorSeverity",
    "RecoveryStrategy",
    "RecoveryAction",
    "ForecastingError",
    "ModelError",
    "BudgetError",
    "APIError",
    "QualityError",
    # Fallback Strategies
    "ModelTierFallbackManager",
    "CrossProviderFallbackManager",
    "EmergencyModeManager",
    "ErrorLoggingAndAlertingSystem",
    "IntelligentFallbackOrchestrator",
    "FallbackOption",
    "FallbackTier",
    "PerformanceLevel",
    "FallbackResult",
    "AlertConfig",
    # Comprehensive Recovery
    "ComprehensiveErrorRecoveryManager",
    "RecoveryConfiguration",
    "RecoveryResult",
]

# Version information
__version__ = "1.0.0"
__author__ = "OpenRouter Tri-Model Optimization Team"
__description__ = (
    "Comprehensive error handling and recovery system for tournament forecasting"
)
