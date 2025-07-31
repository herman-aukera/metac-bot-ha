"""
Prediction-specific exceptions for the tournament optimization system.
"""

from typing import Optional, Dict, Any, List
from .base_exceptions import DomainError


class PredictionError(DomainError):
    """Base class for prediction-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.context.component = "prediction"


class EnsembleError(PredictionError):
    """
    Raised when ensemble operations fail.

    Includes information about ensemble configuration and
    the agents that failed to contribute.
    """

    def __init__(
        self,
        message: str,
        ensemble_method: Optional[str] = None,
        failed_agents: Optional[List[str]] = None,
        successful_agents: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.ensemble_method = ensemble_method
        self.failed_agents = failed_agents or []
        self.successful_agents = successful_agents or []
        self.context.metadata.update({
            "ensemble_method": ensemble_method,
            "failed_agents": failed_agents,
            "successful_agents": successful_agents,
        })
        self.context.operation = "ensemble_prediction"
        self.recoverable = bool(successful_agents)


class AggregationError(PredictionError):
    """
    Raised when prediction aggregation fails.

    Includes information about aggregation method and
    the predictions that could not be aggregated.
    """

    def __init__(
        self,
        message: str,
        aggregation_method: Optional[str] = None,
        prediction_count: Optional[int] = None,
        invalid_predictions: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.aggregation_method = aggregation_method
        self.prediction_count = prediction_count
        self.invalid_predictions = invalid_predictions or []
        self.context.metadata.update({
            "aggregation_method": aggregation_method,
            "prediction_count": prediction_count,
            "invalid_predictions": invalid_predictions,
        })
        self.context.operation = "prediction_aggregation"


class CalibrationError(PredictionError):
    """
    Raised when calibration operations fail.

    Includes information about calibration metrics and
    the calibration issues identified.
    """

    def __init__(
        self,
        message: str,
        calibration_score: Optional[float] = None,
        expected_calibration: Optional[float] = None,
        calibration_issues: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.calibration_score = calibration_score
        self.expected_calibration = expected_calibration
        self.calibration_issues = calibration_issues or []
        self.context.metadata.update({
            "calibration_score": calibration_score,
            "expected_calibration": expected_calibration,
            "calibration_issues": calibration_issues,
        })
        self.context.operation = "calibration_validation"
        self.recoverable = True


class ConfidenceError(PredictionError):
    """
    Raised when confidence calculation or validation fails.

    Includes information about confidence metrics and
    the specific confidence issues identified.
    """

    def __init__(
        self,
        message: str,
        confidence_level: Optional[float] = None,
        confidence_basis: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.confidence_level = confidence_level
        self.confidence_basis = confidence_basis
        self.validation_errors = validation_errors or []
        self.context.metadata.update({
            "confidence_level": confidence_level,
            "confidence_basis": confidence_basis,
            "validation_errors": validation_errors,
        })
        self.context.operation = "confidence_validation"


class PredictionValidationError(PredictionError):
    """
    Raised when prediction validation fails.

    Includes information about validation rules and
    the specific validation failures.
    """

    def __init__(
        self,
        message: str,
        prediction_value: Optional[Any] = None,
        validation_rules: Optional[List[str]] = None,
        validation_failures: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.prediction_value = prediction_value
        self.validation_rules = validation_rules or []
        self.validation_failures = validation_failures or {}
        self.context.metadata.update({
            "prediction_value": str(prediction_value) if prediction_value is not None else None,
            "validation_rules": validation_rules,
            "validation_failures": validation_failures,
        })
        self.context.operation = "prediction_validation"


class PredictionTimeoutError(PredictionError):
    """
    Raised when prediction generation exceeds time limits.

    Includes information about timeout duration and
    partial predictions that may be available.
    """

    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        partial_prediction_available: bool = False,
        completed_agents: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        self.partial_prediction_available = partial_prediction_available
        self.completed_agents = completed_agents or []
        self.context.metadata.update({
            "timeout_duration": timeout_duration,
            "partial_prediction_available": partial_prediction_available,
            "completed_agents": completed_agents,
        })
        self.context.operation = "prediction_timeout"
        self.recoverable = partial_prediction_available
        self.retry_after = 180  # Retry after 3 minutes
