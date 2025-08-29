"""
Submission validation and audit trail system for Metaculus predictions.
"""

import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import structlog

from ...domain.entities.prediction import Prediction
from ...domain.entities.question import Question, QuestionType
from ...domain.value_objects.probability import Probability

logger = structlog.get_logger(__name__)


class ValidationResult(Enum):
    """Validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


class SubmissionStatus(Enum):
    """Submission status tracking."""

    PENDING = "pending"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    FAILED = "failed"
    DRY_RUN = "dry_run"


@dataclass
class ValidationError:
    """Validation error details."""

    field: str
    message: str
    severity: ValidationResult
    code: str


@dataclass
class SubmissionRecord:
    """Audit trail record for submissions."""

    submission_id: str
    question_id: str
    prediction_value: Union[float, int, str]
    confidence: Optional[float]
    reasoning: str
    timestamp: datetime
    status: SubmissionStatus
    validation_errors: List[ValidationError]
    metadata: Dict[str, Any]
    dry_run: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "submission_id": self.submission_id,
            "question_id": self.question_id,
            "prediction_value": self.prediction_value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "validation_errors": [
                {
                    "field": error.field,
                    "message": error.message,
                    "severity": error.severity.value,
                    "code": error.code,
                }
                for error in self.validation_errors
            ],
            "metadata": self.metadata,
            "dry_run": self.dry_run,
        }


class SubmissionValidator:
    """Validates predictions before submission to Metaculus."""

    def __init__(self, tournament_mode: bool = False):
        self.validation_rules = self._initialize_validation_rules()
        self.tournament_mode = tournament_mode

    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules for different question types."""
        return {
            "binary": {
                "min_value": 0.0,
                "max_value": 1.0,
                "required_fields": ["prediction_value"],
                "value_type": float,
            },
            "continuous": {
                "required_fields": ["prediction_value", "min_value", "max_value"],
                "value_type": float,
            },
            "multiple_choice": {
                "required_fields": ["prediction_value", "choices"],
                "value_type": int,
            },
            "general": {
                "max_reasoning_length": 10000,
                "min_reasoning_length": 10,
                "required_fields": ["question_id", "prediction_value"],
            },
        }

    def validate_prediction(
        self, question: Question, prediction: Union[Prediction, Dict[str, Any]]
    ) -> tuple[ValidationResult, List[ValidationError]]:
        """
        Validate a prediction against question requirements.

        Args:
            question: The question being predicted on
            prediction: The prediction to validate

        Returns:
            Tuple of (validation_result, list_of_errors)
        """
        logger.info(
            "Validating prediction",
            question_id=question.id,
            question_type=question.question_type.value,
        )

        errors = []

        # Extract prediction data
        if isinstance(prediction, Prediction):
            pred_data = self._extract_prediction_data(prediction)
        else:
            pred_data = prediction

        # Validate required fields
        errors.extend(self._validate_required_fields(pred_data))

        # Validate question-specific requirements
        errors.extend(self._validate_question_specific(question, pred_data))

        # Validate prediction value
        errors.extend(self._validate_prediction_value(question, pred_data))

        # Validate reasoning
        errors.extend(self._validate_reasoning(pred_data))

        # Validate timing
        errors.extend(self._validate_timing(question))

        # Determine overall result
        if any(error.severity == ValidationResult.INVALID for error in errors):
            result = ValidationResult.INVALID
        elif any(error.severity == ValidationResult.WARNING for error in errors):
            result = ValidationResult.WARNING
        else:
            result = ValidationResult.VALID

        logger.info(
            "Prediction validation completed",
            result=result.value,
            error_count=len(errors),
        )

        return result, errors

    def _extract_prediction_data(self, prediction: Prediction) -> Dict[str, Any]:
        """Extract data from Prediction object."""
        data = {
            "question_id": prediction.question_id,
            "reasoning": prediction.reasoning,
            "confidence": (
                prediction.confidence.value if prediction.confidence else None
            ),
        }

        # Extract prediction value based on type
        if prediction.result.binary_probability is not None:
            data["prediction_value"] = prediction.result.binary_probability
        elif prediction.result.numeric_value is not None:
            data["prediction_value"] = prediction.result.numeric_value
        elif prediction.result.choice_index is not None:
            data["prediction_value"] = prediction.result.choice_index

        return data

    def _validate_required_fields(
        self, pred_data: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate required fields are present."""
        errors = []
        required_fields = self.validation_rules["general"]["required_fields"]

        for field in required_fields:
            if field not in pred_data or pred_data[field] is None:
                errors.append(
                    ValidationError(
                        field=field,
                        message=f"Required field '{field}' is missing or None",
                        severity=ValidationResult.INVALID,
                        code="MISSING_REQUIRED_FIELD",
                    )
                )

        return errors

    def _validate_question_specific(
        self, question: Question, pred_data: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate question-specific requirements."""
        errors = []
        question_type = question.question_type.value.lower()

        if question_type in self.validation_rules:
            rules = self.validation_rules[question_type]

            # Check required fields for this question type
            for field in rules.get("required_fields", []):
                if field not in pred_data:
                    # For min/max values, check question metadata
                    if field in ["min_value", "max_value"]:
                        if (
                            not hasattr(question, field)
                            or getattr(question, field) is None
                        ):
                            errors.append(
                                ValidationError(
                                    field=field,
                                    message=f"Question missing {field} for continuous prediction",
                                    severity=ValidationResult.INVALID,
                                    code="MISSING_QUESTION_BOUNDS",
                                )
                            )
                    elif field == "choices":
                        if not question.choices:
                            errors.append(
                                ValidationError(
                                    field=field,
                                    message="Question missing choices for multiple choice prediction",
                                    severity=ValidationResult.INVALID,
                                    code="MISSING_QUESTION_CHOICES",
                                )
                            )

        return errors

    def _validate_prediction_value(
        self, question: Question, pred_data: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate the prediction value itself."""
        errors = []

        if "prediction_value" not in pred_data:
            return errors  # Already handled in required fields

        value = pred_data["prediction_value"]
        question_type = question.question_type

        if question_type == QuestionType.BINARY:
            errors.extend(self._validate_binary_prediction(value))
        elif question_type == QuestionType.CONTINUOUS:
            errors.extend(self._validate_continuous_prediction(question, value))
        elif question_type == QuestionType.MULTIPLE_CHOICE:
            errors.extend(self._validate_multiple_choice_prediction(question, value))

        return errors

    def _validate_binary_prediction(self, value: Any) -> List[ValidationError]:
        """Validate binary prediction value."""
        errors = []

        # Check type
        if not isinstance(value, (int, float)):
            errors.append(
                ValidationError(
                    field="prediction_value",
                    message=f"Binary prediction must be numeric, got {type(value).__name__}",
                    severity=ValidationResult.INVALID,
                    code="INVALID_VALUE_TYPE",
                )
            )
            return errors

        # Check range
        if not (0.0 <= value <= 1.0):
            errors.append(
                ValidationError(
                    field="prediction_value",
                    message=f"Binary prediction must be between 0.0 and 1.0, got {value}",
                    severity=ValidationResult.INVALID,
                    code="VALUE_OUT_OF_RANGE",
                )
            )

        # Check for extreme values (warning)
        if value < 0.01 or value > 0.99:
            errors.append(
                ValidationError(
                    field="prediction_value",
                    message=f"Extreme prediction value {value} - consider if this is intended",
                    severity=ValidationResult.WARNING,
                    code="EXTREME_PREDICTION_VALUE",
                )
            )

        return errors

    def _validate_continuous_prediction(
        self, question: Question, value: Any
    ) -> List[ValidationError]:
        """Validate continuous prediction value."""
        errors = []

        # Check type
        if not isinstance(value, (int, float)):
            errors.append(
                ValidationError(
                    field="prediction_value",
                    message=f"Continuous prediction must be numeric, got {type(value).__name__}",
                    severity=ValidationResult.INVALID,
                    code="INVALID_VALUE_TYPE",
                )
            )
            return errors

        # Check bounds if available
        if hasattr(question, "min_value") and question.min_value is not None:
            if value < question.min_value:
                errors.append(
                    ValidationError(
                        field="prediction_value",
                        message=f"Prediction {value} below minimum {question.min_value}",
                        severity=ValidationResult.INVALID,
                        code="VALUE_BELOW_MINIMUM",
                    )
                )

        if hasattr(question, "max_value") and question.max_value is not None:
            if value > question.max_value:
                errors.append(
                    ValidationError(
                        field="prediction_value",
                        message=f"Prediction {value} above maximum {question.max_value}",
                        severity=ValidationResult.INVALID,
                        code="VALUE_ABOVE_MAXIMUM",
                    )
                )

        return errors

    def _validate_multiple_choice_prediction(
        self, question: Question, value: Any
    ) -> List[ValidationError]:
        """Validate multiple choice prediction value."""
        errors = []

        # Check type
        if not isinstance(value, int):
            errors.append(
                ValidationError(
                    field="prediction_value",
                    message=f"Multiple choice prediction must be integer, got {type(value).__name__}",
                    severity=ValidationResult.INVALID,
                    code="INVALID_VALUE_TYPE",
                )
            )
            return errors

        # Check choice index bounds
        if question.choices:
            if not (0 <= value < len(question.choices)):
                errors.append(
                    ValidationError(
                        field="prediction_value",
                        message=f"Choice index {value} out of range [0, {len(question.choices)-1}]",
                        severity=ValidationResult.INVALID,
                        code="CHOICE_INDEX_OUT_OF_RANGE",
                    )
                )

        return errors

    def _validate_reasoning(self, pred_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate reasoning text."""
        errors = []

        reasoning = pred_data.get("reasoning", "")
        if not reasoning:
            errors.append(
                ValidationError(
                    field="reasoning",
                    message="Reasoning is required for all predictions",
                    severity=ValidationResult.WARNING,
                    code="MISSING_REASONING",
                )
            )
            return errors

        # Check length
        min_length = self.validation_rules["general"]["min_reasoning_length"]
        max_length = self.validation_rules["general"]["max_reasoning_length"]

        if len(reasoning) < min_length:
            errors.append(
                ValidationError(
                    field="reasoning",
                    message=f"Reasoning too short ({len(reasoning)} chars, minimum {min_length})",
                    severity=ValidationResult.WARNING,
                    code="REASONING_TOO_SHORT",
                )
            )

        if len(reasoning) > max_length:
            errors.append(
                ValidationError(
                    field="reasoning",
                    message=f"Reasoning too long ({len(reasoning)} chars, maximum {max_length})",
                    severity=ValidationResult.INVALID,
                    code="REASONING_TOO_LONG",
                )
            )

        return errors

    def _validate_timing(self, question: Question) -> List[ValidationError]:
        """Validate submission timing."""
        errors = []

        now = datetime.now(timezone.utc)

        # Check if question is still open
        if question.close_time and now >= question.close_time:
            errors.append(
                ValidationError(
                    field="timing",
                    message=f"Question closed at {question.close_time.isoformat()}",
                    severity=ValidationResult.INVALID,
                    code="QUESTION_CLOSED",
                )
            )

        # Warning for questions closing soon
        if question.close_time:
            time_until_close = question.close_time - now
            if time_until_close.total_seconds() < 3600:  # 1 hour
                errors.append(
                    ValidationError(
                        field="timing",
                        message=f"Question closes in {time_until_close.total_seconds()/60:.0f} minutes",
                        severity=ValidationResult.WARNING,
                        code="QUESTION_CLOSING_SOON",
                    )
                )

        return errors

    def format_prediction_for_submission(
        self, question: Question, prediction: Union[Prediction, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format prediction data for Metaculus API submission.

        Args:
            question: The question being predicted on
            prediction: The prediction to format

        Returns:
            Formatted prediction data for API submission
        """
        logger.info("Formatting prediction for submission", question_id=question.id)

        # Extract prediction data
        if isinstance(prediction, Prediction):
            pred_data = self._extract_prediction_data(prediction)
        else:
            pred_data = prediction.copy()

        # Format according to Metaculus API requirements
        formatted = {
            "prediction": pred_data["prediction_value"],
            "comment": pred_data.get("reasoning", ""),
        }

        # Add question-type specific formatting
        if question.question_type == QuestionType.BINARY:
            # Ensure prediction is float for binary questions
            formatted["prediction"] = float(formatted["prediction"])
            formatted["void"] = False

        elif question.question_type == QuestionType.CONTINUOUS:
            # For continuous questions, may need additional formatting
            formatted["prediction"] = float(formatted["prediction"])

        elif question.question_type == QuestionType.MULTIPLE_CHOICE:
            # For multiple choice, prediction should be choice index
            formatted["prediction"] = int(formatted["prediction"])

        # Add confidence if available
        if pred_data.get("confidence") is not None:
            formatted["confidence"] = pred_data["confidence"]

        # Add tournament-specific formatting
        if self.tournament_mode:
            formatted = self._apply_tournament_formatting(
                formatted, question, pred_data
            )

        logger.info(
            "Prediction formatted for submission",
            question_type=question.question_type.value,
            formatted_keys=list(formatted.keys()),
        )

        return formatted

    def _apply_tournament_formatting(
        self, formatted: Dict[str, Any], question: Question, pred_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply tournament-specific formatting enhancements."""
        # Add tournament metadata
        formatted["tournament_metadata"] = {
            "submission_timestamp": datetime.now(timezone.utc).isoformat(),
            "question_category": question.metadata.get("category", "unknown"),
            "tournament_priority": question.metadata.get(
                "tournament_priority", "medium"
            ),
            "agent_type": pred_data.get("agent_type", "unknown"),
            "reasoning_method": pred_data.get("reasoning_method", "standard"),
        }

        # Add validation checksums for integrity
        formatted["validation_checksum"] = self._calculate_validation_checksum(
            formatted
        )

        # Add submission context
        formatted["submission_context"] = {
            "time_until_close": (
                (question.close_time - datetime.now(timezone.utc)).total_seconds()
                if question.close_time
                else None
            ),
            "question_age": (
                datetime.now(timezone.utc)
                - (question.created_at or datetime.now(timezone.utc))
            ).total_seconds(),
            "urgency_score": question.metadata.get("urgency_score", 0.0),
        }

        return formatted

    def _calculate_validation_checksum(self, formatted: Dict[str, Any]) -> str:
        """Calculate validation checksum for submission integrity."""
        import hashlib

        # Create a deterministic string from core prediction data
        core_data = {
            "prediction": formatted.get("prediction"),
            "comment": formatted.get("comment", ""),
            "confidence": formatted.get("confidence"),
        }

        data_string = json.dumps(core_data, sort_keys=True)
        return hashlib.md5(data_string.encode()).hexdigest()

    def validate_submission_integrity(
        self,
        formatted_prediction: Dict[str, Any],
        expected_checksum: Optional[str] = None,
    ) -> bool:
        """Validate submission integrity using checksum."""
        if not expected_checksum:
            expected_checksum = formatted_prediction.get("validation_checksum")

        if not expected_checksum:
            return False

        calculated_checksum = self._calculate_validation_checksum(formatted_prediction)
        return calculated_checksum == expected_checksum

    def simulate_tournament_conditions(
        self,
        question: Question,
        prediction: Union[Prediction, Dict[str, Any]],
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate tournament conditions for dry-run validation.

        Args:
            question: The question being predicted on
            prediction: The prediction to simulate
            tournament_context: Tournament context for simulation

        Returns:
            Simulation results with validation and timing analysis
        """
        logger.info(
            "Simulating tournament conditions",
            question_id=question.id,
            tournament_context=bool(tournament_context),
        )

        simulation_results = {
            "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
            "question_analysis": self._analyze_question_for_simulation(question),
            "prediction_analysis": self._analyze_prediction_for_simulation(
                question, prediction
            ),
            "tournament_simulation": self._simulate_tournament_dynamics(
                question, tournament_context
            ),
            "validation_results": None,
            "timing_analysis": self._analyze_submission_timing(question),
            "risk_assessment": self._assess_submission_risk(question, prediction),
            "recommendations": [],
        }

        # Run full validation
        result, errors = self.validate_prediction(question, prediction)
        simulation_results["validation_results"] = {
            "result": result.value,
            "errors": [
                {
                    "field": error.field,
                    "message": error.message,
                    "severity": error.severity.value,
                    "code": error.code,
                }
                for error in errors
            ],
        }

        # Generate recommendations
        simulation_results["recommendations"] = (
            self._generate_simulation_recommendations(
                question, prediction, simulation_results
            )
        )

        logger.info(
            "Tournament simulation completed",
            validation_result=result.value,
            error_count=len(errors),
            recommendation_count=len(simulation_results["recommendations"]),
        )

        return simulation_results

    def _analyze_question_for_simulation(self, question: Question) -> Dict[str, Any]:
        """Analyze question characteristics for simulation."""
        now = datetime.now(timezone.utc)

        return {
            "question_id": question.id,
            "question_type": question.question_type.value,
            "category": question.metadata.get("category", "unknown"),
            "close_time": (
                question.close_time.isoformat() if question.close_time else None
            ),
            "time_until_close": (
                (question.close_time - now).total_seconds()
                if question.close_time
                else None
            ),
            "is_urgent": (
                (question.close_time - now).total_seconds() < 86400
                if question.close_time
                else False
            ),
            "is_critical": (
                (question.close_time - now).total_seconds() < 21600
                if question.close_time
                else False
            ),
            "community_prediction": question.metadata.get("community_prediction"),
            "prediction_count": question.metadata.get("prediction_count", 0),
            "tournament_priority": question.metadata.get(
                "tournament_priority", "medium"
            ),
        }

    def _analyze_prediction_for_simulation(
        self, question: Question, prediction: Union[Prediction, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze prediction characteristics for simulation."""
        if isinstance(prediction, Prediction):
            pred_data = self._extract_prediction_data(prediction)
        else:
            pred_data = prediction

        analysis = {
            "prediction_value": pred_data.get("prediction_value"),
            "confidence": pred_data.get("confidence"),
            "reasoning_length": len(pred_data.get("reasoning", "")),
            "agent_type": pred_data.get("agent_type", "unknown"),
            "reasoning_method": pred_data.get("reasoning_method", "standard"),
        }

        # Add prediction-specific analysis
        if question.question_type == QuestionType.BINARY:
            analysis["extremeness"] = min(
                pred_data.get("prediction_value", 0.5),
                1 - pred_data.get("prediction_value", 0.5),
            )
            analysis["is_contrarian"] = self._is_contrarian_prediction(
                question, pred_data.get("prediction_value")
            )

        return analysis

    def _is_contrarian_prediction(
        self, question: Question, prediction_value: float
    ) -> bool:
        """Check if prediction is contrarian to community consensus."""
        community_pred = question.metadata.get("community_prediction")
        if community_pred is None:
            return False

        # Consider contrarian if more than 0.2 away from community prediction
        return abs(prediction_value - community_pred) > 0.2

    def _simulate_tournament_dynamics(
        self, question: Question, tournament_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simulate tournament-specific dynamics."""
        simulation = {
            "tournament_active": tournament_context is not None,
            "competitive_pressure": (
                "high"
                if question.metadata.get("tournament_priority") == "critical"
                else "medium"
            ),
            "market_efficiency": self._estimate_market_efficiency(question),
            "scoring_impact": self._estimate_scoring_impact(
                question, tournament_context
            ),
            "strategic_considerations": self._analyze_strategic_considerations(
                question, tournament_context
            ),
        }

        return simulation

    def _estimate_market_efficiency(self, question: Question) -> str:
        """Estimate market efficiency based on question characteristics."""
        prediction_count = question.metadata.get("prediction_count", 0)
        community_pred = question.metadata.get("community_prediction")

        if prediction_count < 10:
            return "low"
        elif prediction_count < 50:
            return "medium"
        elif community_pred and (community_pred < 0.1 or community_pred > 0.9):
            return "potentially_inefficient"
        else:
            return "high"

    def _estimate_scoring_impact(
        self, question: Question, tournament_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate potential scoring impact of prediction."""
        if not tournament_context:
            return {"impact": "unknown", "reason": "no_tournament_context"}

        # Base impact on question priority and tournament position
        priority = question.metadata.get("tournament_priority", "medium")
        current_ranking = tournament_context.get("current_ranking", 50)
        total_participants = tournament_context.get("participant_count", 100)

        if priority == "critical":
            impact = "high"
        elif priority == "high" or current_ranking > total_participants * 0.7:
            impact = "medium"
        else:
            impact = "low"

        return {
            "impact": impact,
            "priority": priority,
            "ranking_context": f"{current_ranking}/{total_participants}",
            "potential_rank_change": self._estimate_rank_change_potential(
                question, tournament_context
            ),
        }

    def _estimate_rank_change_potential(
        self, question: Question, tournament_context: Dict[str, Any]
    ) -> int:
        """Estimate potential ranking change from this prediction."""
        # Simplified estimation based on question priority and current position
        priority_multiplier = {"critical": 3, "high": 2, "medium": 1, "low": 0}

        base_change = priority_multiplier.get(
            question.metadata.get("tournament_priority", "medium"), 1
        )

        # Adjust based on current ranking (more potential if lower ranked)
        current_ranking = tournament_context.get("current_ranking", 50)
        total_participants = tournament_context.get("participant_count", 100)

        if current_ranking > total_participants * 0.8:
            base_change *= 2
        elif current_ranking < total_participants * 0.2:
            base_change = max(1, base_change // 2)

        return base_change

    def _analyze_strategic_considerations(
        self, question: Question, tournament_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Analyze strategic considerations for tournament play."""
        considerations = []

        # Timing considerations
        if question.close_time:
            time_until_close = (
                question.close_time - datetime.now(timezone.utc)
            ).total_seconds()
            if time_until_close < 21600:  # 6 hours
                considerations.append("urgent_deadline")
            elif time_until_close < 86400:  # 24 hours
                considerations.append("approaching_deadline")

        # Market considerations
        community_pred = question.metadata.get("community_prediction")
        if community_pred and (community_pred < 0.1 or community_pred > 0.9):
            considerations.append("extreme_consensus")

        prediction_count = question.metadata.get("prediction_count", 0)
        if prediction_count < 10:
            considerations.append("low_participation")

        # Tournament considerations
        if tournament_context:
            completion_rate = tournament_context.get("completion_rate", 0)
            if completion_rate < 0.5:
                considerations.append("low_completion_rate")

            current_ranking = tournament_context.get("current_ranking", 50)
            total_participants = tournament_context.get("participant_count", 100)
            if current_ranking > total_participants * 0.8:
                considerations.append("need_aggressive_strategy")

        return considerations

    def _analyze_submission_timing(self, question: Question) -> Dict[str, Any]:
        """Analyze optimal submission timing."""
        now = datetime.now(timezone.utc)

        if not question.close_time:
            return {"status": "no_deadline", "recommendation": "submit_when_ready"}

        time_until_close = (question.close_time - now).total_seconds()
        hours_until_close = time_until_close / 3600

        if hours_until_close < 1:
            status = "critical"
            recommendation = "submit_immediately"
        elif hours_until_close < 6:
            status = "urgent"
            recommendation = "submit_within_hour"
        elif hours_until_close < 24:
            status = "soon"
            recommendation = "submit_within_6_hours"
        else:
            status = "normal"
            recommendation = "monitor_and_optimize"

        return {
            "status": status,
            "hours_until_close": hours_until_close,
            "recommendation": recommendation,
            "optimal_window": self._calculate_optimal_submission_window(question),
        }

    def _calculate_optimal_submission_window(
        self, question: Question
    ) -> Dict[str, Any]:
        """Calculate optimal submission window."""
        if not question.close_time or not question.created_at:
            return {"start": None, "end": None, "reason": "insufficient_timing_data"}

        # Optimal window is typically the last 25% of question lifetime
        question_lifetime = question.close_time - question.created_at
        optimal_start = question.close_time - (question_lifetime * 0.25)

        return {
            "start": optimal_start.isoformat(),
            "end": question.close_time.isoformat(),
            "reason": "last_quarter_of_lifetime",
            "currently_in_window": datetime.now(timezone.utc) >= optimal_start,
        }

    def _assess_submission_risk(
        self, question: Question, prediction: Union[Prediction, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess risks associated with submission."""
        if isinstance(prediction, Prediction):
            pred_data = self._extract_prediction_data(prediction)
        else:
            pred_data = prediction

        risks = []
        risk_level = "low"

        # Confidence-based risk
        confidence = pred_data.get("confidence", 0.5)
        if confidence < 0.3:
            risks.append("low_confidence_prediction")
            risk_level = "high"
        elif confidence < 0.5:
            risks.append("moderate_confidence_prediction")
            risk_level = max(risk_level, "medium")

        # Extreme prediction risk
        if question.question_type == QuestionType.BINARY:
            pred_value = pred_data.get("prediction_value", 0.5)
            if pred_value < 0.05 or pred_value > 0.95:
                risks.append("extreme_prediction_value")
                risk_level = max(risk_level, "medium")

        # Timing risk
        if question.close_time:
            time_until_close = (
                question.close_time - datetime.now(timezone.utc)
            ).total_seconds()
            if time_until_close < 3600:  # 1 hour
                risks.append("very_tight_deadline")
                risk_level = max(risk_level, "medium")

        # Contrarian risk
        if self._is_contrarian_prediction(
            question, pred_data.get("prediction_value", 0.5)
        ):
            risks.append("contrarian_position")
            risk_level = max(risk_level, "medium")

        return {
            "risk_level": risk_level,
            "identified_risks": risks,
            "mitigation_suggestions": self._suggest_risk_mitigations(risks),
        }

    def _suggest_risk_mitigations(self, risks: List[str]) -> List[str]:
        """Suggest risk mitigation strategies."""
        mitigations = []

        if "low_confidence_prediction" in risks:
            mitigations.append("consider_additional_research")
            mitigations.append("use_conservative_prediction")

        if "extreme_prediction_value" in risks:
            mitigations.append("double_check_reasoning")
            mitigations.append("consider_moderate_adjustment")

        if "very_tight_deadline" in risks:
            mitigations.append("submit_immediately")
            mitigations.append("use_existing_analysis")

        if "contrarian_position" in risks:
            mitigations.append("verify_unique_insights")
            mitigations.append("document_reasoning_thoroughly")

        return mitigations

    def _generate_simulation_recommendations(
        self,
        question: Question,
        prediction: Union[Prediction, Dict[str, Any]],
        simulation_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on simulation results."""
        recommendations = []

        # Validation-based recommendations
        validation_result = simulation_results["validation_results"]["result"]
        if validation_result == "invalid":
            recommendations.append(
                {
                    "type": "validation",
                    "priority": "critical",
                    "message": "Prediction failed validation - must fix before submission",
                    "action": "fix_validation_errors",
                }
            )

        # Timing recommendations
        timing = simulation_results["timing_analysis"]
        if timing["status"] == "critical":
            recommendations.append(
                {
                    "type": "timing",
                    "priority": "critical",
                    "message": "Question closes very soon - submit immediately",
                    "action": "submit_now",
                }
            )

        # Risk-based recommendations
        risk_assessment = simulation_results["risk_assessment"]
        if risk_assessment["risk_level"] == "high":
            recommendations.append(
                {
                    "type": "risk",
                    "priority": "high",
                    "message": "High-risk prediction detected",
                    "action": "review_and_mitigate",
                    "mitigations": risk_assessment["mitigation_suggestions"],
                }
            )

        # Strategic recommendations
        strategic_considerations = simulation_results["tournament_simulation"][
            "strategic_considerations"
        ]
        if "extreme_consensus" in strategic_considerations:
            recommendations.append(
                {
                    "type": "strategy",
                    "priority": "medium",
                    "message": "Extreme community consensus detected - consider contrarian opportunity",
                    "action": "evaluate_contrarian_position",
                }
            )

        return recommendations


class AuditTrailManager:
    """Manages audit trail for prediction submissions."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "logs/submission_audit.jsonl"
        self.submissions: Dict[str, SubmissionRecord] = {}
        self.confirmation_callbacks: List[callable] = []

    def create_submission_record(
        self,
        question_id: str,
        prediction_value: Union[float, int, str],
        reasoning: str,
        confidence: Optional[float] = None,
        dry_run: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SubmissionRecord:
        """Create a new submission record."""
        submission_id = str(uuid.uuid4())

        record = SubmissionRecord(
            submission_id=submission_id,
            question_id=question_id,
            prediction_value=prediction_value,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(timezone.utc),
            status=SubmissionStatus.DRY_RUN if dry_run else SubmissionStatus.PENDING,
            validation_errors=[],
            metadata=metadata or {},
            dry_run=dry_run,
        )

        self.submissions[submission_id] = record

        logger.info(
            "Created submission record",
            submission_id=submission_id,
            question_id=question_id,
            dry_run=dry_run,
        )

        return record

    def update_submission_status(
        self,
        submission_id: str,
        status: SubmissionStatus,
        validation_errors: Optional[List[ValidationError]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update submission record status."""
        if submission_id not in self.submissions:
            logger.error("Submission record not found", submission_id=submission_id)
            return

        record = self.submissions[submission_id]
        record.status = status

        if validation_errors is not None:
            record.validation_errors = validation_errors

        if metadata:
            record.metadata.update(metadata)

        logger.info(
            "Updated submission status",
            submission_id=submission_id,
            status=status.value,
        )

    def get_submission_record(self, submission_id: str) -> Optional[SubmissionRecord]:
        """Get submission record by ID."""
        return self.submissions.get(submission_id)

    def get_submissions_by_question(self, question_id: str) -> List[SubmissionRecord]:
        """Get all submissions for a question."""
        return [
            record
            for record in self.submissions.values()
            if record.question_id == question_id
        ]

    def get_recent_submissions(self, limit: int = 100) -> List[SubmissionRecord]:
        """Get recent submissions ordered by timestamp."""
        sorted_submissions = sorted(
            self.submissions.values(), key=lambda x: x.timestamp, reverse=True
        )
        return sorted_submissions[:limit]

    def persist_to_storage(self) -> None:
        """Persist audit trail to storage."""
        try:
            import os

            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

            with open(self.storage_path, "w") as f:
                for record in self.submissions.values():
                    f.write(json.dumps(record.to_dict()) + "\n")

            logger.info(
                "Audit trail persisted",
                path=self.storage_path,
                record_count=len(self.submissions),
            )

        except Exception as e:
            logger.error("Failed to persist audit trail", error=str(e))

    def load_from_storage(self) -> None:
        """Load audit trail from storage."""
        try:
            if not os.path.exists(self.storage_path):
                return

            with open(self.storage_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())

                    # Reconstruct validation errors
                    validation_errors = [
                        ValidationError(
                            field=error["field"],
                            message=error["message"],
                            severity=ValidationResult(error["severity"]),
                            code=error["code"],
                        )
                        for error in data["validation_errors"]
                    ]

                    record = SubmissionRecord(
                        submission_id=data["submission_id"],
                        question_id=data["question_id"],
                        prediction_value=data["prediction_value"],
                        confidence=data["confidence"],
                        reasoning=data["reasoning"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        status=SubmissionStatus(data["status"]),
                        validation_errors=validation_errors,
                        metadata=data["metadata"],
                        dry_run=data["dry_run"],
                    )

                    self.submissions[record.submission_id] = record

            logger.info(
                "Audit trail loaded",
                path=self.storage_path,
                record_count=len(self.submissions),
            )

        except Exception as e:
            logger.error("Failed to load audit trail", error=str(e))

    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate audit report summary."""
        total_submissions = len(self.submissions)

        status_counts = {}
        for status in SubmissionStatus:
            status_counts[status.value] = sum(
                1 for record in self.submissions.values() if record.status == status
            )

        dry_run_count = sum(1 for record in self.submissions.values() if record.dry_run)

        validation_error_counts = {}
        for record in self.submissions.values():
            for error in record.validation_errors:
                validation_error_counts[error.code] = (
                    validation_error_counts.get(error.code, 0) + 1
                )

        return {
            "total_submissions": total_submissions,
            "status_distribution": status_counts,
            "dry_run_submissions": dry_run_count,
            "validation_error_distribution": validation_error_counts,
            "recent_submissions": [
                {
                    "submission_id": record.submission_id,
                    "question_id": record.question_id,
                    "timestamp": record.timestamp.isoformat(),
                    "status": record.status.value,
                    "dry_run": record.dry_run,
                }
                for record in self.get_recent_submissions(10)
            ],
        }

    def add_confirmation_callback(self, callback: callable) -> None:
        """Add callback for submission confirmations."""
        self.confirmation_callbacks.append(callback)

    def confirm_submission(
        self, submission_id: str, api_response: Dict[str, Any], success: bool = True
    ) -> None:
        """
        Confirm submission with API response details.

        Args:
            submission_id: ID of the submission
            api_response: Response from Metaculus API
            success: Whether submission was successful
        """
        if submission_id not in self.submissions:
            logger.error(
                "Cannot confirm unknown submission", submission_id=submission_id
            )
            return

        record = self.submissions[submission_id]

        # Update status based on success
        new_status = SubmissionStatus.SUBMITTED if success else SubmissionStatus.FAILED

        # Add confirmation metadata
        confirmation_metadata = {
            "api_response": api_response,
            "confirmation_timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "response_code": api_response.get("status_code"),
            "response_message": api_response.get("message", ""),
            "metaculus_prediction_id": api_response.get("prediction_id"),
            "submission_confirmed": success,
        }

        self.update_submission_status(
            submission_id, new_status, metadata=confirmation_metadata
        )

        # Trigger confirmation callbacks
        for callback in self.confirmation_callbacks:
            try:
                callback(record, api_response, success)
            except Exception as e:
                logger.error("Confirmation callback failed", error=str(e))

        logger.info(
            "Submission confirmation recorded",
            submission_id=submission_id,
            success=success,
            status=new_status.value,
        )

    def track_submission_attempt(
        self, submission_id: str, attempt_number: int, error: Optional[str] = None
    ) -> None:
        """Track submission attempts for retry logic."""
        if submission_id not in self.submissions:
            logger.error(
                "Cannot track attempt for unknown submission",
                submission_id=submission_id,
            )
            return

        record = self.submissions[submission_id]

        # Initialize attempt tracking if not exists
        if "submission_attempts" not in record.metadata:
            record.metadata["submission_attempts"] = []

        attempt_info = {
            "attempt_number": attempt_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "success": error is None,
        }

        record.metadata["submission_attempts"].append(attempt_info)

        logger.info(
            "Submission attempt tracked",
            submission_id=submission_id,
            attempt=attempt_number,
            success=error is None,
        )

    def get_submission_history(
        self,
        question_id: Optional[str] = None,
        status_filter: Optional[SubmissionStatus] = None,
        dry_run_filter: Optional[bool] = None,
        limit: int = 100,
    ) -> List[SubmissionRecord]:
        """
        Get filtered submission history.

        Args:
            question_id: Filter by question ID
            status_filter: Filter by submission status
            dry_run_filter: Filter by dry run status
            limit: Maximum number of records to return

        Returns:
            Filtered list of submission records
        """
        filtered_submissions = []

        for record in self.submissions.values():
            # Apply filters
            if question_id and record.question_id != question_id:
                continue
            if status_filter and record.status != status_filter:
                continue
            if dry_run_filter is not None and record.dry_run != dry_run_filter:
                continue

            filtered_submissions.append(record)

        # Sort by timestamp (most recent first)
        filtered_submissions.sort(key=lambda x: x.timestamp, reverse=True)

        return filtered_submissions[:limit]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from audit trail."""
        total_submissions = len(self.submissions)

        if total_submissions == 0:
            return {"error": "No submissions to analyze"}

        # Success rate
        successful_submissions = sum(
            1
            for record in self.submissions.values()
            if record.status == SubmissionStatus.SUBMITTED and not record.dry_run
        )

        real_submissions = sum(
            1 for record in self.submissions.values() if not record.dry_run
        )

        success_rate = (
            successful_submissions / real_submissions if real_submissions > 0 else 0
        )

        # Average validation errors
        total_validation_errors = sum(
            len(record.validation_errors) for record in self.submissions.values()
        )
        avg_validation_errors = total_validation_errors / total_submissions

        # Submission timing analysis
        timing_analysis = self._analyze_submission_timing()

        # Question category performance
        category_performance = self._analyze_category_performance()

        return {
            "total_submissions": total_submissions,
            "real_submissions": real_submissions,
            "dry_run_submissions": total_submissions - real_submissions,
            "success_rate": success_rate,
            "average_validation_errors": avg_validation_errors,
            "timing_analysis": timing_analysis,
            "category_performance": category_performance,
            "recent_performance": self._analyze_recent_performance(),
        }

    def _analyze_submission_timing(self) -> Dict[str, Any]:
        """Analyze submission timing patterns."""
        timing_data = []

        for record in self.submissions.values():
            if record.dry_run:
                continue

            # Extract timing information from metadata
            submission_context = record.metadata.get("submission_context", {})
            time_until_close = submission_context.get("time_until_close")

            if time_until_close is not None:
                timing_data.append(
                    {
                        "time_until_close": time_until_close,
                        "success": record.status == SubmissionStatus.SUBMITTED,
                    }
                )

        if not timing_data:
            return {"error": "No timing data available"}

        # Analyze timing patterns
        urgent_submissions = [
            t for t in timing_data if t["time_until_close"] < 21600
        ]  # 6 hours
        normal_submissions = [t for t in timing_data if t["time_until_close"] >= 21600]

        urgent_success_rate = (
            sum(1 for t in urgent_submissions if t["success"]) / len(urgent_submissions)
            if urgent_submissions
            else 0
        )
        normal_success_rate = (
            sum(1 for t in normal_submissions if t["success"]) / len(normal_submissions)
            if normal_submissions
            else 0
        )

        return {
            "total_timed_submissions": len(timing_data),
            "urgent_submissions": len(urgent_submissions),
            "normal_submissions": len(normal_submissions),
            "urgent_success_rate": urgent_success_rate,
            "normal_success_rate": normal_success_rate,
            "timing_recommendation": (
                "avoid_urgent_submissions"
                if urgent_success_rate < normal_success_rate
                else "timing_not_critical"
            ),
        }

    def _analyze_category_performance(self) -> Dict[str, Any]:
        """Analyze performance by question category."""
        category_stats = {}

        for record in self.submissions.values():
            if record.dry_run:
                continue

            # Extract category from metadata
            category = "unknown"
            if "tournament_metadata" in record.metadata:
                category = record.metadata["tournament_metadata"].get(
                    "question_category", "unknown"
                )

            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "successful": 0,
                    "validation_errors": 0,
                }

            category_stats[category]["total"] += 1
            if record.status == SubmissionStatus.SUBMITTED:
                category_stats[category]["successful"] += 1
            category_stats[category]["validation_errors"] += len(
                record.validation_errors
            )

        # Calculate success rates
        for category, stats in category_stats.items():
            stats["success_rate"] = (
                stats["successful"] / stats["total"] if stats["total"] > 0 else 0
            )
            stats["avg_validation_errors"] = (
                stats["validation_errors"] / stats["total"] if stats["total"] > 0 else 0
            )

        return category_stats

    def _analyze_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Analyze recent performance trends."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        recent_submissions = [
            record
            for record in self.submissions.values()
            if record.timestamp >= cutoff_date and not record.dry_run
        ]

        if not recent_submissions:
            return {"error": f"No submissions in last {days} days"}

        successful_recent = sum(
            1
            for record in recent_submissions
            if record.status == SubmissionStatus.SUBMITTED
        )

        return {
            "period_days": days,
            "recent_submissions": len(recent_submissions),
            "recent_success_rate": successful_recent / len(recent_submissions),
            "recent_validation_errors": sum(
                len(r.validation_errors) for r in recent_submissions
            ),
            "trend": self._calculate_performance_trend(recent_submissions),
        }

    def _calculate_performance_trend(
        self, recent_submissions: List[SubmissionRecord]
    ) -> str:
        """Calculate performance trend from recent submissions."""
        if len(recent_submissions) < 4:
            return "insufficient_data"

        # Split into two halves and compare success rates
        mid_point = len(recent_submissions) // 2
        first_half = recent_submissions[:mid_point]
        second_half = recent_submissions[mid_point:]

        first_success_rate = sum(
            1 for r in first_half if r.status == SubmissionStatus.SUBMITTED
        ) / len(first_half)
        second_success_rate = sum(
            1 for r in second_half if r.status == SubmissionStatus.SUBMITTED
        ) / len(second_half)

        if second_success_rate > first_success_rate + 0.1:
            return "improving"
        elif second_success_rate < first_success_rate - 0.1:
            return "declining"
        else:
            return "stable"

    def export_audit_trail(
        self,
        format: str = "json",
        include_dry_runs: bool = True,
        include_metadata: bool = True,
    ) -> str:
        """
        Export audit trail in specified format.

        Args:
            format: Export format ("json", "csv", "summary")
            include_dry_runs: Whether to include dry run submissions
            include_metadata: Whether to include detailed metadata

        Returns:
            Formatted audit trail data
        """
        submissions_to_export = [
            record
            for record in self.submissions.values()
            if include_dry_runs or not record.dry_run
        ]

        if format == "json":
            return self._export_json(submissions_to_export, include_metadata)
        elif format == "csv":
            return self._export_csv(submissions_to_export, include_metadata)
        elif format == "summary":
            return self._export_summary(submissions_to_export)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(
        self, submissions: List[SubmissionRecord], include_metadata: bool
    ) -> str:
        """Export submissions as JSON."""
        export_data = []

        for record in submissions:
            record_data = record.to_dict()
            if not include_metadata:
                record_data.pop("metadata", None)
            export_data.append(record_data)

        return json.dumps(export_data, indent=2)

    def _export_csv(
        self, submissions: List[SubmissionRecord], include_metadata: bool
    ) -> str:
        """Export submissions as CSV."""
        import csv
        import io

        output = io.StringIO()

        if not submissions:
            return ""

        # Define CSV columns
        columns = [
            "submission_id",
            "question_id",
            "prediction_value",
            "confidence",
            "timestamp",
            "status",
            "dry_run",
            "validation_error_count",
        ]

        if include_metadata:
            columns.extend(["agent_type", "reasoning_method", "tournament_priority"])

        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()

        for record in submissions:
            row = {
                "submission_id": record.submission_id,
                "question_id": record.question_id,
                "prediction_value": record.prediction_value,
                "confidence": record.confidence,
                "timestamp": record.timestamp.isoformat(),
                "status": record.status.value,
                "dry_run": record.dry_run,
                "validation_error_count": len(record.validation_errors),
            }

            if include_metadata:
                tournament_metadata = record.metadata.get("tournament_metadata", {})
                row.update(
                    {
                        "agent_type": tournament_metadata.get("agent_type", ""),
                        "reasoning_method": tournament_metadata.get(
                            "reasoning_method", ""
                        ),
                        "tournament_priority": tournament_metadata.get(
                            "tournament_priority", ""
                        ),
                    }
                )

            writer.writerow(row)

        return output.getvalue()

    def _export_summary(self, submissions: List[SubmissionRecord]) -> str:
        """Export audit trail summary."""
        if not submissions:
            return "No submissions to summarize."

        total = len(submissions)
        successful = sum(
            1 for r in submissions if r.status == SubmissionStatus.SUBMITTED
        )
        dry_runs = sum(1 for r in submissions if r.dry_run)

        summary_lines = [
            "=== Audit Trail Summary ===",
            f"Total Submissions: {total}",
            f"Successful Submissions: {successful}",
            f"Dry Run Submissions: {dry_runs}",
            (
                f"Success Rate: {successful / (total - dry_runs) * 100:.1f}%"
                if total > dry_runs
                else "Success Rate: N/A"
            ),
            "",
            "Status Distribution:",
        ]

        # Status distribution
        status_counts = {}
        for record in submissions:
            status_counts[record.status.value] = (
                status_counts.get(record.status.value, 0) + 1
            )

        for status, count in status_counts.items():
            summary_lines.append(f"  {status}: {count}")

        # Recent activity
        recent_submissions = [
            r
            for r in submissions
            if r.timestamp >= datetime.now(timezone.utc) - timedelta(days=7)
        ]
        summary_lines.extend(
            ["", f"Recent Activity (7 days): {len(recent_submissions)} submissions"]
        )

        return "\n".join(summary_lines)


class DryRunManager:
    """
    Manages dry-run mode with tournament condition simulation.

    This class provides comprehensive dry-run capabilities that simulate
    tournament conditions without making actual submissions to Metaculus.
    """

    def __init__(
        self,
        validator: SubmissionValidator,
        audit_manager: AuditTrailManager,
        tournament_client: Optional[Any] = None,
    ):
        self.validator = validator
        self.audit_manager = audit_manager
        self.tournament_client = tournament_client
        self.dry_run_sessions: Dict[str, Dict[str, Any]] = {}

    def start_dry_run_session(
        self, session_name: str, tournament_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new dry-run session.

        Args:
            session_name: Name for the dry-run session
            tournament_context: Tournament context for simulation

        Returns:
            Session ID for tracking
        """
        session_id = str(uuid.uuid4())

        self.dry_run_sessions[session_id] = {
            "session_name": session_name,
            "session_id": session_id,
            "start_time": datetime.now(timezone.utc),
            "tournament_context": tournament_context,
            "submissions": [],
            "simulation_results": {},
            "performance_metrics": {},
            "status": "active",
        }

        logger.info(
            "Dry-run session started", session_id=session_id, session_name=session_name
        )

        return session_id

    def simulate_submission(
        self,
        session_id: str,
        question: Question,
        prediction: Union[Prediction, Dict[str, Any]],
        agent_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate a prediction submission in dry-run mode.

        Args:
            session_id: Dry-run session ID
            question: Question being predicted on
            prediction: Prediction to simulate
            agent_metadata: Additional agent metadata

        Returns:
            Comprehensive simulation results
        """
        if session_id not in self.dry_run_sessions:
            raise ValueError(f"Unknown dry-run session: {session_id}")

        session = self.dry_run_sessions[session_id]

        logger.info(
            "Simulating submission", session_id=session_id, question_id=question.id
        )

        # Create submission record in dry-run mode
        if isinstance(prediction, Prediction):
            pred_data = self.validator._extract_prediction_data(prediction)
        else:
            pred_data = prediction.copy()

        # Add agent metadata
        if agent_metadata:
            pred_data.update(agent_metadata)

        submission_record = self.audit_manager.create_submission_record(
            question_id=question.id,
            prediction_value=pred_data.get("prediction_value"),
            reasoning=pred_data.get("reasoning", ""),
            confidence=pred_data.get("confidence"),
            dry_run=True,
            metadata={
                "session_id": session_id,
                "agent_metadata": agent_metadata or {},
                "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Run comprehensive simulation
        simulation_results = self.validator.simulate_tournament_conditions(
            question, prediction, session["tournament_context"]
        )

        # Add submission-specific simulation data
        simulation_results.update(
            {
                "submission_id": submission_record.submission_id,
                "session_id": session_id,
                "api_simulation": self._simulate_api_interaction(question, pred_data),
                "competitive_analysis": self._simulate_competitive_impact(
                    question, pred_data, session["tournament_context"]
                ),
                "learning_opportunities": self._identify_learning_opportunities(
                    question, prediction, simulation_results
                ),
            }
        )

        # Update session
        session["submissions"].append(submission_record.submission_id)
        session["simulation_results"][
            submission_record.submission_id
        ] = simulation_results

        # Update submission record with simulation results
        self.audit_manager.update_submission_status(
            submission_record.submission_id,
            SubmissionStatus.DRY_RUN,
            metadata={"simulation_results": simulation_results},
        )

        logger.info(
            "Submission simulation completed",
            session_id=session_id,
            submission_id=submission_record.submission_id,
            validation_result=simulation_results["validation_results"]["result"],
        )

        return simulation_results

    def _simulate_api_interaction(
        self, question: Question, prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate API interaction without making actual requests."""
        # Format prediction as it would be sent to API
        formatted_prediction = self.validator.format_prediction_for_submission(
            question, prediction_data
        )

        # Simulate API response based on validation
        validation_result, validation_errors = self.validator.validate_prediction(
            question, prediction_data
        )

        if validation_result == ValidationResult.INVALID:
            # Simulate API rejection
            api_response = {
                "status_code": 400,
                "success": False,
                "message": "Validation failed",
                "errors": [error.message for error in validation_errors],
                "prediction_id": None,
            }
        else:
            # Simulate successful API response
            api_response = {
                "status_code": 200,
                "success": True,
                "message": "Prediction submitted successfully",
                "prediction_id": f"sim_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "formatted_prediction": formatted_prediction,
            "simulated_api_response": api_response,
            "would_succeed": validation_result != ValidationResult.INVALID,
            "estimated_response_time": self._estimate_api_response_time(question),
        }

    def _estimate_api_response_time(self, question: Question) -> float:
        """Estimate API response time based on question characteristics."""
        # Base response time
        base_time = 0.5  # 500ms

        # Add complexity factors
        if question.question_type == QuestionType.CONTINUOUS:
            base_time += 0.1
        elif question.question_type == QuestionType.MULTIPLE_CHOICE:
            base_time += 0.05

        # Add load factors (simulated)
        import random

        load_factor = random.uniform(0.8, 1.5)

        return base_time * load_factor

    def _simulate_competitive_impact(
        self,
        question: Question,
        prediction_data: Dict[str, Any],
        tournament_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Simulate competitive impact of the prediction."""
        if not tournament_context:
            return {"impact": "unknown", "reason": "no_tournament_context"}

        # Simulate ranking change
        current_ranking = tournament_context.get("current_ranking", 50)
        total_participants = tournament_context.get("participant_count", 100)

        # Estimate impact based on question priority and prediction quality
        priority = question.metadata.get("tournament_priority", "medium")
        confidence = prediction_data.get("confidence", 0.5)

        # Simulate potential ranking change
        if priority == "critical" and confidence > 0.8:
            potential_change = random.randint(3, 8)
        elif priority == "high" and confidence > 0.7:
            potential_change = random.randint(1, 5)
        elif confidence > 0.6:
            potential_change = random.randint(0, 3)
        else:
            potential_change = random.randint(-2, 2)

        # Adjust based on current position
        if current_ranking > total_participants * 0.8:
            potential_change = max(
                potential_change, 1
            )  # Always positive for low-ranked
        elif current_ranking < total_participants * 0.2:
            potential_change = min(
                potential_change, 2
            )  # Limited upside for high-ranked

        new_ranking = max(
            1, min(total_participants, current_ranking - potential_change)
        )

        return {
            "current_ranking": current_ranking,
            "estimated_new_ranking": new_ranking,
            "potential_change": potential_change,
            "percentile_change": (current_ranking - new_ranking)
            / total_participants
            * 100,
            "impact_confidence": min(
                confidence * 0.8, 0.9
            ),  # Slightly lower than prediction confidence
            "factors": {
                "question_priority": priority,
                "prediction_confidence": confidence,
                "current_position": (
                    "low" if current_ranking > total_participants * 0.7 else "high"
                ),
            },
        }

    def _identify_learning_opportunities(
        self,
        question: Question,
        prediction: Union[Prediction, Dict[str, Any]],
        simulation_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Identify learning opportunities from the simulation."""
        opportunities = []

        # Validation learning opportunities
        validation_errors = simulation_results["validation_results"]["errors"]
        if validation_errors:
            opportunities.append(
                {
                    "type": "validation_improvement",
                    "priority": "high",
                    "description": "Address validation errors to improve submission success rate",
                    "specific_errors": [error["code"] for error in validation_errors],
                    "learning_action": "review_validation_requirements",
                }
            )

        # Risk management opportunities
        risk_assessment = simulation_results["risk_assessment"]
        if risk_assessment["risk_level"] == "high":
            opportunities.append(
                {
                    "type": "risk_management",
                    "priority": "medium",
                    "description": "High-risk prediction detected - learn risk mitigation strategies",
                    "identified_risks": risk_assessment["identified_risks"],
                    "learning_action": "study_risk_mitigation",
                }
            )

        # Strategic opportunities
        strategic_considerations = simulation_results["tournament_simulation"][
            "strategic_considerations"
        ]
        if "extreme_consensus" in strategic_considerations:
            opportunities.append(
                {
                    "type": "contrarian_strategy",
                    "priority": "medium",
                    "description": "Learn to identify and evaluate contrarian opportunities",
                    "learning_action": "study_market_inefficiencies",
                }
            )

        # Timing opportunities
        timing_analysis = simulation_results["timing_analysis"]
        if timing_analysis["status"] in ["critical", "urgent"]:
            opportunities.append(
                {
                    "type": "time_management",
                    "priority": "high",
                    "description": "Improve submission timing to avoid deadline pressure",
                    "learning_action": "optimize_workflow_timing",
                }
            )

        return opportunities

    def end_dry_run_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a dry-run session and generate comprehensive report.

        Args:
            session_id: Session to end

        Returns:
            Session summary and analysis
        """
        if session_id not in self.dry_run_sessions:
            raise ValueError(f"Unknown dry-run session: {session_id}")

        session = self.dry_run_sessions[session_id]
        session["status"] = "completed"
        session["end_time"] = datetime.now(timezone.utc)
        session["duration"] = (
            session["end_time"] - session["start_time"]
        ).total_seconds()

        # Generate session report
        report = self._generate_session_report(session)

        logger.info(
            "Dry-run session ended",
            session_id=session_id,
            duration=session["duration"],
            submissions=len(session["submissions"]),
        )

        return report

    def _generate_session_report(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive session report."""
        submissions = session["submissions"]
        simulation_results = session["simulation_results"]

        # Overall statistics
        total_submissions = len(submissions)
        successful_validations = sum(
            1
            for result in simulation_results.values()
            if result["validation_results"]["result"] != "invalid"
        )

        # Risk analysis
        high_risk_submissions = sum(
            1
            for result in simulation_results.values()
            if result["risk_assessment"]["risk_level"] == "high"
        )

        # Learning opportunities aggregation
        all_opportunities = []
        for result in simulation_results.values():
            all_opportunities.extend(result["learning_opportunities"])

        opportunity_types = {}
        for opp in all_opportunities:
            opp_type = opp["type"]
            opportunity_types[opp_type] = opportunity_types.get(opp_type, 0) + 1

        # Competitive impact analysis
        competitive_impacts = [
            result["competitive_analysis"]
            for result in simulation_results.values()
            if "potential_change" in result["competitive_analysis"]
        ]

        avg_ranking_change = 0
        if competitive_impacts:
            avg_ranking_change = sum(
                impact["potential_change"] for impact in competitive_impacts
            ) / len(competitive_impacts)

        return {
            "session_summary": {
                "session_id": session["session_id"],
                "session_name": session["session_name"],
                "duration_seconds": session["duration"],
                "total_submissions": total_submissions,
                "successful_validations": successful_validations,
                "validation_success_rate": (
                    successful_validations / total_submissions
                    if total_submissions > 0
                    else 0
                ),
            },
            "risk_analysis": {
                "high_risk_submissions": high_risk_submissions,
                "risk_rate": (
                    high_risk_submissions / total_submissions
                    if total_submissions > 0
                    else 0
                ),
                "risk_distribution": self._analyze_risk_distribution(
                    simulation_results
                ),
            },
            "learning_analysis": {
                "total_opportunities": len(all_opportunities),
                "opportunity_types": opportunity_types,
                "priority_distribution": self._analyze_priority_distribution(
                    all_opportunities
                ),
                "top_learning_areas": self._identify_top_learning_areas(
                    opportunity_types
                ),
            },
            "competitive_analysis": {
                "analyzed_submissions": len(competitive_impacts),
                "average_ranking_change": avg_ranking_change,
                "potential_ranking_improvement": max(
                    [impact["potential_change"] for impact in competitive_impacts],
                    default=0,
                ),
                "competitive_readiness": self._assess_competitive_readiness(
                    simulation_results
                ),
            },
            "recommendations": self._generate_session_recommendations(
                session, simulation_results
            ),
            "detailed_results": (
                simulation_results
                if len(simulation_results) <= 10
                else "truncated_for_brevity"
            ),
        }

    def _analyze_risk_distribution(
        self, simulation_results: Dict[str, Any]
    ) -> Dict[str, int]:
        """Analyze distribution of risk levels."""
        risk_distribution = {"low": 0, "medium": 0, "high": 0}

        for result in simulation_results.values():
            risk_level = result["risk_assessment"]["risk_level"]
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

        return risk_distribution

    def _analyze_priority_distribution(
        self, opportunities: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze distribution of learning opportunity priorities."""
        priority_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for opp in opportunities:
            priority = opp.get("priority", "medium")
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1

        return priority_distribution

    def _identify_top_learning_areas(
        self, opportunity_types: Dict[str, int]
    ) -> List[str]:
        """Identify top learning areas by frequency."""
        sorted_types = sorted(
            opportunity_types.items(), key=lambda x: x[1], reverse=True
        )
        return [opp_type for opp_type, count in sorted_types[:3]]

    def _assess_competitive_readiness(self, simulation_results: Dict[str, Any]) -> str:
        """Assess overall competitive readiness based on simulation results."""
        total_results = len(simulation_results)
        if total_results == 0:
            return "unknown"

        # Count successful validations
        successful_validations = sum(
            1
            for result in simulation_results.values()
            if result["validation_results"]["result"] != "invalid"
        )

        # Count low-risk submissions
        low_risk_submissions = sum(
            1
            for result in simulation_results.values()
            if result["risk_assessment"]["risk_level"] == "low"
        )

        validation_rate = successful_validations / total_results
        low_risk_rate = low_risk_submissions / total_results

        if validation_rate >= 0.9 and low_risk_rate >= 0.7:
            return "high"
        elif validation_rate >= 0.8 and low_risk_rate >= 0.5:
            return "medium"
        else:
            return "low"

    def _generate_session_recommendations(
        self, session: Dict[str, Any], simulation_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on session results."""
        recommendations = []

        # Validation recommendations
        validation_success_rate = (
            sum(
                1
                for result in simulation_results.values()
                if result["validation_results"]["result"] != "invalid"
            )
            / len(simulation_results)
            if simulation_results
            else 0
        )

        if validation_success_rate < 0.8:
            recommendations.append(
                {
                    "type": "validation_improvement",
                    "priority": "high",
                    "message": f"Validation success rate is {validation_success_rate:.1%}. Focus on improving prediction formatting and validation.",
                    "action": "review_validation_requirements",
                }
            )

        # Risk management recommendations
        high_risk_rate = (
            sum(
                1
                for result in simulation_results.values()
                if result["risk_assessment"]["risk_level"] == "high"
            )
            / len(simulation_results)
            if simulation_results
            else 0
        )

        if high_risk_rate > 0.3:
            recommendations.append(
                {
                    "type": "risk_management",
                    "priority": "medium",
                    "message": f"{high_risk_rate:.1%} of predictions are high-risk. Consider more conservative strategies.",
                    "action": "implement_risk_controls",
                }
            )

        # Learning recommendations
        all_opportunities = []
        for result in simulation_results.values():
            all_opportunities.extend(result["learning_opportunities"])

        if len(all_opportunities) > len(simulation_results) * 2:
            recommendations.append(
                {
                    "type": "learning_focus",
                    "priority": "medium",
                    "message": "Multiple learning opportunities identified. Prioritize systematic improvement.",
                    "action": "create_learning_plan",
                }
            )

        return recommendations

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a dry-run session."""
        if session_id not in self.dry_run_sessions:
            return {"error": "Session not found"}

        session = self.dry_run_sessions[session_id]

        return {
            "session_id": session_id,
            "session_name": session["session_name"],
            "status": session["status"],
            "start_time": session["start_time"].isoformat(),
            "submissions_count": len(session["submissions"]),
            "duration": (
                datetime.now(timezone.utc) - session["start_time"]
            ).total_seconds(),
        }

    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active dry-run sessions."""
        active_sessions = []

        for session_id, session in self.dry_run_sessions.items():
            if session["status"] == "active":
                active_sessions.append(
                    {
                        "session_id": session_id,
                        "session_name": session["session_name"],
                        "start_time": session["start_time"].isoformat(),
                        "submissions_count": len(session["submissions"]),
                    }
                )

        return active_sessions
