"""
Tests for submission validation and audit trail system.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from src.domain.entities.prediction import Prediction, PredictionResult
from src.domain.entities.question import Question, QuestionType
from src.domain.value_objects.confidence import ConfidenceLevel
from src.domain.value_objects.probability import Probability
from src.infrastructure.external_apis.submission_validator import (
    AuditTrailManager,
    SubmissionRecord,
    SubmissionStatus,
    SubmissionValidator,
    ValidationError,
    ValidationResult,
)


@pytest.fixture
def validator():
    """Create submission validator."""
    return SubmissionValidator()


@pytest.fixture
def audit_manager():
    """Create audit trail manager with temporary storage."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        temp_path = f.name

    manager = AuditTrailManager(storage_path=temp_path)
    yield manager

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def binary_question():
    """Create a binary question for testing."""
    return Question.create(
        title="Will AI achieve AGI by 2030?",
        description="This question asks about artificial general intelligence development.",
        question_type=QuestionType.BINARY,
        resolution_criteria="AGI is defined as...",
        close_time=datetime.now(timezone.utc) + timedelta(hours=24),
        resolve_time=datetime.now(timezone.utc) + timedelta(days=30),
        created_at=datetime.now(timezone.utc) - timedelta(days=7),
    )


@pytest.fixture
def continuous_question():
    """Create a continuous question for testing."""
    return Question.create(
        title="What will be the global temperature anomaly in 2030?",
        description="Temperature anomaly relative to 1951-1980 baseline.",
        question_type=QuestionType.CONTINUOUS,
        resolution_criteria="Based on NASA GISS data.",
        close_time=datetime.now(timezone.utc) + timedelta(hours=24),
        resolve_time=datetime.now(timezone.utc) + timedelta(days=30),
        created_at=datetime.now(timezone.utc) - timedelta(days=7),
        min_value=-2.0,
        max_value=5.0,
    )


@pytest.fixture
def multiple_choice_question():
    """Create a multiple choice question for testing."""
    return Question.create(
        title="Which party will win the 2024 election?",
        description="US Presidential election winner.",
        question_type=QuestionType.MULTIPLE_CHOICE,
        resolution_criteria="Based on electoral college results.",
        close_time=datetime.now(timezone.utc) + timedelta(hours=24),
        resolve_time=datetime.now(timezone.utc) + timedelta(days=30),
        created_at=datetime.now(timezone.utc) - timedelta(days=7),
        choices=["Democratic", "Republican", "Other"],
    )


@pytest.fixture
def valid_binary_prediction():
    """Create a valid binary prediction."""
    return {
        "question_id": "test_question",
        "prediction_value": 0.75,
        "reasoning": "Based on current AI development trends and expert opinions, there is a significant probability of AGI by 2030.",
        "confidence": 0.8,
    }


class TestSubmissionValidator:
    """Test submission validation functionality."""

    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator.validation_rules is not None
        assert "binary" in validator.validation_rules
        assert "continuous" in validator.validation_rules
        assert "multiple_choice" in validator.validation_rules
        assert "general" in validator.validation_rules
        assert validator.tournament_mode is False

    def test_tournament_mode_initialization(self):
        """Test validator initialization in tournament mode."""
        tournament_validator = SubmissionValidator(tournament_mode=True)
        assert tournament_validator.tournament_mode is True

    def test_valid_binary_prediction(
        self, validator, binary_question, valid_binary_prediction
    ):
        """Test validation of valid binary prediction."""
        result, errors = validator.validate_prediction(
            binary_question, valid_binary_prediction
        )

        assert result == ValidationResult.VALID
        assert len(errors) == 0

    def test_invalid_binary_prediction_out_of_range(self, validator, binary_question):
        """Test validation of binary prediction out of range."""
        invalid_prediction = {
            "question_id": "test_question",
            "prediction_value": 1.5,  # Invalid: > 1.0
            "reasoning": "This prediction is out of range.",
        }

        result, errors = validator.validate_prediction(
            binary_question, invalid_prediction
        )

        assert result == ValidationResult.INVALID
        assert len(errors) > 0
        assert any(error.code == "VALUE_OUT_OF_RANGE" for error in errors)

    def test_extreme_binary_prediction_warning(self, validator, binary_question):
        """Test warning for extreme binary prediction values."""
        extreme_prediction = {
            "question_id": "test_question",
            "prediction_value": 0.005,  # Extreme value
            "reasoning": "This is an extreme prediction.",
        }

        result, errors = validator.validate_prediction(
            binary_question, extreme_prediction
        )

        assert result == ValidationResult.WARNING
        assert any(error.code == "EXTREME_PREDICTION_VALUE" for error in errors)

    def test_missing_required_fields(self, validator, binary_question):
        """Test validation with missing required fields."""
        incomplete_prediction = {"reasoning": "Missing prediction value."}

        result, errors = validator.validate_prediction(
            binary_question, incomplete_prediction
        )

        assert result == ValidationResult.INVALID
        assert any(error.code == "MISSING_REQUIRED_FIELD" for error in errors)

    def test_continuous_prediction_validation(self, validator, continuous_question):
        """Test validation of continuous predictions."""
        valid_prediction = {
            "question_id": "test_question",
            "prediction_value": 1.5,
            "reasoning": "Based on climate models and current trends.",
        }

        result, errors = validator.validate_prediction(
            continuous_question, valid_prediction
        )

        assert result == ValidationResult.VALID
        assert len(errors) == 0

    def test_continuous_prediction_out_of_bounds(self, validator, continuous_question):
        """Test continuous prediction outside question bounds."""
        out_of_bounds_prediction = {
            "question_id": "test_question",
            "prediction_value": 10.0,  # Above max_value of 5.0
            "reasoning": "This prediction exceeds the maximum bound.",
        }

        result, errors = validator.validate_prediction(
            continuous_question, out_of_bounds_prediction
        )

        assert result == ValidationResult.INVALID
        assert any(error.code == "VALUE_ABOVE_MAXIMUM" for error in errors)

    def test_multiple_choice_prediction_validation(
        self, validator, multiple_choice_question
    ):
        """Test validation of multiple choice predictions."""
        valid_prediction = {
            "question_id": "test_question",
            "prediction_value": 1,  # Republican
            "reasoning": "Based on polling data and historical trends.",
        }

        result, errors = validator.validate_prediction(
            multiple_choice_question, valid_prediction
        )

        assert result == ValidationResult.VALID
        assert len(errors) == 0

    def test_multiple_choice_invalid_index(self, validator, multiple_choice_question):
        """Test multiple choice prediction with invalid choice index."""
        invalid_prediction = {
            "question_id": "test_question",
            "prediction_value": 5,  # Invalid: only 3 choices (0-2)
            "reasoning": "Invalid choice index.",
        }

        result, errors = validator.validate_prediction(
            multiple_choice_question, invalid_prediction
        )

        assert result == ValidationResult.INVALID
        assert any(error.code == "CHOICE_INDEX_OUT_OF_RANGE" for error in errors)

    def test_reasoning_validation(self, validator, binary_question):
        """Test reasoning validation."""
        # Too short reasoning
        short_reasoning_prediction = {
            "question_id": "test_question",
            "prediction_value": 0.7,
            "reasoning": "Short",  # Too short
        }

        result, errors = validator.validate_prediction(
            binary_question, short_reasoning_prediction
        )

        assert result == ValidationResult.WARNING
        assert any(error.code == "REASONING_TOO_SHORT" for error in errors)

        # Too long reasoning
        long_reasoning_prediction = {
            "question_id": "test_question",
            "prediction_value": 0.7,
            "reasoning": "x" * 15000,  # Too long
        }

        result, errors = validator.validate_prediction(
            binary_question, long_reasoning_prediction
        )

        assert result == ValidationResult.INVALID
        assert any(error.code == "REASONING_TOO_LONG" for error in errors)

    def test_timing_validation_closed_question(self, validator):
        """Test validation of predictions on closed questions."""
        closed_question = Question.create(
            title="Closed question",
            description="This question is already closed.",
            question_type=QuestionType.BINARY,
            close_time=datetime.now(timezone.utc)
            - timedelta(hours=1),  # Closed 1 hour ago
            created_at=datetime.now(timezone.utc) - timedelta(days=7),
        )

        prediction = {
            "question_id": "test_question",
            "prediction_value": 0.7,
            "reasoning": "This question is closed.",
        }

        result, errors = validator.validate_prediction(closed_question, prediction)

        assert result == ValidationResult.INVALID
        assert any(error.code == "QUESTION_CLOSED" for error in errors)

    def test_timing_validation_closing_soon(self, validator):
        """Test warning for questions closing soon."""
        closing_soon_question = Question.create(
            title="Closing soon question",
            description="This question closes soon.",
            question_type=QuestionType.BINARY,
            close_time=datetime.now(timezone.utc)
            + timedelta(minutes=30),  # Closes in 30 minutes
            created_at=datetime.now(timezone.utc) - timedelta(days=7),
        )

        prediction = {
            "question_id": "test_question",
            "prediction_value": 0.7,
            "reasoning": "This question closes soon.",
        }

        result, errors = validator.validate_prediction(
            closing_soon_question, prediction
        )

        assert result == ValidationResult.WARNING
        assert any(error.code == "QUESTION_CLOSING_SOON" for error in errors)

    def test_prediction_object_validation(self, validator, binary_question):
        """Test validation with Prediction object instead of dict."""
        prediction_result = PredictionResult(
            binary_probability=0.75, numeric_value=None, choice_index=None
        )

        from src.domain.entities.prediction import PredictionConfidence, PredictionMethod
        from uuid import uuid4
        from datetime import datetime

        prediction = Prediction(
            id=uuid4(),
            question_id=binary_question.id,
            research_report_id=uuid4(),
            result=prediction_result,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Based on analysis of current trends.",
            reasoning_steps=["Step 1: Analysis", "Step 2: Conclusion"],
            created_at=datetime.now(),
            created_by="test-agent",
        )

        result, errors = validator.validate_prediction(binary_question, prediction)

        assert result == ValidationResult.VALID
        assert len(errors) == 0

    def test_format_binary_prediction_for_submission(
        self, validator, binary_question, valid_binary_prediction
    ):
        """Test formatting binary prediction for API submission."""
        formatted = validator.format_prediction_for_submission(
            binary_question, valid_binary_prediction
        )

        assert "prediction" in formatted
        assert "comment" in formatted
        assert "void" in formatted
        assert formatted["prediction"] == 0.75
        assert formatted["void"] is False
        assert isinstance(formatted["prediction"], float)

    def test_format_continuous_prediction_for_submission(
        self, validator, continuous_question
    ):
        """Test formatting continuous prediction for API submission."""
        prediction = {
            "question_id": "test_question",
            "prediction_value": 1.5,
            "reasoning": "Based on climate models.",
        }

        formatted = validator.format_prediction_for_submission(
            continuous_question, prediction
        )

        assert "prediction" in formatted
        assert "comment" in formatted
        assert formatted["prediction"] == 1.5
        assert isinstance(formatted["prediction"], float)

    def test_format_multiple_choice_prediction_for_submission(
        self, validator, multiple_choice_question
    ):
        """Test formatting multiple choice prediction for API submission."""
        prediction = {
            "question_id": "test_question",
            "prediction_value": 1,
            "reasoning": "Based on polling data.",
        }

        formatted = validator.format_prediction_for_submission(
            multiple_choice_question, prediction
        )

        assert "prediction" in formatted
        assert "comment" in formatted
        assert formatted["prediction"] == 1
        assert isinstance(formatted["prediction"], int)

    def test_tournament_formatting(self, binary_question, valid_binary_prediction):
        """Test tournament-specific formatting."""
        tournament_validator = SubmissionValidator(tournament_mode=True)

        # Add tournament metadata to question
        binary_question.metadata.update(
            {
                "category": "technology",
                "tournament_priority": "high",
                "urgency_score": 0.8,
            }
        )

        # Add agent metadata to prediction
        prediction_with_metadata = valid_binary_prediction.copy()
        prediction_with_metadata.update(
            {"agent_type": "ensemble", "reasoning_method": "chain_of_thought"}
        )

        formatted = tournament_validator.format_prediction_for_submission(
            binary_question, prediction_with_metadata
        )

        assert "tournament_metadata" in formatted
        assert "validation_checksum" in formatted
        assert "submission_context" in formatted

        tournament_metadata = formatted["tournament_metadata"]
        assert tournament_metadata["question_category"] == "technology"
        assert tournament_metadata["tournament_priority"] == "high"
        assert tournament_metadata["agent_type"] == "ensemble"

    def test_validation_checksum(self, validator):
        """Test validation checksum calculation and verification."""
        formatted_prediction = {
            "prediction": 0.75,
            "comment": "Test reasoning",
            "confidence": 0.8,
        }

        checksum = validator._calculate_validation_checksum(formatted_prediction)
        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5 hash length

        # Test integrity validation
        formatted_prediction["validation_checksum"] = checksum
        assert validator.validate_submission_integrity(formatted_prediction)

        # Test with tampered data
        formatted_prediction["prediction"] = 0.8  # Changed value
        assert not validator.validate_submission_integrity(
            formatted_prediction, checksum
        )

    def test_tournament_condition_simulation(
        self, validator, binary_question, valid_binary_prediction
    ):
        """Test tournament condition simulation."""
        tournament_context = {
            "tournament_id": "test_tournament",
            "current_ranking": 25,
            "participant_count": 100,
            "completion_rate": 0.6,
        }

        # Add tournament metadata to question
        binary_question.metadata.update(
            {
                "category": "technology",
                "tournament_priority": "high",
                "community_prediction": 0.3,
                "prediction_count": 50,
            }
        )

        simulation_results = validator.simulate_tournament_conditions(
            binary_question, valid_binary_prediction, tournament_context
        )

        assert "simulation_timestamp" in simulation_results
        assert "question_analysis" in simulation_results
        assert "prediction_analysis" in simulation_results
        assert "tournament_simulation" in simulation_results
        assert "validation_results" in simulation_results
        assert "timing_analysis" in simulation_results
        assert "risk_assessment" in simulation_results
        assert "recommendations" in simulation_results

        # Check question analysis
        question_analysis = simulation_results["question_analysis"]
        assert question_analysis["question_type"] == "binary"
        assert question_analysis["category"] == "technology"
        assert question_analysis["tournament_priority"] == "high"

        # Check tournament simulation
        tournament_sim = simulation_results["tournament_simulation"]
        assert tournament_sim["tournament_active"] is True
        assert "market_efficiency" in tournament_sim
        assert "scoring_impact" in tournament_sim

    def test_contrarian_prediction_detection(self, validator, binary_question):
        """Test contrarian prediction detection."""
        # Set community prediction
        binary_question.metadata["community_prediction"] = 0.2

        # Test contrarian prediction (far from community)
        assert validator._is_contrarian_prediction(binary_question, 0.8) is True

        # Test non-contrarian prediction (close to community)
        assert validator._is_contrarian_prediction(binary_question, 0.25) is False

        # Test with no community prediction
        binary_question.metadata.pop("community_prediction", None)
        assert validator._is_contrarian_prediction(binary_question, 0.8) is False

    def test_market_efficiency_estimation(self, validator, binary_question):
        """Test market efficiency estimation."""
        # Low participation
        binary_question.metadata.update(
            {"prediction_count": 5, "community_prediction": 0.5}
        )
        assert validator._estimate_market_efficiency(binary_question) == "low"

        # Medium participation
        binary_question.metadata["prediction_count"] = 25
        assert validator._estimate_market_efficiency(binary_question) == "medium"

        # High participation with extreme consensus
        binary_question.metadata.update(
            {"prediction_count": 100, "community_prediction": 0.05}
        )
        assert (
            validator._estimate_market_efficiency(binary_question)
            == "potentially_inefficient"
        )

        # High participation with balanced consensus
        binary_question.metadata["community_prediction"] = 0.5
        assert validator._estimate_market_efficiency(binary_question) == "high"

    def test_submission_timing_analysis(self, validator):
        """Test submission timing analysis."""
        now = datetime.now(timezone.utc)

        # Critical timing (closes in 30 minutes)
        critical_question = Question.create(
            title="Critical question",
            description="Closes very soon",
            question_type=QuestionType.BINARY,
            close_time=now + timedelta(minutes=30),
            created_at=now - timedelta(days=7),
        )

        timing_analysis = validator._analyze_submission_timing(critical_question)
        assert timing_analysis["status"] == "critical"
        assert timing_analysis["recommendation"] == "submit_immediately"

        # Normal timing (closes in 2 days)
        normal_question = Question.create(
            title="Normal question",
            description="Closes in 2 days",
            question_type=QuestionType.BINARY,
            close_time=now + timedelta(days=2),
            created_at=now - timedelta(days=7),
        )

        timing_analysis = validator._analyze_submission_timing(normal_question)
        assert timing_analysis["status"] == "normal"
        assert timing_analysis["recommendation"] == "monitor_and_optimize"

    def test_risk_assessment(self, validator, binary_question, valid_binary_prediction):
        """Test submission risk assessment."""
        # Low confidence prediction
        low_confidence_prediction = valid_binary_prediction.copy()
        low_confidence_prediction["confidence"] = 0.2

        risk_assessment = validator._assess_submission_risk(
            binary_question, low_confidence_prediction
        )
        assert risk_assessment["risk_level"] == "high"
        assert "low_confidence_prediction" in risk_assessment["identified_risks"]

        # Extreme prediction value
        extreme_prediction = valid_binary_prediction.copy()
        extreme_prediction["prediction_value"] = 0.02

        risk_assessment = validator._assess_submission_risk(
            binary_question, extreme_prediction
        )
        assert "extreme_prediction_value" in risk_assessment["identified_risks"]

        # Normal prediction
        risk_assessment = validator._assess_submission_risk(
            binary_question, valid_binary_prediction
        )
        assert risk_assessment["risk_level"] in ["low", "medium"]


class TestAuditTrailManager:
    """Test audit trail management functionality."""

    def test_create_submission_record(self, audit_manager):
        """Test creating submission records."""
        record = audit_manager.create_submission_record(
            question_id="test_question",
            prediction_value=0.75,
            reasoning="Test reasoning",
            confidence=0.8,
            dry_run=False,
        )

        assert record.submission_id is not None
        assert record.question_id == "test_question"
        assert record.prediction_value == 0.75
        assert record.reasoning == "Test reasoning"
        assert record.confidence == 0.8
        assert record.status == SubmissionStatus.PENDING
        assert not record.dry_run

        # Check it's stored in manager
        assert record.submission_id in audit_manager.submissions

    def test_create_dry_run_record(self, audit_manager):
        """Test creating dry run submission records."""
        record = audit_manager.create_submission_record(
            question_id="test_question",
            prediction_value=0.75,
            reasoning="Test reasoning",
            dry_run=True,
        )

        assert record.status == SubmissionStatus.DRY_RUN
        assert record.dry_run is True

    def test_update_submission_status(self, audit_manager):
        """Test updating submission status."""
        record = audit_manager.create_submission_record(
            question_id="test_question",
            prediction_value=0.75,
            reasoning="Test reasoning",
        )

        validation_errors = [
            ValidationError(
                field="test_field",
                message="Test error",
                severity=ValidationResult.WARNING,
                code="TEST_ERROR",
            )
        ]

        audit_manager.update_submission_status(
            record.submission_id,
            SubmissionStatus.VALIDATED,
            validation_errors=validation_errors,
            metadata={"test_key": "test_value"},
        )

        updated_record = audit_manager.get_submission_record(record.submission_id)
        assert updated_record.status == SubmissionStatus.VALIDATED
        assert len(updated_record.validation_errors) == 1
        assert updated_record.metadata["test_key"] == "test_value"

    def test_get_submissions_by_question(self, audit_manager):
        """Test retrieving submissions by question ID."""
        # Create multiple submissions for same question
        record1 = audit_manager.create_submission_record(
            question_id="question_1", prediction_value=0.6, reasoning="First prediction"
        )

        record2 = audit_manager.create_submission_record(
            question_id="question_1",
            prediction_value=0.7,
            reasoning="Second prediction",
        )

        record3 = audit_manager.create_submission_record(
            question_id="question_2",
            prediction_value=0.8,
            reasoning="Different question",
        )

        question_1_submissions = audit_manager.get_submissions_by_question("question_1")

        assert len(question_1_submissions) == 2
        submission_ids = [r.submission_id for r in question_1_submissions]
        assert record1.submission_id in submission_ids
        assert record2.submission_id in submission_ids
        assert record3.submission_id not in submission_ids

    def test_get_recent_submissions(self, audit_manager):
        """Test retrieving recent submissions."""
        # Create multiple submissions
        records = []
        for i in range(5):
            record = audit_manager.create_submission_record(
                question_id=f"question_{i}",
                prediction_value=0.5 + i * 0.1,
                reasoning=f"Prediction {i}",
            )
            records.append(record)

        recent = audit_manager.get_recent_submissions(limit=3)

        assert len(recent) == 3
        # Should be ordered by timestamp (most recent first)
        assert recent[0].timestamp >= recent[1].timestamp >= recent[2].timestamp

    def test_persist_and_load_audit_trail(self, audit_manager):
        """Test persisting and loading audit trail."""
        # Create some records
        record1 = audit_manager.create_submission_record(
            question_id="question_1",
            prediction_value=0.75,
            reasoning="Test reasoning 1",
            confidence=0.8,
        )

        record2 = audit_manager.create_submission_record(
            question_id="question_2",
            prediction_value=0.65,
            reasoning="Test reasoning 2",
            dry_run=True,
        )

        # Add validation errors to one record
        validation_errors = [
            ValidationError(
                field="test_field",
                message="Test error",
                severity=ValidationResult.WARNING,
                code="TEST_ERROR",
            )
        ]
        audit_manager.update_submission_status(
            record1.submission_id,
            SubmissionStatus.VALIDATED,
            validation_errors=validation_errors,
        )

        # Persist to storage
        audit_manager.persist_to_storage()

        # Create new manager and load
        new_manager = AuditTrailManager(storage_path=audit_manager.storage_path)
        new_manager.load_from_storage()

        # Verify data was loaded correctly
        assert len(new_manager.submissions) == 2

        loaded_record1 = new_manager.get_submission_record(record1.submission_id)
        assert loaded_record1 is not None
        assert loaded_record1.question_id == "question_1"
        assert loaded_record1.prediction_value == 0.75
        assert loaded_record1.status == SubmissionStatus.VALIDATED
        assert len(loaded_record1.validation_errors) == 1
        assert loaded_record1.validation_errors[0].code == "TEST_ERROR"

        loaded_record2 = new_manager.get_submission_record(record2.submission_id)
        assert loaded_record2 is not None
        assert loaded_record2.dry_run is True

    def test_generate_audit_report(self, audit_manager):
        """Test generating audit report."""
        # Create submissions with different statuses
        record1 = audit_manager.create_submission_record(
            question_id="question_1",
            prediction_value=0.75,
            reasoning="Test reasoning 1",
        )

        record2 = audit_manager.create_submission_record(
            question_id="question_2",
            prediction_value=0.65,
            reasoning="Test reasoning 2",
            dry_run=True,
        )

        # Update statuses
        audit_manager.update_submission_status(
            record1.submission_id, SubmissionStatus.SUBMITTED
        )
        audit_manager.update_submission_status(
            record2.submission_id, SubmissionStatus.FAILED
        )

        report = audit_manager.generate_audit_report()

        assert "total_submissions" in report
        assert "status_distribution" in report
        assert "dry_run_submissions" in report
        assert "validation_error_distribution" in report
        assert "recent_submissions" in report

        assert report["total_submissions"] == 2
        assert report["dry_run_submissions"] == 1
        assert report["status_distribution"]["submitted"] == 1
        assert report["status_distribution"]["failed"] == 1
        assert len(report["recent_submissions"]) == 2


class TestSubmissionRecord:
    """Test SubmissionRecord functionality."""

    def test_to_dict_conversion(self):
        """Test converting submission record to dictionary."""
        validation_errors = [
            ValidationError(
                field="test_field",
                message="Test error",
                severity=ValidationResult.WARNING,
                code="TEST_ERROR",
            )
        ]

        record = SubmissionRecord(
            submission_id="test_id",
            question_id="question_1",
            prediction_value=0.75,
            confidence=0.8,
            reasoning="Test reasoning",
            timestamp=datetime.now(timezone.utc),
            status=SubmissionStatus.VALIDATED,
            validation_errors=validation_errors,
            metadata={"test_key": "test_value"},
            dry_run=False,
        )

        record_dict = record.to_dict()

        assert record_dict["submission_id"] == "test_id"
        assert record_dict["question_id"] == "question_1"
        assert record_dict["prediction_value"] == 0.75
        assert record_dict["confidence"] == 0.8
        assert record_dict["reasoning"] == "Test reasoning"
        assert record_dict["status"] == "validated"
        assert len(record_dict["validation_errors"]) == 1
        assert record_dict["validation_errors"][0]["code"] == "TEST_ERROR"
        assert record_dict["metadata"]["test_key"] == "test_value"
        assert record_dict["dry_run"] is False

    def test_submission_confirmation(self, audit_manager):
        """Test submission confirmation functionality."""
        record = audit_manager.create_submission_record(
            question_id="test_question",
            prediction_value=0.75,
            reasoning="Test reasoning",
        )

        # Test successful confirmation
        api_response = {
            "status_code": 200,
            "message": "Success",
            "prediction_id": "pred_12345",
        }

        audit_manager.confirm_submission(
            record.submission_id, api_response, success=True
        )

        updated_record = audit_manager.get_submission_record(record.submission_id)
        assert updated_record.status == SubmissionStatus.SUBMITTED
        assert updated_record.metadata["api_response"] == api_response
        assert updated_record.metadata["success"] is True
        assert updated_record.metadata["metaculus_prediction_id"] == "pred_12345"

        # Test failed confirmation
        failed_record = audit_manager.create_submission_record(
            question_id="test_question_2",
            prediction_value=0.6,
            reasoning="Test reasoning 2",
        )

        failed_response = {"status_code": 400, "message": "Validation error"}

        audit_manager.confirm_submission(
            failed_record.submission_id, failed_response, success=False
        )

        updated_failed_record = audit_manager.get_submission_record(
            failed_record.submission_id
        )
        assert updated_failed_record.status == SubmissionStatus.FAILED
        assert updated_failed_record.metadata["success"] is False

    def test_submission_attempt_tracking(self, audit_manager):
        """Test submission attempt tracking."""
        record = audit_manager.create_submission_record(
            question_id="test_question",
            prediction_value=0.75,
            reasoning="Test reasoning",
        )

        # Track first attempt (failed)
        audit_manager.track_submission_attempt(record.submission_id, 1, "Network error")

        # Track second attempt (successful)
        audit_manager.track_submission_attempt(record.submission_id, 2)

        updated_record = audit_manager.get_submission_record(record.submission_id)
        attempts = updated_record.metadata["submission_attempts"]

        assert len(attempts) == 2
        assert attempts[0]["attempt_number"] == 1
        assert attempts[0]["error"] == "Network error"
        assert attempts[0]["success"] is False
        assert attempts[1]["attempt_number"] == 2
        assert attempts[1]["error"] is None
        assert attempts[1]["success"] is True

    def test_submission_history_filtering(self, audit_manager):
        """Test filtered submission history retrieval."""
        # Create submissions with different characteristics
        record1 = audit_manager.create_submission_record(
            question_id="question_1", prediction_value=0.6, reasoning="First prediction"
        )
        audit_manager.update_submission_status(
            record1.submission_id, SubmissionStatus.SUBMITTED
        )

        record2 = audit_manager.create_submission_record(
            question_id="question_1",
            prediction_value=0.7,
            reasoning="Second prediction",
            dry_run=True,
        )

        record3 = audit_manager.create_submission_record(
            question_id="question_2",
            prediction_value=0.8,
            reasoning="Different question",
        )
        audit_manager.update_submission_status(
            record3.submission_id, SubmissionStatus.FAILED
        )

        # Test question filtering
        question_1_history = audit_manager.get_submission_history(
            question_id="question_1"
        )
        assert len(question_1_history) == 2

        # Test status filtering
        submitted_history = audit_manager.get_submission_history(
            status_filter=SubmissionStatus.SUBMITTED
        )
        assert len(submitted_history) == 1
        assert submitted_history[0].submission_id == record1.submission_id

        # Test dry run filtering
        dry_run_history = audit_manager.get_submission_history(dry_run_filter=True)
        assert len(dry_run_history) == 1
        assert dry_run_history[0].submission_id == record2.submission_id

        real_submission_history = audit_manager.get_submission_history(
            dry_run_filter=False
        )
        assert len(real_submission_history) == 2

    def test_performance_metrics(self, audit_manager):
        """Test performance metrics calculation."""
        # Create submissions with different outcomes
        record1 = audit_manager.create_submission_record(
            question_id="question_1", prediction_value=0.6, reasoning="First prediction"
        )
        audit_manager.update_submission_status(
            record1.submission_id, SubmissionStatus.SUBMITTED
        )

        record2 = audit_manager.create_submission_record(
            question_id="question_2",
            prediction_value=0.7,
            reasoning="Second prediction",
        )
        audit_manager.update_submission_status(
            record2.submission_id, SubmissionStatus.FAILED
        )

        record3 = audit_manager.create_submission_record(
            question_id="question_3",
            prediction_value=0.8,
            reasoning="Dry run prediction",
            dry_run=True,
        )

        metrics = audit_manager.get_performance_metrics()

        assert metrics["total_submissions"] == 3
        assert metrics["real_submissions"] == 2
        assert metrics["dry_run_submissions"] == 1
        assert metrics["success_rate"] == 0.5  # 1 success out of 2 real submissions

    def test_audit_trail_export(self, audit_manager):
        """Test audit trail export functionality."""
        # Create test submissions
        record1 = audit_manager.create_submission_record(
            question_id="question_1",
            prediction_value=0.75,
            reasoning="Test reasoning 1",
            confidence=0.8,
        )

        record2 = audit_manager.create_submission_record(
            question_id="question_2",
            prediction_value=0.65,
            reasoning="Test reasoning 2",
            dry_run=True,
        )

        # Test JSON export
        json_export = audit_manager.export_audit_trail(format="json")
        assert isinstance(json_export, str)

        import json

        exported_data = json.loads(json_export)
        assert len(exported_data) == 2

        # Test CSV export
        csv_export = audit_manager.export_audit_trail(format="csv")
        assert isinstance(csv_export, str)
        assert "submission_id" in csv_export
        assert "question_id" in csv_export

        # Test summary export
        summary_export = audit_manager.export_audit_trail(format="summary")
        assert isinstance(summary_export, str)
        assert "Total Submissions: 2" in summary_export

        # Test export without dry runs
        json_export_no_dry = audit_manager.export_audit_trail(
            format="json", include_dry_runs=False
        )
        exported_data_no_dry = json.loads(json_export_no_dry)
        assert len(exported_data_no_dry) == 1

    def test_confirmation_callbacks(self, audit_manager):
        """Test submission confirmation callbacks."""
        callback_called = False
        callback_record = None
        callback_response = None
        callback_success = None

        def test_callback(record, api_response, success):
            nonlocal callback_called, callback_record, callback_response, callback_success
            callback_called = True
            callback_record = record
            callback_response = api_response
            callback_success = success

        audit_manager.add_confirmation_callback(test_callback)

        record = audit_manager.create_submission_record(
            question_id="test_question",
            prediction_value=0.75,
            reasoning="Test reasoning",
        )

        api_response = {"status_code": 200, "message": "Success"}
        audit_manager.confirm_submission(
            record.submission_id, api_response, success=True
        )

        assert callback_called is True
        assert callback_record.submission_id == record.submission_id
        assert callback_response == api_response
        assert callback_success is True


class TestDryRunManager:
    """Test dry-run manager functionality."""

    @pytest.fixture
    def dry_run_manager(self, validator, audit_manager):
        """Create dry-run manager."""
        from src.infrastructure.external_apis.submission_validator import DryRunManager

        return DryRunManager(validator, audit_manager)

    def test_start_dry_run_session(self, dry_run_manager):
        """Test starting a dry-run session."""
        tournament_context = {
            "tournament_id": "test_tournament",
            "current_ranking": 25,
            "participant_count": 100,
        }

        session_id = dry_run_manager.start_dry_run_session(
            "Test Session", tournament_context
        )

        assert session_id is not None
        assert session_id in dry_run_manager.dry_run_sessions

        session = dry_run_manager.dry_run_sessions[session_id]
        assert session["session_name"] == "Test Session"
        assert session["tournament_context"] == tournament_context
        assert session["status"] == "active"
        assert len(session["submissions"]) == 0

    def test_simulate_submission(
        self, dry_run_manager, binary_question, valid_binary_prediction
    ):
        """Test submission simulation."""
        session_id = dry_run_manager.start_dry_run_session("Test Session")

        # Add tournament metadata to question
        binary_question.metadata.update(
            {
                "category": "technology",
                "tournament_priority": "high",
                "community_prediction": 0.3,
            }
        )

        agent_metadata = {
            "agent_type": "ensemble",
            "reasoning_method": "chain_of_thought",
        }

        simulation_results = dry_run_manager.simulate_submission(
            session_id, binary_question, valid_binary_prediction, agent_metadata
        )

        assert "submission_id" in simulation_results
        assert "session_id" in simulation_results
        assert "api_simulation" in simulation_results
        assert "competitive_analysis" in simulation_results
        assert "learning_opportunities" in simulation_results

        # Check that submission was added to session
        session = dry_run_manager.dry_run_sessions[session_id]
        assert len(session["submissions"]) == 1
        assert simulation_results["submission_id"] in session["submissions"]

        # Check API simulation
        api_sim = simulation_results["api_simulation"]
        assert "formatted_prediction" in api_sim
        assert "simulated_api_response" in api_sim
        assert "would_succeed" in api_sim

        # Check competitive analysis
        competitive_analysis = simulation_results["competitive_analysis"]
        if competitive_analysis["impact"] != "unknown":
            assert "current_ranking" in competitive_analysis
            assert "estimated_new_ranking" in competitive_analysis

    def test_end_dry_run_session(
        self, dry_run_manager, binary_question, valid_binary_prediction
    ):
        """Test ending a dry-run session."""
        session_id = dry_run_manager.start_dry_run_session("Test Session")

        # Add some simulations
        dry_run_manager.simulate_submission(
            session_id, binary_question, valid_binary_prediction
        )

        report = dry_run_manager.end_dry_run_session(session_id)

        assert "session_summary" in report
        assert "risk_analysis" in report
        assert "learning_analysis" in report
        assert "competitive_analysis" in report
        assert "recommendations" in report

        # Check session status
        session = dry_run_manager.dry_run_sessions[session_id]
        assert session["status"] == "completed"
        assert "end_time" in session
        assert "duration" in session

        # Check session summary
        session_summary = report["session_summary"]
        assert session_summary["total_submissions"] == 1
        assert session_summary["session_id"] == session_id

    def test_session_status_tracking(self, dry_run_manager):
        """Test session status tracking."""
        session_id = dry_run_manager.start_dry_run_session("Test Session")

        status = dry_run_manager.get_session_status(session_id)
        assert status["session_id"] == session_id
        assert status["status"] == "active"
        assert status["submissions_count"] == 0

        # Test unknown session
        unknown_status = dry_run_manager.get_session_status("unknown_session")
        assert "error" in unknown_status

    def test_list_active_sessions(self, dry_run_manager):
        """Test listing active sessions."""
        # Start multiple sessions
        session1_id = dry_run_manager.start_dry_run_session("Session 1")
        session2_id = dry_run_manager.start_dry_run_session("Session 2")

        active_sessions = dry_run_manager.list_active_sessions()
        assert len(active_sessions) == 2

        session_ids = [session["session_id"] for session in active_sessions]
        assert session1_id in session_ids
        assert session2_id in session_ids

        # End one session
        dry_run_manager.end_dry_run_session(session1_id)

        active_sessions = dry_run_manager.list_active_sessions()
        assert len(active_sessions) == 1
        assert active_sessions[0]["session_id"] == session2_id

    def test_api_response_simulation(self, dry_run_manager, binary_question):
        """Test API response simulation."""
        # Test with valid prediction
        valid_prediction = {
            "question_id": binary_question.id,
            "prediction_value": 0.75,
            "reasoning": "Valid reasoning",
        }

        api_sim = dry_run_manager._simulate_api_interaction(
            binary_question, valid_prediction
        )
        assert api_sim["would_succeed"] is True
        assert api_sim["simulated_api_response"]["success"] is True
        assert api_sim["simulated_api_response"]["status_code"] == 200

        # Test with invalid prediction
        invalid_prediction = {
            "question_id": binary_question.id,
            "prediction_value": 1.5,  # Invalid for binary
            "reasoning": "Invalid reasoning",
        }

        api_sim = dry_run_manager._simulate_api_interaction(
            binary_question, invalid_prediction
        )
        assert api_sim["would_succeed"] is False
        assert api_sim["simulated_api_response"]["success"] is False
        assert api_sim["simulated_api_response"]["status_code"] == 400

    def test_competitive_impact_simulation(self, dry_run_manager, binary_question):
        """Test competitive impact simulation."""
        tournament_context = {"current_ranking": 50, "participant_count": 100}

        prediction_data = {"prediction_value": 0.75, "confidence": 0.8}

        # Add tournament priority to question
        binary_question.metadata["tournament_priority"] = "high"

        competitive_impact = dry_run_manager._simulate_competitive_impact(
            binary_question, prediction_data, tournament_context
        )

        assert competitive_impact["current_ranking"] == 50
        assert "estimated_new_ranking" in competitive_impact
        assert "potential_change" in competitive_impact
        assert competitive_impact["estimated_new_ranking"] >= 1
        assert competitive_impact["estimated_new_ranking"] <= 100

        # Test without tournament context
        no_context_impact = dry_run_manager._simulate_competitive_impact(
            binary_question, prediction_data, None
        )
        assert no_context_impact["impact"] == "unknown"

    def test_learning_opportunities_identification(
        self, dry_run_manager, binary_question, valid_binary_prediction
    ):
        """Test learning opportunities identification."""
        # Create simulation results with various issues
        simulation_results = {
            "validation_results": {
                "result": "invalid",
                "errors": [{"code": "VALUE_OUT_OF_RANGE"}],
            },
            "risk_assessment": {
                "risk_level": "high",
                "identified_risks": ["low_confidence_prediction"],
            },
            "tournament_simulation": {
                "strategic_considerations": ["extreme_consensus"]
            },
            "timing_analysis": {"status": "critical"},
        }

        opportunities = dry_run_manager._identify_learning_opportunities(
            binary_question, valid_binary_prediction, simulation_results
        )

        # Should identify multiple learning opportunities
        assert len(opportunities) > 0

        opportunity_types = [opp["type"] for opp in opportunities]
        assert "validation_improvement" in opportunity_types
        assert "risk_management" in opportunity_types
        assert "time_management" in opportunity_types

    def test_session_report_generation(
        self, dry_run_manager, binary_question, valid_binary_prediction
    ):
        """Test comprehensive session report generation."""
        session_id = dry_run_manager.start_dry_run_session("Test Session")

        # Add multiple simulations with different characteristics
        for i in range(3):
            prediction = valid_binary_prediction.copy()
            prediction["prediction_value"] = 0.5 + i * 0.1
            dry_run_manager.simulate_submission(session_id, binary_question, prediction)

        report = dry_run_manager.end_dry_run_session(session_id)

        # Verify report structure
        assert "session_summary" in report
        assert "risk_analysis" in report
        assert "learning_analysis" in report
        assert "competitive_analysis" in report
        assert "recommendations" in report

        # Verify session summary
        session_summary = report["session_summary"]
        assert session_summary["total_submissions"] == 3
        assert "validation_success_rate" in session_summary

        # Verify risk analysis
        risk_analysis = report["risk_analysis"]
        assert "risk_distribution" in risk_analysis
        assert "risk_rate" in risk_analysis

        # Verify learning analysis
        learning_analysis = report["learning_analysis"]
        assert "total_opportunities" in learning_analysis
        assert "opportunity_types" in learning_analysis

        # Verify recommendations
        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
