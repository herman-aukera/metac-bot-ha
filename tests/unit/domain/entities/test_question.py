"""Unit tests for Question entity."""

import pytest
from datetime import datetime, timedelta

from src.domain.entities.question import Question, QuestionType, QuestionStatus, QuestionCategory


class TestQuestion:
    """Test cases for Question entity."""

    def test_question_initialization_valid(self):
        """Test valid question initialization."""
        deadline = datetime.utcnow() + timedelta(days=30)

        question = Question(
            id=1,
            text="Will AI achieve AGI by 2030?",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background="Background information about AGI development",
            resolution_criteria="Clear criteria for resolution",
            scoring_weight=2.0,
            metadata={"source": "test"},
            min_value=None,
            max_value=None,
            choices=None
        )

        assert question.id == 1
        assert question.text == "Will AI achieve AGI by 2030?"
        assert question.question_type == QuestionType.BINARY
        assert question.category == QuestionCategory.AI_DEVELOPMENT
        assert question.deadline == deadline
        assert question.scoring_weight == 2.0
        assert question.metadata == {"source": "test"}

    def test_question_validation_invalid_id(self):
        """Test question validation with invalid ID."""
        deadline = datetime.utcnow() + timedelta(days=30)

        with pytest.raises(ValueError, match="Question ID must be positive"):
            Question(
                id=0,
                text="Test question",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=deadline,
                background="Background",
                resolution_criteria="Criteria"
            )

    def test_question_validation_empty_text(self):
        """Test question validation with empty text."""
        deadline = datetime.utcnow() + timedelta(days=30)

        with pytest.raises(ValueError, match="Question text cannot be empty"):
            Question(
                id=1,
                text="",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=deadline,
                background="Background",
                resolution_criteria="Criteria"
            )

    def test_question_validation_empty_background(self):
        """Test question validation with empty background."""
        deadline = datetime.utcnow() + timedelta(days=30)

        with pytest.raises(ValueError, match="Question background cannot be empty"):
            Question(
                id=1,
                text="Test question",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=deadline,
                background="",
                resolution_criteria="Criteria"
            )

    def test_question_validation_empty_resolution_criteria(self):
        """Test question validation with empty resolution criteria."""
        deadline = datetime.utcnow() + timedelta(days=30)

        with pytest.raises(ValueError, match="Resolution criteria cannot be empty"):
            Question(
                id=1,
                text="Test question",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=deadline,
                background="Background",
                resolution_criteria=""
            )

    def test_question_validation_invalid_scoring_weight(self):
        """Test question validation with invalid scoring weight."""
        deadline = datetime.utcnow() + timedelta(days=30)

        with pytest.raises(ValueError, match="Scoring weight must be positive"):
            Question(
                id=1,
                text="Test question",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=deadline,
                background="Background",
                resolution_criteria="Criteria",
                scoring_weight=0.0
            )

    def test_question_validation_past_deadline(self):
        """Test question validation with past deadline."""
        deadline = datetime.utcnow() - timedelta(days=1)

        with pytest.raises(ValueError, match="Question deadline must be in the future"):
            Question(
                id=1,
                text="Test question",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=deadline,
                background="Background",
                resolution_criteria="Criteria"
            )

    def test_question_numeric_validation_missing_bounds(self):
        """Test numeric question validation with missing bounds."""
        deadline = datetime.utcnow() + timedelta(days=30)

        with pytest.raises(ValueError, match="Numeric questions must have min_value and max_value"):
            Question(
                id=1,
                text="What will be the temperature?",
                question_type=QuestionType.NUMERIC,
                category=QuestionCategory.SCIENCE,
                deadline=deadline,
                background="Background",
                resolution_criteria="Criteria",
                min_value=None,
                max_value=100.0
            )

    def test_question_numeric_validation_invalid_bounds(self):
        """Test numeric question validation with invalid bounds."""
        deadline = datetime.utcnow() + timedelta(days=30)

        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            Question(
                id=1,
                text="What will be the temperature?",
                question_type=QuestionType.NUMERIC,
                category=QuestionCategory.SCIENCE,
                deadline=deadline,
                background="Background",
                resolution_criteria="Criteria",
                min_value=100.0,
                max_value=50.0
            )

    def test_question_multiple_choice_validation_missing_choices(self):
        """Test multiple choice question validation with missing choices."""
        deadline = datetime.utcnow() + timedelta(days=30)

        with pytest.raises(ValueError, match="Multiple choice questions must have at least 2 choices"):
            Question(
                id=1,
                text="Which option is correct?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                category=QuestionCategory.OTHER,
                deadline=deadline,
                background="Background",
                resolution_criteria="Criteria",
                choices=["A"]
            )

    def test_question_type_checks(self):
        """Test question type checking methods."""
        deadline = datetime.utcnow() + timedelta(days=30)

        binary_question = Question(
            id=1,
            text="Binary question?",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria"
        )

        assert binary_question.is_binary()
        assert not binary_question.is_numeric()
        assert not binary_question.is_multiple_choice()
        assert not binary_question.is_date()
        assert not binary_question.is_conditional()

        numeric_question = Question(
            id=2,
            text="Numeric question?",
            question_type=QuestionType.NUMERIC,
            category=QuestionCategory.SCIENCE,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria",
            min_value=0.0,
            max_value=100.0
        )

        assert not numeric_question.is_binary()
        assert numeric_question.is_numeric()
        assert not numeric_question.is_multiple_choice()

    def test_question_time_methods(self):
        """Test time-related methods."""
        deadline = datetime.utcnow() + timedelta(hours=12)

        question = Question(
            id=1,
            text="Test question",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria"
        )

        # Test time until deadline
        time_remaining = question.time_until_deadline()
        assert 11.5 < time_remaining < 12.5  # Allow some tolerance

        # Test deadline approaching
        assert question.is_deadline_approaching(24.0)  # Within 24 hours
        assert not question.is_deadline_approaching(6.0)  # Not within 6 hours

    def test_question_value_assessment(self):
        """Test value assessment methods."""
        deadline = datetime.utcnow() + timedelta(days=30)

        high_value_question = Question(
            id=1,
            text="High value question",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria",
            scoring_weight=3.0
        )

        low_value_question = Question(
            id=2,
            text="Low value question",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.OTHER,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria",
            scoring_weight=1.0
        )

        assert high_value_question.is_high_value(2.0)
        assert not low_value_question.is_high_value(2.0)

    def test_question_specialized_knowledge(self):
        """Test specialized knowledge requirement check."""
        deadline = datetime.utcnow() + timedelta(days=30)

        specialized_question = Question(
            id=1,
            text="AI question",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria"
        )

        general_question = Question(
            id=2,
            text="General question",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.OTHER,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria"
        )

        assert specialized_question.requires_specialized_knowledge()
        assert not general_question.requires_specialized_knowledge()

    def test_question_complexity_score(self):
        """Test complexity score calculation."""
        deadline = datetime.utcnow() + timedelta(days=30)

        simple_question = Question(
            id=1,
            text="Simple binary question?",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.OTHER,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria"
        )

        complex_question = Question(
            id=2,
            text="This is a very complex conditional question with lots of details and nuanced considerations that require deep analysis and specialized knowledge to properly evaluate and understand the implications of various outcomes and scenarios that might unfold over time.",
            question_type=QuestionType.CONDITIONAL,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria"
        )

        simple_score = simple_question.get_complexity_score()
        complex_score = complex_question.get_complexity_score()

        assert complex_score > simple_score
        assert simple_score >= 1.0
        assert complex_score > 2.0

    def test_question_summary(self):
        """Test question summary generation."""
        deadline = datetime.utcnow() + timedelta(days=30)

        question = Question(
            id=123,
            text="Test question",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria"
        )

        summary = question.to_summary()
        assert "Q123:" in summary
        assert "binary" in summary
        assert "ai_development" in summary
        assert "deadline:" in summary

    def test_question_defaults(self):
        """Test question default values."""
        deadline = datetime.utcnow() + timedelta(days=30)

        question = Question(
            id=1,
            text="Test question",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background="Background",
            resolution_criteria="Criteria"
        )

        # Test default values
        assert question.scoring_weight == 1.0
        assert question.metadata == {}
        assert question.created_at is not None
        assert isinstance(question.created_at, datetime)

    def test_question_enums(self):
        """Test question enum values."""
        # Test QuestionType enum
        assert QuestionType.BINARY.value == "binary"
        assert QuestionType.NUMERIC.value == "numeric"
        assert QuestionType.MULTIPLE_CHOICE.value == "multiple_choice"
        assert QuestionType.DATE.value == "date"
        assert QuestionType.CONDITIONAL.value == "conditional"

        # Test QuestionStatus enum
        assert QuestionStatus.ACTIVE.value == "active"
        assert QuestionStatus.CLOSED.value == "closed"
        assert QuestionStatus.RESOLVED.value == "resolved"
        assert QuestionStatus.CANCELLED.value == "cancelled"

        # Test QuestionCategory enum
        assert QuestionCategory.AI_DEVELOPMENT.value == "ai_development"
        assert QuestionCategory.TECHNOLOGY.value == "technology"
        assert QuestionCategory.OTHER.value == "other"
