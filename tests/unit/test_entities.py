"""Unit tests for domain entities."""

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from src.domain.entities.question import Question, QuestionStatus, QuestionType
from src.domain.value_objects.probability import Probability


class TestQuestionType:
    """Test QuestionType enum."""

    def test_question_type_values(self):
        """Test that question type enum has expected values."""
        assert QuestionType.BINARY.value == "binary"
        assert QuestionType.MULTIPLE_CHOICE.value == "multiple_choice"
        assert QuestionType.NUMERIC.value == "numeric"
        assert QuestionType.DATE.value == "date"


class TestQuestion:
    """Test Question domain entity."""

    def test_create_binary_question(self):
        """Test creating a binary question."""
        question_id = uuid4()
        close_time = datetime.utcnow() + timedelta(days=30)

        question = Question(
            id=question_id,
            metaculus_id=12345,
            title="Will AI achieve AGI by 2030?",
            description="Test description",
            question_type=QuestionType.BINARY,
            status=QuestionStatus.OPEN,
            url="https://metaculus.com/questions/12345",
            close_time=close_time,
            resolve_time=None,
            categories=["AI", "Technology"],
            metadata={"source": "metaculus"},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert question.id == question_id
        assert question.metaculus_id == 12345
        assert question.title == "Will AI achieve AGI by 2030?"
        assert question.question_type == QuestionType.BINARY
        assert question.categories == ["AI", "Technology"]

    def test_create_multiple_choice_question(self):
        """Test creating a multiple choice question."""
        choices = ["Option A", "Option B", "Option C"]

        question = Question(
            id=uuid4(),
            metaculus_id=12346,
            title="Which AI company will achieve AGI first?",
            description="Test description",
            question_type=QuestionType.MULTIPLE_CHOICE,
            status=QuestionStatus.OPEN,
            url="https://metaculus.com/questions/12346",
            close_time=datetime.utcnow() + timedelta(days=30),
            resolve_time=None,
            categories=["AI"],
            metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            choices=choices,
        )

        assert question.choices == choices
        assert question.question_type == QuestionType.MULTIPLE_CHOICE

    def test_create_numeric_question(self):
        """Test creating a numeric question."""
        question = Question(
            id=uuid4(),
            metaculus_id=12347,
            title="What will be the global temperature increase by 2030?",
            description="Test description",
            question_type=QuestionType.NUMERIC,
            status=QuestionStatus.OPEN,
            url="https://metaculus.com/questions/12347",
            close_time=datetime.utcnow() + timedelta(days=30),
            resolve_time=None,
            categories=["Climate"],
            metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            min_value=0.0,
            max_value=5.0,
        )

        assert question.min_value == 0.0
        assert question.max_value == 5.0
        assert question.question_type == QuestionType.NUMERIC

    def test_multiple_choice_validation_error(self):
        """Test that multiple choice questions require choices."""
        with pytest.raises(
            ValueError, match="Multiple choice questions must have choices"
        ):
            Question(
                id=uuid4(),
                metaculus_id=12346,
                title="Test question",
                description="Test description",
                question_type=QuestionType.MULTIPLE_CHOICE,
                status=QuestionStatus.OPEN,
                url="https://metaculus.com/questions/12346",
                close_time=datetime.utcnow() + timedelta(days=30),
                resolve_time=None,
                categories=["Test"],
                metadata={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                # Missing choices
            )

    def test_numeric_validation_error(self):
        """Test that numeric questions require min/max values."""
        with pytest.raises(
            ValueError, match="Numeric questions must have min and max values"
        ):
            Question(
                id=uuid4(),
                metaculus_id=12347,
                title="Test question",
                description="Test description",
                question_type=QuestionType.NUMERIC,
                status=QuestionStatus.OPEN,
                url="https://metaculus.com/questions/12347",
                close_time=datetime.utcnow() + timedelta(days=30),
                resolve_time=None,
                categories=["Test"],
                metadata={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                # Missing min_value and max_value
            )

    def test_create_new_factory_method(self):
        """Test Question.create_new factory method."""
        close_time = datetime.utcnow() + timedelta(days=30)

        question = Question.create_new(
            metaculus_id=12345,
            title="Test Question",
            description="Test description",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=close_time,
            categories=["AI"],
            metadata={"test": "value"},
        )

        assert isinstance(question.id, UUID)
        assert question.metaculus_id == 12345
        assert question.title == "Test Question"
        assert question.metadata == {"test": "value"}
        assert isinstance(question.created_at, datetime)
        assert isinstance(question.updated_at, datetime)

    def test_is_open(self):
        """Test is_open method."""
        future_close = datetime.utcnow() + timedelta(days=30)
        past_close = datetime.utcnow() - timedelta(days=1)

        open_question = Question.create_new(
            metaculus_id=12345,
            title="Open Question",
            description="Test",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=future_close,
            categories=["Test"],
        )

        closed_question = Question.create_new(
            metaculus_id=12346,
            title="Closed Question",
            description="Test",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12346",
            close_time=past_close,
            categories=["Test"],
        )

        assert open_question.is_open() is True
        assert closed_question.is_open() is False

    def test_is_resolved(self):
        """Test is_resolved method."""
        past_resolve = datetime.utcnow() - timedelta(days=1)
        future_resolve = datetime.utcnow() + timedelta(days=30)

        resolved_question = Question.create_new(
            metaculus_id=12345,
            title="Resolved Question",
            description="Test",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=datetime.utcnow() + timedelta(days=60),
            categories=["Test"],
            resolve_time=past_resolve,
        )

        unresolved_question = Question.create_new(
            metaculus_id=12346,
            title="Unresolved Question",
            description="Test",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12346",
            close_time=datetime.utcnow() + timedelta(days=60),
            categories=["Test"],
            resolve_time=future_resolve,
        )

        no_resolve_question = Question.create_new(
            metaculus_id=12347,
            title="No Resolve Question",
            description="Test",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12347",
            close_time=datetime.utcnow() + timedelta(days=60),
            categories=["Test"],
        )

        assert resolved_question.is_resolved() is True
        assert unresolved_question.is_resolved() is False
        assert no_resolve_question.is_resolved() is False

    def test_days_until_close(self):
        """Test days_until_close method."""
        future_close = datetime.now(timezone.utc) + timedelta(days=30)
        past_close = datetime.now(timezone.utc) - timedelta(days=1)

        open_question = Question.create_new(
            metaculus_id=12345,
            title="Open Question",
            description="Test",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=future_close,
            categories=["Test"],
        )

        closed_question = Question.create_new(
            metaculus_id=12346,
            title="Closed Question",
            description="Test",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12346",
            close_time=past_close,
            categories=["Test"],
        )

        assert open_question.days_until_close() >= 29  # Allow for test execution time
        assert closed_question.days_until_close() == 0

    def test_update_metadata(self):
        """Test update_metadata method."""
        question = Question.create_new(
            metaculus_id=12345,
            title="Test Question",
            description="Test",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=datetime.utcnow() + timedelta(days=30),
            categories=["Test"],
        )

        original_updated_at = question.updated_at

        # Wait a small amount to ensure timestamp difference
        import time

        time.sleep(0.01)

        question.update_metadata("new_key", "new_value")

        assert question.metadata["new_key"] == "new_value"
        assert question.updated_at > original_updated_at


class TestProbability:
    """Test Probability value object."""

    def test_valid_probability_creation(self):
        """Test creating valid probability values."""
        prob_0 = Probability(0.0)
        prob_half = Probability(0.5)
        prob_1 = Probability(1.0)

        assert prob_0.value == 0.0
        assert prob_half.value == 0.5
        assert prob_1.value == 1.0

    def test_invalid_probability_creation(self):
        """Test that invalid probabilities raise ValueError."""
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            Probability(-0.1)

        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            Probability(1.1)

    def test_probability_operations(self):
        """Test probability arithmetic operations."""
        prob1 = Probability(0.3)
        prob2 = Probability(0.4)

        # Test complement
        complement = prob1.complement()
        assert complement.value == 0.7

        # Test addition
        result = prob1 + prob2
        assert result.value == 0.7

        # Test multiplication
        result_mul = prob1 * prob2
        assert result_mul.value == 0.12

        # Test with float
        result_float = prob1 + 0.2
        assert result_float.value == 0.5

    def test_probability_comparison(self):
        """Test probability comparison operations."""
        prob1 = Probability(0.3)
        prob2 = Probability(0.7)
        prob3 = Probability(0.3)

        assert prob1 < prob2
        assert prob2 > prob1
        assert prob1 == prob3
        assert prob1 != prob2

    def test_probability_string_representation(self):
        """Test probability string representation."""
        prob = Probability(0.42)
        assert str(prob) == "42.0%"
        assert repr(prob) == "Probability(value=0.42)"

    def test_probability_from_percentage(self):
        """Test creating probability from percentage."""
        prob = Probability.from_percentage(42.0)
        assert prob.value == 0.42
        assert prob.to_percentage() == 42.0
