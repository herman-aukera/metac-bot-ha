"""Unit tests for Question domain entity."""

from datetime import datetime, timedelta, timezone


from src.domain.entities.question import Question, QuestionType


def test_can_create_binary_question():
    """Test that we can create a binary question type."""
    question = Question.create_new(
        metaculus_id=123,
        title="Will it rain tomorrow?",
        description="Simple binary question about weather",
        question_type=QuestionType.BINARY,
        url="https://example.com/question/123",
        close_time=datetime.now(timezone.utc) + timedelta(days=7),
        categories=["weather"],
    )

    assert question.question_type == QuestionType.BINARY
    assert question.title == "Will it rain tomorrow?"
    assert question.metaculus_id == 123
    assert len(question.categories) == 1
    assert question.categories[0] == "weather"


def test_is_resolved_false_when_no_resolution():
    """Test that is_resolved returns False when there's no resolution time."""
    question = Question.create_new(
        metaculus_id=123,
        title="Will it rain tomorrow?",
        description="Simple binary question about weather",
        question_type=QuestionType.BINARY,
        url="https://example.com/question/123",
        close_time=datetime.now(timezone.utc) + timedelta(days=7),
        categories=["weather"],
    )

    assert not question.is_resolved()


def test_is_resolved_true_when_resolution_in_past():
    """Test that is_resolved returns True when resolution time is in the past."""
    question = Question.create_new(
        metaculus_id=123,
        title="Will it rain tomorrow?",
        description="Simple binary question about weather",
        question_type=QuestionType.BINARY,
        url="https://example.com/question/123",
        close_time=datetime.now(timezone.utc) - timedelta(days=1),
        categories=["weather"],
        resolve_time=datetime.now(timezone.utc) - timedelta(hours=1),  # Past resolution
    )

    assert question.is_resolved()
