"""
Unit tests for IngestionService application component.
Tests JSON parsing, validation, and error handling.
"""

import pytest
from datetime import datetime, timezone
from src.application.ingestion_service import (
    IngestionService, ValidationLevel, IngestionStats, IngestionError
)
from src.domain.entities import Question, QuestionType


class TestIngestionService:
    """Test suite for IngestionService class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.service = IngestionService(validation_level=ValidationLevel.LENIENT)

    def test_parse_question_valid_binary(self):
        """Test parsing a valid binary question."""
        # Arrange
        question_data = {
            "id": 123,
            "title": "Will it rain tomorrow?",
            "description": "A test question about weather",
            "type": "binary",
            "url": "https://example.com/123",
            "close_time": "2024-12-31T23:59:59Z",
            "community_prediction": 0.7,
            "num_forecasters": 100,
            "status": "open"
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.id == 123
        assert result.title == "Will it rain tomorrow?"
        assert result.question_type == QuestionType.BINARY
        assert result.community_prediction == 0.7

    def test_parse_question_valid_multiple_choice(self):
        """Test parsing a valid multiple choice question."""
        # Arrange
        question_data = {
            "id": 124,
            "title": "Which team will win?",
            "description": "A test question about sports",
            "question_type": "multiple_choice",
            "url": "https://example.com/124",
            "close_time": "2024-12-31T23:59:59Z",
            "created_time": "2024-01-01T00:00:00Z",
            "possibilities": {
                "type": "multiple_choice",
                "choices": ["Team A", "Team B", "Team C"]
            },
            "status": "open"
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.question_type == QuestionType.MULTIPLE_CHOICE
        assert "choices" in result.possibilities

    def test_parse_question_valid_numeric(self):
        """Test parsing a valid numeric question."""
        # Arrange
        question_data = {
            "id": 125,
            "title": "What will be the temperature?",
            "description": "A test question about temperature",
            "question_type": "numeric",
            "url": "https://example.com/125",
            "close_time": "2024-12-31T23:59:59Z",
            "created_time": "2024-01-01T00:00:00Z",
            "possibilities": {
                "type": "numeric",
                "min": -10,
                "max": 50
            },
            "status": "open"
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.question_type == QuestionType.NUMERIC
        assert "min" in result.possibilities
        assert "max" in result.possibilities

    def test_parse_question_missing_required_field_strict(self):
        """Test parsing fails with missing required field in strict mode."""
        # Arrange
        service = IngestionService(validation_level=ValidationLevel.STRICT)
        question_data = {
            "id": 126,
            "title": "Incomplete question",
            # Missing description, url, etc.
        }

        # Act & Assert
        with pytest.raises(IngestionError, match="Missing required field"):
            service.parse_question(question_data)

    def test_parse_question_missing_optional_field_lenient(self):
        """Test parsing succeeds with missing optional field in lenient mode."""
        # Arrange
        question_data = {
            "id": 127,
            "title": "Minimal question",
            "question_type": "binary",
            "status": "open"
            # Missing many optional fields
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.id == 127
        assert result.title == "Minimal question"

    def test_parse_question_invalid_question_type(self):
        """Test parsing fails with invalid question type."""
        # Arrange
        question_data = {
            "id": 128,
            "title": "Invalid type question",
            "question_type": "invalid_type",
            "status": "open"
        }

        # Act & Assert
        with pytest.raises(IngestionError, match="Unknown question type"):
            self.service.parse_question(question_data)

    def test_parse_question_invalid_community_prediction(self):
        """Test parsing fails with invalid community prediction."""
        # Arrange
        question_data = {
            "id": 129,
            "title": "Invalid prediction question",
            "question_type": "binary",
            "community_prediction": 1.5,  # Invalid: > 1.0
            "status": "open"
        }

        # Act & Assert
        with pytest.raises(IngestionError, match="Community prediction must be between 0 and 1"):
            self.service.parse_question(question_data)

    def test_parse_question_invalid_date_format(self):
        """Test parsing fails with invalid date format."""
        # Arrange
        question_data = {
            "id": 130,
            "title": "Invalid date question",
            "question_type": "binary",
            "close_time": "invalid-date-format",
            "status": "open"
        }

        # Act & Assert
        with pytest.raises(IngestionError, match="Invalid date format"):
            self.service.parse_question(question_data)

    def test_parse_questions_multiple_valid(self):
        """Test parsing multiple valid questions."""
        # Arrange
        questions_data = [
            {
                "id": 131,
                "title": "Question 1",
                "question_type": "binary",
                "status": "open"
            },
            {
                "id": 132,
                "title": "Question 2",
                "question_type": "binary",
                "status": "open"
            }
        ]

        # Act
        results, stats = self.service.parse_questions(questions_data)

        # Assert
        assert len(results) == 2
        assert stats.total_processed == 2
        assert stats.successful_parses == 2
        assert stats.failed_parses == 0

    def test_parse_questions_with_failures(self):
        """Test parsing questions with some failures."""
        # Arrange
        questions_data = [
            {
                "id": 133,
                "title": "Valid question",
                "question_type": "binary",
                "status": "open"
            },
            {
                "id": 134,
                "title": "Invalid question",
                "question_type": "invalid_type",  # Will fail
                "status": "open"
            }
        ]

        # Act
        results, stats = self.service.parse_questions(questions_data)

        # Assert
        assert len(results) == 1  # Only the valid one
        assert stats.total_processed == 2
        assert stats.successful_parses == 1
        assert stats.failed_parses == 1
        assert len(stats.errors) == 1

    def test_parse_questions_empty_list(self):
        """Test parsing empty questions list."""
        # Act
        results, stats = self.service.parse_questions([])

        # Assert
        assert results == []
        assert stats.total_processed == 0
        assert stats.successful_parses == 0
        assert stats.failed_parses == 0

    def test_validation_level_strict_requirements(self):
        """Test strict validation level enforces all required fields."""
        # Arrange
        service = IngestionService(validation_level=ValidationLevel.STRICT)
        minimal_data = {
            "id": 135,
            "title": "Minimal",
            "question_type": "binary"
        }

        # Act & Assert
        with pytest.raises(IngestionError):
            service.parse_question(minimal_data)

    def test_validation_level_minimal_requirements(self):
        """Test minimal validation level allows very basic questions."""
        # Arrange
        service = IngestionService(validation_level=ValidationLevel.MINIMAL)
        minimal_data = {
            "id": 136,
            "title": "Very minimal"
        }

        # Act
        result = service.parse_question(minimal_data)

        # Assert
        assert result is not None
        assert result.id == 136

    def test_ingestion_stats_tracking(self):
        """Test that ingestion statistics are properly tracked."""
        # Arrange
        questions_data = [
            {"id": 137, "title": "Q1", "question_type": "binary", "status": "open"},
            {"id": 138, "title": "Q2", "question_type": "invalid"},  # Will fail
            {"id": 139, "title": "Q3", "question_type": "binary", "status": "open"}
        ]

        # Act
        results, stats = self.service.parse_questions(questions_data)

        # Assert
        assert stats.total_processed == 3
        assert stats.successful_parses == 2
        assert stats.failed_parses == 1
        assert stats.success_rate == 2/3
        assert len(stats.errors) == 1
        assert isinstance(stats.processing_time, float)

    def test_ingestion_error_inheritance(self):
        """Test IngestionError is properly defined."""
        # Test that it's a proper exception
        error = IngestionError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_validation_level_enum(self):
        """Test ValidationLevel enum values."""
        assert ValidationLevel.STRICT == "strict"
        assert ValidationLevel.LENIENT == "lenient"
        assert ValidationLevel.MINIMAL == "minimal"

    def test_parse_question_with_tags(self):
        """Test parsing question with tags."""
        # Arrange
        question_data = {
            "id": 140,
            "title": "Tagged question",
            "question_type": "binary",
            "tags": ["tag1", "tag2", "tag3"],
            "status": "open"
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.tags == ["tag1", "tag2", "tag3"]

    def test_parse_question_resolved_with_resolution(self):
        """Test parsing resolved question with resolution value."""
        # Arrange
        question_data = {
            "id": 141,
            "title": "Resolved question",
            "question_type": "binary",
            "status": "resolved",
            "is_resolved": True,
            "resolution": True
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.is_resolved is True
        assert result.resolution is True

    def test_parse_question_with_category(self):
        """Test parsing question with category."""
        # Arrange
        question_data = {
            "id": 142,
            "title": "Categorized question",
            "question_type": "binary",
            "category": "technology",
            "status": "open"
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.category == "technology"

    def test_concurrent_parsing_safety(self):
        """Test that parsing is thread-safe (basic check)."""
        # This is a basic test - in a real scenario you'd use threading
        # Arrange
        question_data = {
            "id": 143,
            "title": "Concurrent test",
            "question_type": "binary",
            "status": "open"
        }

        # Act - parse the same question multiple times
        results = []
        for _ in range(5):
            result = self.service.parse_question(question_data)
            results.append(result)

        # Assert - all results should be identical
        for result in results:
            assert result.id == 143
            assert result.title == "Concurrent test"
