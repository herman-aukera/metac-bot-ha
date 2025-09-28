"""
Unit tests for IngestionService application component.
Tests JSON parsing, validation, and error handling.
"""

import pytest

from src.application.ingestion_service import (
    IngestionError,
    IngestionService,
    IngestionStats,
    ParseError,
    ValidationError,
    ValidationLevel,
)
from src.domain.entities.question import QuestionStatus, QuestionType


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
            "status": "open",
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.metaculus_id == 123
        assert result.title == "Will it rain tomorrow?"
        assert result.question_type == QuestionType.BINARY
        assert result.status == QuestionStatus.OPEN
        assert result.url == "https://example.com/123"
        assert result.choices is None
        assert result.min_value is None
        assert result.max_value is None

    def test_parse_question_valid_multiple_choice(self):
        """Test parsing a valid multiple choice question."""
        # Arrange
        question_data = {
            "id": 124,
            "title": "What will be the weather?",
            "description": "Multiple choice weather question",
            "type": "multiple_choice",
            "url": "https://example.com/124",
            "close_time": "2024-12-31T23:59:59Z",
            "choices": ["Sunny", "Rainy", "Cloudy"],
            "status": "open",
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.metaculus_id == 124
        assert result.question_type == QuestionType.MULTIPLE_CHOICE
        assert result.choices == ["Sunny", "Rainy", "Cloudy"]
        assert result.min_value is None
        assert result.max_value is None

    def test_parse_question_valid_numeric(self):
        """Test parsing a valid numeric question."""
        # Arrange
        question_data = {
            "id": 125,
            "title": "What will be the temperature?",
            "description": "Numeric temperature question",
            "type": "numeric",
            "url": "https://example.com/125",
            "close_time": "2024-12-31T23:59:59Z",
            "min_value": -10.0,
            "max_value": 50.0,
            "status": "open",
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.metaculus_id == 125
        assert result.question_type == QuestionType.NUMERIC
        assert result.min_value == -10.0
        assert result.max_value == 50.0
        assert result.choices is None

    def test_parse_questions_multiple_valid(self):
        """Test parsing multiple valid questions."""
        # Arrange
        questions_data = [
            {
                "id": 123,
                "title": "Question 1",
                "description": "First question",
                "type": "binary",
                "url": "https://example.com/123",
                "close_time": "2024-12-31T23:59:59Z",
                "status": "open",
            },
            {
                "id": 124,
                "title": "Question 2",
                "description": "Second question",
                "type": "multiple_choice",
                "url": "https://example.com/124",
                "close_time": "2024-12-31T23:59:59Z",
                "choices": ["A", "B", "C"],
                "status": "open",
            },
        ]

        # Act
        questions, stats = self.service.parse_questions(questions_data)

        # Assert
        assert len(questions) == 2
        assert stats.total_processed == 2
        assert stats.successful_parsed == 2
        assert stats.failed_parsing == 0
        assert stats.success_rate == 100.0
        assert stats.processing_time_seconds > 0

    def test_parse_questions_with_errors(self):
        """Test parsing questions with some errors."""
        # Arrange
        questions_data = [
            {
                "id": 123,
                "title": "Valid Question",
                "description": "This is valid",
                "type": "binary",
                "url": "https://example.com/123",
                "close_time": "2024-12-31T23:59:59Z",
                "status": "open",
            },
            {
                # Missing required fields
                "id": 124,
                "title": "",  # Empty title
                "description": "Invalid question",
            },
        ]

        # Act
        questions, stats = self.service.parse_questions(questions_data)

        # Assert
        assert len(questions) == 1  # Only one valid question
        assert stats.total_processed == 2
        assert stats.successful_parsed == 1
        assert stats.failed_parsing == 1
        assert stats.success_rate == 50.0

    def test_parse_questions_empty_list(self):
        """Test parsing empty list of questions."""
        # Arrange
        questions_data = []

        # Act
        questions, stats = self.service.parse_questions(questions_data)

        # Assert
        assert len(questions) == 0
        assert stats.total_processed == 0
        assert stats.successful_parsed == 0
        assert stats.failed_parsing == 0
        assert stats.success_rate == 0.0

    def test_parse_questions_mixed_success_failure(self):
        """Test parsing with mixed success and failure cases."""
        # Arrange
        questions_data = [
            {
                "id": 123,
                "title": "Valid Question 1",
                "description": "Valid",
                "type": "binary",
                "url": "https://example.com/123",
                "close_time": "2024-12-31T23:59:59Z",
                "status": "open",
            },
            {
                "id": 124,
                "title": "Valid Question 2",
                "description": "Also valid",
                "type": "numeric",
                "url": "https://example.com/124",
                "close_time": "2024-12-31T23:59:59Z",
                "min_value": 0,
                "max_value": 100,
                "status": "open",
            },
            {
                # Invalid question - malformed data that will cause parsing failure
                "id": "not_a_number",  # Invalid ID type that will cause parsing error
                "title": "Invalid Question",
            },
        ]

        # Act
        questions, stats = self.service.parse_questions(questions_data)

        # Assert
        assert len(questions) == 2
        assert stats.total_processed == 3
        assert stats.successful_parsed == 2
        assert stats.failed_parsing == 1
        assert abs(stats.success_rate - 66.67) < 0.1  # Approximately 66.67%

    def test_parse_question_with_categories(self):
        """Test parsing question with categories."""
        # Arrange
        question_data = {
            "id": 123,
            "title": "Question with categories",
            "description": "Has categories",
            "type": "binary",
            "url": "https://example.com/123",
            "close_time": "2024-12-31T23:59:59Z",
            "categories": ["Politics", "Economics"],
            "status": "open",
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert set(result.categories) == {
            "Politics",
            "Economics",
        }  # Categories as set (order not guaranteed)

    def test_parse_question_with_resolve_time(self):
        """Test parsing question with resolve time."""
        # Arrange
        question_data = {
            "id": 123,
            "title": "Resolved question",
            "description": "Already resolved",
            "type": "binary",
            "url": "https://example.com/123",
            "close_time": "2024-12-31T23:59:59Z",
            "resolve_time": "2024-01-15T12:00:00Z",
            "status": "resolved",
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.status == QuestionStatus.RESOLVED
        assert result.resolve_time is not None
        assert result.resolve_time.year == 2024
        assert result.resolve_time.month == 1
        assert result.resolve_time.day == 15

    def test_parse_question_missing_required_field_strict(self):
        """Test parsing with missing required field in strict mode."""
        # Arrange
        strict_service = IngestionService(validation_level=ValidationLevel.STRICT)
        question_data = {
            # Missing id field
            "title": "Question without ID",
            "description": "Missing ID",
            "type": "binary",
            "url": "https://example.com/test",
            "close_time": "2024-12-31T23:59:59Z",
            "status": "open",
        }

        # Act & Assert
        with pytest.raises((ValidationError, ParseError)):
            strict_service.parse_question(question_data)

    def test_parse_question_missing_required_field_lenient(self):
        """Test parsing with missing required field in lenient mode."""
        # Arrange
        question_data = {
            # Missing id field
            "title": "Question without ID",
            "description": "Missing ID",
            "type": "binary",
            "url": "https://example.com/test",
            "close_time": "2024-12-31T23:59:59Z",
            "status": "open",
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.metaculus_id == -1  # Default value for missing ID

    def test_parse_question_invalid_question_type(self):
        """Test parsing with invalid question type."""
        # Arrange
        question_data = {
            "id": 123,
            "title": "Invalid type question",
            "description": "Has invalid type",
            "type": "invalid_type",
            "url": "https://example.com/123",
            "close_time": "2024-12-31T23:59:59Z",
            "status": "open",
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.question_type == QuestionType.BINARY  # Default value

    def test_parse_question_invalid_status(self):
        """Test parsing with invalid status."""
        # Arrange
        question_data = {
            "id": 123,
            "title": "Invalid status question",
            "description": "Has invalid status",
            "type": "binary",
            "url": "https://example.com/123",
            "close_time": "2024-12-31T23:59:59Z",
            "status": "invalid_status",
        }

        # Act
        result = self.service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.status == QuestionStatus.OPEN  # Default value

    def test_validation_level_minimal(self):
        """Test parsing with minimal validation level."""
        # Arrange
        minimal_service = IngestionService(validation_level=ValidationLevel.MINIMAL)
        question_data = {
            "id": 123,
            "title": "Minimal validation",
            "description": "Very basic question",
            "type": "binary",
            # Missing many optional fields
        }

        # Act
        result = minimal_service.parse_question(question_data)

        # Assert
        assert result is not None
        assert result.metaculus_id == 123
        assert result.title == "Minimal validation"

    def test_ingestion_stats_properties(self):
        """Test IngestionStats properties and calculations."""
        # Arrange
        stats = IngestionStats()
        stats.total_processed = 10
        stats.successful_parsed = 7
        stats.failed_parsing = 3

        # Act & Assert
        assert stats.success_rate == 70.0

        # Test zero division
        empty_stats = IngestionStats()
        assert empty_stats.success_rate == 0.0


class TestValidationLevels:
    """Test suite for different validation levels."""

    def test_strict_validation_requires_all_fields(self):
        """Test that strict validation requires all fields."""
        # Arrange
        service = IngestionService(validation_level=ValidationLevel.STRICT)
        minimal_data = {
            "id": 123,
            "title": "Minimal",
            # Missing many required fields
        }

        # Act & Assert
        with pytest.raises((ValidationError, ParseError)):
            service.parse_question(minimal_data)

    def test_lenient_validation_allows_defaults(self):
        """Test that lenient validation allows defaults."""
        # Arrange
        service = IngestionService(validation_level=ValidationLevel.LENIENT)
        minimal_data = {
            "id": 123,
            "title": "Minimal with some fields",
            "description": "Basic description",
            "type": "binary",
        }

        # Act
        result = service.parse_question(minimal_data)

        # Assert
        assert result is not None
        assert result.metaculus_id == 123

    def test_minimal_validation_very_permissive(self):
        """Test that minimal validation is very permissive."""
        # Arrange
        service = IngestionService(validation_level=ValidationLevel.MINIMAL)
        basic_data = {"id": 123, "title": "Very minimal", "type": "binary"}

        # Act
        result = service.parse_question(basic_data)

        # Assert
        assert result is not None
        assert result.metaculus_id == 123


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def test_ingestion_error_inheritance(self):
        """Test that custom exceptions inherit properly."""
        assert issubclass(ValidationError, IngestionError)
        assert issubclass(ParseError, IngestionError)

    def test_parse_error_on_invalid_data_type(self):
        """Test parse error when data is not a dictionary."""
        # Arrange
        service = IngestionService()

        # Act & Assert
        with pytest.raises((ParseError, IngestionError, TypeError)):
            service.parse_question("not a dictionary")

    def test_validation_error_on_empty_title_strict(self):
        """Test validation error on empty title in strict mode."""
        # Arrange
        service = IngestionService(validation_level=ValidationLevel.STRICT)
        data = {
            "id": 123,
            "title": "",  # Empty title
            "description": "Has description",
            "type": "binary",
            "url": "https://example.com/123",
            "close_time": "2024-12-31T23:59:59Z",
            "status": "open",
        }

        # Act & Assert
        with pytest.raises(ValidationError):
            service.parse_question(data)

    def test_handles_malformed_datetime(self):
        """Test handling of malformed datetime strings."""
        # Arrange
        service = IngestionService(validation_level=ValidationLevel.LENIENT)
        data = {
            "id": 123,
            "title": "Bad datetime",
            "description": "Has bad datetime",
            "type": "binary",
            "url": "https://example.com/123",
            "close_time": "not-a-datetime",
            "status": "open",
        }

        # Act
        result = service.parse_question(data)

        # Assert
        assert result is not None
        # Should use default datetime (2030-12-31)
        assert result.close_time.year == 2030
