"""
Application service for ingesting raw question data into domain entities.

This service handles parsing JSON question data from the Metaculus API
and converting it into Question domain objects with validation.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from src.domain.entities.question import Question, QuestionStatus, QuestionType

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels for question parsing."""

    STRICT = "strict"  # All required fields must be present and valid
    LENIENT = "lenient"  # Some fields can be missing, use defaults
    MINIMAL = "minimal"  # Only essential fields required


@dataclass
class IngestionStats:
    """Statistics from the ingestion process."""

    total_processed: int = 0
    successful_parsed: int = 0
    failed_parsing: int = 0
    validation_errors: int = 0
    type_conversion_errors: int = 0
    missing_required_fields: int = 0
    processing_time_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.successful_parsed / self.total_processed) * 100


class IngestionError(Exception):
    """Base exception for ingestion-related errors."""

    pass


class ValidationError(IngestionError):
    """Exception raised when question validation fails."""

    pass


class ParseError(IngestionError):
    """Exception raised when JSON parsing fails."""

    pass


class IngestionService:
    """
    Service for ingesting raw question data and converting to domain entities.

    Handles JSON parsing, validation, and conversion to Question objects
    with configurable validation levels and comprehensive error handling.
    """

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.LENIENT):
        """
        Initialize the ingestion service.

        Args:
            validation_level: How strict to be with validation
        """
        self.validation_level = validation_level
        self.stats = IngestionStats()

    def parse_questions(
        self, raw_data: List[Dict[str, Any]]
    ) -> Tuple[List[Question], IngestionStats]:
        """
        Parse a list of raw question dictionaries into Question objects.

        Args:
            raw_data: List of question dictionaries from API

        Returns:
            Tuple of (parsed questions, ingestion statistics)
        """
        start_time = datetime.now()
        questions = []
        errors = []

        self.stats = IngestionStats()
        self.stats.total_processed = len(raw_data)

        for raw_question in raw_data:
            try:
                question = self.parse_question(raw_question)
                questions.append(question)
                self.stats.successful_parsed += 1

            except ValidationError as e:
                logger.warning(
                    f"Validation error for question {raw_question.get('id', 'unknown')}: {e}"
                )
                self.stats.validation_errors += 1
                self.stats.failed_parsing += 1
                errors.append(str(e))

            except ParseError as e:
                logger.warning(
                    f"Parse error for question {raw_question.get('id', 'unknown')}: {e}"
                )
                self.stats.type_conversion_errors += 1
                self.stats.failed_parsing += 1
                errors.append(str(e))

            except Exception as e:
                logger.error(
                    f"Unexpected error parsing question {raw_question.get('id', 'unknown')}: {e}"
                )
                self.stats.failed_parsing += 1
                errors.append(str(e))

        # Calculate processing time
        end_time = datetime.now()
        self.stats.processing_time_seconds = (end_time - start_time).total_seconds()

        logger.info(
            f"Ingestion completed: {self.stats.successful_parsed}/{self.stats.total_processed} "
            f"questions parsed successfully ({self.stats.success_rate:.1f}%)"
        )

        if errors:
            logger.warning(f"Encountered {len(errors)} errors during ingestion")

        return questions, self.stats

    def parse_question(self, raw_data: Dict[str, Any]) -> Question:
        """
        Parse a single raw question dictionary into a Question object.

        Args:
            raw_data: Raw question data from API

        Returns:
            Question domain object

        Raises:
            ValidationError: If validation fails
            ParseError: If required data is missing or invalid
        """
        try:
            # Extract and validate required fields
            question_id = self._extract_id(raw_data)
            title = self._extract_title(raw_data)
            description = self._extract_description(raw_data)
            question_type = self._extract_question_type(raw_data)
            url = self._extract_url(raw_data)
            close_time = self._extract_close_time(raw_data)

            # Extract optional fields
            resolve_time = self._extract_resolve_time(raw_data)
            categories = self._extract_categories(raw_data)
            metadata = self._extract_metadata(raw_data)

            # Extract type-specific fields
            min_value, max_value = self._extract_numeric_bounds(raw_data, question_type)
            choices = self._extract_choices(raw_data, question_type)

            # Create Question object
            now = datetime.now(timezone.utc)
            question = Question(
                id=uuid4(),
                metaculus_id=question_id,
                title=title,
                description=description,
                question_type=question_type,
                status=self._extract_status(raw_data),
                url=url,
                close_time=close_time,
                resolve_time=resolve_time,
                categories=categories,
                metadata=metadata,
                created_at=now,
                updated_at=now,
                min_value=min_value,
                max_value=max_value,
                choices=choices,
            )

            return question

        except ValidationError:
            # Let ValidationErrors propagate as-is for test verification
            raise
        except KeyError as e:
            raise ParseError(f"Missing required field: {e}")
        except (ValueError, TypeError) as e:
            raise ParseError(f"Invalid data format: {e}")
        except Exception as e:
            raise IngestionError(f"Unexpected error during parsing: {e}")

    def _extract_id(self, data: Dict[str, Any]) -> int:
        """Extract and validate question ID."""
        if "id" not in data:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError("Question ID is required")
            return -1  # Default for missing ID

        try:
            return int(data["id"])
        except (ValueError, TypeError):
            raise ParseError(f"Invalid question ID: {data['id']}")

    def _extract_title(self, data: Dict[str, Any]) -> str:
        """Extract and validate question title."""
        if "title" not in data:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError("Question title is required")
            return "Untitled Question"

        title = str(data["title"]).strip()
        if not title and self.validation_level in [
            ValidationLevel.STRICT,
            ValidationLevel.LENIENT,
        ]:
            raise ValidationError("Question title cannot be empty")

        return title or "Untitled Question"

    def _extract_description(self, data: Dict[str, Any]) -> str:
        """Extract and validate question description."""
        if "description" not in data:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError("Question description is required")
            return "No description provided"

        description = str(data["description"]).strip()
        return description or "No description provided"

    def _extract_question_type(self, data: Dict[str, Any]) -> QuestionType:
        """Extract and validate question type."""
        type_mappings = {
            "binary": QuestionType.BINARY,
            "multiple_choice": QuestionType.MULTIPLE_CHOICE,
            "numeric": QuestionType.NUMERIC,
            "date": QuestionType.DATE,
            # Additional mappings for API variations
            "multiple-choice": QuestionType.MULTIPLE_CHOICE,
            "continuous": QuestionType.NUMERIC,
        }

        raw_type = data.get("type") or data.get("question_type")
        if not raw_type:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError("Question type is required")
            return QuestionType.BINARY  # Default

        raw_type = str(raw_type).lower().strip()
        if raw_type not in type_mappings:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError(f"Unknown question type: {raw_type}")
            return QuestionType.BINARY  # Default for unknown types

        return type_mappings[raw_type]

    def _extract_status(self, data: Dict[str, Any]) -> QuestionStatus:
        """Extract and validate question status."""
        status_mappings = {
            "open": QuestionStatus.OPEN,
            "closed": QuestionStatus.CLOSED,
            "resolved": QuestionStatus.RESOLVED,
            "cancelled": QuestionStatus.CANCELLED,
        }

        raw_status = data.get("status")
        if not raw_status:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError("Question status is required")
            return QuestionStatus.OPEN  # Default

        raw_status = str(raw_status).lower().strip()
        if raw_status not in status_mappings:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError(f"Unknown question status: {raw_status}")
            return QuestionStatus.OPEN  # Default for unknown status

        return status_mappings[raw_status]

    def _extract_url(self, data: Dict[str, Any]) -> str:
        """Extract and validate question URL."""
        if "url" not in data:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError("Question URL is required")
            return f"https://metaculus.com/questions/{data.get('id', 'unknown')}/"

        url = str(data["url"]).strip()
        if not url.startswith(("http://", "https://")):
            if self.validation_level in [
                ValidationLevel.STRICT,
                ValidationLevel.LENIENT,
            ]:
                raise ValidationError(f"Invalid URL format: {url}")

        return url

    def _extract_close_time(self, data: Dict[str, Any]) -> datetime:
        """Extract and validate question close time."""
        close_time_fields = ["close_time", "scheduled_close_time", "closes_at"]
        close_time_str = None

        for field in close_time_fields:
            if field in data:
                close_time_str = data[field]
                break

        if not close_time_str:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError("Question close time is required")
            # Default to far future
            return datetime(2030, 12, 31, tzinfo=timezone.utc)

        try:
            # Handle various datetime formats
            if isinstance(close_time_str, datetime):
                return (
                    close_time_str.replace(tzinfo=timezone.utc)
                    if close_time_str.tzinfo is None
                    else close_time_str
                )

            # Parse ISO format strings
            if "T" in str(close_time_str):
                dt = datetime.fromisoformat(str(close_time_str).replace("Z", "+00:00"))
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

            # Fallback parsing
            dt = datetime.strptime(str(close_time_str), "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)

        except (ValueError, TypeError) as e:
            if self.validation_level == ValidationLevel.STRICT:
                raise ParseError(f"Invalid close time format: {close_time_str}")
            return datetime(2030, 12, 31, tzinfo=timezone.utc)

    def _extract_resolve_time(self, data: Dict[str, Any]) -> Optional[datetime]:
        """Extract question resolve time if available."""
        resolve_fields = ["resolve_time", "resolved_at", "resolution_time"]

        for field in resolve_fields:
            if field in data and data[field]:
                try:
                    resolve_time = data[field]
                    if isinstance(resolve_time, datetime):
                        return (
                            resolve_time.replace(tzinfo=timezone.utc)
                            if resolve_time.tzinfo is None
                            else resolve_time
                        )

                    if "T" in str(resolve_time):
                        dt = datetime.fromisoformat(
                            str(resolve_time).replace("Z", "+00:00")
                        )
                        return (
                            dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
                        )

                except (ValueError, TypeError):
                    continue

        return None

    def _extract_categories(self, data: Dict[str, Any]) -> List[str]:
        """Extract question categories."""
        categories = []

        # Try different field names
        for field in ["categories", "category", "tags"]:
            if field in data:
                raw_categories = data[field]
                if isinstance(raw_categories, list):
                    categories.extend(
                        [str(cat).strip() for cat in raw_categories if cat]
                    )
                elif isinstance(raw_categories, str) and raw_categories.strip():
                    categories.append(raw_categories.strip())

        return list(set(categories))  # Remove duplicates

    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional metadata."""
        metadata = {}

        # Extract common metadata fields
        metadata_fields = [
            "community_prediction",
            "num_forecasters",
            "status",
            "resolution",
            "is_resolved",
            "created_time",
            "created_at",
            "updated_at",
        ]

        for field in metadata_fields:
            if field in data:
                metadata[field] = data[field]

        return metadata

    def _extract_numeric_bounds(
        self, data: Dict[str, Any], question_type: QuestionType
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract numeric bounds for numeric questions."""
        if question_type != QuestionType.NUMERIC:
            return None, None

        # Use is not None to handle 0 values correctly
        min_value = data.get("min_value")
        if min_value is None:
            min_value = data.get("min")
        if min_value is None:
            min_value = data.get("lower_bound")
        if min_value is None:
            # Check nested possibilities structure
            possibilities = data.get("possibilities", {})
            min_value = possibilities.get("min")

        max_value = data.get("max_value")
        if max_value is None:
            max_value = data.get("max")
        if max_value is None:
            max_value = data.get("upper_bound")
        if max_value is None:
            # Check nested possibilities structure
            possibilities = data.get("possibilities", {})
            max_value = possibilities.get("max")

        try:
            min_val = float(min_value) if min_value is not None else None
            max_val = float(max_value) if max_value is not None else None

            if min_val is not None and max_val is not None and min_val >= max_val:
                if self.validation_level == ValidationLevel.STRICT:
                    raise ValidationError(
                        f"Invalid bounds: min ({min_val}) >= max ({max_val})"
                    )

            return min_val, max_val

        except (ValueError, TypeError):
            if self.validation_level == ValidationLevel.STRICT:
                raise ParseError("Invalid numeric bounds format")
            return None, None

    def _extract_choices(
        self, data: Dict[str, Any], question_type: QuestionType
    ) -> Optional[List[str]]:
        """Extract choices for multiple choice questions."""
        if question_type != QuestionType.MULTIPLE_CHOICE:
            return None

        choices_raw = (
            data.get("choices")
            or data.get("options")
            or data.get("possibilities", {}).get("choices")
        )

        if not choices_raw:
            if self.validation_level == ValidationLevel.STRICT:
                raise ValidationError("Multiple choice questions must have choices")
            return None

        if isinstance(choices_raw, list):
            choices = [str(choice).strip() for choice in choices_raw if choice]
            if len(choices) < 2 and self.validation_level in [
                ValidationLevel.STRICT,
                ValidationLevel.LENIENT,
            ]:
                raise ValidationError(
                    "Multiple choice questions must have at least 2 choices"
                )
            return choices

        if self.validation_level == ValidationLevel.STRICT:
            raise ParseError("Invalid choices format")
        return None

    async def convert_question_data(self, question_data: Dict[str, Any]) -> Question:
        """
        Convert raw question data to a Question domain object (async wrapper).

        This method provides an async interface to the synchronous parse_question method,
        as required by the forecasting pipeline integration tests.

        Args:
            question_data: Raw question data from API

        Returns:
            Question domain object

        Raises:
            ValidationError: If validation fails
            ParseError: If required data is missing or invalid
        """
        return self.parse_question(question_data)
