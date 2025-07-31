"""Question entity for representing tournament questions."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class QuestionType(Enum):
    """Types of questions in tournaments."""
    BINARY = "binary"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"
    DATE = "date"
    CONDITIONAL = "conditional"


class QuestionStatus(Enum):
    """Status of questions in tournaments."""
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class QuestionCategory(Enum):
    """Categories of questions for specialized strategies."""
    AI_DEVELOPMENT = "ai_development"
    TECHNOLOGY = "technology"
    POLITICS = "politics"
    ECONOMICS = "economics"
    SCIENCE = "science"
    SOCIAL = "social"
    GEOPOLITICS = "geopolitics"
    CLIMATE = "climate"
    HEALTH = "health"
    OTHER = "other"


@dataclass
class Question:
    """Represents a tournament question with all relevant metadata.

    Attributes:
        id: Unique identifier for the question
        text: The question text
        question_type: Type of question (binary, numeric, etc.)
        category: Category for strategy selection
        deadline: When predictions must be submitted
        background: Background information for the question
        resolution_criteria: How the question will be resolved
        scoring_weight: Relative importance in tournament scoring
        metadata: Additional question-specific data
        created_at: When the question was created
        min_value: Minimum value for numeric questions
        max_value: Maximum value for numeric questions
        choices: Available choices for multiple choice questions
    """
    id: int
    text: str
    question_type: QuestionType
    category: QuestionCategory
    deadline: datetime
    background: str
    resolution_criteria: str
    scoring_weight: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate question data and set defaults."""
        if self.id <= 0:
            raise ValueError(f"Question ID must be positive, got {self.id}")

        if not self.text or not self.text.strip():
            raise ValueError("Question text cannot be empty")

        if not self.background or not self.background.strip():
            raise ValueError("Question background cannot be empty")

        if not self.resolution_criteria or not self.resolution_criteria.strip():
            raise ValueError("Resolution criteria cannot be empty")

        if self.scoring_weight <= 0:
            raise ValueError(f"Scoring weight must be positive, got {self.scoring_weight}")

        if self.deadline <= datetime.utcnow():
            raise ValueError("Question deadline must be in the future")

        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.utcnow())

        # Validate type-specific constraints
        if self.question_type == QuestionType.NUMERIC:
            if self.min_value is None or self.max_value is None:
                raise ValueError("Numeric questions must have min_value and max_value")
            if self.min_value >= self.max_value:
                raise ValueError("min_value must be less than max_value")

        if self.question_type == QuestionType.MULTIPLE_CHOICE:
            if not self.choices or len(self.choices) < 2:
                raise ValueError("Multiple choice questions must have at least 2 choices")

    def is_binary(self) -> bool:
        """Check if this is a binary question."""
        return self.question_type == QuestionType.BINARY

    def is_numeric(self) -> bool:
        """Check if this is a numeric question."""
        return self.question_type == QuestionType.NUMERIC

    def is_multiple_choice(self) -> bool:
        """Check if this is a multiple choice question."""
        return self.question_type == QuestionType.MULTIPLE_CHOICE

    def is_date(self) -> bool:
        """Check if this is a date question."""
        return self.question_type == QuestionType.DATE

    def is_conditional(self) -> bool:
        """Check if this is a conditional question."""
        return self.question_type == QuestionType.CONDITIONAL

    def time_until_deadline(self) -> float:
        """Get time until deadline in hours."""
        delta = self.deadline - datetime.utcnow()
        return delta.total_seconds() / 3600

    def is_deadline_approaching(self, hours_threshold: float = 24.0) -> bool:
        """Check if deadline is approaching within threshold."""
        return self.time_until_deadline() <= hours_threshold

    def is_high_value(self, weight_threshold: float = 2.0) -> bool:
        """Check if this is a high-value question based on scoring weight."""
        return self.scoring_weight >= weight_threshold

    def requires_specialized_knowledge(self) -> bool:
        """Check if question requires specialized domain knowledge."""
        specialized_categories = {
            QuestionCategory.AI_DEVELOPMENT,
            QuestionCategory.SCIENCE,
            QuestionCategory.TECHNOLOGY,
            QuestionCategory.CLIMATE,
            QuestionCategory.HEALTH
        }
        return self.category in specialized_categories

    def get_complexity_score(self) -> float:
        """Calculate complexity score based on question characteristics."""
        base_score = 1.0

        # Add complexity for question type
        type_complexity = {
            QuestionType.BINARY: 1.0,
            QuestionType.MULTIPLE_CHOICE: 1.2,
            QuestionType.NUMERIC: 1.5,
            QuestionType.DATE: 1.3,
            QuestionType.CONDITIONAL: 2.0
        }
        base_score *= type_complexity.get(self.question_type, 1.0)

        # Add complexity for specialized categories
        if self.requires_specialized_knowledge():
            base_score *= 1.3

        # Add complexity based on text length (proxy for complexity)
        text_length_factor = min(len(self.text) / 500, 2.0)  # Cap at 2x
        base_score *= (1.0 + text_length_factor * 0.2)

        return base_score

    def to_summary(self) -> str:
        """Create a brief summary of the question."""
        deadline_str = self.deadline.strftime("%Y-%m-%d %H:%M")
        return f"Q{self.id}: {self.question_type.value} ({self.category.value}) - deadline: {deadline_str}"
