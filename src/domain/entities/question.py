"""Question domain entity."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4


class QuestionType(Enum):
    """Types of questions that can be forecasted."""
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    DATE = "date"


class QuestionStatus(Enum):
    """Status of questions."""
    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


@dataclass
class Question:
    """
    Domain entity representing a forecasting question.
    
    This is the core entity that encapsulates all the information
    about a question that needs to be forecasted.
    """
    id: UUID
    metaculus_id: int
    title: str
    description: str
    question_type: QuestionType
    status: QuestionStatus
    url: str
    close_time: datetime
    resolve_time: Optional[datetime]
    categories: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    resolution_criteria: Optional[str] = None
    
    # Question-specific fields
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate question data after initialization."""
        if self.question_type == QuestionType.MULTIPLE_CHOICE and not self.choices:
            raise ValueError("Multiple choice questions must have choices")
        
        if self.question_type == QuestionType.NUMERIC and (
            self.min_value is None or self.max_value is None
        ):
            raise ValueError("Numeric questions must have min and max values")
    
    @classmethod
    def create_new(
        cls,
        metaculus_id: int,
        title: str,
        description: str,
        question_type: QuestionType,
        url: str,
        close_time: datetime,
        categories: List[str],
        **kwargs
    ) -> "Question":
        """Factory method to create a new question."""
        now = datetime.now(timezone.utc)
        return cls(
            id=uuid4(),
            metaculus_id=metaculus_id,
            title=title,
            description=description,
            question_type=question_type,
            status=kwargs.get("status", QuestionStatus.OPEN),
            url=url,
            close_time=close_time,
            resolve_time=kwargs.get("resolve_time"),
            categories=categories,
            metadata=kwargs.get("metadata", {}),
            created_at=now,
            updated_at=now,
            min_value=kwargs.get("min_value"),
            max_value=kwargs.get("max_value"),
            choices=kwargs.get("choices"),
        )
    
    def is_open(self) -> bool:
        """Check if the question is still open for forecasting."""
        now = datetime.now(timezone.utc)
        close_time = self.close_time
        
        # Handle naive datetime comparison
        if close_time.tzinfo is None:
            now = now.replace(tzinfo=None)
        
        return now < close_time
    
    def is_resolved(self) -> bool:
        """Check if the question has been resolved."""
        if self.resolve_time is None:
            return False
            
        now = datetime.now(timezone.utc)
        resolve_time = self.resolve_time
        
        # Handle naive datetime comparison
        if resolve_time.tzinfo is None:
            now = now.replace(tzinfo=None)
            
        return now > resolve_time
    
    def days_until_close(self) -> int:
        """Calculate days until the question closes."""
        if not self.is_open():
            return 0
        return (self.close_time - datetime.now(timezone.utc)).days
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update question metadata."""
        self.metadata[key] = value
        self.updated_at = datetime.now(timezone.utc)
