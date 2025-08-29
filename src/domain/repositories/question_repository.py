"""Question repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.question import Question, QuestionType


class QuestionRepository(ABC):
    """Repository interface for question persistence."""

    @abstractmethod
    async def save(self, question: Question) -> None:
        """Save a question to the repository."""
        pass

    @abstractmethod
    async def find_by_id(self, question_id: UUID) -> Optional[Question]:
        """Find a question by its ID."""
        pass

    @abstractmethod
    async def find_by_metaculus_id(self, metaculus_id: int) -> Optional[Question]:
        """Find a question by its Metaculus ID."""
        pass

    @abstractmethod
    async def find_open_questions(self) -> List[Question]:
        """Find all open questions."""
        pass

    @abstractmethod
    async def find_by_type(self, question_type: QuestionType) -> List[Question]:
        """Find questions by type."""
        pass

    @abstractmethod
    async def find_by_categories(self, categories: List[str]) -> List[Question]:
        """Find questions by categories."""
        pass

    @abstractmethod
    async def list_all(self, limit: Optional[int] = None) -> List[Question]:
        """List all questions with optional limit."""
        pass

    @abstractmethod
    async def delete(self, question_id: UUID) -> None:
        """Delete a question."""
        pass
