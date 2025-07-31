"""Tournament entity for representing tournament structure and rules."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from .question import Question


class ScoringMethod(Enum):
    """Different scoring methods used in tournaments."""
    BRIER_SCORE = "brier_score"
    LOG_SCORE = "log_score"
    QUADRATIC_SCORE = "quadratic_score"
    SPHERICAL_SCORE = "spherical_score"
    RELATIVE_SCORE = "relative_score"


@dataclass
class ScoringRules:
    """Defines how scoring works in a tournament.

    Attributes:
        method: Primary scoring method used
        weight_by_question: Whether to weight questions differently
        bonus_for_early: Whether early submissions get bonus points
        penalty_for_late: Whether late submissions are penalized
        minimum_confidence: Minimum confidence required for submissions
        maximum_submissions: Maximum number of submissions per question
        resolution_bonus: Bonus points for questions that resolve quickly
        metadata: Additional scoring rule parameters
    """
    method: ScoringMethod
    weight_by_question: bool = True
    bonus_for_early: bool = False
    penalty_for_late: bool = False
    minimum_confidence: float = 0.0
    maximum_submissions: int = 1
    resolution_bonus: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate scoring rules."""
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(f"Minimum confidence must be between 0.0 and 1.0, got {self.minimum_confidence}")

        if self.maximum_submissions < 1:
            raise ValueError(f"Maximum submissions must be at least 1, got {self.maximum_submissions}")

        if self.resolution_bonus < 0:
            raise ValueError(f"Resolution bonus cannot be negative, got {self.resolution_bonus}")

        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

    def allows_multiple_submissions(self) -> bool:
        """Check if multiple submissions are allowed."""
        return self.maximum_submissions > 1

    def requires_minimum_confidence(self) -> bool:
        """Check if minimum confidence is required."""
        return self.minimum_confidence > 0.0

    def has_timing_incentives(self) -> bool:
        """Check if there are timing-based incentives."""
        return self.bonus_for_early or self.penalty_for_late


@dataclass
class Tournament:
    """Represents a forecasting tournament with questions and rules.

    Attributes:
        id: Unique identifier for the tournament
        name: Human-readable name of the tournament
        questions: List of questions in this tournament
        scoring_rules: How scoring works in this tournament
        start_date: When the tournament starts
        end_date: When the tournament ends
        current_standings: Current participant standings
        metadata: Additional tournament-specific data
        description: Description of the tournament
        max_participants: Maximum number of participants allowed
        entry_requirements: Requirements to participate
        prize_structure: Information about prizes/rewards
    """
    id: int
    name: str
    questions: List[Question]
    scoring_rules: ScoringRules
    start_date: datetime
    end_date: datetime
    current_standings: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    max_participants: Optional[int] = None
    entry_requirements: Optional[List[str]] = None
    prize_structure: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate tournament data."""
        if self.id <= 0:
            raise ValueError(f"Tournament ID must be positive, got {self.id}")

        if not self.name or not self.name.strip():
            raise ValueError("Tournament name cannot be empty")

        if not isinstance(self.questions, list):
            raise ValueError("Questions must be a list")

        if not isinstance(self.current_standings, dict):
            raise ValueError("Current standings must be a dictionary")

        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")

        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

        if self.max_participants is not None and self.max_participants <= 0:
            raise ValueError(f"Max participants must be positive, got {self.max_participants}")

        # Validate all questions belong to this tournament timeframe
        for question in self.questions:
            if question.deadline > self.end_date:
                raise ValueError(f"Question {question.id} deadline is after tournament end")

    def is_active(self) -> bool:
        """Check if tournament is currently active."""
        now = datetime.utcnow()
        return self.start_date <= now <= self.end_date

    def is_upcoming(self) -> bool:
        """Check if tournament hasn't started yet."""
        return datetime.utcnow() < self.start_date

    def is_finished(self) -> bool:
        """Check if tournament has ended."""
        return datetime.utcnow() > self.end_date

    def time_remaining(self) -> float:
        """Get time remaining in tournament in hours."""
        if self.is_finished():
            return 0.0

        delta = self.end_date - datetime.utcnow()
        return delta.total_seconds() / 3600

    def get_active_questions(self) -> List[Question]:
        """Get questions that are still accepting predictions."""
        now = datetime.utcnow()
        return [q for q in self.questions if q.deadline > now]

    def get_resolved_questions(self) -> List[Question]:
        """Get questions that have passed their deadline."""
        now = datetime.utcnow()
        return [q for q in self.questions if q.deadline <= now]

    def get_high_value_questions(self, weight_threshold: float = 2.0) -> List[Question]:
        """Get questions with high scoring weight."""
        return [q for q in self.questions if q.is_high_value(weight_threshold)]

    def get_questions_by_category(self, category) -> List[Question]:
        """Get questions in a specific category."""
        return [q for q in self.questions if q.category == category]

    def get_questions_by_type(self, question_type) -> List[Question]:
        """Get questions of a specific type."""
        return [q for q in self.questions if q.question_type == question_type]

    def get_urgent_questions(self, hours_threshold: float = 24.0) -> List[Question]:
        """Get questions with approaching deadlines."""
        return [q for q in self.questions if q.is_deadline_approaching(hours_threshold)]

    def get_participant_count(self) -> int:
        """Get number of participants with standings."""
        return len(self.current_standings)

    def get_participant_rank(self, participant_id: str) -> Optional[int]:
        """Get rank of a specific participant."""
        if participant_id not in self.current_standings:
            return None

        # Sort by score descending
        sorted_standings = sorted(
            self.current_standings.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for rank, (pid, _) in enumerate(sorted_standings, 1):
            if pid == participant_id:
                return rank

        return None

    def get_top_participants(self, n: int = 10) -> List[tuple[str, float]]:
        """Get top N participants by score."""
        sorted_standings = sorted(
            self.current_standings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_standings[:n]

    def get_participant_score(self, participant_id: str) -> Optional[float]:
        """Get score for a specific participant."""
        return self.current_standings.get(participant_id)

    def update_standings(self, new_standings: Dict[str, float]) -> "Tournament":
        """Create updated tournament with new standings."""
        return Tournament(
            id=self.id,
            name=self.name,
            questions=self.questions,
            scoring_rules=self.scoring_rules,
            start_date=self.start_date,
            end_date=self.end_date,
            current_standings=new_standings,
            metadata=self.metadata,
            description=self.description,
            max_participants=self.max_participants,
            entry_requirements=self.entry_requirements,
            prize_structure=self.prize_structure
        )

    def add_question(self, question: Question) -> "Tournament":
        """Create updated tournament with additional question."""
        if question.deadline > self.end_date:
            raise ValueError(f"Question deadline {question.deadline} is after tournament end {self.end_date}")

        new_questions = self.questions + [question]

        return Tournament(
            id=self.id,
            name=self.name,
            questions=new_questions,
            scoring_rules=self.scoring_rules,
            start_date=self.start_date,
            end_date=self.end_date,
            current_standings=self.current_standings,
            metadata=self.metadata,
            description=self.description,
            max_participants=self.max_participants,
            entry_requirements=self.entry_requirements,
            prize_structure=self.prize_structure
        )

    def get_tournament_stats(self) -> Dict[str, Any]:
        """Get comprehensive tournament statistics."""
        active_questions = self.get_active_questions()
        resolved_questions = self.get_resolved_questions()

        return {
            "total_questions": len(self.questions),
            "active_questions": len(active_questions),
            "resolved_questions": len(resolved_questions),
            "participants": self.get_participant_count(),
            "time_remaining_hours": self.time_remaining(),
            "is_active": self.is_active(),
            "high_value_questions": len(self.get_high_value_questions()),
            "urgent_questions": len(self.get_urgent_questions()),
            "question_types": {
                qtype.value: len(self.get_questions_by_type(qtype))
                for qtype in set(q.question_type for q in self.questions)
            },
            "question_categories": {
                cat.value: len(self.get_questions_by_category(cat))
                for cat in set(q.category for q in self.questions)
            }
        }

    def to_summary(self) -> str:
        """Create a brief summary of the tournament."""
        status = "active" if self.is_active() else "upcoming" if self.is_upcoming() else "finished"
        return f"Tournament {self.id}: {self.name} ({status}) - {len(self.questions)} questions, {self.get_participant_count()} participants"
