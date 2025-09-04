"""Tournament strategy value objects for competitive forecasting."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class QuestionCategory(Enum):
    """Categories of forecasting questions for specialized strategies."""

    TECHNOLOGY = "technology"
    ECONOMICS = "economics"
    POLITICS = "politics"
    HEALTH = "health"
    CLIMATE = "climate"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    GEOPOLITICS = "geopolitics"
    BUSINESS = "business"
    SOCIAL = "social"
    OTHER = "other"


class TournamentPhase(Enum):
    """Phases of tournament for strategy adaptation."""

    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"
    FINAL = "final"


class RiskProfile(Enum):
    """Risk profiles for tournament strategy."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class QuestionPriority:
    """Priority assessment for tournament questions."""

    question_id: UUID
    category: QuestionCategory
    confidence_level: float
    scoring_potential: float
    resource_allocation: float
    deadline_urgency: float
    competitive_advantage: float

    def __post_init__(self):
        """Validate priority values."""
        for field_name, value in [
            ("confidence_level", self.confidence_level),
            ("scoring_potential", self.scoring_potential),
            ("resource_allocation", self.resource_allocation),
            ("deadline_urgency", self.deadline_urgency),
            ("competitive_advantage", self.competitive_advantage),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1, got {value}")

    @classmethod
    def create(
        cls,
        question_id: UUID,
        category: QuestionCategory,
        confidence_level: float,
        scoring_potential: float,
        resource_allocation: float = 0.5,
        deadline_urgency: float = 0.5,
        competitive_advantage: float = 0.5,
    ) -> "QuestionPriority":
        """Factory method to create question priority."""
        return cls(
            question_id=question_id,
            category=category,
            confidence_level=confidence_level,
            scoring_potential=scoring_potential,
            resource_allocation=resource_allocation,
            deadline_urgency=deadline_urgency,
            competitive_advantage=competitive_advantage,
        )

    def get_overall_priority_score(self) -> float:
        """Calculate overall priority score for resource allocation."""
        weights = {
            "scoring_potential": 0.3,
            "confidence_level": 0.25,
            "competitive_advantage": 0.2,
            "deadline_urgency": 0.15,
            "resource_allocation": 0.1,
        }

        return (
            self.scoring_potential * weights["scoring_potential"]
            + self.confidence_level * weights["confidence_level"]
            + self.competitive_advantage * weights["competitive_advantage"]
            + self.deadline_urgency * weights["deadline_urgency"]
            + self.resource_allocation * weights["resource_allocation"]
        )


@dataclass(frozen=True)
class TournamentStrategy:
    """Tournament-specific strategy configuration."""

    id: UUID
    tournament_id: str
    phase: TournamentPhase
    risk_profile: RiskProfile
    category_specializations: Dict[QuestionCategory, float]
    resource_allocation_weights: Dict[str, float]
    confidence_thresholds: Dict[str, float]
    submission_timing_strategy: str
    competitive_positioning: str
    created_at: datetime

    @classmethod
    def create_default(
        cls,
        tournament_id: str,
        phase: TournamentPhase = TournamentPhase.EARLY,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
    ) -> "TournamentStrategy":
        """Create default tournament strategy."""
        return cls(
            id=uuid4(),
            tournament_id=tournament_id,
            phase=phase,
            risk_profile=risk_profile,
            category_specializations={
                QuestionCategory.TECHNOLOGY: 0.8,
                QuestionCategory.ECONOMICS: 0.7,
                QuestionCategory.POLITICS: 0.6,
                QuestionCategory.SCIENCE: 0.8,
                QuestionCategory.HEALTH: 0.7,
                QuestionCategory.CLIMATE: 0.6,
                QuestionCategory.OTHER: 0.5,
            },
            resource_allocation_weights={
                "research_depth": 0.4,
                "ensemble_diversity": 0.3,
                "validation_rigor": 0.2,
                "speed_optimization": 0.1,
            },
            confidence_thresholds={
                "minimum_submission": 0.6,
                "high_confidence": 0.8,
                "abstention": 0.4,
                "additional_research": 0.5,
            },
            submission_timing_strategy="optimal_window",
            competitive_positioning="balanced",
            created_at=datetime.now(timezone.utc),
        )

    @classmethod
    def create_aggressive(
        cls, tournament_id: str, phase: TournamentPhase = TournamentPhase.LATE
    ) -> "TournamentStrategy":
        """Create aggressive tournament strategy for late-phase competition."""
        return cls(
            id=uuid4(),
            tournament_id=tournament_id,
            phase=phase,
            risk_profile=RiskProfile.AGGRESSIVE,
            category_specializations={
                QuestionCategory.TECHNOLOGY: 0.9,
                QuestionCategory.ECONOMICS: 0.8,
                QuestionCategory.POLITICS: 0.7,
                QuestionCategory.SCIENCE: 0.9,
                QuestionCategory.HEALTH: 0.8,
                QuestionCategory.CLIMATE: 0.7,
                QuestionCategory.OTHER: 0.6,
            },
            resource_allocation_weights={
                "research_depth": 0.5,
                "ensemble_diversity": 0.3,
                "validation_rigor": 0.1,
                "speed_optimization": 0.1,
            },
            confidence_thresholds={
                "minimum_submission": 0.5,
                "high_confidence": 0.75,
                "abstention": 0.3,
                "additional_research": 0.4,
            },
            submission_timing_strategy="early_advantage",
            competitive_positioning="aggressive",
            created_at=datetime.now(timezone.utc),
        )

    @classmethod
    def create_conservative(
        cls, tournament_id: str, phase: TournamentPhase = TournamentPhase.EARLY
    ) -> "TournamentStrategy":
        """Create conservative tournament strategy for risk management."""
        return cls(
            id=uuid4(),
            tournament_id=tournament_id,
            phase=phase,
            risk_profile=RiskProfile.CONSERVATIVE,
            category_specializations={
                QuestionCategory.TECHNOLOGY: 0.7,
                QuestionCategory.ECONOMICS: 0.8,
                QuestionCategory.POLITICS: 0.5,
                QuestionCategory.SCIENCE: 0.8,
                QuestionCategory.HEALTH: 0.7,
                QuestionCategory.CLIMATE: 0.6,
                QuestionCategory.OTHER: 0.4,
            },
            resource_allocation_weights={
                "research_depth": 0.3,
                "ensemble_diversity": 0.2,
                "validation_rigor": 0.4,
                "speed_optimization": 0.1,
            },
            confidence_thresholds={
                "minimum_submission": 0.7,
                "high_confidence": 0.85,
                "abstention": 0.5,
                "additional_research": 0.6,
            },
            submission_timing_strategy="late_validation",
            competitive_positioning="conservative",
            created_at=datetime.now(timezone.utc),
        )

    def get_category_confidence_threshold(self, category: QuestionCategory) -> float:
        """Get confidence threshold adjusted for category specialization."""
        base_threshold = self.confidence_thresholds["minimum_submission"]
        specialization = self.category_specializations.get(category, 0.5)

        # Lower threshold for specialized categories
        adjustment = (specialization - 0.5) * 0.2
        return max(0.1, min(0.9, base_threshold - adjustment))

    def should_prioritize_question(
        self, category: QuestionCategory, confidence: float, scoring_potential: float
    ) -> bool:
        """Determine if a question should be prioritized based on strategy."""
        category_threshold = self.get_category_confidence_threshold(category)

        if confidence < category_threshold:
            return False

        # Consider scoring potential and risk profile
        if self.risk_profile == RiskProfile.AGGRESSIVE:
            return scoring_potential > 0.6
        elif self.risk_profile == RiskProfile.CONSERVATIVE:
            return scoring_potential > 0.7 and confidence > 0.75
        else:  # MODERATE or ADAPTIVE
            return scoring_potential > 0.65 and confidence > category_threshold


@dataclass(frozen=True)
class CompetitiveIntelligence:
    """Intelligence about tournament competition and market dynamics."""

    tournament_id: str
    current_standings: Dict[str, float]
    market_inefficiencies: List[str]
    competitor_patterns: Dict[str, Any]
    scoring_trends: Dict[str, float]
    question_difficulty_distribution: Dict[QuestionCategory, float]
    timestamp: datetime

    @classmethod
    def create_empty(cls, tournament_id: str) -> "CompetitiveIntelligence":
        """Create empty competitive intelligence for initialization."""
        return cls(
            tournament_id=tournament_id,
            current_standings={},
            market_inefficiencies=[],
            competitor_patterns={},
            scoring_trends={},
            question_difficulty_distribution={},
            timestamp=datetime.now(timezone.utc),
        )

    def get_competitive_advantage_score(self, category: QuestionCategory) -> float:
        """Calculate competitive advantage score for a question category."""
        difficulty = self.question_difficulty_distribution.get(category, 0.5)

        # Higher advantage in categories where others struggle (high difficulty)
        # but we have specialization
        base_advantage = difficulty * 0.5

        # Add market inefficiency bonus
        inefficiency_bonus = (
            0.1 if str(category.value) in self.market_inefficiencies else 0.0
        )

        return min(1.0, base_advantage + inefficiency_bonus)
