"""Question domain entity."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from ..value_objects.tournament_strategy import QuestionCategory, QuestionPriority


class QuestionType(Enum):
    """Types of questions that can be forecasted."""
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    CONTINUOUS = "continuous"
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

    # Tournament-specific enhancements
    tournament_id: Optional[str] = None
    question_category: Optional[QuestionCategory] = None
    difficulty_score: Optional[float] = None
    scoring_weight: Optional[float] = None
    priority: Optional[QuestionPriority] = None

    # Advanced tournament capabilities
    market_inefficiency_score: Optional[float] = None
    competitive_advantage_potential: Optional[float] = None
    research_complexity_score: Optional[float] = None
    historical_performance_data: Optional[Dict[str, Any]] = None
    similar_questions_analysis: Optional[List[Dict[str, Any]]] = None

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

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        question_type: QuestionType,
        resolution_criteria: Optional[str] = None,
        close_time: Optional[datetime] = None,
        resolve_time: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "Question":
        """Factory method to create a question from Metaculus API data."""
        if metadata is None:
            metadata = {}

        # Extract required fields from metadata
        metaculus_id = metadata.get('metaculus_id', 0)
        url = metadata.get('url', '')
        category = metadata.get('category', '')
        categories = [category] if category else []

        # Handle close_time default
        if close_time is None:
            close_time = datetime.now(timezone.utc)

        # Handle created_at default
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            id=uuid4(),
            metaculus_id=metaculus_id,
            title=title,
            description=description,
            question_type=question_type,
            status=QuestionStatus.OPEN,
            url=url,
            close_time=close_time,
            resolve_time=resolve_time,
            categories=categories,
            metadata=metadata,
            created_at=created_at,
            updated_at=created_at,
            resolution_criteria=resolution_criteria,
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

    def categorize_question(self) -> QuestionCategory:
        """Automatically categorize question based on content."""
        if self.question_category:
            return self.question_category

        title_lower = self.title.lower()
        description_lower = self.description.lower()
        combined_text = f"{title_lower} {description_lower}"

        # Technology keywords
        if any(keyword in combined_text for keyword in [
            "ai", "artificial intelligence", "technology", "software", "tech",
            "algorithm", "computer", "digital", "internet", "blockchain"
        ]):
            return QuestionCategory.TECHNOLOGY

        # Economics keywords
        elif any(keyword in combined_text for keyword in [
            "economy", "gdp", "market", "finance", "economic", "inflation",
            "recession", "stock", "price", "trade", "currency"
        ]):
            return QuestionCategory.ECONOMICS

        # Politics keywords
        elif any(keyword in combined_text for keyword in [
            "election", "political", "government", "policy", "president",
            "congress", "vote", "democracy", "republican", "democrat"
        ]):
            return QuestionCategory.POLITICS

        # Health keywords
        elif any(keyword in combined_text for keyword in [
            "health", "medical", "disease", "pandemic", "vaccine", "hospital",
            "medicine", "treatment", "covid", "virus", "drug"
        ]):
            return QuestionCategory.HEALTH

        # Climate keywords
        elif any(keyword in combined_text for keyword in [
            "climate", "environment", "carbon", "emission", "temperature",
            "global warming", "renewable", "energy", "pollution", "green"
        ]):
            return QuestionCategory.CLIMATE

        # Science keywords
        elif any(keyword in combined_text for keyword in [
            "science", "research", "study", "experiment", "discovery",
            "physics", "chemistry", "biology", "space", "nasa"
        ]):
            return QuestionCategory.SCIENCE

        # Geopolitics keywords
        elif any(keyword in combined_text for keyword in [
            "war", "conflict", "international", "country", "nation",
            "diplomacy", "treaty", "sanctions", "military", "peace"
        ]):
            return QuestionCategory.GEOPOLITICS

        # Business keywords
        elif any(keyword in combined_text for keyword in [
            "business", "company", "corporation", "startup", "revenue",
            "profit", "merger", "acquisition", "ipo", "ceo"
        ]):
            return QuestionCategory.BUSINESS

        else:
            return QuestionCategory.OTHER

    def calculate_difficulty_score(self) -> float:
        """Calculate difficulty score based on question characteristics."""
        if self.difficulty_score is not None:
            return self.difficulty_score

        base_difficulty = 0.5

        # Adjust based on question type
        if self.question_type == QuestionType.BINARY:
            base_difficulty += 0.0  # Binary questions are baseline
        elif self.question_type == QuestionType.NUMERIC:
            base_difficulty += 0.2  # Numeric questions are harder
        elif self.question_type == QuestionType.MULTIPLE_CHOICE:
            base_difficulty += 0.1  # Multiple choice is moderately harder

        # Adjust based on time horizon
        if self.close_time:
            days_to_close = (self.close_time - datetime.now(timezone.utc)).days
            if days_to_close < 30:
                base_difficulty += 0.1  # Short-term predictions are harder
            elif days_to_close > 365:
                base_difficulty += 0.2  # Long-term predictions are harder

        # Adjust based on category
        category = self.categorize_question()
        category_difficulty_adjustments = {
            QuestionCategory.TECHNOLOGY: 0.1,
            QuestionCategory.ECONOMICS: 0.15,
            QuestionCategory.POLITICS: 0.2,
            QuestionCategory.HEALTH: 0.1,
            QuestionCategory.CLIMATE: 0.15,
            QuestionCategory.SCIENCE: 0.1,
            QuestionCategory.GEOPOLITICS: 0.25,
            QuestionCategory.BUSINESS: 0.1,
            QuestionCategory.OTHER: 0.05
        }

        base_difficulty += category_difficulty_adjustments.get(category, 0.0)

        return min(1.0, base_difficulty)

    def calculate_scoring_potential(self, tournament_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate scoring potential based on question characteristics and tournament context."""
        base_potential = 0.5

        # Higher potential for questions we can categorize well
        category = self.categorize_question()
        if category != QuestionCategory.OTHER:
            base_potential += 0.1

        # Adjust based on difficulty (moderate difficulty has highest potential)
        difficulty = self.calculate_difficulty_score()
        if 0.4 <= difficulty <= 0.7:
            base_potential += 0.2  # Sweet spot for scoring
        elif difficulty < 0.4:
            base_potential += 0.1  # Easy questions have some potential
        else:
            base_potential -= 0.1  # Very hard questions are risky

        # Adjust based on time to close
        if self.close_time:
            days_to_close = (self.close_time - datetime.now(timezone.utc)).days
            if 7 <= days_to_close <= 90:
                base_potential += 0.1  # Good time window for research

        # Tournament context adjustments
        if tournament_context:
            # Adjust based on competition level
            competition_level = tournament_context.get("competition_level", 0.5)
            base_potential += (1 - competition_level) * 0.1

            # Adjust based on question weight in tournament
            question_weight = tournament_context.get("question_weight", 1.0)
            base_potential *= question_weight

        return min(1.0, max(0.0, base_potential))

    def create_priority_assessment(
        self,
        confidence_level: float,
        tournament_context: Optional[Dict[str, Any]] = None
    ) -> QuestionPriority:
        """Create priority assessment for tournament resource allocation."""
        category = self.categorize_question()
        scoring_potential = self.calculate_scoring_potential(tournament_context)

        # Calculate deadline urgency
        if self.close_time:
            days_to_close = (self.close_time - datetime.now(timezone.utc)).days
            if days_to_close <= 1:
                deadline_urgency = 1.0
            elif days_to_close <= 7:
                deadline_urgency = 0.8
            elif days_to_close <= 30:
                deadline_urgency = 0.6
            else:
                deadline_urgency = 0.4
        else:
            deadline_urgency = 0.5

        # Calculate competitive advantage based on category specialization
        competitive_advantage = 0.5  # Default
        if tournament_context and "category_specializations" in tournament_context:
            specializations = tournament_context["category_specializations"]
            competitive_advantage = specializations.get(category, 0.5)

        # Calculate resource allocation recommendation
        difficulty = self.calculate_difficulty_score()
        if difficulty > 0.8:
            resource_allocation = 0.8  # High difficulty needs more resources
        elif difficulty < 0.3:
            resource_allocation = 0.3  # Easy questions need fewer resources
        else:
            resource_allocation = 0.5 + (difficulty - 0.5) * 0.4

        return QuestionPriority.create(
            question_id=self.id,
            category=category,
            confidence_level=confidence_level,
            scoring_potential=scoring_potential,
            resource_allocation=resource_allocation,
            deadline_urgency=deadline_urgency,
            competitive_advantage=competitive_advantage
        )

    def calculate_market_inefficiency_score(self, market_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate market inefficiency score for competitive advantage."""
        if self.market_inefficiency_score is not None:
            return self.market_inefficiency_score

        base_score = 0.5

        # Adjust based on question complexity
        difficulty = self.calculate_difficulty_score()
        if difficulty > 0.7:
            base_score += 0.2  # High complexity may have inefficiencies

        # Adjust based on category
        category = self.categorize_question()
        category_inefficiency_potential = {
            QuestionCategory.TECHNOLOGY: 0.3,
            QuestionCategory.ECONOMICS: 0.2,
            QuestionCategory.POLITICS: 0.4,
            QuestionCategory.HEALTH: 0.2,
            QuestionCategory.CLIMATE: 0.3,
            QuestionCategory.SCIENCE: 0.2,
            QuestionCategory.GEOPOLITICS: 0.4,
            QuestionCategory.BUSINESS: 0.3,
            QuestionCategory.OTHER: 0.1
        }

        base_score += category_inefficiency_potential.get(category, 0.1)

        # Market data adjustments
        if market_data:
            prediction_variance = market_data.get("prediction_variance", 0.1)
            base_score += min(0.3, prediction_variance * 2)  # Higher variance = more inefficiency

        self.market_inefficiency_score = min(1.0, base_score)
        return self.market_inefficiency_score

    def calculate_research_complexity_score(self) -> float:
        """Calculate research complexity score."""
        if self.research_complexity_score is not None:
            return self.research_complexity_score

        base_complexity = 0.5

        # Adjust based on question type
        if self.question_type == QuestionType.NUMERIC:
            base_complexity += 0.2
        elif self.question_type == QuestionType.MULTIPLE_CHOICE:
            base_complexity += 0.1

        # Adjust based on category
        category = self.categorize_question()
        category_complexity = {
            QuestionCategory.TECHNOLOGY: 0.3,
            QuestionCategory.ECONOMICS: 0.4,
            QuestionCategory.POLITICS: 0.3,
            QuestionCategory.HEALTH: 0.3,
            QuestionCategory.CLIMATE: 0.4,
            QuestionCategory.SCIENCE: 0.4,
            QuestionCategory.GEOPOLITICS: 0.5,
            QuestionCategory.BUSINESS: 0.3,
            QuestionCategory.OTHER: 0.2
        }

        base_complexity += category_complexity.get(category, 0.2)

        # Adjust based on time horizon
        if self.close_time:
            days_to_close = (self.close_time - datetime.now(timezone.utc)).days
            if days_to_close > 365:
                base_complexity += 0.2  # Long-term predictions need more research
            elif days_to_close < 7:
                base_complexity += 0.1  # Short-term may need rapid research

        self.research_complexity_score = min(1.0, base_complexity)
        return self.research_complexity_score

    def analyze_similar_questions(self, historical_questions: List["Question"]) -> List[Dict[str, Any]]:
        """Analyze similar questions for pattern recognition."""
        if not historical_questions:
            return []

        similar_questions = []
        my_category = self.categorize_question()

        for question in historical_questions:
            if question.id == self.id:
                continue

            similarity_score = self._calculate_question_similarity(question)
            if similarity_score > 0.3:  # Threshold for similarity
                similar_questions.append({
                    "question_id": str(question.id),
                    "title": question.title,
                    "category": question.categorize_question(),
                    "similarity_score": similarity_score,
                    "difficulty_score": question.calculate_difficulty_score(),
                    "historical_performance": question.historical_performance_data
                })

        # Sort by similarity score
        similar_questions.sort(key=lambda x: x["similarity_score"], reverse=True)
        self.similar_questions_analysis = similar_questions[:10]  # Keep top 10
        return self.similar_questions_analysis

    def _calculate_question_similarity(self, other_question: "Question") -> float:
        """Calculate similarity score between questions."""
        similarity = 0.0

        # Category similarity
        if self.categorize_question() == other_question.categorize_question():
            similarity += 0.4

        # Type similarity
        if self.question_type == other_question.question_type:
            similarity += 0.2

        # Title/description similarity (simple keyword matching)
        my_keywords = set(self.title.lower().split() + self.description.lower().split())
        other_keywords = set(other_question.title.lower().split() + other_question.description.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "will", "be", "is", "are", "was", "were"}
        my_keywords -= common_words
        other_keywords -= common_words

        if my_keywords and other_keywords:
            keyword_overlap = len(my_keywords.intersection(other_keywords)) / len(my_keywords.union(other_keywords))
            similarity += keyword_overlap * 0.4

        return min(1.0, similarity)

    def update_historical_performance(self, performance_data: Dict[str, Any]) -> None:
        """Update historical performance data."""
        if self.historical_performance_data is None:
            self.historical_performance_data = {}

        self.historical_performance_data.update(performance_data)
        self.updated_at = datetime.now(timezone.utc)
