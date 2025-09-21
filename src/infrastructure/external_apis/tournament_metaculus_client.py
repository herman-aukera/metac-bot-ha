"""
Tournament-focused Metaculus API client with enhanced tournament operations.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import httpx
import structlog

from ...domain.entities.question import Question
from ...domain.value_objects.time_range import TimeRange
from ..config.settings import Settings
from .metaculus_client import MetaculusClient

logger = structlog.get_logger(__name__)


class TournamentPriority(Enum):
    """Priority levels for tournament questions."""

    CRITICAL = "critical"  # High-impact, closing soon
    HIGH = "high"  # High-impact or closing soon
    MEDIUM = "medium"  # Standard priority
    LOW = "low"  # Low impact or far deadline


@dataclass
class TournamentContext:
    """Context information for tournament operations."""

    tournament_id: Optional[str]
    tournament_name: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    scoring_method: str
    participant_count: int
    current_ranking: Optional[int]
    total_questions: int
    answered_questions: int

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate."""
        if self.total_questions == 0:
            return 0.0
        return self.answered_questions / self.total_questions

    @property
    def time_remaining(self) -> Optional[timedelta]:
        """Calculate time remaining in tournament."""
        if not self.end_time:
            return None
        return self.end_time - datetime.now(timezone.utc)


@dataclass
class QuestionDeadlineInfo:
    """Deadline tracking information for questions."""

    question_id: str
    close_time: datetime
    resolve_time: Optional[datetime]
    time_until_close: timedelta
    urgency_score: float
    submission_window: TimeRange

    @property
    def is_urgent(self) -> bool:
        """Check if question is urgent (closing within 24 hours)."""
        return self.time_until_close.total_seconds() < 86400  # 24 hours

    @property
    def is_critical(self) -> bool:
        """Check if question is critical (closing within 6 hours)."""
        return self.time_until_close.total_seconds() < 21600  # 6 hours


@dataclass
class QuestionCategory:
    """Question categorization for tournament strategy."""

    category_name: str
    confidence_threshold: float
    expected_accuracy: float
    resource_allocation: float
    strategy_type: str


class TournamentMetaculusClient(MetaculusClient):
    """Enhanced Metaculus client with tournament-specific capabilities."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.tournament_context: Optional[TournamentContext] = None
        self.question_categories: Dict[str, QuestionCategory] = {}
        self.deadline_tracker: Dict[str, QuestionDeadlineInfo] = {}
        self._initialize_categories()

    def _initialize_categories(self) -> None:
        """Initialize question categories with tournament strategies."""
        self.question_categories = {
            "technology": QuestionCategory(
                category_name="Technology",
                confidence_threshold=0.7,
                expected_accuracy=0.75,
                resource_allocation=0.25,
                strategy_type="research_intensive",
            ),
            "politics": QuestionCategory(
                category_name="Politics",
                confidence_threshold=0.6,
                expected_accuracy=0.65,
                resource_allocation=0.20,
                strategy_type="news_focused",
            ),
            "economics": QuestionCategory(
                category_name="Economics",
                confidence_threshold=0.65,
                expected_accuracy=0.70,
                resource_allocation=0.20,
                strategy_type="data_driven",
            ),
            "science": QuestionCategory(
                category_name="Science",
                confidence_threshold=0.75,
                expected_accuracy=0.80,
                resource_allocation=0.25,
                strategy_type="expert_consensus",
            ),
            "other": QuestionCategory(
                category_name="Other",
                confidence_threshold=0.5,
                expected_accuracy=0.60,
                resource_allocation=0.10,
                strategy_type="conservative",
            ),
        }

    async def fetch_tournament_questions(
        self,
        tournament_id: Optional[str] = None,
        include_resolved: bool = False,
        priority_filter: Optional[TournamentPriority] = None,
        limit: int = 100,
    ) -> List[Question]:
        """
        Fetch questions with tournament-specific filtering and prioritization.

        Args:
            tournament_id: Specific tournament to filter by
            include_resolved: Whether to include resolved questions
            priority_filter: Filter by priority level
            limit: Maximum number of questions to fetch

        Returns:
            List of tournament questions with enhanced metadata
        """
        logger.info(
            "Fetching tournament questions",
            tournament_id=tournament_id,
            priority_filter=priority_filter,
            limit=limit,
        )

        # Build query parameters
        params: Dict[str, Any] = {"limit": limit}
        if tournament_id:
            params["tournament"] = tournament_id
        if not include_resolved:
            params["status"] = "open"

        # Pagination integration: allow environment to control page size and max questions
        try:
            page_size = int(os.getenv("TOURNAMENT_PAGE_SIZE", str(min(limit, 50))))
        except Exception:
            page_size = min(limit, 50)
        try:
            max_questions = int(os.getenv("TOURNAMENT_MAX_QUESTIONS", str(limit)))
        except Exception:
            max_questions = limit
        max_questions = min(max_questions, limit)

        async def _fetch_page(page: int, size: int) -> List[Question]:
            offset = page * size
            remaining = max_questions - offset
            if remaining <= 0:
                return []
            page_limit = min(size, remaining)
            status_val = params.get("status", "open")
            return await self.fetch_questions(status=status_val, limit=page_limit, offset=offset)

        questions: List[Question] = []
        if page_size >= max_questions:
            questions = await self.fetch_questions(status=params.get("status", "open"), limit=max_questions, offset=0)
        else:
            page = 0
            seen_ids: Set[str] = set()
            while len(questions) < max_questions:
                batch = await _fetch_page(page, page_size)
                if not batch:
                    break
                for q in batch:
                    qid = str(q.id)
                    if qid in seen_ids:
                        continue
                    seen_ids.add(qid)
                    questions.append(q)
                    if len(questions) >= max_questions:
                        break
                if len(batch) < page_size:
                    break
                page += 1
            logger.info(
                "Pagination summary",
                pages=page + 1,
                retrieved=len(questions),
                target=max_questions,
                page_size=page_size,
            )

        enhanced_questions: List[Question] = []
        for question in questions:
            enhanced = await self._enhance_question_with_tournament_data(question)
            if priority_filter:
                if self._calculate_question_priority(enhanced) != priority_filter:
                    continue
            enhanced_questions.append(enhanced)

        enhanced_questions.sort(key=self._get_priority_sort_key, reverse=True)
        logger.info(
            "Fetched tournament questions",
            total=len(enhanced_questions),
            critical=len([q for q in enhanced_questions if self._is_critical_question(q)]),
        )
        return enhanced_questions

    async def _enhance_question_with_tournament_data(
        self, question: Question
    ) -> Question:
        """Enhance question with tournament-specific metadata."""
        # Calculate deadline information
        if question.close_time:
            deadline_info = self._calculate_deadline_info(question)
            # Use string key to avoid UUID vs str typing mismatches
            self.deadline_tracker[str(question.id)] = deadline_info

        # Categorize question
        category = self._categorize_question(question)

        # Calculate tournament priority
        priority = self._calculate_question_priority(question)

        # Add tournament metadata
        tournament_metadata = {
            "tournament_priority": priority.value,
            "category": category.category_name if category else "unknown",
            "urgency_score": self.deadline_tracker.get(
                str(question.id),
                QuestionDeadlineInfo(
                    question_id=str(question.id),
                    close_time=question.close_time or datetime.now(timezone.utc),
                    resolve_time=question.resolve_time,
                    time_until_close=timedelta(days=30),
                    urgency_score=0.0,
                    submission_window=TimeRange(
                        start=datetime.now(timezone.utc),
                        end=question.close_time
                        or datetime.now(timezone.utc) + timedelta(days=30),
                    ),
                ),
            ).urgency_score,
            "expected_accuracy": category.expected_accuracy if category else 0.6,
            "resource_allocation": category.resource_allocation if category else 0.1,
        }

        # Update question metadata
        question.metadata.update(tournament_metadata)

        return question

    def _calculate_deadline_info(self, question: Question) -> QuestionDeadlineInfo:
        """Calculate deadline tracking information for a question."""
        now = datetime.now(timezone.utc)
        close_time = question.close_time or (now + timedelta(days=30))
        time_until_close = close_time - now

        # Calculate urgency score (0-1, higher = more urgent)
        hours_until_close = time_until_close.total_seconds() / 3600
        if hours_until_close <= 6:
            urgency_score = 1.0
        elif hours_until_close <= 24:
            urgency_score = 0.8
        elif hours_until_close <= 72:
            urgency_score = 0.6
        elif hours_until_close <= 168:  # 1 week
            urgency_score = 0.4
        else:
            urgency_score = 0.2

        # Define optimal submission window (last 25% of question lifetime)
        question_lifetime = close_time - (question.created_at or now)
        submission_start = close_time - (question_lifetime * 0.25)

        return QuestionDeadlineInfo(
            question_id=str(question.id),
            close_time=close_time,
            resolve_time=question.resolve_time,
            time_until_close=time_until_close,
            urgency_score=urgency_score,
            submission_window=TimeRange(start=submission_start, end=close_time),
        )

    def _categorize_question(self, question: Question) -> Optional[QuestionCategory]:
        """Categorize question based on content and metadata."""
        title_lower = question.title.lower()
        description_lower = question.description.lower()

        # Check metadata category first
        if question.metadata and question.metadata.get("category"):
            category_name = question.metadata["category"].lower()
            if category_name in self.question_categories:
                return self.question_categories[category_name]

        # Technology keywords
        tech_keywords = [
            "ai",
            "artificial intelligence",
            "technology",
            "software",
            "computer",
            "algorithm",
            "machine learning",
            "blockchain",
            "cryptocurrency",
            "agi",
        ]
        if any(
            keyword in title_lower or keyword in description_lower
            for keyword in tech_keywords
        ):
            return self.question_categories["technology"]

        # Politics keywords
        politics_keywords = [
            "election",
            "president",
            "congress",
            "senate",
            "vote",
            "political",
            "government",
            "policy",
            "law",
            "regulation",
            "biden",
            "trump",
        ]
        if any(
            keyword in title_lower or keyword in description_lower
            for keyword in politics_keywords
        ):
            return self.question_categories["politics"]

        # Economics keywords
        econ_keywords = [
            "economy",
            "gdp",
            "inflation",
            "market",
            "stock",
            "price",
            "economic",
            "recession",
            "growth",
            "unemployment",
            "federal reserve",
            "interest rate",
        ]
        if any(
            keyword in title_lower or keyword in description_lower
            for keyword in econ_keywords
        ):
            return self.question_categories["economics"]

        # Science keywords
        science_keywords = [
            "climate",
            "covid",
            "vaccine",
            "research",
            "study",
            "scientific",
            "medicine",
            "health",
            "disease",
            "temperature",
            "carbon",
            "energy",
        ]
        if any(
            keyword in title_lower or keyword in description_lower
            for keyword in science_keywords
        ):
            return self.question_categories["science"]

        return self.question_categories["other"]

    def _calculate_question_priority(self, question: Question) -> TournamentPriority:
        """Calculate tournament priority for a question."""
        deadline_info = self.deadline_tracker.get(str(question.id))

        if not deadline_info:
            return TournamentPriority.LOW

        # Critical: closing within 6 hours
        if deadline_info.is_critical:
            return TournamentPriority.CRITICAL

        # High: closing within 24 hours or high-impact category
        if deadline_info.is_urgent:
            return TournamentPriority.HIGH

        # Check category impact
        category = self._categorize_question(question)
        if category and category.resource_allocation >= 0.2:
            return TournamentPriority.HIGH

        # Medium: closing within a week
        if deadline_info.time_until_close.total_seconds() < 604800:  # 1 week
            return TournamentPriority.MEDIUM

        return TournamentPriority.LOW

    def _is_critical_question(self, question: Question) -> bool:
        """Check if question is critical priority."""
        return (
            self._calculate_question_priority(question) == TournamentPriority.CRITICAL
        )

    def _get_priority_sort_key(self, question: Question) -> int:
        """Get sort key for priority ordering."""
        priority = self._calculate_question_priority(question)
        priority_values = {
            TournamentPriority.CRITICAL: 4,
            TournamentPriority.HIGH: 3,
            TournamentPriority.MEDIUM: 2,
            TournamentPriority.LOW: 1,
        }
        return priority_values.get(priority, 0)

    async def get_tournament_context(
        self, tournament_id: Optional[str] = None
    ) -> Optional[TournamentContext]:
        """
        Retrieve tournament context and competitive information.

        Args:
            tournament_id: Tournament to analyze

        Returns:
            Tournament context with competitive analysis
        """
        logger.info("Fetching tournament context", tournament_id=tournament_id)

        try:
            # Fetch tournament information
            if tournament_id:
                tournament_data = await self._fetch_tournament_data(tournament_id)
            else:
                tournament_data = await self._fetch_current_tournament_data()

            if not tournament_data:
                logger.warning("No tournament data available")
                return None

            # Get user's current performance
            user_predictions = await self.fetch_user_predictions()
            answered_questions = len(user_predictions)

            context = TournamentContext(
                tournament_id=tournament_data.get("id"),
                tournament_name=tournament_data.get("name"),
                start_time=self._parse_datetime(tournament_data.get("start_time")),
                end_time=self._parse_datetime(tournament_data.get("end_time")),
                scoring_method=tournament_data.get("scoring_method", "brier"),
                participant_count=tournament_data.get("participant_count", 0),
                current_ranking=tournament_data.get("user_ranking"),
                total_questions=tournament_data.get("question_count", 0),
                answered_questions=answered_questions,
            )

            self.tournament_context = context
            logger.info(
                "Tournament context retrieved",
                tournament_name=context.tournament_name,
                completion_rate=context.completion_rate,
                ranking=context.current_ranking,
            )

            return context

        except Exception as e:
            logger.error("Failed to fetch tournament context", error=str(e))
            return None

    async def _fetch_tournament_data(
        self, tournament_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch specific tournament data."""
        try:
            async with httpx.AsyncClient() as client:
                headers = self._get_headers()
                response = await client.get(
                    f"{self.base_url}/tournaments/{tournament_id}/", headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(
                "Failed to fetch tournament data",
                tournament_id=tournament_id,
                error=str(e),
            )
            return None

    async def _fetch_current_tournament_data(self) -> Optional[Dict[str, Any]]:
        """Fetch current active tournament data."""
        try:
            async with httpx.AsyncClient() as client:
                headers = self._get_headers()
                response = await client.get(
                    f"{self.base_url}/tournaments/",
                    headers=headers,
                    params={"status": "active", "limit": 1},
                )
                response.raise_for_status()

                data = response.json()
                tournaments = data.get("results", [])

                if tournaments:
                    return tournaments[0]

                return None
        except Exception as e:
            logger.error("Failed to fetch current tournament data", error=str(e))
            return None

    async def optimize_submission_timing(
        self, question_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize submission timing for multiple questions.

        Args:
            question_ids: List of question IDs to optimize

        Returns:
            Dictionary mapping question IDs to timing recommendations
        """
        logger.info("Optimizing submission timing", question_count=len(question_ids))

        timing_recommendations = {}

        for question_id in question_ids:
            deadline_info = self.deadline_tracker.get(question_id)

            if not deadline_info:
                # Fetch question to get deadline info
                question = await self.fetch_question(int(question_id))
                if question:
                    deadline_info = self._calculate_deadline_info(question)
                    self.deadline_tracker[question_id] = deadline_info

            if deadline_info:
                recommendation = self._calculate_optimal_submission_time(deadline_info)
                timing_recommendations[question_id] = recommendation

        logger.info(
            "Submission timing optimized", recommendations=len(timing_recommendations)
        )

        return timing_recommendations

    def _calculate_optimal_submission_time(
        self, deadline_info: QuestionDeadlineInfo
    ) -> Dict[str, Any]:
        """Calculate optimal submission timing for a question."""
        now = datetime.now(timezone.utc)

        # Determine optimal submission strategy
        if deadline_info.is_critical:
            # Submit immediately
            optimal_time = now
            strategy = "immediate"
            reason = "Critical deadline - submit now"
        elif deadline_info.is_urgent:
            # Submit within next few hours
            optimal_time = now + timedelta(hours=2)
            strategy = "urgent"
            reason = "Urgent deadline - submit within 2 hours"
        elif (
            deadline_info.submission_window.start
            <= now
            <= deadline_info.submission_window.end
        ):
            # We're in optimal window
            optimal_time = now + timedelta(hours=6)
            strategy = "optimal_window"
            reason = "In optimal submission window"
        else:
            # Wait for optimal window
            optimal_time = deadline_info.submission_window.start
            strategy = "wait_for_window"
            reason = "Wait for optimal submission window"

        return {
            "optimal_time": optimal_time.isoformat(),
            "strategy": strategy,
            "reason": reason,
            "urgency_score": deadline_info.urgency_score,
            "time_until_close": deadline_info.time_until_close.total_seconds(),
            "submission_window": {
                "start": deadline_info.submission_window.start.isoformat(),
                "end": deadline_info.submission_window.end.isoformat(),
            },
        }

    async def analyze_competitive_landscape(
        self, tournament_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze competitive landscape and market inefficiencies.

        Args:
            tournament_id: Tournament to analyze

        Returns:
            Competitive analysis with strategic recommendations
        """
        logger.info("Analyzing competitive landscape", tournament_id=tournament_id)

        try:
            # Get tournament context
            context = await self.get_tournament_context(tournament_id)
            if not context:
                return {"error": "No tournament context available"}

            # Analyze question distribution
            questions = await self.fetch_tournament_questions(tournament_id, limit=200)

            # Calculate category distribution
            category_distribution: Dict[str, int] = {}
            for question in questions:
                category = question.metadata.get("category", "unknown")
                category_distribution[category] = (
                    category_distribution.get(category, 0) + 1
                )

            # Identify high-value opportunities
            high_value_questions = [
                q
                for q in questions
                if q.metadata.get("tournament_priority") in ["critical", "high"]
            ]

            # Calculate competitive metrics
            analysis = {
                "tournament_context": {
                    "name": context.tournament_name,
                    "completion_rate": context.completion_rate,
                    "ranking": context.current_ranking,
                    "participant_count": context.participant_count,
                    "time_remaining": (
                        context.time_remaining.total_seconds()
                        if context.time_remaining
                        else None
                    ),
                },
                "question_analysis": {
                    "total_questions": len(questions),
                    "high_priority_questions": len(high_value_questions),
                    "category_distribution": category_distribution,
                    "urgent_questions": len(
                        [q for q in questions if self._is_critical_question(q)]
                    ),
                },
                "strategic_recommendations": self._generate_strategic_recommendations(
                    context, questions, category_distribution
                ),
                "market_inefficiencies": self._identify_market_inefficiencies(
                    questions
                ),
            }

            logger.info(
                "Competitive analysis completed",
                total_questions=len(questions),
                high_priority=len(high_value_questions),
            )

            return analysis

        except Exception as e:
            logger.error("Failed to analyze competitive landscape", error=str(e))
            return {"error": str(e)}

    def _generate_strategic_recommendations(
        self,
        context: TournamentContext,
        questions: List[Question],
        category_distribution: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Generate strategic recommendations based on analysis."""
        recommendations = []

        # Completion rate recommendation
        if context.completion_rate < 0.5:
            recommendations.append(
                {
                    "type": "completion_rate",
                    "priority": "high",
                    "message": f"Low completion rate ({context.completion_rate:.1%}). Focus on answering more questions.",
                    "action": "increase_question_volume",
                }
            )

        # Category focus recommendation
        dominant_category = max(category_distribution.items(), key=lambda x: x[1])
        if dominant_category[1] > len(questions) * 0.4:
            recommendations.append(
                {
                    "type": "category_focus",
                    "priority": "medium",
                    "message": f"High concentration in {dominant_category[0]} ({dominant_category[1]} questions). Consider specialization.",
                    "action": "specialize_category",
                    "category": dominant_category[0],
                }
            )

        # Urgency recommendation
        urgent_questions = [q for q in questions if self._is_critical_question(q)]
        if urgent_questions:
            recommendations.append(
                {
                    "type": "urgency",
                    "priority": "critical",
                    "message": f"{len(urgent_questions)} questions closing soon. Prioritize immediate action.",
                    "action": "handle_urgent_questions",
                    "question_count": str(len(urgent_questions)),
                }
            )

        return recommendations

    def _identify_market_inefficiencies(
        self, questions: List[Question]
    ) -> List[Dict[str, Any]]:
        """Identify potential market inefficiencies."""
        inefficiencies = []

        # Look for questions with extreme community predictions
        for question in questions:
            community_pred = question.metadata.get("community_prediction")
            if community_pred:
                if isinstance(community_pred, (int, float)):
                    if community_pred < 0.1 or community_pred > 0.9:
                        inefficiencies.append(
                            {
                                "question_id": question.id,
                                "type": "extreme_consensus",
                                "community_prediction": community_pred,
                                "opportunity": (
                                    "contrarian_position"
                                    if community_pred > 0.9
                                    else "confirmation_bias"
                                ),
                            }
                        )

        # Look for questions with low prediction counts
        low_participation = [
            q for q in questions if q.metadata.get("prediction_count", 0) < 10
        ]

        if low_participation:
            inefficiencies.append(
                {
                    "type": "low_participation",
                    "question_count": len(low_participation),
                    "opportunity": "early_mover_advantage",
                }
            )

        return inefficiencies

    def get_deadline_summary(self) -> Dict[str, Any]:
        """Get summary of question deadlines and urgency."""
        now = datetime.now(timezone.utc)

        summary: Dict[str, List[Any]] = {
            "critical": [],  # < 6 hours
            "urgent": [],  # < 24 hours
            "soon": [],  # < 72 hours
            "upcoming": [],  # < 1 week
        }

        for question_id, deadline_info in self.deadline_tracker.items():
            hours_remaining = deadline_info.time_until_close.total_seconds() / 3600

            deadline_summary = {
                "question_id": question_id,
                "close_time": deadline_info.close_time.isoformat(),
                "hours_remaining": hours_remaining,
                "urgency_score": deadline_info.urgency_score,
            }

            if hours_remaining < 6:
                summary["critical"].append(deadline_summary)
            elif hours_remaining < 24:
                summary["urgent"].append(deadline_summary)
            elif hours_remaining < 72:
                summary["soon"].append(deadline_summary)
            elif hours_remaining < 168:
                summary["upcoming"].append(deadline_summary)

        return summary
