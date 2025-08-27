"""Tournament-specific question filtering and prioritization service."""

from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
import logging

from ..entities.question import Question
from ...infrastructure.config.tournament_config import get_tournament_config

logger = logging.getLogger(__name__)


class TournamentQuestionFilter:
    """Service for filtering and prioritizing questions for tournament participation."""

    def __init__(self):
        """Initialize the tournament question filter."""
        self.config = get_tournament_config()
        self.logger = logging.getLogger(__name__)

    def filter_and_prioritize_questions(
        self,
        questions: List[Question],
        max_questions: int = None
    ) -> List[Tuple[Question, float]]:
        """
        Filter and prioritize questions for tournament forecasting.

        Args:
            questions: List of questions to filter and prioritize
            max_questions: Maximum number of questions to return (uses config default if None)

        Returns:
            List of (question, priority_score) tuples, sorted by priority (highest first)
        """
        if max_questions is None:
            max_questions = self.config.max_concurrent_questions

        # If not in tournament mode, return all questions with equal priority
        if not self.config.should_filter_questions():
            return [(q, 1.0) for q in questions[:max_questions]]

        # Calculate priority scores for all questions
        scored_questions = []
        for question in questions:
            try:
                score = self._calculate_question_priority(question)
                if score >= self.config.min_confidence_threshold:
                    scored_questions.append((question, score))
            except Exception as e:
                self.logger.warning(f"Error scoring question {question.id}: {e}")
                # Include with default score if scoring fails
                scored_questions.append((question, 0.5))

        # Sort by priority score (highest first)
        scored_questions.sort(key=lambda x: x[1], reverse=True)

        # Return top questions up to max_questions limit
        result = scored_questions[:max_questions]

        self.logger.info(
            f"Filtered {len(questions)} questions to {len(result)} high-priority questions "
            f"(min_score: {self.config.min_confidence_threshold})"
        )

        return result

    def _calculate_question_priority(self, question: Question) -> float:
        """
        Calculate priority score for a question based on tournament criteria.

        Args:
            question: Question to score

        Returns:
            Priority score between 0.0 and 1.0
        """
        score = 0.0

        # Base score for all questions
        score += 0.1

        # Category-based scoring
        score += self._score_by_categories(question)

        # Question type scoring
        score += self._score_by_question_type(question)

        # Timing scoring (questions closing soon get higher priority)
        score += self._score_by_timing(question)

        # Activity scoring (questions with engagement get higher priority)
        score += self._score_by_activity(question)

        # Complexity scoring (avoid overly complex questions in tournament mode)
        score += self._score_by_complexity(question)

        return min(score, 1.0)  # Cap at 1.0

    def _score_by_categories(self, question: Question) -> float:
        """Score question based on category relevance."""
        if not question.categories:
            return 0.1  # Default score for uncategorized questions

        category_score = 0.0
        for category in question.categories:
            for priority_cat in self.config.priority_categories:
                if priority_cat.lower() in category.lower():
                    category_score += 0.15
                    break  # Avoid double-counting

        return min(category_score, 0.3)  # Cap category contribution

    def _score_by_question_type(self, question: Question) -> float:
        """Score question based on type (binary, numeric, multiple choice)."""
        type_scores = {
            "binary": 0.25,      # Highest priority - easier to forecast accurately
            "numeric": 0.20,     # Medium priority - good for calibration
            "multiple_choice": 0.15  # Lower priority - more complex
        }

        question_type = question.question_type.value.lower() if question.question_type else "unknown"
        return type_scores.get(question_type, 0.1)

    def _score_by_timing(self, question: Question) -> float:
        """Score question based on timing considerations."""
        if not question.close_time:
            return 0.1  # Default for questions without close time

        now = datetime.now(timezone.utc)
        time_to_close = (question.close_time - now).total_seconds()

        # Questions closing in 1-7 days get highest priority
        if 86400 <= time_to_close <= 604800:  # 1-7 days
            return 0.2
        # Questions closing in 7-30 days get medium priority
        elif 604800 < time_to_close <= 2592000:  # 7-30 days
            return 0.15
        # Questions closing very soon (< 1 day) get lower priority (might be too late)
        elif time_to_close < 86400:
            return 0.05
        # Questions closing far in future get lower priority
        else:
            return 0.1

    def _score_by_activity(self, question: Question) -> float:
        """Score question based on community activity."""
        # This would ideally use actual prediction counts, comments, etc.
        # For now, use metadata if available
        metadata = question.metadata or {}

        # Look for activity indicators in metadata
        num_predictions = metadata.get("num_predictions", 0)
        num_comments = metadata.get("num_comments", 0)

        activity_score = 0.0

        # Prediction count scoring
        if num_predictions > 100:
            activity_score += 0.1
        elif num_predictions > 50:
            activity_score += 0.05

        # Comment count scoring
        if num_comments > 20:
            activity_score += 0.05
        elif num_comments > 10:
            activity_score += 0.025

        return min(activity_score, 0.15)  # Cap activity contribution

    def _score_by_complexity(self, question: Question) -> float:
        """Score question based on complexity (simpler questions preferred in tournament)."""
        complexity_score = 0.1  # Base complexity score

        # Analyze question text length (very long questions might be complex)
        question_length = len(question.description or "") + len(question.title or "")

        if question_length < 500:
            complexity_score += 0.05  # Shorter questions might be simpler
        elif question_length > 2000:
            complexity_score -= 0.05  # Very long questions might be complex

        # Analyze title for complexity indicators
        title_lower = (question.title or "").lower()
        complexity_indicators = [
            "conditional", "if and only if", "multiple", "complex",
            "various", "several", "numerous", "detailed"
        ]

        for indicator in complexity_indicators:
            if indicator in title_lower:
                complexity_score -= 0.02

        # Simplicity indicators
        simplicity_indicators = ["will", "by", "before", "after", "yes", "no"]
        for indicator in simplicity_indicators:
            if indicator in title_lower:
                complexity_score += 0.01

        return max(complexity_score, 0.0)  # Don't go negative

    def get_filtering_stats(self, questions: List[Question]) -> Dict[str, Any]:
        """Get statistics about question filtering for monitoring."""
        if not questions:
            return {"total_questions": 0}

        stats = {
            "total_questions": len(questions),
            "tournament_mode": self.config.is_tournament_mode(),
            "filtering_enabled": self.config.should_filter_questions(),
            "min_confidence_threshold": self.config.min_confidence_threshold,
            "max_concurrent_questions": self.config.max_concurrent_questions
        }

        if self.config.should_filter_questions():
            # Calculate score distribution
            scores = []
            for question in questions:
                try:
                    score = self._calculate_question_priority(question)
                    scores.append(score)
                except Exception:
                    scores.append(0.0)

            if scores:
                stats.update({
                    "avg_priority_score": sum(scores) / len(scores),
                    "max_priority_score": max(scores),
                    "min_priority_score": min(scores),
                    "questions_above_threshold": sum(1 for s in scores if s >= self.config.min_confidence_threshold)
                })

        return stats
