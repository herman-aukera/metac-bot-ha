"""Tournament-specific configuration and utilities."""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional


class TournamentMode(Enum):
    """Tournament operation modes."""

    DEVELOPMENT = "development"
    TOURNAMENT = "tournament"
    QUARTERLY_CUP = "quarterly_cup"
    TEST = "test"


@dataclass
class TournamentConfig:
    """Tournament-specific configuration."""

    # Tournament identification
    tournament_id: int = 32813  # Fall 2025 AI Forecasting Benchmark
    tournament_slug: str = "fall-aib-2025"
    tournament_name: str = "Fall 2025 AI Forecasting Benchmark"

    # Operation mode
    mode: TournamentMode = TournamentMode.DEVELOPMENT

    # Scheduling and resource management
    scheduling_interval_hours: int = 4  # Updated default to 4 hours
    deadline_aware_scheduling: bool = True
    critical_period_frequency_hours: int = 2
    final_24h_frequency_hours: int = 1
    tournament_scope: str = "seasonal"
    max_concurrent_questions: int = 5
    max_research_reports_per_question: int = 1
    max_predictions_per_report: int = 5

    # Tournament scope management - NEW
    tournament_start_date: Optional[str] = None  # ISO format: "2025-09-01"
    tournament_end_date: Optional[str] = None  # ISO format: "2025-12-31"
    expected_total_questions: int = 75  # 50-100 questions for entire tournament
    min_expected_questions: int = 50
    max_expected_questions: int = 100
    questions_processed: int = 0  # Track progress
    sustainable_daily_rate: float = 1.0  # Questions per day on average

    # Compliance settings
    publish_reports: bool = True
    dry_run: bool = False
    skip_previously_forecasted: bool = True

    # Question filtering and prioritization
    enable_question_filtering: bool = True
    priority_categories: list = None
    min_confidence_threshold: float = 0.6

    # API and resource limits
    enable_proxy_credits: bool = True
    asknews_quota_limit: int = 9000

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.priority_categories is None:
            self.priority_categories = [
                "AI",
                "Technology",
                "Economics",
                "Politics",
                "Science",
            ]

    @classmethod
    def from_environment(cls) -> "TournamentConfig":
        """Create tournament configuration from environment variables."""

        # Determine mode
        mode_str = os.getenv("TOURNAMENT_MODE", "development").lower()
        if mode_str in ("true", "tournament"):
            mode = TournamentMode.TOURNAMENT
        elif mode_str == "quarterly_cup":
            mode = TournamentMode.QUARTERLY_CUP
        elif mode_str == "test":
            mode = TournamentMode.TEST
        else:
            mode = TournamentMode.DEVELOPMENT

        return cls(
            tournament_id=int(os.getenv("AIB_TOURNAMENT_ID", "32813")),
            tournament_slug=os.getenv("TOURNAMENT_SLUG", "fall-aib-2025"),
            tournament_name=os.getenv(
                "TOURNAMENT_NAME", "Fall 2025 AI Forecasting Benchmark"
            ),
            mode=mode,
            scheduling_interval_hours=int(os.getenv("SCHEDULING_FREQUENCY_HOURS", "4")),
            deadline_aware_scheduling=os.getenv(
                "DEADLINE_AWARE_SCHEDULING", "true"
            ).lower()
            == "true",
            critical_period_frequency_hours=int(
                os.getenv("CRITICAL_PERIOD_FREQUENCY_HOURS", "2")
            ),
            final_24h_frequency_hours=int(os.getenv("FINAL_24H_FREQUENCY_HOURS", "1")),
            tournament_scope=os.getenv("TOURNAMENT_SCOPE", "seasonal"),
            max_concurrent_questions=int(os.getenv("TOURNAMENT_MAX_QUESTIONS", "5")),
            max_research_reports_per_question=int(
                os.getenv("TOURNAMENT_MAX_RESEARCH_REPORTS", "1")
            ),
            max_predictions_per_report=int(
                os.getenv("TOURNAMENT_MAX_PREDICTIONS", "5")
            ),
            # Tournament scope management
            tournament_start_date=os.getenv(
                "TOURNAMENT_START_DATE"
            ),  # e.g., "2025-09-01"
            tournament_end_date=os.getenv("TOURNAMENT_END_DATE"),  # e.g., "2025-12-31"
            expected_total_questions=int(os.getenv("EXPECTED_TOTAL_QUESTIONS", "75")),
            min_expected_questions=int(os.getenv("MIN_EXPECTED_QUESTIONS", "50")),
            max_expected_questions=int(os.getenv("MAX_EXPECTED_QUESTIONS", "100")),
            questions_processed=int(os.getenv("QUESTIONS_PROCESSED", "0")),
            sustainable_daily_rate=float(os.getenv("SUSTAINABLE_DAILY_RATE", "1.0")),
            # Other settings
            publish_reports=os.getenv("PUBLISH_REPORTS", "true").lower() == "true",
            dry_run=os.getenv("DRY_RUN", "false").lower() == "true",
            skip_previously_forecasted=os.getenv(
                "SKIP_PREVIOUSLY_FORECASTED", "true"
            ).lower()
            == "true",
            enable_question_filtering=os.getenv(
                "ENABLE_QUESTION_FILTERING", "true"
            ).lower()
            == "true",
            min_confidence_threshold=float(
                os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.6")
            ),
            enable_proxy_credits=os.getenv("ENABLE_PROXY_CREDITS", "true").lower()
            == "true",
            asknews_quota_limit=int(os.getenv("ASKNEWS_QUOTA_LIMIT", "9000")),
        )

    def is_tournament_mode(self) -> bool:
        """Check if running in tournament mode."""
        return self.mode == TournamentMode.TOURNAMENT

    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return self.mode == TournamentMode.DEVELOPMENT

    def get_cron_schedule(self) -> str:
        """Get cron schedule based on tournament configuration."""
        if self.mode == TournamentMode.TOURNAMENT:
            return f"0 */{self.scheduling_interval_hours} * * *"
        elif self.mode == TournamentMode.QUARTERLY_CUP:
            return "0 0 */2 * *"  # Every 2 days at midnight
        else:
            return "0 */6 * * *"  # Every 6 hours for development

    def get_deadline_aware_frequency(self, hours_until_deadline: float) -> int:
        """Get scheduling frequency based on time until deadline."""
        if not self.deadline_aware_scheduling:
            return self.scheduling_interval_hours

        if hours_until_deadline <= 24:
            # Final 24 hours: most frequent updates
            return self.final_24h_frequency_hours
        elif hours_until_deadline <= 72:
            # Critical period (72 hours): more frequent updates
            return self.critical_period_frequency_hours
        else:
            # Normal period: standard frequency
            return self.scheduling_interval_hours

    def should_run_now(
        self, hours_until_deadline: float, hours_since_last_run: float
    ) -> bool:
        """Determine if the bot should run now based on deadline-aware scheduling."""
        required_frequency = self.get_deadline_aware_frequency(hours_until_deadline)
        return hours_since_last_run >= required_frequency

    def get_scheduling_strategy(self) -> Dict[str, Any]:
        """Get the current scheduling strategy configuration."""
        return {
            "base_frequency_hours": self.scheduling_interval_hours,
            "deadline_aware": self.deadline_aware_scheduling,
            "critical_period_frequency_hours": self.critical_period_frequency_hours,
            "final_24h_frequency_hours": self.final_24h_frequency_hours,
            "tournament_scope": self.tournament_scope,
            "cron_schedule": self.get_cron_schedule(),
        }

    def should_filter_questions(self) -> bool:
        """Check if question filtering should be applied."""
        return self.enable_question_filtering and self.is_tournament_mode()

    def get_question_priority_score(self, question_data: Dict[str, Any]) -> float:
        """Calculate priority score for a question based on tournament criteria."""
        if not self.should_filter_questions():
            return 1.0

        score = 0.0

        # Category-based scoring
        categories = question_data.get("categories", [])
        for category in categories:
            if any(
                priority_cat.lower() in category.lower()
                for priority_cat in self.priority_categories
            ):
                score += 0.3

        # Recency scoring (newer questions get higher priority)
        # This would need actual question creation date
        score += 0.2

        # Complexity scoring (binary questions might be prioritized)
        question_type = question_data.get("type", "")
        if question_type == "binary":
            score += 0.2
        elif question_type == "numeric":
            score += 0.15
        else:  # multiple choice
            score += 0.1

        # Activity scoring (questions with more activity might be prioritized)
        num_predictions = question_data.get("num_predictions", 0)
        if num_predictions > 100:
            score += 0.2
        elif num_predictions > 50:
            score += 0.1

        # Close time scoring (questions closing soon get higher priority)
        # This would need actual close time analysis
        score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def get_tournament_duration_days(self) -> Optional[int]:
        """Calculate tournament duration in days."""
        if not self.tournament_start_date or not self.tournament_end_date:
            return None

        try:
            start = datetime.fromisoformat(self.tournament_start_date)
            end = datetime.fromisoformat(self.tournament_end_date)
            return (end - start).days
        except (ValueError, TypeError):
            return None

    def get_tournament_progress(self) -> Dict[str, Any]:
        """Get current tournament progress and estimates."""
        duration_days = self.get_tournament_duration_days()

        if duration_days is None:
            # Default seasonal tournament estimate (4 months)
            duration_days = 120

        # Calculate progress
        progress_percentage = (
            self.questions_processed / self.expected_total_questions
        ) * 100
        remaining_questions = max(
            0, self.expected_total_questions - self.questions_processed
        )

        # Calculate sustainable rate
        if duration_days > 0:
            calculated_daily_rate = self.expected_total_questions / duration_days
        else:
            calculated_daily_rate = self.sustainable_daily_rate

        return {
            "tournament_duration_days": duration_days,
            "questions_processed": self.questions_processed,
            "expected_total_questions": self.expected_total_questions,
            "remaining_questions": remaining_questions,
            "progress_percentage": round(progress_percentage, 2),
            "sustainable_daily_rate": round(calculated_daily_rate, 2),
            "current_daily_rate": self.sustainable_daily_rate,
            "is_on_track": self.questions_processed
            <= (self.expected_total_questions * 0.8),  # Allow 20% buffer
        }

    def calculate_sustainable_forecasting_rate(
        self, days_elapsed: int = 0
    ) -> Dict[str, float]:
        """Calculate sustainable forecasting rates based on tournament scope."""
        duration_days = self.get_tournament_duration_days() or 120  # Default 4 months

        # Base calculations for seasonal tournament (not daily)
        total_questions_target = self.expected_total_questions

        # Calculate rates
        questions_per_day = total_questions_target / duration_days
        questions_per_week = questions_per_day * 7
        questions_per_month = questions_per_day * 30

        # Adjust based on current progress if days_elapsed provided
        if days_elapsed > 0 and days_elapsed < duration_days:
            remaining_days = duration_days - days_elapsed
            remaining_questions = max(
                0, total_questions_target - self.questions_processed
            )
            adjusted_daily_rate = (
                remaining_questions / remaining_days if remaining_days > 0 else 0
            )
        else:
            adjusted_daily_rate = questions_per_day

        return {
            "questions_per_day": round(questions_per_day, 2),
            "questions_per_week": round(questions_per_week, 2),
            "questions_per_month": round(questions_per_month, 2),
            "adjusted_daily_rate": round(adjusted_daily_rate, 2),
            "total_target": total_questions_target,
            "tournament_duration_days": duration_days,
        }

    def should_throttle_forecasting(self, current_daily_rate: float) -> bool:
        """Determine if forecasting should be throttled based on sustainable rates."""
        sustainable_rates = self.calculate_sustainable_forecasting_rate()
        target_daily_rate = sustainable_rates["adjusted_daily_rate"]

        # Throttle if current rate is more than 50% above sustainable rate
        return current_daily_rate > (target_daily_rate * 1.5)

    def get_recommended_scheduling_frequency(self) -> int:
        """Get recommended scheduling frequency based on tournament scope."""
        sustainable_rates = self.calculate_sustainable_forecasting_rate()
        questions_per_day = sustainable_rates["questions_per_day"]

        # For seasonal tournaments with low daily rates, less frequent scheduling is appropriate
        if questions_per_day <= 0.5:  # Less than 0.5 questions per day
            return 8  # Every 8 hours
        elif questions_per_day <= 1.0:  # 0.5-1 questions per day
            return 6  # Every 6 hours
        elif questions_per_day <= 2.0:  # 1-2 questions per day
            return 4  # Every 4 hours
        else:  # More than 2 questions per day
            return 2  # Every 2 hours

    def update_questions_processed(self, count: int):
        """Update the count of questions processed."""
        self.questions_processed = count

    def increment_questions_processed(self, increment: int = 1):
        """Increment the count of questions processed."""
        self.questions_processed += increment

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "tournament_id": self.tournament_id,
            "tournament_slug": self.tournament_slug,
            "tournament_name": self.tournament_name,
            "mode": self.mode.value,
            "scheduling_interval_hours": self.scheduling_interval_hours,
            "deadline_aware_scheduling": self.deadline_aware_scheduling,
            "critical_period_frequency_hours": self.critical_period_frequency_hours,
            "final_24h_frequency_hours": self.final_24h_frequency_hours,
            "tournament_scope": self.tournament_scope,
            "max_concurrent_questions": self.max_concurrent_questions,
            "max_research_reports_per_question": self.max_research_reports_per_question,
            "max_predictions_per_report": self.max_predictions_per_report,
            # Tournament scope management
            "tournament_start_date": self.tournament_start_date,
            "tournament_end_date": self.tournament_end_date,
            "expected_total_questions": self.expected_total_questions,
            "min_expected_questions": self.min_expected_questions,
            "max_expected_questions": self.max_expected_questions,
            "questions_processed": self.questions_processed,
            "sustainable_daily_rate": self.sustainable_daily_rate,
            # Other settings
            "publish_reports": self.publish_reports,
            "dry_run": self.dry_run,
            "skip_previously_forecasted": self.skip_previously_forecasted,
            "enable_question_filtering": self.enable_question_filtering,
            "priority_categories": self.priority_categories,
            "min_confidence_threshold": self.min_confidence_threshold,
            "enable_proxy_credits": self.enable_proxy_credits,
            "asknews_quota_limit": self.asknews_quota_limit,
        }


class TournamentScopeManager:
    """Utility class for managing tournament scope and question volume expectations."""

    def __init__(self, config: TournamentConfig):
        self.config = config

    def validate_seasonal_scope(self) -> Dict[str, Any]:
        """Validate that the tournament is configured for seasonal scope, not daily."""
        issues = []
        recommendations = []

        # Check scope setting
        if self.config.tournament_scope != "seasonal":
            issues.append(
                f"Tournament scope is '{self.config.tournament_scope}', should be 'seasonal'"
            )
            recommendations.append("Set TOURNAMENT_SCOPE=seasonal in environment")

        # Check question expectations
        sustainable_rates = self.config.calculate_sustainable_forecasting_rate()
        daily_rate = sustainable_rates["questions_per_day"]

        if (
            daily_rate > 5.0
        ):  # More than 5 questions per day seems too high for seasonal
            issues.append(
                f"Daily question rate of {daily_rate:.1f} seems too high for seasonal tournament"
            )
            recommendations.append(
                "Reduce EXPECTED_TOTAL_QUESTIONS or increase tournament duration"
            )

        # Check scheduling frequency
        if self.config.scheduling_interval_hours < 2:
            issues.append(
                f"Scheduling interval of {self.config.scheduling_interval_hours}h is too frequent for seasonal scope"
            )
            recommendations.append("Increase SCHEDULING_FREQUENCY_HOURS to 4 or higher")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "current_scope": self.config.tournament_scope,
            "expected_daily_rate": daily_rate,
            "scheduling_frequency": self.config.scheduling_interval_hours,
        }

    def get_scope_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of tournament scope configuration."""
        progress = self.config.get_tournament_progress()
        rates = self.config.calculate_sustainable_forecasting_rate()
        validation = self.validate_seasonal_scope()

        return {
            "tournament_info": {
                "id": self.config.tournament_id,
                "name": self.config.tournament_name,
                "scope": self.config.tournament_scope,
                "start_date": self.config.tournament_start_date,
                "end_date": self.config.tournament_end_date,
            },
            "question_expectations": {
                "total_expected": self.config.expected_total_questions,
                "range": f"{self.config.min_expected_questions}-{self.config.max_expected_questions}",
                "processed": self.config.questions_processed,
                "remaining": progress["remaining_questions"],
            },
            "sustainable_rates": rates,
            "progress": progress,
            "validation": validation,
            "recommended_frequency": self.config.get_recommended_scheduling_frequency(),
        }


# Global tournament configuration instance
_tournament_config: Optional[TournamentConfig] = None


def get_tournament_config() -> TournamentConfig:
    """Get the global tournament configuration instance."""
    global _tournament_config
    if _tournament_config is None:
        _tournament_config = TournamentConfig.from_environment()
    return _tournament_config


def get_tournament_scope_manager() -> TournamentScopeManager:
    """Get a tournament scope manager instance."""
    return TournamentScopeManager(get_tournament_config())


def reset_tournament_config():
    """Reset the global tournament configuration (useful for testing)."""
    global _tournament_config
    _tournament_config = None
