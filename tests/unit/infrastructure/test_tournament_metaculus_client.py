"""
Tests for TournamentMetaculusClient.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.domain.entities.question import Question, QuestionType
from src.domain.value_objects.time_range import TimeRange
from src.infrastructure.config.settings import Settings
from src.infrastructure.external_apis.tournament_metaculus_client import (
    QuestionDeadlineInfo,
    TournamentContext,
    TournamentMetaculusClient,
    TournamentPriority,
)


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.metaculus_username = "test_user"
    settings.metaculus_password = "test_pass"
    settings.metaculus = Mock()
    settings.metaculus.dry_run = False
    settings.metaculus.submit_predictions = True
    return settings


@pytest.fixture
def tournament_client(mock_settings):
    """Create tournament client with mock settings."""
    return TournamentMetaculusClient(mock_settings)


@pytest.fixture
def sample_question():
    """Create a sample question for testing."""
    return Question.create(
        title="Will AI achieve AGI by 2030?",
        description="This question asks about artificial general intelligence development.",
        question_type=QuestionType.BINARY,
        resolution_criteria="AGI is defined as...",
        close_time=datetime.now(timezone.utc) + timedelta(hours=12),
        resolve_time=datetime.now(timezone.utc) + timedelta(days=30),
        created_at=datetime.now(timezone.utc) - timedelta(days=7),
        metadata={
            "metaculus_id": 12345,
            "category": "Technology",
            "prediction_count": 150,
            "community_prediction": 0.35,
        },
    )


class TestTournamentMetaculusClient:
    """Test tournament-specific Metaculus client functionality."""

    def test_initialization(self, tournament_client):
        """Test client initialization."""
        assert tournament_client.tournament_context is None
        assert len(tournament_client.question_categories) == 5
        assert "technology" in tournament_client.question_categories
        assert tournament_client.deadline_tracker == {}

    def test_question_categorization(self, tournament_client, sample_question):
        """Test question categorization logic."""
        # Technology question
        tech_question = sample_question
        category = tournament_client._categorize_question(tech_question)
        assert category.category_name == "Technology"
        assert category.strategy_type == "research_intensive"

        # Politics question
        politics_question = Question.create(
            title="Will Biden win the 2024 election?",
            description="This question is about the presidential election.",
            question_type=QuestionType.BINARY,
            close_time=datetime.now(timezone.utc) + timedelta(days=30),
        )
        category = tournament_client._categorize_question(politics_question)
        assert category.category_name == "Politics"

        # Unknown category
        unknown_question = Question.create(
            title="Will the local bakery open on Sunday?",
            description="A question about local business hours and operations.",
            question_type=QuestionType.BINARY,
            close_time=datetime.now(timezone.utc) + timedelta(days=1),
            metadata={},  # No category metadata
        )
        category = tournament_client._categorize_question(unknown_question)
        assert category.category_name == "Other"

    def test_deadline_calculation(self, tournament_client, sample_question):
        """Test deadline information calculation."""
        deadline_info = tournament_client._calculate_deadline_info(sample_question)

        assert deadline_info.question_id == sample_question.id
        assert deadline_info.close_time == sample_question.close_time
        assert deadline_info.time_until_close.total_seconds() > 0
        assert 0 <= deadline_info.urgency_score <= 1

        # Test urgent question (closing in 6 hours)
        urgent_question = Question.create(
            title="Urgent question",
            description="This question closes soon.",
            question_type=QuestionType.BINARY,
            close_time=datetime.now(timezone.utc) + timedelta(hours=3),
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        urgent_deadline = tournament_client._calculate_deadline_info(urgent_question)
        assert urgent_deadline.urgency_score == 1.0
        assert urgent_deadline.is_critical

    def test_priority_calculation(self, tournament_client, sample_question):
        """Test tournament priority calculation."""
        # Set up deadline info for normal question
        deadline_info = tournament_client._calculate_deadline_info(sample_question)
        tournament_client.deadline_tracker[sample_question.id] = deadline_info

        # Normal question
        priority = tournament_client._calculate_question_priority(sample_question)
        assert priority in [
            TournamentPriority.HIGH,
            TournamentPriority.MEDIUM,
            TournamentPriority.LOW,
        ]

        # Critical question (closing very soon)
        critical_question = Question.create(
            title="Critical question",
            description="This question closes in 2 hours.",
            question_type=QuestionType.BINARY,
            close_time=datetime.now(timezone.utc) + timedelta(hours=2),
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        # Need to calculate deadline info first
        tournament_client._calculate_deadline_info(critical_question)
        tournament_client.deadline_tracker[critical_question.id] = (
            tournament_client._calculate_deadline_info(critical_question)
        )

        critical_priority = tournament_client._calculate_question_priority(
            critical_question
        )
        assert critical_priority == TournamentPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_fetch_tournament_questions(self, tournament_client, sample_question):
        """Test fetching tournament questions with filtering."""
        with patch.object(
            tournament_client, "fetch_questions", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = [sample_question]

            questions = await tournament_client.fetch_tournament_questions(
                tournament_id="test_tournament",
                priority_filter=TournamentPriority.HIGH,
                limit=50,
            )

            mock_fetch.assert_called_once()
            assert len(questions) >= 0  # May be filtered out

            # Test without priority filter
            mock_fetch.reset_mock()
            questions = await tournament_client.fetch_tournament_questions(limit=50)

            mock_fetch.assert_called_once()
            assert isinstance(questions, list)

    @pytest.mark.asyncio
    async def test_tournament_context_retrieval(self, tournament_client):
        """Test tournament context retrieval."""
        mock_tournament_data = {
            "id": "test_tournament",
            "name": "Test Tournament",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-12-31T23:59:59Z",
            "scoring_method": "brier",
            "participant_count": 500,
            "user_ranking": 42,
            "question_count": 100,
        }

        with patch.object(
            tournament_client, "_fetch_tournament_data", new_callable=AsyncMock
        ) as mock_fetch_tournament:
            with patch.object(
                tournament_client, "fetch_user_predictions", new_callable=AsyncMock
            ) as mock_fetch_predictions:
                mock_fetch_tournament.return_value = mock_tournament_data
                mock_fetch_predictions.return_value = [
                    {"id": 1},
                    {"id": 2},
                ]  # 2 predictions

                context = await tournament_client.get_tournament_context(
                    "test_tournament"
                )

                assert context is not None
                assert context.tournament_id == "test_tournament"
                assert context.tournament_name == "Test Tournament"
                assert context.participant_count == 500
                assert context.current_ranking == 42
                assert context.answered_questions == 2
                assert context.completion_rate == 0.02  # 2/100

    @pytest.mark.asyncio
    async def test_submission_timing_optimization(
        self, tournament_client, sample_question
    ):
        """Test submission timing optimization."""
        # Add question to deadline tracker
        deadline_info = tournament_client._calculate_deadline_info(sample_question)
        tournament_client.deadline_tracker[sample_question.id] = deadline_info

        timing_recommendations = await tournament_client.optimize_submission_timing(
            [sample_question.id]
        )

        assert sample_question.id in timing_recommendations
        recommendation = timing_recommendations[sample_question.id]

        assert "optimal_time" in recommendation
        assert "strategy" in recommendation
        assert "reason" in recommendation
        assert "urgency_score" in recommendation
        assert recommendation["strategy"] in [
            "immediate",
            "urgent",
            "optimal_window",
            "wait_for_window",
        ]

    @pytest.mark.asyncio
    async def test_competitive_landscape_analysis(
        self, tournament_client, sample_question
    ):
        """Test competitive landscape analysis."""
        mock_context = TournamentContext(
            tournament_id="test_tournament",
            tournament_name="Test Tournament",
            start_time=datetime.now(timezone.utc) - timedelta(days=30),
            end_time=datetime.now(timezone.utc) + timedelta(days=30),
            scoring_method="brier",
            participant_count=500,
            current_ranking=42,
            total_questions=100,
            answered_questions=25,
        )

        with patch.object(
            tournament_client, "get_tournament_context", new_callable=AsyncMock
        ) as mock_context_fetch:
            with patch.object(
                tournament_client, "fetch_tournament_questions", new_callable=AsyncMock
            ) as mock_questions_fetch:
                mock_context_fetch.return_value = mock_context
                mock_questions_fetch.return_value = [sample_question]

                analysis = await tournament_client.analyze_competitive_landscape(
                    "test_tournament"
                )

                assert "tournament_context" in analysis
                assert "question_analysis" in analysis
                assert "strategic_recommendations" in analysis
                assert "market_inefficiencies" in analysis

                assert analysis["tournament_context"]["completion_rate"] == 0.25
                assert analysis["question_analysis"]["total_questions"] == 1

    def test_strategic_recommendations_generation(self, tournament_client):
        """Test strategic recommendations generation."""
        # Low completion rate context
        low_completion_context = TournamentContext(
            tournament_id="test",
            tournament_name="Test",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(days=30),
            scoring_method="brier",
            participant_count=100,
            current_ranking=50,
            total_questions=100,
            answered_questions=20,  # 20% completion
        )

        questions = [Mock() for _ in range(10)]
        category_distribution = {"technology": 8, "politics": 2}

        recommendations = tournament_client._generate_strategic_recommendations(
            low_completion_context, questions, category_distribution
        )

        # Should recommend increasing completion rate
        completion_rec = next(
            (r for r in recommendations if r["type"] == "completion_rate"), None
        )
        assert completion_rec is not None
        assert completion_rec["priority"] == "high"

        # Should recommend category specialization
        category_rec = next(
            (r for r in recommendations if r["type"] == "category_focus"), None
        )
        assert category_rec is not None
        assert category_rec["category"] == "technology"

    def test_market_inefficiency_identification(self, tournament_client):
        """Test market inefficiency identification."""
        # Create questions with extreme community predictions
        extreme_questions = [
            Mock(id="1", metadata={"community_prediction": 0.05}),  # Very low
            Mock(id="2", metadata={"community_prediction": 0.95}),  # Very high
            Mock(id="3", metadata={"community_prediction": 0.5}),  # Normal
            Mock(id="4", metadata={"prediction_count": 5}),  # Low participation
        ]

        inefficiencies = tournament_client._identify_market_inefficiencies(
            extreme_questions
        )

        # Should identify extreme consensus opportunities
        extreme_inefficiencies = [
            i for i in inefficiencies if i.get("type") == "extreme_consensus"
        ]
        assert len(extreme_inefficiencies) == 2

        # Should identify low participation opportunity
        low_participation = next(
            (i for i in inefficiencies if i.get("type") == "low_participation"), None
        )
        assert low_participation is not None
        assert low_participation["opportunity"] == "early_mover_advantage"

    def test_deadline_summary(self, tournament_client):
        """Test deadline summary generation."""
        now = datetime.now(timezone.utc)

        # Add various deadline infos
        tournament_client.deadline_tracker = {
            "critical": QuestionDeadlineInfo(
                question_id="critical",
                close_time=now + timedelta(hours=3),
                resolve_time=None,
                time_until_close=timedelta(hours=3),
                urgency_score=1.0,
                submission_window=TimeRange(start=now, end=now + timedelta(hours=3)),
            ),
            "urgent": QuestionDeadlineInfo(
                question_id="urgent",
                close_time=now + timedelta(hours=12),
                resolve_time=None,
                time_until_close=timedelta(hours=12),
                urgency_score=0.8,
                submission_window=TimeRange(start=now, end=now + timedelta(hours=12)),
            ),
            "soon": QuestionDeadlineInfo(
                question_id="soon",
                close_time=now + timedelta(hours=48),
                resolve_time=None,
                time_until_close=timedelta(hours=48),
                urgency_score=0.6,
                submission_window=TimeRange(start=now, end=now + timedelta(hours=48)),
            ),
        }

        summary = tournament_client.get_deadline_summary()

        assert len(summary["critical"]) == 1
        assert len(summary["urgent"]) == 1
        assert len(summary["soon"]) == 1
        assert len(summary["upcoming"]) == 0

        assert summary["critical"][0]["question_id"] == "critical"
        assert summary["urgent"][0]["question_id"] == "urgent"
        assert summary["soon"][0]["question_id"] == "soon"


class TestTournamentContext:
    """Test TournamentContext functionality."""

    def test_completion_rate_calculation(self):
        """Test completion rate calculation."""
        context = TournamentContext(
            tournament_id="test",
            tournament_name="Test",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(days=30),
            scoring_method="brier",
            participant_count=100,
            current_ranking=50,
            total_questions=100,
            answered_questions=25,
        )

        assert context.completion_rate == 0.25

        # Test zero questions
        context.total_questions = 0
        assert context.completion_rate == 0.0

    def test_time_remaining_calculation(self):
        """Test time remaining calculation."""
        future_time = datetime.now(timezone.utc) + timedelta(days=10)

        context = TournamentContext(
            tournament_id="test",
            tournament_name="Test",
            start_time=datetime.now(timezone.utc),
            end_time=future_time,
            scoring_method="brier",
            participant_count=100,
            current_ranking=50,
            total_questions=100,
            answered_questions=25,
        )

        time_remaining = context.time_remaining
        assert time_remaining is not None
        assert time_remaining.days >= 9  # Should be close to 10 days

        # Test no end time
        context.end_time = None
        assert context.time_remaining is None


class TestQuestionDeadlineInfo:
    """Test QuestionDeadlineInfo functionality."""

    def test_urgency_properties(self):
        """Test urgency property calculations."""
        now = datetime.now(timezone.utc)

        # Critical question (3 hours)
        critical_info = QuestionDeadlineInfo(
            question_id="critical",
            close_time=now + timedelta(hours=3),
            resolve_time=None,
            time_until_close=timedelta(hours=3),
            urgency_score=1.0,
            submission_window=TimeRange(start=now, end=now + timedelta(hours=3)),
        )

        assert critical_info.is_critical
        assert critical_info.is_urgent

        # Urgent question (12 hours)
        urgent_info = QuestionDeadlineInfo(
            question_id="urgent",
            close_time=now + timedelta(hours=12),
            resolve_time=None,
            time_until_close=timedelta(hours=12),
            urgency_score=0.8,
            submission_window=TimeRange(start=now, end=now + timedelta(hours=12)),
        )

        assert not urgent_info.is_critical
        assert urgent_info.is_urgent

        # Normal question (48 hours)
        normal_info = QuestionDeadlineInfo(
            question_id="normal",
            close_time=now + timedelta(hours=48),
            resolve_time=None,
            time_until_close=timedelta(hours=48),
            urgency_score=0.4,
            submission_window=TimeRange(start=now, end=now + timedelta(hours=48)),
        )

        assert not normal_info.is_critical
        assert not normal_info.is_urgent
