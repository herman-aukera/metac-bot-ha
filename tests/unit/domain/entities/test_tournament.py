"""Unit
tests for Tournament entity."""

import pytest
from datetime import datetime, timedelta

from src.domain.entities.tournament import Tournament, ScoringRules, ScoringMethod
from src.domain.entities.question import Question, QuestionType, QuestionCategory


class TestScoringRules:
    """Test cases for ScoringRules."""

    def test_scoring_rules_initialization_valid(self):
        """Test valid scoring rules initialization."""
        rules = ScoringRules(
            method=ScoringMethod.BRIER_SCORE,
            weight_by_question=True,
            bonus_for_early=True,
            penalty_for_late=False,
            minimum_confidence=0.1,
            maximum_submissions=3,
            resolution_bonus=0.05
        )

        assert rules.method == ScoringMethod.BRIER_SCORE
        assert rules.weight_by_question is True
        assert rules.bonus_for_early is True
        assert rules.penalty_for_late is False
        assert rules.minimum_confidence == 0.1
        assert rules.maximum_submissions == 3
        assert rules.resolution_bonus == 0.05

    def test_scoring_rules_validation_invalid_confidence(self):
        """Test scoring rules validation with invalid confidence."""
        with pytest.raises(ValueError, match="Minimum confidence must be between 0.0 and 1.0"):
            ScoringRules(
                method=ScoringMethod.BRIER_SCORE,
                minimum_confidence=1.5
            )

    def test_scoring_rules_validation_invalid_submissions(self):
        """Test scoring rules validation with invalid submissions."""
        with pytest.raises(ValueError, match="Maximum submissions must be at least 1"):
            ScoringRules(
                method=ScoringMethod.BRIER_SCORE,
                maximum_submissions=0
            )

    def test_scoring_rules_validation_negative_bonus(self):
        """Test scoring rules validation with negative bonus."""
        with pytest.raises(ValueError, match="Resolution bonus cannot be negative"):
            ScoringRules(
                method=ScoringMethod.BRIER_SCORE,
                resolution_bonus=-0.1
            )

    def test_scoring_rules_methods(self):
        """Test scoring rules utility methods."""
        rules = ScoringRules(
            method=ScoringMethod.BRIER_SCORE,
            minimum_confidence=0.1,
            maximum_submissions=3,
            bonus_for_early=True,
            penalty_for_late=True
        )

        assert rules.allows_multiple_submissions()
        assert rules.requires_minimum_confidence()
        assert rules.has_timing_incentives()

        single_submission_rules = ScoringRules(
            method=ScoringMethod.LOG_SCORE,
            minimum_confidence=0.0,
            maximum_submissions=1,
            bonus_for_early=False,
            penalty_for_late=False
        )

        assert not single_submission_rules.allows_multiple_submissions()
        assert not single_submission_rules.requires_minimum_confidence()
        assert not single_submission_rules.has_timing_incentives()


class TestTournament:
    """Test cases for Tournament entity."""

    def create_sample_question(self, question_id: int, deadline_offset_days: int = 30) -> Question:
        """Helper to create a sample question."""
        deadline = datetime.utcnow() + timedelta(days=deadline_offset_days)
        return Question(
            id=question_id,
            text=f"Sample question {question_id}",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background="Sample background",
            resolution_criteria="Sample criteria"
        )

    def test_tournament_initialization_valid(self):
        """Test valid tournament initialization."""
        start_date = datetime.utcnow() + timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=60)
        questions = [self.create_sample_question(1), self.create_sample_question(2)]
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        tournament = Tournament(
            id=1,
            name="Test Tournament",
            questions=questions,
            scoring_rules=scoring_rules,
            start_date=start_date,
            end_date=end_date,
            current_standings={"agent1": 0.8, "agent2": 0.6},
            description="A test tournament",
            max_participants=100
        )

        assert tournament.id == 1
        assert tournament.name == "Test Tournament"
        assert len(tournament.questions) == 2
        assert tournament.scoring_rules == scoring_rules
        assert tournament.start_date == start_date
        assert tournament.end_date == end_date
        assert tournament.current_standings == {"agent1": 0.8, "agent2": 0.6}
        assert tournament.description == "A test tournament"
        assert tournament.max_participants == 100

    def test_tournament_validation_invalid_id(self):
        """Test tournament validation with invalid ID."""
        start_date = datetime.utcnow() + timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=60)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        with pytest.raises(ValueError, match="Tournament ID must be positive"):
            Tournament(
                id=0,
                name="Test Tournament",
                questions=[],
                scoring_rules=scoring_rules,
                start_date=start_date,
                end_date=end_date,
                current_standings={}
            )

    def test_tournament_validation_empty_name(self):
        """Test tournament validation with empty name."""
        start_date = datetime.utcnow() + timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=60)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        with pytest.raises(ValueError, match="Tournament name cannot be empty"):
            Tournament(
                id=1,
                name="",
                questions=[],
                scoring_rules=scoring_rules,
                start_date=start_date,
                end_date=end_date,
                current_standings={}
            )

    def test_tournament_validation_invalid_dates(self):
        """Test tournament validation with invalid dates."""
        start_date = datetime.utcnow() + timedelta(days=60)
        end_date = datetime.utcnow() + timedelta(days=1)  # End before start
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        with pytest.raises(ValueError, match="Start date must be before end date"):
            Tournament(
                id=1,
                name="Test Tournament",
                questions=[],
                scoring_rules=scoring_rules,
                start_date=start_date,
                end_date=end_date,
                current_standings={}
            )

    def test_tournament_validation_question_deadline_after_end(self):
        """Test tournament validation with question deadline after tournament end."""
        start_date = datetime.utcnow() + timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=30)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        # Question deadline after tournament end
        question = self.create_sample_question(1, deadline_offset_days=60)

        with pytest.raises(ValueError, match="Question .* deadline is after tournament end"):
            Tournament(
                id=1,
                name="Test Tournament",
                questions=[question],
                scoring_rules=scoring_rules,
                start_date=start_date,
                end_date=end_date,
                current_standings={}
            )

    def test_tournament_status_methods(self):
        """Test tournament status checking methods."""
        # Active tournament
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=30)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        active_tournament = Tournament(
            id=1,
            name="Active Tournament",
            questions=[],
            scoring_rules=scoring_rules,
            start_date=start_date,
            end_date=end_date,
            current_standings={}
        )

        assert active_tournament.is_active()
        assert not active_tournament.is_upcoming()
        assert not active_tournament.is_finished()

        # Upcoming tournament
        upcoming_start = datetime.utcnow() + timedelta(days=1)
        upcoming_end = datetime.utcnow() + timedelta(days=30)

        upcoming_tournament = Tournament(
            id=2,
            name="Upcoming Tournament",
            questions=[],
            scoring_rules=scoring_rules,
            start_date=upcoming_start,
            end_date=upcoming_end,
            current_standings={}
        )

        assert not upcoming_tournament.is_active()
        assert upcoming_tournament.is_upcoming()
        assert not upcoming_tournament.is_finished()

        # Finished tournament
        finished_start = datetime.utcnow() - timedelta(days=30)
        finished_end = datetime.utcnow() - timedelta(days=1)

        finished_tournament = Tournament(
            id=3,
            name="Finished Tournament",
            questions=[],
            scoring_rules=scoring_rules,
            start_date=finished_start,
            end_date=finished_end,
            current_standings={}
        )

        assert not finished_tournament.is_active()
        assert not finished_tournament.is_upcoming()
        assert finished_tournament.is_finished()

    def test_tournament_time_remaining(self):
        """Test time remaining calculation."""
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(hours=12)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        tournament = Tournament(
            id=1,
            name="Test Tournament",
            questions=[],
            scoring_rules=scoring_rules,
            start_date=start_date,
            end_date=end_date,
            current_standings={}
        )

        time_remaining = tournament.time_remaining()
        assert 11.5 < time_remaining < 12.5  # Allow some tolerance

        # Finished tournament should return 0
        finished_tournament = Tournament(
            id=2,
            name="Finished Tournament",
            questions=[],
            scoring_rules=scoring_rules,
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow() - timedelta(days=1),
            current_standings={}
        )

        assert finished_tournament.time_remaining() == 0.0
    def test_tournament_question_filtering(self):
        """Test question filtering methods."""
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=60)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        # Create questions with different deadlines
        active_question = self.create_sample_question(1, deadline_offset_days=30)
        resolved_question = Question(
            id=2,
            text="Resolved question",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.TECHNOLOGY,
            deadline=datetime.utcnow() - timedelta(days=1),  # Past deadline
            background="Background",
            resolution_criteria="Criteria",
            scoring_weight=2.0
        )
        urgent_question = Question(
            id=3,
            text="Urgent question",
            question_type=QuestionType.NUMERIC,
            category=QuestionCategory.SCIENCE,
            deadline=datetime.utcnow() + timedelta(hours=12),  # Urgent
            background="Background",
            resolution_criteria="Criteria",
            min_value=0.0,
            max_value=100.0
        )

        tournament = Tournament(
            id=1,
            name="Test Tournament",
            questions=[active_question, resolved_question, urgent_question],
            scoring_rules=scoring_rules,
            start_date=start_date,
            end_date=end_date,
            current_standings={}
        )

        # Test active questions
        active_questions = tournament.get_active_questions()
        assert len(active_questions) == 2  # active_question and urgent_question
        assert active_question in active_questions
        assert urgent_question in active_questions

        # Test resolved questions
        resolved_questions = tournament.get_resolved_questions()
        assert len(resolved_questions) == 1
        assert resolved_question in resolved_questions

        # Test high value questions
        high_value_questions = tournament.get_high_value_questions(1.5)
        assert len(high_value_questions) == 1
        assert resolved_question in high_value_questions

        # Test questions by category
        tech_questions = tournament.get_questions_by_category(QuestionCategory.TECHNOLOGY)
        assert len(tech_questions) == 1
        assert resolved_question in tech_questions

        # Test questions by type
        binary_questions = tournament.get_questions_by_type(QuestionType.BINARY)
        assert len(binary_questions) == 2

        # Test urgent questions
        urgent_questions = tournament.get_urgent_questions(24.0)
        assert len(urgent_questions) == 1
        assert urgent_question in urgent_questions

    def test_tournament_participant_methods(self):
        """Test participant-related methods."""
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=60)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        standings = {
            "agent1": 0.85,
            "agent2": 0.72,
            "agent3": 0.68,
            "agent4": 0.91,
            "agent5": 0.55
        }

        tournament = Tournament(
            id=1,
            name="Test Tournament",
            questions=[],
            scoring_rules=scoring_rules,
            start_date=start_date,
            end_date=end_date,
            current_standings=standings
        )

        # Test participant count
        assert tournament.get_participant_count() == 5

        # Test participant rank
        assert tournament.get_participant_rank("agent4") == 1  # Highest score
        assert tournament.get_participant_rank("agent1") == 2
        assert tournament.get_participant_rank("agent5") == 5  # Lowest score
        assert tournament.get_participant_rank("nonexistent") is None

        # Test top participants
        top_3 = tournament.get_top_participants(3)
        assert len(top_3) == 3
        assert top_3[0] == ("agent4", 0.91)
        assert top_3[1] == ("agent1", 0.85)
        assert top_3[2] == ("agent2", 0.72)

        # Test participant score
        assert tournament.get_participant_score("agent3") == 0.68
        assert tournament.get_participant_score("nonexistent") is None

    def test_tournament_update_methods(self):
        """Test tournament update methods."""
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=60)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)
        original_question = self.create_sample_question(1)

        tournament = Tournament(
            id=1,
            name="Test Tournament",
            questions=[original_question],
            scoring_rules=scoring_rules,
            start_date=start_date,
            end_date=end_date,
            current_standings={"agent1": 0.5}
        )

        # Test update standings
        new_standings = {"agent1": 0.8, "agent2": 0.6}
        updated_tournament = tournament.update_standings(new_standings)

        assert updated_tournament.current_standings == new_standings
        assert updated_tournament.id == tournament.id  # Other fields unchanged
        assert updated_tournament.name == tournament.name

        # Test add question
        new_question = self.create_sample_question(2)
        tournament_with_question = tournament.add_question(new_question)

        assert len(tournament_with_question.questions) == 2
        assert new_question in tournament_with_question.questions
        assert original_question in tournament_with_question.questions

        # Test add question with invalid deadline
        invalid_question = self.create_sample_question(3, deadline_offset_days=90)  # After tournament end

        with pytest.raises(ValueError, match="Question deadline .* is after tournament end"):
            tournament.add_question(invalid_question)

    def test_tournament_statistics(self):
        """Test tournament statistics generation."""
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=60)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        # Create diverse questions
        questions = [
            self.create_sample_question(1, deadline_offset_days=30),  # Active
            Question(
                id=2,
                text="Resolved question",
                question_type=QuestionType.NUMERIC,
                category=QuestionCategory.TECHNOLOGY,
                deadline=datetime.utcnow() - timedelta(days=1),  # Resolved
                background="Background",
                resolution_criteria="Criteria",
                scoring_weight=3.0,  # High value
                min_value=0.0,
                max_value=100.0
            ),
            Question(
                id=3,
                text="Urgent question",
                question_type=QuestionType.MULTIPLE_CHOICE,
                category=QuestionCategory.SCIENCE,
                deadline=datetime.utcnow() + timedelta(hours=12),  # Urgent
                background="Background",
                resolution_criteria="Criteria",
                choices=["A", "B", "C"]
            )
        ]

        tournament = Tournament(
            id=1,
            name="Test Tournament",
            questions=questions,
            scoring_rules=scoring_rules,
            start_date=start_date,
            end_date=end_date,
            current_standings={"agent1": 0.8, "agent2": 0.6}
        )

        stats = tournament.get_tournament_stats()

        assert stats["total_questions"] == 3
        assert stats["active_questions"] == 2
        assert stats["resolved_questions"] == 1
        assert stats["participants"] == 2
        assert stats["is_active"] is True
        assert stats["high_value_questions"] == 1
        assert stats["urgent_questions"] == 1

        # Check question type distribution
        assert stats["question_types"]["binary"] == 1
        assert stats["question_types"]["numeric"] == 1
        assert stats["question_types"]["multiple_choice"] == 1

        # Check category distribution
        assert stats["question_categories"]["ai_development"] == 1
        assert stats["question_categories"]["technology"] == 1
        assert stats["question_categories"]["science"] == 1

    def test_tournament_summary(self):
        """Test tournament summary generation."""
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=60)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        tournament = Tournament(
            id=123,
            name="AI Forecasting Championship",
            questions=[self.create_sample_question(1), self.create_sample_question(2)],
            scoring_rules=scoring_rules,
            start_date=start_date,
            end_date=end_date,
            current_standings={"agent1": 0.8, "agent2": 0.6, "agent3": 0.7}
        )

        summary = tournament.to_summary()
        assert "Tournament 123:" in summary
        assert "AI Forecasting Championship" in summary
        assert "active" in summary
        assert "2 questions" in summary
        assert "3 participants" in summary

    def test_tournament_defaults(self):
        """Test tournament default values."""
        start_date = datetime.utcnow() + timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=60)
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)

        tournament = Tournament(
            id=1,
            name="Test Tournament",
            questions=[],
            scoring_rules=scoring_rules,
            start_date=start_date,
            end_date=end_date,
            current_standings={}
        )

        # Test default values
        assert tournament.metadata == {}
        assert tournament.description is None
        assert tournament.max_participants is None
        assert tournament.entry_requirements is None
        assert tournament.prize_structure is None

    def test_scoring_method_enum(self):
        """Test scoring method enum values."""
        assert ScoringMethod.BRIER_SCORE.value == "brier_score"
        assert ScoringMethod.LOG_SCORE.value == "log_score"
        assert ScoringMethod.QUADRATIC_SCORE.value == "quadratic_score"
        assert ScoringMethod.SPHERICAL_SCORE.value == "spherical_score"
        assert ScoringMethod.RELATIVE_SCORE.value == "relative_score"
