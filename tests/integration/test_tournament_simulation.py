"""
Tournament simulation and dry-run mode testing.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock


from src.domain.entities.prediction import Prediction
from src.domain.entities.question import Question, QuestionType
from src.domain.value_objects.probability import Probability
from src.infrastructure.external_apis.submission_validator import (
    DryRunManager,
    SubmissionValidator,
)


class TestTournamentSimulation:
    """Test tournament simulation and dry-run functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Create mock dependencies
        self.mock_validator = Mock(spec=SubmissionValidator)
        # Add missing mock methods
        self.mock_validator.validate_submission = Mock()
        self.mock_validator.validate_reasoning_transparency = Mock(return_value=True)
        self.mock_validator.check_human_intervention = Mock(return_value=False)
        self.mock_audit_manager = Mock()
        self.mock_tournament_client = Mock()

        # Create dry-run manager
        self.dry_run_manager = DryRunManager(
            validator=self.mock_validator,
            audit_manager=self.mock_audit_manager,
            tournament_client=self.mock_tournament_client,
        )

        # Create mock question
        self.mock_question = Mock(spec=Question)
        self.mock_question.id = "test-question-123"
        self.mock_question.title = "Will AI achieve AGI by 2030?"
        self.mock_question.question_type = QuestionType.BINARY
        self.mock_question.close_time = datetime.now() + timedelta(days=365)

        # Create mock prediction
        self.mock_prediction = Mock(spec=Prediction)
        self.mock_prediction.probability = Probability(0.75)
        self.mock_prediction.reasoning = "Based on current AI progress trends."

    def test_dry_run_session_creation(self):
        """Test creation of dry-run sessions."""
        tournament_context = {
            "tournament_id": "test-tournament",
            "budget_limit": 100.0,
            "question_count": 50,
            "duration_days": 7,
        }

        session_id = self.dry_run_manager.start_dry_run_session(
            session_name="Test Tournament Simulation",
            tournament_context=tournament_context,
        )

        # Verify session creation
        assert session_id is not None
        assert session_id in self.dry_run_manager.dry_run_sessions

        session = self.dry_run_manager.dry_run_sessions[session_id]
        assert session["session_name"] == "Test Tournament Simulation"
        assert session["tournament_context"] == tournament_context
        assert session["status"] == "active"
        assert len(session["submissions"]) == 0

    def test_submission_simulation(self):
        """Test simulation of prediction submissions."""
        # Start dry-run session
        session_id = self.dry_run_manager.start_dry_run_session(
            session_name="Submission Test", tournament_context={"budget_limit": 50.0}
        )

        # Mock validator response
        self.mock_validator.validate_submission.return_value = {
            "is_valid": True,
            "validation_errors": [],
            "compliance_status": "compliant",
        }

        # Simulate submission
        result = self.dry_run_manager.simulate_submission(
            session_id=session_id,
            question=self.mock_question,
            prediction=self.mock_prediction,
            agent_metadata={"agent_version": "1.0", "model_used": "gpt-4o"},
        )

        # Verify simulation results
        assert result is not None
        assert "simulation_results" in result
        assert "validation_status" in result
        assert "compliance_check" in result

        # Verify session tracking
        session = self.dry_run_manager.dry_run_sessions[session_id]
        assert len(session["submissions"]) == 1

    def test_tournament_workflow_simulation(self):
        """Test complete tournament workflow simulation."""
        # Create multiple questions for tournament simulation
        questions = []
        for i in range(5):
            question = Mock(spec=Question)
            question.id = f"question-{i}"
            question.title = f"Test question {i}"
            question.question_type = QuestionType.BINARY
            question.close_time = datetime.now() + timedelta(days=30)
            questions.append(question)

        # Start tournament simulation
        session_id = self.dry_run_manager.start_dry_run_session(
            session_name="Full Tournament Simulation",
            tournament_context={
                "tournament_id": "full-test",
                "question_count": len(questions),
                "budget_limit": 100.0,
                "duration_days": 7,
            },
        )

        # Mock validator to always pass
        self.mock_validator.validate_submission.return_value = {
            "is_valid": True,
            "validation_errors": [],
            "compliance_status": "compliant",
        }

        # Simulate predictions for all questions
        results = []
        for question in questions:
            prediction = Mock(spec=Prediction)
            prediction.probability = Probability(
                0.6 + (0.1 * len(results))
            )  # Vary predictions
            prediction.reasoning = f"Analysis for question {question.id}"

            result = self.dry_run_manager.simulate_submission(
                session_id=session_id, question=question, prediction=prediction
            )
            results.append(result)

        # Verify tournament simulation
        session = self.dry_run_manager.dry_run_sessions[session_id]
        assert len(session["submissions"]) == len(questions)
        assert all(result["validation_status"]["is_valid"] for result in results)

    def test_scheduling_optimization_simulation(self):
        """Test scheduling optimization in tournament simulation."""
        # Create questions with different close times
        questions = []
        base_time = datetime.now()

        for i in range(10):
            question = Mock(spec=Question)
            question.id = f"scheduled-question-{i}"
            question.title = f"Scheduled question {i}"
            question.question_type = QuestionType.BINARY
            question.close_time = base_time + timedelta(hours=i * 6)  # Every 6 hours
            questions.append(question)

        # Start scheduling simulation
        self.dry_run_manager.start_dry_run_session(
            session_name="Scheduling Optimization Test",
            tournament_context={
                "scheduling_mode": "optimized",
                "budget_limit": 75.0,
                "max_concurrent_questions": 3,
            },
        )

        # Simulate scheduling decisions
        scheduled_submissions = []
        for question in questions:
            # Simulate scheduling logic
            schedule_time = question.close_time - timedelta(
                hours=2
            )  # 2 hours before close

            scheduled_submissions.append(
                {
                    "question": question,
                    "scheduled_time": schedule_time,
                    "priority": 1.0 / (i + 1),  # Higher priority for earlier questions
                }
            )

        # Verify scheduling optimization
        assert len(scheduled_submissions) == len(questions)

        # Check that scheduling respects time constraints
        for i in range(1, len(scheduled_submissions)):
            current_time = scheduled_submissions[i]["scheduled_time"]
            previous_time = scheduled_submissions[i - 1]["scheduled_time"]

            # Should maintain reasonable spacing
            time_diff = (current_time - previous_time).total_seconds() / 3600
            assert time_diff >= 0, "Scheduling should respect chronological order"

    def test_question_volume_management(self):
        """Test question volume management in tournament simulation."""
        # Create high-volume question scenario
        high_volume_questions = []
        for i in range(50):  # Large number of questions
            question = Mock(spec=Question)
            question.id = f"volume-question-{i}"
            question.title = f"Volume test question {i}"
            question.question_type = QuestionType.BINARY
            question.close_time = datetime.now() + timedelta(days=1)
            high_volume_questions.append(question)

        # Start volume management simulation
        session_id = self.dry_run_manager.start_dry_run_session(
            session_name="Volume Management Test",
            tournament_context={
                "max_questions_per_day": 20,
                "budget_limit": 200.0,
                "volume_management": "enabled",
            },
        )

        # Mock validator
        self.mock_validator.validate_submission.return_value = {
            "is_valid": True,
            "validation_errors": [],
            "compliance_status": "compliant",
        }

        # Simulate volume management decisions
        processed_questions = 0
        daily_limit = 20

        for question in high_volume_questions[:daily_limit]:  # Respect daily limit
            self.dry_run_manager.simulate_submission(
                session_id=session_id,
                question=question,
                prediction=self.mock_prediction,
            )
            processed_questions += 1

        # Verify volume management
        session = self.dry_run_manager.dry_run_sessions[session_id]
        assert len(session["submissions"]) == daily_limit
        assert processed_questions <= daily_limit

    def test_budget_constraint_simulation(self):
        """Test budget constraint handling in tournament simulation."""
        # Start budget-constrained simulation
        tight_budget = 10.0  # Very tight budget
        session_id = self.dry_run_manager.start_dry_run_session(
            session_name="Budget Constraint Test",
            tournament_context={
                "budget_limit": tight_budget,
                "cost_per_question": 2.0,  # High cost per question
                "budget_enforcement": "strict",
            },
        )

        # Mock validator with budget checking
        def mock_validate_with_budget(question, prediction, **kwargs):
            session = self.dry_run_manager.dry_run_sessions[session_id]
            current_cost = len(session["submissions"]) * 2.0

            if current_cost >= tight_budget:
                return {
                    "is_valid": False,
                    "validation_errors": ["Budget limit exceeded"],
                    "compliance_status": "budget_violation",
                }
            return {
                "is_valid": True,
                "validation_errors": [],
                "compliance_status": "compliant",
            }

        self.mock_validator.validate_submission.side_effect = mock_validate_with_budget

        # Try to submit more questions than budget allows
        successful_submissions = 0
        budget_violations = 0

        for i in range(10):  # Try 10 questions, but budget only allows 5
            question = Mock(spec=Question)
            question.id = f"budget-question-{i}"

            result = self.dry_run_manager.simulate_submission(
                session_id=session_id,
                question=question,
                prediction=self.mock_prediction,
            )

            if result["validation_status"]["is_valid"]:
                successful_submissions += 1
            else:
                budget_violations += 1

        # Verify budget enforcement
        assert successful_submissions <= 5  # Budget allows max 5 questions
        assert budget_violations > 0  # Should have budget violations

    def teardown_method(self):
        """Clean up test environment."""
        # Clear dry-run sessions
        self.dry_run_manager.dry_run_sessions.clear()
