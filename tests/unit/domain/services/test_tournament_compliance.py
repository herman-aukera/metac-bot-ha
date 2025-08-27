"""
Tests for tournament rule compliance and transparency requirements.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.domain.services.tournament_compliance_validator import (
    TournamentComplianceValidator, ComplianceIssue, ComplianceReport
)
from src.domain.services.tournament_rule_compliance_monitor import (
    TournamentRuleComplianceMonitor, ComplianceViolationType, ComplianceViolation
)
from src.domain.entities.prediction import Prediction
from src.domain.entities.question import Question, QuestionType


class TestTournamentCompliance:
    """Test tournament compliance validation and monitoring."""

    def setup_method(self):
        """Set up test environment."""
        self.compliance_validator = TournamentComplianceValidator()
        self.compliance_monitor = TournamentRuleComplianceMonitor()

        # Create mock question
        self.mock_question = Mock(spec=Question)
        self.mock_question.id = "compliance-test-123"
        self.mock_question.title = "Will this prediction be compliant?"
        self.mock_question.question_type = QuestionType.BINARY
        self.mock_question.close_time = datetime.now() + timedelta(days=30)

    def test_reasoning_transparency_validation(self):
        """Test validation of reasoning transparency requirements."""
        # Test compliant reasoning
        compliant_prediction = Mock(spec=Prediction)
        compliant_prediction.reasoning = """
        Based on my analysis of the available data:
        1. Historical trends show a 60% success rate for similar events
        2. Current market conditions suggest positive momentum
        3. Expert opinions are generally optimistic
        Therefore, I estimate a 65% probability.
        """

        report = self.compliance_validator.validate_reasoning_transparency(
            compliant_prediction, self.mock_question
        )

        assert report.is_compliant is True
        assert len(report.issues) == 0

        # Test non-compliant reasoning (too brief)
        non_compliant_prediction = Mock(spec=Prediction)
        non_compliant_prediction.reasoning = "I think it's likely."

        report = self.compliance_validator.validate_reasoning_transparency(
            non_compliant_prediction, self.mock_question
        )

        assert report.is_compliant is False
        assert len(report.issues) > 0
        assert any("insufficient detail" in issue.description.lower() for issue in report.issues)
    def test_human_intervention_detection(self):
        """Test detection of human intervention violations."""
        # Test automated prediction (compliant)
        automated_metadata = {
            "agent_type": "automated",
            "human_review": False,
            "manual_adjustments": [],
            "intervention_flags": []
        }

        violation = self.compliance_monitor.check_human_intervention(
            prediction_metadata=automated_metadata
        )

        assert violation is None  # No violation for automated prediction

        # Test human-assisted prediction (violation)
        human_assisted_metadata = {
            "agent_type": "automated",
            "human_review": True,
            "manual_adjustments": ["probability adjusted from 0.7 to 0.8"],
            "intervention_flags": ["human_override"]
        }

        violation = self.compliance_monitor.check_human_intervention(
            prediction_metadata=human_assisted_metadata
        )

        assert violation is not None
        assert violation.violation_type == ComplianceViolationType.HUMAN_INTERVENTION
        assert "human review" in violation.description.lower()

    def test_automated_decision_validation(self):
        """Test validation of automated decision-making requirements."""
        # Test fully automated process
        automated_process = {
            "research_method": "automated_web_search",
            "analysis_method": "llm_analysis",
            "prediction_method": "algorithmic",
            "human_involvement": "none",
            "decision_points": [
                {"step": "research", "automated": True},
                {"step": "analysis", "automated": True},
                {"step": "prediction", "automated": True}
            ]
        }

        report = self.compliance_validator.validate_automated_decision_making(
            process_metadata=automated_process
        )

        assert report.is_compliant is True
        assert len(report.issues) == 0

        # Test process with human decision points
        human_involved_process = {
            "research_method": "automated_web_search",
            "analysis_method": "human_review",
            "prediction_method": "algorithmic",
            "human_involvement": "analysis_review",
            "decision_points": [
                {"step": "research", "automated": True},
                {"step": "analysis", "automated": False, "human_reviewer": "analyst_1"},
                {"step": "prediction", "automated": True}
            ]
        }

        report = self.compliance_validator.validate_automated_decision_making(
            process_metadata=human_involved_process
        )

        assert report.is_compliant is False
        assert len(report.issues) > 0
        assert any("human involvement" in issue.description.lower() for issue in report.issues)

    def test_submission_timing_compliance(self):
        """Test compliance with submission timing requirements."""
        # Test timely submission (compliant)
        timely_submission = {
            "question_close_time": datetime.now() + timedelta(hours=6),
            "submission_time": datetime.now(),
            "processing_start_time": datetime.now() - timedelta(minutes=30),
            "research_duration": timedelta(minutes=20),
            "analysis_duration": timedelta(minutes=10)
        }

        violation = self.compliance_monitor.check_submission_timing(
            submission_metadata=timely_submission
        )

        assert violation is None  # No timing violation

        # Test late submission (violation)
        late_submission = {
            "question_close_time": datetime.now() - timedelta(hours=1),  # Already closed
            "submission_time": datetime.now(),
            "processing_start_time": datetime.now() - timedelta(minutes=30),
            "research_duration": timedelta(minutes=20),
            "analysis_duration": timedelta(minutes=10)
        }

        violation = self.compliance_monitor.check_submission_timing(
            submission_metadata=late_submission
        )

        assert violation is not None
        assert violation.violation_type == ComplianceViolationType.LATE_SUBMISSION
        assert "after close time" in violation.description.lower()
    def test_data_source_compliance(self):
        """Test compliance with data source restrictions."""
        # Test compliant data sources
        compliant_sources = {
            "sources_used": [
                {"type": "web_search", "url": "https://example.com/public-data"},
                {"type": "api", "service": "public_news_api"},
                {"type": "database", "name": "public_economic_data"}
            ],
            "restricted_sources": [],
            "private_information": False
        }

        report = self.compliance_validator.validate_data_source_compliance(
            data_sources=compliant_sources
        )

        assert report.is_compliant is True
        assert len(report.issues) == 0

        # Test non-compliant data sources
        non_compliant_sources = {
            "sources_used": [
                {"type": "web_search", "url": "https://example.com/public-data"},
                {"type": "private_database", "name": "internal_company_data"},
                {"type": "insider_information", "source": "confidential"}
            ],
            "restricted_sources": ["internal_company_data"],
            "private_information": True
        }

        report = self.compliance_validator.validate_data_source_compliance(
            data_sources=non_compliant_sources
        )

        assert report.is_compliant is False
        assert len(report.issues) > 0
        assert any("private" in issue.description.lower() or "restricted" in issue.description.lower()
                  for issue in report.issues)

    def test_prediction_format_compliance(self):
        """Test compliance with prediction format requirements."""
        # Test compliant prediction format
        compliant_prediction = Mock(spec=Prediction)
        compliant_prediction.probability = 0.75
        compliant_prediction.confidence_interval = (0.65, 0.85)
        compliant_prediction.reasoning = "Detailed analysis with multiple factors considered."
        compliant_prediction.metadata = {
            "format_version": "1.0",
            "required_fields": ["probability", "reasoning"],
            "optional_fields": ["confidence_interval"]
        }

        report = self.compliance_validator.validate_prediction_format(
            compliant_prediction, self.mock_question
        )

        assert report.is_compliant is True
        assert len(report.issues) == 0

        # Test non-compliant prediction format
        non_compliant_prediction = Mock(spec=Prediction)
        non_compliant_prediction.probability = None  # Missing required field
        non_compliant_prediction.reasoning = ""  # Empty reasoning
        non_compliant_prediction.metadata = {
            "format_version": "0.5",  # Outdated format
            "required_fields": ["probability", "reasoning"],
            "missing_fields": ["probability"]
        }

        report = self.compliance_validator.validate_prediction_format(
            non_compliant_prediction, self.mock_question
        )

        assert report.is_compliant is False
        assert len(report.issues) > 0
        assert any("missing" in issue.description.lower() or "required" in issue.description.lower()
                  for issue in report.issues)

    def test_comprehensive_compliance_check(self):
        """Test comprehensive compliance validation."""
        # Create comprehensive test data
        prediction = Mock(spec=Prediction)
        prediction.probability = 0.68
        prediction.reasoning = """
        My analysis considers multiple factors:
        1. Historical data shows 65% success rate for similar events
        2. Current market indicators suggest positive trends
        3. Expert consensus leans toward favorable outcome
        4. Risk factors are manageable within current context
        Based on this analysis, I estimate 68% probability.
        """

        prediction_metadata = {
            "agent_type": "automated",
            "human_review": False,
            "data_sources": [
                {"type": "web_search", "url": "https://public-data.com"},
                {"type": "api", "service": "news_api"}
            ],
            "processing_time": timedelta(minutes=15),
            "submission_time": datetime.now(),
            "format_version": "1.0"
        }

        # Run comprehensive compliance check
        report = self.compliance_validator.run_comprehensive_compliance_check(
            prediction=prediction,
            question=self.mock_question,
            metadata=prediction_metadata
        )

        # Should pass all compliance checks
        assert report.is_compliant is True
        assert len(report.issues) == 0
        assert "transparency" in report.compliance_areas_checked
        assert "automation" in report.compliance_areas_checked
        assert "data_sources" in report.compliance_areas_checked

    def teardown_method(self):
        """Clean up test environment."""
        pass
