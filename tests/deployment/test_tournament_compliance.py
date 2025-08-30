"""
Tournament compliance validation test.

This test verifies that the bot meets all tournament requirements and compliance rules
before deployment to ensure it can participate without violations.
"""

import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

import pytest

from src.infrastructure.config.settings import Config
from src.agents.ensemble_agent import EnsembleAgent
from src.domain.entities.question import Question, QuestionType
from src.domain.entities.forecast import Forecast
from src.domain.services.tournament_compliance_validator import (
    TournamentComplianceValidator,
)
from src.domain.services.tournament_rule_compliance_monitor import (
    TournamentRuleComplianceMonitor,
)


class TestTournamentCompliance:
    """Test tournament compliance validation for deployment readiness."""

    @pytest.fixture
    def tournament_config(self):
        """Create tournament-specific configuration."""
        return {
            "tournament": {
                "id": 32813,
                "name": "AI Forecasting Tournament",
                "max_questions": 50,
                "dry_run": False,
                "submit_predictions": True,
                "compliance_mode": "strict",
            },
            "llm": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "api_key": "test-key",
                "temperature": 0.1,
                "max_tokens": 2000,
                "timeout": 30,
                "structured_output": True,
            },
            "agent": {
                "max_iterations": 5,
                "timeout": 300,
                "confidence_threshold": 0.6,
                "reasoning_transparency": True,
                "human_intervention": False,
                "automated_decision_making": True,
            },
            "compliance": {
                "reasoning_transparency": True,
                "data_source_validation": True,
                "prediction_format_validation": True,
                "submission_timing_validation": True,
                "human_intervention_check": True,
            },
        }

    @pytest.fixture
    def sample_tournament_question(self):
        """Create a sample tournament question."""
        return Question.create_new(
            metaculus_id=98765,
            title="Will global CO2 emissions decrease by 5% in 2025?",
            description="This question asks about the likelihood of a 5% decrease in global CO2 emissions during 2025 compared to 2024 levels.",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/98765",
            close_time=datetime.utcnow() + timedelta(days=30),
            categories=["Environment", "Climate"],
            url="https://metaculus.com/questions/98765/",
            tournament_id=32813,
        )

    @pytest.fixture
    def compliant_forecast(self, sample_tournament_question):
        """Create a compliant forecast for testing."""
        return Forecast(
            question_id=sample_tournament_question.id,
            prediction=0.25,
            confidence=0.78,
            reasoning="Based on comprehensive analysis of current climate policies, economic trends, and historical emission patterns, I estimate a 25% probability of achieving a 5% reduction in global CO2 emissions in 2025. Key factors supporting this prediction include: 1) Accelerating renewable energy adoption, 2) Strengthening climate policies in major economies, 3) Economic incentives for clean technology. However, challenges include continued fossil fuel dependence and potential economic disruptions.",
            method="ensemble_agent",
            sources=[
                "IEA World Energy Outlook 2024",
                "IPCC Climate Change Reports",
                "National climate policy databases",
                "Economic analysis of emission trends",
            ],
            reasoning_steps=[
                "Analyzed historical emission trends and patterns",
                "Reviewed current climate policies and their effectiveness",
                "Evaluated economic factors affecting emission reductions",
                "Assessed technological adoption rates for clean energy",
                "Synthesized expert opinions and forecasting models",
            ],
            metadata={
                "transparency_score": 0.95,
                "data_source_reliability": 0.92,
                "reasoning_depth": "comprehensive",
                "human_intervention": False,
                "automated_generation": True,
                "compliance_validated": True,
            },
        )

    def test_tournament_compliance_validator_initialization(self, tournament_config):
        """Test that tournament compliance validator can be initialized."""

        config = Config.from_dict(tournament_config)
        validator = TournamentComplianceValidator(config)

        assert validator is not None
        assert hasattr(validator, "validate_reasoning_transparency")
        assert hasattr(validator, "validate_automated_decision_making")
        assert hasattr(validator, "validate_data_source_compliance")
        assert hasattr(validator, "validate_prediction_format")
        assert hasattr(validator, "run_comprehensive_compliance_check")

    def test_tournament_rule_compliance_monitor_initialization(self, tournament_config):
        """Test that tournament rule compliance monitor can be initialized."""

        config = Config.from_dict(tournament_config)
        monitor = TournamentRuleComplianceMonitor(config)

        assert monitor is not None
        assert hasattr(monitor, "check_human_intervention")
        assert hasattr(monitor, "check_submission_timing")

    def test_reasoning_transparency_validation(
        self, tournament_config, compliant_forecast
    ):
        """Test reasoning transparency validation."""

        config = Config.from_dict(tournament_config)
        validator = TournamentComplianceValidator(config)

        # Test compliant forecast
        result = validator.validate_reasoning_transparency(compliant_forecast)

        assert result is not None
        assert result.get("compliant", False) is True
        assert result.get("transparency_score", 0) > 0.8
        assert "reasoning_length" in result
        assert "reasoning_steps_count" in result

        # Test non-compliant forecast (insufficient reasoning)
        non_compliant_forecast = Forecast(
            question_id=98765,
            prediction=0.5,
            confidence=0.6,
            reasoning="Maybe.",  # Insufficient reasoning
            method="simple",
            sources=[],
            reasoning_steps=[],
        )

        result = validator.validate_reasoning_transparency(non_compliant_forecast)
        assert result.get("compliant", True) is False
        assert result.get("transparency_score", 1.0) < 0.5

    def test_automated_decision_making_validation(
        self, tournament_config, compliant_forecast
    ):
        """Test automated decision making validation."""

        config = Config.from_dict(tournament_config)
        validator = TournamentComplianceValidator(config)

        # Test compliant forecast (fully automated)
        result = validator.validate_automated_decision_making(compliant_forecast)

        assert result is not None
        assert result.get("compliant", False) is True
        assert result.get("human_intervention", True) is False
        assert result.get("automated_generation", False) is True

    def test_data_source_compliance_validation(
        self, tournament_config, compliant_forecast
    ):
        """Test data source compliance validation."""

        config = Config.from_dict(tournament_config)
        validator = TournamentComplianceValidator(config)

        # Test compliant forecast (reliable sources)
        result = validator.validate_data_source_compliance(compliant_forecast)

        assert result is not None
        assert result.get("compliant", False) is True
        assert result.get("source_count", 0) > 0
        assert result.get("reliability_score", 0) > 0.8
        assert "validated_sources" in result

    def test_prediction_format_validation(self, tournament_config, compliant_forecast):
        """Test prediction format validation."""

        config = Config.from_dict(tournament_config)
        validator = TournamentComplianceValidator(config)

        # Test compliant forecast
        result = validator.validate_prediction_format(compliant_forecast)

        assert result is not None
        assert result.get("compliant", False) is True
        assert result.get("prediction_valid", False) is True
        assert result.get("confidence_valid", False) is True
        assert 0 <= result.get("prediction_value", -1) <= 1
        assert 0 <= result.get("confidence_value", -1) <= 1

    def test_comprehensive_compliance_check(
        self, tournament_config, compliant_forecast
    ):
        """Test comprehensive compliance check."""

        config = Config.from_dict(tournament_config)
        validator = TournamentComplianceValidator(config)

        # Run comprehensive check
        result = validator.run_comprehensive_compliance_check(compliant_forecast)

        assert result is not None
        assert "overall_compliant" in result
        assert "compliance_score" in result
        assert "validation_results" in result

        # Check individual validation results
        validation_results = result["validation_results"]
        assert "reasoning_transparency" in validation_results
        assert "automated_decision_making" in validation_results
        assert "data_source_compliance" in validation_results
        assert "prediction_format" in validation_results

        # Overall compliance should be true for compliant forecast
        assert result.get("overall_compliant", False) is True
        assert result.get("compliance_score", 0) > 0.8

    def test_human_intervention_check(self, tournament_config, compliant_forecast):
        """Test human intervention check."""

        config = Config.from_dict(tournament_config)
        monitor = TournamentRuleComplianceMonitor(config)

        # Test automated forecast (no human intervention)
        result = monitor.check_human_intervention(compliant_forecast)

        assert result is not None
        assert result.get("human_intervention_detected", True) is False
        assert (
            result.get("compliant", False) is True
        )  # No human intervention is compliant

    def test_submission_timing_check(
        self, tournament_config, sample_tournament_question
    ):
        """Test submission timing check."""

        config = Config.from_dict(tournament_config)
        monitor = TournamentRuleComplianceMonitor(config)

        # Test submission timing
        result = monitor.check_submission_timing(sample_tournament_question)

        assert result is not None
        assert "submission_allowed" in result
        assert "time_until_close" in result
        assert "compliant" in result

    @pytest.mark.asyncio
    async def test_end_to_end_compliance_validation(
        self, tournament_config, sample_tournament_question
    ):
        """Test end-to-end compliance validation workflow."""

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "ASKNEWS_CLIENT_ID": "test-client-id",
                "ASKNEWS_SECRET": "test-secret",
            },
        ):
            # Mock compliant LLM response
            mock_llm_response = {
                "reasoning": "Comprehensive analysis of climate data and policy trends suggests moderate probability of emission reduction targets being met in 2025. Key factors include renewable energy adoption rates, policy implementation effectiveness, and economic conditions.",
                "prediction": 0.28,
                "confidence": 0.75,
                "sources": [
                    "IEA reports",
                    "Climate policy databases",
                    "Economic analysis",
                ],
                "reasoning_steps": [
                    "Analyzed historical emission trends",
                    "Reviewed current climate policies",
                    "Evaluated economic factors",
                    "Synthesized expert predictions",
                ],
            }

            # Mock research results
            mock_research_results = [
                {
                    "title": "IEA World Energy Outlook 2024",
                    "url": "https://iea.org/reports/world-energy-outlook-2024",
                    "content": "Global CO2 emissions analysis and projections for 2025",
                    "relevance_score": 0.95,
                    "date": "2024-10-15",
                },
                {
                    "title": "Climate Policy Database",
                    "url": "https://climatepolicyinitiative.org/",
                    "content": "Comprehensive database of climate policies and their effectiveness",
                    "relevance_score": 0.88,
                    "date": "2024-11-20",
                },
            ]

            # Mock LLM and research clients
            mock_llm_client = AsyncMock()
            mock_llm_client.generate_response.return_value = str(mock_llm_response)

            mock_research_client = AsyncMock()
            mock_research_client.search.return_value = mock_research_results

            # Create agent and compliance validators
            with (
                patch(
                    "src.infrastructure.external_apis.llm_client.LLMClient",
                    return_value=mock_llm_client,
                ),
                patch(
                    "src.infrastructure.external_apis.tournament_asknews.TournamentAskNews",
                    return_value=mock_research_client,
                ),
            ):

                config = Config.from_dict(tournament_config)
                agent = EnsembleAgent("tournament-agent", config.llm_config)
                validator = TournamentComplianceValidator(config)
                monitor = TournamentRuleComplianceMonitor(config)

                # Mock the agent's internal clients
                agent.llm_client = mock_llm_client
                agent.research_client = mock_research_client

                # Generate forecast
                forecast = await agent.forecast(sample_tournament_question)

                # Validate compliance
                compliance_result = validator.run_comprehensive_compliance_check(
                    forecast
                )
                timing_result = monitor.check_submission_timing(
                    sample_tournament_question
                )
                intervention_result = monitor.check_human_intervention(forecast)

                # Verify forecast was generated
                assert forecast is not None

                # Verify compliance results
                assert compliance_result.get("overall_compliant", False) is True
                assert compliance_result.get("compliance_score", 0) > 0.7

                # Verify timing compliance
                assert timing_result.get("compliant", False) is True

                # Verify no human intervention
                assert intervention_result.get("compliant", False) is True
                assert (
                    intervention_result.get("human_intervention_detected", True)
                    is False
                )

    def test_non_compliant_forecast_detection(self, tournament_config):
        """Test detection of non-compliant forecasts."""

        config = Config.from_dict(tournament_config)
        validator = TournamentComplianceValidator(config)

        # Create non-compliant forecast
        non_compliant_forecast = Forecast(
            question_id=98765,
            prediction=1.5,  # Invalid probability > 1
            confidence=-0.1,  # Invalid confidence < 0
            reasoning="No reasoning provided.",  # Insufficient reasoning
            method="manual",  # Suggests human intervention
            sources=[],  # No sources
            reasoning_steps=[],  # No reasoning steps
            metadata={
                "human_intervention": True,  # Human intervention detected
                "automated_generation": False,
            },
        )

        # Run compliance check
        result = validator.run_comprehensive_compliance_check(non_compliant_forecast)

        # Should detect non-compliance
        assert result.get("overall_compliant", True) is False
        assert result.get("compliance_score", 1.0) < 0.5

        # Check specific validation failures
        validation_results = result["validation_results"]
        assert validation_results["prediction_format"]["compliant"] is False
        assert validation_results["reasoning_transparency"]["compliant"] is False
        assert validation_results["data_source_compliance"]["compliant"] is False

    def test_tournament_configuration_validation(self, tournament_config):
        """Test tournament configuration validation."""

        config = Config.from_dict(tournament_config)

        # Verify tournament-specific settings
        assert hasattr(config, "tournament_id") or "tournament" in tournament_config
        assert config.llm_config.get("structured_output", False) is True

        # Verify compliance settings
        compliance_config = tournament_config.get("compliance", {})
        assert compliance_config.get("reasoning_transparency", False) is True
        assert compliance_config.get("data_source_validation", False) is True
        assert compliance_config.get("prediction_format_validation", False) is True

    def test_deployment_readiness_checklist(self, tournament_config):
        """Test deployment readiness checklist for tournament compliance."""

        config = Config.from_dict(tournament_config)

        # Check 1: Compliance validators can be initialized
        try:
            validator = TournamentComplianceValidator(config)
            monitor = TournamentRuleComplianceMonitor(config)
            assert validator is not None
            assert monitor is not None
        except Exception as e:
            pytest.fail(f"Compliance validator initialization failed: {e}")

        # Check 2: All required validation methods exist
        required_validator_methods = [
            "validate_reasoning_transparency",
            "validate_automated_decision_making",
            "validate_data_source_compliance",
            "validate_prediction_format",
            "run_comprehensive_compliance_check",
        ]

        for method in required_validator_methods:
            assert hasattr(validator, method), f"Missing validator method: {method}"

        required_monitor_methods = [
            "check_human_intervention",
            "check_submission_timing",
        ]

        for method in required_monitor_methods:
            assert hasattr(monitor, method), f"Missing monitor method: {method}"

        # Check 3: Configuration supports tournament requirements
        assert config.llm_config is not None
        assert config.llm_config.get("provider") is not None
        assert config.llm_config.get("model") is not None

        # Check 4: Agent can be initialized for tournament
        try:
            agent = EnsembleAgent("tournament-agent", config.llm_config)
            assert agent is not None
        except Exception as e:
            pytest.fail(f"Tournament agent initialization failed: {e}")

    def test_compliance_error_handling(self, tournament_config):
        """Test compliance validation error handling."""

        config = Config.from_dict(tournament_config)
        validator = TournamentComplianceValidator(config)

        # Test with None forecast
        result = validator.run_comprehensive_compliance_check(None)
        assert result is not None
        assert result.get("overall_compliant", True) is False
        assert "error" in result

        # Test with malformed forecast
        malformed_forecast = Mock()
        malformed_forecast.prediction = "invalid"  # String instead of float
        malformed_forecast.confidence = None
        malformed_forecast.reasoning = None

        result = validator.validate_prediction_format(malformed_forecast)
        assert result is not None
        assert result.get("compliant", True) is False
