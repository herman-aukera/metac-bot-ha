"""
Tests for Multi-Stage Validation Pipeline Implementation.
Tests task 4 requirements with all three stages integrated.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.domain.services.forecasting_stage_service import ForecastResult
from src.domain.services.multi_stage_research_pipeline import ResearchStageResult
from src.domain.services.multi_stage_validation_pipeline import (
    MultiStageResult,
    MultiStageValidationPipeline,
)
from src.domain.services.validation_stage_service import ValidationResult


class TestMultiStageValidationPipeline:
    """Test the complete multi-stage validation pipeline."""

    @pytest.fixture
    def mock_tri_model_router(self):
        """Mock tri-model router for testing."""
        router = Mock()

        # Mock models
        router.models = {"nano": AsyncMock(), "mini": AsyncMock(), "full": AsyncMock()}

        # Mock model responses
        router.models["nano"].invoke = AsyncMock(
            return_value="Validation passed. Quality score: 8/10"
        )
        router.models["mini"].invoke = AsyncMock(
            return_value="Research synthesis with [Source: Test, 2024]"
        )
        router.models["full"].invoke = AsyncMock(
            return_value="Probability: 65%. Confidence: Medium. Base rate considered."
        )

        return router

    @pytest.fixture
    def mock_tournament_asknews(self):
        """Mock tournament AskNews client."""
        client = Mock()
        client.get_news_research = AsyncMock(
            return_value="Recent news about the topic with citations."
        )
        return client

    @pytest.fixture
    def pipeline(self, mock_tri_model_router, mock_tournament_asknews):
        """Create pipeline instance for testing."""
        return MultiStageValidationPipeline(
            tri_model_router=mock_tri_model_router,
            tournament_asknews=mock_tournament_asknews,
        )

    @pytest.mark.asyncio
    async def test_complete_pipeline_binary_question(self, pipeline):
        """Test complete pipeline processing for binary question."""

        question = "Will AI achieve AGI by 2030?"
        question_type = "binary"
        context = {
            "background_info": "AI development context",
            "resolution_criteria": "AGI definition criteria",
        }

        result = await pipeline.process_question(question, question_type, context)

        # Verify result structure
        assert isinstance(result, MultiStageResult)
        assert result.question == question
        assert result.question_type == question_type

        # Verify all stages executed
        assert result.research_result is not None
        assert result.validation_result is not None
        assert result.forecast_result is not None

        # Verify metrics
        assert isinstance(result.total_execution_time, float)
        assert result.total_execution_time > 0
        assert isinstance(result.total_cost, float)
        assert isinstance(result.quality_score, float)
        assert isinstance(result.tournament_compliant, bool)

    @pytest.mark.asyncio
    async def test_complete_pipeline_multiple_choice_question(self, pipeline):
        """Test complete pipeline processing for multiple choice question."""

        question = "Which technology will have the biggest impact in 2025?"
        question_type = "multiple_choice"
        context = {
            "options": ["AI", "Quantum Computing", "Biotechnology", "Renewable Energy"],
            "background_info": "Technology impact assessment",
        }

        result = await pipeline.process_question(question, question_type, context)

        # Verify result structure
        assert isinstance(result, MultiStageResult)
        assert result.question_type == question_type

        # Verify forecast format for multiple choice
        if isinstance(result.final_forecast, dict):
            # Should have probabilities for each option
            assert len(result.final_forecast) > 0

    @pytest.mark.asyncio
    async def test_complete_pipeline_numeric_question(self, pipeline):
        """Test complete pipeline processing for numeric question."""

        question = "What will be the global temperature increase by 2030?"
        question_type = "numeric"
        context = {
            "unit_of_measure": "degrees Celsius",
            "lower_bound": 0.5,
            "upper_bound": 3.0,
        }

        result = await pipeline.process_question(question, question_type, context)

        # Verify result structure
        assert isinstance(result, MultiStageResult)
        assert result.question_type == question_type

    @pytest.mark.asyncio
    async def test_pipeline_with_research_failure(self, pipeline):
        """Test pipeline behavior when research stage fails."""

        # Mock research pipeline to fail
        pipeline.research_pipeline.execute_research_pipeline = AsyncMock(
            return_value={
                "success": False,
                "error": "Research failed",
                "final_research": "",
                "total_cost": 0.0,
            }
        )

        question = "Test question with research failure"
        question_type = "binary"

        result = await pipeline.process_question(question, question_type)

        # Pipeline should still complete but with degraded quality
        assert isinstance(result, MultiStageResult)
        assert not result.pipeline_success or result.quality_score < 0.6

    @pytest.mark.asyncio
    async def test_pipeline_with_validation_failure(self, pipeline):
        """Test pipeline behavior when validation stage fails."""

        # Mock validation service to fail
        pipeline.validation_service.validate_content = AsyncMock(
            return_value=ValidationResult(
                is_valid=False,
                quality_score=0.2,
                evidence_traceability_score=0.1,
                hallucination_detected=True,
                logical_consistency_score=0.3,
                issues_identified=["Major validation issues"],
                recommendations=["Fix validation issues"],
                confidence_level="low",
                execution_time=1.0,
                cost_estimate=0.001,
            )
        )

        question = "Test question with validation failure"
        question_type = "binary"

        result = await pipeline.process_question(question, question_type)

        # Pipeline should complete but mark validation issues
        assert isinstance(result, MultiStageResult)
        assert not result.validation_result.is_valid
        assert result.validation_result.hallucination_detected

    @pytest.mark.asyncio
    async def test_pipeline_with_forecasting_failure(self, pipeline):
        """Test pipeline behavior when forecasting stage fails."""

        # Mock forecasting service to fail
        pipeline.forecasting_service.generate_forecast = AsyncMock(
            return_value=ForecastResult(
                forecast_type="binary",
                prediction=0.5,
                confidence_score=0.0,
                uncertainty_bounds=None,
                calibration_score=0.0,
                overconfidence_detected=True,
                quality_validation_passed=False,
                tournament_compliant=False,
                reasoning="Forecasting failed",
                execution_time=1.0,
                cost_estimate=0.01,
                model_used="none",
            )
        )

        question = "Test question with forecasting failure"
        question_type = "binary"

        result = await pipeline.process_question(question, question_type)

        # Pipeline should complete but mark forecasting issues
        assert isinstance(result, MultiStageResult)
        assert not result.forecast_result.quality_validation_passed
        assert not result.forecast_result.tournament_compliant

    @pytest.mark.asyncio
    async def test_pipeline_cost_tracking(self, pipeline):
        """Test that pipeline properly tracks costs across all stages."""

        question = "Cost tracking test question"
        question_type = "binary"

        result = await pipeline.process_question(question, question_type)

        # Verify cost tracking
        assert isinstance(result.total_cost, float)
        assert result.total_cost >= 0

        # Cost should be sum of all stages
        expected_cost = (
            result.research_result.cost_estimate
            + result.validation_result.cost_estimate
            + result.forecast_result.cost_estimate
        )

        assert abs(result.total_cost - expected_cost) < 0.001

    @pytest.mark.asyncio
    async def test_pipeline_quality_scoring(self, pipeline):
        """Test quality scoring calculation across all stages."""

        question = "Quality scoring test question"
        question_type = "binary"

        result = await pipeline.process_question(question, question_type)

        # Verify quality score calculation
        assert isinstance(result.quality_score, float)
        assert 0.0 <= result.quality_score <= 1.0

        # Quality score should reflect all stages
        # (This is a weighted average of research, validation, and forecasting quality)

    @pytest.mark.asyncio
    async def test_pipeline_tournament_compliance_checking(self, pipeline):
        """Test tournament compliance checking across all stages."""

        question = "Tournament compliance test question"
        question_type = "binary"

        result = await pipeline.process_question(question, question_type)

        # Verify tournament compliance checking
        assert isinstance(result.tournament_compliant, bool)

        # Tournament compliance should consider all stages
        if result.tournament_compliant:
            # If compliant, validation should be valid and forecasting should be compliant
            assert result.validation_result.is_valid
            assert result.forecast_result.tournament_compliant

    @pytest.mark.asyncio
    async def test_pipeline_reasoning_compilation(self, pipeline):
        """Test final reasoning compilation from all stages."""

        question = "Reasoning compilation test question"
        question_type = "binary"

        result = await pipeline.process_question(question, question_type)

        # Verify reasoning compilation
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

        # Reasoning should include elements from all stages
        assert "Research Summary" in result.reasoning
        assert "Quality Validation" in result.reasoning
        assert "Forecasting Analysis" in result.reasoning
        assert "Final Assessment" in result.reasoning

    @pytest.mark.asyncio
    async def test_pipeline_health_check(self, pipeline):
        """Test pipeline health check functionality."""

        health_status = await pipeline.get_pipeline_health_check()

        # Verify health check structure
        assert isinstance(health_status, dict)
        assert "pipeline" in health_status
        assert "timestamp" in health_status
        assert "components" in health_status
        assert "overall_health" in health_status

        # Verify component checks
        components = health_status["components"]
        assert "research" in components
        assert "validation" in components
        assert "forecasting" in components

        # Each component should have status
        for component in components.values():
            assert "status" in component
            assert component["status"] in ["healthy", "unhealthy"]

    def test_pipeline_configuration(self, pipeline):
        """Test pipeline configuration retrieval."""

        config = pipeline.get_pipeline_configuration()

        # Verify configuration structure
        assert isinstance(config, dict)
        assert "pipeline" in config
        assert "stages" in config
        assert "models_used" in config
        assert "cost_optimization" in config
        assert "quality_thresholds" in config
        assert "tournament_compliance" in config

        # Verify stages
        stages = config["stages"]
        assert "research_with_asknews_and_gpt5_mini" in stages
        assert "validation_with_gpt5_nano" in stages
        assert "forecasting_with_gpt5_full" in stages

        # Verify models
        models = config["models_used"]
        assert "research" in models
        assert "validation" in models
        assert "forecasting" in models

        # Verify cost optimization
        cost_opt = config["cost_optimization"]
        assert "asknews_free_via_metaculusq4" in cost_opt
        assert "free_model_fallbacks" in cost_opt
        assert cost_opt["asknews_free_via_metaculusq4"] is True

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, pipeline):
        """Test pipeline error handling and recovery."""

        # Mock all services to raise exceptions
        pipeline.research_pipeline.execute_research_pipeline = AsyncMock(
            side_effect=Exception("Research service error")
        )

        question = "Error handling test question"
        question_type = "binary"

        result = await pipeline.process_question(question, question_type)

        # Pipeline should handle errors gracefully
        assert isinstance(result, MultiStageResult)
        assert not result.pipeline_success
        assert result.quality_score == 0.0
        assert not result.tournament_compliant
        assert "error" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_pipeline_performance_metrics(self, pipeline):
        """Test pipeline performance metrics collection."""

        question = "Performance metrics test question"
        question_type = "binary"

        start_time = datetime.now()
        result = await pipeline.process_question(question, question_type)
        end_time = datetime.now()

        # Verify performance metrics
        assert isinstance(result.total_execution_time, float)
        assert result.total_execution_time > 0

        # Execution time should be reasonable
        actual_time = (end_time - start_time).total_seconds()
        assert (
            abs(result.total_execution_time - actual_time) < 1.0
        )  # Within 1 second tolerance

    @pytest.mark.asyncio
    async def test_pipeline_with_different_contexts(self, pipeline):
        """Test pipeline with various context configurations."""

        contexts = [
            # Binary with minimal context
            {"question_type": "binary", "context": {}},
            # Multiple choice with options
            {
                "question_type": "multiple_choice",
                "context": {
                    "options": ["Option A", "Option B", "Option C"],
                    "background_info": "Test background",
                },
            },
            # Numeric with bounds
            {
                "question_type": "numeric",
                "context": {
                    "unit_of_measure": "units",
                    "lower_bound": 0,
                    "upper_bound": 100,
                },
            },
        ]

        for test_case in contexts:
            question = f"Test question for {test_case['question_type']}"
            result = await pipeline.process_question(
                question, test_case["question_type"], test_case["context"]
            )

            # Each should complete successfully
            assert isinstance(result, MultiStageResult)
            assert result.question_type == test_case["question_type"]


if __name__ == "__main__":
    pytest.main([__file__])
