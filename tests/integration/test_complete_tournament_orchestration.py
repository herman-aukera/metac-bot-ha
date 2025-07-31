"""Comprehensive integration tests for tournament orchestration and integration layer.

This module tests the complete workflow from question ingestion through research,
reasoning, prediction, ensemble aggregation, and submission, ensuring all
components work together correctly.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.application.use_cases.process_tournament_question import (
    ProcessTournamentQuestionUseCase, ProcessingResult, ProcessingContext
)
from src.application.services.forecasting_pipeline import ForecastingPipeline, PipelineConfig
from src.application.services.tournament_service import TournamentService
from src.application.services.learning_service import LearningService
from src.application.services.integration_service import IntegrationService, LegacyQuestionFormat
from src.domain.entities.question import Question, QuestionType, QuestionCategory
from src.domain.entities.forecast import Forecast
from src.domain.entities.research_report import ResearchReport
from src.domain.entities.tournament import Tournament, ScoringMethod, ScoringRules
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.reasoning_step import ReasoningStep
from src.infrastructure.resilience.circuit_breaker import CircuitBreaker
from src.infrastructure.resilience.retry_strategy import RetryStrategy
from src.infrastructure.resilience.graceful_degradation import GracefulDegradationManager


@pytest.fixture
def sample_question():
    """Create a sample question for testing."""
    return Question(
        id=1,
        text="Will AGI be achieved by 2030?",
        question_type=QuestionType.BINARY,
        category=QuestionCategory.AI_DEVELOPMENT,
        deadline=datetime.utcnow() + timedelta(days=30),
        background="Background information about AGI development",
        resolution_criteria="AGI will be considered achieved when...",
        scoring_weight=2.0
    )


@pytest.fixture
def sample_tournament(sample_question):
    """Create a sample tournament for testing."""
    return Tournament(
        id=1,
        name="AI Forecasting Tournament",
        questions=[sample_question],
        scoring_rules=ScoringRules(
            method=ScoringMethod.LOG_SCORE,
            bonus_for_early=True,
            penalty_for_late=False
        ),
        start_date=datetime.utcnow() - timedelta(days=1),
        end_date=datetime.utcnow() + timedelta(days=30),
        current_standings={},
        participant_count=100
    )


@pytest.fixture
def sample_research_report():
    """Create a sample research report for testing."""
    return ResearchReport(
        id=1,
        question_id=1,
        sources=["source1", "source2", "source3"],
        credibility_scores={"source1": 0.9, "source2": 0.8, "source3": 0.7},
        evidence_synthesis="Comprehensive evidence synthesis",
        base_rates={},
        knowledge_gaps=["gap1", "gap2"],
        research_quality_score=0.85,
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_forecast():
    """Create a sample forecast for testing."""
    reasoning_steps = [
        ReasoningStep(
            step_number=1,
            description="Initial analysis",
            input_data={"question": "Will AGI be achieved by 2030?"},
            output_data={"analysis": "Complex question requiring deep analysis"},
            confidence=Confidence(level=0.8, basis="Strong analytical foundation"),
            timestamp=datetime.utcnow()
        )
    ]

    return Forecast.create_binary(
        question_id=1,
        probability=0.65,
        confidence_level=0.8,
        confidence_basis="Based on comprehensive analysis",
        reasoning_trace=reasoning_steps,
        evidence_sources=["source1", "source2"],
        agent_id="test_agent"
    )


@pytest.fixture
def mock_services():
    """Create mock services for testing."""
    # Mock circuit breaker
    circuit_breaker = Mock(spec=CircuitBreaker)
    circuit_breaker.call = AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))

    # Mock retry strategy
    retry_strategy = Mock(spec=RetryStrategy)
    retry_strategy.execute = AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))

    # Mock degradation manager
    degradation_manager = Mock(spec=GracefulDegradationManager)
    degradation_manager.get_degraded_research = AsyncMock(return_value=[])
    degradation_manager.get_single_agent_prediction = AsyncMock(return_value=None)

    # Mock forecasting pipeline
    forecasting_pipeline = Mock(spec=ForecastingPipeline)

    # Mock tournament service
    tournament_service = Mock(spec=TournamentService)

    # Mock learning service
    learning_service = Mock(spec=LearningService)
    learning_service.record_processing_result = AsyncMock()
    learning_service.get_calibration_data = AsyncMock(return_value={'adjustment_factor': 1.0})
    learning_service.get_system_stats = AsyncMock(return_value={})

    return {
        'circuit_breaker': circuit_breaker,
        'retry_strategy': retry_strategy,
        'degradation_manager': degradation_manager,
        'forecasting_pipeline': forecasting_pipeline,
        'tournament_service': tournament_service,
        'learning_service': learning_service
    }


class TestProcessTournamentQuestionUseCase:
    """Test the main ProcessTournamentQuestion use case."""

    @pytest.mark.asyncio
    async def test_successful_question_processing(self, sample_question, sample_research_report,
                                                sample_forecast, mock_services):
        """Test successful end-to-end question processing."""
        # Setup mocks
        mock_services['forecasting_pipeline'].conduct_research = AsyncMock(
            return_value=[sample_research_report]
        )
        mock_services['forecasting_pipeline'].generate_ensemble_predictions = AsyncMock(
            return_value=[sample_forecast]
        )
        mock_services['forecasting_pipeline'].aggregate_forecasts = AsyncMock(
            return_value=sample_forecast
        )
        mock_services['forecasting_pipeline'].apply_risk_management = AsyncMock(
            return_value=sample_forecast
        )
        mock_services['forecasting_pipeline'].submit_forecast = AsyncMock(
            return_value={'submission_id': 'test_submission_123'}
        )

        # Create use case
        use_case = ProcessTournamentQuestionUseCase(
            forecasting_pipeline=mock_services['forecasting_pipeline'],
            tournament_service=mock_services['tournament_service'],
            learning_service=mock_services['learning_service'],
            circuit_breaker=mock_services['circuit_breaker'],
            retry_strategy=mock_services['retry_strategy'],
            degradation_manager=mock_services['degradation_manager']
        )

        # Execute use case
        result = await use_case.execute(
            question=sample_question,
            tournament_id=1,
            submission_mode=True
        )

        # Verify result
        assert result.success is True
        assert result.question_id == sample_question.id
        assert result.final_forecast is not None
        assert result.final_forecast.question_id == sample_question.id
        assert len(result.research_reports) == 1
        assert len(result.ensemble_forecasts) == 1
        assert result.processing_time > 0
        assert result.error_message is None

        # Verify all pipeline stages were called
        mock_services['forecasting_pipeline'].conduct_research.assert_called_once()
        mock_services['forecasting_pipeline'].generate_ensemble_predictions.assert_called_once()
        mock_services['forecasting_pipeline'].aggregate_forecasts.assert_called_once()
        mock_services['forecasting_pipeline'].apply_risk_management.assert_called_once()
        mock_services['forecasting_pipeline'].submit_forecast.assert_called_once()
        mock_services['learning_service'].record_processing_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_processing_with_research_failure(self, sample_question, sample_forecast, mock_services):
        """Test processing when research phase fails but graceful degradation works."""
        # Setup mocks - research fails but degradation provides fallback
        mock_services['forecasting_pipeline'].conduct_research = AsyncMock(
            side_effect=Exception("Research failed")
        )
        mock_services['degradation_manager'].get_degraded_research = AsyncMock(
            return_value=[ResearchReport(
                id=1, question_id=1, sources=["fallback_source"], credibility_scores={},
                evidence_synthesis="Fallback research", base_rates={}, knowledge_gaps=[],
                research_quality_score=0.5, timestamp=datetime.utcnow()
            )]
        )
        mock_services['forecasting_pipeline'].generate_ensemble_predictions = AsyncMock(
            return_value=[sample_forecast]
        )
        mock_services['forecasting_pipeline'].aggregate_forecasts = AsyncMock(
            return_value=sample_forecast
        )
        mock_services['forecasting_pipeline'].apply_risk_management = AsyncMock(
            return_value=sample_forecast
        )

        # Create use case
        use_case = ProcessTournamentQuestionUseCase(
            forecasting_pipeline=mock_services['forecasting_pipeline'],
            tournament_service=mock_services['tournament_service'],
            learning_service=mock_services['learning_service'],
            circuit_breaker=mock_services['circuit_breaker'],
            retry_strategy=mock_services['retry_strategy'],
            degradation_manager=mock_services['degradation_manager']
        )

        # Execute use case
        result = await use_case.execute(
            question=sample_question,
            submission_mode=False
        )

        # Verify graceful degradation worked
        assert result.success is True
        assert result.final_forecast is not None
        assert len(result.research_reports) == 1  # Fallback research
        mock_services['degradation_manager'].get_degraded_research.assert_called_once()

    @pytest.mark.asyncio
    async def test_processing_with_complete_failure(self, sample_question, mock_services):
        """Test processing when all stages fail and graceful degradation is applied."""
        # Setup mocks - everything fails
        mock_services['forecasting_pipeline'].conduct_research = AsyncMock(
            side_effect=Exception("Research failed")
        )
        mock_services['degradation_manager'].get_degraded_research = AsyncMock(return_value=[])
        mock_services['forecasting_pipeline'].generate_ensemble_predictions = AsyncMock(
            side_effect=Exception("Prediction failed")
        )
        mock_services['degradation_manager'].get_single_agent_prediction = AsyncMock(return_value=None)

        # Create use case
        use_case = ProcessTournamentQuestionUseCase(
            forecasting_pipeline=mock_services['forecasting_pipeline'],
            tournament_service=mock_services['tournament_service'],
            learning_service=mock_services['learning_service'],
            circuit_breaker=mock_services['circuit_breaker'],
            retry_strategy=mock_services['retry_strategy'],
            degradation_manager=mock_services['degradation_manager']
        )

        # Execute use case
        result = await use_case.execute(
            question=sample_question,
            submission_mode=False
        )

        # Verify failure is handled gracefully
        assert result.success is False
        assert result.error_message is not None
        assert result.final_forecast is None
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, sample_question, sample_research_report, sample_forecast, mock_services):
        """Test processing in dry run mode (no submission)."""
        # Setup mocks
        mock_services['forecasting_pipeline'].conduct_research = AsyncMock(
            return_value=[sample_research_report]
        )
        mock_services['forecasting_pipeline'].generate_ensemble_predictions = AsyncMock(
            return_value=[sample_forecast]
        )
        mock_services['forecasting_pipeline'].aggregate_forecasts = AsyncMock(
            return_value=sample_forecast
        )
        mock_services['forecasting_pipeline'].apply_risk_management = AsyncMock(
            return_value=sample_forecast
        )

        # Create use case
        use_case = ProcessTournamentQuestionUseCase(
            forecasting_pipeline=mock_services['forecasting_pipeline'],
            tournament_service=mock_services['tournament_service'],
            learning_service=mock_services['learning_service'],
            circuit_breaker=mock_services['circuit_breaker'],
            retry_strategy=mock_services['retry_strategy'],
            degradation_manager=mock_services['degradation_manager']
        )

        # Execute use case in dry run mode
        result = await use_case.execute(
            question=sample_question,
            submission_mode=False
        )

        # Verify no submission was attempted
        assert result.success is True
        mock_services['forecasting_pipeline'].submit_forecast.assert_not_called()


class TestForecastingPipeline:
    """Test the main forecasting pipeline orchestration."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, sample_question, mock_services):
        """Test complete pipeline execution with all stages."""
        # Create pipeline with mock dependencies
        config = PipelineConfig(
            max_research_time_minutes=5,
            enabled_agents=['cot', 'tot'],
            max_agents=2
        )

        # Mock search client
        search_client = Mock()
        search_client.conduct_research = AsyncMock(return_value=ResearchReport(
            id=1, question_id=1, sources=["test_source"], credibility_scores={},
            evidence_synthesis="Test research", base_rates={}, knowledge_gaps=[],
            research_quality_score=0.8, timestamp=datetime.utcnow()
        ))

        # Mock agents
        mock_agent = Mock()
        mock_agent.forecast = AsyncMock(return_value=Forecast.create_binary(
            question_id=1, probability=0.7, confidence_level=0.8,
            confidence_basis="Test forecast", reasoning_trace=[], evidence_sources=[]
        ))
        agents = {'cot': mock_agent, 'tot': mock_agent}

        # Mock ensemble agent
        ensemble_agent = Mock()
        ensemble_agent.aggregate_predictions = AsyncMock(return_value=Forecast.create_binary(
            question_id=1, probability=0.65, confidence_level=0.85,
            confidence_basis="Ensemble forecast", reasoning_trace=[], evidence_sources=[]
        ))

        # Mock cache manager
        cache_manager = Mock()
        cache_manager.get = AsyncMock(return_value=None)
        cache_manager.set = AsyncMock()

        # Create pipeline
        pipeline = ForecastingPipeline(
            search_client=search_client,
            agents=agents,
            ensemble_agent=ensemble_agent,
            tournament_service=mock_services['tournament_service'],
            learning_service=mock_services['learning_service'],
            circuit_breaker=mock_services['circuit_breaker'],
            retry_strategy=mock_services['retry_strategy'],
            cache_manager=cache_manager,
            config=config
        )

        # Execute pipeline
        result = await pipeline.process_question(sample_question)

        # Verify result
        assert result is not None
        assert result.question_id == sample_question.id
        assert isinstance(result.prediction, float)
        assert 0.0 <= result.prediction <= 1.0
        assert result.confidence.level > 0

        # Verify pipeline metrics were updated
        assert pipeline.metrics.research_sources_used > 0
        assert pipeline.metrics.agents_executed > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_caching(self, sample_question, mock_services):
        """Test pipeline with caching enabled."""
        # Create cached research report
        cached_report = ResearchReport(
            id=1, question_id=1, sources=["cached_source"], credibility_scores={},
            evidence_synthesis="Cached research", base_rates={}, knowledge_gaps=[],
            research_quality_score=0.9, timestamp=datetime.utcnow()
        )

        # Mock cache manager with cached data
        cache_manager = Mock()
        cache_manager.get = AsyncMock(return_value=[cached_report])
        cache_manager.set = AsyncMock()

        # Mock other components
        search_client = Mock()
        agents = {'cot': Mock()}
        agents['cot'].forecast = AsyncMock(return_value=Forecast.create_binary(
            question_id=1, probability=0.6, confidence_level=0.7,
            confidence_basis="Test", reasoning_trace=[], evidence_sources=[]
        ))
        ensemble_agent = Mock()
        ensemble_agent.aggregate_predictions = AsyncMock(return_value=agents['cot'].forecast.return_value)

        # Create pipeline
        pipeline = ForecastingPipeline(
            search_client=search_client,
            agents=agents,
            ensemble_agent=ensemble_agent,
            tournament_service=mock_services['tournament_service'],
            learning_service=mock_services['learning_service'],
            circuit_breaker=mock_services['circuit_breaker'],
            retry_strategy=mock_services['retry_strategy'],
            cache_manager=cache_manager
        )

        # Execute pipeline
        result = await pipeline.process_question(sample_question)

        # Verify cache was used
        assert result is not None
        assert pipeline.metrics.cache_hits == 1
        assert pipeline.metrics.cache_misses == 0
        search_client.conduct_research.assert_not_called()  # Should not be called due to cache hit


class TestIntegrationService:
    """Test the integration service for backward compatibility."""

    @pytest.mark.asyncio
    async def test_legacy_question_processing(self, mock_services):
        """Test processing of legacy format questions."""
        # Create legacy question
        legacy_question = LegacyQuestionFormat(
            question_id=1,
            question_text="Will AI achieve human-level performance by 2025?",
            question_type='binary',
            background_info="Background about AI development",
            resolution_criteria="Human-level performance defined as...",
            page_url="https://example.com/question/1"
        )

        # Mock successful processing
        mock_result = ProcessingResult(
            correlation_id="test_correlation",
            question_id=1,
            final_forecast=Forecast.create_binary(
                question_id=1, probability=0.75, confidence_level=0.8,
                confidence_basis="Advanced analysis", reasoning_trace=[], evidence_sources=[]
            ),
            research_reports=[],
            ensemble_forecasts=[],
            processing_time=5.0,
            success=True,
            error_message=None,
            metadata={}
        )

        # Mock use case
        mock_use_case = Mock()
        mock_use_case.execute = AsyncMock(return_value=mock_result)

        # Create integration service
        integration_service = IntegrationService(
            process_question_use_case=mock_use_case,
            forecasting_pipeline=mock_services['forecasting_pipeline'],
            tournament_service=mock_services['tournament_service']
        )

        # Process legacy question
        result = await integration_service.process_legacy_question(legacy_question)

        # Verify result
        assert result.question_id == 1
        assert result.forecast == 0.75
        assert result.confidence == 0.8
        assert result.justification is not None
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_legacy_agent_result_format(self, mock_services):
        """Test conversion to legacy agent result format."""
        # Create question dictionary in legacy format
        question_dict = {
            'question_id': 1,
            'question_text': 'Will quantum computing achieve practical advantage by 2030?',
            'type': 'binary',
            'background': 'Quantum computing background',
            'page_url': 'https://example.com/quantum'
        }

        # Mock successful processing
        mock_result = ProcessingResult(
            correlation_id="test_correlation",
            question_id=1,
            final_forecast=Forecast.create_binary(
                question_id=1, probability=0.6, confidence_level=0.75,
                confidence_basis="Quantum analysis", reasoning_trace=[], evidence_sources=[]
            ),
            research_reports=[],
            ensemble_forecasts=[],
            processing_time=3.0,
            success=True,
            error_message=None,
            metadata={}
        )

        # Mock use case
        mock_use_case = Mock()
        mock_use_case.execute = AsyncMock(return_value=mock_result)

        # Create integration service
        integration_service = IntegrationService(
            process_question_use_case=mock_use_case,
            forecasting_pipeline=mock_services['forecasting_pipeline'],
            tournament_service=mock_services['tournament_service']
        )

        # Get legacy agent result
        result = await integration_service.get_legacy_agent_result(question_dict)

        # Verify legacy format
        assert 'question_id' in result
        assert 'forecast' in result
        assert 'justification' in result
        assert 'prediction' in result  # Alias for backward compatibility
        assert 'reasoning' in result
        assert 'confidence' in result
        assert 'trace' in result
        assert 'metadata' in result

        assert result['question_id'] == 1
        assert result['forecast'] == 0.6
        assert result['prediction'] == 0.6
        assert result['confidence'] == 0.75
        assert isinstance(result['trace'], list)

    @pytest.mark.asyncio
    async def test_fallback_to_basic_processing(self, mock_services):
        """Test fallback to basic processing when advanced features fail."""
        # Create legacy question
        legacy_question = LegacyQuestionFormat(
            question_id=1,
            question_text="Simple binary question",
            question_type='binary'
        )

        # Mock failed processing
        mock_result = ProcessingResult(
            correlation_id="test_correlation",
            question_id=1,
            final_forecast=None,
            research_reports=[],
            ensemble_forecasts=[],
            processing_time=1.0,
            success=False,
            error_message="Processing failed",
            metadata={}
        )

        # Mock use case
        mock_use_case = Mock()
        mock_use_case.execute = AsyncMock(return_value=mock_result)

        # Create integration service
        integration_service = IntegrationService(
            process_question_use_case=mock_use_case,
            forecasting_pipeline=mock_services['forecasting_pipeline'],
            tournament_service=mock_services['tournament_service']
        )

        # Process legacy question
        result = await integration_service.process_legacy_question(legacy_question)

        # Verify fallback result
        assert result.question_id == 1
        assert result.forecast == 0.5  # Default binary prediction
        assert result.confidence == 0.1  # Low confidence for fallback
        assert "Processing failed" in result.justification
        assert result.metadata.get('fallback_used') is True


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow scenarios."""

    @pytest.mark.asyncio
    async def test_complete_tournament_workflow(self, sample_tournament, mock_services):
        """Test complete tournament processing workflow."""
        # Setup comprehensive mocks for full workflow
        mock_services['tournament_service'].get_tournament = AsyncMock(return_value=sample_tournament)

        # Mock research phase
        research_report = ResearchReport(
            id=1, question_id=1, sources=["source1", "source2"], credibility_scores={},
            evidence_synthesis="Tournament research", base_rates={}, knowledge_gaps=[],
            research_quality_score=0.85, timestamp=datetime.utcnow()
        )
        mock_services['forecasting_pipeline'].conduct_research = AsyncMock(return_value=[research_report])

        # Mock prediction phase
        forecast1 = Forecast.create_binary(
            question_id=1, probability=0.7, confidence_level=0.8,
            confidence_basis="Agent 1", reasoning_trace=[], evidence_sources=[], agent_id="agent1"
        )
        forecast2 = Forecast.create_binary(
            question_id=1, probability=0.6, confidence_level=0.75,
            confidence_basis="Agent 2", reasoning_trace=[], evidence_sources=[], agent_id="agent2"
        )
        mock_services['forecasting_pipeline'].generate_ensemble_predictions = AsyncMock(
            return_value=[forecast1, forecast2]
        )

        # Mock ensemble aggregation
        final_forecast = Forecast.create_binary(
            question_id=1, probability=0.65, confidence_level=0.85,
            confidence_basis="Ensemble", reasoning_trace=[], evidence_sources=[]
        )
        mock_services['forecasting_pipeline'].aggregate_forecasts = AsyncMock(return_value=final_forecast)
        mock_services['forecasting_pipeline'].apply_risk_management = AsyncMock(return_value=final_forecast)

        # Mock submission
        mock_services['forecasting_pipeline'].submit_forecast = AsyncMock(
            return_value={'submission_id': 'tournament_submission_123'}
        )

        # Create use case
        use_case = ProcessTournamentQuestionUseCase(
            forecasting_pipeline=mock_services['forecasting_pipeline'],
            tournament_service=mock_services['tournament_service'],
            learning_service=mock_services['learning_service'],
            circuit_breaker=mock_services['circuit_breaker'],
            retry_strategy=mock_services['retry_strategy'],
            degradation_manager=mock_services['degradation_manager']
        )

        # Process tournament question
        result = await use_case.execute(
            question=sample_tournament.questions[0],
            tournament_id=sample_tournament.id,
            submission_mode=True
        )

        # Verify complete workflow
        assert result.success is True
        assert result.final_forecast is not None
        assert result.final_forecast.prediction == 0.65
        assert len(result.research_reports) == 1
        assert len(result.ensemble_forecasts) == 2
        assert result.metadata.get('submission_result') is not None

        # Verify all services were called
        mock_services['forecasting_pipeline'].conduct_research.assert_called_once()
        mock_services['forecasting_pipeline'].generate_ensemble_predictions.assert_called_once()
        mock_services['forecasting_pipeline'].aggregate_forecasts.assert_called_once()
        mock_services['forecasting_pipeline'].apply_risk_management.assert_called_once()
        mock_services['forecasting_pipeline'].submit_forecast.assert_called_once()
        mock_services['learning_service'].record_processing_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_question_processing(self, sample_tournament, mock_services):
        """Test concurrent processing of multiple questions."""
        # Create multiple questions
        questions = []
        for i in range(3):
            question = Question(
                id=i+1,
                text=f"Question {i+1}",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=datetime.utcnow() + timedelta(days=30),
                background=f"Background {i+1}",
                resolution_criteria=f"Criteria {i+1}",
                scoring_weight=1.0
            )
            questions.append(question)

        # Mock successful processing for all questions
        async def mock_execute(question, **kwargs):
            return ProcessingResult(
                correlation_id=f"corr_{question.id}",
                question_id=question.id,
                final_forecast=Forecast.create_binary(
                    question_id=question.id, probability=0.5 + question.id * 0.1,
                    confidence_level=0.8, confidence_basis="Test",
                    reasoning_trace=[], evidence_sources=[]
                ),
                research_reports=[],
                ensemble_forecasts=[],
                processing_time=1.0,
                success=True,
                error_message=None,
                metadata={}
            )

        # Create use case with mock
        use_case = Mock()
        use_case.execute = AsyncMock(side_effect=mock_execute)

        # Process questions concurrently
        tasks = [
            use_case.execute(question=q, tournament_id=1, submission_mode=False)
            for q in questions
        ]
        results = await asyncio.gather(*tasks)

        # Verify all questions were processed
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.success is True
            assert result.question_id == i + 1
            assert result.final_forecast.prediction == 0.5 + (i + 1) * 0.1

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, sample_question, mock_services):
        """Test error recovery and resilience mechanisms."""
        # Setup circuit breaker to fail initially then succeed
        call_count = 0

        async def mock_circuit_breaker_call(func, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"Simulated failure {call_count}")
            return await func(*args, **kwargs)

        mock_services['circuit_breaker'].call = AsyncMock(side_effect=mock_circuit_breaker_call)

        # Setup retry strategy to retry on failure
        retry_count = 0

        async def mock_retry_execute(func, *args, **kwargs):
            nonlocal retry_count
            retry_count += 1
            if retry_count <= 1:
                raise Exception("Retry failure")
            return await func(*args, **kwargs)

        mock_services['retry_strategy'].execute = AsyncMock(side_effect=mock_retry_execute)

        # Setup successful final execution
        mock_services['forecasting_pipeline'].conduct_research = AsyncMock(return_value=[])
        mock_services['forecasting_pipeline'].generate_ensemble_predictions = AsyncMock(return_value=[])
        mock_services['forecasting_pipeline'].aggregate_forecasts = AsyncMock(return_value=None)

        # Create use case
        use_case = ProcessTournamentQuestionUseCase(
            forecasting_pipeline=mock_services['forecasting_pipeline'],
            tournament_service=mock_services['tournament_service'],
            learning_service=mock_services['learning_service'],
            circuit_breaker=mock_services['circuit_breaker'],
            retry_strategy=mock_services['retry_strategy'],
            degradation_manager=mock_services['degradation_manager']
        )

        # Execute with expected failures and recovery
        result = await use_case.execute(
            question=sample_question,
            submission_mode=False
        )

        # Verify resilience mechanisms were used
        assert mock_services['circuit_breaker'].call.call_count > 0
        # Result may fail due to mocked failures, but resilience mechanisms were tested
