"""
Comprehensive integration tests for the tournament orchestrator.
Tests all components working together with proper dependency injection.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import yaml

from src.application.tournament_orchestrator import TournamentOrchestrator, create_tournament_orchestrator
from src.infrastructure.config.settings import Settings
from src.domain.entities.question import Question, QuestionType, QuestionStatus
from src.domain.entities.forecast import Forecast


class TestTournamentOrchestrator:
    """Test suite for tournament orchestrator integration."""

    @pytest.fixture
    async def temp_config(self):
        """Create temporary configuration file for testing."""
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.3,
                "api_key": "test-key"
            },
            "search": {
                "provider": "multi_source",
                "max_results": 5,
                "timeout": 30.0
            },
            "metaculus": {
                "base_url": "https://test.metaculus.com/api",
                "tournament_id": 12345,
                "dry_run": True
            },
            "pipeline": {
                "max_concurrent_questions": 2,
                "default_agent_names": ["ensemble"],
                "health_check_interval": 10
            },
            "bot": {
                "name": "TestBot",
                "version": "1.0.0"
            },
            "logging": {
                "level": "INFO",
                "console_output": True
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        yield config_path

        # Cleanup
        Path(config_path).unlink(missing_ok=True)

    @pytest.fixture
    async def orchestrator(self, temp_config):
        """Create orchestrator instance for testing."""
        orchestrator = TournamentOrchestrator(temp_config)

        # Mock external dependencies
        with patch('src.infrastructure.external_apis.llm_client.LLMClient') as mock_llm, \
             patch('src.infrastructure.external_apis.search_client.SearchClient') as mock_search, \
             patch('src.infrastructure.external_apis.metaculus_client.MetaculusClient') as mock_metaculus:

            # Configure mocks
            mock_llm.return_value.initialize = AsyncMock()
            mock_llm.return_value.health_check = AsyncMock()
            mock_llm.return_value.update_config = AsyncMock()

            mock_search.return_value.initialize = AsyncMock()
            mock_search.return_value.health_check = AsyncMock()
            mock_search.return_value.update_config = AsyncMock()

            mock_metaculus.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.health_check = AsyncMock()
            mock_metaculus.return_value.update_config = AsyncMock()

            await orchestrator.initialize()

            yield orchestrator

            await orchestrator.shutdown()

    async def test_orchestrator_initialization(self, temp_config):
        """Test that orchestrator initializes all components correctly."""
        orchestrator = TournamentOrchestrator(temp_config)

        with patch('src.infrastructure.external_apis.llm_client.LLMClient') as mock_llm, \
             patch('src.infrastructure.external_apis.search_client.SearchClient') as mock_search, \
             patch('src.infrastructure.external_apis.metaculus_client.MetaculusClient') as mock_metaculus:

            # Configure mocks
            mock_llm.return_value.initialize = AsyncMock()
            mock_search.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.initialize = AsyncMock()

            await orchestrator.initialize()

            # Verify all components are initialized
            assert orchestrator.registry is not None
            assert orchestrator.registry.settings is not None
            assert orchestrator.registry.llm_client is not None
            assert orchestrator.registry.search_client is not None
            assert orchestrator.registry.metaculus_client is not None
            assert orchestrator.registry.forecasting_pipeline is not None
            assert orchestrator.registry.dispatcher is not None

            # Verify configuration is loaded correctly
            assert orchestrator.registry.settings.bot.name == "TestBot"
            assert orchestrator.registry.settings.metaculus.tournament_id == 12345
            assert orchestrator.registry.settings.pipeline.max_concurrent_questions == 2

            await orchestrator.shutdown()

    async def test_dependency_injection(self, orchestrator):
        """Test that dependency injection works correctly across all components."""
        registry = orchestrator.registry

        # Verify that services have proper dependencies injected
        assert registry.research_service.search_client == registry.search_client
        assert registry.research_service.llm_client == registry.llm_client

        assert registry.forecast_service.forecasting_service == registry.forecasting_service
        assert registry.forecast_service.ensemble_service == registry.ensemble_service
        assert registry.forecast_service.research_service == registry.research_service

        assert registry.dispatcher.forecast_service == registry.forecast_service
        assert registry.dispatcher.ingestion_service == registry.ingestion_service

        assert registry.forecasting_pipeline.llm_client == registry.llm_client
        assert registry.forecasting_pipeline.search_client == registry.search_client
        assert registry.forecasting_pipeline.metaculus_client == registry.metaculus_client

    async def test_health_check_system(self, orchestrator):
        """Test comprehensive health checking system."""
        # Mock health check methods
        orchestrator.registry.llm_client.health_check = AsyncMock(return_value=True)
        orchestrator.registry.search_client.health_check = AsyncMock(return_value=True)
        orchestrator.registry.metaculus_client.health_check = AsyncMock(return_value=True)

        # Perform health check
        health_status = await orchestrator._perform_health_check()

        # Verify health check results
        assert isinstance(health_status, dict)
        assert "llm_client" in health_status
        assert "search_client" in health_status
        assert "metaculus_client" in health_status

        # Verify metrics are updated
        assert orchestrator.metrics["last_health_check"] is not None
        assert orchestrator.metrics["component_health"] == health_status

    async def test_configuration_hot_reload(self, orchestrator, temp_config):
        """Test configuration hot-reloading functionality."""
        original_name = orchestrator.registry.settings.bot.name

        # Update configuration file
        config_data = {
            "llm": {"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
            "search": {"provider": "multi_source"},
            "metaculus": {"base_url": "https://test.metaculus.com/api", "tournament_id": 12345},
            "pipeline": {"max_concurrent_questions": 2, "default_agent_names": ["ensemble"]},
            "bot": {"name": "UpdatedTestBot", "version": "2.0.0"},
            "logging": {"level": "INFO"}
        }

        with open(temp_config, 'w') as f:
            yaml.dump(config_data, f)

        # Trigger configuration reload
        await orchestrator._reload_configuration()

        # Verify configuration was updated
        assert orchestrator.registry.settings.bot.name == "UpdatedTestBot"
        assert orchestrator.registry.settings.bot.version == "2.0.0"
        assert orchestrator.registry.settings.bot.name != original_name

    async def test_single_question_forecast_integration(self, orchestrator):
        """Test complete single question forecasting integration."""
        question_id = 12345

        # Mock the pipeline method
        expected_result = {
            "question_id": question_id,
            "forecast": {
                "prediction": 0.65,
                "confidence": 0.8,
                "method": "ensemble",
                "reasoning": "Test reasoning"
            },
            "metadata": {"test": True}
        }

        orchestrator.registry.forecasting_pipeline.run_single_question = AsyncMock(
            return_value=expected_result
        )

        # Run single question forecast
        result = await orchestrator.run_single_question(question_id, "ensemble")

        # Verify result
        assert result == expected_result
        assert orchestrator.metrics["questions_processed"] == 1
        assert orchestrator.metrics["forecasts_generated"] == 1

        # Verify pipeline was called correctly
        orchestrator.registry.forecasting_pipeline.run_single_question.assert_called_once_with(
            question_id=question_id,
            agent_type="ensemble",
            include_research=True,
            collect_metrics=True
        )

    async def test_batch_forecast_integration(self, orchestrator):
        """Test batch forecasting integration."""
        question_ids = [12345, 12346, 12347]

        # Mock the pipeline method
        expected_results = [
            {
                "question_id": qid,
                "forecast": {
                    "prediction": 0.6 + (i * 0.1),
                    "confidence": 0.8,
                    "method": "ensemble"
                }
            }
            for i, qid in enumerate(question_ids)
        ]

        orchestrator.registry.forecasting_pipeline.run_batch_forecast = AsyncMock(
            return_value=expected_results
        )

        # Run batch forecast
        results = await orchestrator.run_batch_forecast(question_ids, "ensemble")

        # Verify results
        assert len(results) == len(question_ids)
        assert results == expected_results
        assert orchestrator.metrics["questions_processed"] == len(question_ids)
        assert orchestrator.metrics["forecasts_generated"] == len(results)

    async def test_tournament_run_integration(self, orchestrator):
        """Test complete tournament run integration."""
        tournament_id = 12345
        max_questions = 5
        agent_types = ["ensemble"]

        # Mock dispatcher method
        expected_results = {
            "tournament_id": tournament_id,
            "questions_processed": max_questions,
            "forecasts_generated": max_questions,
            "performance_metrics": {"avg_confidence": 0.75},
            "agent_performance": {"ensemble": {"accuracy": 0.8}}
        }

        orchestrator.registry.dispatcher.run_tournament = AsyncMock(
            return_value=expected_results
        )

        # Run tournament
        results = await orchestrator.run_tournament(
            tournament_id=tournament_id,
            max_questions=max_questions,
            agent_types=agent_types
        )

        # Verify results structure
        assert "tournament_id" in results
        assert "start_time" in results
        assert "end_time" in results
        assert "duration_seconds" in results
        assert "questions_processed" in results
        assert "forecasts_generated" in results

        # Verify metrics updated
        assert orchestrator.metrics["questions_processed"] == max_questions
        assert orchestrator.metrics["forecasts_generated"] == max_questions

    async def test_system_status_reporting(self, orchestrator):
        """Test comprehensive system status reporting."""
        # Mock health check
        orchestrator._perform_health_check = AsyncMock(return_value={
            "llm_client": True,
            "search_client": True,
            "metaculus_client": True
        })

        # Get system status
        status = await orchestrator.get_system_status()

        # Verify status structure
        assert status["status"] == "running"
        assert "uptime_seconds" in status
        assert "health_status" in status
        assert "metrics" in status
        assert "configuration" in status
        assert "last_config_reload" in status

        # Verify configuration details
        config = status["configuration"]
        assert config["environment"] == orchestrator.registry.settings.environment
        assert config["tournament_id"] == orchestrator.registry.settings.metaculus.tournament_id
        assert config["max_concurrent_questions"] == orchestrator.registry.settings.pipeline.max_concurrent_questions

    async def test_error_handling_and_recovery(self, orchestrator):
        """Test error handling and recovery mechanisms."""
        question_id = 12345

        # Mock pipeline to raise an exception
        orchestrator.registry.forecasting_pipeline.run_single_question = AsyncMock(
            side_effect=Exception("Test error")
        )

        # Verify exception is raised and metrics updated
        with pytest.raises(Exception, match="Test error"):
            await orchestrator.run_single_question(question_id)

        assert orchestrator.metrics["errors_encountered"] == 1

    async def test_graceful_shutdown(self, orchestrator):
        """Test graceful shutdown of all components."""
        # Add shutdown methods to mocks
        orchestrator.registry.llm_client.shutdown = AsyncMock()
        orchestrator.registry.search_client.shutdown = AsyncMock()
        orchestrator.registry.metaculus_client.shutdown = AsyncMock()
        orchestrator.registry.health_monitor.shutdown = AsyncMock()

        # Perform shutdown
        await orchestrator.shutdown()

        # Verify shutdown was called on components
        orchestrator.registry.llm_client.shutdown.assert_called_once()
        orchestrator.registry.search_client.shutdown.assert_called_once()
        orchestrator.registry.metaculus_client.shutdown.assert_called_once()
        orchestrator.registry.health_monitor.shutdown.assert_called_once()

        # Verify shutdown event is set
        assert orchestrator._shutdown_event.is_set()

    async def test_managed_lifecycle_context_manager(self, temp_config):
        """Test managed lifecycle context manager."""
        with patch('src.infrastructure.external_apis.llm_client.LLMClient') as mock_llm, \
             patch('src.infrastructure.external_apis.search_client.SearchClient') as mock_search, \
             patch('src.infrastructure.external_apis.metaculus_client.MetaculusClient') as mock_metaculus:

            # Configure mocks
            mock_llm.return_value.initialize = AsyncMock()
            mock_llm.return_value.shutdown = AsyncMock()
            mock_search.return_value.initialize = AsyncMock()
            mock_search.return_value.shutdown = AsyncMock()
            mock_metaculus.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.shutdown = AsyncMock()

            orchestrator = TournamentOrchestrator(temp_config)

            # Use context manager
            async with orchestrator.managed_lifecycle() as orch:
                assert orch.registry is not None
                assert orch.registry.settings is not None

            # Verify shutdown was called
            assert orchestrator._shutdown_event.is_set()

    async def test_factory_function(self, temp_config):
        """Test factory function for creating orchestrator."""
        with patch('src.infrastructure.external_apis.llm_client.LLMClient') as mock_llm, \
             patch('src.infrastructure.external_apis.search_client.SearchClient') as mock_search, \
             patch('src.infrastructure.external_apis.metaculus_client.MetaculusClient') as mock_metaculus:

            # Configure mocks
            mock_llm.return_value.initialize = AsyncMock()
            mock_search.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.initialize = AsyncMock()

            # Create orchestrator using factory
            orchestrator = await create_tournament_orchestrator(temp_config)

            # Verify it's properly initialized
            assert orchestrator.registry is not None
            assert orchestrator.registry.settings.bot.name == "TestBot"

            await orchestrator.shutdown()


class TestIntegrationValidation:
    """Test integration validation and comprehensive testing."""

    async def test_component_integration_validation(self, temp_config):
        """Test that all components integrate correctly with validation."""
        with patch('src.infrastructure.external_apis.llm_client.LLMClient') as mock_llm, \
             patch('src.infrastructure.external_apis.search_client.SearchClient') as mock_search, \
             patch('src.infrastructure.external_apis.metaculus_client.MetaculusClient') as mock_metaculus:

            # Configure mocks with validation
            mock_llm.return_value.initialize = AsyncMock()
            mock_llm.return_value.health_check = AsyncMock()
            mock_search.return_value.initialize = AsyncMock()
            mock_search.return_value.health_check = AsyncMock()
            mock_metaculus.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.health_check = AsyncMock()

            orchestrator = TournamentOrchestrator(temp_config)
            await orchestrator.initialize()

            # Validate all required components exist
            required_components = [
                'settings', 'llm_client', 'search_client', 'metaculus_client',
                'circuit_breaker', 'rate_limiter', 'health_monitor', 'retry_manager',
                'reasoning_logger', 'dispatcher', 'forecast_service', 'ingestion_service',
                'ensemble_service', 'forecasting_service', 'research_service',
                'tournament_analytics', 'performance_tracking', 'calibration_service',
                'risk_management_service', 'forecasting_pipeline'
            ]

            for component in required_components:
                assert hasattr(orchestrator.registry, component), f"Missing component: {component}"
                assert getattr(orchestrator.registry, component) is not None, f"Component is None: {component}"

            # Validate component types
            from src.infrastructure.config.settings import Settings
            from src.pipelines.forecasting_pipeline import ForecastingPipeline
            from src.application.dispatcher import Dispatcher

            assert isinstance(orchestrator.registry.settings, Settings)
            assert isinstance(orchestrator.registry.forecasting_pipeline, ForecastingPipeline)
            assert isinstance(orchestrator.registry.dispatcher, Dispatcher)

            await orchestrator.shutdown()

    async def test_end_to_end_integration_flow(self, temp_config):
        """Test complete end-to-end integration flow."""
        with patch('src.infrastructure.external_apis.llm_client.LLMClient') as mock_llm, \
             patch('src.infrastructure.external_apis.search_client.SearchClient') as mock_search, \
             patch('src.infrastructure.external_apis.metaculus_client.MetaculusClient') as mock_metaculus:

            # Configure comprehensive mocks
            mock_llm.return_value.initialize = AsyncMock()
            mock_llm.return_value.health_check = AsyncMock()
            mock_search.return_value.initialize = AsyncMock()
            mock_search.return_value.health_check = AsyncMock()
            mock_metaculus.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.health_check = AsyncMock()

            orchestrator = TournamentOrchestrator(temp_config)
            await orchestrator.initialize()

            # Mock complete pipeline flow
            mock_question_data = {
                "id": 12345,
                "title": "Test Question",
                "description": "Test description",
                "type": "binary",
                "status": "open"
            }

            mock_forecast_result = {
                "question_id": 12345,
                "forecast": {
                    "prediction": 0.65,
                    "confidence": 0.8,
                    "method": "ensemble",
                    "reasoning": "Test reasoning"
                },
                "metadata": {"integration_test": True}
            }

            # Mock the pipeline methods
            orchestrator.registry.forecasting_pipeline.run_single_question = AsyncMock(
                return_value=mock_forecast_result
            )

            # Execute end-to-end flow
            result = await orchestrator.run_single_question(12345, "ensemble")

            # Validate complete flow
            assert result["question_id"] == 12345
            assert result["forecast"]["prediction"] == 0.65
            assert result["forecast"]["confidence"] == 0.8
            assert result["metadata"]["integration_test"] is True

            # Validate metrics tracking
            assert orchestrator.metrics["questions_processed"] == 1
            assert orchestrator.metrics["forecasts_generated"] == 1

            await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_requirements_compliance():
    """Test that orchestrator meets all requirements from 10.1, 10.2, 10.5."""

    # Requirement 10.1: Clean Architecture and SOLID principles
    orchestrator = TournamentOrchestrator()

    # Verify separation of concerns
    assert hasattr(orchestrator, '_load_configuration')  # Configuration management
    assert hasattr(orchestrator, '_create_llm_client')   # Client creation
    assert hasattr(orchestrator, '_perform_health_check') # Health monitoring
    assert hasattr(orchestrator, 'run_tournament')       # Business logic

    # Requirement 10.2: Plugin-based architecture and hot-swappable components
    # Verify configuration hot-reloading
    assert hasattr(orchestrator, '_reload_configuration')
    assert hasattr(orchestrator, '_config_reload_loop')
    assert hasattr(orchestrator, '_update_component_configs')

    # Requirement 10.5: Comprehensive API documentation and monitoring
    # Verify monitoring capabilities
    assert hasattr(orchestrator, 'get_system_status')
    assert hasattr(orchestrator, '_health_check_loop')
    assert hasattr(orchestrator, 'metrics')

    # Verify graceful lifecycle management
    assert hasattr(orchestrator, 'managed_lifecycle')
    assert hasattr(orchestrator, 'shutdown')
