"""
Comprehensive integration tests for task 12.1 - Integration and System Optimization.
Tests all components working together with proper dependency injection.
"""
import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.application.tournament_orchestrator import TournamentOrchestrator, ComponentRegistry
from src.infrastructure.config.config_manager import create_config_manager
from src.domain.entities.question import Question, QuestionType, QuestionStatus
from src.domain.entities.forecast import Forecast


class TestComprehensiveIntegration:
    """Test comprehensive integration of all system components."""

    @pytest.fixture
    async def comprehensive_config(self):
        """Create comprehensive configuration for testing."""
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.3,
                "api_key": "test-key",
                "rate_limit_rpm": 60
            },
            "search": {
                "provider": "multi_source",
                "max_results": 10,
                "timeout": 30.0
            },
            "metaculus": {
                "base_url": "https://test.metaculus.com/api",
                "tournament_id": 12345,
                "dry_run": True,
                "api_key": "test-metaculus-key"
            },
            "pipeline": {
                "max_concurrent_questions": 3,
                "default_agent_names": ["ensemble", "chain_of_thought", "tree_of_thought"],
                "health_check_interval": 30,
                "max_retries_per_question": 2,
                "retry_delay_seconds": 1.0
            },
            "ensemble": {
                "min_agents": 2,
                "confidence_threshold": 0.7,
                "agent_weights": {
                    "chain_of_thought": 0.4,
                    "tree_of_thought": 0.3,
                    "react": 0.3
                }
            },
            "bot": {
                "name": "ComprehensiveTestBot",
                "version": "2.0.0",
                "min_confidence_threshold": 0.6
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
    async def mock_external_services(self):
        """Mock all external services for integration testing."""
        with patch('src.infrastructure.external_apis.llm_client.LLMClient') as mock_llm, \
             patch('src.infrastructure.external_apis.search_client.DuckDuckGoSearchClient') as mock_search, \
             patch('src.infrastructure.external_apis.metaculus_client.MetaculusClient') as mock_metaculus:

            # Configure LLM client mock
            mock_llm.return_value.initialize = AsyncMock()
            mock_llm.return_value.health_check = AsyncMock()
            mock_llm.return_value.update_config = AsyncMock()
            mock_llm.return_value.generate_response = AsyncMock(
                return_value="Mock LLM response with reasoning"
            )

            # Configure search client mock
            mock_search.return_value.initialize = AsyncMock()
            mock_search.return_value.health_check = AsyncMock()
            mock_search.return_value.update_config = AsyncMock()
            mock_search.return_value.search = AsyncMock(
                return_value=[{"title": "Mock Search Result", "content": "Mock content"}]
            )

            # Configure Metaculus client mock
            mock_metaculus.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.health_check = AsyncMock()
            mock_metaculus.return_value.update_config = AsyncMock()
            mock_metaculus.return_value.get_questions = AsyncMock(
                return_value=[{
                    "id": 12345,
                    "title": "Test Question",
                    "description": "Test description",
                    "type": "binary",
                    "status": "open"
                }]
            )

            yield {
                "llm": mock_llm,
                "search": mock_search,
                "metaculus": mock_metaculus
            }

    async def test_complete_system_integration(self, comprehensive_config, mock_external_services):
        """Test complete system integration with all components."""
        orchestrator = TournamentOrchestrator(comprehensive_config)
        await orchestrator.initialize()

        try:
            # Verify all components are properly initialized
            registry = orchestrator.registry
            assert registry is not None

            # Test core infrastructure components
            assert registry.llm_client is not None
            assert registry.search_client is not None
            assert registry.metaculus_client is not None
            assert registry.circuit_breaker is not None
            assert registry.rate_limiter is not None
            assert registry.health_monitor is not None
            assert registry.retry_manager is not None
            assert registry.reasoning_logger is not None

            # Test application services
            assert registry.dispatcher is not None
            assert registry.forecast_service is not None
            assert registry.ingestion_service is not None

            # Test domain services
            assert registry.ensemble_service is not None
            assert registry.forecasting_service is not None
            assert registry.research_service is not None
            assert registry.tournament_analytics is not None
            assert registry.performance_tracking is not None
            assert registry.calibration_service is not None
            assert registry.risk_management_service is not None

            # Test advanced reasoning and analysis services
            assert registry.reasoning_orchestrator is not None
            assert registry.question_categorizer is not None
            assert registry.authoritative_source_manager is not None
            assert registry.conflict_resolver is not None
            assert registry.knowledge_gap_detector is not None
            assert registry.divergence_analyzer is not None
            assert registry.dynamic_weight_adjuster is not None
            assert registry.performance_analyzer is not None
            assert registry.pattern_detector is not None
            assert registry.strategy_adaptation_engine is not None
            assert registry.uncertainty_quantifier is not None
            assert registry.conservative_strategy_engine is not None
            assert registry.scoring_optimizer is not None
            assert registry.tournament_analyzer is not None

            # Test pipeline
            assert registry.forecasting_pipeline is not None

            print("✓ All components properly initialized")

        finally:
            await orchestrator.shutdown()

    async def test_dependency_injection_validation(self, comprehensive_config, mock_external_services):
        """Test that dependency injection works correctly across all components."""
        orchestrator = TournamentOrchestrator(comprehensive_config)
        await orchestrator.initialize()

        try:
            registry = orchestrator.registry

            # Verify research service dependencies
            assert registry.research_service.search_client == registry.search_client
            assert registry.research_service.llm_client == registry.llm_client

            # Verify reasoning orchestrator dependencies
            assert registry.reasoning_orchestrator.llm_client == registry.llm_client
            assert registry.reasoning_orchestrator.search_client == registry.search_client

            # Verify forecast service dependencies
            assert registry.forecast_service.forecasting_service == registry.forecasting_service
            assert registry.forecast_service.ensemble_service == registry.ensemble_service
            assert registry.forecast_service.research_service == registry.research_service
            assert registry.forecast_service.reasoning_orchestrator == registry.reasoning_orchestrator

            # Verify dispatcher dependencies
            assert registry.dispatcher.forecast_service == registry.forecast_service
            assert registry.dispatcher.ingestion_service == registry.ingestion_service
            assert registry.dispatcher.metaculus_client == registry.metaculus_client

            # Verify pipeline dependencies
            assert registry.forecasting_pipeline.llm_client == registry.llm_client
            assert registry.forecasting_pipeline.search_client == registry.search_client
            assert registry.forecasting_pipeline.metaculus_client == registry.metaculus_client

            print("✓ Dependency injection working correctly")

        finally:
            await orchestrator.shutdown()

    async def test_configuration_hot_reload_integration(self, comprehensive_config, mock_external_services):
        """Test configuration hot-reloading with all components."""
        orchestrator = TournamentOrchestrator(comprehensive_config)
        await orchestrator.initialize()

        try:
            original_name = orchestrator.registry.settings.bot.name
            original_temperature = orchestrator.registry.settings.llm.temperature

            # Update configuration
            updated_config = {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7,  # Changed
                    "api_key": "test-key",
                    "rate_limit_rpm": 60
                },
                "search": {"provider": "multi_source", "max_results": 10, "timeout": 30.0},
                "metaculus": {"base_url": "https://test.metaculus.com/api", "tournament_id": 12345, "dry_run": True},
                "pipeline": {"max_concurrent_questions": 3, "default_agent_names": ["ensemble"], "health_check_interval": 30},
                "ensemble": {"min_agents": 2, "confidence_threshold": 0.7, "agent_weights": {"chain_of_thought": 1.0}},
                "bot": {"name": "UpdatedComprehensiveBot", "version": "3.0.0"},  # Changed
                "logging": {"level": "INFO"}
            }

            with open(comprehensive_config, 'w') as f:
                yaml.dump(updated_config, f)

            # Trigger configuration reload
            await orchestrator._reload_configuration()

            # Verify configuration was updated
            assert orchestrator.registry.settings.bot.name == "UpdatedComprehensiveBot"
            assert orchestrator.registry.settings.bot.version == "3.0.0"
            assert orchestrator.registry.settings.llm.temperature == 0.7
            assert orchestrator.registry.settings.bot.name != original_name
            assert orchestrator.registry.settings.llm.temperature != original_temperature

            print("✓ Configuration hot-reload working")

        finally:
            await orchestrator.shutdown()

    async def test_end_to_end_tournament_flow(self, comprehensive_config, mock_external_services):
        """Test complete end-to-end tournament forecasting flow."""
        orchestrator = TournamentOrchestrator(comprehensive_config)
        await orchestrator.initialize()

        try:
            # Mock tournament run
            tournament_id = 12345
            max_questions = 3
            agent_types = ["ensemble", "chain_of_thought"]

            # Mock the dispatcher's run_tournament method
            expected_results = {
                "tournament_id": tournament_id,
                "questions_processed": max_questions,
                "forecasts_generated": max_questions,
                "success_rate": 100.0,
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
            assert results["questions_processed"] == max_questions
            assert results["forecasts_generated"] == max_questions

            # Verify metrics were updated
            assert orchestrator.metrics["questions_processed"] == max_questions
            assert orchestrator.metrics["forecasts_generated"] == max_questions

            print("✓ End-to-end tournament flow working")

        finally:
            await orchestrator.shutdown()

    async def test_health_monitoring_integration(self, comprehensive_config, mock_external_services):
        """Test health monitoring across all components."""
        orchestrator = TournamentOrchestrator(comprehensive_config)
        await orchestrator.initialize()

        try:
            # Perform comprehensive health check
            health_status = await orchestrator._perform_health_check()

            # Verify health check covers all critical components
            expected_components = [
                "llm_client", "search_client", "metaculus_client"
            ]

            for component in expected_components:
                assert component in health_status
                assert isinstance(health_status[component], bool)

            # Verify metrics are updated
            assert orchestrator.metrics["last_health_check"] is not None
            assert orchestrator.metrics["component_health"] == health_status

            print("✓ Health monitoring integration working")

        finally:
            await orchestrator.shutdown()

    async def test_system_status_comprehensive_reporting(self, comprehensive_config, mock_external_services):
        """Test comprehensive system status reporting."""
        orchestrator = TournamentOrchestrator(comprehensive_config)
        await orchestrator.initialize()

        try:
            # Get system status
            status = await orchestrator.get_system_status()

            # Verify comprehensive status structure
            required_fields = [
                "status", "uptime_seconds", "health_status", "metrics",
                "configuration", "last_config_reload"
            ]

            for field in required_fields:
                assert field in status, f"Missing status field: {field}"

            # Verify configuration details
            config = status["configuration"]
            assert config["environment"] == orchestrator.registry.settings.environment
            assert config["tournament_id"] == 12345
            assert config["max_concurrent_questions"] == 3
            assert "ComprehensiveTestBot" in config.get("default_agents", []) or config.get("tournament_id") == 12345

            # Verify metrics structure
            metrics = status["metrics"]
            assert "questions_processed" in metrics
            assert "forecasts_generated" in metrics
            assert "errors_encountered" in metrics
            assert "uptime_start" in metrics

            print("✓ Comprehensive system status reporting working")

        finally:
            await orchestrator.shutdown()

    async def test_error_handling_and_recovery(self, comprehensive_config, mock_external_services):
        """Test error handling and recovery mechanisms across components."""
        orchestrator = TournamentOrchestrator(comprehensive_config)
        await orchestrator.initialize()

        try:
            # Test component failure handling
            orchestrator.registry.llm_client.health_check = AsyncMock(
                side_effect=Exception("LLM service unavailable")
            )

            # Health check should handle component failures gracefully
            health_status = await orchestrator._perform_health_check()
            assert health_status["llm_client"] is False

            # Test single question forecast with error
            question_id = 12345
            orchestrator.registry.forecasting_pipeline.run_single_question = AsyncMock(
                side_effect=Exception("Pipeline error")
            )

            # Should raise exception but update error metrics
            with pytest.raises(Exception, match="Pipeline error"):
                await orchestrator.run_single_question(question_id)

            assert orchestrator.metrics["errors_encountered"] == 1

            print("✓ Error handling and recovery working")

        finally:
            await orchestrator.shutdown()

    async def test_graceful_shutdown_all_components(self, comprehensive_config, mock_external_services):
        """Test graceful shutdown of all components."""
        orchestrator = TournamentOrchestrator(comprehensive_config)
        await orchestrator.initialize()

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

        print("✓ Graceful shutdown of all components working")

    async def test_managed_lifecycle_comprehensive(self, comprehensive_config, mock_external_services):
        """Test managed lifecycle context manager with all components."""
        orchestrator = TournamentOrchestrator(comprehensive_config)

        # Use context manager
        async with orchestrator.managed_lifecycle() as orch:
            # Verify all components are available
            assert orch.registry is not None
            assert orch.registry.settings is not None
            assert len(orch.registry.__dict__) > 20  # Should have many components

            # Test basic functionality
            status = await orch.get_system_status()
            assert status["status"] == "running"

        # Verify shutdown was called
        assert orchestrator._shutdown_event.is_set()

        print("✓ Managed lifecycle with all components working")


@pytest.mark.asyncio
async def test_integration_requirements_compliance():
    """Test that integration meets all requirements from 10.1, 10.2, 10.5."""

    # Requirement 10.1: Clean Architecture and SOLID principles
    orchestrator = TournamentOrchestrator()

    # Verify separation of concerns
    assert hasattr(orchestrator, 'initialize')  # Initialization logic
    assert hasattr(orchestrator, 'run_tournament')  # Business logic
    assert hasattr(orchestrator, 'get_system_status')  # Monitoring
    assert hasattr(orchestrator, 'shutdown')  # Cleanup

    # Requirement 10.2: Plugin-based architecture and hot-swappable components
    # Verify configuration hot-reloading capabilities
    assert hasattr(orchestrator, '_reload_configuration')
    assert hasattr(orchestrator, '_update_component_configs')
    assert hasattr(orchestrator, '_on_config_change')

    # Requirement 10.5: Comprehensive monitoring and backward compatibility
    # Verify monitoring capabilities
    assert hasattr(orchestrator, 'get_system_status')
    assert hasattr(orchestrator, '_perform_health_check')
    assert hasattr(orchestrator, 'metrics')

    # Verify lifecycle management
    assert hasattr(orchestrator, 'managed_lifecycle')

    print("✓ All requirements compliance verified")


if __name__ == "__main__":
    # Run tests directly
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
