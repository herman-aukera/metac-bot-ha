#!/usr/bin/env python3
"""
Simple validation script for integration testing without pytest.
"""
import asyncio
import sys
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.application.tournament_orchestrator import TournamentOrchestrator
from src.infrastructure.config.config_manager import create_config_manager


async def test_basic_orchestrator_integration():
    """Test basic orchestrator integration."""
    print("Testing basic orchestrator integration...")

    # Create temporary config
    config_data = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.3,
            "api_key": "test-key",
        },
        "search": {"provider": "multi_source", "max_results": 10},
        "metaculus": {
            "base_url": "https://test.metaculus.com/api",
            "tournament_id": 12345,
            "dry_run": True,
        },
        "pipeline": {
            "max_concurrent_questions": 2,
            "default_agent_names": ["ensemble"],
        },
        "bot": {"name": "TestBot", "version": "1.0.0"},
        "logging": {"level": "INFO"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        with (
            patch("src.infrastructure.external_apis.llm_client.LLMClient") as mock_llm,
            patch(
                "src.infrastructure.external_apis.search_client.SearchClient"
            ) as mock_search,
            patch(
                "src.infrastructure.external_apis.metaculus_client.MetaculusClient"
            ) as mock_metaculus,
        ):

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

            # Test orchestrator initialization
            orchestrator = TournamentOrchestrator(config_path)
            await orchestrator.initialize()

            # Verify components are initialized
            assert orchestrator.registry is not None
            assert orchestrator.registry.settings is not None
            assert orchestrator.registry.settings.bot.name == "TestBot"

            print("✓ Orchestrator initialized successfully")

            # Test health check
            health_status = await orchestrator._perform_health_check()
            assert isinstance(health_status, dict)
            print("✓ Health check completed")

            # Test system status
            status = await orchestrator.get_system_status()
            assert status["status"] == "running"
            assert "configuration" in status
            print("✓ System status reporting works")

            # Test graceful shutdown
            await orchestrator.shutdown()
            print("✓ Graceful shutdown completed")

    finally:
        # Cleanup
        Path(config_path).unlink(missing_ok=True)

    print("Basic orchestrator integration test passed!")


async def test_config_manager_integration():
    """Test configuration manager integration."""
    print("\nTesting configuration manager integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        config_file = config_dir / "config.yaml"

        # Create initial config
        config_data = {
            "llm": {"provider": "openai", "model": "gpt-4", "api_key": "test"},
            "bot": {"name": "TestBot", "version": "1.0.0"},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Test config manager
        config_manager = create_config_manager(
            config_paths=[str(config_file)],
            watch_directories=[str(config_dir)],
            enable_hot_reload=False,  # Disable for testing
            validation_enabled=True,
        )

        settings = await config_manager.initialize()
        assert settings.bot.name == "TestBot"
        print("✓ Config manager initialized")

        # Test manual reload
        config_data["bot"]["name"] = "UpdatedBot"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        new_settings = await config_manager.reload_configuration()
        assert new_settings.bot.name == "UpdatedBot"
        print("✓ Manual configuration reload works")

        # Test status reporting
        status = config_manager.get_status()
        assert status["initialized"] is True
        print("✓ Status reporting works")

        await config_manager.shutdown()
        print("✓ Config manager shutdown completed")

    print("Configuration manager integration test passed!")


async def test_dependency_injection():
    """Test dependency injection works correctly."""
    print("\nTesting dependency injection...")

    config_data = {
        "llm": {"provider": "openai", "model": "gpt-4", "api_key": "test"},
        "search": {"provider": "multi_source"},
        "metaculus": {
            "base_url": "https://test.metaculus.com/api",
            "tournament_id": 12345,
        },
        "pipeline": {
            "max_concurrent_questions": 2,
            "default_agent_names": ["ensemble"],
        },
        "bot": {"name": "TestBot", "version": "1.0.0"},
        "logging": {"level": "INFO"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        with (
            patch("src.infrastructure.external_apis.llm_client.LLMClient") as mock_llm,
            patch(
                "src.infrastructure.external_apis.search_client.SearchClient"
            ) as mock_search,
            patch(
                "src.infrastructure.external_apis.metaculus_client.MetaculusClient"
            ) as mock_metaculus,
        ):

            # Configure mocks
            mock_llm.return_value.initialize = AsyncMock()
            mock_search.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.initialize = AsyncMock()

            orchestrator = TournamentOrchestrator(config_path)
            await orchestrator.initialize()

            registry = orchestrator.registry

            # Verify dependency injection
            required_components = [
                "settings",
                "llm_client",
                "search_client",
                "metaculus_client",
                "circuit_breaker",
                "rate_limiter",
                "health_monitor",
                "retry_manager",
                "reasoning_logger",
                "dispatcher",
                "forecast_service",
                "ingestion_service",
                "ensemble_service",
                "forecasting_service",
                "research_service",
                "tournament_analytics",
                "performance_tracking",
                "calibration_service",
                "risk_management_service",
                "forecasting_pipeline",
            ]

            for component in required_components:
                assert hasattr(registry, component), f"Missing component: {component}"
                assert (
                    getattr(registry, component) is not None
                ), f"Component is None: {component}"

            print("✓ All required components are present")

            # Verify cross-component dependencies
            assert registry.research_service.search_client == registry.search_client
            assert registry.research_service.llm_client == registry.llm_client
            print("✓ Cross-component dependencies are correctly injected")

            await orchestrator.shutdown()

    finally:
        Path(config_path).unlink(missing_ok=True)

    print("Dependency injection test passed!")


async def main():
    """Run all integration tests."""
    print("Starting integration validation...")

    try:
        await test_basic_orchestrator_integration()
        await test_config_manager_integration()
        await test_dependency_injection()

        print("\n🎉 All integration tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
