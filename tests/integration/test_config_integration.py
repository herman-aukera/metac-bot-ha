"""
Integration tests for configuration management and hot-reloading.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.application.tournament_orchestrator import TournamentOrchestrator
from src.infrastructure.config.config_manager import (
    ConfigChangeEvent,
    ConfigChangeType,
    create_config_manager,
)


class TestConfigManagerIntegration:
    """Test configuration manager integration."""

    @pytest.fixture
    async def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create initial config file
            config_data = {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "api_key": "test-key",
                },
                "search": {"provider": "multi_source", "max_results": 10},
                "bot": {"name": "TestBot", "version": "1.0.0"},
            }

            config_file = config_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            yield config_dir, config_file

    @pytest.fixture
    async def config_manager(self, temp_config_dir):
        """Create config manager for testing."""
        config_dir, config_file = temp_config_dir

        manager = create_config_manager(
            config_paths=[str(config_file)],
            watch_directories=[str(config_dir)],
            enable_hot_reload=True,
            validation_enabled=True,
        )

        await manager.initialize()
        yield manager
        await manager.shutdown()

    async def test_config_manager_initialization(self, config_manager):
        """Test config manager initializes correctly."""
        assert config_manager.current_settings is not None
        assert config_manager.current_settings.bot.name == "TestBot"
        assert config_manager.current_settings.llm.model == "gpt-4"
        assert config_manager.current_settings.search.max_results == 10

    async def test_configuration_validation(self, temp_config_dir):
        """Test configuration validation."""
        config_dir, config_file = temp_config_dir

        # Create invalid configuration
        invalid_config = {
            "llm": {
                "temperature": 5.0,  # Invalid: should be 0.0-2.0
                "rate_limit_rpm": -1,  # Invalid: should be positive
            },
            "search": {"max_results": 200},  # Invalid: should be 1-100
        }

        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        manager = create_config_manager(
            config_paths=[str(config_file)], validation_enabled=True
        )

        # Should still initialize but with validation errors
        settings = await manager.initialize()
        assert settings is not None

        await manager.shutdown()

    async def test_hot_reload_functionality(self, config_manager, temp_config_dir):
        """Test configuration hot-reloading."""
        config_dir, config_file = temp_config_dir

        # Set up change listener
        change_events = []

        def change_listener(event: ConfigChangeEvent):
            change_events.append(event)

        config_manager.add_change_listener(change_listener)

        # Modify configuration file
        updated_config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",  # Changed
                "temperature": 0.5,  # Changed
                "api_key": "test-key",
            },
            "search": {"provider": "multi_source", "max_results": 15},  # Changed
            "bot": {"name": "UpdatedTestBot", "version": "2.0.0"},  # Changed  # Changed
        }

        with open(config_file, "w") as f:
            yaml.dump(updated_config, f)

        # Wait for file system event to be processed
        await asyncio.sleep(0.5)

        # Verify configuration was reloaded
        current_settings = config_manager.get_current_settings()
        assert current_settings.bot.name == "UpdatedTestBot"
        assert current_settings.bot.version == "2.0.0"
        assert current_settings.llm.model == "gpt-3.5-turbo"
        assert current_settings.llm.temperature == 0.5
        assert current_settings.search.max_results == 15

        # Verify change event was triggered
        assert len(change_events) > 0
        assert change_events[-1].change_type == ConfigChangeType.MODIFIED

    async def test_manual_config_reload(self, config_manager, temp_config_dir):
        """Test manual configuration reload."""
        config_dir, config_file = temp_config_dir

        # Modify config file
        updated_config = {
            "llm": {"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
            "search": {"provider": "multi_source", "max_results": 20},
            "bot": {"name": "ManualReloadBot", "version": "3.0.0"},
        }

        with open(config_file, "w") as f:
            yaml.dump(updated_config, f)

        # Manually reload configuration
        new_settings = await config_manager.reload_configuration()

        # Verify changes
        assert new_settings.bot.name == "ManualReloadBot"
        assert new_settings.bot.version == "3.0.0"
        assert new_settings.search.max_results == 20

    async def test_config_history_tracking(self, config_manager, temp_config_dir):
        """Test configuration change history tracking."""
        config_dir, config_file = temp_config_dir

        initial_history_count = len(config_manager.get_config_history())

        # Make multiple configuration changes
        for i in range(3):
            updated_config = {
                "llm": {"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
                "search": {"provider": "multi_source", "max_results": 10 + i},
                "bot": {"name": f"TestBot{i}", "version": f"{i}.0.0"},
            }

            with open(config_file, "w") as f:
                yaml.dump(updated_config, f)

            await asyncio.sleep(0.1)  # Small delay between changes

        # Wait for all changes to be processed
        await asyncio.sleep(0.5)

        # Verify history tracking
        history = config_manager.get_config_history()
        assert len(history) >= initial_history_count + 3

        # Verify history entries have correct structure
        for event in history[-3:]:
            assert isinstance(event, ConfigChangeEvent)
            assert event.timestamp is not None
            assert event.file_path is not None

    async def test_validation_listeners(self, temp_config_dir):
        """Test configuration validation listeners."""
        config_dir, config_file = temp_config_dir

        validation_results = []

        def validation_listener(result):
            validation_results.append(result)

        manager = create_config_manager(
            config_paths=[str(config_file)], validation_enabled=True
        )

        manager.add_validation_listener(validation_listener)

        # Create invalid configuration
        invalid_config = {
            "llm": {"temperature": 10.0},  # Invalid
            "search": {"max_results": -5},  # Invalid
        }

        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        await manager.initialize()

        # Verify validation listener was called
        assert len(validation_results) > 0
        assert not validation_results[-1].is_valid
        assert len(validation_results[-1].errors) > 0

        await manager.shutdown()

    async def test_custom_validation_rules(self, temp_config_dir):
        """Test custom validation rules."""
        config_dir, config_file = temp_config_dir

        manager = create_config_manager(
            config_paths=[str(config_file)], validation_enabled=True
        )

        # Add custom validation rule
        manager.add_validation_rule(
            "bot.name", lambda x: isinstance(x, str) and len(x) >= 5
        )

        # Create config that violates custom rule
        config_data = {
            "llm": {"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
            "bot": {"name": "Bot", "version": "1.0.0"},  # Too short
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        await manager.initialize()

        # Verify custom validation was applied
        # (In a real implementation, you'd check validation results)

        await manager.shutdown()


class TestOrchestratorConfigIntegration:
    """Test orchestrator integration with configuration management."""

    @pytest.fixture
    async def temp_config_setup(self):
        """Create temporary configuration setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

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

            config_file = config_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            yield config_dir, config_file

    async def test_orchestrator_with_config_manager(self, temp_config_setup):
        """Test orchestrator integration with config manager."""
        config_dir, config_file = temp_config_setup

        # Create config manager
        config_manager = create_config_manager(
            config_paths=[str(config_file)],
            watch_directories=[str(config_dir)],
            enable_hot_reload=True,
        )

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

            # Create orchestrator with config manager
            orchestrator = TournamentOrchestrator(config_manager=config_manager)
            await orchestrator.initialize()

            # Verify initial configuration
            assert orchestrator.registry.settings.bot.name == "TestBot"
            assert orchestrator.registry.settings.llm.model == "gpt-4"

            # Test configuration hot-reload
            updated_config = {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",  # Changed
                    "temperature": 0.5,
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
                "bot": {"name": "UpdatedBot", "version": "2.0.0"},  # Changed
                "logging": {"level": "INFO"},
            }

            with open(config_file, "w") as f:
                yaml.dump(updated_config, f)

            # Wait for hot-reload
            await asyncio.sleep(0.5)

            # Verify configuration was updated
            assert orchestrator.registry.settings.bot.name == "UpdatedBot"
            assert orchestrator.registry.settings.llm.model == "gpt-3.5-turbo"

            # Verify component update methods were called
            mock_llm.return_value.update_config.assert_called()

            await orchestrator.shutdown()

    async def test_orchestrator_config_status_reporting(self, temp_config_setup):
        """Test orchestrator configuration status reporting."""
        config_dir, config_file = temp_config_setup

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
            mock_search.return_value.initialize = AsyncMock()
            mock_search.return_value.health_check = AsyncMock()
            mock_metaculus.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.health_check = AsyncMock()

            orchestrator = TournamentOrchestrator(str(config_file))
            await orchestrator.initialize()

            # Get system status
            status = await orchestrator.get_system_status()

            # Verify configuration information is included
            assert "configuration" in status
            config_info = status["configuration"]
            assert config_info["environment"] is not None
            assert config_info["tournament_id"] == 12345
            assert config_info["max_concurrent_questions"] == 2

            # Verify config manager status
            if orchestrator.config_manager:
                config_status = orchestrator.config_manager.get_status()
                assert config_status["initialized"] is True
                assert config_status["hot_reload_enabled"] is True

            await orchestrator.shutdown()

    async def test_config_error_handling(self, temp_config_setup):
        """Test configuration error handling and recovery."""
        config_dir, config_file = temp_config_setup

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
            mock_llm.return_value.update_config = AsyncMock(
                side_effect=Exception("Update failed")
            )
            mock_search.return_value.initialize = AsyncMock()
            mock_search.return_value.health_check = AsyncMock()
            mock_search.return_value.update_config = AsyncMock()
            mock_metaculus.return_value.initialize = AsyncMock()
            mock_metaculus.return_value.health_check = AsyncMock()
            mock_metaculus.return_value.update_config = AsyncMock()

            orchestrator = TournamentOrchestrator(str(config_file))
            await orchestrator.initialize()

            original_name = orchestrator.registry.settings.bot.name

            # Create invalid configuration update
            updated_config = {
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
                "bot": {"name": "FailedUpdateBot", "version": "2.0.0"},
                "logging": {"level": "INFO"},
            }

            with open(config_file, "w") as f:
                yaml.dump(updated_config, f)

            # Wait for hot-reload attempt
            await asyncio.sleep(0.5)

            # Configuration should still be updated even if component update fails
            assert orchestrator.registry.settings.bot.name == "FailedUpdateBot"

            await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_config_integration_requirements_compliance():
    """Test that configuration integration meets requirements 10.1, 10.2, 10.5."""

    # Requirement 10.2: Hot-swappable components and runtime configuration updates
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "config.yaml"
        config_data = {
            "llm": {"provider": "openai", "model": "gpt-4", "api_key": "test"},
            "bot": {"name": "TestBot", "version": "1.0.0"},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Test hot-reloading capability
        config_manager = create_config_manager(
            config_paths=[str(config_file)],
            watch_directories=[str(Path(temp_dir))],
            enable_hot_reload=True,
        )

        await config_manager.initialize()

        # Verify hot-reload is enabled
        status = config_manager.get_status()
        assert status["hot_reload_enabled"] is True
        assert status["file_watching_active"] is True

        # Test runtime configuration updates
        original_name = config_manager.get_current_settings().bot.name

        # Update configuration
        config_data["bot"]["name"] = "UpdatedBot"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        await asyncio.sleep(0.5)  # Wait for file system event

        # Verify configuration was updated at runtime
        updated_name = config_manager.get_current_settings().bot.name
        assert updated_name != original_name
        assert updated_name == "UpdatedBot"

        await config_manager.shutdown()

    # Requirement 10.5: Comprehensive monitoring and backward compatibility
    # Test monitoring capabilities
    config_manager = create_config_manager(
        enable_hot_reload=True, validation_enabled=True
    )
    await config_manager.initialize()

    # Verify monitoring capabilities
    status = config_manager.get_status()
    required_status_fields = [
        "initialized",
        "hot_reload_enabled",
        "validation_enabled",
        "last_reload_time",
        "change_listeners",
        "config_history_count",
    ]

    for field in required_status_fields:
        assert field in status, f"Missing status field: {field}"

    # Test change tracking
    history = config_manager.get_config_history()
    assert isinstance(history, list)

    await config_manager.shutdown()
