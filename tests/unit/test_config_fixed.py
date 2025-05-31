"""Unit tests for configuration settings - Fixed to match actual implementation."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.infrastructure.config.settings import (
    Settings,
    Config,
    DatabaseConfig,
    LLMConfig,
    SearchConfig,
    MetaculusConfig,
    AgentConfig,
    EnsembleConfig,
    PipelineConfig,
    BotConfig,
    LoggingConfig,
    AggregationMethod,
    get_settings,
    set_settings,
    reload_settings
)


class TestAggregationMethod:
    """Test AggregationMethod enum."""
    
    def test_aggregation_method_values(self):
        """Test that aggregation method enum has expected values."""
        assert AggregationMethod.SIMPLE_AVERAGE.value == "simple_average"
        assert AggregationMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert AggregationMethod.MEDIAN.value == "median"
        assert AggregationMethod.CONFIDENCE_WEIGHTED.value == "confidence_weighted"
        assert AggregationMethod.META_REASONING.value == "meta_reasoning"


class TestDatabaseConfig:
    """Test DatabaseConfig class."""
    
    def test_database_config_defaults(self):
        """Test database configuration with default values."""
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "forecasting_bot"
        assert config.username == "postgres"
        assert config.password == ""
        assert config.min_connections == 1
        assert config.max_connections == 10
        assert config.connection_timeout == 30.0


class TestLLMConfig:
    """Test LLMConfig class."""
    
    def test_llm_config_defaults(self):
        """Test LLM configuration with default values."""
        config = LLMConfig()
        
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.3
        assert config.api_key == ""
        assert config.backup_models == ["gpt-3.5-turbo"]
        assert config.max_retries == 3
        assert config.timeout == 60.0
        assert config.use_structured_output is True


class TestSearchConfig:
    """Test SearchConfig class."""
    
    def test_search_config_defaults(self):
        """Test search configuration with default values."""
        config = SearchConfig()
        
        assert config.provider == "multi_source"
        assert config.max_results == 10
        assert config.timeout == 30.0
        assert config.duckduckgo_enabled is True
        assert config.wikipedia_enabled is True
        assert config.concurrent_searches is True
        assert config.deduplicate_results is True


class TestMetaculusConfig:
    """Test MetaculusConfig class."""
    
    def test_metaculus_config_defaults(self):
        """Test Metaculus configuration with default values."""
        config = MetaculusConfig()
        
        assert config.api_token == ""
        assert config.base_url == "https://www.metaculus.com/api2"
        assert config.submit_predictions is False
        assert config.dry_run is True
        assert config.include_reasoning is True
        assert config.timeout == 30.0


class TestAgentConfig:
    """Test AgentConfig class."""
    
    def test_agent_config_defaults(self):
        """Test agent configuration with default values."""
        config = AgentConfig()
        
        assert config.enabled is True
        assert config.weight == 1.0
        assert config.confidence_threshold == 0.5
        assert config.react_max_iterations == 5
        assert config.chain_of_thought_steps == 3
        assert config.tree_of_thought_depth == 3
        assert config.auto_cot_examples == 3


class TestEnsembleConfig:
    """Test EnsembleConfig class."""
    
    def test_ensemble_config_defaults(self):
        """Test ensemble configuration with default values."""
        config = EnsembleConfig()
        
        assert config.aggregation_method == "confidence_weighted"
        assert config.min_agents == 2
        assert config.confidence_threshold == 0.6
        assert config.use_meta_reasoning is True
        assert config.fallback_to_single_agent is True


class TestPipelineConfig:
    """Test PipelineConfig class."""
    
    def test_pipeline_config_defaults(self):
        """Test pipeline configuration with default values."""
        config = PipelineConfig()
        
        assert config.max_concurrent_questions == 5
        assert config.batch_delay_seconds == 1.0
        assert config.default_research_depth == 3
        assert config.default_agent_names == ["ensemble"]
        assert config.enable_benchmarking is True


class TestBotConfig:
    """Test BotConfig class."""
    
    def test_bot_config_defaults(self):
        """Test bot configuration with default values."""
        config = BotConfig()
        
        assert config.name == "MetaculusBotHA"
        assert config.version == "1.0.0"
        assert config.research_reports_per_question == 2
        assert config.predictions_per_research_report == 3
        assert config.publish_reports_to_metaculus is False
        assert config.max_concurrent_questions == 2


class TestLoggingConfig:
    """Test LoggingConfig class."""
    
    def test_logging_config_defaults(self):
        """Test logging configuration with default values."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.console_output is True
        assert config.enable_structured_logging is True
        assert config.log_predictions is True


class TestConfig:
    """Test Config class."""
    
    def test_config_defaults(self):
        """Test Config instantiation with defaults."""
        config = Config()
        
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.metaculus, MetaculusConfig)
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.ensemble, EnsembleConfig)
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.bot, BotConfig)
        assert isinstance(config.logging, LoggingConfig)


class TestSettings:
    """Test Settings class."""
    
    def test_settings_defaults(self):
        """Test settings with default values."""
        settings = Settings()
        
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.llm, LLMConfig)
        assert isinstance(settings.search, SearchConfig)
        assert isinstance(settings.metaculus, MetaculusConfig)
        assert isinstance(settings.agent, AgentConfig)
        assert isinstance(settings.ensemble, EnsembleConfig)
        assert isinstance(settings.pipeline, PipelineConfig)
        assert isinstance(settings.bot, BotConfig)
        assert isinstance(settings.logging, LoggingConfig)
        assert settings.environment == "development"
        assert settings.debug is False
    
    def test_settings_from_env(self):
        """Test settings loading from environment variables."""
        env_vars = {
            "LLM_MODEL": "gpt-3.5-turbo",
            "LLM_TEMPERATURE": "0.7",
            "METACULUS_API_TOKEN": "test-token",
            "DATABASE_HOST": "testhost",
            "DATABASE_PORT": "5433"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings.load_from_env()
            
            assert settings.llm.model == "gpt-3.5-turbo"
            assert settings.llm.temperature == 0.7
            assert settings.metaculus.api_token == "test-token"
            assert settings.database.host == "testhost"
            assert settings.database.port == 5433
    
    def test_settings_helper_methods(self):
        """Test settings helper methods."""
        settings = Settings()
        
        assert settings.is_development() is True
        assert settings.is_production() is False
        assert settings.is_testing() is False
        
        # Test API key getter
        settings.llm.openai_api_key = "test-openai-key"
        assert settings.get_api_key("openai") == "test-openai-key"
        assert settings.get_api_key("unknown") == ""
    
    def test_settings_from_config(self):
        """Test creating Settings from Config instance."""
        config = Config()
        settings = Settings.from_config(config)
        
        assert isinstance(settings, Settings)
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.llm, LLMConfig)
    
    def test_global_settings_functions(self):
        """Test global settings functions."""
        # Test get_settings creates a default instance
        settings = get_settings()
        assert isinstance(settings, Settings)
        
        # Test set_settings
        custom_settings = Settings()
        custom_settings.environment = "test"
        set_settings(custom_settings)
        
        retrieved_settings = get_settings()
        assert retrieved_settings.environment == "test"
        
        # Reset for other tests
        set_settings(Settings())


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_data = {
        "llm": {
            "model": "gpt-4",
            "temperature": 0.5,
            "openai_api_key": "test-key"
        },
        "database": {
            "host": "testhost",
            "port": 5433
        },
        "bot": {
            "name": "TestBot",
            "version": "1.0.0"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    yield Path(temp_path)
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestSettingsWithYAML:
    """Test Settings with YAML configuration files."""
    
    def test_settings_from_yaml(self, temp_config_file):
        """Test loading settings from YAML file."""
        settings = Settings.load_from_yaml(temp_config_file)
        
        assert settings.llm.model == "gpt-4"
        assert settings.llm.temperature == 0.5
        assert settings.database.host == "testhost"
        assert settings.database.port == 5433
        assert settings.bot.name == "TestBot"
    
    def test_settings_yaml_with_env_override(self, temp_config_file):
        """Test YAML config with environment variable override."""
        with patch.dict(os.environ, {"LLM_MODEL": "gpt-3.5-turbo"}):
            settings = Settings.load_from_yaml(temp_config_file)
            
            # Environment variable should override YAML
            assert settings.llm.model == "gpt-3.5-turbo"
            # YAML values should still be used for non-overridden values
            assert settings.llm.temperature == 0.5
    
    def test_reload_settings(self, temp_config_file):
        """Test reloading settings."""
        # Set initial settings
        initial_settings = Settings()
        set_settings(initial_settings)
        
        # Reload with config file
        reloaded_settings = reload_settings(temp_config_file)
        
        assert reloaded_settings.llm.model == "gpt-4"
        assert reloaded_settings.llm.temperature == 0.5
