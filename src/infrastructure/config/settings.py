"""Configuration management for the AI forecasting bot."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum

import yaml
from dotenv import load_dotenv


class AggregationMethod(Enum):
    """Ensemble aggregation methods."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    MEDIAN = "median"
    META_REASONING = "meta_reasoning"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "forecasting_bot"
    username: str = "postgres"
    password: str = ""
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: float = 30.0


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    max_retries: int = 3
    timeout: float = 60.0
    rate_limit_rpm: int = 60
    
    # Model-specific configurations
    backup_models: List[str] = field(default_factory=lambda: ["gpt-3.5-turbo"])
    use_structured_output: bool = True
    response_format: Optional[str] = None


@dataclass  
class SearchConfig:
    """Search configuration."""
    provider: str = "multi_source"
    max_results: int = 10
    timeout: float = 30.0
    
    # API Keys
    serpapi_key: str = ""
    duckduckgo_enabled: bool = True
    wikipedia_enabled: bool = True
    
    # Search behavior
    concurrent_searches: bool = True
    deduplicate_results: bool = True
    result_cache_ttl: int = 3600  # seconds
    max_content_length: int = 10000


@dataclass
class MetaculusConfig:
    """Metaculus API configuration."""
    username: str = ""
    password: str = ""
    api_token: str = ""
    base_url: str = "https://www.metaculus.com/api2"
    tournament_id: Optional[int] = None
    timeout: float = 30.0
    
    # Prediction behavior
    submit_predictions: bool = False
    dry_run: bool = True
    include_reasoning: bool = True
    max_prediction_retries: int = 3


@dataclass
class AgentConfig:
    """Individual agent configuration."""
    enabled: bool = True
    weight: float = 1.0
    confidence_threshold: float = 0.5
    max_retries: int = 2
    timeout: float = 300.0  # 5 minutes
    
    # Agent-specific parameters
    chain_of_thought_steps: int = 3
    tree_of_thought_depth: int = 3
    tree_of_thought_breadth: int = 3
    react_max_iterations: int = 5
    auto_cot_examples: int = 3


@dataclass
class EnsembleConfig:
    """Ensemble agent configuration."""
    aggregation_method: str = "confidence_weighted"
    min_agents: int = 2
    confidence_threshold: float = 0.6
    use_meta_reasoning: bool = True
    fallback_to_single_agent: bool = True
    
    # Agent weights (if using weighted aggregation)
    agent_weights: Dict[str, float] = field(default_factory=lambda: {
        "chain_of_thought": 1.0,
        "tree_of_thought": 1.2,
        "react": 1.1,
        "auto_cot": 0.9
    })


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    max_concurrent_questions: int = 5
    batch_delay_seconds: float = 1.0
    default_research_depth: int = 3
    default_agent_names: List[str] = field(default_factory=lambda: ["ensemble"])
    
    # Health checks
    health_check_interval: int = 60  # seconds
    max_failed_health_checks: int = 3
    
    # Performance monitoring
    enable_benchmarking: bool = True
    benchmark_output_path: Optional[str] = None
    
    # Error handling
    max_retries_per_question: int = 3
    retry_delay_seconds: float = 2.0
    enable_circuit_breaker: bool = True


@dataclass
class BotConfig:
    """Bot-specific configuration."""
    name: str = "MetaculusBotHA"
    version: str = "1.0.0"
    research_reports_per_question: int = 2
    predictions_per_research_report: int = 3
    publish_reports_to_metaculus: bool = False
    max_concurrent_questions: int = 2
    
    # Research behavior
    enable_deep_research: bool = True
    research_timeout_minutes: int = 10
    min_sources_per_topic: int = 3
    
    # Prediction behavior
    require_confidence_score: bool = True
    min_confidence_threshold: float = 0.5
    enable_uncertainty_quantification: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    file_path: Optional[str] = None
    console_output: bool = True
    
    # Advanced logging
    enable_structured_logging: bool = True
    log_predictions: bool = True
    log_research_data: bool = False  # Can be large
    max_log_size_mb: int = 100
    backup_count: int = 5


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[Path] = None):
        # Load environment variables
        load_dotenv()
        
        # Load YAML config if provided
        self.yaml_config = {}
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.yaml_config = yaml.safe_load(f) or {}
            except ImportError:
                # If PyYAML is not available, just use environment variables
                self.yaml_config = {}
        
        # Initialize configurations
        self.database = self._load_database_config()
        self.llm = self._load_llm_config()
        self.search = self._load_search_config()
        self.metaculus = self._load_metaculus_config()
        self.pipeline = self._load_pipeline_config()
        self.bot = self._load_bot_config()
        self.logging = self._load_logging_config()
        self.agent = self._load_agent_config()
        self.ensemble = self._load_ensemble_config()
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration."""
        return DatabaseConfig(
            host=self._get_config_value("database.host", "DATABASE_HOST", "localhost"),
            port=self._get_config_int("database.port", "DATABASE_PORT", 5432),
            database=self._get_config_value("database.name", "DATABASE_NAME", "forecasting_bot"),
            username=self._get_config_value("database.username", "DATABASE_USERNAME", "postgres"),
            password=self._get_config_value("database.password", "DATABASE_PASSWORD", ""),
            min_connections=self._get_config_int("database.min_connections", "DATABASE_MIN_CONNECTIONS", 1),
            max_connections=self._get_config_int("database.max_connections", "DATABASE_MAX_CONNECTIONS", 10),
            connection_timeout=self._get_config_float("database.connection_timeout", "DATABASE_CONNECTION_TIMEOUT", 30.0),
        )
    
    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration."""
        return LLMConfig(
            provider=self._get_config_value("llm.provider", "LLM_PROVIDER", "openai"),
            model=self._get_config_value("llm.model", "LLM_MODEL", "gpt-4"),
            temperature=self._get_config_float("llm.temperature", "LLM_TEMPERATURE", 0.3),
            max_tokens=self._get_optional_int("llm.max_tokens", "LLM_MAX_TOKENS"),
            api_key=self._get_config_value("llm.api_key", "OPENAI_API_KEY", ""),
            openai_api_key=self._get_config_value("llm.openai_api_key", "OPENAI_API_KEY", ""),
            anthropic_api_key=self._get_config_value("llm.anthropic_api_key", "ANTHROPIC_API_KEY", ""),
            openrouter_api_key=self._get_config_value("llm.openrouter_api_key", "OPENROUTER_API_KEY", ""),
            max_retries=self._get_config_int("llm.max_retries", "LLM_MAX_RETRIES", 3),
            timeout=self._get_config_float("llm.timeout", "LLM_TIMEOUT", 60.0),
            rate_limit_rpm=self._get_config_int("llm.rate_limit_rpm", "LLM_RATE_LIMIT_RPM", 60),
            backup_models=self._get_config_list("llm.backup_models", "LLM_BACKUP_MODELS", ["gpt-3.5-turbo"]),
            use_structured_output=self._get_config_bool("llm.use_structured_output", "LLM_USE_STRUCTURED_OUTPUT", True),
            response_format=self._get_config_optional_str("llm.response_format", "LLM_RESPONSE_FORMAT"),
        )
    
    def _load_search_config(self) -> SearchConfig:
        """Load search configuration."""
        return SearchConfig(
            provider=self._get_config_value("search.provider", "SEARCH_PROVIDER", "multi_source"),
            max_results=self._get_config_int("search.max_results", "SEARCH_MAX_RESULTS", 10),
            timeout=self._get_config_float("search.timeout", "SEARCH_TIMEOUT", 30.0),
            serpapi_key=self._get_config_value("search.serpapi_key", "SERPAPI_KEY", ""),
            duckduckgo_enabled=self._get_config_bool("search.duckduckgo_enabled", "SEARCH_DUCKDUCKGO_ENABLED", True),
            wikipedia_enabled=self._get_config_bool("search.wikipedia_enabled", "SEARCH_WIKIPEDIA_ENABLED", True),
            concurrent_searches=self._get_config_bool("search.concurrent_searches", "SEARCH_CONCURRENT", True),
            deduplicate_results=self._get_config_bool("search.deduplicate_results", "SEARCH_DEDUPLICATE", True),
            result_cache_ttl=self._get_config_int("search.result_cache_ttl", "SEARCH_CACHE_TTL", 3600),
            max_content_length=self._get_config_int("search.max_content_length", "SEARCH_MAX_CONTENT_LENGTH", 10000),
        )
    
    def _load_metaculus_config(self) -> MetaculusConfig:
        """Load Metaculus configuration."""
        tournament_id_str = self._get_config_value("metaculus.tournament_id", "METACULUS_TOURNAMENT_ID")
        tournament_id = int(tournament_id_str) if tournament_id_str else None
        return MetaculusConfig(
            username=self._get_config_value("metaculus.username", "METACULUS_USERNAME", ""),
            password=self._get_config_value("metaculus.password", "METACULUS_PASSWORD", ""),
            api_token=self._get_config_value("metaculus.api_token", "METACULUS_TOKEN", ""),
            base_url=self._get_config_value("metaculus.base_url", "METACULUS_BASE_URL", "https://www.metaculus.com/api2"),
            tournament_id=tournament_id,
            timeout=self._get_config_float("metaculus.timeout", "METACULUS_TIMEOUT", 30.0),
            submit_predictions=self._get_config_bool("metaculus.submit_predictions", "METACULUS_SUBMIT_PREDICTIONS", False),
            dry_run=self._get_config_bool("metaculus.dry_run", "METACULUS_DRY_RUN", True),
            include_reasoning=self._get_config_bool("metaculus.include_reasoning", "METACULUS_INCLUDE_REASONING", True),
            max_prediction_retries=self._get_config_int("metaculus.max_prediction_retries", "METACULUS_MAX_RETRIES", 3),
        )
    
    def _load_pipeline_config(self) -> PipelineConfig:
        """Load pipeline configuration."""
        default_agents = self._get_config_list("pipeline.default_agent_names", "PIPELINE_DEFAULT_AGENTS", ["ensemble"])
        return PipelineConfig(
            max_concurrent_questions=self._get_config_int("pipeline.max_concurrent_questions", "PIPELINE_MAX_CONCURRENT", 5),
            batch_delay_seconds=self._get_config_float("pipeline.batch_delay_seconds", "PIPELINE_BATCH_DELAY", 1.0),
            default_research_depth=self._get_config_int("pipeline.default_research_depth", "PIPELINE_RESEARCH_DEPTH", 3),
            default_agent_names=default_agents,
            health_check_interval=self._get_config_int("pipeline.health_check_interval", "PIPELINE_HEALTH_CHECK_INTERVAL", 60),
            max_failed_health_checks=self._get_config_int("pipeline.max_failed_health_checks", "PIPELINE_MAX_FAILED_HEALTH_CHECKS", 3),
            enable_benchmarking=self._get_config_bool("pipeline.enable_benchmarking", "PIPELINE_ENABLE_BENCHMARKING", True),
            benchmark_output_path=self._get_config_optional_str("pipeline.benchmark_output_path", "PIPELINE_BENCHMARK_OUTPUT_PATH"),
            max_retries_per_question=self._get_config_int("pipeline.max_retries_per_question", "PIPELINE_MAX_RETRIES", 3),
            retry_delay_seconds=self._get_config_float("pipeline.retry_delay_seconds", "PIPELINE_RETRY_DELAY", 2.0),
            enable_circuit_breaker=self._get_config_bool("pipeline.enable_circuit_breaker", "PIPELINE_ENABLE_CIRCUIT_BREAKER", True),
        )
    
    def _load_bot_config(self) -> BotConfig:
        """Load bot configuration."""
        return BotConfig(
            name=self._get_config_value("bot.name", "BOT_NAME", "MetaculusBotHA"),
            version=self._get_config_value("bot.version", "BOT_VERSION", "1.0.0"),
            research_reports_per_question=self._get_config_int("bot.research_reports_per_question", "RESEARCH_REPORTS_PER_QUESTION", 2),
            predictions_per_research_report=self._get_config_int("bot.predictions_per_research_report", "PREDICTIONS_PER_RESEARCH_REPORT", 3),
            publish_reports_to_metaculus=self._get_config_bool("bot.publish_reports", "PUBLISH_REPORTS", False),
            max_concurrent_questions=self._get_config_int("bot.max_concurrent_questions", "MAX_CONCURRENT_QUESTIONS", 2),
            enable_deep_research=self._get_config_bool("bot.enable_deep_research", "BOT_ENABLE_DEEP_RESEARCH", True),
            research_timeout_minutes=self._get_config_int("bot.research_timeout_minutes", "BOT_RESEARCH_TIMEOUT_MINUTES", 10),
            min_sources_per_topic=self._get_config_int("bot.min_sources_per_topic", "BOT_MIN_SOURCES_PER_TOPIC", 3),
            require_confidence_score=self._get_config_bool("bot.require_confidence_score", "BOT_REQUIRE_CONFIDENCE_SCORE", True),
            min_confidence_threshold=self._get_config_float("bot.min_confidence_threshold", "BOT_MIN_CONFIDENCE_THRESHOLD", 0.5),
            enable_uncertainty_quantification=self._get_config_bool("bot.enable_uncertainty_quantification", "BOT_ENABLE_UNCERTAINTY_QUANTIFICATION", True),
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration."""
        return LoggingConfig(
            level=self._get_config_value("logging.level", "LOG_LEVEL", "INFO"),
            format=self._get_config_value("logging.format", "LOG_FORMAT", "json"),
            file_path=self._get_config_optional_str("logging.file_path", "LOG_FILE_PATH"),
            console_output=self._get_config_bool("logging.console_output", "LOG_CONSOLE_OUTPUT", True),
            enable_structured_logging=self._get_config_bool("logging.enable_structured_logging", "LOG_ENABLE_STRUCTURED", True),
            log_predictions=self._get_config_bool("logging.log_predictions", "LOG_PREDICTIONS", True),
            log_research_data=self._get_config_bool("logging.log_research_data", "LOG_RESEARCH_DATA", False),
            max_log_size_mb=self._get_config_int("logging.max_log_size_mb", "LOG_MAX_SIZE_MB", 100),
            backup_count=self._get_config_int("logging.backup_count", "LOG_BACKUP_COUNT", 5),
        )
    
    def _load_agent_config(self) -> AgentConfig:
        """Load agent configuration."""
        return AgentConfig(
            enabled=self._get_config_bool("agent.enabled", "AGENT_ENABLED", True),
            weight=self._get_config_float("agent.weight", "AGENT_WEIGHT", 1.0),
            confidence_threshold=self._get_config_float("agent.confidence_threshold", "AGENT_CONFIDENCE_THRESHOLD", 0.5),
            max_retries=self._get_config_int("agent.max_retries", "AGENT_MAX_RETRIES", 2),
            timeout=self._get_config_float("agent.timeout", "AGENT_TIMEOUT", 300.0),
            chain_of_thought_steps=self._get_config_int("agent.chain_of_thought_steps", "AGENT_COT_STEPS", 3),
            tree_of_thought_depth=self._get_config_int("agent.tree_of_thought_depth", "AGENT_TOT_DEPTH", 3),
            tree_of_thought_breadth=self._get_config_int("agent.tree_of_thought_breadth", "AGENT_TOT_BREADTH", 3),
            react_max_iterations=self._get_config_int("agent.react_max_iterations", "AGENT_REACT_MAX_ITERATIONS", 5),
            auto_cot_examples=self._get_config_int("agent.auto_cot_examples", "AGENT_AUTO_COT_EXAMPLES", 3),
        )
    
    def _load_ensemble_config(self) -> EnsembleConfig:
        """Load ensemble configuration."""
        agent_weights = {}
        try:
            weights_str = self._get_config_value("ensemble.agent_weights", "ENSEMBLE_AGENT_WEIGHTS", "")
            if weights_str:
                # Parse "agent1:weight1,agent2:weight2" format
                for pair in weights_str.split(","):
                    if ":" in pair:
                        agent, weight = pair.strip().split(":", 1)
                        agent_weights[agent.strip()] = float(weight.strip())
        except (ValueError, AttributeError):
            pass
        
        if not agent_weights:
            agent_weights = {
                "chain_of_thought": 1.0,
                "tree_of_thought": 1.2,
                "react": 1.1,
                "auto_cot": 0.9
            }
        
        return EnsembleConfig(
            aggregation_method=self._get_config_value("ensemble.aggregation_method", "ENSEMBLE_AGGREGATION_METHOD", "confidence_weighted"),
            min_agents=self._get_config_int("ensemble.min_agents", "ENSEMBLE_MIN_AGENTS", 2),
            confidence_threshold=self._get_config_float("ensemble.confidence_threshold", "ENSEMBLE_CONFIDENCE_THRESHOLD", 0.6),
            use_meta_reasoning=self._get_config_bool("ensemble.use_meta_reasoning", "ENSEMBLE_USE_META_REASONING", True),
            fallback_to_single_agent=self._get_config_bool("ensemble.fallback_to_single_agent", "ENSEMBLE_FALLBACK_TO_SINGLE", True),
            agent_weights=agent_weights,
        )
    
    def _get_config_value(self, yaml_path: str, env_var: str, default: str = "") -> str:
        """Get configuration value from YAML or environment variables."""
        # Check environment variable first
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value
        
        # Check YAML config
        yaml_value = self._get_nested_value(self.yaml_config, yaml_path)
        if yaml_value is not None:
            return str(yaml_value)
        
        return default
    
    def _get_config_optional_str(self, yaml_path: str, env_var: str) -> Optional[str]:
        """Get optional string configuration value."""
        # Check environment variable first
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value
        
        # Check YAML config
        yaml_value = self._get_nested_value(self.yaml_config, yaml_path)
        if yaml_value is not None:
            return str(yaml_value)
        
        return None
    
    def _get_config_bool(self, yaml_path: str, env_var: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self._get_config_value(yaml_path, env_var, str(default).lower())
        return value.lower() in ("true", "1", "yes", "on")
    
    def _get_config_int(self, yaml_path: str, env_var: str, default: int) -> int:
        """Get integer configuration value."""
        value = self._get_config_value(yaml_path, env_var, str(default))
        try:
            return int(value)
        except ValueError:
            return default
    
    def _get_config_float(self, yaml_path: str, env_var: str, default: float) -> float:
        """Get float configuration value."""
        value = self._get_config_value(yaml_path, env_var, str(default))
        try:
            return float(value)
        except ValueError:
            return default
    
    def _get_config_list(self, yaml_path: str, env_var: str, default: List[str]) -> List[str]:
        """Get list configuration value."""
        # Check environment variable first (comma-separated)
        env_value = os.getenv(env_var)
        if env_value is not None:
            return [item.strip() for item in env_value.split(",") if item.strip()]
        
        # Check YAML config
        yaml_value = self._get_nested_value(self.yaml_config, yaml_path)
        if yaml_value is not None and isinstance(yaml_value, list):
            return yaml_value
        
        return default
    
    def _get_optional_int(self, yaml_path: str, env_var: str) -> Optional[int]:
        """Get optional integer configuration value."""
        value = self._get_config_value(yaml_path, env_var)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
        return None
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested value from config dictionary using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                "min_connections": self.database.min_connections,
                "max_connections": self.database.max_connections,
                "connection_timeout": self.database.connection_timeout,
                # Don't include password in dict representation
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "max_retries": self.llm.max_retries,
                "timeout": self.llm.timeout,
                "rate_limit_rpm": self.llm.rate_limit_rpm,
                "backup_models": self.llm.backup_models,
                "use_structured_output": self.llm.use_structured_output,
                "response_format": self.llm.response_format,
                # Don't include API keys in dict representation
            },
            "search": {
                "provider": self.search.provider,
                "max_results": self.search.max_results,
                "timeout": self.search.timeout,
                "duckduckgo_enabled": self.search.duckduckgo_enabled,
                "wikipedia_enabled": self.search.wikipedia_enabled,
                "concurrent_searches": self.search.concurrent_searches,
                "deduplicate_results": self.search.deduplicate_results,
                "result_cache_ttl": self.search.result_cache_ttl,
                "max_content_length": self.search.max_content_length,
                # Don't include API credentials in dict representation
            },
            "metaculus": {
                "base_url": self.metaculus.base_url,
                "tournament_id": self.metaculus.tournament_id,
                "timeout": self.metaculus.timeout,
                "submit_predictions": self.metaculus.submit_predictions,
                "dry_run": self.metaculus.dry_run,
                "include_reasoning": self.metaculus.include_reasoning,
                "max_prediction_retries": self.metaculus.max_prediction_retries,
                # Don't include credentials in dict representation
            },
            "pipeline": {
                "max_concurrent_questions": self.pipeline.max_concurrent_questions,
                "batch_delay_seconds": self.pipeline.batch_delay_seconds,
                "default_research_depth": self.pipeline.default_research_depth,
                "default_agent_names": self.pipeline.default_agent_names,
                "health_check_interval": self.pipeline.health_check_interval,
                "max_failed_health_checks": self.pipeline.max_failed_health_checks,
                "enable_benchmarking": self.pipeline.enable_benchmarking,
                "benchmark_output_path": self.pipeline.benchmark_output_path,
                "max_retries_per_question": self.pipeline.max_retries_per_question,
                "retry_delay_seconds": self.pipeline.retry_delay_seconds,
                "enable_circuit_breaker": self.pipeline.enable_circuit_breaker,
            },
            "bot": {
                "name": self.bot.name,
                "version": self.bot.version,
                "research_reports_per_question": self.bot.research_reports_per_question,
                "predictions_per_research_report": self.bot.predictions_per_research_report,
                "publish_reports_to_metaculus": self.bot.publish_reports_to_metaculus,
                "max_concurrent_questions": self.bot.max_concurrent_questions,
                "enable_deep_research": self.bot.enable_deep_research,
                "research_timeout_minutes": self.bot.research_timeout_minutes,
                "min_sources_per_topic": self.bot.min_sources_per_topic,
                "require_confidence_score": self.bot.require_confidence_score,
                "min_confidence_threshold": self.bot.min_confidence_threshold,
                "enable_uncertainty_quantification": self.bot.enable_uncertainty_quantification,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": self.logging.file_path,
                "console_output": self.logging.console_output,
                "enable_structured_logging": self.logging.enable_structured_logging,
                "log_predictions": self.logging.log_predictions,
                "log_research_data": self.logging.log_research_data,
                "max_log_size_mb": self.logging.max_log_size_mb,
                "backup_count": self.logging.backup_count,
            },
            "agent": {
                "enabled": self.agent.enabled,
                "weight": self.agent.weight,
                "confidence_threshold": self.agent.confidence_threshold,
                "max_retries": self.agent.max_retries,
                "timeout": self.agent.timeout,
                "chain_of_thought_steps": self.agent.chain_of_thought_steps,
                "tree_of_thought_depth": self.agent.tree_of_thought_depth,
                "tree_of_thought_breadth": self.agent.tree_of_thought_breadth,
                "react_max_iterations": self.agent.react_max_iterations,
                "auto_cot_examples": self.agent.auto_cot_examples,
            },
            "ensemble": {
                "aggregation_method": self.ensemble.aggregation_method,
                "min_agents": self.ensemble.min_agents,
                "confidence_threshold": self.ensemble.confidence_threshold,
                "use_meta_reasoning": self.ensemble.use_meta_reasoning,
                "fallback_to_single_agent": self.ensemble.fallback_to_single_agent,
                "agent_weights": self.ensemble.agent_weights,
            }
        }


@dataclass
class Settings:
    """Main application settings that aggregates all configuration components."""
    
    # Core configuration components
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    metaculus: MetaculusConfig = field(default_factory=MetaculusConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    @classmethod
    def from_config(cls, config: "Config") -> "Settings":
        """Create Settings instance from Config instance."""
        return cls(
            database=config.database,
            llm=config.llm,
            search=config.search,
            metaculus=config.metaculus,
            pipeline=config.pipeline,
            bot=config.bot,
            logging=config.logging,
            agent=config.agent,
            ensemble=config.ensemble,
            environment=getattr(config, 'environment', 'development'),
            debug=getattr(config, 'debug', False)
        )
    
    @classmethod
    def load_from_yaml(cls, config_path: Optional[Union[str, Path]] = None) -> "Settings":
        """Load settings from YAML configuration file."""
        path_obj = None
        if config_path:
            path_obj = Path(config_path) if isinstance(config_path, str) else config_path
        
        # Create Config instance and convert to Settings
        config = Config(path_obj)
        return cls.from_config(config)
    
    @classmethod
    def load_from_env(cls) -> "Settings":
        """Load settings from environment variables only."""
        config = Config()
        return cls.from_config(config)
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a specific service."""
        api_keys = {
            "openai": self.llm.openai_api_key,
            "anthropic": self.llm.anthropic_api_key,
            "openrouter": self.llm.openrouter_api_key,
            "metaculus": self.metaculus.api_token,
            "serpapi": self.search.serpapi_key,
        }
        return api_keys.get(service, "")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ("production", "prod")
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ("development", "dev")
    
    def is_testing(self) -> bool:
        """Check if running in test environment."""
        return self.environment.lower() in ("testing", "test")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        # Try to load from config file first, then fall back to environment
        config_file = Path("config/config.yaml")
        if config_file.exists():
            _settings = Settings.load_from_yaml(config_file)
        else:
            _settings = Settings.load_from_env()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set global settings instance (mainly for testing)."""
    global _settings
    _settings = settings


def reload_settings(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """Reload settings from configuration."""
    global _settings
    if config_path:
        _settings = Settings.load_from_yaml(config_path)
    else:
        _settings = Settings.load_from_env()
    return _settings