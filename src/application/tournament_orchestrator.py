"""
Tournament orchestration system that integrates all components with proper dependency injection.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import structlog  # type: ignore[import]

    logger = structlog.get_logger(__name__)
except (
    Exception
):  # pragma: no cover - fallback if structlog isn't available during analysis
    import logging as _logging

    class _KwLogger:
        def __init__(self, name: str):
            self._logger = _logging.getLogger(name)

        def debug(self, msg: str, **kwargs):
            self._logger.debug(self._fmt(msg, kwargs))

        def info(self, msg: str, **kwargs):
            self._logger.info(self._fmt(msg, kwargs))

        def warning(self, msg: str, **kwargs):
            self._logger.warning(self._fmt(msg, kwargs))

        def error(self, msg: str, **kwargs):
            self._logger.error(self._fmt(msg, kwargs))

        def _fmt(self, msg: str, kv: dict) -> str:
            return f"{msg} | {kv}" if kv else msg

    logger = _KwLogger(__name__)

from ..application.dispatcher import Dispatcher
from ..application.forecast_service import ForecastService
from ..application.ingestion_service import IngestionService
from ..domain.services.authoritative_source_manager import AuthoritativeSourceManager
from ..domain.services.calibration_service import CalibrationTracker
from ..domain.services.conflict_resolver import ConflictResolver
from ..domain.services.conservative_strategy_engine import ConservativeStrategyEngine
from ..domain.services.divergence_analyzer import DivergenceAnalyzer
from ..domain.services.dynamic_weight_adjuster import DynamicWeightAdjuster
from ..domain.services.ensemble_service import EnsembleService
from ..domain.services.forecasting_service import ForecastingService
from ..domain.services.knowledge_gap_detector import KnowledgeGapDetector
from ..domain.services.pattern_detector import PatternDetector
from ..domain.services.performance_analyzer import PerformanceAnalyzer
from ..domain.services.performance_tracking_service import PerformanceTrackingService
from ..domain.services.question_categorizer import QuestionCategorizer
from ..domain.services.reasoning_orchestrator import ReasoningOrchestrator
from ..domain.services.research_service import ResearchService
from ..domain.services.risk_management_service import RiskManagementService
from ..domain.services.scoring_optimizer import ScoringOptimizer
from ..domain.services.strategy_adaptation_engine import StrategyAdaptationEngine
from ..domain.services.tournament_analytics import TournamentAnalytics
from ..domain.services.tournament_analyzer import TournamentAnalyzer
from ..domain.services.uncertainty_quantifier import UncertaintyQuantifier
from ..infrastructure.config.config_manager import (
    ConfigChangeEvent,
    ConfigManager,
    create_config_manager,
)
from ..infrastructure.config.settings import Settings
from ..infrastructure.external_apis.llm_client import LLMClient
from ..infrastructure.external_apis.metaculus_client import MetaculusClient
from ..infrastructure.external_apis.search_client import (
    SearchClient,
    create_search_client,
)
from ..infrastructure.logging.reasoning_logger import ReasoningLogger
from ..infrastructure.reliability.circuit_breaker import CircuitBreaker
from ..infrastructure.reliability.health_monitor import HealthMonitor
from ..infrastructure.reliability.rate_limiter import TokenBucketRateLimiter
from ..infrastructure.reliability.retry_manager import RetryManager
from ..pipelines.forecasting_pipeline import ForecastingPipeline

"""Logger initialized above (structlog preferred, fallback to stdlib logging)."""


@dataclass
class ComponentRegistry:
    """Registry for all system components with proper dependency injection."""

    # Configuration
    settings: Settings

    # Infrastructure clients
    llm_client: LLMClient
    search_client: SearchClient
    metaculus_client: MetaculusClient

    # Reliability components
    circuit_breaker: CircuitBreaker
    rate_limiter: TokenBucketRateLimiter
    health_monitor: HealthMonitor
    retry_manager: RetryManager
    reasoning_logger: ReasoningLogger

    # Application services
    dispatcher: Dispatcher
    forecast_service: ForecastService
    ingestion_service: IngestionService

    # Domain services
    ensemble_service: EnsembleService
    forecasting_service: ForecastingService
    research_service: ResearchService
    tournament_analytics: TournamentAnalytics
    performance_tracking: PerformanceTrackingService
    calibration_service: CalibrationTracker
    risk_management_service: RiskManagementService

    # Advanced reasoning and analysis services
    reasoning_orchestrator: ReasoningOrchestrator
    question_categorizer: QuestionCategorizer
    authoritative_source_manager: AuthoritativeSourceManager
    conflict_resolver: ConflictResolver
    knowledge_gap_detector: KnowledgeGapDetector
    divergence_analyzer: DivergenceAnalyzer
    dynamic_weight_adjuster: DynamicWeightAdjuster
    performance_analyzer: PerformanceAnalyzer
    pattern_detector: PatternDetector
    strategy_adaptation_engine: StrategyAdaptationEngine
    uncertainty_quantifier: UncertaintyQuantifier
    conservative_strategy_engine: ConservativeStrategyEngine
    scoring_optimizer: ScoringOptimizer
    tournament_analyzer: TournamentAnalyzer

    # Pipeline
    forecasting_pipeline: ForecastingPipeline


class TournamentOrchestrator:
    """
    Main orchestrator that coordinates all tournament operations with proper dependency injection.
    Implements hot-reloading configuration and comprehensive integration testing.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        """Initialize the tournament orchestrator with configuration."""
        self.config_path = config_path
        self.config_manager = config_manager
        self.registry: Optional[ComponentRegistry] = None
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._config_reload_task: Optional[asyncio.Task] = None
        self._last_config_reload = datetime.now(timezone.utc)

        # Performance metrics
        self.metrics = {
            "questions_processed": 0,
            "forecasts_generated": 0,
            "errors_encountered": 0,
            "uptime_start": datetime.now(timezone.utc),
            "last_health_check": None,
            "component_health": {},
        }

    async def initialize(self) -> None:
        """Initialize all components with proper dependency injection."""
        logger.info("Initializing tournament orchestrator")

        try:
            # Initialize configuration manager if not provided
            if not self.config_manager:
                config_paths = [Path(self.config_path)] if self.config_path else []
                watch_dirs = [
                    Path("config"),
                    Path("."),
                ]  # Watch config directory and current directory
                self.config_manager = create_config_manager(
                    config_paths=[str(p) for p in config_paths],
                    watch_directories=[str(p) for p in watch_dirs if p.exists()],
                    enable_hot_reload=True,
                    validation_enabled=True,
                )

            # Load configuration
            settings = await self.config_manager.initialize()

            # Add configuration change listener
            self.config_manager.add_change_listener(self._on_config_change)

            # Initialize infrastructure clients
            llm_client = await self._create_llm_client(settings)
            search_client = await self._create_search_client(settings)
            metaculus_client = await self._create_metaculus_client(settings)

            # Initialize reliability components
            from ..infrastructure.reliability.circuit_breaker import (
                CircuitBreakerConfig,
            )

            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=5, recovery_timeout=60.0, expected_exception=Exception
            )
            circuit_breaker = CircuitBreaker("main", circuit_breaker_config)

            from ..infrastructure.reliability.rate_limiter import RateLimitConfig

            rate_limit_config = RateLimitConfig(
                requests_per_second=settings.llm.rate_limit_rpm / 60.0,
                burst_size=min(settings.llm.rate_limit_rpm, 20),
                enabled=True,
            )
            rate_limiter = TokenBucketRateLimiter("main", rate_limit_config)

            health_monitor = HealthMonitor(
                check_interval=settings.pipeline.health_check_interval
            )

            from ..infrastructure.reliability.retry_manager import RetryPolicy

            retry_policy = RetryPolicy(
                max_attempts=settings.pipeline.max_retries_per_question,
                base_delay=settings.pipeline.retry_delay_seconds,
                max_delay=30.0,
                backoff_multiplier=2.0,
            )
            retry_manager = RetryManager("main", retry_policy)

            from pathlib import Path as PathLib

            reasoning_logger = ReasoningLogger(base_dir=PathLib("logs/reasoning"))

            # Initialize domain services with proper dependency injection
            ensemble_service = EnsembleService()
            forecasting_service = ForecastingService()
            research_service = ResearchService(
                search_client=search_client, llm_client=llm_client
            )
            tournament_analytics = TournamentAnalytics()
            performance_tracking = PerformanceTrackingService()
            calibration_service = CalibrationTracker()
            risk_management_service = RiskManagementService()

            # Initialize advanced reasoning and analysis services
            reasoning_orchestrator = ReasoningOrchestrator()
            reasoning_orchestrator.llm_client = llm_client
            reasoning_orchestrator.search_client = search_client

            question_categorizer = QuestionCategorizer()
            question_categorizer.llm_client = llm_client

            authoritative_source_manager = AuthoritativeSourceManager()
            authoritative_source_manager.search_client = search_client
            authoritative_source_manager.llm_client = llm_client

            conflict_resolver = ConflictResolver()
            conflict_resolver.llm_client = llm_client

            knowledge_gap_detector = KnowledgeGapDetector()
            knowledge_gap_detector.llm_client = llm_client
            knowledge_gap_detector.search_client = search_client
            divergence_analyzer = DivergenceAnalyzer()
            dynamic_weight_adjuster = DynamicWeightAdjuster()
            performance_analyzer = PerformanceAnalyzer()
            pattern_detector = PatternDetector()
            strategy_adaptation_engine = StrategyAdaptationEngine(
                performance_analyzer, pattern_detector
            )
            uncertainty_quantifier = UncertaintyQuantifier()
            conservative_strategy_engine = ConservativeStrategyEngine()
            scoring_optimizer = ScoringOptimizer()
            tournament_analyzer = TournamentAnalyzer()

            # Initialize application services with proper dependency injection
            from ..application.ingestion_service import ValidationLevel

            ingestion_service = IngestionService(ValidationLevel.LENIENT)

            forecast_service = ForecastService(
                forecasting_service=forecasting_service,
                ensemble_service=ensemble_service,
                research_service=research_service,
                reasoning_orchestrator=reasoning_orchestrator,
                question_categorizer=question_categorizer,
                risk_management_service=risk_management_service,
                performance_tracking=performance_tracking,
                calibration_service=calibration_service,
            )

            dispatcher = Dispatcher(
                forecast_service=forecast_service,
                ingestion_service=ingestion_service,
                metaculus_client=metaculus_client,
                tournament_analytics=tournament_analytics,
                performance_tracking=performance_tracking,
            )

            # Initialize forecasting pipeline
            forecasting_pipeline = ForecastingPipeline(
                settings=settings,
                llm_client=llm_client,
                search_client=search_client,
                metaculus_client=metaculus_client,
            )

            # Create component registry
            self.registry = ComponentRegistry(
                settings=settings,
                llm_client=llm_client,
                search_client=search_client,
                metaculus_client=metaculus_client,
                circuit_breaker=circuit_breaker,
                rate_limiter=rate_limiter,
                health_monitor=health_monitor,
                retry_manager=retry_manager,
                reasoning_logger=reasoning_logger,
                dispatcher=dispatcher,
                forecast_service=forecast_service,
                ingestion_service=ingestion_service,
                ensemble_service=ensemble_service,
                forecasting_service=forecasting_service,
                research_service=research_service,
                tournament_analytics=tournament_analytics,
                performance_tracking=performance_tracking,
                calibration_service=calibration_service,
                risk_management_service=risk_management_service,
                reasoning_orchestrator=reasoning_orchestrator,
                question_categorizer=question_categorizer,
                authoritative_source_manager=authoritative_source_manager,
                conflict_resolver=conflict_resolver,
                knowledge_gap_detector=knowledge_gap_detector,
                divergence_analyzer=divergence_analyzer,
                dynamic_weight_adjuster=dynamic_weight_adjuster,
                performance_analyzer=performance_analyzer,
                pattern_detector=pattern_detector,
                strategy_adaptation_engine=strategy_adaptation_engine,
                uncertainty_quantifier=uncertainty_quantifier,
                conservative_strategy_engine=conservative_strategy_engine,
                scoring_optimizer=scoring_optimizer,
                tournament_analyzer=tournament_analyzer,
                forecasting_pipeline=forecasting_pipeline,
            )

            # Start background tasks
            await self._start_background_tasks()

            logger.info("Tournament orchestrator initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize tournament orchestrator", error=str(e))
            raise

    async def _on_config_change(self, change_event: ConfigChangeEvent) -> None:
        """Handle configuration change events."""
        logger.info(
            "Configuration change detected",
            change_type=change_event.change_type.value,
            file_path=str(change_event.file_path),
        )

        try:
            if self.registry and change_event.new_config:
                # Update registry settings
                new_settings = self.config_manager.get_current_settings()
                if new_settings:
                    old_settings = self.registry.settings
                    self.registry.settings = new_settings

                    # Update component configurations
                    await self._update_component_configs(old_settings, new_settings)

                    logger.info("Components updated with new configuration")

        except Exception as e:
            logger.error("Failed to handle configuration change", error=str(e))

    async def _create_llm_client(self, settings: Settings) -> LLMClient:
        """Create and configure LLM client."""
        llm_client = LLMClient(settings.llm)
        # Initialize if method exists
        if hasattr(llm_client, "initialize"):
            await llm_client.initialize()
        return llm_client

    async def _create_search_client(self, settings: Settings) -> SearchClient:
        """Create and configure search client."""
        # Use factory; policy enforces NoOp (AskNews-first handled in pipeline)
        search_client = create_search_client(settings)
        # Initialize if method exists
        if hasattr(search_client, "initialize"):
            await search_client.initialize()
        return search_client

    async def _create_metaculus_client(self, settings: Settings) -> MetaculusClient:
        """Create and configure Metaculus client."""
        metaculus_client = MetaculusClient(settings.metaculus)
        # Initialize if method exists
        if hasattr(metaculus_client, "initialize"):
            await metaculus_client.initialize()
        return metaculus_client

    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        if not self.registry:
            raise RuntimeError("Registry not initialized")

        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        # Start configuration hot-reloading
        self._config_reload_task = asyncio.create_task(self._config_reload_loop())

        logger.info("Background tasks started")

    async def _health_check_loop(self) -> None:
        """Continuous health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(
                    self.registry.settings.pipeline.health_check_interval
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check failed", error=str(e))
                await asyncio.sleep(30)  # Retry after 30 seconds on error

    async def _config_reload_loop(self) -> None:
        """Configuration hot-reloading loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_config_reload()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Config reload check failed", error=str(e))
                await asyncio.sleep(60)

    async def _perform_health_check(self) -> Dict[str, bool]:
        """Perform comprehensive health check of all components."""
        if not self.registry:
            return {"orchestrator": False}

        health_status = {}

        try:
            # Check infrastructure clients
            health_status["llm_client"] = await self._check_component_health(
                self.registry.llm_client, "health_check"
            )
            health_status["search_client"] = await self._check_component_health(
                self.registry.search_client, "health_check"
            )
            health_status["metaculus_client"] = await self._check_component_health(
                self.registry.metaculus_client, "health_check"
            )

            # Check pipeline health
            pipeline_health = await self.registry.forecasting_pipeline.health_check()
            health_status.update(pipeline_health)

            # Update metrics
            self.metrics["last_health_check"] = datetime.now(timezone.utc)
            self.metrics["component_health"] = health_status

            # Log health status
            healthy_components = sum(1 for status in health_status.values() if status)
            total_components = len(health_status)

            logger.info(
                "Health check completed",
                healthy_components=healthy_components,
                total_components=total_components,
                health_status=health_status,
            )

            return health_status

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {"orchestrator": False}

    async def _check_component_health(self, component: Any, method_name: str) -> bool:
        """Check health of individual component."""
        try:
            if hasattr(component, method_name):
                health_method = getattr(component, method_name)
                if asyncio.iscoroutinefunction(health_method):
                    await health_method()
                else:
                    health_method()
                return True
            return True  # Assume healthy if no health check method
        except Exception:
            return False

    async def _check_config_reload(self) -> None:
        """Check if configuration needs to be reloaded."""
        if not self.config_path:
            return

        try:
            from pathlib import Path

            config_file = Path(self.config_path)

            if config_file.exists():
                file_modified = datetime.fromtimestamp(
                    config_file.stat().st_mtime, tz=timezone.utc
                )

                if file_modified > self._last_config_reload:
                    logger.info("Configuration file changed, reloading...")
                    await self._reload_configuration()
                    self._last_config_reload = file_modified

        except Exception as e:
            logger.error("Config reload check failed", error=str(e))

    async def _load_configuration(self) -> Settings:
        """Load configuration from config manager."""
        if not self.config_manager:
            raise RuntimeError("Config manager not initialized")

        settings = self.config_manager.get_current_settings()
        if not settings:
            raise RuntimeError("Failed to load settings from config manager")

        return settings

    async def _reload_configuration(self) -> None:
        """Reload configuration and update components."""
        try:
            new_settings = await self._load_configuration()

            if self.registry:
                # Update settings in registry
                old_settings = self.registry.settings
                self.registry.settings = new_settings

                # Update components that support configuration updates
                await self._update_component_configs(old_settings, new_settings)

                logger.info("Configuration reloaded successfully")

        except Exception as e:
            logger.error("Failed to reload configuration", error=str(e))
            # Revert to old settings if reload fails
            if self.registry and hasattr(self, "_last_good_settings"):
                self.registry.settings = self._last_good_settings

    async def _update_component_configs(
        self, old_settings: Settings, new_settings: Settings
    ) -> None:
        """Update component configurations after reload."""
        if not self.registry:
            return

        # Update LLM client if configuration changed
        if old_settings.llm != new_settings.llm:
            await self.registry.llm_client.update_config(new_settings.llm)

        # Update search client if configuration changed
        if old_settings.search != new_settings.search:
            await self.registry.search_client.update_config(new_settings.search)

        # Update Metaculus client if configuration changed
        if old_settings.metaculus != new_settings.metaculus:
            await self.registry.metaculus_client.update_config(new_settings.metaculus)

    async def run_tournament(
        self,
        tournament_id: Optional[int] = None,
        max_questions: Optional[int] = None,
        agent_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete tournament forecasting with all integrated components.

        Args:
            tournament_id: Metaculus tournament ID
            max_questions: Maximum number of questions to process
            agent_types: List of agent types to use

        Returns:
            Tournament results with comprehensive metrics
        """
        if not self.registry:
            raise RuntimeError("Orchestrator not initialized")

        tournament_id = tournament_id or self.registry.settings.metaculus.tournament_id
        max_questions = max_questions or 10
        agent_types = agent_types or self.registry.settings.pipeline.default_agent_names

        logger.info(
            "Starting tournament run",
            tournament_id=tournament_id,
            max_questions=max_questions,
            agent_types=agent_types,
        )

        start_time = datetime.now(timezone.utc)
        results = {
            "tournament_id": tournament_id,
            "start_time": start_time.isoformat(),
            "questions_processed": 0,
            "forecasts_generated": 0,
            "errors": [],
            "performance_metrics": {},
            "agent_performance": {},
        }

        try:
            # Use dispatcher to orchestrate the tournament
            tournament_results = await self.registry.dispatcher.run_tournament(
                tournament_id=tournament_id,
                max_questions=max_questions,
                agent_types=agent_types,
            )

            # Update results
            results.update(tournament_results)
            results["end_time"] = datetime.now(timezone.utc).isoformat()
            results["duration_seconds"] = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()

            # Update metrics
            self.metrics["questions_processed"] += results["questions_processed"]
            self.metrics["forecasts_generated"] += results["forecasts_generated"]

            logger.info(
                "Tournament run completed",
                tournament_id=tournament_id,
                questions_processed=results["questions_processed"],
                forecasts_generated=results["forecasts_generated"],
                duration_seconds=results["duration_seconds"],
            )

            return results

        except Exception as e:
            logger.error(
                "Tournament run failed", tournament_id=tournament_id, error=str(e)
            )
            results["error"] = str(e)
            results["end_time"] = datetime.now(timezone.utc).isoformat()
            self.metrics["errors_encountered"] += 1
            raise

    async def run_single_question(
        self, question_id: int, agent_type: str = "ensemble"
    ) -> Dict[str, Any]:
        """Run forecasting for a single question."""
        if not self.registry:
            raise RuntimeError("Orchestrator not initialized")

        logger.info(
            "Running single question forecast",
            question_id=question_id,
            agent_type=agent_type,
        )

        try:
            result = await self.registry.forecasting_pipeline.run_single_question(
                question_id=question_id,
                agent_type=agent_type,
                include_research=True,
                collect_metrics=True,
            )

            self.metrics["questions_processed"] += 1
            self.metrics["forecasts_generated"] += 1

            return result

        except Exception as e:
            logger.error(
                "Single question forecast failed", question_id=question_id, error=str(e)
            )
            self.metrics["errors_encountered"] += 1
            raise

    async def run_batch_forecast(
        self, question_ids: List[int], agent_type: str = "ensemble"
    ) -> List[Dict[str, Any]]:
        """Run batch forecasting for multiple questions."""
        if not self.registry:
            raise RuntimeError("Orchestrator not initialized")

        logger.info(
            "Running batch forecast",
            question_count=len(question_ids),
            agent_type=agent_type,
        )

        try:
            results = await self.registry.forecasting_pipeline.run_batch_forecast(
                question_ids=question_ids,
                agent_type=agent_type,
                include_research=True,
                batch_size=self.registry.settings.pipeline.max_concurrent_questions,
            )

            self.metrics["questions_processed"] += len(question_ids)
            self.metrics["forecasts_generated"] += len(results)

            return results

        except Exception as e:
            logger.error(
                "Batch forecast failed", question_ids=question_ids, error=str(e)
            )
            self.metrics["errors_encountered"] += 1
            raise

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and metrics."""
        if not self.registry:
            return {"status": "not_initialized"}

        # Perform health check
        health_status = await self._perform_health_check()

        # Calculate uptime
        uptime_seconds = (
            datetime.now(timezone.utc) - self.metrics["uptime_start"]
        ).total_seconds()

        return {
            "status": "running",
            "uptime_seconds": uptime_seconds,
            "health_status": health_status,
            "metrics": self.metrics.copy(),
            "configuration": {
                "environment": self.registry.settings.environment,
                "tournament_id": self.registry.settings.metaculus.tournament_id,
                "max_concurrent_questions": self.registry.settings.pipeline.max_concurrent_questions,
                "default_agents": self.registry.settings.pipeline.default_agent_names,
            },
            "last_config_reload": self._last_config_reload.isoformat(),
        }

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for proper lifecycle management."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        logger.info("Shutting down tournament orchestrator")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._config_reload_task:
            self._config_reload_task.cancel()
            try:
                await self._config_reload_task
            except asyncio.CancelledError:
                pass

        # Shutdown configuration manager
        if self.config_manager:
            try:
                await self.config_manager.shutdown()
            except Exception as e:
                logger.error("Error during config manager shutdown", error=str(e))

        # Shutdown components
        if self.registry:
            try:
                # Shutdown clients
                if hasattr(self.registry.llm_client, "shutdown"):
                    await self.registry.llm_client.shutdown()
                if hasattr(self.registry.search_client, "shutdown"):
                    await self.registry.search_client.shutdown()
                if hasattr(self.registry.metaculus_client, "shutdown"):
                    await self.registry.metaculus_client.shutdown()

                # Shutdown reliability components
                if hasattr(self.registry.health_monitor, "shutdown"):
                    await self.registry.health_monitor.shutdown()

            except Exception as e:
                logger.error("Error during component shutdown", error=str(e))

        logger.info("Tournament orchestrator shutdown complete")


# Factory function for easy instantiation
async def create_tournament_orchestrator(
    config_path: Optional[str] = None,
) -> TournamentOrchestrator:
    """Factory function to create and initialize tournament orchestrator."""
    orchestrator = TournamentOrchestrator(config_path)
    await orchestrator.initialize()
    return orchestrator
