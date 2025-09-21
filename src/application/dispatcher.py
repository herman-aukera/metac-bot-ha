"""
Application service for dispatching questions through the forecasting pipeline.

The Dispatcher orchestrates the flow from raw API data through ingestion
to forecast generation, handling errors and batching appropriately.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.application.forecast_service import ForecastService
from src.application.ingestion_service import (
    IngestionService,
    ValidationLevel,
)
from src.domain.entities.forecast import Forecast
from src.domain.entities.question import Question
from src.infrastructure.config.settings import Settings
from src.infrastructure.logging.reasoning_logger import (
    log_agent_reasoning,
)
from src.infrastructure.metaculus_api import APIConfig, MetaculusAPI, MetaculusAPIError

# Import ensemble and reasoning logging capabilities
from src.pipelines.forecasting_pipeline import ForecastingPipeline

logger = logging.getLogger(__name__)


@dataclass
class DispatcherConfig:
    """Configuration for the dispatcher."""

    batch_size: int = 10
    validation_level: ValidationLevel = ValidationLevel.LENIENT
    max_retries: int = 3
    enable_dry_run: bool = False
    api_config: Optional[APIConfig] = None
    # Ensemble forecasting options
    enable_ensemble: bool = False
    ensemble_agents: Optional[List[str]] = None
    ensemble_aggregation_method: str = "weighted_average"
    enable_reasoning_logs: bool = True

    def __post_init__(self):
        if self.ensemble_agents is None:
            self.ensemble_agents = ["chain_of_thought", "tree_of_thought", "react"]


@dataclass
class DispatcherStats:
    """Statistics from dispatcher execution."""

    total_questions_fetched: int = 0
    questions_successfully_parsed: int = 0
    questions_failed_parsing: int = 0
    forecasts_generated: int = 0
    forecasts_failed: int = 0
    total_processing_time_seconds: float = 0.0

    def __post_init__(self):
        self.errors: List[str] = []

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate as percentage."""
        if self.total_questions_fetched == 0:
            return 0.0
        return (self.forecasts_generated / self.total_questions_fetched) * 100


class DispatcherError(Exception):
    """Exception raised by the dispatcher."""

    pass


class Dispatcher:
    """
    Orchestrates the complete forecasting pipeline.

    Fetches questions from API -> Ingests/parses them -> Generates forecasts
    Handles errors, batching, and provides comprehensive statistics.
    """

    def __init__(
        self,
        forecast_service: Optional[ForecastService] = None,
        ingestion_service: Optional[IngestionService] = None,
        metaculus_client=None,
        tournament_analytics=None,
        performance_tracking=None,
        config: Optional[DispatcherConfig] = None,
    ):
        """
        Initialize the dispatcher.

        Args:
            forecast_service: Optional forecast service instance for dependency injection
            ingestion_service: Optional ingestion service instance
            metaculus_client: Optional Metaculus client instance
            tournament_analytics: Optional tournament analytics service
            performance_tracking: Optional performance tracking service
            config: Dispatcher configuration. Uses defaults if None.
        """
        self.config = config or DispatcherConfig()

        # Initialize services with dependency injection
        self.api = MetaculusAPI(config=self.config.api_config)
        self.ingestion_service = ingestion_service or IngestionService(
            validation_level=self.config.validation_level
        )
        self.forecast_service = forecast_service or ForecastService()
        self.metaculus_client = metaculus_client
        self.tournament_analytics = tournament_analytics
        self.performance_tracking = performance_tracking

        # Initialize ensemble forecasting pipeline if enabled
        self.forecasting_pipeline = None
        if self.config.enable_ensemble:
            try:
                settings = Settings()
                self.forecasting_pipeline = ForecastingPipeline(settings=settings)
            except Exception as e:
                logger.warning(f"Failed to initialize forecasting pipeline: {e}")
                self.forecasting_pipeline = None

        # State
        self.stats = DispatcherStats()

    def dispatch(self, question: Question) -> Forecast:
        """
        Dispatch a single question for forecast generation.

        Args:
            question: The question to generate a forecast for

        Returns:
            Generated forecast

        Raises:
            DispatcherError: If forecast generation fails
        """
        try:
            # Use ensemble forecasting if enabled and available
            if self.config.enable_ensemble and self.forecasting_pipeline:
                return self.dispatch_ensemble(question)
            else:
                return self.forecast_service.generate_forecast(question)
        except Exception as e:
            error_msg = f"Failed to generate forecast for question {question.metaculus_id}: {str(e)}"
            logger.error(error_msg)
            raise DispatcherError(error_msg) from e

    def dispatch_ensemble(self, question: Question) -> Forecast:
        """
        Dispatch a single question for ensemble forecast generation.

        Args:
            question: The question to generate a forecast for

        Returns:
            Generated ensemble forecast

        Raises:
            DispatcherError: If ensemble forecast generation fails
        """
        try:
            logger.info(
                f"Generating ensemble forecast for question {question.metaculus_id}"
            )

            # Try to use the forecasting pipeline for ensemble forecasting
            if self.forecasting_pipeline:
                try:
                    # Convert to async call - for now use asyncio.run as a bridge
                    # TODO: Make dispatcher fully async in future iterations
                    import asyncio

                    # Determine agent types to use
                    agent_types = self.config.ensemble_agents or [
                        "chain_of_thought",
                        "tree_of_thought",
                        "react",
                    ]

                    # Use the pipeline's generate_forecast method directly with the Question entity
                    # This avoids the need for Metaculus client and works with local questions
                    ensemble_forecast = asyncio.run(
                        self.forecasting_pipeline.generate_forecast(
                            question=question,
                            agent_names=agent_types,
                            include_research=True,
                        )
                    )

                    # The pipeline already returns a proper Forecast object
                    forecast = ensemble_forecast

                    # Update metadata to indicate ensemble success
                    if forecast.metadata is None:
                        forecast.metadata = {}
                    forecast.metadata.update(
                        {
                            "ensemble_attempted": True,
                            "ensemble_agents": agent_types,
                            "aggregation_method": self.config.ensemble_aggregation_method,
                            "ensemble_success": True,
                            "offline_mode": True,
                        }
                    )

                except Exception as pipeline_error:
                    logger.warning(
                        f"ForecastingPipeline failed ({pipeline_error}), falling back to ForecastService ensemble"
                    )
                    # Fall back to the standard ForecastService which has its own ensemble logic
                    forecast = self.forecast_service.generate_forecast(question)

                    # Update metadata to indicate ensemble fallback
                    if forecast.metadata is None:
                        forecast.metadata = {}
                    forecast.metadata.update(
                        {
                            "ensemble_attempted": True,
                            "ensemble_agents": self.config.ensemble_agents
                            or ["ai_forecast_service"],
                            "aggregation_method": self.config.ensemble_aggregation_method,
                            "ensemble_success": True,
                            "pipeline_fallback": True,
                            "offline_mode": True,
                        }
                    )
            else:
                # Fallback to standard forecasting if pipeline not available
                logger.warning(
                    "Ensemble forecasting pipeline not available, falling back to standard forecasting"
                )
                forecast = self.forecast_service.generate_forecast(question)

                # Add ensemble metadata to indicate fallback
                if forecast.metadata is None:
                    forecast.metadata = {}
                forecast.metadata.update(
                    {
                        "ensemble_attempted": True,
                        "ensemble_agents": self.config.ensemble_agents,
                        "aggregation_method": self.config.ensemble_aggregation_method,
                        "fallback_used": True,
                        "ensemble_success": False,
                    }
                )

            # Log reasoning if enabled
            if self.config.enable_reasoning_logs:
                try:
                    # Log individual agent reasoning if available in forecast metadata
                    if (
                        hasattr(forecast, "metadata")
                        and forecast.metadata
                        and forecast.metadata.get("agents_used")
                    ):
                        agents_used = forecast.metadata.get("agents_used", [])
                        for agent_name in agents_used:
                            # Find prediction for this agent
                            agent_prediction = None
                            for pred in forecast.predictions:
                                if (
                                    pred.created_by == agent_name
                                    or agent_name in pred.reasoning
                                ):
                                    agent_prediction = pred
                                    break

                            if agent_prediction:
                                reasoning_data = {
                                    "reasoning": agent_prediction.reasoning
                                    or "No detailed reasoning available",
                                    "method": agent_name,
                                    "confidence": (
                                        agent_prediction.confidence.value
                                        if hasattr(agent_prediction.confidence, "value")
                                        else str(agent_prediction.confidence)
                                    ),
                                }

                                prediction_result = {
                                    "probability": agent_prediction.result.binary_probability,
                                    "confidence": (
                                        agent_prediction.confidence.value
                                        if hasattr(agent_prediction.confidence, "value")
                                        else str(agent_prediction.confidence)
                                    ),
                                    "method": agent_name,
                                }

                                log_agent_reasoning(
                                    question_id=question.metaculus_id or question.id,
                                    agent_name=agent_name,
                                    reasoning_data=reasoning_data,
                                    prediction_result=prediction_result,
                                )

                    # Log ensemble reasoning
                    reasoning_data = {
                        "reasoning": forecast.reasoning_summary
                        or "Ensemble forecast with multiple agents",
                        "method": "ensemble",
                        "agents_used": self.config.ensemble_agents
                        or ["chain_of_thought", "tree_of_thought", "react"],
                        "aggregation_method": forecast.ensemble_method
                        or self.config.ensemble_aggregation_method,
                        "prediction_count": len(forecast.predictions),
                    }

                    prediction_result = {
                        "probability": (
                            forecast.final_prediction.result.binary_probability
                            if forecast.final_prediction
                            else None
                        ),
                        "confidence": (
                            forecast.final_prediction.confidence.value
                            if forecast.final_prediction
                            and hasattr(forecast.final_prediction.confidence, "value")
                            else (
                                str(forecast.final_prediction.confidence)
                                if forecast.final_prediction
                                else None
                            )
                        ),
                        "method": "ensemble",
                    }

                    log_agent_reasoning(
                        question_id=question.metaculus_id or question.id,
                        agent_name="ensemble",
                        reasoning_data=reasoning_data,
                        prediction_result=prediction_result,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log reasoning trace: {e}")

            return forecast

        except Exception as e:
            error_msg = f"Failed to generate ensemble forecast for question {question.metaculus_id}: {str(e)}"
            logger.error(error_msg)
            raise DispatcherError(error_msg) from e

    def run(
        self,
        limit: Optional[int] = None,
        status: str = "open",
        category: Optional[str] = None,
    ) -> Tuple[List[Forecast], DispatcherStats]:
        """
        Run the complete forecasting pipeline.

        Args:
            limit: Maximum number of questions to process
            status: Question status filter
            category: Question category filter

        Returns:
            Tuple of (generated forecasts, execution statistics)

        Raises:
            DispatcherError: If critical errors occur during processing
        """
        start_time = datetime.now()
        forecasts = []

        try:
            logger.info(
                f"Starting dispatcher run with limit={limit}, status={status}, category={category}"
            )

            # Reset stats
            self.stats = DispatcherStats()

            # Step 1: Fetch questions from API
            raw_questions = self._fetch_questions(limit, status, category)
            self.stats.total_questions_fetched = len(raw_questions)

            if not raw_questions:
                logger.warning("No questions fetched from API")
                return forecasts, self.stats

            # Step 2: Parse questions into domain objects
            questions = self._parse_questions(raw_questions)

            # Step 3: Generate forecasts for parsed questions
            forecasts = self._generate_forecasts(questions)

            # Calculate total processing time
            end_time = datetime.now()
            self.stats.total_processing_time_seconds = (
                end_time - start_time
            ).total_seconds()

            logger.info(
                f"Dispatcher run completed: {len(forecasts)} forecasts generated "
                f"from {self.stats.total_questions_fetched} questions "
                f"({self.stats.success_rate:.1f}% success rate)"
            )

            return forecasts, self.stats

        except Exception as e:
            error_msg = f"Critical error in dispatcher: {str(e)}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            raise DispatcherError(error_msg) from e

    def run_batch(
        self, total_limit: int, status: str = "open", category: Optional[str] = None
    ) -> Tuple[List[Forecast], DispatcherStats]:
        """
        Run the pipeline in batches for large datasets.

        Args:
            total_limit: Total number of questions to process
            status: Question status filter
            category: Question category filter

        Returns:
            Tuple of (all generated forecasts, combined statistics)
        """
        all_forecasts = []
        combined_stats = DispatcherStats()

        processed = 0
        batch_num = 1

        while processed < total_limit:
            current_batch_size = min(self.config.batch_size, total_limit - processed)

            logger.info(f"Processing batch {batch_num}, size {current_batch_size}")

            try:
                batch_forecasts, batch_stats = self.run(
                    limit=current_batch_size, status=status, category=category
                )

                all_forecasts.extend(batch_forecasts)
                self._merge_stats(combined_stats, batch_stats)

                processed += current_batch_size
                batch_num += 1

            except DispatcherError as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                combined_stats.errors.append(f"Batch {batch_num}: {str(e)}")
                processed += current_batch_size  # Skip this batch

        return all_forecasts, combined_stats

    def _fetch_questions(
        self, limit: Optional[int], status: str, category: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Fetch questions from the API with error handling."""
        try:
            raw_questions = self.api.fetch_questions(
                limit=limit, status=status, category=category
            )

            logger.info(f"Fetched {len(raw_questions)} questions from API")
            return raw_questions

        except MetaculusAPIError as e:
            error_msg = f"Failed to fetch questions: {str(e)}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return []
        except Exception as e:
            # For critical errors, let them propagate so they can be caught by the main run method
            # which will wrap them in DispatcherError
            error_msg = f"Unexpected error fetching questions: {str(e)}"
            logger.error(error_msg)
            raise

    def _parse_questions(self, raw_questions: List[Dict[str, Any]]) -> List[Question]:
        """Parse raw questions into domain objects."""
        try:
            questions, ingestion_stats = self.ingestion_service.parse_questions(
                raw_questions
            )

            # Update dispatcher stats with ingestion results
            self.stats.questions_successfully_parsed = ingestion_stats.successful_parsed
            self.stats.questions_failed_parsing = ingestion_stats.failed_parsing

            # Merge ingestion errors
            if ingestion_stats.failed_parsing > 0:
                self.stats.errors.append(
                    f"Ingestion failed for {ingestion_stats.failed_parsing} questions"
                )

            logger.info(f"Parsed {len(questions)} questions successfully")
            return questions

        except Exception as e:
            error_msg = f"Critical error during question parsing: {str(e)}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return []

    def _generate_forecasts(self, questions: List[Question]) -> List[Forecast]:
        """Generate forecasts for all questions."""
        forecasts = []

        for question in questions:
            try:
                if self.config.enable_dry_run:
                    logger.info(
                        f"DRY RUN: Would generate forecast for question {question.metaculus_id}"
                    )
                    continue

                # Generate forecast using ForecastService
                forecast = self.forecast_service.generate_forecast(question)
                forecasts.append(forecast)
                self.stats.forecasts_generated += 1

                logger.debug(f"Generated forecast for question {question.metaculus_id}")

            except Exception as e:
                error_msg = f"Failed to generate forecast for question {question.metaculus_id}: {str(e)}"
                logger.warning(error_msg)
                self.stats.forecasts_failed += 1
                self.stats.errors.append(error_msg)

        logger.info(
            f"Generated {len(forecasts)} forecasts, {self.stats.forecasts_failed} failed"
        )
        return forecasts

    def _merge_stats(self, combined: DispatcherStats, batch: DispatcherStats) -> None:
        """Merge batch statistics into combined statistics."""
        combined.total_questions_fetched += batch.total_questions_fetched
        combined.questions_successfully_parsed += batch.questions_successfully_parsed
        combined.questions_failed_parsing += batch.questions_failed_parsing
        combined.forecasts_generated += batch.forecasts_generated
        combined.forecasts_failed += batch.forecasts_failed
        combined.total_processing_time_seconds += batch.total_processing_time_seconds
        combined.errors.extend(batch.errors)

    def get_status(self) -> Dict[str, Any]:
        """Get current dispatcher status and configuration."""
        return {
            "config": {
                "batch_size": self.config.batch_size,
                "validation_level": self.config.validation_level.value,
                "max_retries": self.config.max_retries,
                "enable_dry_run": self.config.enable_dry_run,
            },
            "stats": {
                "total_questions_fetched": self.stats.total_questions_fetched,
                "questions_successfully_parsed": self.stats.questions_successfully_parsed,
                "questions_failed_parsing": self.stats.questions_failed_parsing,
                "forecasts_generated": self.stats.forecasts_generated,
                "forecasts_failed": self.stats.forecasts_failed,
                "success_rate": self.stats.success_rate,
                "total_processing_time_seconds": self.stats.total_processing_time_seconds,
                "error_count": len(self.stats.errors),
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    async def run_tournament(
        self,
        tournament_id: int,
        max_questions: int = 10,
        agent_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run tournament forecasting with all integrated components.

        Args:
            tournament_id: Metaculus tournament ID
            max_questions: Maximum number of questions to process
            agent_types: List of agent types to use

        Returns:
            Tournament results with comprehensive metrics
        """
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
            # Use existing run method but with tournament-specific logic
            forecasts, stats = self.run(limit=max_questions, status="open")

            # Update results with stats
            results["questions_processed"] = stats.questions_successfully_parsed
            results["forecasts_generated"] = stats.forecasts_generated
            results["errors"] = stats.errors
            results["success_rate"] = stats.success_rate
            results["processing_time_seconds"] = stats.total_processing_time_seconds

            # Add tournament-specific analytics if available
            if self.tournament_analytics:
                try:
                    tournament_metrics = (
                        await self.tournament_analytics.analyze_tournament_performance(
                            tournament_id, forecasts
                        )
                    )
                    results["performance_metrics"] = tournament_metrics
                except Exception as e:
                    logger.warning(f"Failed to generate tournament analytics: {e}")

            # Add performance tracking if available
            if self.performance_tracking:
                try:
                    agent_performance = (
                        await self.performance_tracking.get_agent_performance_summary()
                    )
                    results["agent_performance"] = agent_performance
                except Exception as e:
                    logger.warning(f"Failed to get agent performance: {e}")

            return results

        except Exception as e:
            logger.error(f"Tournament run failed: {e}")
            results["error"] = str(e)
            results["end_time"] = datetime.now(timezone.utc).isoformat()
            raise
