"""
Application service for dispatching questions through the forecasting pipeline.

The Dispatcher orchestrates the flow from raw API data through ingestion
to forecast generation, handling errors and batching appropriately.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from src.infrastructure.metaculus_api import MetaculusAPI, APIConfig, MetaculusAPIError
from src.application.ingestion_service import IngestionService, ValidationLevel, IngestionStats
from src.application.forecast_service import ForecastService
from src.domain.entities.question import Question
from src.domain.entities.forecast import Forecast
from src.domain.value_objects.probability import Probability
from src.domain.value_objects.confidence import ConfidenceLevel


logger = logging.getLogger(__name__)


@dataclass
class DispatcherConfig:
    """Configuration for the dispatcher."""
    batch_size: int = 10
    validation_level: ValidationLevel = ValidationLevel.LENIENT
    max_retries: int = 3
    enable_dry_run: bool = False
    api_config: Optional[APIConfig] = None


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

    def __init__(self, 
                 forecast_service: Optional[ForecastService] = None,
                 config: Optional[DispatcherConfig] = None):
        """
        Initialize the dispatcher.
        
        Args:
            forecast_service: Optional forecast service instance for dependency injection
            config: Dispatcher configuration. Uses defaults if None.
        """
        self.config = config or DispatcherConfig()
        
        # Initialize services
        self.api = MetaculusAPI(config=self.config.api_config)
        self.ingestion_service = IngestionService(
            validation_level=self.config.validation_level
        )
        self.forecast_service = forecast_service or ForecastService()
        
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
            return self.forecast_service.generate_forecast(question)
        except Exception as e:
            error_msg = f"Failed to generate forecast for question {question.metaculus_id}: {str(e)}"
            logger.error(error_msg)
            raise DispatcherError(error_msg) from e

    def run(self, 
            limit: Optional[int] = None,
            status: str = "open",
            category: Optional[str] = None) -> Tuple[List[Forecast], DispatcherStats]:
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
            logger.info(f"Starting dispatcher run with limit={limit}, status={status}, category={category}")
            
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
            self.stats.total_processing_time_seconds = (end_time - start_time).total_seconds()
            
            logger.info(f"Dispatcher run completed: {len(forecasts)} forecasts generated "
                       f"from {self.stats.total_questions_fetched} questions "
                       f"({self.stats.success_rate:.1f}% success rate)")
            
            return forecasts, self.stats
            
        except Exception as e:
            error_msg = f"Critical error in dispatcher: {str(e)}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            raise DispatcherError(error_msg) from e

    def run_batch(self, 
                  total_limit: int,
                  status: str = "open", 
                  category: Optional[str] = None) -> Tuple[List[Forecast], DispatcherStats]:
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
                    limit=current_batch_size,
                    status=status,
                    category=category
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

    def _fetch_questions(self, limit: Optional[int], status: str, category: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch questions from the API with error handling."""
        try:
            raw_questions = self.api.fetch_questions(
                limit=limit,
                status=status,
                category=category
            )
            
            logger.info(f"Fetched {len(raw_questions)} questions from API")
            return raw_questions
            
        except MetaculusAPIError as e:
            error_msg = f"Failed to fetch questions: {str(e)}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return []
        except Exception as e:
            error_msg = f"Unexpected error fetching questions: {str(e)}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return []

    def _parse_questions(self, raw_questions: List[Dict[str, Any]]) -> List[Question]:
        """Parse raw questions into domain objects."""
        try:
            questions, ingestion_stats = self.ingestion_service.parse_questions(raw_questions)
            
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
                    logger.info(f"DRY RUN: Would generate forecast for question {question.metaculus_id}")
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
        
        logger.info(f"Generated {len(forecasts)} forecasts, {self.stats.forecasts_failed} failed")
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
                "enable_dry_run": self.config.enable_dry_run
            },
            "stats": {
                "total_questions_fetched": self.stats.total_questions_fetched,
                "questions_successfully_parsed": self.stats.questions_successfully_parsed,
                "questions_failed_parsing": self.stats.questions_failed_parsing,
                "forecasts_generated": self.stats.forecasts_generated,
                "forecasts_failed": self.stats.forecasts_failed,
                "success_rate": self.stats.success_rate,
                "total_processing_time_seconds": self.stats.total_processing_time_seconds,
                "error_count": len(self.stats.errors)
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }