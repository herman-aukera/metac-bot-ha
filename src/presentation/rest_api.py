"""REST API for Tournament Optimization System.

This module provides a comprehensive REST API for interacting with the tournament
optimization system, including endpoints for question processing, tournament analysis,
system monitoring, and result management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from ..application.use_cases.process_tournament_question import (
    ProcessTournamentQuestionUseCase, ProcessingResult
)
from ..application.services.forecasting_pipeline import ForecastingPipeline
from ..application.services.tournament_service import TournamentService
from ..application.services.learning_service import LearningService
from ..domain.entities.question import QuestionType, QuestionCategory
from ..infrastructure.logging.structured_logger import StructuredLogger
from ..infrastructure.monitoring.health_check_manager import HealthCheckManager


logger = StructuredLogger(__name__)


# Pydantic models for API requests/responses

class QuestionProcessRequest(BaseModel):
    """Request model for processing a question."""
    question_id: int = Field(..., description="ID of the question to process")
    tournament_id: Optional[int] = Field(None, description="Optional tournament context")
    force_reprocess: bool = Field(False, description="Force reprocessing if already processed")
    submission_mode: bool = Field(True, description="Whether to submit the forecast")
    strategy_override: Optional[str] = Field(None, description="Override strategy type")


class TournamentProcessRequest(BaseModel):
    """Request model for processing a tournament."""
    tournament_id: int = Field(..., description="Tournament ID to process")
    max_questions: Optional[int] = Field(None, description="Maximum questions to process")
    parallel_processing: bool = Field(True, description="Whether to process in parallel")
    dry_run: bool = Field(False, description="Run without submitting forecasts")
    priority_filter: Optional[str] = Field(None, description="Filter by priority level")


class StrategyAnalysisRequest(BaseModel):
    """Request model for strategy analysis."""
    tournament_id: int = Field(..., description="Tournament ID to analyze")
    include_alternatives: bool = Field(True, description="Include alternative strategies")
    risk_tolerance: Optional[float] = Field(None, description="Risk tolerance override")


class ExportRequest(BaseModel):
    """Request model for exporting results."""
    tournament_id: Optional[int] = Field(None, description="Tournament ID to export")
    question_ids: Optional[List[int]] = Field(None, description="Specific question IDs")
    date_from: Optional[datetime] = Field(None, description="Start date for results")
    date_to: Optional[datetime] = Field(None, description="End date for results")
    format: str = Field("json", description="Export format")
    include_metadata: bool = Field(True, description="Include detailed metadata")


class ProcessingResponse(BaseModel):
    """Response model for question processing."""
    correlation_id: str
    question_id: int
    success: bool
    processing_time: float
    final_forecast: Optional[Dict[str, Any]] = None
    research_reports_count: int
    ensemble_forecasts_count: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class TournamentResponse(BaseModel):
    """Response model for tournament processing."""
    tournament_id: int
    total_questions: int
    successful: int
    failed: int
    success_rate: float
    total_processing_time: float
    average_processing_time: float
    results: List[ProcessingResponse]


class StrategyResponse(BaseModel):
    """Response model for strategy analysis."""
    tournament_id: int
    strategy_type: str
    confidence: float
    expected_score_impact: float
    risk_level: float
    reasoning: str
    question_allocation: Dict[int, float]
    alternatives: List[Dict[str, Any]]
    resource_requirements: Dict[str, float]


class HealthResponse(BaseModel):
    """Response model for health checks."""
    timestamp: datetime
    overall_health: str
    health_checks: Dict[str, Dict[str, Any]]
    pipeline_metrics: Dict[str, Any]
    learning_stats: Dict[str, Any]


class TournamentAPI:
    """Main API class for tournament optimization system."""

    def __init__(self,
                 process_question_use_case: ProcessTournamentQuestionUseCase,
                 forecasting_pipeline: ForecastingPipeline,
                 tournament_service: TournamentService,
                 learning_service: LearningService,
                 health_check_manager: HealthCheckManager):
        """Initialize API with required services.

        Args:
            process_question_use_case: Main question processing use case
            forecasting_pipeline: Forecasting pipeline service
            tournament_service: Tournament strategy service
            learning_service: Learning and adaptation service
            health_check_manager: Health check manager
        """
        self.process_question_use_case = process_question_use_case
        self.forecasting_pipeline = forecasting_pipeline
        self.tournament_service = tournament_service
        self.learning_service = learning_service
        self.health_check_manager = health_check_manager

        # Initialize FastAPI app
        self.app = FastAPI(
            title="Tournament Optimization API",
            description="REST API for the Tournament Optimization System",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register all API routes."""

        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "Tournament Optimization API",
                "version": "1.0.0",
                "status": "operational",
                "timestamp": datetime.utcnow().isoformat(),
                "endpoints": {
                    "health": "/health",
                    "docs": "/docs",
                    "questions": "/questions",
                    "tournaments": "/tournaments",
                    "strategy": "/strategy",
                    "export": "/export"
                }
            }

        @self.app.get("/health", response_model=HealthResponse, tags=["System"])
        async def health_check():
            """Comprehensive health check endpoint."""
            try:
                # Run health checks
                health_results = await self.health_check_manager.run_all_checks()

                # Get system metrics
                pipeline_metrics = {
                    'cache_hit_rate': (
                        self.forecasting_pipeline.metrics.cache_hits /
                        (self.forecasting_pipeline.metrics.cache_hits +
                         self.forecasting_pipeline.metrics.cache_misses)
                        if (self.forecasting_pipeline.metrics.cache_hits +
                            self.forecasting_pipeline.metrics.cache_misses) > 0 else 0
                    ),
                    'consensus_strength': self.forecasting_pipeline.metrics.consensus_strength,
                    'agents_executed': self.forecasting_pipeline.metrics.agents_executed,
                    'research_sources_used': self.forecasting_pipeline.metrics.research_sources_used,
                    'recent_errors': self.forecasting_pipeline.metrics.errors_encountered[-5:]
                }

                # Get learning stats
                learning_stats = await self.learning_service.get_system_stats()

                overall_health = "healthy" if all(
                    check.is_healthy for check in health_results.values()
                ) else "degraded"

                return HealthResponse(
                    timestamp=datetime.utcnow(),
                    overall_health=overall_health,
                    health_checks={
                        name: {
                            'status': 'healthy' if check.is_healthy else 'unhealthy',
                            'message': check.message,
                            'last_check': check.timestamp.isoformat()
                        }
                        for name, check in health_results.items()
                    },
                    pipeline_metrics=pipeline_metrics,
                    learning_stats=learning_stats
                )

            except Exception as e:
                logger.error("Health check failed", extra={'error': str(e)})
                raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

        @self.app.post("/questions/process", response_model=ProcessingResponse, tags=["Questions"])
        async def process_question(request: QuestionProcessRequest, background_tasks: BackgroundTasks):
            """Process a single question through the forecasting pipeline."""
            try:
                # Load question (in real implementation, this would fetch from API)
                question = await self._load_question(request.question_id)
                if not question:
                    raise HTTPException(status_code=404, detail=f"Question {request.question_id} not found")

                # Execute processing
                result = await self.process_question_use_case.execute(
                    question=question,
                    tournament_id=request.tournament_id,
                    force_reprocess=request.force_reprocess,
                    submission_mode=request.submission_mode
                )

                # Log processing result
                logger.info(
                    "Question processed via API",
                    extra={
                        'question_id': request.question_id,
                        'tournament_id': request.tournament_id,
                        'success': result.success,
                        'processing_time': result.processing_time
                    }
                )

                # Convert to response model
                return ProcessingResponse(
                    correlation_id=result.correlation_id,
                    question_id=result.question_id,
                    success=result.success,
                    processing_time=result.processing_time,
                    final_forecast=self._serialize_forecast(result.final_forecast) if result.final_forecast else None,
                    research_reports_count=len(result.research_reports),
                    ensemble_forecasts_count=len(result.ensemble_forecasts),
                    error_message=result.error_message,
                    metadata=result.metadata
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    "Question processing failed via API",
                    extra={'question_id': request.question_id, 'error': str(e)},
                    exc_info=True
                )
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

        @self.app.post("/tournaments/process", response_model=TournamentResponse, tags=["Tournaments"])
        async def process_tournament(request: TournamentProcessRequest, background_tasks: BackgroundTasks):
            """Process all questions in a tournament."""
            try:
                # Get tournament
                tournament = await self.tournament_service.get_tournament(request.tournament_id)
                if not tournament:
                    raise HTTPException(status_code=404, detail=f"Tournament {request.tournament_id} not found")

                # Get questions to process
                active_questions = tournament.get_active_questions()
                if request.max_questions:
                    active_questions = active_questions[:request.max_questions]

                # Process questions
                if request.parallel_processing:
                    results = await self._process_questions_parallel(
                        active_questions, request.tournament_id, request.dry_run
                    )
                else:
                    results = await self._process_questions_sequential(
                        active_questions, request.tournament_id, request.dry_run
                    )

                # Aggregate results
                successful = [r for r in results if r.success]
                failed = [r for r in results if not r.success]

                logger.info(
                    "Tournament processed via API",
                    extra={
                        'tournament_id': request.tournament_id,
                        'total_questions': len(active_questions),
                        'successful': len(successful),
                        'failed': len(failed)
                    }
                )

                return TournamentResponse(
                    tournament_id=request.tournament_id,
                    total_questions=len(active_questions),
                    successful=len(successful),
                    failed=len(failed),
                    success_rate=len(successful) / len(active_questions) if active_questions else 0,
                    total_processing_time=sum(r.processing_time for r in results),
                    average_processing_time=sum(r.processing_time for r in results) / len(results) if results else 0,
                    results=[self._convert_to_processing_response(r) for r in results]
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    "Tournament processing failed via API",
                    extra={'tournament_id': request.tournament_id, 'error': str(e)},
                    exc_info=True
                )
                raise HTTPException(status_code=500, detail=f"Tournament processing failed: {str(e)}")

        @self.app.post("/strategy/analyze", response_model=StrategyResponse, tags=["Strategy"])
        async def analyze_strategy(request: StrategyAnalysisRequest):
            """Analyze tournament strategy and provide recommendations."""
            try:
                # Get tournament
                tournament = await self.tournament_service.get_tournament(request.tournament_id)
                if not tournament:
                    raise HTTPException(status_code=404, detail=f"Tournament {request.tournament_id} not found")

                # Analyze strategy
                strategy_recommendation = await self.tournament_service.analyze_tournament_strategy(tournament)

                logger.info(
                    "Strategy analyzed via API",
                    extra={
                        'tournament_id': request.tournament_id,
                        'strategy_type': strategy_recommendation.strategy_type.value,
                        'confidence': strategy_recommendation.confidence.level
                    }
                )

                return StrategyResponse(
                    tournament_id=request.tournament_id,
                    strategy_type=strategy_recommendation.strategy_type.value,
                    confidence=strategy_recommendation.confidence.level,
                    expected_score_impact=strategy_recommendation.expected_score_impact,
                    risk_level=strategy_recommendation.risk_level,
                    reasoning=strategy_recommendation.reasoning,
                    question_allocation=strategy_recommendation.question_allocation,
                    alternatives=[
                        {'strategy': alt[0].value, 'score': alt[1]}
                        for alt in strategy_recommendation.alternatives
                    ] if request.include_alternatives else [],
                    resource_requirements=strategy_recommendation.resource_requirements
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    "Strategy analysis failed via API",
                    extra={'tournament_id': request.tournament_id, 'error': str(e)},
                    exc_info=True
                )
                raise HTTPException(status_code=500, detail=f"Strategy analysis failed: {str(e)}")

        @self.app.get("/questions/{question_id}/status", tags=["Questions"])
        async def get_question_status(question_id: int = Path(..., description="Question ID")):
            """Get processing status for a specific question."""
            try:
                # Get question status from learning service
                status = await self.learning_service.get_question_status(question_id)

                if not status:
                    raise HTTPException(status_code=404, detail=f"Question {question_id} not found")

                return status

            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    "Question status retrieval failed",
                    extra={'question_id': question_id, 'error': str(e)}
                )
                raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

        @self.app.get("/tournaments/{tournament_id}/status", tags=["Tournaments"])
        async def get_tournament_status(tournament_id: int = Path(..., description="Tournament ID")):
            """Get processing status for a tournament."""
            try:
                # Get tournament status
                status = await self.learning_service.get_tournament_status(tournament_id)

                if not status:
                    raise HTTPException(status_code=404, detail=f"Tournament {tournament_id} not found")

                return status

            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    "Tournament status retrieval failed",
                    extra={'tournament_id': tournament_id, 'error': str(e)}
                )
                raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

        @self.app.post("/export", tags=["Export"])
        async def export_results(request: ExportRequest, background_tasks: BackgroundTasks):
            """Export processing results to file."""
            try:
                # Collect results based on request parameters
                if request.tournament_id:
                    results = await self.learning_service.get_tournament_results(
                        request.tournament_id,
                        date_from=request.date_from,
                        date_to=request.date_to
                    )
                elif request.question_ids:
                    results = await self.learning_service.get_question_results(
                        request.question_ids,
                        date_from=request.date_from,
                        date_to=request.date_to
                    )
                else:
                    results = await self.learning_service.get_all_recent_results(
                        date_from=request.date_from,
                        date_to=request.date_to
                    )

                # Generate export file
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filename = f"export_{timestamp}.{request.format}"

                # Schedule background export task
                background_tasks.add_task(
                    self._export_results_background,
                    results,
                    filename,
                    request.format,
                    request.include_metadata
                )

                return {
                    'export_id': timestamp,
                    'filename': filename,
                    'records_count': len(results),
                    'format': request.format,
                    'status': 'processing',
                    'download_url': f"/export/download/{filename}"
                }

            except Exception as e:
                logger.error(
                    "Export request failed",
                    extra={'error': str(e)},
                    exc_info=True
                )
                raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

        @self.app.get("/export/download/{filename}", tags=["Export"])
        async def download_export(filename: str = Path(..., description="Export filename")):
            """Download exported results file."""
            try:
                file_path = f"/tmp/{filename}"  # Configure appropriate export directory
                return FileResponse(
                    path=file_path,
                    filename=filename,
                    media_type='application/octet-stream'
                )

            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Export file not found")
            except Exception as e:
                logger.error(
                    "Export download failed",
                    extra={'filename': filename, 'error': str(e)}
                )
                raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

        @self.app.get("/metrics", tags=["System"])
        async def get_metrics():
            """Get system metrics in Prometheus format."""
            try:
                # Get pipeline metrics
                metrics = self.forecasting_pipeline.metrics

                # Format as Prometheus metrics
                prometheus_metrics = []
                prometheus_metrics.append(f"# HELP cache_hit_rate Cache hit rate")
                prometheus_metrics.append(f"# TYPE cache_hit_rate gauge")
                cache_hit_rate = (
                    metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
                    if (metrics.cache_hits + metrics.cache_misses) > 0 else 0
                )
                prometheus_metrics.append(f"cache_hit_rate {cache_hit_rate}")

                prometheus_metrics.append(f"# HELP consensus_strength Average consensus strength")
                prometheus_metrics.append(f"# TYPE consensus_strength gauge")
                prometheus_metrics.append(f"consensus_strength {metrics.consensus_strength}")

                prometheus_metrics.append(f"# HELP agents_executed Total agents executed")
                prometheus_metrics.append(f"# TYPE agents_executed counter")
                prometheus_metrics.append(f"agents_executed {metrics.agents_executed}")

                return "\n".join(prometheus_metrics)

            except Exception as e:
                logger.error("Metrics retrieval failed", extra={'error': str(e)})
                raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

    # Helper methods

    async def _load_question(self, question_id: int):
        """Load question by ID (placeholder implementation)."""
        # In real implementation, this would fetch from Metaculus API or database
        from ..domain.entities.question import Question, QuestionType, QuestionCategory
        return Question(
            id=question_id,
            text=f"Sample question {question_id}",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=datetime.utcnow().replace(hour=23, minute=59),
            background="Sample background information",
            resolution_criteria="Sample resolution criteria",
            scoring_weight=1.0
        )

    async def _process_questions_parallel(self, questions, tournament_id, dry_run):
        """Process questions in parallel."""
        semaphore = asyncio.Semaphore(5)

        async def process_single(question):
            async with semaphore:
                return await self.process_question_use_case.execute(
                    question=question,
                    tournament_id=tournament_id,
                    submission_mode=not dry_run
                )

        tasks = [process_single(q) for q in questions]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _process_questions_sequential(self, questions, tournament_id, dry_run):
        """Process questions sequentially."""
        results = []
        for question in questions:
            result = await self.process_question_use_case.execute(
                question=question,
                tournament_id=tournament_id,
                submission_mode=not dry_run
            )
            results.append(result)
        return results

    def _convert_to_processing_response(self, result: ProcessingResult) -> ProcessingResponse:
        """Convert ProcessingResult to ProcessingResponse."""
        return ProcessingResponse(
            correlation_id=result.correlation_id,
            question_id=result.question_id,
            success=result.success,
            processing_time=result.processing_time,
            final_forecast=self._serialize_forecast(result.final_forecast) if result.final_forecast else None,
            research_reports_count=len(result.research_reports),
            ensemble_forecasts_count=len(result.ensemble_forecasts),
            error_message=result.error_message,
            metadata=result.metadata
        )

    def _serialize_forecast(self, forecast) -> Dict[str, Any]:
        """Serialize forecast for API response."""
        if not forecast:
            return None

        return {
            'question_id': forecast.question_id,
            'prediction': forecast.prediction,
            'confidence': {
                'level': forecast.confidence.level,
                'basis': forecast.confidence.basis
            },
            'agent_id': forecast.agent_id,
            'timestamp': forecast.timestamp.isoformat(),
            'reasoning_steps': len(forecast.reasoning_trace),
            'evidence_sources': len(forecast.evidence_sources),
            'is_final': forecast.is_final,
            'metadata': forecast.metadata
        }

    async def _export_results_background(self, results, filename, format, include_metadata):
        """Background task for exporting results."""
        try:
            file_path = f"/tmp/{filename}"  # Configure appropriate export directory

            if format == 'json':
                import json
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format == 'csv':
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(file_path, index=False)
            elif format == 'xlsx':
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_excel(file_path, index=False)

            logger.info(
                "Export completed",
                extra={
                    'filename': filename,
                    'format': format,
                    'records': len(results)
                }
            )

        except Exception as e:
            logger.error(
                "Background export failed",
                extra={
                    'filename': filename,
                    'error': str(e)
                }
            )


def create_app(
    process_question_use_case: ProcessTournamentQuestionUseCase,
    forecasting_pipeline: ForecastingPipeline,
    tournament_service: TournamentService,
    learning_service: LearningService,
    health_check_manager: HealthCheckManager
) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        process_question_use_case: Main question processing use case
        forecasting_pipeline: Forecasting pipeline service
        tournament_service: Tournament strategy service
        learning_service: Learning and adaptation service
        health_check_manager: Health check manager

    Returns:
        Configured FastAPI application
    """
    api = TournamentAPI(
        process_question_use_case=process_question_use_case,
        forecasting_pipeline=forecasting_pipeline,
        tournament_service=tournament_service,
        learning_service=learning_service,
        health_check_manager=health_check_manager
    )

    return api.app


def run_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the API server.

    Args:
        app: FastAPI application
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    uvicorn.run(
        app,
        host=host,
        port=port,
        debug=debug,
        log_level="info" if not debug else "debug"
    )


if __name__ == "__main__":
    # Example usage (in real implementation, use dependency injection)
    app = create_app(None, None, None, None, None)
    run_server(app, debug=True)
