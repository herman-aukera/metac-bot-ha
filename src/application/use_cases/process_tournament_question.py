"""Process Tournament Question Use Case.

This module implements the comprehensive ProcessTournamentQuestion use case that
orchestrates the complete forecasting workflow from question ingestion through
research, reasoning, prediction, ensemble aggregation, and submission.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

from ...domain.entities.question import Question
from ...domain.entities.forecast import Forecast
from ...domain.entities.research_report import ResearchReport
from ...domain.value_objects.confidence import Confidence
from ...domain.value_objects.reasoning_step import ReasoningStep
from ...domain.exceptions.base_exceptions import TournamentOptimizationError
from ...infrastructure.logging.correlation_context import CorrelationContext
from ...infrastructure.logging.structured_logger import StructuredLogger
from ...infrastructure.resilience.circuit_breaker import CircuitBreaker
from ...infrastructure.resilience.retry_strategy import RetryStrategy
from ...infrastructure.resilience.graceful_degradation import GracefulDegradationManager
from ..services.tournament_service import TournamentService
from ..services.learning_service import LearningService


logger = StructuredLogger(__name__)


@dataclass
class ProcessingContext:
    """Context for question processing workflow."""
    correlation_id: str
    question: Question
    tournament_id: Optional[int]
    processing_start: datetime
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.correlation_id:
            object.__setattr__(self, 'correlation_id', str(uuid4()))
        if not self.processing_start:
            object.__setattr__(self, 'processing_start', datetime.utcnow())
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass
class ProcessingResult:
    """Result of question processing workflow."""
    correlation_id: str
    question_id: int
    final_forecast: Optional[Forecast]
    research_reports: List[ResearchReport]
    ensemble_forecasts: List[Forecast]
    processing_time: float
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]


class ProcessTournamentQuestionUseCase:
    """Comprehensive use case for processing tournament questions.

    This use case orchestrates the complete workflow:
    1. Question ingestion and validation
    2. Research phase with multi-provider evidence gathering
    3. Reasoning phase with multiple agent approaches
    4. Prediction generation with confidence assessment
    5. Ensemble aggregation with consensus analysis
    6. Risk management and calibration
    7. Submission preparation and execution
    """

    def __init__(self,
                 forecasting_pipeline: 'ForecastingPipeline',
                 tournament_service: TournamentService,
                 learning_service: LearningService,
                 circuit_breaker: CircuitBreaker,
                 retry_strategy: RetryStrategy,
                 degradation_manager: GracefulDegradationManager):
        """Initialize the use case with required services.

        Args:
            forecasting_pipeline: Main forecasting pipeline orchestrator
            tournament_service: Tournament strategy and optimization service
            learning_service: Learning and adaptation service
            circuit_breaker: Circuit breaker for external service protection
            retry_strategy: Retry strategy for transient failures
            degradation_manager: Graceful degradation manager
        """
        self.forecasting_pipeline = forecasting_pipeline
        self.tournament_service = tournament_service
        self.learning_service = learning_service
        self.circuit_breaker = circuit_breaker
        self.retry_strategy = retry_strategy
        self.degradation_manager = degradation_manager

    async def execute(self,
                     question: Question,
                     tournament_id: Optional[int] = None,
                     force_reprocess: bool = False,
                     submission_mode: bool = True) -> ProcessingResult:
        """Execute the complete question processing workflow.

        Args:
            question: Question to process
            tournament_id: Optional tournament context
            force_reprocess: Whether to force reprocessing if already processed
            submission_mode: Whether to submit the final forecast

        Returns:
            ProcessingResult containing all outputs and metadata
        """
        # Initialize processing context
        context = ProcessingContext(
            correlation_id=str(uuid4()),
            question=question,
            tournament_id=tournament_id,
            processing_start=datetime.utcnow(),
            metadata={
                'force_reprocess': force_reprocess,
                'submission_mode': submission_mode,
                'question_type': question.question_type.value,
                'question_category': question.category.value
            }
        )

        # Set correlation context for distributed tracing
        with CorrelationContext(context.correlation_id):
            logger.info(
                "Starting question processing workflow",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'tournament_id': tournament_id,
                    'question_type': question.question_type.value
                }
            )

            try:
                # Execute workflow with circuit breaker protection
                result = await self.circuit_breaker.call(
                    self._execute_workflow,
                    context
                )

                logger.info(
                    "Question processing completed successfully",
                    extra={
                        'correlation_id': context.correlation_id,
                        'question_id': question.id,
                        'processing_time': result.processing_time,
                        'success': result.success
                    }
                )

                return result

            except Exception as e:
                logger.error(
                    "Question processing failed",
                    extra={
                        'correlation_id': context.correlation_id,
                        'question_id': question.id,
                        'error': str(e),
                        'error_type': type(e).__name__
                    },
                    exc_info=True
                )

                # Return failure result
                processing_time = (datetime.utcnow() - context.processing_start).total_seconds()
                return ProcessingResult(
                    correlation_id=context.correlation_id,
                    question_id=question.id,
                    final_forecast=None,
                    research_reports=[],
                    ensemble_forecasts=[],
                    processing_time=processing_time,
                    success=False,
                    error_message=str(e),
                    metadata=context.metadata
                )

    async def _execute_workflow(self, context: ProcessingContext) -> ProcessingResult:
        """Execute the main processing workflow.

        Args:
            context: Processing context

        Returns:
            ProcessingResult with all outputs
        """
        question = context.question
        research_reports = []
        ensemble_forecasts = []
        final_forecast = None

        try:
            # Phase 1: Question Analysis and Strategy Selection
            logger.info(
                "Phase 1: Analyzing question and selecting strategy",
                extra={'correlation_id': context.correlation_id, 'question_id': question.id}
            )

            strategy_analysis = await self._analyze_question_strategy(question, context)
            context.metadata.update(strategy_analysis)

            # Phase 2: Research and Evidence Gathering
            logger.info(
                "Phase 2: Conducting research and evidence gathering",
                extra={'correlation_id': context.correlation_id, 'question_id': question.id}
            )

            research_reports = await self._conduct_research_phase(question, context)

            # Phase 3: Multi-Agent Reasoning and Prediction
            logger.info(
                "Phase 3: Multi-agent reasoning and prediction generation",
                extra={'correlation_id': context.correlation_id, 'question_id': question.id}
            )

            ensemble_forecasts = await self._generate_ensemble_predictions(
                question, research_reports, context
            )

            # Phase 4: Ensemble Aggregation and Consensus Analysis
            logger.info(
                "Phase 4: Ensemble aggregation and consensus analysis",
                extra={'correlation_id': context.correlation_id, 'question_id': question.id}
            )

            final_forecast = await self._aggregate_ensemble_forecasts(
                question, ensemble_forecasts, context
            )

            # Phase 5: Risk Management and Calibration
            logger.info(
                "Phase 5: Risk management and calibration",
                extra={'correlation_id': context.correlation_id, 'question_id': question.id}
            )

            final_forecast = await self._apply_risk_management(
                question, final_forecast, context
            )

            # Phase 6: Submission Preparation and Execution
            if context.metadata.get('submission_mode', True):
                logger.info(
                    "Phase 6: Submission preparation and execution",
                    extra={'correlation_id': context.correlation_id, 'question_id': question.id}
                )

                await self._prepare_and_submit_forecast(
                    question, final_forecast, context
                )

            # Phase 7: Learning and Adaptation
            logger.info(
                "Phase 7: Learning and adaptation",
                extra={'correlation_id': context.correlation_id, 'question_id': question.id}
            )

            await self._update_learning_systems(
                question, final_forecast, research_reports, context
            )

            # Calculate processing time
            processing_time = (datetime.utcnow() - context.processing_start).total_seconds()

            return ProcessingResult(
                correlation_id=context.correlation_id,
                question_id=question.id,
                final_forecast=final_forecast,
                research_reports=research_reports,
                ensemble_forecasts=ensemble_forecasts,
                processing_time=processing_time,
                success=True,
                error_message=None,
                metadata=context.metadata
            )

        except Exception as e:
            # Handle workflow failures with graceful degradation
            logger.error(
                "Workflow execution failed, attempting graceful degradation",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'error': str(e)
                }
            )

            # Attempt to provide partial results
            degraded_result = await self._handle_workflow_failure(
                question, research_reports, ensemble_forecasts, context, e
            )

            return degraded_result

    async def _analyze_question_strategy(self,
                                       question: Question,
                                       context: ProcessingContext) -> Dict[str, Any]:
        """Analyze question and select optimal strategy.

        Args:
            question: Question to analyze
            context: Processing context

        Returns:
            Dictionary containing strategy analysis results
        """
        try:
            # Get tournament context if available
            tournament = None
            if context.tournament_id:
                tournament = await self.tournament_service.get_tournament(context.tournament_id)

            # Analyze question characteristics
            complexity_score = question.get_complexity_score()
            specialization_required = question.requires_specialized_knowledge()
            time_remaining = question.time_until_deadline()

            # Select optimal strategy based on analysis
            if tournament:
                strategy_recommendation = await self.tournament_service.analyze_tournament_strategy(tournament)
                strategy_type = strategy_recommendation.strategy_type
            else:
                # Default strategy selection based on question characteristics
                if complexity_score > 2.0 and specialization_required:
                    strategy_type = "specialized_research"
                elif time_remaining < 6:  # Less than 6 hours
                    strategy_type = "rapid_response"
                else:
                    strategy_type = "comprehensive_analysis"

            return {
                'strategy_type': strategy_type,
                'complexity_score': complexity_score,
                'specialization_required': specialization_required,
                'time_remaining_hours': time_remaining,
                'tournament_context': tournament.id if tournament else None
            }

        except Exception as e:
            logger.warning(
                "Strategy analysis failed, using default strategy",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'error': str(e)
                }
            )
            return {
                'strategy_type': 'default',
                'complexity_score': 1.0,
                'specialization_required': False,
                'time_remaining_hours': 24.0,
                'tournament_context': None
            }

    async def _conduct_research_phase(self,
                                    question: Question,
                                    context: ProcessingContext) -> List[ResearchReport]:
        """Conduct comprehensive research phase.

        Args:
            question: Question to research
            context: Processing context

        Returns:
            List of research reports from different sources
        """
        try:
            # Execute research with retry strategy
            research_reports = await self.retry_strategy.execute(
                self.forecasting_pipeline.conduct_research,
                question,
                context.metadata.get('strategy_type', 'default')
            )

            logger.info(
                "Research phase completed",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'reports_generated': len(research_reports),
                    'total_sources': sum(len(report.sources) for report in research_reports)
                }
            )

            return research_reports

        except Exception as e:
            logger.warning(
                "Research phase failed, attempting degraded research",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'error': str(e)
                }
            )

            # Attempt degraded research with available providers
            degraded_reports = await self.degradation_manager.get_degraded_research(question)
            return degraded_reports or []

    async def _generate_ensemble_predictions(self,
                                           question: Question,
                                           research_reports: List[ResearchReport],
                                           context: ProcessingContext) -> List[Forecast]:
        """Generate predictions using ensemble of agents.

        Args:
            question: Question to predict
            research_reports: Research reports to use
            context: Processing context

        Returns:
            List of forecasts from different agents
        """
        try:
            # Generate ensemble predictions with circuit breaker protection
            ensemble_forecasts = await self.circuit_breaker.call(
                self.forecasting_pipeline.generate_ensemble_predictions,
                question,
                research_reports,
                context.metadata.get('strategy_type', 'default')
            )

            logger.info(
                "Ensemble prediction generation completed",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'forecasts_generated': len(ensemble_forecasts),
                    'agent_types': [f.agent_id for f in ensemble_forecasts if f.agent_id]
                }
            )

            return ensemble_forecasts

        except Exception as e:
            logger.warning(
                "Ensemble prediction failed, attempting single agent fallback",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'error': str(e)
                }
            )

            # Fallback to single agent prediction
            fallback_forecast = await self.degradation_manager.get_single_agent_prediction(
                question, research_reports
            )
            return [fallback_forecast] if fallback_forecast else []

    async def _aggregate_ensemble_forecasts(self,
                                          question: Question,
                                          ensemble_forecasts: List[Forecast],
                                          context: ProcessingContext) -> Optional[Forecast]:
        """Aggregate ensemble forecasts into final prediction.

        Args:
            question: Question being forecasted
            ensemble_forecasts: Individual agent forecasts
            context: Processing context

        Returns:
            Final aggregated forecast
        """
        if not ensemble_forecasts:
            logger.warning(
                "No ensemble forecasts available for aggregation",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id
                }
            )
            return None

        try:
            # Aggregate forecasts using the pipeline
            final_forecast = await self.forecasting_pipeline.aggregate_forecasts(
                question,
                ensemble_forecasts,
                context.metadata.get('strategy_type', 'default')
            )

            logger.info(
                "Forecast aggregation completed",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'input_forecasts': len(ensemble_forecasts),
                    'final_confidence': final_forecast.confidence.level if final_forecast else None
                }
            )

            return final_forecast

        except Exception as e:
            logger.error(
                "Forecast aggregation failed",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'error': str(e)
                }
            )

            # Return best individual forecast as fallback
            if ensemble_forecasts:
                best_forecast = max(ensemble_forecasts, key=lambda f: f.confidence.level)
                logger.info(
                    "Using best individual forecast as fallback",
                    extra={
                        'correlation_id': context.correlation_id,
                        'question_id': question.id,
                        'fallback_agent': best_forecast.agent_id,
                        'fallback_confidence': best_forecast.confidence.level
                    }
                )
                return best_forecast

            return None

    async def _apply_risk_management(self,
                                   question: Question,
                                   forecast: Optional[Forecast],
                                   context: ProcessingContext) -> Optional[Forecast]:
        """Apply risk management and calibration to forecast.

        Args:
            question: Question being forecasted
            forecast: Forecast to apply risk management to
            context: Processing context

        Returns:
            Risk-adjusted forecast
        """
        if not forecast:
            return None

        try:
            # Apply risk management through the pipeline
            risk_adjusted_forecast = await self.forecasting_pipeline.apply_risk_management(
                question,
                forecast,
                context.metadata.get('strategy_type', 'default')
            )

            logger.info(
                "Risk management applied",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'original_confidence': forecast.confidence.level,
                    'adjusted_confidence': risk_adjusted_forecast.confidence.level if risk_adjusted_forecast else None
                }
            )

            return risk_adjusted_forecast

        except Exception as e:
            logger.warning(
                "Risk management failed, using original forecast",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'error': str(e)
                }
            )
            return forecast

    async def _prepare_and_submit_forecast(self,
                                         question: Question,
                                         forecast: Optional[Forecast],
                                         context: ProcessingContext) -> None:
        """Prepare and submit forecast to tournament platform.

        Args:
            question: Question being forecasted
            forecast: Final forecast to submit
            context: Processing context
        """
        if not forecast:
            logger.warning(
                "No forecast available for submission",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id
                }
            )
            return

        try:
            # Submit forecast through the pipeline
            submission_result = await self.forecasting_pipeline.submit_forecast(
                question,
                forecast,
                context.tournament_id
            )

            # Update forecast with submission ID
            if submission_result and submission_result.get('submission_id'):
                forecast = forecast.with_submission_id(submission_result['submission_id'])

            logger.info(
                "Forecast submitted successfully",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'submission_id': submission_result.get('submission_id') if submission_result else None,
                    'submission_time': datetime.utcnow().isoformat()
                }
            )

            # Update context metadata
            context.metadata['submission_result'] = submission_result

        except Exception as e:
            logger.error(
                "Forecast submission failed",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'error': str(e)
                }
            )
            # Don't raise exception - submission failure shouldn't fail entire workflow
            context.metadata['submission_error'] = str(e)

    async def _update_learning_systems(self,
                                     question: Question,
                                     forecast: Optional[Forecast],
                                     research_reports: List[ResearchReport],
                                     context: ProcessingContext) -> None:
        """Update learning and adaptation systems with processing results.

        Args:
            question: Question that was processed
            forecast: Final forecast generated
            research_reports: Research reports used
            context: Processing context
        """
        try:
            # Update learning service with processing results
            await self.learning_service.record_processing_result(
                question=question,
                forecast=forecast,
                research_reports=research_reports,
                processing_metadata=context.metadata,
                correlation_id=context.correlation_id
            )

            logger.info(
                "Learning systems updated",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'forecast_available': forecast is not None,
                    'research_reports_count': len(research_reports)
                }
            )

        except Exception as e:
            logger.warning(
                "Learning system update failed",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'error': str(e)
                }
            )
            # Don't raise exception - learning failure shouldn't fail workflow

    async def _handle_workflow_failure(self,
                                     question: Question,
                                     research_reports: List[ResearchReport],
                                     ensemble_forecasts: List[Forecast],
                                     context: ProcessingContext,
                                     error: Exception) -> ProcessingResult:
        """Handle workflow failure with graceful degradation.

        Args:
            question: Question being processed
            research_reports: Any research reports generated
            ensemble_forecasts: Any ensemble forecasts generated
            context: Processing context
            error: The error that caused the failure

        Returns:
            ProcessingResult with partial results and error information
        """
        logger.info(
            "Attempting graceful degradation after workflow failure",
            extra={
                'correlation_id': context.correlation_id,
                'question_id': question.id,
                'available_research_reports': len(research_reports),
                'available_forecasts': len(ensemble_forecasts)
            }
        )

        # Try to provide best available forecast
        final_forecast = None
        if ensemble_forecasts:
            # Use best available forecast
            final_forecast = max(ensemble_forecasts, key=lambda f: f.confidence.level)
            logger.info(
                "Using best available forecast from partial results",
                extra={
                    'correlation_id': context.correlation_id,
                    'question_id': question.id,
                    'selected_agent': final_forecast.agent_id,
                    'confidence': final_forecast.confidence.level
                }
            )

        processing_time = (datetime.utcnow() - context.processing_start).total_seconds()

        return ProcessingResult(
            correlation_id=context.correlation_id,
            question_id=question.id,
            final_forecast=final_forecast,
            research_reports=research_reports,
            ensemble_forecasts=ensemble_forecasts,
            processing_time=processing_time,
            success=False,
            error_message=f"Workflow failed: {str(error)}",
            metadata={
                **context.metadata,
                'partial_results': True,
                'failure_phase': self._identify_failure_phase(error),
                'degradation_applied': True
            }
        )

    def _identify_failure_phase(self, error: Exception) -> str:
        """Identify which phase of the workflow failed.

        Args:
            error: The error that occurred

        Returns:
            String identifying the failure phase
        """
        error_message = str(error).lower()

        if 'research' in error_message or 'search' in error_message:
            return 'research_phase'
        elif 'prediction' in error_message or 'forecast' in error_message:
            return 'prediction_phase'
        elif 'ensemble' in error_message or 'aggregation' in error_message:
            return 'aggregation_phase'
        elif 'submission' in error_message:
            return 'submission_phase'
        else:
            return 'unknown_phase'
