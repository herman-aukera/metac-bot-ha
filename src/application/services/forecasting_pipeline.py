"""Forecasting Pipeline for orchestrating all tournament optimization components.

This module implements the main ForecastingPipeline that coordinates:
- Multi-provider research and evidence gathering
- Multi-agent reasoning and prediction generation
- Ensemble aggregation with sophisticated methods
- Risk management and calibration
- Tournament strategy optimization
- Submission and feedback loops
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from ...domain.entities.question import Question, QuestionType
from ...domain.entities.forecast import Forecast
from ...domain.entities.research_report import ResearchReport
from ...domain.entities.agent import Agent
from ...domain.value_objects.confidence import Confidence
from ...domain.value_objects.reasoning_step import ReasoningStep
from ...domain.value_objects.consensus_metrics import ConsensusMetrics
from ...domain.services.agent_orchestration import (
    BaseAgent, ChainOfThoughtAgent, TreeOfThoughtAgent,
    ReActAgent, EnsembleAgent, AggregationMethod
)
from ...infrastructure.research.search_client import SearchClient
from ...infrastructure.logging.structured_logger import StructuredLogger
from ...infrastructure.resilience.circuit_breaker import CircuitBreaker
from ...infrastructure.resilience.retry_strategy import RetryStrategy
from ...infrastructure.cache.cache_manager import CacheManager
from .tournament_service import TournamentService
from .learning_service import LearningService


logger = StructuredLogger(__name__)


class PipelineStage(Enum):
    """Stages of the forecasting pipeline."""
    INITIALIZATION = "initialization"
    RESEARCH = "research"
    REASONING = "reasoning"
    PREDICTION = "prediction"
    ENSEMBLE = "ensemble"
    RISK_MANAGEMENT = "risk_management"
    SUBMISSION = "submission"
    LEARNING = "learning"


@dataclass
class PipelineConfig:
    """Configuration for the forecasting pipeline."""
    # Research configuration
    max_research_time_minutes: int = 15
    min_research_sources: int = 3
    research_providers: List[str] = None

    # Agent configuration
    enabled_agents: List[str] = None
    max_agents: int = 5
    agent_timeout_minutes: int = 10

    # Ensemble configuration
    aggregation_method: AggregationMethod = AggregationMethod.CONFIDENCE_WEIGHTED
    min_consensus_threshold: float = 0.6

    # Risk management
    confidence_adjustment_factor: float = 0.9
    extreme_prediction_threshold: float = 0.05

    # Performance
    max_concurrent_operations: int = 10
    cache_ttl_hours: int = 24

    def __post_init__(self):
        if self.research_providers is None:
            self.research_providers = ['asknews', 'perplexity', 'exa', 'serpapi']
        if self.enabled_agents is None:
            self.enabled_agents = ['cot', 'tot', 'react', 'ensemble']


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    stage_timings: Dict[PipelineStage, float]
    research_sources_used: int
    agents_executed: int
    consensus_strength: float
    final_confidence: float
    cache_hits: int
    cache_misses: int
    errors_encountered: List[str]

    def __post_init__(self):
        if not hasattr(self, 'stage_timings'):
            self.stage_timings = {}
        if not hasattr(self, 'errors_encountered'):
            self.errors_encountered = []


class ForecastingPipeline:
    """Main forecasting pipeline orchestrating all components.

    This pipeline coordinates the complete forecasting workflow:
    1. Research: Multi-provider evidence gathering with source credibility analysis
    2. Reasoning: Multi-agent reasoning with different approaches (CoT, ToT, ReAct)
    3. Prediction: Individual agent prediction generation with confidence assessment
    4. Ensemble: Sophisticated aggregation of multiple predictions
    5. Risk Management: Calibration and risk adjustment
    6. Submission: Tournament platform submission with timing optimization
    7. Learning: Feedback incorporation and strategy adaptation
    """

    def __init__(self,
                 search_client: SearchClient,
                 agents: Dict[str, BaseAgent],
                 ensemble_agent: EnsembleAgent,
                 tournament_service: TournamentService,
                 learning_service: LearningService,
                 circuit_breaker: CircuitBreaker,
                 retry_strategy: RetryStrategy,
                 cache_manager: CacheManager,
                 config: Optional[PipelineConfig] = None):
        """Initialize the forecasting pipeline.

        Args:
            search_client: Multi-provider search client
            agents: Dictionary of available agents by type
            ensemble_agent: Ensemble aggregation agent
            tournament_service: Tournament strategy service
            learning_service: Learning and adaptation service
            circuit_breaker: Circuit breaker for resilience
            retry_strategy: Retry strategy for transient failures
            cache_manager: Cache manager for performance
            config: Pipeline configuration
        """
        self.search_client = search_client
        self.agents = agents
        self.ensemble_agent = ensemble_agent
        self.tournament_service = tournament_service
        self.learning_service = learning_service
        self.circuit_breaker = circuit_breaker
        self.retry_strategy = retry_strategy
        self.cache_manager = cache_manager
        self.config = config or PipelineConfig()

        # Initialize metrics
        self.metrics = PipelineMetrics(
            stage_timings={},
            research_sources_used=0,
            agents_executed=0,
            consensus_strength=0.0,
            final_confidence=0.0,
            cache_hits=0,
            cache_misses=0,
            errors_encountered=[]
        )

    async def process_question(self,
                             question: Question,
                             strategy_type: str = 'default',
                             tournament_id: Optional[int] = None) -> Forecast:
        """Process a question through the complete forecasting pipeline.

        Args:
            question: Question to process
            strategy_type: Strategy type for processing
            tournament_id: Optional tournament context

        Returns:
            Final forecast with all metadata
        """
        pipeline_start = datetime.utcnow()

        logger.info(
            "Starting forecasting pipeline",
            extra={
                'question_id': question.id,
                'question_type': question.question_type.value,
                'strategy_type': strategy_type,
                'tournament_id': tournament_id
            }
        )

        try:
            # Stage 1: Research Phase
            research_reports = await self._execute_stage(
                PipelineStage.RESEARCH,
                self.conduct_research,
                question,
                strategy_type
            )

            # Stage 2: Multi-Agent Prediction Generation
            ensemble_forecasts = await self._execute_stage(
                PipelineStage.PREDICTION,
                self.generate_ensemble_predictions,
                question,
                research_reports,
                strategy_type
            )

            # Stage 3: Ensemble Aggregation
            final_forecast = await self._execute_stage(
                PipelineStage.ENSEMBLE,
                self.aggregate_forecasts,
                question,
                ensemble_forecasts,
                strategy_type
            )

            # Stage 4: Risk Management
            final_forecast = await self._execute_stage(
                PipelineStage.RISK_MANAGEMENT,
                self.apply_risk_management,
                question,
                final_forecast,
                strategy_type
            )

            # Update final metrics
            pipeline_time = (datetime.utcnow() - pipeline_start).total_seconds()
            self.metrics.final_confidence = final_forecast.confidence.level if final_forecast else 0.0

            logger.info(
                "Forecasting pipeline completed successfully",
                extra={
                    'question_id': question.id,
                    'pipeline_time': pipeline_time,
                    'final_confidence': self.metrics.final_confidence,
                    'research_sources': self.metrics.research_sources_used,
                    'agents_executed': self.metrics.agents_executed
                }
            )

            return final_forecast

        except Exception as e:
            logger.error(
                "Forecasting pipeline failed",
                extra={
                    'question_id': question.id,
                    'error': str(e),
                    'pipeline_time': (datetime.utcnow() - pipeline_start).total_seconds()
                },
                exc_info=True
            )
            raise

    async def conduct_research(self,
                             question: Question,
                             strategy_type: str = 'default') -> List[ResearchReport]:
        """Conduct comprehensive research using multiple providers.

        Args:
            question: Question to research
            strategy_type: Research strategy type

        Returns:
            List of research reports from different sources
        """
        # Check cache first
        cache_key = f"research:{question.id}:{strategy_type}"
        cached_reports = await self.cache_manager.get(cache_key)

        if cached_reports:
            self.metrics.cache_hits += 1
            logger.info(
                "Using cached research reports",
                extra={'question_id': question.id, 'reports_count': len(cached_reports)}
            )
            return cached_reports

        self.metrics.cache_misses += 1

        # Determine research parameters based on strategy
        research_params = self._get_research_parameters(question, strategy_type)

        # Execute research with circuit breaker protection
        research_reports = await self.circuit_breaker.call(
            self._execute_research_with_providers,
            question,
            research_params
        )

        # Cache results
        await self.cache_manager.set(
            cache_key,
            research_reports,
            ttl_hours=self.config.cache_ttl_hours
        )

        self.metrics.research_sources_used = sum(
            len(report.sources) for report in research_reports
        )

        logger.info(
            "Research phase completed",
            extra={
                'question_id': question.id,
                'reports_generated': len(research_reports),
                'total_sources': self.metrics.research_sources_used
            }
        )

        return research_reports

    async def generate_ensemble_predictions(self,
                                          question: Question,
                                          research_reports: List[ResearchReport],
                                          strategy_type: str = 'default') -> List[Forecast]:
        """Generate predictions using ensemble of agents.

        Args:
            question: Question to predict
            research_reports: Research reports to use
            strategy_type: Prediction strategy type

        Returns:
            List of forecasts from different agents
        """
        # Select agents based on strategy and question characteristics
        selected_agents = self._select_agents_for_question(question, strategy_type)

        # Generate predictions concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)

        async def generate_agent_prediction(agent_id: str, agent: BaseAgent) -> Optional[Forecast]:
            async with semaphore:
                try:
                    # Execute with timeout and retry
                    forecast = await asyncio.wait_for(
                        self.retry_strategy.execute(
                            agent.forecast,
                            question,
                            research_reports
                        ),
                        timeout=self.config.agent_timeout_minutes * 60
                    )

                    # Set agent ID for tracking
                    if forecast:
                        forecast.agent_id = agent_id

                    return forecast

                except Exception as e:
                    logger.warning(
                        f"Agent {agent_id} prediction failed",
                        extra={
                            'question_id': question.id,
                            'agent_id': agent_id,
                            'error': str(e)
                        }
                    )
                    self.metrics.errors_encountered.append(f"Agent {agent_id}: {str(e)}")
                    return None

        # Execute all agents concurrently
        tasks = [
            generate_agent_prediction(agent_id, agent)
            for agent_id, agent in selected_agents.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful predictions
        ensemble_forecasts = [
            result for result in results
            if isinstance(result, Forecast) and result is not None
        ]

        self.metrics.agents_executed = len(ensemble_forecasts)

        logger.info(
            "Ensemble prediction generation completed",
            extra={
                'question_id': question.id,
                'agents_attempted': len(selected_agents),
                'successful_predictions': len(ensemble_forecasts),
                'failed_predictions': len(selected_agents) - len(ensemble_forecasts)
            }
        )

        return ensemble_forecasts

    async def aggregate_forecasts(self,
                                question: Question,
                                ensemble_forecasts: List[Forecast],
                                strategy_type: str = 'default') -> Optional[Forecast]:
        """Aggregate ensemble forecasts into final prediction.

        Args:
            question: Question being forecasted
            ensemble_forecasts: Individual agent forecasts
            strategy_type: Aggregation strategy type

        Returns:
            Final aggregated forecast
        """
        if not ensemble_forecasts:
            logger.warning(
                "No forecasts available for aggregation",
                extra={'question_id': question.id}
            )
            return None

        if len(ensemble_forecasts) == 1:
            logger.info(
                "Single forecast available, using directly",
                extra={
                    'question_id': question.id,
                    'agent_id': ensemble_forecasts[0].agent_id
                }
            )
            return ensemble_forecasts[0]

        # Calculate consensus metrics
        consensus_metrics = self._calculate_consensus_metrics(ensemble_forecasts)
        self.metrics.consensus_strength = consensus_metrics.consensus_strength

        # Select aggregation method based on consensus and strategy
        aggregation_method = self._select_aggregation_method(
            consensus_metrics, strategy_type
        )

        # Perform aggregation
        aggregated_forecast = await self.ensemble_agent.aggregate_predictions(
            question,
            ensemble_forecasts,
            aggregation_method
        )

        # Add ensemble metadata
        if aggregated_forecast:
            aggregated_forecast.metadata = {
                **aggregated_forecast.metadata,
                'ensemble_size': len(ensemble_forecasts),
                'consensus_metrics': consensus_metrics.to_dict(),
                'aggregation_method': aggregation_method.value,
                'individual_confidences': [f.confidence.level for f in ensemble_forecasts],
                'agent_ids': [f.agent_id for f in ensemble_forecasts if f.agent_id]
            }

        logger.info(
            "Forecast aggregation completed",
            extra={
                'question_id': question.id,
                'input_forecasts': len(ensemble_forecasts),
                'aggregation_method': aggregation_method.value,
                'consensus_strength': consensus_metrics.consensus_strength,
                'final_confidence': aggregated_forecast.confidence.level if aggregated_forecast else None
            }
        )

        return aggregated_forecast

    async def apply_risk_management(self,
                                  question: Question,
                                  forecast: Optional[Forecast],
                                  strategy_type: str = 'default') -> Optional[Forecast]:
        """Apply risk management and calibration to forecast.

        Args:
            question: Question being forecasted
            forecast: Forecast to apply risk management to
            strategy_type: Risk management strategy type

        Returns:
            Risk-adjusted forecast
        """
        if not forecast:
            return None

        # Apply confidence adjustment based on question characteristics
        confidence_adjustment = self._calculate_confidence_adjustment(question, forecast)

        # Check for extreme predictions that might need moderation
        prediction_adjustment = self._calculate_prediction_adjustment(question, forecast)

        # Apply calibration based on historical performance
        calibration_adjustment = await self._get_calibration_adjustment(question, forecast)

        # Create risk-adjusted forecast
        adjusted_confidence_level = max(0.1, min(1.0,
            forecast.confidence.level * confidence_adjustment * calibration_adjustment
        ))

        adjusted_confidence = Confidence(
            level=adjusted_confidence_level,
            basis=f"Risk-adjusted from {forecast.confidence.level:.3f} "
                  f"(confidence_adj: {confidence_adjustment:.3f}, "
                  f"calibration_adj: {calibration_adjustment:.3f})"
        )

        # Apply prediction adjustment if needed
        adjusted_prediction = self._apply_prediction_adjustment(
            forecast.prediction, prediction_adjustment
        )

        # Create new forecast with adjustments
        risk_adjusted_forecast = Forecast(
            question_id=forecast.question_id,
            prediction=adjusted_prediction,
            confidence=adjusted_confidence,
            reasoning_trace=forecast.reasoning_trace + [
                ReasoningStep(
                    step_number=len(forecast.reasoning_trace) + 1,
                    description="Risk management and calibration adjustment",
                    input_data={
                        'original_confidence': forecast.confidence.level,
                        'original_prediction': forecast.prediction
                    },
                    output_data={
                        'adjusted_confidence': adjusted_confidence_level,
                        'adjusted_prediction': adjusted_prediction,
                        'confidence_adjustment': confidence_adjustment,
                        'prediction_adjustment': prediction_adjustment,
                        'calibration_adjustment': calibration_adjustment
                    },
                    confidence=adjusted_confidence,
                    timestamp=datetime.utcnow(),
                    reasoning_type="risk_management"
                )
            ],
            evidence_sources=forecast.evidence_sources,
            timestamp=datetime.utcnow(),
            agent_id=forecast.agent_id,
            metadata={
                **forecast.metadata,
                'risk_adjustments': {
                    'confidence_adjustment': confidence_adjustment,
                    'prediction_adjustment': prediction_adjustment,
                    'calibration_adjustment': calibration_adjustment
                },
                'original_forecast': {
                    'prediction': forecast.prediction,
                    'confidence': forecast.confidence.level
                }
            }
        )

        logger.info(
            "Risk management applied",
            extra={
                'question_id': question.id,
                'original_confidence': forecast.confidence.level,
                'adjusted_confidence': adjusted_confidence_level,
                'confidence_adjustment': confidence_adjustment,
                'prediction_adjustment': prediction_adjustment,
                'calibration_adjustment': calibration_adjustment
            }
        )

        return risk_adjusted_forecast

    async def submit_forecast(self,
                            question: Question,
                            forecast: Forecast,
                            tournament_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Submit forecast to tournament platform.

        Args:
            question: Question being forecasted
            forecast: Final forecast to submit
            tournament_id: Optional tournament context

        Returns:
            Submission result dictionary
        """
        try:
            # Optimize submission timing if tournament context available
            if tournament_id:
                tournament = await self.tournament_service.get_tournament(tournament_id)
                if tournament:
                    optimal_time, timing_reason = self.tournament_service.optimize_submission_timing(
                        question, tournament
                    )

                    # Check if we should delay submission
                    if optimal_time > datetime.utcnow():
                        logger.info(
                            "Delaying submission for optimal timing",
                            extra={
                                'question_id': question.id,
                                'optimal_time': optimal_time.isoformat(),
                                'delay_seconds': (optimal_time - datetime.utcnow()).total_seconds(),
                                'timing_reason': timing_reason
                            }
                        )

                        # In a real implementation, this would schedule the submission
                        # For now, we'll submit immediately but log the optimization
                        forecast.metadata['optimal_submission_time'] = optimal_time.isoformat()
                        forecast.metadata['timing_reason'] = timing_reason

            # Submit through tournament service
            submission_result = await self.tournament_service.submit_forecast(
                question, forecast, tournament_id
            )

            logger.info(
                "Forecast submitted successfully",
                extra={
                    'question_id': question.id,
                    'tournament_id': tournament_id,
                    'submission_id': submission_result.get('submission_id') if submission_result else None
                }
            )

            return submission_result

        except Exception as e:
            logger.error(
                "Forecast submission failed",
                extra={
                    'question_id': question.id,
                    'tournament_id': tournament_id,
                    'error': str(e)
                },
                exc_info=True
            )
            raise

    # Helper methods for pipeline execution

    async def _execute_stage(self, stage: PipelineStage, func, *args, **kwargs):
        """Execute a pipeline stage with timing and error handling.

        Args:
            stage: Pipeline stage being executed
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        stage_start = datetime.utcnow()

        try:
            result = await func(*args, **kwargs)
            stage_time = (datetime.utcnow() - stage_start).total_seconds()
            self.metrics.stage_timings[stage] = stage_time

            logger.debug(
                f"Pipeline stage {stage.value} completed",
                extra={
                    'stage': stage.value,
                    'execution_time': stage_time
                }
            )

            return result

        except Exception as e:
            stage_time = (datetime.utcnow() - stage_start).total_seconds()
            self.metrics.stage_timings[stage] = stage_time
            self.metrics.errors_encountered.append(f"{stage.value}: {str(e)}")

            logger.error(
                f"Pipeline stage {stage.value} failed",
                extra={
                    'stage': stage.value,
                    'execution_time': stage_time,
                    'error': str(e)
                }
            )
            raise

    def _get_research_parameters(self, question: Question, strategy_type: str) -> Dict[str, Any]:
        """Get research parameters based on question and strategy.

        Args:
            question: Question to research
            strategy_type: Research strategy type

        Returns:
            Dictionary of research parameters
        """
        base_params = {
            'max_sources': 10,
            'max_time_minutes': self.config.max_research_time_minutes,
            'providers': self.config.research_providers.copy()
        }

        # Adjust based on strategy type
        if strategy_type == 'rapid_response':
            base_params['max_time_minutes'] = min(5, base_params['max_time_minutes'])
            base_params['max_sources'] = 5
        elif strategy_type == 'comprehensive_analysis':
            base_params['max_time_minutes'] = min(20, base_params['max_time_minutes'])
            base_params['max_sources'] = 15
        elif strategy_type == 'specialized_research':
            # Add specialized providers for technical questions
            if question.requires_specialized_knowledge():
                base_params['providers'].extend(['arxiv', 'pubmed'])

        # Adjust based on question characteristics
        if question.is_deadline_approaching(hours_threshold=6):
            base_params['max_time_minutes'] = min(3, base_params['max_time_minutes'])

        return base_params

    async def _execute_research_with_providers(self,
                                             question: Question,
                                             research_params: Dict[str, Any]) -> List[ResearchReport]:
        """Execute research using multiple providers.

        Args:
            question: Question to research
            research_params: Research parameters

        Returns:
            List of research reports
        """
        # Execute research with each provider concurrently
        semaphore = asyncio.Semaphore(len(research_params['providers']))

        async def research_with_provider(provider: str) -> Optional[ResearchReport]:
            async with semaphore:
                try:
                    return await self.search_client.conduct_research(
                        question,
                        provider=provider,
                        max_sources=research_params['max_sources'] // len(research_params['providers']),
                        timeout_minutes=research_params['max_time_minutes']
                    )
                except Exception as e:
                    logger.warning(
                        f"Research with provider {provider} failed",
                        extra={
                            'question_id': question.id,
                            'provider': provider,
                            'error': str(e)
                        }
                    )
                    return None

        # Execute all providers concurrently
        tasks = [
            research_with_provider(provider)
            for provider in research_params['providers']
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful research reports
        research_reports = [
            result for result in results
            if isinstance(result, ResearchReport) and result is not None
        ]

        # Ensure minimum number of sources
        if len(research_reports) < self.config.min_research_sources:
            logger.warning(
                "Insufficient research reports generated",
                extra={
                    'question_id': question.id,
                    'reports_generated': len(research_reports),
                    'minimum_required': self.config.min_research_sources
                }
            )

        return research_reports

    def _select_agents_for_question(self,
                                  question: Question,
                                  strategy_type: str) -> Dict[str, BaseAgent]:
        """Select appropriate agents for a question.

        Args:
            question: Question to select agents for
            strategy_type: Strategy type

        Returns:
            Dictionary of selected agents
        """
        selected_agents = {}

        # Always include chain-of-thought for baseline reasoning
        if 'cot' in self.agents and 'cot' in self.config.enabled_agents:
            selected_agents['cot'] = self.agents['cot']

        # Add tree-of-thought for complex questions
        if (question.get_complexity_score() > 1.5 and
            'tot' in self.agents and 'tot' in self.config.enabled_agents):
            selected_agents['tot'] = self.agents['tot']

        # Add ReAct for questions requiring iterative reasoning
        if (question.requires_specialized_knowledge() and
            'react' in self.agents and 'react' in self.config.enabled_agents):
            selected_agents['react'] = self.agents['react']

        # Limit to max agents
        if len(selected_agents) > self.config.max_agents:
            # Keep highest priority agents
            priority_order = ['cot', 'tot', 'react']
            selected_agents = {
                agent_id: agent for agent_id, agent in selected_agents.items()
                if agent_id in priority_order[:self.config.max_agents]
            }

        return selected_agents

    def _calculate_consensus_metrics(self, forecasts: List[Forecast]) -> ConsensusMetrics:
        """Calculate consensus metrics for ensemble forecasts.

        Args:
            forecasts: List of forecasts to analyze

        Returns:
            ConsensusMetrics object
        """
        if not forecasts:
            return ConsensusMetrics(
                consensus_strength=0.0,
                prediction_variance=1.0,
                agent_diversity_score=0.0,
                confidence_alignment=0.0
            )

        # Extract predictions and confidences
        predictions = []
        confidences = [f.confidence.level for f in forecasts]

        for forecast in forecasts:
            if isinstance(forecast.prediction, (int, float)):
                predictions.append(forecast.prediction)
            elif isinstance(forecast.prediction, dict):
                # For multiple choice, use entropy or most likely choice probability
                max_prob = max(forecast.prediction.values())
                predictions.append(max_prob)
            else:
                predictions.append(0.5)  # Default for unknown types

        # Calculate metrics
        import numpy as np

        prediction_variance = float(np.var(predictions)) if len(predictions) > 1 else 0.0
        confidence_variance = float(np.var(confidences)) if len(confidences) > 1 else 0.0

        # Consensus strength (inverse of prediction variance, normalized)
        consensus_strength = max(0.0, 1.0 - prediction_variance)

        # Agent diversity (based on prediction spread)
        agent_diversity = min(1.0, prediction_variance * 2.0)

        # Confidence alignment (inverse of confidence variance)
        confidence_alignment = max(0.0, 1.0 - confidence_variance)

        return ConsensusMetrics(
            consensus_strength=consensus_strength,
            prediction_variance=prediction_variance,
            agent_diversity_score=agent_diversity,
            confidence_alignment=confidence_alignment
        )

    def _select_aggregation_method(self,
                                 consensus_metrics: ConsensusMetrics,
                                 strategy_type: str) -> AggregationMethod:
        """Select appropriate aggregation method based on consensus and strategy.

        Args:
            consensus_metrics: Consensus metrics for the ensemble
            strategy_type: Strategy type

        Returns:
            Selected aggregation method
        """
        # High consensus - use simple average
        if consensus_metrics.consensus_strength > 0.8:
            return AggregationMethod.SIMPLE_AVERAGE

        # Low consensus but high confidence alignment - use confidence weighting
        elif (consensus_metrics.consensus_strength < 0.4 and
              consensus_metrics.confidence_alignment > 0.6):
            return AggregationMethod.CONFIDENCE_WEIGHTED

        # High diversity - use median to reduce outlier impact
        elif consensus_metrics.agent_diversity_score > 0.7:
            return AggregationMethod.MEDIAN

        # Default to configured method
        else:
            return self.config.aggregation_method

    def _calculate_confidence_adjustment(self,
                                       question: Question,
                                       forecast: Forecast) -> float:
        """Calculate confidence adjustment factor.

        Args:
            question: Question being forecasted
            forecast: Forecast to adjust

        Returns:
            Confidence adjustment factor (0.0 to 1.0)
        """
        adjustment = 1.0

        # Reduce confidence for complex questions
        complexity_penalty = min(0.2, question.get_complexity_score() * 0.05)
        adjustment -= complexity_penalty

        # Reduce confidence for specialized knowledge questions
        if question.requires_specialized_knowledge():
            adjustment -= 0.1

        # Reduce confidence for approaching deadlines (rushed analysis)
        if question.is_deadline_approaching(hours_threshold=6):
            adjustment -= 0.15

        # Apply base adjustment factor from config
        adjustment *= self.config.confidence_adjustment_factor

        return max(0.1, adjustment)

    def _calculate_prediction_adjustment(self,
                                       question: Question,
                                       forecast: Forecast) -> float:
        """Calculate prediction adjustment for extreme values.

        Args:
            question: Question being forecasted
            forecast: Forecast to adjust

        Returns:
            Prediction adjustment factor
        """
        if not isinstance(forecast.prediction, (int, float)):
            return 0.0  # No adjustment for non-numeric predictions

        prediction = forecast.prediction

        # Check for extreme predictions
        if question.is_binary():
            if prediction < self.config.extreme_prediction_threshold:
                return 0.05 - prediction  # Pull toward 5%
            elif prediction > (1.0 - self.config.extreme_prediction_threshold):
                return (0.95 - prediction)  # Pull toward 95%

        return 0.0  # No adjustment needed

    async def _get_calibration_adjustment(self,
                                        question: Question,
                                        forecast: Forecast) -> float:
        """Get calibration adjustment based on historical performance.

        Args:
            question: Question being forecasted
            forecast: Forecast to calibrate

        Returns:
            Calibration adjustment factor
        """
        try:
            # Get historical calibration data from learning service
            calibration_data = await self.learning_service.get_calibration_data(
                question_category=question.category,
                question_type=question.question_type,
                agent_id=forecast.agent_id
            )

            if calibration_data and calibration_data.get('adjustment_factor'):
                return calibration_data['adjustment_factor']

        except Exception as e:
            logger.warning(
                "Failed to get calibration adjustment",
                extra={
                    'question_id': question.id,
                    'error': str(e)
                }
            )

        return 1.0  # No adjustment if calibration data unavailable

    def _apply_prediction_adjustment(self,
                                   prediction: Union[float, Dict[str, float], List[float]],
                                   adjustment: float) -> Union[float, Dict[str, float], List[float]]:
        """Apply prediction adjustment.

        Args:
            prediction: Original prediction
            adjustment: Adjustment to apply

        Returns:
            Adjusted prediction
        """
        if adjustment == 0.0:
            return prediction

        if isinstance(prediction, (int, float)):
            return max(0.0, min(1.0, prediction + adjustment))
        elif isinstance(prediction, dict):
            # For multiple choice, adjust proportionally
            total_adjustment = sum(prediction.values()) + adjustment
            if total_adjustment > 0:
                return {
                    choice: (prob + adjustment / len(prediction)) / total_adjustment
                    for choice, prob in prediction.items()
                }
        elif isinstance(prediction, list):
            # For list predictions, adjust each element
            return [max(0.0, min(1.0, p + adjustment)) for p in prediction]

        return prediction
