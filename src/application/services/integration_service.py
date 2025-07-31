"""Integration Service for backward compatibility with existing main entry points.

This service provides integration between the new tournament optimization system
and the existing main.py and main_agent.py entry points, ensuring backward
compatibility while enabling the new advanced features.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from ...domain.entities.question import Question, QuestionType, QuestionCategory
from ...domain.entities.forecast import Forecast
from ...domain.value_objects.confidence import Confidence
from ...domain.value_objects.reasoning_step import ReasoningStep
from ..use_cases.process_tournament_question import ProcessTournamentQuestionUseCase
from .forecasting_pipeline import ForecastingPipeline
from .tournament_service import TournamentService
from ...infrastructure.logging.structured_logger import StructuredLogger


logger = StructuredLogger(__name__)


@dataclass
class LegacyQuestionFormat:
    """Legacy question format for backward compatibility."""
    question_id: int
    question_text: str
    question_type: str = 'binary'
    background_info: str = ''
    resolution_criteria: str = ''
    fine_print: str = ''
    options: Optional[List[str]] = None
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    open_upper_bound: bool = False
    open_lower_bound: bool = False
    unit_of_measure: Optional[str] = None
    page_url: str = ''


@dataclass
class LegacyForecastResult:
    """Legacy forecast result format for backward compatibility."""
    question_id: int
    forecast: Union[float, List[float], Dict[str, float]]
    justification: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    trace: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class IntegrationService:
    """Service for integrating with existing main entry points.

    This service provides backward compatibility with the existing main.py and
    main_agent.py interfaces while leveraging the new tournament optimization
    capabilities under the hood.
    """

    def __init__(self,
                 process_question_use_case: ProcessTournamentQuestionUseCase,
                 forecasting_pipeline: ForecastingPipeline,
                 tournament_service: TournamentService):
        """Initialize integration service.

        Args:
            process_question_use_case: Main question processing use case
            forecasting_pipeline: Forecasting pipeline service
            tournament_service: Tournament strategy service
        """
        self.process_question_use_case = process_question_use_case
        self.forecasting_pipeline = forecasting_pipeline
        self.tournament_service = tournament_service

    async def process_legacy_question(self,
                                    legacy_question: LegacyQuestionFormat,
                                    tournament_id: Optional[int] = None,
                                    use_advanced_features: bool = True) -> LegacyForecastResult:
        """Process a question in legacy format.

        Args:
            legacy_question: Question in legacy format
            tournament_id: Optional tournament context
            use_advanced_features: Whether to use advanced optimization features

        Returns:
            Forecast result in legacy format
        """
        try:
            # Convert legacy question to new format
            question = self._convert_legacy_question(legacy_question)

            if use_advanced_features:
                # Use full tournament optimization pipeline
                result = await self.process_question_use_case.execute(
                    question=question,
                    tournament_id=tournament_id,
                    submission_mode=False  # Don't submit in legacy mode
                )

                if result.success and result.final_forecast:
                    return self._convert_to_legacy_result(
                        result.final_forecast,
                        legacy_question.question_id,
                        result.metadata
                    )
                else:
                    # Fallback to basic processing
                    logger.warning(
                        "Advanced processing failed, falling back to basic mode",
                        extra={
                            'question_id': legacy_question.question_id,
                            'error': result.error_message
                        }
                    )
                    return await self._process_basic_question(legacy_question)
            else:
                # Use basic processing for backward compatibility
                return await self._process_basic_question(legacy_question)

        except Exception as e:
            logger.error(
                "Legacy question processing failed",
                extra={
                    'question_id': legacy_question.question_id,
                    'error': str(e)
                },
                exc_info=True
            )

            # Return error result in legacy format
            return LegacyForecastResult(
                question_id=legacy_question.question_id,
                forecast=0.5 if legacy_question.question_type == 'binary' else 0.0,
                justification=f"Processing failed: {str(e)}",
                confidence=0.1,
                metadata={'error': str(e), 'fallback_used': True}
            )

    async def process_tournament_legacy(self,
                                      tournament_id: int,
                                      max_questions: Optional[int] = None,
                                      use_advanced_features: bool = True) -> List[LegacyForecastResult]:
        """Process tournament questions in legacy format.

        Args:
            tournament_id: Tournament ID to process
            max_questions: Maximum number of questions to process
            use_advanced_features: Whether to use advanced optimization features

        Returns:
            List of forecast results in legacy format
        """
        try:
            if use_advanced_features:
                # Use tournament optimization
                tournament = await self.tournament_service.get_tournament(tournament_id)
                if not tournament:
                    raise ValueError(f"Tournament {tournament_id} not found")

                active_questions = tournament.get_active_questions()
                if max_questions:
                    active_questions = active_questions[:max_questions]

                # Process questions with tournament optimization
                results = []
                for question in active_questions:
                    result = await self.process_question_use_case.execute(
                        question=question,
                        tournament_id=tournament_id,
                        submission_mode=False
                    )

                    if result.success and result.final_forecast:
                        legacy_result = self._convert_to_legacy_result(
                            result.final_forecast,
                            question.id,
                            result.metadata
                        )
                    else:
                        # Create fallback result
                        legacy_result = LegacyForecastResult(
                            question_id=question.id,
                            forecast=0.5 if question.is_binary() else 0.0,
                            justification=f"Processing failed: {result.error_message}",
                            confidence=0.1,
                            metadata={'error': result.error_message, 'fallback_used': True}
                        )

                    results.append(legacy_result)

                return results
            else:
                # Use basic processing
                return await self._process_tournament_basic(tournament_id, max_questions)

        except Exception as e:
            logger.error(
                "Legacy tournament processing failed",
                extra={
                    'tournament_id': tournament_id,
                    'error': str(e)
                },
                exc_info=True
            )
            return []

    async def get_legacy_agent_result(self,
                                    question_dict: Dict[str, Any],
                                    agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get agent result in legacy format for main_agent.py compatibility.

        Args:
            question_dict: Question dictionary in legacy format
            agent_config: Optional agent configuration

        Returns:
            Agent result dictionary in legacy format
        """
        try:
            # Convert question dictionary to legacy format
            legacy_question = LegacyQuestionFormat(
                question_id=question_dict.get('question_id', 1),
                question_text=question_dict.get('question_text', ''),
                question_type=question_dict.get('type', 'binary'),
                background_info=question_dict.get('background', ''),
                resolution_criteria=question_dict.get('resolution_criteria', ''),
                options=question_dict.get('options'),
                page_url=question_dict.get('page_url', '')
            )

            # Process question
            use_advanced = agent_config.get('use_advanced_features', True) if agent_config else True
            result = await self.process_legacy_question(
                legacy_question,
                use_advanced_features=use_advanced
            )

            # Convert to main_agent.py format
            agent_result = {
                'question_id': result.question_id,
                'forecast': result.forecast,
                'justification': result.justification,
                'prediction': result.forecast,  # Alias for backward compatibility
                'reasoning': result.reasoning or result.justification,
                'confidence': result.confidence,
                'trace': result.trace or [],
                'metadata': result.metadata or {}
            }

            # Add trace information if available
            if result.trace:
                agent_result['trace'] = result.trace
            else:
                # Create basic trace for compatibility
                agent_result['trace'] = [
                    {
                        'type': 'reasoning',
                        'input': {'question': question_dict},
                        'output': {'forecast': result.forecast, 'justification': result.justification},
                        'timestamp': datetime.utcnow().isoformat()
                    }
                ]

            return agent_result

        except Exception as e:
            logger.error(
                "Legacy agent result generation failed",
                extra={
                    'question_id': question_dict.get('question_id'),
                    'error': str(e)
                },
                exc_info=True
            )

            # Return error result in legacy format
            return {
                'question_id': question_dict.get('question_id', 1),
                'forecast': 0.5,
                'justification': f"Processing failed: {str(e)}",
                'prediction': 0.5,
                'reasoning': f"Error occurred during processing: {str(e)}",
                'confidence': 0.1,
                'trace': [],
                'metadata': {'error': str(e)}
            }

    # Helper methods

    def _convert_legacy_question(self, legacy_question: LegacyQuestionFormat) -> Question:
        """Convert legacy question format to new Question entity.

        Args:
            legacy_question: Question in legacy format

        Returns:
            Question entity
        """
        # Determine question type
        if legacy_question.question_type.lower() in ['binary', 'bool', 'boolean']:
            question_type = QuestionType.BINARY
        elif legacy_question.question_type.lower() in ['numeric', 'number', 'continuous']:
            question_type = QuestionType.NUMERIC
        elif legacy_question.question_type.lower() in ['mc', 'multiple_choice', 'categorical']:
            question_type = QuestionType.MULTIPLE_CHOICE
        else:
            question_type = QuestionType.BINARY  # Default fallback

        # Determine category based on question text (simple heuristic)
        category = self._infer_question_category(legacy_question.question_text)

        # Set deadline (default to end of day if not specified)
        deadline = datetime.utcnow().replace(hour=23, minute=59, second=59)

        return Question(
            id=legacy_question.question_id,
            text=legacy_question.question_text,
            question_type=question_type,
            category=category,
            deadline=deadline,
            background=legacy_question.background_info or "No background provided",
            resolution_criteria=legacy_question.resolution_criteria or "No criteria provided",
            scoring_weight=1.0,
            metadata={
                'legacy_format': True,
                'page_url': legacy_question.page_url,
                'fine_print': legacy_question.fine_print,
                'unit_of_measure': legacy_question.unit_of_measure
            },
            min_value=legacy_question.lower_bound,
            max_value=legacy_question.upper_bound,
            choices=legacy_question.options
        )

    def _infer_question_category(self, question_text: str) -> QuestionCategory:
        """Infer question category from text using simple heuristics.

        Args:
            question_text: Question text to analyze

        Returns:
            Inferred question category
        """
        text_lower = question_text.lower()

        # AI/Technology keywords
        if any(keyword in text_lower for keyword in [
            'ai', 'artificial intelligence', 'machine learning', 'neural network',
            'deep learning', 'agi', 'algorithm', 'robot', 'automation'
        ]):
            return QuestionCategory.AI_DEVELOPMENT

        # Politics keywords
        elif any(keyword in text_lower for keyword in [
            'election', 'president', 'congress', 'senate', 'vote', 'political',
            'government', 'policy', 'legislation'
        ]):
            return QuestionCategory.POLITICS

        # Economics keywords
        elif any(keyword in text_lower for keyword in [
            'economy', 'gdp', 'inflation', 'market', 'stock', 'price',
            'economic', 'recession', 'growth', 'unemployment'
        ]):
            return QuestionCategory.ECONOMICS

        # Science keywords
        elif any(keyword in text_lower for keyword in [
            'research', 'study', 'experiment', 'discovery', 'scientific',
            'medicine', 'drug', 'treatment', 'vaccine'
        ]):
            return QuestionCategory.SCIENCE

        # Climate keywords
        elif any(keyword in text_lower for keyword in [
            'climate', 'temperature', 'warming', 'carbon', 'emission',
            'renewable', 'energy', 'environment'
        ]):
            return QuestionCategory.CLIMATE

        # Default to OTHER
        else:
            return QuestionCategory.OTHER

    def _convert_to_legacy_result(self,
                                forecast: Forecast,
                                question_id: int,
                                metadata: Dict[str, Any]) -> LegacyForecastResult:
        """Convert new Forecast to legacy result format.

        Args:
            forecast: Forecast entity
            question_id: Question ID
            metadata: Processing metadata

        Returns:
            Legacy forecast result
        """
        # Extract justification from reasoning trace
        justification = ""
        if forecast.reasoning_trace:
            justification_parts = []
            for step in forecast.reasoning_trace:
                if step.description and step.description != "Risk management and calibration adjustment":
                    justification_parts.append(f"{step.description}")
            justification = " -> ".join(justification_parts)

        if not justification:
            justification = forecast.confidence.basis or "Forecast generated using advanced reasoning"

        # Create trace information
        trace = []
        for i, step in enumerate(forecast.reasoning_trace):
            trace.append({
                'type': 'reasoning_step',
                'step_number': step.step_number,
                'description': step.description,
                'confidence': step.confidence.level,
                'timestamp': step.timestamp.isoformat(),
                'input': step.input_data,
                'output': step.output_data
            })

        return LegacyForecastResult(
            question_id=question_id,
            forecast=forecast.prediction,
            justification=justification,
            confidence=forecast.confidence.level,
            reasoning=justification,
            trace=trace,
            metadata={
                **metadata,
                'agent_id': forecast.agent_id,
                'evidence_sources': len(forecast.evidence_sources),
                'processing_timestamp': forecast.timestamp.isoformat(),
                'is_final': forecast.is_final
            }
        )

    async def _process_basic_question(self, legacy_question: LegacyQuestionFormat) -> LegacyForecastResult:
        """Process question using basic pipeline without advanced features.

        Args:
            legacy_question: Question in legacy format

        Returns:
            Basic forecast result
        """
        # Simple fallback processing
        if legacy_question.question_type.lower() == 'binary':
            # Simple binary prediction
            forecast_value = 0.5  # Neutral prediction
            justification = f"Basic binary forecast for: {legacy_question.question_text}"
        elif legacy_question.question_type.lower() == 'numeric':
            # Simple numeric prediction
            if legacy_question.lower_bound is not None and legacy_question.upper_bound is not None:
                forecast_value = (legacy_question.lower_bound + legacy_question.upper_bound) / 2
            else:
                forecast_value = 0.0
            justification = f"Basic numeric forecast for: {legacy_question.question_text}"
        elif legacy_question.options:
            # Simple multiple choice - equal probabilities
            num_options = len(legacy_question.options)
            forecast_value = {option: 1.0 / num_options for option in legacy_question.options}
            justification = f"Basic multiple choice forecast with equal probabilities"
        else:
            # Default fallback
            forecast_value = 0.5
            justification = "Basic fallback forecast"

        return LegacyForecastResult(
            question_id=legacy_question.question_id,
            forecast=forecast_value,
            justification=justification,
            confidence=0.5,
            reasoning=justification,
            metadata={'basic_processing': True, 'fallback_used': True}
        )

    async def _process_tournament_basic(self,
                                      tournament_id: int,
                                      max_questions: Optional[int]) -> List[LegacyForecastResult]:
        """Process tournament using basic pipeline.

        Args:
            tournament_id: Tournament ID
            max_questions: Maximum questions to process

        Returns:
            List of basic forecast results
        """
        # In a real implementation, this would fetch actual tournament questions
        # For now, return empty list as fallback
        logger.warning(
            "Basic tournament processing not fully implemented",
            extra={'tournament_id': tournament_id}
        )
        return []
