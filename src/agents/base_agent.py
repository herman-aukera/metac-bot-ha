"""Base agent class for forecasting."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import structlog

from ..domain.entities.forecast import Forecast, ForecastStatus
from ..domain.entities.prediction import Prediction
from ..domain.entities.question import Question
from ..domain.entities.research_report import ResearchReport

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all forecasting agents.

    Defines the common interface that all agents must implement
    for research and prediction generation.
    """

    def __init__(self, name: str, model_config: Dict[str, Any]):
        self.name = name
        self.model_config = model_config
        self.logger = logger.bind(agent=name)

    @abstractmethod
    async def conduct_research(
        self, question: Question, search_config: Optional[Dict[str, Any]] = None
    ) -> ResearchReport:
        """
        Conduct research for a given question.

        Args:
            question: The question to research
            search_config: Optional configuration for search behavior

        Returns:
            Research report with findings and analysis
        """
        pass

    @abstractmethod
    async def generate_prediction(
        self, question: Question, research_report: ResearchReport
    ) -> Prediction:
        """
        Generate a prediction based on research.

        Args:
            question: The question to predict
            research_report: Research findings to base prediction on

        Returns:
            Prediction with reasoning and confidence
        """
        pass

    async def full_forecast_cycle(
        self, question: Question, search_config: Optional[Dict[str, Any]] = None
    ) -> tuple[ResearchReport, Prediction]:
        """
        Complete forecasting cycle: research + prediction.

        Args:
            question: Question to forecast
            search_config: Optional search configuration

        Returns:
            Tuple of (research_report, prediction)
        """
        self.logger.info("Starting full forecast cycle", question_id=str(question.id))

        try:
            # Conduct research
            research_report = await self.conduct_research(question, search_config)
            self.logger.info("Research completed", sources=len(research_report.sources))

            # Generate prediction
            prediction = await self.generate_prediction(question, research_report)
            self.logger.info("Prediction generated", method=prediction.method.value)

            return research_report, prediction

        except Exception as e:
            self.logger.error("Forecast cycle failed", error=str(e))
            raise

    async def forecast(
        self, question: Question, search_config: Optional[Dict[str, Any]] = None
    ) -> Forecast:
        """
        Generate a complete forecast for a question.

        Args:
            question: Question to forecast
            search_config: Optional search configuration

        Returns:
            Complete forecast with research and prediction
        """
        self.logger.info("Starting forecast", question_id=str(question.id))

        try:
            # Conduct research and generate prediction
            research_report, prediction = await self.full_forecast_cycle(
                question, search_config
            )

            # Create forecast object
            forecast = Forecast.create_new(
                question_id=question.id,
                research_reports=[research_report],
                predictions=[prediction],
                final_prediction=prediction,
                ensemble_method="single_agent",
                weight_distribution={self.name: 1.0},
                consensus_strength=1.0,
                reasoning_summary=prediction.reasoning,
            )

            self.logger.info("Forecast completed", forecast_id=str(forecast.id))
            return forecast

        except Exception as e:
            self.logger.error("Forecast failed", error=str(e))
            raise

    def get_agent_metadata(self) -> Dict[str, Any]:
        """Get metadata about this agent."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "model_config": self.model_config,
        }
