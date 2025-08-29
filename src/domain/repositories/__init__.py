"""Repository interfaces for the forecasting domain."""

from .forecast_repository import ForecastRepository
from .question_repository import QuestionRepository
from .research_repository import ResearchRepository

__all__ = [
    "QuestionRepository",
    "ForecastRepository",
    "ResearchRepository",
]
