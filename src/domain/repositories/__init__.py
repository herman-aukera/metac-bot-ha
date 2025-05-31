"""Repository interfaces for the forecasting domain."""

from .question_repository import QuestionRepository
from .forecast_repository import ForecastRepository
from .research_repository import ResearchRepository

__all__ = [
    "QuestionRepository",
    "ForecastRepository", 
    "ResearchRepository",
]
