"""
Date Question Forecasting Service

Implements forecasting logic for date-type questions on Metaculus.
Date questions ask "When will X happen?" and require predicting a probability distribution over dates.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DateForecast:
    """Result of a date question forecast."""
    percentiles: Dict[float, datetime]  # percentile -> date mapping
    reasoning: str
    confidence: float


class DateQuestionForecaster:
    """Handles forecasting for date-type questions."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def forecast_date_question(
        self,
        question_text: str,
        background_info: str,
        resolution_criteria: str,
        lower_bound: datetime,
        upper_bound: datetime,
        research_data: str = "",
        fine_print: str = ""
    ) -> DateForecast:
        """
        Generate a forecast for a date question.

        Args:
            question_text: The main question text
            background_info: Question background/context
            resolution_criteria: How the question resolves
            lower_bound: Earliest possible date
            upper_bound: Latest possible date
            research_data: Research findings
            fine_print: Additional question details

        Returns:
            DateForecast with percentiles and reasoning
        """
        self.logger.info(f"Forecasting date question: {question_text[:60]}...")

        try:
            # Analyze the date range
            date_range_days = (upper_bound - lower_bound).days
            self.logger.info(f"Date range: {lower_bound.date()} to {upper_bound.date()} ({date_range_days} days)")

            # Generate reasoning-based forecast
            reasoning = self._generate_date_reasoning(
                question_text, background_info, resolution_criteria,
                lower_bound, upper_bound, research_data, fine_print
            )

            # Extract date estimates from reasoning
            percentiles = self._extract_date_percentiles(
                reasoning, lower_bound, upper_bound
            )

            # Calculate confidence based on reasoning quality
            confidence = self._assess_forecast_confidence(reasoning, date_range_days)

            return DateForecast(
                percentiles=percentiles,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            self.logger.error(f"Error forecasting date question: {e}")
            # Return conservative default forecast
            return self._create_default_forecast(lower_bound, upper_bound)

    def _generate_date_reasoning(
        self,
        question_text: str,
        background_info: str,
        resolution_criteria: str,
        lower_bound: datetime,
        upper_bound: datetime,
        research_data: str,
        fine_print: str
    ) -> str:
        """Generate reasoning for the date forecast using structured analysis."""

        # For now, create structured reasoning that can be enhanced with LLM calls later
        reasoning_parts = []

        reasoning_parts.append("Date Question Analysis:")
        reasoning_parts.append(f"Question: {question_text}")
        reasoning_parts.append(f"Date Range: {lower_bound.date()} to {upper_bound.date()}")

        if background_info:
            reasoning_parts.append(f"Background: {background_info[:200]}...")

        if research_data:
            reasoning_parts.append(f"Research: {research_data[:200]}...")

        # Analyze timeframe
        total_days = (upper_bound - lower_bound).days
        if total_days < 365:
            timeframe_analysis = "Short timeframe (less than 1 year) - events likely to be driven by immediate factors."
        elif total_days < 1095:  # 3 years
            timeframe_analysis = "Medium timeframe (1-3 years) - allows for moderate planning and development cycles."
        else:
            timeframe_analysis = "Long timeframe (3+ years) - subject to significant uncertainty and changing conditions."

        reasoning_parts.append(f"Timeframe Analysis: {timeframe_analysis}")

        # Default forecasting approach for date questions
        reasoning_parts.append("\nForecasting Approach:")
        reasoning_parts.append("- Earlier dates receive higher probability (typical pattern for uncertain events)")
        reasoning_parts.append("- Distribution favors first third of time range unless specific reasons suggest otherwise")
        reasoning_parts.append("- Maintaining uncertainty across the full range to account for unknown factors")

        # Add specific analysis based on question content
        if "leave" in question_text.lower() or "depart" in question_text.lower():
            reasoning_parts.append("- Political/administrative departures often occur during transition periods")
            reasoning_parts.append("- Higher probability in first year, moderate in years 2-3, lower for full term completion")

        reasoning_parts.append(f"\nDate Estimates:")

        # Calculate reasonable percentile estimates
        # For most "when will X happen" questions, we expect earlier dates to be more likely
        range_start = lower_bound
        range_end = upper_bound

        # Create a distribution that favors earlier dates
        p10_date = range_start + (range_end - range_start) * 0.15
        p25_date = range_start + (range_end - range_start) * 0.30
        p50_date = range_start + (range_end - range_start) * 0.45
        p75_date = range_start + (range_end - range_start) * 0.65
        p90_date = range_start + (range_end - range_start) * 0.85

        reasoning_parts.append(f"10th percentile (early scenario): {p10_date.strftime('%Y-%m-%d')}")
        reasoning_parts.append(f"25th percentile: {p25_date.strftime('%Y-%m-%d')}")
        reasoning_parts.append(f"50th percentile (median): {p50_date.strftime('%Y-%m-%d')}")
        reasoning_parts.append(f"75th percentile: {p75_date.strftime('%Y-%m-%d')}")
        reasoning_parts.append(f"90th percentile (late scenario): {p90_date.strftime('%Y-%m-%d')}")

        return "\n".join(reasoning_parts)

    def _extract_date_percentiles(
        self,
        reasoning: str,
        lower_bound: datetime,
        upper_bound: datetime
    ) -> Dict[float, datetime]:
        """Extract date percentiles from reasoning text."""

        percentiles = {}

        try:
            # For now, use the calculated percentiles from reasoning generation
            # This could be enhanced to parse LLM-generated reasoning later

            range_start = lower_bound
            range_end = upper_bound

            # Create distribution that favors earlier dates (common for "when will X happen")
            percentiles[0.1] = range_start + (range_end - range_start) * 0.15
            percentiles[0.25] = range_start + (range_end - range_start) * 0.30
            percentiles[0.5] = range_start + (range_end - range_start) * 0.45
            percentiles[0.75] = range_start + (range_end - range_start) * 0.65
            percentiles[0.9] = range_start + (range_end - range_start) * 0.85

            self.logger.info(f"Generated date percentiles from {range_start.date()} to {range_end.date()}")

        except Exception as e:
            self.logger.error(f"Error extracting date percentiles: {e}")
            # Fallback to uniform distribution
            percentiles = self._create_uniform_date_distribution(lower_bound, upper_bound)

        return percentiles

    def _assess_forecast_confidence(self, reasoning: str, date_range_days: int) -> float:
        """Assess confidence in the date forecast based on reasoning quality and timeframe."""

        base_confidence = 0.6  # Moderate confidence for date questions

        # Adjust based on timeframe
        if date_range_days < 365:
            timeframe_adjustment = 0.1  # Higher confidence for shorter timeframes
        elif date_range_days > 2190:  # 6+ years
            timeframe_adjustment = -0.15  # Lower confidence for very long timeframes
        else:
            timeframe_adjustment = 0.0

        # Adjust based on reasoning content quality
        quality_indicators = [
            "research", "analysis", "historical", "pattern", "trend", "evidence"
        ]
        quality_score = sum(1 for indicator in quality_indicators if indicator in reasoning.lower())
        quality_adjustment = min(quality_score * 0.05, 0.2)  # Up to 0.2 bonus

        final_confidence = base_confidence + timeframe_adjustment + quality_adjustment
        return max(0.3, min(0.9, final_confidence))  # Clamp between 0.3 and 0.9

    def _create_uniform_date_distribution(
        self,
        lower_bound: datetime,
        upper_bound: datetime
    ) -> Dict[float, datetime]:
        """Create a uniform distribution across the date range."""

        percentiles = {}
        total_duration = upper_bound - lower_bound

        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            percentiles[p] = lower_bound + total_duration * p

        return percentiles

    def _create_default_forecast(
        self,
        lower_bound: datetime,
        upper_bound: datetime
    ) -> DateForecast:
        """Create a conservative default forecast when analysis fails."""

        percentiles = self._create_uniform_date_distribution(lower_bound, upper_bound)

        reasoning = f"""
        Default Date Forecast (Analysis Limited):
        - Using uniform distribution across date range
        - Range: {lower_bound.date()} to {upper_bound.date()}
        - This is a conservative approach given limited analysis capability
        - Median estimate: {percentiles[0.5].strftime('%Y-%m-%d')}
        """

        return DateForecast(
            percentiles=percentiles,
            reasoning=reasoning.strip(),
            confidence=0.3
        )

    def format_percentiles_for_metaculus(self, percentiles: Dict[float, datetime]) -> List[Tuple[float, str]]:
        """Format percentiles for Metaculus API submission."""

        formatted = []
        for percentile in sorted(percentiles.keys()):
            date_obj = percentiles[percentile]
            # Metaculus expects ISO format dates
            date_str = date_obj.strftime('%Y-%m-%d')
            formatted.append((percentile, date_str))

        return formatted
