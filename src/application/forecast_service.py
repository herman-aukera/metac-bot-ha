"""ForecastService application layer for managing forecasts."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4, UUID

from src.domain.entities.question import Question, QuestionType
from src.domain.entities.forecast import Forecast, ForecastStatus, calculate_brier_score
from src.domain.entities.research_report import ResearchReport
from src.domain.value_objects.probability import Probability
from src.domain.value_objects.confidence import ConfidenceLevel


class ForecastValidationError(Exception):
    """Exception raised when forecast validation fails."""
    pass


class ForecastService:
    """Application service for managing forecasts."""

    def validate_forecast(
        self,
        question: Question,
        probability: Probability,
        confidence: ConfidenceLevel,
        reasoning: str
    ) -> None:
        """
        Validate a forecast before creation.
        
        Args:
            question: The question being forecasted
            probability: The probability estimate
            confidence: The confidence level
            reasoning: The reasoning behind the forecast
            
        Raises:
            ForecastValidationError: If validation fails
        """
        # Check if question is open
        if not question.is_open():
            raise ForecastValidationError("Question is closed and cannot accept new forecasts")
        
        # Only support binary questions for now
        if question.question_type != QuestionType.BINARY:
            raise ForecastValidationError("Only binary questions are supported for forecasting")
        
        # Validate reasoning
        if not reasoning or reasoning.strip() == "":
            raise ForecastValidationError("Reasoning cannot be empty")
        
        # Check for extreme probability values (discourage overconfidence)
        if self._is_extreme_probability(probability):
            raise ForecastValidationError(
                "Extreme probability values (< 0.05 or > 0.95) are discouraged. "
                "Please reconsider your confidence level."
            )

    def create_forecast(
        self,
        question: Question,
        forecaster_id: UUID,
        probability: Probability,
        confidence: ConfidenceLevel,
        reasoning: str
    ) -> Forecast:
        """
        Create a new forecast after validation.
        
        Args:
            question: The question being forecasted
            forecaster_id: ID of the forecaster
            probability: The probability estimate
            confidence: The confidence level
            reasoning: The reasoning behind the forecast
            
        Returns:
            The created forecast
            
        Raises:
            ForecastValidationError: If validation fails
        """
        # Validate the forecast
        self.validate_forecast(question, probability, confidence, reasoning)
        
        # Create using the new generate_forecast method which uses proper domain structure
        # This is a simplified wrapper that creates a basic forecast
        # For more sophisticated forecasts, use generate_forecast instead
        from src.domain.entities.prediction import (
            Prediction, PredictionResult, PredictionConfidence, PredictionMethod
        )
        
        # Convert confidence level to prediction confidence
        confidence_mapping = {
            0.0: PredictionConfidence.VERY_LOW,
            0.25: PredictionConfidence.LOW,
            0.5: PredictionConfidence.MEDIUM,
            0.75: PredictionConfidence.HIGH,
            1.0: PredictionConfidence.VERY_HIGH,
        }
        # Find closest confidence level
        closest_conf = min(confidence_mapping.keys(), key=lambda x: abs(x - confidence.value))
        pred_confidence = confidence_mapping[closest_conf]
        
        # Create a simple research report
        research_report = self._create_mock_research_report(question, probability.value)
        
        # Create prediction
        prediction = Prediction.create_binary_prediction(
            question_id=question.id,
            research_report_id=research_report.id,
            probability=probability.value,
            confidence=pred_confidence,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning=reasoning,
            created_by=str(forecaster_id)
        )
        
        # Create forecast using the factory method
        return Forecast.create_new(
            question_id=question.id,
            research_reports=[research_report],
            predictions=[prediction],
            final_prediction=prediction,
            reasoning_summary=reasoning
        )

    def score_forecast(self, forecast: Forecast, question: Question) -> Optional[float]:
        """
        Score a forecast against the actual outcome.
        
        Args:
            forecast: The forecast to score
            question: The question with resolution
            
        Returns:
            Brier score (lower is better) or None if question is not resolved
        """
        if not question.is_resolved():
            return None
        
        # For binary questions, need to get the actual outcome
        # This is a placeholder - need to understand how Question stores resolution
        outcome = None  # TODO: Get actual outcome from resolved question
        
        if outcome is None:
            return None
        
        # Use the final prediction's binary probability for scoring
        final_prob = forecast.final_prediction.result.binary_probability
        if final_prob is None:
            return None
        
        # Use the existing calculate_brier_score function
        return calculate_brier_score(
            forecast=final_prob,
            outcome=1 if outcome else 0
        )

    def batch_score_forecasts(
        self, 
        forecasts: List[Forecast], 
        questions: List[Question]
    ) -> List[Optional[float]]:
        """
        Score multiple forecasts in batch.
        
        Args:
            forecasts: List of forecasts to score
            questions: List of corresponding questions
            
        Returns:
            List of scores (or None for unresolved questions)
        """
        # Create a mapping of question_id to question for efficient lookup
        question_map = {q.id: q for q in questions}
        
        scores = []
        for forecast in forecasts:
            question = question_map.get(forecast.question_id)
            if question:
                score = self.score_forecast(forecast, question)
                scores.append(score)
            else:
                scores.append(None)
        
        return scores

    def calculate_average_score(self, scores: List[Optional[float]]) -> Optional[float]:
        """
        Calculate the average score from a list of scores.
        
        Args:
            scores: List of scores (may contain None values)
            
        Returns:
            Average score or None if no valid scores
        """
        valid_scores = [score for score in scores if score is not None]
        
        if not valid_scores:
            return None
        
        return sum(valid_scores) / len(valid_scores)

    def get_forecast_summary(
        self, 
        forecasts: List[Forecast], 
        questions: List[Question]
    ) -> Dict[str, Any]:
        """
        Generate a summary of forecasts and their performance.
        
        Args:
            forecasts: List of forecasts
            questions: List of corresponding questions
            
        Returns:
            Dictionary containing summary statistics
        """
        scores = self.batch_score_forecasts(forecasts, questions)
        average_score = self.calculate_average_score(scores)
        
        return {
            "total_forecasts": len(forecasts),
            "scored_forecasts": len([s for s in scores if s is not None]),
            "average_score": average_score,
            "scores": scores
        }

    def _is_extreme_probability(self, probability: Probability) -> bool:
        """
        Check if a probability is considered extreme.
        
        Args:
            probability: The probability to check
            
        Returns:
            True if probability is extreme (< 0.05 or > 0.95)
        """
        return probability.value < 0.05 or probability.value > 0.95

    def generate_forecast(self, question: Question) -> Forecast:
        """
        Generate a forecast for a question using mock AI prediction logic.
        
        This is a mock implementation that simulates AI forecasting by using
        the community prediction as a base and adding some random variation.
        
        Args:
            question: The question to generate a forecast for
            
        Returns:
            Generated forecast
            
        Raises:
            ForecastValidationError: If the question cannot be forecasted
        """
        import random
        from src.domain.entities.prediction import (
            Prediction, PredictionResult, PredictionConfidence, PredictionMethod
        )
        from src.domain.entities.research_report import (
            ResearchReport, ResearchSource, ResearchQuality
        )
        
        # Validate that we can forecast this question
        if not question.is_open():
            raise ForecastValidationError("Cannot generate forecast for closed question")
        
        if question.question_type != QuestionType.BINARY:
            raise ForecastValidationError("Can only generate forecasts for binary questions")
        
        # Mock AI prediction logic: use community prediction if available,
        # otherwise use a baseline probability with some variation
        base_probability = 0.5  # Default neutral position
        
        # Extract community prediction from metadata if available
        if question.metadata and "community_prediction" in question.metadata:
            community_pred = question.metadata["community_prediction"]
            if isinstance(community_pred, (int, float)) and 0 <= community_pred <= 1:
                base_probability = float(community_pred)
        
        # Create a mock research report
        research_report = self._create_mock_research_report(question, base_probability)
        
        # Generate multiple prediction variants with different methods
        predictions = []
        methods = [PredictionMethod.CHAIN_OF_THOUGHT, PredictionMethod.AUTO_COT]
        
        for i, method in enumerate(methods):
            # Add some random variation to simulate different approaches
            variation = random.uniform(-0.1, 0.1)
            ai_probability = max(0.01, min(0.99, base_probability + variation))
            
            # Generate confidence based on distance from neutral (0.5)
            distance_from_neutral = abs(ai_probability - 0.5)
            if distance_from_neutral > 0.3:
                confidence = PredictionConfidence.HIGH
            elif distance_from_neutral > 0.2:
                confidence = PredictionConfidence.MEDIUM
            else:
                confidence = PredictionConfidence.LOW
            
            # Create prediction using factory method
            prediction = Prediction.create_binary_prediction(
                question_id=question.id,
                research_report_id=research_report.id,
                probability=ai_probability,
                confidence=confidence,
                method=method,
                reasoning=self._generate_mock_reasoning(question, ai_probability, base_probability),
                created_by="ai_forecast_service",
                method_metadata={"base_probability": base_probability, "variation": variation}
            )
            predictions.append(prediction)
        
        # Create final prediction (ensemble of all predictions)
        final_probability = sum(p.result.binary_probability for p in predictions) / len(predictions)
        
        final_prediction = Prediction.create_binary_prediction(
            question_id=question.id,
            research_report_id=research_report.id,
            probability=final_probability,
            confidence=PredictionConfidence.MEDIUM,
            method=PredictionMethod.ENSEMBLE,
            reasoning=f"Ensemble of {len(predictions)} predictions with final probability {final_probability:.3f}",
            created_by="ai_forecast_service",
            method_metadata={"component_predictions": len(predictions)}
        )
        
        # Create forecast using the factory method
        return Forecast.create_new(
            question_id=question.id,
            research_reports=[research_report],
            predictions=predictions,
            final_prediction=final_prediction,
            reasoning_summary=f"AI-generated forecast with {len(predictions)} prediction methods",
            ensemble_method="simple_average",
            weight_distribution={method.value: 1.0/len(methods) for method in methods},
            consensus_strength=1.0 - (max(p.result.binary_probability for p in predictions) - 
                                     min(p.result.binary_probability for p in predictions))
        )

    def _generate_mock_reasoning(self, question: Question, ai_probability: float, base_probability: float) -> str:
        """
        Generate mock reasoning for a forecast.
        
        Args:
            question: The question being forecasted
            ai_probability: The AI's predicted probability
            base_probability: The base probability used
            
        Returns:
            Generated reasoning text
        """
        reasoning_parts = [
            f"Analysis of question: {question.title}",
            f"",
            f"Key factors considered:",
            f"- Current base rate/community prediction: {base_probability:.1%}",
            f"- Historical precedents and trends",
            f"- Expert opinions and market indicators",
            f"- Time horizon until resolution: {question.close_time}",
            f"",
            f"Assessment:",
            f"Based on available information and analytical reasoning, "
            f"the probability of a positive outcome is estimated at {ai_probability:.1%}.",
            f"",
            f"This assessment considers both the base rate and specific factors "
            f"relevant to this particular question. The confidence level reflects "
            f"the quality and quantity of available evidence.",
        ]
        
        return "\n".join(reasoning_parts)

    def _create_mock_research_report(self, question: Question, base_probability: float) -> 'ResearchReport':
        """
        Create a mock research report for a question.
        
        Args:
            question: The question to create a research report for
            base_probability: The base probability used in analysis
            
        Returns:
            Mock research report
        """
        from src.domain.entities.research_report import (
            ResearchReport, ResearchSource, ResearchQuality
        )
        
        # Create mock sources
        sources = [
            ResearchSource(
                url="https://example.com/source1",
                title="Historical Analysis of Similar Events",
                summary="Analysis of historical precedents for this type of question",
                credibility_score=0.8,
                source_type="analysis"
            ),
            ResearchSource(
                url="https://example.com/source2", 
                title="Expert Opinions and Market Indicators",
                summary="Current expert opinions and prediction market data",
                credibility_score=0.7,
                source_type="expert_opinion"
            )
        ]
        
        # Create research report
        return ResearchReport.create_new(
            question_id=question.id,
            title=f"Research Report: {question.title}",
            executive_summary=f"Comprehensive analysis suggests base probability of {base_probability:.1%}",
            detailed_analysis=f"Detailed analysis of question: {question.title}\n\n"
                              f"Key considerations include historical precedents, expert opinions, "
                              f"and current market indicators. Base rate analysis suggests "
                              f"probability around {base_probability:.1%}.",
            sources=sources,
            created_by="ai_research_service",
            key_factors=[
                "Historical precedents",
                "Expert opinions", 
                "Market indicators",
                "Time horizon considerations"
            ],
            base_rates={"historical_rate": base_probability},
            quality=ResearchQuality.MEDIUM,
            confidence_level=0.7,
            research_methodology="Historical analysis combined with expert opinion synthesis",
            reasoning_steps=[
                "Identified relevant historical precedents",
                "Gathered expert opinions and market data", 
                "Analyzed base rates and trends",
                "Synthesized findings into probability estimate"
            ],
            evidence_for=["Supporting historical cases", "Positive expert sentiment"],
            evidence_against=["Contrarian expert views", "Market uncertainty"],
            uncertainties=["Limited historical data", "Evolving circumstances"]
        )
