"""
Ensemble agent that combines predictions from multiple reasoning strategies.
"""
import asyncio
from typing import List, Dict, Any, Optional
from statistics import median
import structlog

from .base_agent import BaseAgent
from ..domain.entities.question import Question
from ..domain.entities.prediction import Prediction, PredictionConfidence, PredictionMethod
from ..domain.value_objects.probability import Probability
from ..domain.services.forecasting_service import ForecastingService
from ..infrastructure.external_apis.llm_client import LLMClient
from ..infrastructure.external_apis.search_client import SearchClient
from ..domain.entities.research_report import ResearchReport

logger = structlog.get_logger(__name__)


class EnsembleAgent(BaseAgent):
    """
    Agent that combines predictions from multiple other agents using various aggregation strategies.
    """
    
    def __init__(
        self,
        name: str,
        model_config: Dict[str, Any],
        agents: List[BaseAgent],
        forecasting_service: ForecastingService,
        llm_client: Optional[LLMClient] = None,
        search_client: Optional[SearchClient] = None,
        aggregation_strategy: str = "confidence_weighted"
    ):
        # For ensemble agent, LLM client is optional (only needed for meta-reasoning)
        super().__init__(name, model_config)
        # Try to get LLM client from first agent if available, otherwise use provided one
        self.llm_client = llm_client
        if not self.llm_client and agents:
            # Check if first agent has llm_client attribute
            first_agent = agents[0]
            try:
                # Try to access llm_client attribute safely
                self.llm_client = getattr(first_agent, 'llm_client', None)
            except AttributeError:
                self.llm_client = None
        self.search_client = search_client
        self.agents = agents
        self.forecasting_service = forecasting_service
        self.aggregation_strategy = aggregation_strategy
        
        if not agents:
            raise ValueError("Ensemble agent requires at least one base agent")
    
    async def conduct_research(
        self, 
        question: Question,
        search_config: Optional[Dict[str, Any]] = None
    ) -> ResearchReport:
        """
        Conduct research by delegating to base agents and combining their research.
        """
        logger.info("Starting ensemble research", question_id=question.id, agent_count=len(self.agents))
        
        # For now, delegate to the first agent's research
        # In a more sophisticated implementation, we could combine research from multiple agents
        if self.agents:
            return await self.agents[0].conduct_research(question, search_config)
        else:
            # Fallback - create minimal research report
            return ResearchReport.create_new(
                question_id=question.id,
                title="Ensemble Research",
                executive_summary="Ensemble agent with no base agents - minimal research",
                detailed_analysis="Ensemble agent with no base agents - minimal research",
                sources=[],
                confidence_level=0.5,
                research_depth=0,
                research_time_spent=0.0,
                created_by=self.name
            )
    
    async def generate_prediction(
        self, 
        question: Question, 
        research_report: ResearchReport
    ) -> Prediction:
        """
        Generate prediction by combining predictions from multiple agents.
        """
        # For now, create a simple placeholder prediction
        from ..domain.entities.prediction import Prediction, PredictionConfidence, PredictionMethod
        
        return Prediction.create_binary_prediction(
            question_id=question.id,
            research_report_id=research_report.id,
            probability=0.5,  # Placeholder
            confidence=PredictionConfidence.MEDIUM,
            method=PredictionMethod.ENSEMBLE,
            reasoning="Ensemble prediction - placeholder implementation",
            created_by=self.name
        )
    
    async def predict(
        self,
        question: Question,
        include_research: bool = True,
        max_research_depth: int = 3
    ) -> Prediction:
        """Generate ensemble prediction by combining multiple agent predictions."""
        logger.info(
            "Starting ensemble prediction",
            question_id=question.id,
            agent_count=len(self.agents),
            aggregation_strategy=self.aggregation_strategy
        )
        
        try:
            # Generate predictions from all agents concurrently
            agent_tasks = [
                self._safe_agent_predict(agent, question, include_research, max_research_depth)
                for agent in self.agents
            ]
            
            agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Filter out failed predictions and collect successful ones
            successful_predictions = []
            failed_agents = []
            
            for i, result in enumerate(agent_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Agent prediction failed",
                        agent_type=self.agents[i].__class__.__name__,
                        error=str(result)
                    )
                    failed_agents.append(self.agents[i].__class__.__name__)
                else:
                    successful_predictions.append(result)
            
            if not successful_predictions:
                raise RuntimeError("All agent predictions failed")
            
            # Aggregate predictions
            ensemble_prediction = await self._aggregate_predictions(
                question, successful_predictions, failed_agents
            )
            
            logger.info(
                "Generated ensemble prediction",
                question_id=question.id,
                probability=ensemble_prediction.result.binary_probability,
                confidence=ensemble_prediction.confidence,
                successful_agents=len(successful_predictions),
                failed_agents=len(failed_agents)
            )
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error("Failed to generate ensemble prediction", question_id=question.id, error=str(e))
            raise
    
    async def _safe_agent_predict(
        self,
        agent: BaseAgent,
        question: Question,
        include_research: bool,
        max_research_depth: int
    ) -> Prediction:
        """Safely execute agent prediction with error handling."""
        try:
            # Use BaseAgent's full_forecast_cycle method which returns research_report and prediction
            research_report, prediction = await agent.full_forecast_cycle(question)
            return prediction
        except Exception as e:
            logger.error(
                "Agent prediction failed",
                agent_type=agent.__class__.__name__,
                question_id=question.id,
                error=str(e)
            )
            raise
    
    async def _aggregate_predictions(
        self,
        question: Question,
        predictions: List[Prediction],
        failed_agents: List[str]
    ) -> Prediction:
        """Aggregate multiple predictions using the specified strategy."""
        if len(predictions) == 1:
            # Single prediction - just enhance metadata
            prediction = predictions[0]
            enhanced_metadata = prediction.method_metadata.copy() if prediction.method_metadata else {}
            enhanced_metadata.update({
                "ensemble_agent": True,
                "predictions_used": 1,
                "failed_agents": failed_agents,
                "aggregation_strategy": "single"
            })
            
            return Prediction.create_binary_prediction(
                question_id=question.id,
                research_report_id=prediction.research_report_id,
                probability=prediction.result.binary_probability or 0.5,
                reasoning=prediction.reasoning,
                confidence=prediction.confidence,
                method=PredictionMethod.ENSEMBLE,
                created_by=self.name,
                method_metadata=enhanced_metadata
            )
        
        # Multiple predictions - aggregate based on strategy
        if self.aggregation_strategy == "simple_average":
            aggregated_prediction = self.forecasting_service.aggregate_predictions(predictions, "weighted_average")
            final_probability = aggregated_prediction.result.binary_probability
        elif self.aggregation_strategy == "weighted_average":
            aggregated_prediction = self.forecasting_service.aggregate_predictions(predictions, "weighted_average")
            final_probability = aggregated_prediction.result.binary_probability
        elif self.aggregation_strategy == "confidence_weighted":
            aggregated_prediction = self.forecasting_service.aggregate_predictions(predictions, "confidence_weighted")
            final_probability = aggregated_prediction.result.binary_probability
        elif self.aggregation_strategy == "median":
            aggregated_prediction = self.forecasting_service.aggregate_predictions(predictions, "median")
            final_probability = aggregated_prediction.result.binary_probability
        elif self.aggregation_strategy == "meta_reasoning":
            return await self._meta_reasoning_aggregation(question, predictions, failed_agents)
        else:
            # Default to confidence weighted
            aggregated_prediction = self.forecasting_service.aggregate_predictions(predictions, "confidence_weighted")
            final_probability = aggregated_prediction.result.binary_probability
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(predictions)
        
        # Ensure final_probability is not None
        if final_probability is None:
            final_probability = 0.5  # Default fallback
        
        # Generate ensemble reasoning
        ensemble_reasoning = self._generate_ensemble_reasoning(predictions, final_probability)
        
        # Create ensemble metadata
        ensemble_metadata = {
            "ensemble_agent": True,
            "aggregation_strategy": self.aggregation_strategy,
            "predictions_used": len(predictions),
            "failed_agents": failed_agents,
            "agent_predictions": [
                {
                    "agent_type": pred.method_metadata.get("agent_type", "unknown") if pred.method_metadata else "unknown",
                    "probability": pred.result.binary_probability,
                    "confidence": pred.get_confidence_score()
                }
                for pred in predictions
            ],
            "probability_spread": max(p.result.binary_probability for p in predictions if p.result.binary_probability is not None) - 
                              min(p.result.binary_probability for p in predictions if p.result.binary_probability is not None),
            "confidence_range": [min(p.get_confidence_score() for p in predictions), 
                               max(p.get_confidence_score() for p in predictions)]
        }
        
        # Convert float confidence to enum
        if ensemble_confidence <= 0.2:
            confidence_enum = PredictionConfidence.VERY_LOW
        elif ensemble_confidence <= 0.4:
            confidence_enum = PredictionConfidence.LOW
        elif ensemble_confidence <= 0.6:
            confidence_enum = PredictionConfidence.MEDIUM
        elif ensemble_confidence <= 0.8:
            confidence_enum = PredictionConfidence.HIGH
        else:
            confidence_enum = PredictionConfidence.VERY_HIGH

        return Prediction.create_binary_prediction(
            question_id=question.id,
            research_report_id=predictions[0].research_report_id,  # Use first prediction's report
            probability=final_probability,
            reasoning=ensemble_reasoning,
            confidence=confidence_enum,
            method=PredictionMethod.ENSEMBLE,
            created_by=self.name,
            method_metadata=ensemble_metadata
        )
    
    async def _meta_reasoning_aggregation(
        self,
        question: Question,
        predictions: List[Prediction],
        failed_agents: List[str]
    ) -> Prediction:
        """Use LLM to perform meta-reasoning over agent predictions."""
        if not self.llm_client:
            logger.warning("No LLM client available for meta-reasoning, falling back to confidence weighting")
            aggregated_prediction = self.forecasting_service.aggregate_predictions(predictions, "confidence_weighted")
            final_probability = aggregated_prediction.result.binary_probability or 0.5
            ensemble_confidence = self._calculate_ensemble_confidence(predictions)
            ensemble_reasoning = self._generate_ensemble_reasoning(predictions, final_probability)
        else:
            # Format predictions for meta-reasoning
            predictions_summary = self._format_predictions_for_meta_reasoning(predictions)
            
            meta_prompt = f"""
Question: {question.title}
Description: {question.description}

You have predictions from multiple AI forecasting agents using different reasoning strategies:

{predictions_summary}

As a meta-reasoner, analyze these predictions and provide your own assessment:

1. Which predictions seem most/least reliable and why?
2. Are there consistent patterns or significant disagreements?
3. What does the spread of predictions tell us about uncertainty?
4. How should we weight different reasoning approaches for this specific question?

Provide your meta-analysis:

PROBABILITY: [your final probability 0-1 or percentage]
CONFIDENCE: [0-1 confidence in your meta-prediction]
REASONING: [detailed explanation of how you analyzed and synthesized the different agent predictions]
"""
            
            response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a meta-reasoning expert analyzing predictions from multiple AI agents."
                    },
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=0.3
            )
            
            # Parse meta-reasoning response
            final_probability, ensemble_confidence, ensemble_reasoning = self._parse_meta_response(response)
        
        # Create meta-reasoning metadata
        meta_metadata = {
            "ensemble_agent": True,
            "aggregation_strategy": "meta_reasoning",
            "predictions_used": len(predictions),
            "failed_agents": failed_agents,
            "agent_predictions": [
                {
                    "agent_type": pred.method_metadata.get("agent_type", "unknown") if pred.method_metadata else "unknown",
                    "probability": pred.result.binary_probability,
                    "confidence": pred.get_confidence_score(),
                    "reasoning_summary": pred.reasoning[:200] + "..." if len(pred.reasoning) > 200 else pred.reasoning
                }
                for pred in predictions
            ]
        }
        
        # Convert float confidence to PredictionConfidence enum
        if ensemble_confidence <= 0.2:
            confidence_enum = PredictionConfidence.VERY_LOW
        elif ensemble_confidence <= 0.4:
            confidence_enum = PredictionConfidence.LOW
        elif ensemble_confidence <= 0.6:
            confidence_enum = PredictionConfidence.MEDIUM
        elif ensemble_confidence <= 0.8:
            confidence_enum = PredictionConfidence.HIGH
        else:
            confidence_enum = PredictionConfidence.VERY_HIGH

        # Create a PredictionResult with the final probability  
        from ..domain.entities.prediction import PredictionResult
        result = PredictionResult(binary_probability=final_probability)

        return Prediction.create(
            question_id=question.id,
            research_report_id=predictions[0].research_report_id,
            result=result,
            confidence=confidence_enum,
            method=PredictionMethod.ENSEMBLE,
            reasoning=ensemble_reasoning,
            created_by=self.name,
            method_metadata=meta_metadata
        )
    
    def _calculate_ensemble_confidence(self, predictions: List[Prediction]) -> float:
        """Calculate ensemble confidence based on agreement and individual confidences."""
        if not predictions:
            return 0.0
        
        if len(predictions) == 1:
            return predictions[0].get_confidence_score()
        
        # Calculate agreement (inverse of spread)
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]
        if not probabilities:
            return 0.0
        prob_spread = max(probabilities) - min(probabilities)
        agreement_factor = 1.0 - min(prob_spread, 1.0)  # Higher agreement = higher confidence
        
        # Average confidence (convert enum to float)
        confidence_scores = [p.get_confidence_score() for p in predictions]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Ensemble confidence combines individual confidence and agreement
        ensemble_confidence = (avg_confidence * 0.7) + (agreement_factor * 0.3)
        
        return min(max(ensemble_confidence, 0.0), 1.0)
    
    def _generate_ensemble_reasoning(
        self,
        predictions: List[Prediction],
        final_probability: float
    ) -> str:
        """Generate reasoning explaining the ensemble prediction."""
        reasoning_parts = [
            f"Ensemble prediction combining {len(predictions)} agent predictions:",
            f"Final probability: {final_probability:.3f}"
        ]
        
        # Summarize individual predictions
        reasoning_parts.append("\nIndividual agent predictions:")
        for i, pred in enumerate(predictions):
            agent_type = pred.method_metadata.get("agent_type", f"Agent {i+1}") if pred.method_metadata else f"Agent {i+1}"
            reasoning_parts.append(
                f"- {agent_type}: {pred.result.binary_probability:.3f} (confidence: {pred.get_confidence_score():.2f})"
            )
        
        # Analysis
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]
        reasoning_parts.extend([
            f"\nPrediction spread: {max(probabilities) - min(probabilities):.3f}",
            f"Median prediction: {median(probabilities):.3f}",
            f"Average prediction: {sum(probabilities) / len(probabilities):.3f}"
        ])
        
        # Agreement analysis
        if max(probabilities) - min(probabilities) < 0.1:
            reasoning_parts.append("High agreement between agents suggests robust prediction.")
        elif max(probabilities) - min(probabilities) > 0.3:
            reasoning_parts.append("Significant disagreement between agents indicates high uncertainty.")
        else:
            reasoning_parts.append("Moderate agreement between agents.")
        
        return "\n".join(reasoning_parts)
    
    def _format_predictions_for_meta_reasoning(self, predictions: List[Prediction]) -> str:
        """Format predictions for meta-reasoning prompt."""
        formatted_predictions = []
        
        for i, pred in enumerate(predictions):
            agent_type = pred.method_metadata.get("agent_type", f"Agent {i+1}") if pred.method_metadata else f"Agent {i+1}"
            formatted_predictions.append(
                f"{agent_type.upper()} AGENT:\n"
                f"Probability: {pred.result.binary_probability:.3f}\n"
                f"Confidence: {pred.get_confidence_score():.2f}\n"
                f"Key reasoning: {pred.reasoning[:300]}...\n"
            )
        
        return "\n" + "="*50 + "\n".join(formatted_predictions)
    
    def _parse_meta_response(self, response: str) -> tuple[float, float, str]:
        """Parse meta-reasoning response."""
        lines = response.strip().split('\n')
        
        probability_value = 0.5
        confidence = 0.5
        reasoning = response  # Default to full response
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('PROBABILITY:'):
                try:
                    prob_text = line.split(':', 1)[1].strip()
                    if '%' in prob_text:
                        probability_value = float(prob_text.replace('%', '')) / 100
                    else:
                        probability_value = float(prob_text)
                except ValueError:
                    probability_value = 0.5
                    
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 0.5
                    
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        return probability_value, confidence, reasoning
    
    def get_agent_types(self) -> List[str]:
        """Get list of agent types in the ensemble."""
        return [agent.__class__.__name__ for agent in self.agents]
    
    def set_aggregation_strategy(self, strategy: str) -> None:
        """Change the aggregation strategy."""
        valid_strategies = [
            "simple_average", "weighted_average", "confidence_weighted", 
            "median", "meta_reasoning"
        ]
        
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        
        self.aggregation_strategy = strategy
        logger.info("Changed aggregation strategy", new_strategy=strategy)
