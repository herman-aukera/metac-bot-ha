"""
Forecasting pipeline that orchestrates the end-to-end forecasting process.
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from ..domain.entities.question import Question
from ..domain.entities.forecast import Forecast
from ..domain.services.forecasting_service import ForecastingService
from ..agents.base_agent import BaseAgent
from ..agents.cot_agent import ChainOfThoughtAgent
from ..agents.tot_agent import TreeOfThoughtAgent
from ..agents.react_agent import ReActAgent
from ..agents.ensemble_agent import EnsembleAgent
from ..infrastructure.config.settings import Settings
from ..infrastructure.external_apis.llm_client import LLMClient
from ..infrastructure.external_apis.search_client import SearchClient
from ..infrastructure.external_apis.metaculus_client import MetaculusClient

logger = structlog.get_logger(__name__)


class ForecastingPipeline:
    """Main pipeline for generating forecasts using multiple agents and aggregation strategies."""
    
    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        search_client: Optional[SearchClient] = None,
        metaculus_client: Optional[MetaculusClient] = None
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.search_client = search_client
        self.metaculus_client = metaculus_client
        self.forecasting_service = ForecastingService()
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()
        
    def _initialize_agents(self) -> None:
        """Initialize all forecasting agents."""
        try:
            # Individual reasoning agents
            self.agents["cot"] = ChainOfThoughtAgent(
                llm_client=self.llm_client,
                search_client=self.search_client
            )
            
            self.agents["tot"] = TreeOfThoughtAgent(
                llm_client=self.llm_client,
                search_client=self.search_client
            )
            
            self.agents["react"] = ReActAgent(
                llm_client=self.llm_client,
                search_client=self.search_client
            )
            
            # Ensemble agent that combines multiple approaches
            base_agents = [self.agents["cot"], self.agents["tot"], self.agents["react"]]
            self.agents["ensemble"] = EnsembleAgent(
                agents=base_agents,
                forecasting_service=self.forecasting_service
            )
            
            logger.info("Initialized all forecasting agents", agent_count=len(self.agents))
            
        except Exception as e:
            logger.error("Failed to initialize agents", error=str(e))
            raise
    
    async def generate_forecast(
        self,
        question: Question,
        agent_names: Optional[List[str]] = None,
        include_research: bool = True,
        max_research_depth: int = 3
    ) -> Forecast:
        """
        Generate a forecast for a given question using specified agents.
        
        Args:
            question: The question to forecast
            agent_names: List of agent names to use. If None, uses ensemble agent
            include_research: Whether to perform research before forecasting
            max_research_depth: Maximum depth for research queries
            
        Returns:
            Final aggregated forecast
        """
        logger.info(
            "Starting forecast generation",
            question_id=question.id,
            question_title=question.title,
            agent_names=agent_names,
            include_research=include_research
        )
        
        try:
            # Use ensemble agent by default if no specific agents specified
            if agent_names is None:
                agent_names = ["ensemble"]
            
            # Validate agent names
            invalid_agents = [name for name in agent_names if name not in self.agents]
            if invalid_agents:
                raise ValueError(f"Invalid agent names: {invalid_agents}")
            
            # Generate predictions from each agent
            predictions = []
            for agent_name in agent_names:
                agent = self.agents[agent_name]
                
                logger.info("Generating prediction", agent=agent_name)
                prediction = await agent.predict(
                    question=question,
                    include_research=include_research,
                    max_research_depth=max_research_depth
                )
                predictions.append(prediction)
                
                logger.info(
                    "Generated prediction",
                    agent=agent_name,
                    probability=prediction.probability.value,
                    confidence=prediction.confidence
                )
            
            # Aggregate predictions into final forecast
            if len(predictions) == 1:
                # Single prediction - convert to forecast
                prediction = predictions[0]
                forecast = Forecast.create(
                    question_id=question.id,
                    predictions=[prediction],
                    final_probability=prediction.probability,
                    aggregation_method="single",
                    metadata={
                        "agent_used": agent_names[0],
                        "pipeline_version": "1.0",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            else:
                # Multiple predictions - aggregate using confidence weighting
                final_probability = self.forecasting_service.confidence_weighted_average(predictions)
                
                forecast = Forecast.create(
                    question_id=question.id,
                    predictions=predictions,
                    final_probability=final_probability,
                    aggregation_method="confidence_weighted",
                    metadata={
                        "agents_used": agent_names,
                        "pipeline_version": "1.0",
                        "timestamp": datetime.utcnow().isoformat(),
                        "prediction_count": len(predictions)
                    }
                )
            
            logger.info(
                "Generated final forecast",
                question_id=question.id,
                final_probability=forecast.final_probability.value,
                prediction_count=len(predictions),
                aggregation_method=forecast.aggregation_method
            )
            
            return forecast
            
        except Exception as e:
            logger.error(
                "Failed to generate forecast",
                question_id=question.id,
                error=str(e),
                agent_names=agent_names
            )
            raise
    
    async def batch_forecast(
        self,
        questions: List[Question],
        agent_names: Optional[List[str]] = None,
        include_research: bool = True,
        max_research_depth: int = 3,
        batch_size: int = 5
    ) -> List[Forecast]:
        """
        Generate forecasts for multiple questions in batches.
        
        Args:
            questions: List of questions to forecast
            agent_names: List of agent names to use
            include_research: Whether to perform research
            max_research_depth: Maximum research depth
            batch_size: Number of questions to process concurrently
            
        Returns:
            List of forecasts
        """
        logger.info(
            "Starting batch forecast",
            question_count=len(questions),
            agent_names=agent_names,
            batch_size=batch_size
        )
        
        forecasts = []
        
        # Process questions in batches to avoid overwhelming APIs
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(questions) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}", batch_size=len(batch))
            
            # Create tasks for concurrent processing
            tasks = [
                self.generate_forecast(
                    question=question,
                    agent_names=agent_names,
                    include_research=include_research,
                    max_research_depth=max_research_depth
                )
                for question in batch
            ]
            
            # Execute batch concurrently
            batch_forecasts = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for j, result in enumerate(batch_forecasts):
                if isinstance(result, Exception):
                    logger.error(
                        "Failed to generate forecast in batch",
                        question_id=batch[j].id,
                        question_title=batch[j].title,
                        error=str(result)
                    )
                    # Could add error handling strategy here (retry, skip, etc.)
                else:
                    forecasts.append(result)
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(questions):
                await asyncio.sleep(self.settings.batch_delay_seconds)
        
        logger.info(
            "Completed batch forecast",
            total_questions=len(questions),
            successful_forecasts=len(forecasts),
            failed_forecasts=len(questions) - len(forecasts)
        )
        
        return forecasts
    
    async def benchmark_agents(
        self,
        questions: List[Question],
        agent_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark different agents against a set of questions.
        
        Args:
            questions: Questions to use for benchmarking
            agent_names: Specific agents to benchmark. If None, benchmarks all
            
        Returns:
            Benchmark results with performance metrics
        """
        if agent_names is None:
            agent_names = list(self.agents.keys())
        
        logger.info("Starting agent benchmarking", agents=agent_names, question_count=len(questions))
        
        results = {}
        
        for agent_name in agent_names:
            logger.info(f"Benchmarking agent: {agent_name}")
            
            agent_forecasts = await self.batch_forecast(
                questions=questions,
                agent_names=[agent_name],
                include_research=True,
                max_research_depth=2  # Reduced for benchmarking speed
            )
            
            # Calculate performance metrics
            total_time = sum(
                (forecast.created_at - questions[i].created_at).total_seconds()
                for i, forecast in enumerate(agent_forecasts)
                if i < len(questions)
            )
            
            avg_confidence = sum(
                pred.confidence for forecast in agent_forecasts
                for pred in forecast.predictions
            ) / max(1, sum(len(forecast.predictions) for forecast in agent_forecasts))
            
            results[agent_name] = {
                "forecasts_generated": len(agent_forecasts),
                "avg_processing_time": total_time / max(1, len(agent_forecasts)),
                "avg_confidence": avg_confidence,
                "success_rate": len(agent_forecasts) / len(questions)
            }
        
        logger.info("Completed agent benchmarking", results=results)
        return results
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.agents.keys())
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components."""
        health = {}
        
        # Check LLM client
        try:
            await self.llm_client.health_check()
            health["llm_client"] = True
        except Exception:
            health["llm_client"] = False
        
        # Check search client
        if self.search_client:
            try:
                await self.search_client.health_check()
                health["search_client"] = True
            except Exception:
                health["search_client"] = False
        
        # Check Metaculus client
        if self.metaculus_client:
            try:
                await self.metaculus_client.health_check()
                health["metaculus_client"] = True
            except Exception:
                health["metaculus_client"] = False
        
        # Check agents
        for agent_name in self.agents:
            health[f"agent_{agent_name}"] = True  # Agents are local, assume healthy
        
        return health