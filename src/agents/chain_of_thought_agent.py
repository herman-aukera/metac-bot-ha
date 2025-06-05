"""Chain of Thought agent implementation."""

from typing import Dict, Any, Optional, List
import json
from datetime import datetime

from .base_agent import BaseAgent
from ..domain.entities.question import Question
from ..domain.entities.research_report import ResearchReport, ResearchSource, ResearchQuality
from ..domain.entities.prediction import Prediction, PredictionMethod, PredictionConfidence
from ..prompts.cot_prompts import ChainOfThoughtPrompts
from ..infrastructure.external_apis.llm_client import LLMClient
from ..infrastructure.external_apis.search_client import SearchClient


class ChainOfThoughtAgent(BaseAgent):
    """
    Agent that uses Chain of Thought reasoning for forecasting.
    
    Breaks down complex problems into step-by-step reasoning,
    encouraging the model to show its work and think through
    each aspect of the forecast systematically.
    """
    
    def __init__(
        self, 
        name: str, 
        model_config: Dict[str, Any],
        llm_client: LLMClient,
        search_client: SearchClient
    ):
        super().__init__(name, model_config)
        self.llm_client = llm_client
        self.search_client = search_client
        self.prompts = ChainOfThoughtPrompts()
    
    async def conduct_research(
        self, 
        question: Question,
        search_config: Optional[Dict[str, Any]] = None
    ) -> ResearchReport:
        """Conduct research using Chain of Thought reasoning."""
        self.logger.info("Starting CoT research", question_title=question.title)
        
        # 1. Deconstruct question
        deconstruct_prompt = self.prompts.deconstruct_question(question)
        breakdown_response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": deconstruct_prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        question_breakdown = breakdown_response
        
        # 2. Identify research areas
        research_areas_prompt = self.prompts.identify_research_areas(question, question_breakdown)
        research_areas_response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": research_areas_prompt}],
            temperature=0.4,
            max_tokens=300
        )
        
        research_areas = research_areas_response
        
        # 3. Gather information (simplified for now)
        if self.search_client:
            search_queries = self._parse_search_queries(research_areas_response) # Assuming LLM provides queries
            
            all_search_results = []
            for query in search_queries[:3]: # Limit number of queries
                try:
                    results = await self.search_client.search(query, max_results=3)
                    all_search_results.extend(results)
                except Exception as e:
                    self.logger.error("Search query failed", query=query, error=str(e))

            sources = [
                ResearchSource(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    summary=result.get("snippet", ""), # Using snippet as summary
                    credibility_score=0.7, # Placeholder
                    publish_date=None # Placeholder
                )
                for result in all_search_results
            ]
        else:
            sources = []
            self.logger.warn("No search client available, research will be limited.")

        # 4. Synthesize findings
        synthesis_prompt = self.prompts.synthesize_findings(question, sources)
        synthesis_response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.5,
            max_tokens=1000
        )
        
        # Parse synthesis response
        analysis_data = self._parse_analysis_response(synthesis_response)
        
        # Create research report
        research_report = ResearchReport.create_new(
            question_id=question.id,
            title=f"CoT Research: {question.title}",
            executive_summary=analysis_data.get("executive_summary", ""),
            detailed_analysis=analysis_data.get("detailed_analysis", ""),
            sources=sources,
            created_by=self.name,
            key_factors=analysis_data.get("key_factors", []),
            base_rates=analysis_data.get("base_rates", {}),
            quality=ResearchQuality.HIGH,
            confidence_level=analysis_data.get("confidence_level", 0.7),
            research_methodology="Chain of Thought systematic breakdown",
            reasoning_steps=analysis_data.get("reasoning_steps", []),
            evidence_for=analysis_data.get("evidence_for", []),
            evidence_against=analysis_data.get("evidence_against", []),
            uncertainties=analysis_data.get("uncertainties", []),
        )
        
        self.logger.info("CoT research completed", sources_found=len(sources))
        return research_report
    
    async def generate_prediction(
        self, 
        question: Question, 
        research_report: ResearchReport
    ) -> Prediction:
        """Generate prediction using Chain of Thought based on research."""
        self.logger.info("Starting CoT prediction generation", question_id=question.id)
        
        # Build prompt using research report
        prediction_prompt = self.prompts.generate_prediction_prompt(question, research_report)
        
        llm_response_str = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prediction_prompt}],
            temperature=self.model_config.get("temperature", 0.2),
            max_tokens=self.model_config.get("max_tokens", 300)
        )
        
        # Parse LLM response
        prediction_data = self._parse_prediction_response(llm_response_str)
        
        # Create prediction based on question type
        if question.question_type.value == "binary":
            prediction = Prediction.create_binary_prediction(
                question_id=question.id,
                research_report_id=research_report.id,
                probability=prediction_data["probability"],
                confidence=prediction_data["confidence"],
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning=prediction_data["reasoning"],
                created_by=self.name,
                reasoning_steps=prediction_data["reasoning_steps"],
                lower_bound=prediction_data.get("lower_bound"),
                upper_bound=prediction_data.get("upper_bound"),
                confidence_interval=prediction_data.get("confidence_interval"),
                method_metadata={
                    "model": self.model_config.get("model", "gpt-4"),
                    "temperature": 0.1,
                    "reasoning_depth": len(prediction_data["reasoning_steps"]),
                }
            )
        else:
            # Handle other question types as needed
            raise NotImplementedError(f"Question type {question.question_type} not yet supported")
        
        self.logger.info("CoT prediction completed", probability=prediction_data["probability"])
        return prediction
    
    def _parse_research_areas(self, breakdown_response: str) -> List[str]:
        """Parse the question breakdown to extract key research areas."""
        try:
            # Try to parse as JSON first
            data = json.loads(breakdown_response)
            return data.get("research_areas", [])
        except json.JSONDecodeError:
            # Fallback to text parsing
            lines = breakdown_response.strip().split('\n')
            areas = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ["area:", "topic:", "factor:"]):
                    # Extract the area after the keyword
                    area = line.split(":", 1)[-1].strip()
                    if area:
                        areas.append(area)
            return areas[:5]  # Limit to 5 key areas
    
    def _parse_analysis_response(self, analysis_response: str) -> Dict[str, Any]:
        """Parse the research analysis response."""
        try:
            # Try JSON parsing first
            return json.loads(analysis_response)
        except json.JSONDecodeError:
            # Fallback to structured text parsing
            return {
                "executive_summary": analysis_response[:500] + "...",
                "detailed_analysis": analysis_response,
                "key_factors": [],
                "base_rates": {},
                "confidence_level": 0.7,
                "reasoning_steps": [analysis_response],
                "evidence_for": [],
                "evidence_against": [],
                "uncertainties": [],
            }
    
    def _parse_prediction_response(self, prediction_response: str) -> Dict[str, Any]:
        """Parse the prediction response."""
        try:
            # Try JSON parsing first
            data = json.loads(prediction_response)
            
            # Convert confidence string to enum if needed
            confidence_str = data.get("confidence", "medium").lower()
            confidence_mapping = {
                "very_low": PredictionConfidence.VERY_LOW,
                "low": PredictionConfidence.LOW,
                "medium": PredictionConfidence.MEDIUM,
                "high": PredictionConfidence.HIGH,
                "very_high": PredictionConfidence.VERY_HIGH,
            }
            
            data["confidence"] = confidence_mapping.get(
                confidence_str, PredictionConfidence.MEDIUM
            )
            
            return data
            
        except json.JSONDecodeError:
            # Fallback parsing for non-JSON responses
            # Look for probability patterns
            import re
            prob_match = re.search(r'(\d+(?:\.\d+)?)\s*%', prediction_response)
            if prob_match:
                probability = float(prob_match.group(1)) / 100.0
            else:
                probability = 0.5  # Default fallback
            
            return {
                "probability": probability,
                "confidence": PredictionConfidence.MEDIUM,
                "reasoning": prediction_response,
                "reasoning_steps": [prediction_response],
            }
