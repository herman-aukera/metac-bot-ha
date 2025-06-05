"""
ReAct (Reasoning and Acting) agent implementation for interactive research and reasoning.
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
import structlog

from .base_agent import BaseAgent
from ..domain.entities.question import Question
from ..domain.entities.prediction import Prediction, PredictionMethod, PredictionConfidence
from ..domain.entities.research_report import ResearchReport, ResearchSource
from ..domain.value_objects.probability import Probability
from ..infrastructure.external_apis.llm_client import LLMClient
from ..infrastructure.external_apis.search_client import SearchClient
from ..prompts.react_prompts import REACT_SYSTEM_PROMPT, REACT_REASONING_PROMPT, REACT_ACTION_PROMPT

logger = structlog.get_logger(__name__)


class ActionType(Enum):
    """Types of actions the ReAct agent can take."""
    SEARCH = "search"
    THINK = "think"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    FINALIZE = "finalize"


@dataclass
class ReActStep:
    """Represents a single step in the ReAct process."""
    step_number: int
    thought: str
    action: ActionType
    action_input: str
    observation: str
    reasoning: str


class ReActAgent(BaseAgent):
    """
    Agent that uses ReAct (Reasoning and Acting) methodology to interleave
    reasoning, action-taking, and observation for dynamic problem solving.
    """
    
    def __init__(
        self,
        name: str,
        model_config: Dict[str, Any],
        llm_client: LLMClient,
        search_client: Optional[SearchClient] = None,
        max_steps: int = 8,
        max_search_results: int = 5
    ):
        super().__init__(name, model_config)
        self.llm_client = llm_client
        self.search_client = search_client
        self.max_steps = max_steps
        self.max_search_results = max_search_results
        
    async def predict(
        self,
        question: Question,
        include_research: bool = True,
        max_research_depth: int = 3
    ) -> Prediction:
        """Generate prediction using ReAct reasoning and action loop."""
        logger.info(
            "Starting ReAct prediction",
            question_id=question.id,
            max_steps=self.max_steps,
            include_research=include_research
        )
        
        try:
            # Execute ReAct loop
            react_steps = await self._execute_react_loop(question, include_research)
            
            # Generate final prediction from ReAct trace
            prediction = await self._generate_final_prediction(question, react_steps)
            
            logger.info(
                "Generated ReAct prediction",
                question_id=question.id,
                probability=prediction.result.binary_probability,
                confidence=prediction.confidence,
                steps_taken=len(react_steps)
            )
            
            return prediction
            
        except Exception as e:
            logger.error("Failed to generate ReAct prediction", question_id=question.id, error=str(e))
            raise
    
    async def _execute_react_loop(
        self,
        question: Question,
        include_research: bool
    ) -> List[ReActStep]:
        """Execute the main ReAct reasoning and action loop."""
        steps = []
        current_context = self._build_initial_context(question)
        
        for step_num in range(1, self.max_steps + 1):
            logger.info(f"Executing ReAct step {step_num}")
            
            # Generate thought and determine next action
            thought, action, action_input = await self._reason_and_plan(
                question, current_context, steps, step_num
            )
            
            # Execute the action
            observation = await self._execute_action(action, action_input, question)
            
            # Generate reasoning about the observation
            reasoning = await self._reflect_on_observation(
                question, thought, action, action_input, observation
            )
            
            # Create step record
            step = ReActStep(
                step_number=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                reasoning=reasoning
            )
            steps.append(step)
            
            # Update context with new information
            current_context = self._update_context(current_context, step)
            
            # Check if we should finalize (action is FINALIZE or we have enough info)
            if action == ActionType.FINALIZE or self._should_finalize(steps, question):
                logger.info(f"Finalizing ReAct loop at step {step_num}")
                break
        
        return steps
    
    async def _reason_and_plan(
        self,
        question: Question,
        context: str,
        previous_steps: List[ReActStep],
        step_number: int
    ) -> Tuple[str, ActionType, str]:
        """Generate reasoning and plan the next action."""
        # Build prompt with previous steps
        steps_summary = self._format_previous_steps(previous_steps)
        
        prompt = REACT_REASONING_PROMPT.format(
            question_title=question.title,
            question_description=question.description,
            question_type=question.question_type,
            resolution_criteria=question.resolution_criteria or "Not specified",
            context=context,
            previous_steps=steps_summary,
            step_number=step_number,
            max_steps=self.max_steps
        )
        
        response = await self.llm_client.chat_completion(
            messages=[
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return self._parse_reasoning_response(response)
    
    async def _execute_action(
        self,
        action: ActionType,
        action_input: str,
        question: Question
    ) -> str:
        """Execute the specified action and return observation."""
        if action == ActionType.SEARCH:
            return await self._execute_search_action(action_input)
        
        elif action == ActionType.THINK:
            return await self._execute_think_action(action_input, question)
        
        elif action == ActionType.ANALYZE:
            return await self._execute_analyze_action(action_input, question)
        
        elif action == ActionType.SYNTHESIZE:
            return await self._execute_synthesize_action(action_input, question)
        
        elif action == ActionType.FINALIZE:
            return "Ready to finalize prediction based on gathered information."
        
        else:
            return f"Unknown action type: {action}. Continuing with available information."
    
    async def _execute_search_action(self, query: str) -> str:
        """Execute a search action."""
        if not self.search_client:
            return "Search not available. No search client configured."
        
        try:
            search_results = await self.search_client.search(
                query=query,
                max_results=self.max_search_results
            )
            
            if not search_results:
                return f"No search results found for query: {query}"
            
            # Format search results
            formatted_results = []
            for i, result in enumerate(search_results[:self.max_search_results], 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   {result.get('snippet', 'No snippet')}\n"
                    f"   Source: {result.get('url', 'Unknown')}"
                )
            
            return f"Search results for '{query}':\n\n" + "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error("Search action failed", query=query, error=str(e))
            return f"Search failed: {str(e)}"
    
    async def _execute_think_action(self, thought_focus: str, question: Question) -> str:
        """Execute a thinking/reasoning action."""
        prompt = f"""
Think deeply about this aspect of the forecasting question: {thought_focus}

Question: {question.title}
Description: {question.description}

Provide your detailed thoughts and analysis on this specific aspect.
Consider relevant factors, potential outcomes, and implications.
"""
        
        response = await self.llm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a thoughtful analyst providing deep insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        
        return f"Thinking about '{thought_focus}':\n{response}"
    
    async def _execute_analyze_action(self, analysis_target: str, question: Question) -> str:
        """Execute an analysis action."""
        prompt = f"""
Analyze the following information in the context of this forecasting question: {analysis_target}

Question: {question.title}
Description: {question.description}
Resolution Criteria: {question.resolution_criteria or 'Not specified'}

Provide a structured analysis including:
1. Key insights relevant to the prediction
2. Supporting evidence or arguments
3. Potential counterarguments or limitations
4. Implications for probability assessment
"""
        
        response = await self.llm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert analyst providing structured analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        return f"Analysis of '{analysis_target}':\n{response}"
    
    async def _execute_synthesize_action(self, synthesis_focus: str, question: Question) -> str:
        """Execute a synthesis action."""
        prompt = f"""
Synthesize insights about: {synthesis_focus}

Question: {question.title}

Integrate multiple perspectives and pieces of evidence to form coherent insights.
Focus on how different factors interact and what they collectively suggest about the outcome probability.
"""
        
        response = await self.llm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are skilled at synthesizing complex information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        
        return f"Synthesis on '{synthesis_focus}':\n{response}"
    
    async def _reflect_on_observation(
        self,
        question: Question,
        thought: str,
        action: ActionType,
        action_input: str,
        observation: str
    ) -> str:
        """Generate reasoning about the observation."""
        prompt = f"""
Reflect on this ReAct step and its outcome:

Thought: {thought}
Action: {action.value} - {action_input}
Observation: {observation}

Question Context: {question.title}

Provide reasoning about:
1. What did this step reveal?
2. How does it relate to the forecasting question?
3. What are the implications for your prediction?
4. What should be the next focus?
"""
        
        response = await self.llm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are reflecting on reasoning steps and their outcomes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response
    
    async def _generate_final_prediction(
        self,
        question: Question,
        react_steps: List[ReActStep]
    ) -> Prediction:
        """Generate final prediction from ReAct trace."""
        # Format the complete ReAct trace
        trace_summary = self._format_react_trace(react_steps)
        
        prompt = REACT_ACTION_PROMPT.format(
            question_title=question.title,
            question_description=question.description,
            question_type=question.question_type,
            resolution_criteria=question.resolution_criteria or "Not specified",
            react_trace=trace_summary
        )
        
        response = await self.llm_client.chat_completion(
            messages=[
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Parse response for probability and reasoning
        probability, confidence, reasoning = self._parse_final_response(response)
        
        metadata = {
            "agent_type": "react",
            "steps_taken": len(react_steps),
            "max_steps": self.max_steps,
            "actions_by_type": self._count_actions_by_type(react_steps),
            "react_trace": [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "action": step.action.value,
                    "action_input": step.action_input
                }
                for step in react_steps
            ]
        }
        
        return Prediction.create_binary_prediction(
            question_id=question.id,
            research_report_id=uuid4(),  # ReAct doesn't have a separate research report
            probability=probability.value,
            confidence=PredictionConfidence(confidence) if isinstance(confidence, str) else PredictionConfidence.MEDIUM,
            method=PredictionMethod.REACT,
            reasoning=reasoning,
            created_by=self.name,
            method_metadata=metadata
        )
    
    async def conduct_research(
        self, 
        question: Question,
        search_config: Optional[Dict[str, Any]] = None
    ) -> ResearchReport:
        """
        Conduct research for a given question using ReAct methodology.
        
        Args:
            question: The question to research
            search_config: Optional configuration for search behavior
            
        Returns:
            Research report with findings and analysis
        """
        # Simple implementation - use search client to gather information
        if not self.search_client:
            # Return empty research report if no search client
            return ResearchReport.create_new(
                question_id=question.id,
                title=f"Research for: {question.title}",
                executive_summary="No research conducted - search client not available",
                detailed_analysis="No detailed analysis available",
                sources=[],
                created_by=self.name
            )
            
        try:
            search_results = await self.search_client.search(question.title)
            sources = [
                ResearchSource(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    summary=result.get("snippet", ""),
                    credibility_score=0.8,  # Default score
                    publish_date=None
                )
                for result in search_results[:5]  # Limit to top 5 results
            ]
            
            return ResearchReport.create_new(
                question_id=question.id,
                title=f"Research for: {question.title}",
                executive_summary=f"Found {len(sources)} relevant sources for research",
                detailed_analysis="Basic research conducted using search results",
                sources=sources,
                created_by=self.name
            )
        except Exception as e:
            logger.error("Research failed", error=str(e))
            return ResearchReport.create_new(
                question_id=question.id,
                title=f"Research for: {question.title}",
                executive_summary=f"Research failed: {str(e)}",
                detailed_analysis="Research could not be completed due to error",
                sources=[],
                created_by=self.name
            )

    async def generate_prediction(
        self, 
        question: Question, 
        research_report: ResearchReport
    ) -> Prediction:
        """
        Generate a prediction based on research using ReAct reasoning.
        
        Args:
            question: The question to predict
            research_report: Research findings to base prediction on
            
        Returns:
            Prediction with reasoning and confidence
        """
        # Use the existing predict method which implements ReAct logic
        return await self.predict(question)
    
    def _build_initial_context(self, question: Question) -> str:
        """Build initial context string from the question."""
        context_parts = []
        
        if question.description:
            context_parts.append(f"Description: {question.description}")
        
        if question.resolution_criteria:
            context_parts.append(f"Resolution Criteria: {question.resolution_criteria}")
        
        if question.categories:
            context_parts.append(f"Categories: {', '.join(question.categories)}")
        
        if question.close_time:
            context_parts.append(f"Close Time: {question.close_time}")
        
        context_parts.append(f"Question Type: {question.question_type.value}")
        
        return "\n".join(context_parts) if context_parts else "No additional context available."

    def _update_context(self, current_context: str, step: ReActStep) -> str:
        """Update context with information from a completed step."""
        new_info = f"\nStep {step.step_number} ({step.action.value}): {step.observation[:200]}..."
        return current_context + new_info

    def _format_previous_steps(self, steps: List[ReActStep]) -> str:
        """Format previous steps for inclusion in prompts."""
        if not steps:
            return "No previous steps."
        
        formatted = []
        for step in steps:
            formatted.append(
                f"Step {step.step_number}:\n"
                f"Thought: {step.thought}\n"
                f"Action: {step.action.value} - {step.action_input}\n"
                f"Observation: {step.observation}\n"
                f"Reasoning: {step.reasoning}"
            )
        return "\n\n".join(formatted)

    def _should_finalize(self, steps: List[ReActStep], question: Question) -> bool:
        """Determine if we should finalize based on collected information."""
        # Finalize if we've gathered substantial information
        if len(steps) >= 5:
            return True
        
        # Finalize if last few steps didn't add much new information
        if len(steps) >= 3:
            recent_steps = steps[-2:]
            if all("no" in step.observation.lower() or "not found" in step.observation.lower() 
                  for step in recent_steps):
                return True
        
        return False

    def _parse_reasoning_response(self, response: str) -> Tuple[str, ActionType, str]:
        """Parse LLM response to extract thought, action, and action input."""
        try:
            # Look for structured format
            lines = response.strip().split('\n')
            thought = ""
            action = ActionType.THINK
            action_input = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Thought:"):
                    thought = line[8:].strip()
                elif line.startswith("Action:"):
                    action_str = line[7:].strip().upper()
                    try:
                        action = ActionType(action_str.lower())
                    except ValueError:
                        action = ActionType.THINK
                elif line.startswith("Action Input:"):
                    action_input = line[13:].strip()
            
            return thought or "Continuing analysis", action, action_input or "general analysis"
            
        except Exception:
            # Fallback to default
            return "Analyzing the question", ActionType.THINK, "question analysis"

    def _format_react_trace(self, steps: List[ReActStep]) -> str:
        """Format complete ReAct trace for final prediction."""
        if not steps:
            return "No reasoning steps completed."
        
        trace = []
        for step in steps:
            trace.append(
                f"Step {step.step_number}:\n"
                f"Thought: {step.thought}\n"
                f"Action: {step.action.value} - {step.action_input}\n"
                f"Observation: {step.observation}\n"
                f"Reflection: {step.reasoning}"
            )
        return "\n\n".join(trace)

    def _parse_final_response(self, response: str) -> Tuple[Probability, float, str]:
        """Parse final prediction response to extract probability, confidence, and reasoning."""
        try:
            # Try to find probability in the response
            import re
            
            # Look for probability patterns
            prob_match = re.search(r'(?:probability|chance|likelihood).*?(\d+(?:\.\d+)?)', response.lower())
            if prob_match:
                prob_value = float(prob_match.group(1))
                # Normalize to 0-1 range if needed
                if prob_value > 1:
                    prob_value = prob_value / 100
            else:
                prob_value = 0.5  # Default neutral
            
            # Look for confidence patterns
            conf_match = re.search(r'confidence.*?(\d+(?:\.\d+)?)', response.lower())
            if conf_match:
                confidence = float(conf_match.group(1))
                if confidence > 1:
                    confidence = confidence / 100
            else:
                confidence = 0.7  # Default moderate confidence
            
            probability = Probability(prob_value)
            reasoning = response.strip()
            
            return probability, confidence, reasoning
            
        except Exception:
            # Fallback to defaults
            return Probability(0.5), 0.6, response.strip() or "ReAct analysis completed"

    def _count_actions_by_type(self, steps: List[ReActStep]) -> Dict[str, int]:
        """Count actions by type for metadata."""
        counts = {}
        for step in steps:
            action_name = step.action.value
            counts[action_name] = counts.get(action_name, 0) + 1
        return counts
