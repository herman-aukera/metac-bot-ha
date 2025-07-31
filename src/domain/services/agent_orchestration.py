"""Agent orchestration system with multiple reasoning approaches."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from uuid import UUID, uuid4
import asyncio
import statistics
from collections import defaultdict

from ..entities.question import Question
from ..entities.forecast import Forecast
from ..entities.agent import Agent, ReasoningStyle
from ..value_objects.confidence import Confidence
from ..value_objects.reasoning_step import ReasoningStep


class AggregationMethod(Enum):
    """Methods for aggregating ensemble predictions."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    META_REASONING = "meta_reasoning"


@dataclass
class ResearchReport:
    """Comprehensive research findings for a question."""
    id: UUID
    question_id: int
    sources: List[Dict[str, Any]]
    evidence_synthesis: str
    base_rates: Dict[str, float]
    knowledge_gaps: List[str]
    research_quality_score: float
    timestamp: datetime

    def __post_init__(self):
        if self.id is None:
            object.__setattr__(self, 'id', uuid4())
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.utcnow())


@dataclass
class ReasoningBranch:
    """A single branch in tree-of-thought reasoning."""
    branch_id: str
    reasoning_steps: List[ReasoningStep]
    confidence: Confidence
    conclusion: str

    def get_branch_quality_score(self) -> float:
        """Calculate quality score for this reasoning branch."""
        if not self.reasoning_steps:
            return 0.0

        avg_confidence = sum(step.confidence.level for step in self.reasoning_steps) / len(self.reasoning_steps)
        step_coherence = len(self.reasoning_steps) / 10.0  # Normalize by expected steps

        return (avg_confidence + min(step_coherence, 1.0)) / 2.0


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for ReAct agent."""
    iterations: List[Dict[str, Any]]
    final_reasoning: List[ReasoningStep]
    convergence_score: float

    def get_iteration_count(self) -> int:
        """Get number of reasoning iterations."""
        return len(self.iterations)


@dataclass
class ConsensusMetrics:
    """Metrics for measuring ensemble consensus quality."""
    consensus_strength: float  # 0.0 to 1.0
    prediction_variance: float
    agent_diversity_score: float
    confidence_alignment: float

    def __post_init__(self):
        """Validate consensus metrics."""
        if not 0.0 <= self.consensus_strength <= 1.0:
            raise ValueError(f"Consensus strength must be between 0.0 and 1.0, got {self.consensus_strength}")

        if self.prediction_variance < 0.0:
            raise ValueError(f"Prediction variance cannot be negative, got {self.prediction_variance}")

    def is_high_consensus(self, threshold: float = 0.8) -> bool:
        """Check if consensus is high."""
        return self.consensus_strength >= threshold

    def is_diverse_ensemble(self, threshold: float = 0.6) -> bool:
        """Check if ensemble has good diversity."""
        return self.agent_diversity_score >= threshold


class BaseAgent(ABC):
    """Abstract base class defining standardized agent interface."""

    def __init__(self, agent_id: str, name: str, reasoning_style: ReasoningStyle,
                 knowledge_domains: List[str], configuration: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.name = name
        self.reasoning_style = reasoning_style
        self.knowledge_domains = knowledge_domains
        self.configuration = configuration or {}
        self.performance_history = []

    @abstractmethod
    async def conduct_research(self, question: Question) -> ResearchReport:
        """Conduct comprehensive research for the given question."""
        pass

    @abstractmethod
    async def generate_prediction(self, question: Question, research: ResearchReport) -> Dict[str, Any]:
        """Generate prediction based on question and research."""
        pass

    @abstractmethod
    async def forecast(self, question: Question) -> Forecast:
        """Complete forecasting workflow."""
        pass

    def get_specialization_score(self, question_category: str) -> float:
        """Calculate specialization score for a question category."""
        if question_category.lower() in [domain.lower() for domain in self.knowledge_domains]:
            return 0.9

        # Check for related domains
        related_domains = {
            'ai_development': ['technology', 'science'],
            'technology': ['ai_development', 'science'],
            'science': ['technology', 'ai_development', 'health'],
            'economics': ['politics', 'social'],
            'politics': ['economics', 'geopolitics', 'social'],
            'geopolitics': ['politics', 'economics']
        }

        category_lower = question_category.lower()
        if category_lower in related_domains:
            for domain in self.knowledge_domains:
                if domain.lower() in related_domains[category_lower]:
                    return 0.6

        return 0.3  # Base score for general knowledge

    def update_performance(self, accuracy: float, confidence_calibration: float):
        """Update agent's performance history."""
        self.performance_history.append({
            'timestamp': datetime.utcnow(),
            'accuracy': accuracy,
            'confidence_calibration': confidence_calibration
        })

        # Keep only last 100 records
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_recent_performance(self, days: int = 30) -> Dict[str, float]:
        """Get recent performance metrics."""
        cutoff_date = datetime.utcnow().timestamp() - (days * 24 * 3600)
        recent_records = [
            record for record in self.performance_history
            if record['timestamp'].timestamp() > cutoff_date
        ]

        if not recent_records:
            return {'accuracy': 0.5, 'confidence_calibration': 0.5, 'sample_size': 0}

        return {
            'accuracy': sum(r['accuracy'] for r in recent_records) / len(recent_records),
            'confidence_calibration': sum(r['confidence_calibration'] for r in recent_records) / len(recent_records),
            'sample_size': len(recent_records)
        }


class ChainOfThoughtAgent(BaseAgent):
    """Step-by-step reasoning agent with explicit thought chains."""

    def __init__(self, agent_id: str, name: str, knowledge_domains: List[str],
                 llm_client=None, configuration: Dict[str, Any] = None):
        super().__init__(agent_id, name, ReasoningStyle.CHAIN_OF_THOUGHT, knowledge_domains, configuration)
        self.llm_client = llm_client

    async def conduct_research(self, question: Question) -> ResearchReport:
        """Systematic research with documented reasoning steps."""
        # Placeholder implementation - would integrate with actual research pipeline
        return ResearchReport(
            id=uuid4(),
            question_id=question.id,
            sources=[],
            evidence_synthesis=f"Research conducted for question: {question.text[:100]}...",
            base_rates={},
            knowledge_gaps=[],
            research_quality_score=0.7,
            timestamp=datetime.utcnow()
        )

    async def generate_prediction(self, question: Question, research: ResearchReport) -> Dict[str, Any]:
        """Generate prediction using chain-of-thought reasoning."""
        reasoning_steps = []

        # Step 1: Problem understanding
        reasoning_steps.append(ReasoningStep.create(
            step_number=1,
            description="Understanding the question and context",
            input_data={"question": question.text, "background": question.background},
            output_data={"understanding": "Question analyzed and context established"},
            confidence_level=0.8,
            confidence_basis="Clear question structure and sufficient context",
            reasoning_type="analysis"
        ))

        # Step 2: Evidence evaluation
        reasoning_steps.append(ReasoningStep.create(
            step_number=2,
            description="Evaluating available evidence",
            input_data={"research": research.evidence_synthesis},
            output_data={"evidence_quality": research.research_quality_score},
            confidence_level=research.research_quality_score,
            confidence_basis=f"Research quality score: {research.research_quality_score}",
            reasoning_type="evaluation"
        ))

        # Step 3: Prediction generation
        if question.is_binary():
            prediction_value = 0.6  # Placeholder - would use actual reasoning
            confidence_level = 0.7
        elif question.is_numeric():
            prediction_value = 50.0  # Placeholder
            confidence_level = 0.6
        else:
            prediction_value = {"option_a": 0.4, "option_b": 0.6}  # Placeholder
            confidence_level = 0.65

        reasoning_steps.append(ReasoningStep.create(
            step_number=3,
            description="Generating final prediction",
            input_data={"evidence": research.evidence_synthesis},
            output_data={"prediction": prediction_value},
            confidence_level=confidence_level,
            confidence_basis="Based on systematic analysis of available evidence",
            reasoning_type="prediction"
        ))

        return {
            "prediction": prediction_value,
            "confidence": Confidence(level=confidence_level, basis="Chain-of-thought reasoning"),
            "reasoning_steps": reasoning_steps
        }

    async def forecast(self, question: Question) -> Forecast:
        """Complete forecasting workflow."""
        research = await self.conduct_research(question)
        prediction_result = await self.generate_prediction(question, research)

        return Forecast.create_binary(
            question_id=question.id,
            probability=prediction_result["prediction"] if question.is_binary() else 0.5,
            confidence_level=prediction_result["confidence"].level,
            confidence_basis=prediction_result["confidence"].basis,
            reasoning_trace=prediction_result["reasoning_steps"],
            evidence_sources=[],
            agent_id=self.agent_id
        )


class TreeOfThoughtAgent(BaseAgent):
    """Parallel reasoning paths agent exploring multiple solution branches."""

    def __init__(self, agent_id: str, name: str, knowledge_domains: List[str],
                 llm_client=None, configuration: Dict[str, Any] = None):
        super().__init__(agent_id, name, ReasoningStyle.TREE_OF_THOUGHT, knowledge_domains, configuration)
        self.llm_client = llm_client
        self.max_branches = configuration.get('max_branches', 3) if configuration else 3
        self.max_depth = configuration.get('max_depth', 3) if configuration else 3

    async def conduct_research(self, question: Question) -> ResearchReport:
        """Multi-perspective research exploration."""
        return ResearchReport(
            id=uuid4(),
            question_id=question.id,
            sources=[],
            evidence_synthesis=f"Multi-branch research for: {question.text[:100]}...",
            base_rates={},
            knowledge_gaps=[],
            research_quality_score=0.75,
            timestamp=datetime.utcnow()
        )

    async def explore_branches(self, question: Question, depth: int = 3) -> List[ReasoningBranch]:
        """Explore multiple reasoning paths in parallel."""
        branches = []

        for i in range(self.max_branches):
            branch_steps = []

            for step_num in range(1, depth + 1):
                branch_steps.append(ReasoningStep.create(
                    step_number=step_num,
                    description=f"Branch {i+1} reasoning step {step_num}",
                    input_data={"question": question.text, "branch": i+1},
                    output_data={"analysis": f"Branch {i+1} analysis at depth {step_num}"},
                    confidence_level=0.7 + (i * 0.05),  # Slight variation per branch
                    confidence_basis=f"Branch {i+1} reasoning path",
                    reasoning_type="tree_exploration"
                ))

            branch = ReasoningBranch(
                branch_id=f"branch_{i+1}",
                reasoning_steps=branch_steps,
                confidence=Confidence(level=0.7 + (i * 0.05), basis=f"Branch {i+1} confidence"),
                conclusion=f"Branch {i+1} conclusion based on {depth} reasoning steps"
            )
            branches.append(branch)

        return branches

    async def generate_prediction(self, question: Question, research: ResearchReport) -> Dict[str, Any]:
        """Generate prediction by exploring multiple reasoning branches."""
        branches = await self.explore_branches(question, self.max_depth)

        # Select best branch based on quality score
        best_branch = max(branches, key=lambda b: b.get_branch_quality_score())

        # Generate prediction based on best branch
        if question.is_binary():
            prediction_value = 0.65  # Placeholder
            confidence_level = best_branch.confidence.level
        elif question.is_numeric():
            prediction_value = 45.0  # Placeholder
            confidence_level = best_branch.confidence.level * 0.9
        else:
            prediction_value = {"option_a": 0.35, "option_b": 0.65}
            confidence_level = best_branch.confidence.level * 0.95

        return {
            "prediction": prediction_value,
            "confidence": Confidence(level=confidence_level, basis="Tree-of-thought best branch"),
            "reasoning_steps": best_branch.reasoning_steps,
            "branches_explored": len(branches),
            "best_branch_id": best_branch.branch_id
        }

    async def forecast(self, question: Question) -> Forecast:
        """Complete forecasting workflow with tree exploration."""
        research = await self.conduct_research(question)
        prediction_result = await self.generate_prediction(question, research)

        return Forecast.create_binary(
            question_id=question.id,
            probability=prediction_result["prediction"] if question.is_binary() else 0.5,
            confidence_level=prediction_result["confidence"].level,
            confidence_basis=prediction_result["confidence"].basis,
            reasoning_trace=prediction_result["reasoning_steps"],
            evidence_sources=[],
            agent_id=self.agent_id
        )


class ReActAgent(BaseAgent):
    """Reasoning + Acting agent with iterative refinement."""

    def __init__(self, agent_id: str, name: str, knowledge_domains: List[str],
                 llm_client=None, configuration: Dict[str, Any] = None):
        super().__init__(agent_id, name, ReasoningStyle.REACT, knowledge_domains, configuration)
        self.llm_client = llm_client
        self.max_iterations = configuration.get('max_iterations', 5) if configuration else 5

    async def conduct_research(self, question: Question) -> ResearchReport:
        """Iterative research with reasoning-action cycles."""
        return ResearchReport(
            id=uuid4(),
            question_id=question.id,
            sources=[],
            evidence_synthesis=f"ReAct research for: {question.text[:100]}...",
            base_rates={},
            knowledge_gaps=[],
            research_quality_score=0.8,
            timestamp=datetime.utcnow()
        )

    async def reason_act_cycle(self, question: Question, max_iterations: int = 5) -> ReasoningTrace:
        """Iterative reasoning and action cycle."""
        iterations = []
        reasoning_steps = []

        for i in range(max_iterations):
            # Reasoning phase
            reasoning_step = ReasoningStep.create(
                step_number=i * 2 + 1,
                description=f"Reasoning iteration {i+1}",
                input_data={"question": question.text, "iteration": i+1},
                output_data={"reasoning": f"Analysis for iteration {i+1}"},
                confidence_level=0.6 + (i * 0.05),
                confidence_basis=f"Iterative reasoning step {i+1}",
                reasoning_type="reasoning"
            )
            reasoning_steps.append(reasoning_step)

            # Action phase
            action_step = ReasoningStep.create(
                step_number=i * 2 + 2,
                description=f"Action iteration {i+1}",
                input_data={"reasoning": f"Analysis for iteration {i+1}"},
                output_data={"action": f"Action taken in iteration {i+1}"},
                confidence_level=0.65 + (i * 0.05),
                confidence_basis=f"Action based on reasoning step {i+1}",
                reasoning_type="action"
            )
            reasoning_steps.append(action_step)

            iterations.append({
                "iteration": i + 1,
                "reasoning": reasoning_step,
                "action": action_step,
                "convergence": 0.7 + (i * 0.05)
            })

            # Check for convergence
            if i > 0 and iterations[i]["convergence"] > 0.9:
                break

        convergence_score = iterations[-1]["convergence"] if iterations else 0.5

        return ReasoningTrace(
            iterations=iterations,
            final_reasoning=reasoning_steps,
            convergence_score=convergence_score
        )

    async def generate_prediction(self, question: Question, research: ResearchReport) -> Dict[str, Any]:
        """Generate prediction through reasoning-action cycles."""
        reasoning_trace = await self.reason_act_cycle(question, self.max_iterations)

        # Generate prediction based on final reasoning
        if question.is_binary():
            prediction_value = 0.55 + (reasoning_trace.convergence_score * 0.2)
            confidence_level = reasoning_trace.convergence_score
        elif question.is_numeric():
            prediction_value = 40.0 + (reasoning_trace.convergence_score * 20)
            confidence_level = reasoning_trace.convergence_score * 0.9
        else:
            base_prob = 0.3 + (reasoning_trace.convergence_score * 0.2)
            prediction_value = {"option_a": base_prob, "option_b": 1.0 - base_prob}
            confidence_level = reasoning_trace.convergence_score * 0.95

        return {
            "prediction": prediction_value,
            "confidence": Confidence(level=confidence_level, basis="ReAct iterative refinement"),
            "reasoning_steps": reasoning_trace.final_reasoning,
            "iterations": reasoning_trace.get_iteration_count(),
            "convergence_score": reasoning_trace.convergence_score
        }

    async def forecast(self, question: Question) -> Forecast:
        """Complete forecasting workflow with ReAct cycles."""
        research = await self.conduct_research(question)
        prediction_result = await self.generate_prediction(question, research)

        return Forecast.create_binary(
            question_id=question.id,
            probability=prediction_result["prediction"] if question.is_binary() else 0.5,
            confidence_level=prediction_result["confidence"].level,
            confidence_basis=prediction_result["confidence"].basis,
            reasoning_trace=prediction_result["reasoning_steps"],
            evidence_sources=[],
            agent_id=self.agent_id
        )


class AutoCoTAgent(BaseAgent):
    """Automatic chain-of-thought agent with adaptive reasoning."""

    def __init__(self, agent_id: str, name: str, knowledge_domains: List[str],
                 llm_client=None, configuration: Dict[str, Any] = None):
        super().__init__(agent_id, name, ReasoningStyle.CHAIN_OF_THOUGHT, knowledge_domains, configuration)
        self.llm_client = llm_client
        self.adaptive_depth = configuration.get('adaptive_depth', True) if configuration else True

    async def conduct_research(self, question: Question) -> ResearchReport:
        """Adaptive research based on question complexity."""
        complexity_score = question.get_complexity_score()
        research_depth = min(int(complexity_score * 3), 5)  # Scale research depth

        return ResearchReport(
            id=uuid4(),
            question_id=question.id,
            sources=[],
            evidence_synthesis=f"AutoCoT research (depth: {research_depth}) for: {question.text[:100]}...",
            base_rates={},
            knowledge_gaps=[],
            research_quality_score=0.7 + (research_depth * 0.05),
            timestamp=datetime.utcnow()
        )

    async def generate_prediction(self, question: Question, research: ResearchReport) -> Dict[str, Any]:
        """Generate prediction with automatic chain-of-thought adaptation."""
        complexity_score = question.get_complexity_score()
        reasoning_depth = max(2, min(int(complexity_score * 4), 6))

        reasoning_steps = []

        for i in range(reasoning_depth):
            step_confidence = 0.7 + (i * 0.03) - (complexity_score * 0.1)
            step_confidence = max(0.4, min(0.95, step_confidence))

            reasoning_steps.append(ReasoningStep.create(
                step_number=i + 1,
                description=f"AutoCoT reasoning step {i+1} (adaptive depth: {reasoning_depth})",
                input_data={"question": question.text, "complexity": complexity_score},
                output_data={"analysis": f"Adaptive analysis step {i+1}"},
                confidence_level=step_confidence,
                confidence_basis=f"Adaptive reasoning step {i+1} based on complexity {complexity_score:.2f}",
                reasoning_type="adaptive_cot"
            ))

        # Generate prediction based on adaptive reasoning
        base_confidence = sum(step.confidence.level for step in reasoning_steps) / len(reasoning_steps)

        if question.is_binary():
            prediction_value = 0.5 + (base_confidence - 0.5) * 0.4
            confidence_level = base_confidence
        elif question.is_numeric():
            prediction_value = 35.0 + (base_confidence * 30)
            confidence_level = base_confidence * 0.9
        else:
            base_prob = 0.25 + (base_confidence * 0.3)
            prediction_value = {"option_a": base_prob, "option_b": 1.0 - base_prob}
            confidence_level = base_confidence * 0.95

        return {
            "prediction": prediction_value,
            "confidence": Confidence(level=confidence_level, basis="AutoCoT adaptive reasoning"),
            "reasoning_steps": reasoning_steps,
            "adaptive_depth": reasoning_depth,
            "complexity_score": complexity_score
        }

    async def forecast(self, question: Question) -> Forecast:
        """Complete forecasting workflow with adaptive reasoning."""
        research = await self.conduct_research(question)
        prediction_result = await self.generate_prediction(question, research)

        return Forecast.create_binary(
            question_id=question.id,
            probability=prediction_result["prediction"] if question.is_binary() else 0.5,
            confidence_level=prediction_result["confidence"].level,
            confidence_basis=prediction_result["confidence"].basis,
            reasoning_trace=prediction_result["reasoning_steps"],
            evidence_sources=[],
            agent_id=self.agent_id
        )


class EnsembleAgent:
    """Orchestrates multiple agents and aggregates their predictions."""

    def __init__(self, agents: List[BaseAgent], aggregation_method: AggregationMethod = AggregationMethod.CONFIDENCE_WEIGHTED):
        if not agents:
            raise ValueError("Ensemble must have at least one agent")

        self.agents = agents
        self.aggregation_method = aggregation_method
        self.agent_weights = {agent.agent_id: 1.0 for agent in agents}
        self.performance_tracker = defaultdict(list)

    async def generate_ensemble_forecast(self, question: Question) -> Forecast:
        """Generate ensemble forecast from multiple agents."""
        # Get forecasts from all agents
        agent_forecasts = []
        for agent in self.agents:
            try:
                forecast = await agent.forecast(question)
                agent_forecasts.append((agent, forecast))
            except Exception as e:
                # Log error and continue with other agents
                print(f"Agent {agent.agent_id} failed: {e}")
                continue

        if not agent_forecasts:
            raise RuntimeError("All agents failed to generate forecasts")

        # Aggregate predictions
        aggregated_prediction = await self.aggregate_predictions(
            [forecast for _, forecast in agent_forecasts], question
        )

        # Calculate consensus metrics
        consensus_metrics = self.calculate_consensus_metrics(
            [forecast for _, forecast in agent_forecasts]
        )

        # Combine reasoning traces
        all_reasoning_steps = []
        for agent, forecast in agent_forecasts:
            for step in forecast.reasoning_trace:
                # Add agent context to reasoning step
                enhanced_step = ReasoningStep.create(
                    step_number=len(all_reasoning_steps) + 1,
                    description=f"[{agent.name}] {step.description}",
                    input_data=step.input_data,
                    output_data=step.output_data,
                    confidence_level=step.confidence.level,
                    confidence_basis=f"[{agent.name}] {step.confidence.basis}",
                    reasoning_type=step.reasoning_type
                )
                all_reasoning_steps.append(enhanced_step)

        # Create ensemble forecast
        if question.is_binary():
            return Forecast.create_binary(
                question_id=question.id,
                probability=aggregated_prediction["value"],
                confidence_level=aggregated_prediction["confidence"],
                confidence_basis=f"Ensemble of {len(agent_forecasts)} agents using {self.aggregation_method.value}",
                reasoning_trace=all_reasoning_steps,
                evidence_sources=[],
                agent_id="ensemble"
            )
        elif question.is_numeric():
            return Forecast.create_numeric(
                question_id=question.id,
                value=aggregated_prediction["value"],
                confidence_level=aggregated_prediction["confidence"],
                confidence_basis=f"Ensemble of {len(agent_forecasts)} agents using {self.aggregation_method.value}",
                reasoning_trace=all_reasoning_steps,
                evidence_sources=[],
                agent_id="ensemble"
            )
        else:
            return Forecast.create_multiple_choice(
                question_id=question.id,
                choice_probabilities=aggregated_prediction["value"],
                confidence_level=aggregated_prediction["confidence"],
                confidence_basis=f"Ensemble of {len(agent_forecasts)} agents using {self.aggregation_method.value}",
                reasoning_trace=all_reasoning_steps,
                evidence_sources=[],
                agent_id="ensemble"
            )

    async def aggregate_predictions(self, forecasts: List[Forecast], question: Question) -> Dict[str, Any]:
        """Aggregate predictions using the specified method."""
        if not forecasts:
            raise ValueError("Cannot aggregate empty forecast list")

        if self.aggregation_method == AggregationMethod.SIMPLE_AVERAGE:
            return self._simple_average(forecasts, question)
        elif self.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(forecasts, question)
        elif self.aggregation_method == AggregationMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_average(forecasts, question)
        elif self.aggregation_method == AggregationMethod.MEDIAN:
            return self._median_aggregation(forecasts, question)
        elif self.aggregation_method == AggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean(forecasts, question)
        elif self.aggregation_method == AggregationMethod.META_REASONING:
            return await self._meta_reasoning_aggregation(forecasts, question)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def _simple_average(self, forecasts: List[Forecast], question: Question) -> Dict[str, Any]:
        """Simple average of all predictions."""
        if question.is_binary():
            values = [f.get_binary_probability() for f in forecasts]
            avg_value = sum(values) / len(values)
            avg_confidence = sum(f.confidence.level for f in forecasts) / len(forecasts)

            return {
                "value": avg_value,
                "confidence": avg_confidence
            }
        elif question.is_numeric():
            values = [f.get_numeric_value() for f in forecasts]
            avg_value = sum(values) / len(values)
            avg_confidence = sum(f.confidence.level for f in forecasts) / len(forecasts)

            return {
                "value": avg_value,
                "confidence": avg_confidence
            }
        else:
            # Multiple choice - average probabilities for each choice
            all_choices = set()
            for f in forecasts:
                all_choices.update(f.get_choice_probabilities().keys())

            avg_probs = {}
            for choice in all_choices:
                probs = [f.get_choice_probabilities().get(choice, 0.0) for f in forecasts]
                avg_probs[choice] = sum(probs) / len(probs)

            # Normalize probabilities
            total_prob = sum(avg_probs.values())
            if total_prob > 0:
                avg_probs = {k: v / total_prob for k, v in avg_probs.items()}

            avg_confidence = sum(f.confidence.level for f in forecasts) / len(forecasts)

            return {
                "value": avg_probs,
                "confidence": avg_confidence
            }

    def _confidence_weighted_average(self, forecasts: List[Forecast], question: Question) -> Dict[str, Any]:
        """Confidence-weighted average of predictions."""
        weights = [f.confidence.level for f in forecasts]
        total_weight = sum(weights)

        if total_weight == 0:
            return self._simple_average(forecasts, question)

        if question.is_binary():
            values = [f.get_binary_probability() for f in forecasts]
            weighted_value = sum(v * w for v, w in zip(values, weights)) / total_weight
            weighted_confidence = sum(f.confidence.level * w for f, w in zip(forecasts, weights)) / total_weight

            return {
                "value": weighted_value,
                "confidence": weighted_confidence
            }
        elif question.is_numeric():
            values = [f.get_numeric_value() for f in forecasts]
            weighted_value = sum(v * w for v, w in zip(values, weights)) / total_weight
            weighted_confidence = sum(f.confidence.level * w for f, w in zip(forecasts, weights)) / total_weight

            return {
                "value": weighted_value,
                "confidence": weighted_confidence
            }
        else:
            # Multiple choice - weighted average of probabilities
            all_choices = set()
            for f in forecasts:
                all_choices.update(f.get_choice_probabilities().keys())

            weighted_probs = {}
            for choice in all_choices:
                weighted_sum = sum(
                    f.get_choice_probabilities().get(choice, 0.0) * w
                    for f, w in zip(forecasts, weights)
                )
                weighted_probs[choice] = weighted_sum / total_weight

            # Normalize probabilities
            total_prob = sum(weighted_probs.values())
            if total_prob > 0:
                weighted_probs = {k: v / total_prob for k, v in weighted_probs.items()}

            weighted_confidence = sum(f.confidence.level * w for f, w in zip(forecasts, weights)) / total_weight

            return {
                "value": weighted_probs,
                "confidence": weighted_confidence
            }

    def _weighted_average(self, forecasts: List[Forecast], question: Question) -> Dict[str, Any]:
        """Weighted average using agent performance weights."""
        weights = [self.agent_weights.get(f.agent_id, 1.0) for f in forecasts]
        total_weight = sum(weights)

        if total_weight == 0:
            return self._simple_average(forecasts, question)

        if question.is_binary():
            values = [f.get_binary_probability() for f in forecasts]
            weighted_value = sum(v * w for v, w in zip(values, weights)) / total_weight
            weighted_confidence = sum(f.confidence.level * w for f, w in zip(forecasts, weights)) / total_weight

            return {
                "value": weighted_value,
                "confidence": weighted_confidence
            }
        # Similar implementation for numeric and multiple choice...
        else:
            return self._simple_average(forecasts, question)  # Fallback

    def _median_aggregation(self, forecasts: List[Forecast], question: Question) -> Dict[str, Any]:
        """Median aggregation of predictions."""
        if question.is_binary():
            values = [f.get_binary_probability() for f in forecasts]
            median_value = statistics.median(values)
            median_confidence = statistics.median([f.confidence.level for f in forecasts])

            return {
                "value": median_value,
                "confidence": median_confidence
            }
        elif question.is_numeric():
            values = [f.get_numeric_value() for f in forecasts]
            median_value = statistics.median(values)
            median_confidence = statistics.median([f.confidence.level for f in forecasts])

            return {
                "value": median_value,
                "confidence": median_confidence
            }
        else:
            # For multiple choice, use mode or fallback to average
            return self._simple_average(forecasts, question)

    def _trimmed_mean(self, forecasts: List[Forecast], question: Question, trim_percent: float = 0.2) -> Dict[str, Any]:
        """Trimmed mean aggregation (removes outliers)."""
        if len(forecasts) < 3:
            return self._simple_average(forecasts, question)

        trim_count = max(1, int(len(forecasts) * trim_percent))

        if question.is_binary():
            values = sorted([f.get_binary_probability() for f in forecasts])
            trimmed_values = values[trim_count:-trim_count] if trim_count > 0 else values

            confidences = sorted([f.confidence.level for f in forecasts])
            trimmed_confidences = confidences[trim_count:-trim_count] if trim_count > 0 else confidences

            return {
                "value": sum(trimmed_values) / len(trimmed_values),
                "confidence": sum(trimmed_confidences) / len(trimmed_confidences)
            }
        # Similar for other types...
        else:
            return self._simple_average(forecasts, question)  # Fallback

    async def _meta_reasoning_aggregation(self, forecasts: List[Forecast], question: Question) -> Dict[str, Any]:
        """Meta-reasoning aggregation considering reasoning quality."""
        # Analyze reasoning quality for each forecast
        reasoning_scores = []
        for forecast in forecasts:
            score = self._evaluate_reasoning_quality(forecast.reasoning_trace)
            reasoning_scores.append(score)

        # Weight by reasoning quality
        total_score = sum(reasoning_scores)
        if total_score == 0:
            return self._simple_average(forecasts, question)

        weights = [score / total_score for score in reasoning_scores]

        if question.is_binary():
            values = [f.get_binary_probability() for f in forecasts]
            weighted_value = sum(v * w for v, w in zip(values, weights))
            weighted_confidence = sum(f.confidence.level * w for f, w in zip(forecasts, weights))

            return {
                "value": weighted_value,
                "confidence": weighted_confidence
            }
        # Similar for other types...
        else:
            return self._simple_average(forecasts, question)  # Fallback

    def _evaluate_reasoning_quality(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Evaluate the quality of reasoning steps."""
        if not reasoning_steps:
            return 0.1

        # Factors: step count, confidence levels, reasoning coherence
        step_count_score = min(len(reasoning_steps) / 5.0, 1.0)  # Normalize by expected steps
        avg_confidence = sum(step.confidence.level for step in reasoning_steps) / len(reasoning_steps)

        # Simple coherence check - steps should build on each other
        coherence_score = 0.8  # Placeholder - would implement actual coherence analysis

        return (step_count_score + avg_confidence + coherence_score) / 3.0

    def calculate_consensus_metrics(self, forecasts: List[Forecast]) -> ConsensusMetrics:
        """Calculate consensus quality metrics for the ensemble."""
        if not forecasts:
            return ConsensusMetrics(
                consensus_strength=0.0,
                prediction_variance=1.0,
                agent_diversity_score=0.0,
                confidence_alignment=0.0
            )

        # Extract predictions for analysis
        predictions = []
        confidences = [f.confidence.level for f in forecasts]

        # Get prediction values based on first forecast type
        first_forecast = forecasts[0]
        if first_forecast.is_binary_forecast():
            predictions = [f.get_binary_probability() for f in forecasts]
        elif first_forecast.is_numeric_forecast():
            predictions = [f.get_numeric_value() for f in forecasts]
        else:
            # For multiple choice, use entropy or most likely choice probability
            predictions = [max(f.get_choice_probabilities().values()) for f in forecasts]

        # Calculate consensus strength (inverse of variance)
        if len(predictions) > 1:
            prediction_variance = statistics.variance(predictions)
            consensus_strength = 1.0 / (1.0 + prediction_variance)
        else:
            prediction_variance = 0.0
            consensus_strength = 1.0

        # Calculate agent diversity (based on reasoning styles)
        reasoning_styles = set()
        for forecast in forecasts:
            # Extract reasoning style from agent_id or reasoning trace
            if hasattr(forecast, 'agent_id') and forecast.agent_id:
                reasoning_styles.add(forecast.agent_id)

        agent_diversity_score = len(reasoning_styles) / len(forecasts) if forecasts else 0.0

        # Calculate confidence alignment
        if len(confidences) > 1:
            confidence_variance = statistics.variance(confidences)
            confidence_alignment = 1.0 / (1.0 + confidence_variance)
        else:
            confidence_alignment = 1.0

        return ConsensusMetrics(
            consensus_strength=consensus_strength,
            prediction_variance=prediction_variance,
            agent_diversity_score=agent_diversity_score,
            confidence_alignment=confidence_alignment
        )

    def update_agent_weights(self, performance_data: Dict[str, Dict[str, float]]):
        """Update agent weights based on performance data."""
        for agent_id, performance in performance_data.items():
            if agent_id in self.agent_weights:
                # Weight based on accuracy and calibration
                accuracy = performance.get('accuracy', 0.5)
                calibration = performance.get('calibration', 0.5)
                sample_size = performance.get('sample_size', 1)

                # Combine metrics with sample size weighting
                weight = (accuracy + calibration) / 2.0
                weight = weight * min(sample_size / 10.0, 1.0)  # Scale by sample size

                self.agent_weights[agent_id] = max(0.1, min(2.0, weight))  # Clamp weights


class AgentOrchestrator:
    """High-level orchestrator for managing multiple agents and ensembles."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.ensembles: Dict[str, EnsembleAgent] = {}
        self.performance_tracker = defaultdict(list)

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent

    def create_ensemble(self, ensemble_id: str, agent_ids: List[str],
                       aggregation_method: AggregationMethod = AggregationMethod.CONFIDENCE_WEIGHTED) -> EnsembleAgent:
        """Create an ensemble from registered agents."""
        agents = []
        for agent_id in agent_ids:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not registered")
            agents.append(self.agents[agent_id])

        ensemble = EnsembleAgent(agents, aggregation_method)
        self.ensembles[ensemble_id] = ensemble
        return ensemble

    def get_best_agents_for_question(self, question: Question, max_agents: int = 3) -> List[BaseAgent]:
        """Select best agents for a specific question based on specialization and performance."""
        agent_scores = []

        for agent in self.agents.values():
            # Calculate specialization score
            specialization_score = agent.get_specialization_score(question.category.value)

            # Get recent performance
            performance = agent.get_recent_performance()
            performance_score = (performance['accuracy'] + performance['confidence_calibration']) / 2.0

            # Weight by sample size
            sample_weight = min(performance['sample_size'] / 10.0, 1.0)

            # Combined score
            combined_score = (specialization_score * 0.4 + performance_score * 0.6) * sample_weight

            agent_scores.append((agent, combined_score))

        # Sort by score and return top agents
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in agent_scores[:max_agents]]

    async def generate_optimal_forecast(self, question: Question) -> Forecast:
        """Generate optimal forecast using best available agents."""
        # Select best agents for this question
        best_agents = self.get_best_agents_for_question(question)

        if not best_agents:
            raise RuntimeError("No suitable agents available")

        # Create temporary ensemble
        temp_ensemble = EnsembleAgent(best_agents, AggregationMethod.CONFIDENCE_WEIGHTED)

        # Generate ensemble forecast
        return await temp_ensemble.generate_ensemble_forecast(question)

    def update_performance_tracking(self, agent_id: str, question_id: int,
                                  accuracy: float, confidence_calibration: float):
        """Update performance tracking for an agent."""
        if agent_id in self.agents:
            self.agents[agent_id].update_performance(accuracy, confidence_calibration)

            # Also update ensemble weights
            for ensemble in self.ensembles.values():
                if any(agent.agent_id == agent_id for agent in ensemble.agents):
                    performance_data = {
                        agent_id: {
                            'accuracy': accuracy,
                            'calibration': confidence_calibration,
                            'sample_size': len(self.agents[agent_id].performance_history)
                        }
                    }
                    ensemble.update_agent_weights(performance_data)
