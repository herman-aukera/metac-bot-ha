"""
Enhanced Tree of Thought reasoning agent with systematic exploration.

This implementation provides:
- Parallel reasoning path exploration with configurable breadth/depth
- Systematic sub-component analysis and problem decomposition
- Advanced reasoning path evaluation and selection mechanisms
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import structlog

from ..domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
)
from ..domain.entities.question import Question
from ..domain.entities.research_report import ResearchReport
from ..domain.value_objects.reasoning_trace import (
    ReasoningStep,
    ReasoningStepType,
    ReasoningTrace,
)
from ..infrastructure.external_apis.llm_client import LLMClient
from ..infrastructure.external_apis.search_client import SearchClient
from .base_agent import BaseAgent

logger = structlog.get_logger(__name__)


class ReasoningPathType(Enum):
    """Types of reasoning paths in the tree."""

    ANALYTICAL = "analytical"
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    HISTORICAL = "historical"
    SYSTEMATIC = "systematic"


class PathEvaluationCriteria(Enum):
    """Criteria for evaluating reasoning paths."""

    LOGICAL_COHERENCE = "logical_coherence"
    EVIDENCE_STRENGTH = "evidence_strength"
    NOVELTY = "novelty"
    COMPLETENESS = "completeness"
    ACCURACY_POTENTIAL = "accuracy_potential"
    UNCERTAINTY_HANDLING = "uncertainty_handling"


@dataclass
class ReasoningPath:
    """Represents a complete reasoning path in the tree."""

    id: UUID = field(default_factory=uuid4)
    path_type: ReasoningPathType = ReasoningPathType.ANALYTICAL
    steps: List[ReasoningStep] = field(default_factory=list)
    depth: int = 0
    parent_path_id: Optional[UUID] = None
    sub_components: List[str] = field(default_factory=list)
    confidence: float = 0.5
    evaluation_scores: Dict[PathEvaluationCriteria, float] = field(default_factory=dict)
    is_complete: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to this path."""
        self.steps.append(step)
        self.depth = len(self.steps)

    def get_overall_score(self) -> float:
        """Calculate overall score from evaluation criteria."""
        if not self.evaluation_scores:
            return self.confidence

        # Weight different criteria
        weights = {
            PathEvaluationCriteria.LOGICAL_COHERENCE: 0.25,
            PathEvaluationCriteria.EVIDENCE_STRENGTH: 0.20,
            PathEvaluationCriteria.ACCURACY_POTENTIAL: 0.20,
            PathEvaluationCriteria.COMPLETENESS: 0.15,
            PathEvaluationCriteria.NOVELTY: 0.10,
            PathEvaluationCriteria.UNCERTAINTY_HANDLING: 0.10,
        }

        # Calculate weighted score only for criteria that exist
        total_weight = 0.0
        weighted_score = 0.0

        for criteria, score in self.evaluation_scores.items():
            weight = weights.get(criteria, 0.1)
            weighted_score += score * weight
            total_weight += weight

        # Normalize by actual total weight used
        if total_weight > 0:
            return min(1.0, weighted_score / total_weight)
        else:
            return self.confidence

    def get_reasoning_summary(self) -> str:
        """Get a summary of the reasoning in this path."""
        if not self.steps:
            return "Empty reasoning path"

        summary_parts = [f"Path Type: {self.path_type.value}"]

        if self.sub_components:
            summary_parts.append(f"Sub-components: {', '.join(self.sub_components)}")

        summary_parts.append("Key reasoning steps:")
        for i, step in enumerate(self.steps[:3]):  # Show first 3 steps
            summary_parts.append(f"{i+1}. {step.content[:100]}...")

        if len(self.steps) > 3:
            summary_parts.append(f"... and {len(self.steps) - 3} more steps")

        return "\n".join(summary_parts)


@dataclass
class TreeExplorationConfig:
    """Configuration for tree exploration parameters."""

    max_depth: int = 4
    max_breadth: int = 3
    max_parallel_paths: int = 6
    evaluation_threshold: float = 0.6
    path_selection_top_k: int = 2
    enable_sub_component_analysis: bool = True
    enable_parallel_exploration: bool = True
    reasoning_path_types: List[ReasoningPathType] = field(
        default_factory=lambda: [
            ReasoningPathType.ANALYTICAL,
            ReasoningPathType.EMPIRICAL,
            ReasoningPathType.PROBABILISTIC,
        ]
    )


class TreeOfThoughtAgent(BaseAgent):
    """
    Enhanced Tree of Thought agent with systematic exploration capabilities.

    Features:
    - Parallel reasoning path exploration with configurable breadth/depth
    - Systematic sub-component analysis and problem decomposition
    - Advanced reasoning path evaluation and selection mechanisms
    - Integration with reasoning orchestrator for bias detection
    """

    def __init__(
        self,
        name: str,
        model_config: Dict[str, Any],
        llm_client: LLMClient,
        search_client: Optional[SearchClient] = None,
        exploration_config: Optional[TreeExplorationConfig] = None,
    ):
        super().__init__(name, model_config)
        self.llm_client = llm_client
        self.search_client = search_client
        self.exploration_config = exploration_config or TreeExplorationConfig()
        self.reasoning_paths: Dict[UUID, ReasoningPath] = {}

    async def conduct_research(
        self, question: Question, search_config: Optional[Dict[str, Any]] = None
    ) -> ResearchReport:
        """Conduct research using systematic exploration approach."""
        self.logger.info("Starting systematic research", question_id=str(question.id))

        if not self.search_client:
            return ResearchReport.create_new(
                question_id=question.id,
                title=f"Research for: {question.title}",
                executive_summary="No research conducted - search client not available",
                detailed_analysis="No detailed analysis available",
                sources=[],
                created_by=self.name,
            )

        try:
            # Decompose question into sub-components for targeted research
            sub_components = await self._decompose_question(question)

            # Conduct research for each sub-component
            research_tasks = []
            for component in sub_components[:5]:  # Limit to 5 components
                research_tasks.append(self._research_component(component, question))

            component_results = await asyncio.gather(
                *research_tasks, return_exceptions=True
            )

            # Aggregate research results
            all_sources = []
            component_summaries = []

            for i, result in enumerate(component_results):
                if isinstance(result, Exception):
                    self.logger.warning(
                        f"Research failed for component {i}", error=str(result)
                    )
                    continue

                sources, summary = result
                all_sources.extend(sources)
                component_summaries.append(f"{sub_components[i]}: {summary}")

            executive_summary = f"Systematic research conducted on {len(sub_components)} sub-components: {', '.join(sub_components)}"
            detailed_analysis = "\n\n".join(component_summaries)

            return ResearchReport.create_new(
                question_id=question.id,
                title=f"Systematic Research: {question.title}",
                executive_summary=executive_summary,
                detailed_analysis=detailed_analysis,
                sources=all_sources[:20],  # Limit sources
                created_by=self.name,
                key_factors=sub_components,
            )

        except Exception as e:
            self.logger.error("Systematic research failed", error=str(e))
            return ResearchReport.create_new(
                question_id=question.id,
                title=f"Research for: {question.title}",
                executive_summary=f"Research failed: {str(e)}",
                detailed_analysis="Research could not be completed due to error",
                sources=[],
                created_by=self.name,
            )

    async def generate_prediction(
        self, question: Question, research_report: ResearchReport
    ) -> Prediction:
        """Generate prediction using systematic tree exploration."""
        self.logger.info(
            "Starting Tree of Thought prediction", question_id=str(question.id)
        )

        try:
            # Initialize reasoning tree
            await self._initialize_reasoning_tree(question, research_report)

            # Explore reasoning paths systematically
            await self._explore_reasoning_tree(question, research_report)

            # Evaluate and select best paths
            best_paths = await self._evaluate_and_select_paths()

            # Synthesize final prediction
            prediction = await self._synthesize_prediction(
                question, research_report, best_paths
            )

            self.logger.info(
                "Tree of Thought prediction completed",
                question_id=str(question.id),
                paths_explored=len(self.reasoning_paths),
                best_paths_used=len(best_paths),
            )

            return prediction

        except Exception as e:
            self.logger.error("Tree of Thought prediction failed", error=str(e))
            raise

    async def _decompose_question(self, question: Question) -> List[str]:
        """Decompose question into sub-components for systematic analysis."""
        decomposition_prompt = f"""
        Analyze this forecasting question and decompose it into key sub-components that need to be analyzed separately:

        Question: {question.title}
        Description: {question.description or 'No description provided'}
        Type: {question.question_type.value}

        Identify 3-5 key sub-components or aspects that should be analyzed to answer this question effectively.
        Each component should be a specific, analyzable aspect of the main question.

        Format your response as a simple list:
        1. [Component 1]
        2. [Component 2]
        3. [Component 3]
        etc.
        """

        response = await self.llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at breaking down complex questions into analyzable components.",
                },
                {"role": "user", "content": decomposition_prompt},
            ],
            temperature=0.3,
        )

        # Parse response to extract components
        components = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("*")
            ):
                # Remove numbering/bullets and extract component
                component = line.split(".", 1)[-1].strip()
                component = component.lstrip("- *").strip()
                if component:
                    components.append(component)

        return components[:5]  # Limit to 5 components

    async def _research_component(
        self, component: str, question: Question
    ) -> Tuple[List[Any], str]:
        """Research a specific component of the question."""
        try:
            search_query = f"{component} {question.title}"
            search_results = await self.search_client.search(search_query)

            # Convert search results to sources (simplified)
            sources = []
            for result in search_results[:3]:  # Limit per component
                sources.append(
                    {
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "summary": result.get("snippet", ""),
                        "component": component,
                    }
                )

            summary = f"Found {len(sources)} sources related to {component}"
            return sources, summary

        except Exception as e:
            self.logger.warning(
                f"Component research failed for {component}", error=str(e)
            )
            return [], f"Research failed for {component}: {str(e)}"

    async def _initialize_reasoning_tree(
        self, question: Question, research_report: ResearchReport
    ) -> None:
        """Initialize the reasoning tree with diverse initial paths."""
        self.reasoning_paths.clear()

        # Create initial reasoning paths for different types
        initialization_tasks = []
        for path_type in self.exploration_config.reasoning_path_types:
            initialization_tasks.append(
                self._create_initial_reasoning_path(
                    question, research_report, path_type
                )
            )

        initial_paths = await asyncio.gather(
            *initialization_tasks, return_exceptions=True
        )

        for path in initial_paths:
            if isinstance(path, Exception):
                self.logger.warning("Failed to create initial path", error=str(path))
                continue

            self.reasoning_paths[path.id] = path

        self.logger.info(
            f"Initialized reasoning tree with {len(self.reasoning_paths)} paths"
        )

    async def _create_initial_reasoning_path(
        self,
        question: Question,
        research_report: ResearchReport,
        path_type: ReasoningPathType,
    ) -> ReasoningPath:
        """Create an initial reasoning path of a specific type."""
        path = ReasoningPath(path_type=path_type)

        # Add sub-components based on research
        if research_report.key_factors:
            path.sub_components = research_report.key_factors[:3]

        # Create initial reasoning step based on path type
        initial_step = await self._generate_initial_step(
            question, research_report, path_type
        )
        path.add_step(initial_step)

        return path

    async def _generate_initial_step(
        self,
        question: Question,
        research_report: ResearchReport,
        path_type: ReasoningPathType,
    ) -> ReasoningStep:
        """Generate initial reasoning step for a specific path type."""
        path_prompts = {
            ReasoningPathType.ANALYTICAL: "Analyze this question using logical decomposition and systematic reasoning",
            ReasoningPathType.EMPIRICAL: "Approach this question by examining empirical evidence and data patterns",
            ReasoningPathType.THEORETICAL: "Apply relevant theories and models to understand this question",
            ReasoningPathType.COMPARATIVE: "Compare this situation to similar historical cases or analogies",
            ReasoningPathType.CAUSAL: "Identify causal relationships and mechanisms relevant to this question",
            ReasoningPathType.PROBABILISTIC: "Apply probabilistic reasoning and statistical thinking",
            ReasoningPathType.HISTORICAL: "Examine historical patterns and trends relevant to this question",
            ReasoningPathType.SYSTEMATIC: "Use systematic methodology to break down and analyze this question",
        }

        prompt = f"""
        {path_prompts.get(path_type, "Analyze this question systematically")}:

        Question: {question.title}
        Description: {question.description or 'No description provided'}

        Research Summary: {research_report.executive_summary}

        Provide your initial reasoning step for this {path_type.value} approach.
        Focus on the specific methodology and initial insights this approach would provide.
        """

        response = await self.llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert using {path_type.value} reasoning approach.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        )

        return ReasoningStep.create(
            step_type=ReasoningStepType.ANALYSIS,
            content=response.strip(),
            confidence=0.6,
            metadata={"path_type": path_type.value, "step_number": 1},
        )

    async def _explore_reasoning_tree(
        self, question: Question, research_report: ResearchReport
    ) -> None:
        """Systematically explore the reasoning tree through multiple iterations."""
        for depth in range(1, self.exploration_config.max_depth):
            self.logger.info(f"Exploring reasoning tree at depth {depth}")

            # Get paths that can be expanded
            expandable_paths = [
                path
                for path in self.reasoning_paths.values()
                if not path.is_complete
                and path.depth < self.exploration_config.max_depth
            ]

            if not expandable_paths:
                break

            # Evaluate current paths
            await self._evaluate_reasoning_paths(expandable_paths)

            # Select promising paths for expansion
            selected_paths = self._select_paths_for_expansion(expandable_paths)

            if not selected_paths:
                break

            # Expand selected paths
            if self.exploration_config.enable_parallel_exploration:
                expansion_tasks = [
                    self._expand_reasoning_path(path, question, research_report)
                    for path in selected_paths
                ]
                await asyncio.gather(*expansion_tasks, return_exceptions=True)
            else:
                for path in selected_paths:
                    await self._expand_reasoning_path(path, question, research_report)

            # Limit total number of paths
            if (
                len(self.reasoning_paths)
                > self.exploration_config.max_parallel_paths * 2
            ):
                await self._prune_reasoning_paths()

    async def _evaluate_reasoning_paths(self, paths: List[ReasoningPath]) -> None:
        """Evaluate reasoning paths against multiple criteria."""
        evaluation_tasks = []
        for path in paths:
            evaluation_tasks.append(self._evaluate_single_path(path))

        await asyncio.gather(*evaluation_tasks, return_exceptions=True)

    async def _evaluate_single_path(self, path: ReasoningPath) -> None:
        """Evaluate a single reasoning path."""
        if not path.steps:
            return

        # Get reasoning content for evaluation
        reasoning_content = "\n".join([step.content for step in path.steps])

        evaluation_prompt = f"""
        Evaluate this reasoning path on the following criteria (score 0-1 for each):

        Reasoning Path ({path.path_type.value}):
        {reasoning_content}

        Criteria:
        1. LOGICAL_COHERENCE: How logically consistent and well-structured is this reasoning?
        2. EVIDENCE_STRENGTH: How well does this reasoning use and integrate evidence?
        3. NOVELTY: How novel or insightful are the perspectives in this reasoning?
        4. COMPLETENESS: How complete and thorough is this reasoning approach?
        5. ACCURACY_POTENTIAL: How likely is this reasoning to lead to accurate predictions?
        6. UNCERTAINTY_HANDLING: How well does this reasoning handle uncertainty and limitations?

        Provide scores in this format:
        LOGICAL_COHERENCE: 0.X
        EVIDENCE_STRENGTH: 0.X
        NOVELTY: 0.X
        COMPLETENESS: 0.X
        ACCURACY_POTENTIAL: 0.X
        UNCERTAINTY_HANDLING: 0.X
        """

        try:
            response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator of reasoning quality.",
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
                temperature=0.2,
            )

            # Parse evaluation scores
            scores = {}
            for line in response.strip().split("\n"):
                line = line.strip()
                if ":" in line:
                    criterion, score_str = line.split(":", 1)
                    criterion = criterion.strip().upper()
                    try:
                        score = float(score_str.strip())
                        if criterion in [c.name for c in PathEvaluationCriteria]:
                            scores[PathEvaluationCriteria[criterion]] = min(
                                1.0, max(0.0, score)
                            )
                    except ValueError:
                        continue

            path.evaluation_scores = scores

        except Exception as e:
            self.logger.warning(f"Path evaluation failed for {path.id}", error=str(e))
            # Set default scores
            path.evaluation_scores = {
                criteria: 0.5 for criteria in PathEvaluationCriteria
            }

    def _select_paths_for_expansion(
        self, paths: List[ReasoningPath]
    ) -> List[ReasoningPath]:
        """Select the most promising paths for further expansion."""
        # Score paths based on evaluation criteria
        scored_paths = []
        for path in paths:
            overall_score = path.get_overall_score()
            if overall_score >= self.exploration_config.evaluation_threshold:
                scored_paths.append((overall_score, path))

        # Sort by score and select top k
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        selected_paths = [
            path
            for _, path in scored_paths[: self.exploration_config.path_selection_top_k]
        ]

        self.logger.info(
            f"Selected {len(selected_paths)} paths for expansion from {len(paths)} candidates"
        )
        return selected_paths

    async def _expand_reasoning_path(
        self, path: ReasoningPath, question: Question, research_report: ResearchReport
    ) -> None:
        """Expand a reasoning path with additional steps."""
        try:
            # Generate next reasoning step
            next_step = await self._generate_next_reasoning_step(
                path, question, research_report
            )
            path.add_step(next_step)

            # Check if path should be marked as complete
            if (
                path.depth >= self.exploration_config.max_depth
                or await self._is_path_complete(path)
            ):
                path.is_complete = True

        except Exception as e:
            self.logger.warning(f"Failed to expand path {path.id}", error=str(e))
            path.is_complete = (
                True  # Mark as complete to avoid further expansion attempts
            )

    async def _generate_next_reasoning_step(
        self, path: ReasoningPath, question: Question, research_report: ResearchReport
    ) -> ReasoningStep:
        """Generate the next reasoning step for a path."""
        previous_steps = "\n".join(
            [f"Step {i+1}: {step.content}" for i, step in enumerate(path.steps)]
        )

        prompt = f"""
        Continue this {path.path_type.value} reasoning path with the next logical step:

        Question: {question.title}

        Previous reasoning steps:
        {previous_steps}

        Sub-components being analyzed: {', '.join(path.sub_components) if path.sub_components else 'None specified'}

        Provide the next reasoning step that builds upon the previous analysis.
        Focus on deepening the {path.path_type.value} approach and moving toward a conclusion.
        """

        response = await self.llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": f"You are continuing a {path.path_type.value} reasoning analysis.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )

        # Determine step type based on depth and content
        step_type = ReasoningStepType.ANALYSIS
        if path.depth >= self.exploration_config.max_depth - 1:
            step_type = ReasoningStepType.CONCLUSION
        elif "hypothesis" in response.lower():
            step_type = ReasoningStepType.HYPOTHESIS
        elif "synthesis" in response.lower() or "combining" in response.lower():
            step_type = ReasoningStepType.SYNTHESIS

        return ReasoningStep.create(
            step_type=step_type,
            content=response.strip(),
            confidence=0.7,
            metadata={
                "path_type": path.path_type.value,
                "step_number": path.depth + 1,
                "parent_path_id": str(path.id),
            },
        )

    async def _is_path_complete(self, path: ReasoningPath) -> bool:
        """Check if a reasoning path is complete."""
        if not path.steps:
            return False

        last_step = path.steps[-1]

        # Check if last step is a conclusion
        if last_step.step_type == ReasoningStepType.CONCLUSION:
            return True

        # Check if content suggests completion
        completion_indicators = [
            "conclusion",
            "therefore",
            "in summary",
            "final assessment",
        ]
        return any(
            indicator in last_step.content.lower()
            for indicator in completion_indicators
        )

    async def _prune_reasoning_paths(self) -> None:
        """Prune less promising reasoning paths to manage memory."""
        if len(self.reasoning_paths) <= self.exploration_config.max_parallel_paths:
            return

        # Score all paths
        scored_paths = []
        for path in self.reasoning_paths.values():
            score = path.get_overall_score()
            scored_paths.append((score, path))

        # Keep top paths
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        paths_to_keep = scored_paths[: self.exploration_config.max_parallel_paths]

        # Update reasoning_paths dict
        self.reasoning_paths = {path.id: path for _, path in paths_to_keep}

        self.logger.info(f"Pruned reasoning paths to {len(self.reasoning_paths)}")

    async def _evaluate_and_select_paths(self) -> List[ReasoningPath]:
        """Evaluate all paths and select the best ones for synthesis."""
        # Final evaluation of all paths
        all_paths = list(self.reasoning_paths.values())
        await self._evaluate_reasoning_paths(all_paths)

        # Select best paths for synthesis
        scored_paths = []
        for path in all_paths:
            score = path.get_overall_score()
            scored_paths.append((score, path))

        scored_paths.sort(key=lambda x: x[0], reverse=True)

        # Select top paths, ensuring diversity
        selected_paths = []
        used_types = set()

        for score, path in scored_paths:
            if len(selected_paths) >= 3:  # Limit to top 3 paths
                break

            # Prefer diversity in path types
            if path.path_type not in used_types or len(selected_paths) == 0:
                selected_paths.append(path)
                used_types.add(path.path_type)

        self.logger.info(f"Selected {len(selected_paths)} best paths for synthesis")
        return selected_paths

    async def _synthesize_prediction(
        self,
        question: Question,
        research_report: ResearchReport,
        best_paths: List[ReasoningPath],
    ) -> Prediction:
        """Synthesize final prediction from the best reasoning paths."""
        # Prepare synthesis context
        paths_summary = []
        for i, path in enumerate(best_paths):
            paths_summary.append(
                f"Path {i+1} ({path.path_type.value}):\n{path.get_reasoning_summary()}"
            )

        synthesis_prompt = f"""
        Synthesize a final prediction based on these diverse reasoning paths:

        Question: {question.title}
        Description: {question.description or 'No description provided'}
        Type: {question.question_type.value}

        Research Summary: {research_report.executive_summary}

        Reasoning Paths Explored:
        {chr(10).join(paths_summary)}

        Based on the convergence and divergence across these reasoning approaches, provide:
        1. A probability estimate (0-1 for binary questions)
        2. Your confidence in this prediction (0-1)
        3. A synthesis of the key insights from all reasoning paths
        4. Main sources of uncertainty

        Format:
        PROBABILITY: [0-1 value]
        CONFIDENCE: [0-1 value]
        REASONING: [detailed synthesis]
        UNCERTAINTIES: [key uncertainties]
        """

        response = await self.llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are synthesizing insights from multiple reasoning approaches to make a final prediction.",
                },
                {"role": "user", "content": synthesis_prompt},
            ],
            temperature=0.3,
        )

        # Parse response
        probability_value, confidence_value, reasoning, uncertainties = (
            self._parse_synthesis_response(response)
        )

        # Create reasoning trace from best paths
        reasoning_trace = self._create_reasoning_trace(question, best_paths)

        # Create prediction
        metadata = {
            "agent_type": "tree_of_thought_enhanced",
            "paths_explored": len(self.reasoning_paths),
            "best_paths_used": len(best_paths),
            "path_types_used": [path.path_type.value for path in best_paths],
            "exploration_config": {
                "max_depth": self.exploration_config.max_depth,
                "max_breadth": self.exploration_config.max_breadth,
                "max_parallel_paths": self.exploration_config.max_parallel_paths,
            },
            "reasoning_trace_id": str(reasoning_trace.id) if reasoning_trace else None,
        }

        return Prediction.create_binary_prediction(
            question_id=question.id,
            research_report_id=research_report.id,
            probability=probability_value,
            confidence=(
                PredictionConfidence.HIGH
                if confidence_value > 0.7
                else (
                    PredictionConfidence.MEDIUM
                    if confidence_value > 0.4
                    else PredictionConfidence.LOW
                )
            ),
            method=PredictionMethod.TREE_OF_THOUGHT,
            reasoning=reasoning,
            created_by=self.name,
            method_metadata=metadata,
        )

    def _parse_synthesis_response(self, response: str) -> Tuple[float, float, str, str]:
        """Parse the synthesis response into components."""
        probability_value = 0.5
        confidence_value = 0.5
        reasoning = response
        uncertainties = "No specific uncertainties identified"

        lines = response.strip().split("\n")
        current_section = None
        reasoning_lines = []
        uncertainty_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("PROBABILITY:"):
                try:
                    prob_text = line.split(":", 1)[1].strip()
                    if "%" in prob_text:
                        probability_value = float(prob_text.replace("%", "")) / 100
                    else:
                        probability_value = float(prob_text)
                except ValueError:
                    probability_value = 0.5

            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence_value = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence_value = 0.5

            elif line.startswith("REASONING:"):
                current_section = "reasoning"
                reasoning_content = line.split(":", 1)[1].strip()
                if reasoning_content:
                    reasoning_lines.append(reasoning_content)

            elif line.startswith("UNCERTAINTIES:"):
                current_section = "uncertainties"
                uncertainty_content = line.split(":", 1)[1].strip()
                if uncertainty_content:
                    uncertainty_lines.append(uncertainty_content)

            elif current_section == "reasoning" and line:
                reasoning_lines.append(line)

            elif current_section == "uncertainties" and line:
                uncertainty_lines.append(line)

        if reasoning_lines:
            reasoning = "\n".join(reasoning_lines)

        if uncertainty_lines:
            uncertainties = "\n".join(uncertainty_lines)

        return probability_value, confidence_value, reasoning, uncertainties

    def _create_reasoning_trace(
        self, question: Question, best_paths: List[ReasoningPath]
    ) -> Optional[ReasoningTrace]:
        """Create a reasoning trace from the best paths."""
        try:
            all_steps = []

            # Collect steps from all best paths
            for path in best_paths:
                for step in path.steps:
                    # Add path information to metadata
                    enhanced_metadata = {
                        **step.metadata,
                        "path_id": str(path.id),
                        "path_type": path.path_type.value,
                    }

                    enhanced_step = ReasoningStep.create(
                        step_type=step.step_type,
                        content=step.content,
                        confidence=step.confidence,
                        metadata=enhanced_metadata,
                    )
                    all_steps.append(enhanced_step)

            if not all_steps:
                return None

            # Create reasoning trace
            return ReasoningTrace.create(
                question_id=question.id,
                agent_id=self.name,
                reasoning_method="tree_of_thought_enhanced",
                steps=all_steps,
                final_conclusion=f"Synthesized conclusion from {len(best_paths)} reasoning paths",
                overall_confidence=sum(path.get_overall_score() for path in best_paths)
                / len(best_paths),
                bias_checks=[
                    f"Multiple reasoning path types used: {[p.path_type.value for p in best_paths]}"
                ],
                uncertainty_sources=[
                    "Path selection uncertainty",
                    "Synthesis uncertainty",
                ],
            )

        except Exception as e:
            self.logger.warning("Failed to create reasoning trace", error=str(e))
            return None
