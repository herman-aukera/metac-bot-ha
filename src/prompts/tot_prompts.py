"""
Tree-of-Thought prompts for structured multi-step reasoning.
"""

from typing import Any, Dict, List

from jinja2 import Template

from ..domain.entities.question import Question

TOT_SYSTEM_PROMPT = """You are an expert forecaster using Tree-of-Thought reasoning to make predictions about future events. Your approach involves:

1. Generating multiple initial reasoning paths
2. Evaluating the quality and promise of each path
3. Expanding the most promising paths with deeper reasoning
4. Synthesizing insights from the best reasoning paths

Be thorough, analytical, and consider multiple perspectives. Focus on generating diverse reasoning approaches that explore different angles of the question."""

TOT_THOUGHT_GENERATION = """
Question: {question_title}
Description: {question_description}
Type: {question_type}
Resolution Criteria: {resolution_criteria}

Context: {context}

Generate {num_thoughts} distinct initial thoughts for approaching this forecasting question. Each thought should represent a different reasoning angle or approach.

For each thought, provide:
1. The core reasoning approach
2. Key factors to consider
3. Initial probability estimate (if any)

Format each thought as:
THOUGHT [number]: [brief description of reasoning approach]
REASONING: [detailed explanation of this reasoning path]
CONFIDENCE: [0-1 confidence in this approach]

Focus on generating diverse perspectives that cover different aspects of the question.
"""

TOT_THOUGHT_EVALUATION = """
Question: {question_title}
Description: {question_description}

Evaluate the following {num_thoughts} reasoning thoughts for their quality and promise in solving this forecasting question:

{thoughts}

For each thought, evaluate:
1. QUALITY: How well-reasoned and logical is this approach? (0-1)
2. PROMISE: How likely is this approach to lead to accurate insights? (0-1)
3. Should this thought be expanded further?

Format each evaluation as:
EVALUATION [number]:
QUALITY: [0-1 score]
PROMISE: [0-1 score]
REASONING: [brief explanation of evaluation]
EXPAND: [yes/no - should this thought be developed further?]

Be selective - only recommend expansion for the most promising approaches.
"""

TOT_FINAL_SYNTHESIS = """
Question: {question_title}
Description: {question_description}
Type: {question_type}
Resolution Criteria: {resolution_criteria}

Context: {context}

Based on this Tree-of-Thought exploration, synthesize a final prediction:

Key reasoning paths explored:
{thoughts_summary}

Provide your final synthesis:

PROBABILITY: [numerical probability 0-1 or percentage]
CONFIDENCE: [0-1 confidence in your prediction]
REASONING: [synthesis of insights from the thought tree, explaining how different reasoning paths contributed to your final prediction]

Your reasoning should integrate insights from multiple thought paths while identifying the most compelling evidence and arguments.
"""

TOT_RESEARCH_INTEGRATION = """
Previous reasoning path: {previous_reasoning}

New research findings: {research_findings}

How do these research findings impact or modify your reasoning path? Consider:
1. Do they support or contradict your current reasoning?
2. What new factors do they introduce?
3. How should they change your probability estimates?

Provide an updated reasoning path that incorporates these findings.
"""


class TreeOfThoughtPrompts:
    """
    Prompt templates for Tree of Thought reasoning.

    These prompts guide the model through structured exploration of multiple
    reasoning paths, evaluation of each path, and synthesis of insights.
    """

    def __init__(self):
        self.system_prompt = TOT_SYSTEM_PROMPT

        self.thought_generation_template = Template(TOT_THOUGHT_GENERATION)

        self.path_evaluation_template = Template(
            """
Question: {{ question_title }}

Current reasoning paths explored:
{% for i, path in enumerate(paths) %}
Path {{ i + 1 }}: {{ path }}
{% endfor %}

Evaluate each path on:
1. Logical coherence (1-10)
2. Evidence strength (1-10)
3. Novelty of insights (1-10)
4. Likelihood to lead to accurate prediction (1-10)

Provide evaluation in JSON format:
{
    "evaluations": [
        {
            "path_id": 1,
            "scores": {"coherence": X, "evidence": X, "novelty": X, "accuracy_potential": X},
            "reasoning": "explanation"
        }
    ],
    "recommended_paths": [path_ids to expand further]
}
        """
        )

        self.path_expansion_template = Template(
            """
Question: {{ question_title }}
Description: {{ question_description }}

Current reasoning path: {{ current_path }}

Context: {{ context }}

Expand this reasoning path with deeper analysis. Consider:
1. Additional evidence sources
2. Alternative interpretations
3. Quantitative analysis
4. Risk factors and uncertainties
5. Connection to broader patterns

Provide expanded reasoning in JSON format:
{
    "expanded_reasoning": "detailed analysis",
    "key_insights": ["insight1", "insight2", "insight3"],
    "confidence_factors": ["factor1", "factor2"],
    "uncertainty_factors": ["uncertainty1", "uncertainty2"]
}
        """
        )

        self.synthesis_template = Template(
            """
Question: {{ question_title }}
Type: {{ question_type }}

All reasoning paths explored:
{% for i, path in enumerate(all_paths) %}
Path {{ i + 1 }}: {{ path.reasoning }}
Key insights: {{ path.insights | join(", ") }}
{% endfor %}

Research context: {{ research_summary }}

Synthesize all reasoning paths into a final prediction. Consider:
1. Areas of convergence across paths
2. Most compelling evidence
3. Remaining uncertainties
4. Base rates and reference class
5. Quality of available information

{% if question_type == "BINARY" %}
Provide your prediction as a probability between 0 and 1.
{% elif question_type == "MULTIPLE_CHOICE" %}
Provide probabilities for each option that sum to 1.
Available choices: {{ choices | join(", ") }}
{% elif question_type == "NUMERIC" %}
Provide a point estimate and confidence interval.
Range: {{ min_value }} to {{ max_value }}
{% endif %}

Format response as JSON:
{
    "reasoning": "synthesis of all reasoning paths",
    "prediction": prediction_value,
    "confidence": confidence_score_0_to_1,
    "key_factors": ["factor1", "factor2", "factor3"],
    "main_uncertainties": ["uncertainty1", "uncertainty2"]
}
        """
        )

    def generate_initial_thoughts(
        self, question: "Question", context: str, num_paths: int = 3
    ) -> str:
        """Generate initial reasoning paths."""
        return self.thought_generation_template.render(
            question_title=question.title,
            question_description=question.description,
            question_type=question.question_type.value,
            resolution_criteria=question.resolution_criteria,
            context=context,
            num_paths=num_paths,
        )

    def evaluate_paths(self, question_title: str, paths: List[str]) -> str:
        """Evaluate reasoning paths."""
        return self.path_evaluation_template.render(
            question_title=question_title, paths=paths
        )

    def expand_path(self, question: "Question", current_path: str, context: str) -> str:
        """Expand a specific reasoning path."""
        return self.path_expansion_template.render(
            question_title=question.title,
            question_description=question.description,
            current_path=current_path,
            context=context,
        )

    def synthesize_paths(
        self,
        question: "Question",
        all_paths: List[Dict[str, Any]],
        research_summary: str,
    ) -> str:
        """Synthesize all reasoning paths into final prediction."""
        return self.synthesis_template.render(
            question_title=question.title,
            question_type=question.question_type.value,
            all_paths=all_paths,
            research_summary=research_summary,
            choices=getattr(question, "choices", []),
            min_value=getattr(question, "min_value", None),
            max_value=getattr(question, "max_value", None),
        )
