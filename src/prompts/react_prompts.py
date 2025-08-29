"""
ReAct (Reasoning and Acting) prompts for interactive research and reasoning.
"""

REACT_SYSTEM_PROMPT = """You are an expert forecaster using the ReAct (Reasoning and Acting) methodology. You work by:

1. Thinking about what you need to know
2. Taking actions to gather information or analyze
3. Observing the results
4. Reasoning about what you learned
5. Repeating this cycle until you have enough information

Available actions:
- SEARCH: Search for relevant information online
- THINK: Deep thinking about a specific aspect
- ANALYZE: Analyze specific information or data
- SYNTHESIZE: Combine insights from multiple sources
- FINALIZE: Indicate you're ready to make the final prediction

Be systematic and thorough. Each step should build on previous observations."""

REACT_REASONING_PROMPT = """
Question: {question_title}
Description: {question_description}
Type: {question_type}
Resolution Criteria: {resolution_criteria}

Context: {context}

Previous steps:
{previous_steps}

You are at step {step_number} of {max_steps}. Based on what you've learned so far, what should you do next?

Provide your response in this format:
THOUGHT: [What are you thinking about? What do you need to know or do next?]
ACTION: [Choose one: search, think, analyze, synthesize, finalize]
ACTION_INPUT: [Specific input for the action - search query, topic to think about, data to analyze, etc.]

Remember: You're building toward making an accurate probability prediction for this question.
"""

REACT_ACTION_PROMPT = """
Question: {question_title}
Description: {question_description}
Type: {question_type}
Resolution Criteria: {resolution_criteria}

Complete ReAct trace:
{react_trace}

Based on your systematic ReAct exploration above, provide your final prediction:

PROBABILITY: [numerical probability 0-1 or percentage]
CONFIDENCE: [0-1 confidence in your prediction]
REASONING: [comprehensive reasoning that synthesizes insights from your ReAct process, explaining how each major step contributed to your final assessment]

Your reasoning should clearly show how your step-by-step investigation led to this prediction.
"""

REACT_SEARCH_FOLLOW_UP = """
Based on these search results: {search_results}

What are the key insights relevant to the forecasting question: {question_title}

Identify:
1. Most relevant pieces of information
2. How they support or oppose different outcomes
3. What additional information might be needed
4. Specific aspects that warrant deeper investigation
"""

REACT_SYNTHESIS_PROMPT = """
You've gathered the following information through your ReAct investigation:

{information_summary}

Question: {question_title}

Synthesize this information to:
1. Identify the strongest predictive factors
2. Assess the balance of evidence for different outcomes
3. Highlight key uncertainties or missing information
4. Form preliminary probability estimates

Focus on how different pieces of evidence interact and what they collectively suggest.
"""

from typing import Any, Dict, List

from jinja2 import Template


class ReActPrompts:
    """
    Prompt templates for ReAct (Reasoning and Acting) methodology.

    These prompts guide the model through iterative cycles of
    reasoning, acting, and observing to gather information and
    develop predictions.
    """

    def __init__(self):
        self.system_prompt = REACT_SYSTEM_PROMPT

        self.initial_prompt_template = Template(
            """
Question: {{ question.title }}
Description: {{ question.description }}
Type: {{ question.question_type.value }}
{% if question.choices %}
Choices: {{ question.choices | join(", ") }}
{% endif %}
Resolution Criteria: {{ question.resolution_criteria }}
Close Date: {{ question.close_time.strftime('%Y-%m-%d') }}

Your task is to forecast this question using the ReAct methodology.

Start by thinking about what information you need, then take your first action.

Format your response as:
Thought: [Your reasoning about what to do next]
Action: [SEARCH/THINK/ANALYZE/SYNTHESIZE/FINALIZE]
Action Input: [Specific details for the action]
        """
        )

        self.continue_prompt_template = Template(
            """
Previous steps:
{% for step in previous_steps %}
{{ step.thought }}
{{ step.action }}: {{ step.action_input }}
Observation: {{ step.observation }}
{% endfor %}

Continue your reasoning process. What should you do next?

Thought: [Your reasoning about what to do next]
Action: [SEARCH/THINK/ANALYZE/SYNTHESIZE/FINALIZE]  
Action Input: [Specific details for the action]
        """
        )

        self.finalize_prompt_template = Template(
            """
Question: {{ question.title }}
Type: {{ question.question_type.value }}

Complete reasoning process:
{% for step in all_steps %}
Thought: {{ step.thought }}
Action: {{ step.action }}: {{ step.action_input }}
Observation: {{ step.observation }}
{% endfor %}

Based on all your research and reasoning, provide your final prediction.

{% if question.question_type.value == "BINARY" %}
Provide your prediction as a probability between 0 and 1.
{% elif question.question_type.value == "MULTIPLE_CHOICE" %}
Provide probabilities for each option that sum to 1.
Available choices: {{ question.choices | join(", ") }}
{% elif question.question_type.value == "NUMERIC" %}
Provide a point estimate and confidence interval.
Range: {{ question.min_value }} to {{ question.max_value }}
{% endif %}

Format response as JSON:
{
    "final_reasoning": "comprehensive summary of your reasoning",
    "prediction": prediction_value,
    "confidence": confidence_score_0_to_1,
    "key_evidence": ["evidence1", "evidence2", "evidence3"],
    "main_uncertainties": ["uncertainty1", "uncertainty2"]
}
        """
        )

    def get_initial_prompt(self, question: "Question") -> str:
        """Get initial ReAct prompt for a question."""
        return self.initial_prompt_template.render(question=question)

    def get_continue_prompt(self, previous_steps: List[Dict[str, str]]) -> str:
        """Get continuation prompt with previous steps."""
        return self.continue_prompt_template.render(previous_steps=previous_steps)

    def get_finalize_prompt(
        self, question: "Question", all_steps: List[Dict[str, str]]
    ) -> str:
        """Get finalization prompt with all steps."""
        return self.finalize_prompt_template.render(
            question=question, all_steps=all_steps
        )
