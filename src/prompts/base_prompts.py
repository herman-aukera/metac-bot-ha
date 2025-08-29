"""Base prompt templates and utilities."""

from typing import Any, Dict

from jinja2 import Template


class BasePrompts:
    """
    Base prompt templates and utilities shared across different reasoning strategies.
    """

    def __init__(self):
        self.research_summary_template = Template(
            """
Based on the following research sources, provide a comprehensive summary:

{% for source in research_sources %}
Source: {{ source.title }}
URL: {{ source.url }}
Summary: {{ source.summary }}
Relevance Score: {{ source.relevance_score }}

{% endfor %}

Synthesize the key findings, noting areas of agreement and disagreement among sources.
Highlight the most reliable and relevant information for forecasting.

Format as JSON:
{
    "key_findings": ["finding1", "finding2", "finding3"],
    "consensus_areas": ["area1", "area2"],
    "disagreements": ["disagreement1", "disagreement2"],
    "reliability_assessment": "assessment of source quality",
    "forecasting_relevance": "how this relates to the prediction"
}
        """
        )

        self.confidence_calibration_template = Template(
            """
Question: {{ question_title }}
Your prediction: {{ prediction }}

Calibrate your confidence by considering:

1. **Information Quality**: How reliable and comprehensive is your evidence?
2. **Historical Base Rates**: How often do similar events occur?
3. **Uncertainty Sources**: What are the main sources of uncertainty?
4. **Time Horizon**: How does the prediction timeline affect uncertainty?
5. **Expert Consensus**: How aligned is your view with expert opinions?

Rate each factor from 1-5 and provide overall confidence (0-1):

{
    "information_quality": rating_1_to_5,
    "base_rate_knowledge": rating_1_to_5,
    "uncertainty_understanding": rating_1_to_5,
    "time_horizon_consideration": rating_1_to_5,
    "expert_alignment": rating_1_to_5,
    "overall_confidence": confidence_0_to_1,
    "reasoning": "explanation of confidence level"
}
        """
        )

        self.prediction_format_template = Template(
            """
{% if question_type == "BINARY" %}
For this binary question, provide your answer as a probability between 0 and 1.
Example: 0.65 means 65% chance the event will occur.

{% elif question_type == "MULTIPLE_CHOICE" %}
For this multiple choice question, provide probabilities for each option that sum to 1.
Available choices: {{ choices | join(", ") }}
Example: {"Option A": 0.4, "Option B": 0.35, "Option C": 0.25}

{% elif question_type == "NUMERIC" %}
For this numeric question, provide a point estimate and confidence interval.
Range: {{ min_value }} to {{ max_value }}
Example: {"point_estimate": 150, "confidence_interval": [120, 180], "confidence_level": 0.8}

{% endif %}

Always include:
- Your reasoning process
- Key factors that influenced your prediction  
- Main sources of uncertainty
- Confidence level in your prediction
        """
        )

    def format_research_summary(self, research_sources: list) -> str:
        """Format research sources into summary prompt."""
        return self.research_summary_template.render(research_sources=research_sources)

    def format_confidence_calibration(
        self, question_title: str, prediction: Any
    ) -> str:
        """Format confidence calibration prompt."""
        return self.confidence_calibration_template.render(
            question_title=question_title, prediction=prediction
        )

    def format_prediction_guidelines(
        self,
        question_type: str,
        choices: list = None,
        min_value: float = None,
        max_value: float = None,
    ) -> str:
        """Format prediction guidelines based on question type."""
        return self.prediction_format_template.render(
            question_type=question_type,
            choices=choices or [],
            min_value=min_value,
            max_value=max_value,
        )
