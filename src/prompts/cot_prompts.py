"""Chain of Thought prompt templates."""

from typing import List
from jinja2 import Template

from ..domain.entities.question import Question
from ..domain.entities.research_report import ResearchReport, ResearchSource


class ChainOfThoughtPrompts:
    """
    Prompt templates for Chain of Thought reasoning.
    
    These prompts guide the model through step-by-step thinking,
    encouraging explicit reasoning at each stage.
    """
    
    def __init__(self):
        self.question_breakdown_template = Template("""
You are an expert forecaster analyzing a prediction question. Your task is to break down this question into key research areas that need investigation.

QUESTION: {{ question.title }}

DESCRIPTION: {{ question.description }}

QUESTION TYPE: {{ question.question_type.value }}
{% if question.choices %}
CHOICES: {{ question.choices | join(", ") }}
{% endif %}
{% if question.min_value and question.max_value %}
RANGE: {{ question.min_value }} to {{ question.max_value }}
{% endif %}

CATEGORIES: {{ question.categories | join(", ") }}
CLOSE DATE: {{ question.close_time.strftime('%Y-%m-%d') }}

Think step by step to identify the key areas that need research:

1. **Core Factors**: What are the main factors that would influence this outcome?
2. **Historical Context**: What historical data or precedents are relevant?
3. **Current Trends**: What current trends or developments should be analyzed?
4. **Expert Opinions**: What expert views or institutional forecasts exist?
5. **Leading Indicators**: What metrics or signals should be monitored?

Provide your analysis in the following JSON format:
{
    "research_areas": [
        "area1", "area2", "area3", "area4", "area5"
    ],
    "reasoning": "Your step-by-step reasoning for why these areas are important"
}
""")

        self.research_analysis_template = Template("""
You are an expert forecaster analyzing research sources to understand a prediction question. Use systematic, step-by-step reasoning.

QUESTION: {{ question.title }}
DESCRIPTION: {{ question.description }}

RESEARCH SOURCES:
{% for source in sources %}
---
Title: {{ source.title }}
URL: {{ source.url }}
Summary: {{ source.summary }}
Credibility: {{ source.credibility_score }}
{% if source.publish_date %}
Published: {{ source.publish_date.strftime('%Y-%m-%d') }}
{% endif %}
---
{% endfor %}

Analyze this information step by step:

STEP 1: SOURCE EVALUATION
- Evaluate the credibility and relevance of each source
- Identify any potential biases or limitations
- Note the recency and quality of information

STEP 2: EVIDENCE SYNTHESIS
- What evidence supports a positive outcome?
- What evidence suggests a negative outcome?
- What are the key uncertainties and unknowns?

STEP 3: FACTOR ANALYSIS
- What are the most important factors influencing this question?
- How do current trends relate to historical patterns?
- What are the mechanisms that would lead to each outcome?

STEP 4: BASE RATE RESEARCH
- What are relevant base rates from similar situations?
- How does this case compare to historical precedents?

STEP 5: CONFIDENCE ASSESSMENT
- How confident can we be in the available evidence?
- What are the main sources of uncertainty?

Provide your analysis in JSON format:
{
    "executive_summary": "Brief overview of findings",
    "detailed_analysis": "Comprehensive analysis following the 5 steps above",
    "key_factors": ["factor1", "factor2", "factor3"],
    "base_rates": {"similar_event_1": 0.X, "similar_event_2": 0.Y},
    "confidence_level": 0.X,
    "reasoning_steps": ["step1", "step2", "step3", "step4", "step5"],
    "evidence_for": ["evidence supporting positive outcome"],
    "evidence_against": ["evidence supporting negative outcome"],
    "uncertainties": ["key unknowns and limitations"]
}
""")

        self.prediction_template = Template("""
You are an expert forecaster making a prediction. Use clear, step-by-step reasoning to arrive at your forecast.

QUESTION: {{ question.title }}
DESCRIPTION: {{ question.description }}
TYPE: {{ question.question_type.value }}
{% if question.choices %}
CHOICES: {{ question.choices | join(", ") }}
{% endif %}

RESEARCH SUMMARY:
{{ research_report.executive_summary }}

KEY FACTORS:
{% for factor in research_report.key_factors %}
- {{ factor }}
{% endfor %}

EVIDENCE FOR:
{% for evidence in research_report.evidence_for %}
- {{ evidence }}
{% endfor %}

EVIDENCE AGAINST:
{% for evidence in research_report.evidence_against %}
- {{ evidence }}
{% endfor %}

BASE RATES:
{% for event, rate in research_report.base_rates.items() %}
- {{ event }}: {{ rate }}
{% endfor %}

Now, think through your prediction step by step:

STEP 1: BASELINE ASSESSMENT
Start with relevant base rates and historical precedents. What would be a reasonable starting probability based on similar cases?

STEP 2: FACTOR ADJUSTMENT
Consider how each key factor should adjust your probability up or down from the baseline:
- For each factor, determine its impact direction and magnitude
- Consider factor interactions and dependencies

STEP 3: EVIDENCE WEIGHTING
Weigh the evidence for and against:
- How strong and reliable is each piece of evidence?
- Are there any decisive factors or deal-breakers?

STEP 4: UNCERTAINTY QUANTIFICATION
Consider what you don't know:
- What are the key uncertainties?
- How might these affect your confidence?

STEP 5: FINAL CALIBRATION
Arrive at your final prediction:
- What probability best reflects your analysis?
- What confidence level is appropriate?
- What range captures your uncertainty?

{% if question.question_type.value == "binary" %}
Provide your prediction in JSON format:
{
    "probability": 0.XX,
    "confidence": "very_low|low|medium|high|very_high",
    "reasoning": "Your complete step-by-step reasoning",
    "reasoning_steps": [
        "Step 1: Baseline assessment - ...",
        "Step 2: Factor adjustment - ...",
        "Step 3: Evidence weighting - ...",
        "Step 4: Uncertainty quantification - ...",
        "Step 5: Final calibration - ..."
    ],
    "lower_bound": 0.XX,
    "upper_bound": 0.XX,
    "confidence_interval": 0.90
}
{% endif %}

Remember: Be precise in your reasoning, acknowledge uncertainties, and ensure your probability reflects your true belief about the outcome.
""")
    
    def get_question_breakdown_prompt(self, question: Question) -> str:
        """Generate question breakdown prompt."""
        return self.question_breakdown_template.render(question=question)
    
    def deconstruct_question(self, question: Question) -> str:
        """Generate question deconstruction prompt - alias for get_question_breakdown_prompt."""
        return self.get_question_breakdown_prompt(question)
    
    def get_research_analysis_prompt(
        self, 
        question: Question, 
        sources: List[ResearchSource]
    ) -> str:
        """Generate research analysis prompt."""
        return self.research_analysis_template.render(
            question=question, 
            sources=sources
        )
    
    def get_prediction_prompt(
        self, 
        question: Question, 
        research_report: ResearchReport
    ) -> str:
        """Generate prediction prompt."""
        return self.prediction_template.render(
            question=question,
            research_report=research_report
        )
    
    def identify_research_areas(self, question: Question, question_breakdown: str) -> str:
        """Generate prompt to identify research areas based on question breakdown."""
        template = Template("""
Based on the question breakdown below, identify 3-5 key research areas that need investigation:

QUESTION: {{ question.title }}
BREAKDOWN: {{ question_breakdown }}

Identify the most important research areas needed to make an accurate forecast:

1. **Primary Factors**: What are the main drivers?
2. **Data Sources**: What data should be gathered?
3. **Expert Sources**: Who are the relevant experts?
4. **Trend Analysis**: What trends matter?
5. **Risk Factors**: What could go wrong?

Return your response as a JSON list of research areas:
{"research_areas": ["area1", "area2", "area3"]}
""")
        return template.render(question=question, question_breakdown=question_breakdown)
