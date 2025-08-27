"""Calibrated forecasting prompt templates for budget-efficient predictions.

This module provides forecasting prompts designed to improve calibration and
reduce overconfidence while maintaining token efficiency.
"""

from typing import List, Dict, Any, Optional
from jinja2 import Template
from ..domain.entities.question import Question
from ..domain.entities.research_report import ResearchReport


class CalibratedForecastingPrompts:
    """
    Token-efficient forecasting prompts with explicit calibration instructions.

    These prompts are designed to:
    - Reduce overconfidence through structured calibration
    - Include scenario analysis and uncertainty assessment
    - Provide explicit confidence calibration instructions
    - Maintain budget efficiency while improving accuracy
    """

    def __init__(self):
        pass
        # Basic calibrated forecasting template
        self.basic_calibrated_template = Template("""
Make a calibrated forecast for this question:

Q: {{ question.title }}
Type: {{ question.question_type.value }}
Close: {{ question.close_time.strftime('%Y-%m-%d') }}

Research Summary: {{ research_summary }}

CALIBRATION STEPS:
1. BASE RATE: What's the historical frequency of similar events?
2. SPECIFICS: How does this case differ from the base rate?
3. SCENARIOS: Consider best/worst/most likely outcomes
4. CONFIDENCE: Rate your certainty (avoid overconfidence)

{% if question.question_type.value == "BINARY" %}
Provide probability (0-1) with confidence interval:
{
  "probability": 0.XX,
  "confidence_interval": [0.XX, 0.XX],
  "confidence_level": 0.XX,
  "reasoning": "brief explanation"
}
{% endif %}

Remember: Well-calibrated forecasters are right X% of the time when they say X%.
""")
        # Scenario analysis template for uncertainty assessment
        self.scenario_analysis_template = Template("""
Forecast with scenario analysis:

QUESTION: {{ question.title }}
RESEARCH: {{ research_summary }}

SCENARIO ANALYSIS:
1. OPTIMISTIC: What if things go better than expected?
2. PESSIMISTIC: What if key risks materialize?
3. BASELINE: Most likely outcome based on current trends

For each scenario:
- Probability of scenario occurring
- Outcome if scenario occurs
- Key factors that would trigger it

UNCERTAINTY FACTORS:
- Information gaps
- Unpredictable variables
- Time horizon effects

{% if question.question_type.value == "BINARY" %}
Final forecast:
{
  "scenarios": {
    "optimistic": {"probability": 0.XX, "outcome_prob": 0.XX},
    "baseline": {"probability": 0.XX, "outcome_prob": 0.XX},
    "pessimistic": {"probability": 0.XX, "outcome_prob": 0.XX}
  },
  "weighted_probability": 0.XX,
  "confidence": 0.XX,
  "key_uncertainties": ["uncertainty1", "uncertainty2"]
}
{% endif %}
""")
        # Overconfidence reduction template
        self.overconfidence_reduction_template = Template("""
Make a well-calibrated forecast (avoid overconfidence):

Q: {{ question.title }}
Research: {{ research_summary }}

OVERCONFIDENCE CHECKS:
1. DEVIL'S ADVOCATE: What evidence contradicts your initial view?
2. OUTSIDE VIEW: How often are similar predictions correct?
3. REFERENCE CLASS: What's the base rate for this type of event?
4. INFORMATION QUALITY: How reliable is your evidence?
5. COMPLEXITY: How many things need to go right/wrong?

CALIBRATION ANCHORS:
- 50%: Coin flip uncertainty
- 70%: Moderately confident
- 90%: Very confident (rare!)
- 95%: Extremely confident (very rare!)

{% if question.question_type.value == "BINARY" %}
Calibrated forecast:
{
  "initial_intuition": 0.XX,
  "after_devils_advocate": 0.XX,
  "base_rate_adjusted": 0.XX,
  "final_probability": 0.XX,
  "confidence_justification": "why this confidence level",
  "could_be_wrong_because": ["reason1", "reason2"]
}
{% endif %}

Remember: Overconfidence is the #1 bias in forecasting. Be humble about uncertainty.
""")
        # Reference class forecasting template
        self.reference_class_template = Template("""
Use reference class forecasting:

Q: {{ question.title }}
Research: {{ research_summary }}

REFERENCE CLASS ANALYSIS:
1. IDENTIFY: What's the reference class of similar events?
2. BASE RATE: What's the historical success/failure rate?
3. ADJUSTMENTS: How is this case different?
4. FINAL: Adjust base rate for specific factors

STEPS:
Step 1: Reference class = "{{ reference_class_hint or 'similar events' }}"
Step 2: Historical base rate = ?
Step 3: This case is different because...
Step 4: Adjust base rate up/down by...

{% if question.question_type.value == "BINARY" %}
Reference class forecast:
{
  "reference_class": "description",
  "base_rate": 0.XX,
  "adjustment_factors": [
    {"factor": "factor1", "direction": "up/down", "magnitude": "small/medium/large"}
  ],
  "adjusted_probability": 0.XX,
  "confidence": 0.XX
}
{% endif %}

Base rates are powerful. Start there, then adjust carefully.
""")
    def get_basic_calibrated_prompt(self, question: Question, research_summary: str) -> str:
        """Get basic calibrated forecasting prompt."""
        return self.basic_calibrated_template.render(
            question=question,
            research_summary=research_summary
        )

    def get_scenario_analysis_prompt(self, question: Question, research_summary: str) -> str:
        """Get scenario analysis forecasting prompt."""
        return self.scenario_analysis_template.render(
            question=question,
            research_summary=research_summary
        )

    def get_overconfidence_reduction_prompt(self, question: Question, research_summary: str) -> str:
        """Get overconfidence reduction forecasting prompt."""
        return self.overconfidence_reduction_template.render(
            question=question,
            research_summary=research_summary
        )

    def get_reference_class_prompt(self, question: Question, research_summary: str,
                                 reference_class_hint: Optional[str] = None) -> str:
        """Get reference class forecasting prompt."""
        return self.reference_class_template.render(
            question=question,
            research_summary=research_summary,
            reference_class_hint=reference_class_hint
        )
        # Comprehensive calibrated template combining all techniques
        self.comprehensive_calibrated_template = Template("""
Make a well-calibrated forecast using multiple debiasing techniques:

QUESTION: {{ question.title }}
DESCRIPTION: {{ question.description }}
TYPE: {{ question.question_type.value }}
CLOSE: {{ question.close_time.strftime('%Y-%m-%d') }}

RESEARCH SUMMARY: {{ research_summary }}

CALIBRATED FORECASTING PROCESS:

STEP 1: REFERENCE CLASS & BASE RATE
- What's the reference class of similar events?
- What's the historical base rate?
- Sample size and quality of historical data?

STEP 2: CASE-SPECIFIC FACTORS
- How does this case differ from the reference class?
- What factors should adjust the base rate up or down?
- Strength of evidence for each adjustment?

STEP 3: SCENARIO ANALYSIS
- Optimistic scenario (10-20% chance): What if things go very well?
- Baseline scenario (60-80% chance): Most likely outcome?
- Pessimistic scenario (10-20% chance): What if key risks materialize?

STEP 4: OVERCONFIDENCE CHECKS
- Devil's advocate: What evidence contradicts my view?
- Outside view: How often are forecasters right in similar cases?
- Complexity: How many things need to align for my prediction?
- Information quality: How reliable and complete is my evidence?

STEP 5: CALIBRATION
- Initial intuition probability
- After reference class adjustment
- After scenario weighting
- After overconfidence correction
- Final calibrated probability

{% if question.question_type.value == "BINARY" %}
Provide comprehensive calibrated forecast:
{
  "reference_class": {
    "description": "...",
    "base_rate": 0.XX,
    "sample_size": N,
    "confidence_in_base_rate": 0.XX
  },
  "adjustments": [
    {"factor": "...", "direction": "up/down", "magnitude": 0.XX, "confidence": 0.XX}
  ],
  "scenarios": {
    "optimistic": {"probability": 0.XX, "outcome_if_occurs": 0.XX},
    "baseline": {"probability": 0.XX, "outcome_if_occurs": 0.XX},
    "pessimistic": {"probability": 0.XX, "outcome_if_occurs": 0.XX}
  },
  "calibration_steps": {
    "initial_intuition": 0.XX,
    "base_rate_adjusted": 0.XX,
    "scenario_weighted": 0.XX,
    "overconfidence_corrected": 0.XX
  },
  "final_probability": 0.XX,
  "confidence_interval": [0.XX, 0.XX],
  "confidence_level": 0.XX,
  "key_uncertainties": ["...", "...", "..."],
  "could_be_wrong_because": ["...", "...", "..."]
}
{% endif %}

Remember: Good calibration means being right X% of the time when you say X%.
""")
    def get_comprehensive_calibrated_prompt(self, question: Question, research_summary: str) -> str:
        """Get comprehensive calibrated forecasting prompt with all debiasing techniques."""
        return self.comprehensive_calibrated_template.render(
            question=question,
            research_summary=research_summary
        )

    def get_calibrated_prompt(self, question: Question, research_summary: str,
                            calibration_type: str = "basic") -> str:
        """
        Get calibrated forecasting prompt based on type.

        Args:
            question: The forecasting question
            research_summary: Summary of research findings
            calibration_type: Type of calibration ("basic", "scenario", "overconfidence",
                            "reference_class", "comprehensive")

        Returns:
            Calibrated forecasting prompt string
        """
        if calibration_type == "scenario":
            return self.get_scenario_analysis_prompt(question, research_summary)
        elif calibration_type == "overconfidence":
            return self.get_overconfidence_reduction_prompt(question, research_summary)
        elif calibration_type == "reference_class":
            return self.get_reference_class_prompt(question, research_summary)
        elif calibration_type == "comprehensive":
            return self.get_comprehensive_calibrated_prompt(question, research_summary)
        else:  # basic
            return self.get_basic_calibrated_prompt(question, research_summary)

    def estimate_calibration_token_usage(self, calibration_type: str) -> Dict[str, int]:
        """
        Estimate token usage for different calibration prompt types.

        Returns:
            Dictionary with estimated input and expected output tokens
        """
        estimates = {
            "basic": {"input_tokens": 200, "expected_output": 150},
            "scenario": {"input_tokens": 280, "expected_output": 300},
            "overconfidence": {"input_tokens": 320, "expected_output": 250},
            "reference_class": {"input_tokens": 250, "expected_output": 200},
            "comprehensive": {"input_tokens": 450, "expected_output": 500}
        }
        return estimates.get(calibration_type, estimates["basic"])


class CalibrationPromptManager:
    """
    Manages selection of calibrated forecasting prompts based on question characteristics.
    """

    def __init__(self):
        self.calibrated_prompts = CalibratedForecastingPrompts()

    def select_optimal_calibration_type(self, question: Question,
                                      budget_remaining: Optional[float] = None) -> str:
        """
        Select optimal calibration type based on question characteristics and budget.

        Args:
            question: The forecasting question
            budget_remaining: Remaining budget in dollars

        Returns:
            Recommended calibration type
        """
        # Budget constraints
        if budget_remaining is not None and budget_remaining < 5:
            return "basic"
        elif budget_remaining is not None and budget_remaining < 15:
            return "overconfidence"  # Good bang for buck

        # Question complexity analysis
        complexity_score = self._analyze_calibration_needs(question)

        if complexity_score <= 2:
            return "basic"
        elif complexity_score <= 4:
            return "overconfidence"
        elif complexity_score <= 6:
            return "scenario"
        else:
            return "comprehensive"

    def _analyze_calibration_needs(self, question: Question) -> int:
        """Analyze how much calibration support a question needs."""
        score = 0

        # Long-term questions benefit from scenario analysis
        if question.close_time:
            from datetime import datetime, timezone
            days_until_close = (question.close_time - datetime.now(timezone.utc)).days
            if days_until_close > 180:
                score += 2

        # Complex question types need more calibration
        if question.question_type.value in ["NUMERIC", "MULTIPLE_CHOICE"]:
            score += 1

        # Technical topics prone to overconfidence
        technical_categories = ["science", "technology", "economics", "politics"]
        if any(cat.lower() in technical_categories for cat in question.categories):
            score += 2

        # Questions with many choices need scenario analysis
        if hasattr(question, 'choices') and question.choices and len(question.choices) > 3:
            score += 1

        return score
    def get_optimal_calibrated_prompt(self, question: Question, research_summary: str,
                                    budget_remaining: Optional[float] = None,
                                    force_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get optimal calibrated forecasting prompt with metadata.

        Args:
            question: The forecasting question
            research_summary: Summary of research findings
            budget_remaining: Remaining budget in dollars
            force_type: Force specific calibration type

        Returns:
            Dictionary containing prompt, metadata, and cost estimates
        """
        # Select calibration type
        calibration_type = force_type or self.select_optimal_calibration_type(
            question, budget_remaining
        )

        # Get the prompt
        prompt = self.calibrated_prompts.get_calibrated_prompt(
            question, research_summary, calibration_type
        )

        # Get token estimates
        token_estimates = self.calibrated_prompts.estimate_calibration_token_usage(calibration_type)

        # Calculate cost estimates (using same rates as research prompts)
        model_costs = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
        }

        cost_estimates = {}
        for model, rates in model_costs.items():
            input_cost = (token_estimates["input_tokens"] / 1000) * rates["input"]
            output_cost = (token_estimates["expected_output"] / 1000) * rates["output"]
            cost_estimates[model] = {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost
            }

        return {
            "prompt": prompt,
            "calibration_type": calibration_type,
            "token_estimates": token_estimates,
            "cost_estimates": cost_estimates,
            "recommended_model": "gpt-4o-mini",  # Forecasting can use mini effectively
            "metadata": {
                "question_id": getattr(question, 'id', None),
                "question_type": question.question_type.value,
                "calibration_techniques": self._get_techniques_used(calibration_type)
            }
        }

    def _get_techniques_used(self, calibration_type: str) -> List[str]:
        """Get list of calibration techniques used in each type."""
        techniques = {
            "basic": ["base_rate_anchoring", "confidence_intervals"],
            "scenario": ["scenario_analysis", "uncertainty_assessment"],
            "overconfidence": ["devils_advocate", "outside_view", "reference_class"],
            "reference_class": ["reference_class_forecasting", "base_rate_adjustment"],
            "comprehensive": ["reference_class", "scenario_analysis", "overconfidence_reduction", "calibration_steps"]
        }
        return techniques.get(calibration_type, [])
