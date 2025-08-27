"""
Anti-slop prompt templates for tournament forecasting.
Implements quality guard directives for GPT-5 tri-model system.
"""

from datetime import datetime
from typing import Dict, Optional
from forecasting_tools import clean_indents


class AntiSlopPrompts:
    """
    Anti-slop prompt templates with quality guard directives.
    Designed for competitive tournament forecasting with GPT-5 variants.
    """

    @staticmethod
    def get_base_anti_slop_directives() -> str:
        """Base anti-slop quality guard directives for all tasks."""
        return clean_indents("""
            # ANTI-SLOP / QUALITY GUARD
            • Think step-by-step internally, then output only final, clear reasoning
            • Ground every claim with specific evidence sources - no hallucinations
            • If uncertain about anything, acknowledge it explicitly
            • Use bullet points (•) for structure, keep response ≤ 300 words unless complex analysis required
            • Maintain human, helpful tone while being precise
            • Pre-check: Does every statement trace to verifiable evidence?
            • Question your own reasoning - could there be edge cases or alternatives?
            • Avoid overconfidence - calibrate predictions carefully for tournament scoring
        """)

    @staticmethod
    def get_research_prompt(question_text: str, model_tier: str = "mini") -> str:
        """
        Research prompt with anti-slop directives for GPT-5 variants.

        Args:
            question_text: The forecasting question to research
            model_tier: Model tier (nano/mini/full) for tier-specific optimization
        """

        tier_specific = {
            "nano": "• Prioritize speed and essential facts only\n• Focus on most recent and credible sources\n",
            "mini": "• Balance depth with efficiency\n• Provide structured synthesis with key insights\n",
            "full": "• Use comprehensive analysis capability\n• Provide detailed research with multiple perspectives\n"
        }

        return clean_indents(f"""
            You are a research assistant to an elite forecasting expert competing in the Metaculus AI Benchmark Tournament.

            {AntiSlopPrompts.get_base_anti_slop_directives()}

            # RESEARCH-SPECIFIC DIRECTIVES
            • Cite every factual claim with sources (URLs, publication names, dates)
            • Acknowledge information gaps explicitly - don't fill with speculation
            • Prioritize recent developments (last 48 hours most important)
            • Synthesize information without adding unsupported interpretations
            • Focus on factors that could influence the question's resolution
            {tier_specific.get(model_tier, "")}

            # RESEARCH TASK
            Research this forecasting question systematically:

            **QUESTION**: {question_text}

            **RESEARCH REQUIREMENTS**:
            • Recent news and developments (last 48 hours) - cite sources with dates
            • Historical precedents with specific dates and outcomes
            • Expert opinions from credible sources (name the experts and publications)
            • Quantitative data with verification (specify data sources)
            • Key uncertainty factors that could shift the outcome

            **OUTPUT FORMAT**:
            • Recent Developments: [Bullet points with source citations]
            • Historical Context: [Relevant precedents with dates]
            • Expert Views: [Credible expert opinions with sources]
            • Data Points: [Quantitative information with sources]
            • Uncertainty Factors: [Key unknowns that could change outcome]

            Provide only verified, traceable information. If information is unavailable, state this explicitly.
        """)

    @staticmethod
    def get_binary_forecast_prompt(question_text: str, background_info: str,
                                 resolution_criteria: str, fine_print: str,
                                 research: str, model_tier: str = "full") -> str:
        """
        Binary forecasting prompt with anti-slop directives.

        Args:
            question_text: The binary question to forecast
            background_info: Question background information
            resolution_criteria: How the question will be resolved
            fine_print: Additional question details
            research: Research findings from research phase
            model_tier: Model tier for tier-specific optimization
        """

        tier_specific = {
            "nano": "• Focus on essential factors only\n• Provide quick, calibrated assessment\n",
            "mini": "• Balance thoroughness with efficiency\n• Consider key scenarios and base rates\n",
            "full": "• Use maximum reasoning capability\n• Comprehensive scenario analysis and calibration\n"
        }

        return clean_indents(f"""
            You are an elite forecasting expert competing in the Metaculus AI Benchmark Tournament.
            Your goal is to provide the most accurate probability estimate possible.

            {AntiSlopPrompts.get_base_anti_slop_directives()}

            # FORECASTING-SPECIFIC DIRECTIVES
            • Base predictions on verifiable evidence and historical precedents
            • Acknowledge uncertainty explicitly - avoid overconfidence penalties
            • Consider multiple scenarios and weight them by probability
            • Use base rates and reference class forecasting when applicable
            • Calibrate confidence carefully - tournament uses log scoring
            • Think in terms of frequencies: "X out of 100 similar situations"
            {tier_specific.get(model_tier, "")}

            # FORECASTING TASK

            **QUESTION**: {question_text}

            **BACKGROUND**: {background_info}

            **RESOLUTION CRITERIA**: {resolution_criteria}

            **FINE PRINT**: {fine_print}

            **RESEARCH FINDINGS**: {research}

            **TODAY'S DATE**: {datetime.now().strftime("%Y-%m-%d")}

            # FORECASTING PROTOCOL
            Before providing your final probability, analyze:

            (a) **Time Horizon**: How much time remains until resolution?
            (b) **Status Quo**: What happens if current trends continue unchanged?
            (c) **No Scenario**: Describe a plausible path to a "No" outcome
            (d) **Yes Scenario**: Describe a plausible path to a "Yes" outcome
            (e) **Base Rate**: What's the historical frequency of similar events?
            (f) **Key Uncertainties**: What factors could significantly change the outcome?

            # CALIBRATION REMINDER
            Good forecasters:
            • Put extra weight on status quo (world changes slowly)
            • Avoid extreme probabilities unless evidence is overwhelming
            • Consider reference class of similar questions
            • Account for unknown unknowns with appropriate uncertainty

            **FINAL OUTPUT**:
            Probability: XX% (where XX is your calibrated probability estimate)

            **Confidence Level**: [Low/Medium/High] - Why this confidence vs higher/lower?

            **Key Reasoning**: [2-3 sentences citing specific evidence and scenarios]
        """)

    @staticmethod
    def get_multiple_choice_prompt(question_text: str, options: list, background_info: str,
                                 resolution_criteria: str, fine_print: str, research: str,
                                 model_tier: str = "full") -> str:
        """Multiple choice forecasting prompt with anti-slop directives."""

        options_str = ", ".join(options)

        tier_specific = {
            "nano": "• Focus on most likely outcomes\n• Quick probability assignment\n",
            "mini": "• Consider all options systematically\n• Balanced probability distribution\n",
            "full": "• Comprehensive analysis of all scenarios\n• Detailed probability reasoning\n"
        }

        return clean_indents(f"""
            You are an elite forecasting expert competing in the Metaculus AI Benchmark Tournament.

            {AntiSlopPrompts.get_base_anti_slop_directives()}

            # MULTIPLE CHOICE FORECASTING DIRECTIVES
            • Assign probabilities to ALL options (must sum to 100%)
            • Leave moderate probability on most options for unexpected outcomes
            • Base probability assignments on evidence strength
            • Consider correlation between options
            • Avoid overconfidence in any single outcome
            {tier_specific.get(model_tier, "")}

            **QUESTION**: {question_text}

            **OPTIONS**: {options_str}

            **BACKGROUND**: {background_info}

            **RESOLUTION CRITERIA**: {resolution_criteria}

            **FINE PRINT**: {fine_print}

            **RESEARCH FINDINGS**: {research}

            **TODAY'S DATE**: {datetime.now().strftime("%Y-%m-%d")}

            # ANALYSIS PROTOCOL
            Before assigning probabilities, consider:

            (a) **Time Horizon**: How much time until resolution?
            (b) **Status Quo**: Which outcome if trends continue?
            (c) **Unexpected Scenarios**: What could cause surprising outcomes?
            (d) **Evidence Strength**: Which options have strongest support?
            (e) **Historical Patterns**: What do similar cases suggest?

            # PROBABILITY ASSIGNMENT
            Remember:
            • Good forecasters leave some probability on most options
            • Avoid putting >80% on any single option unless evidence is overwhelming
            • Consider that unexpected outcomes happen ~20% of the time
            • Weight evidence quality, not just quantity

            **FINAL PROBABILITIES** (must sum to 100%):
            {chr(10).join([f"{option}: XX%" for option in options])}

            **Reasoning**: [Brief explanation of probability assignments with evidence]
        """)

    @staticmethod
    def get_numeric_forecast_prompt(question_text: str, background_info: str,
                                  resolution_criteria: str, fine_print: str,
                                  research: str, unit_of_measure: Optional[str] = None,
                                  lower_bound: Optional[float] = None,
                                  upper_bound: Optional[float] = None,
                                  model_tier: str = "full") -> str:
        """Numeric forecasting prompt with anti-slop directives."""

        bounds_info = ""
        if lower_bound is not None:
            bounds_info += f"• Lower bound: {lower_bound}\n"
        if upper_bound is not None:
            bounds_info += f"• Upper bound: {upper_bound}\n"
        if unit_of_measure:
            bounds_info += f"• Units: {unit_of_measure}\n"

        tier_specific = {
            "nano": "• Focus on central estimate and basic range\n• Quick percentile assignment\n",
            "mini": "• Consider multiple scenarios for range\n• Balanced uncertainty assessment\n",
            "full": "• Comprehensive uncertainty quantification\n• Detailed percentile reasoning\n"
        }

        return clean_indents(f"""
            You are an elite forecasting expert competing in the Metaculus AI Benchmark Tournament.

            {AntiSlopPrompts.get_base_anti_slop_directives()}

            # NUMERIC FORECASTING DIRECTIVES
            • Provide wide confidence intervals to account for unknown unknowns
            • Base estimates on quantitative evidence when available
            • Consider multiple scenarios (low, medium, high outcomes)
            • Remember: good forecasters are humble about numeric predictions
            • Use reference class forecasting for similar quantities
            {tier_specific.get(model_tier, "")}

            **QUESTION**: {question_text}

            **BACKGROUND**: {background_info}

            **RESOLUTION CRITERIA**: {resolution_criteria}

            **FINE PRINT**: {fine_print}

            {bounds_info}

            **RESEARCH FINDINGS**: {research}

            **TODAY'S DATE**: {datetime.now().strftime("%Y-%m-%d")}

            # ANALYSIS PROTOCOL
            Before providing percentiles, analyze:

            (a) **Time Horizon**: How much time until resolution?
            (b) **Current Baseline**: What's the current value/status?
            (c) **Trend Analysis**: What does current trend suggest?
            (d) **Expert Expectations**: What do markets/experts predict?
            (e) **Low Scenario**: What could cause unexpectedly low outcome?
            (f) **High Scenario**: What could cause unexpectedly high outcome?
            (g) **Reference Class**: What do similar historical cases suggest?

            # UNCERTAINTY REMINDER
            • Good forecasters set WIDE 90/10 confidence intervals
            • Account for unknown unknowns and black swan events
            • Consider that extreme outcomes happen more than expected
            • Base ranges on evidence, not just intuition

            **FINAL PERCENTILES** (in ascending order):
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX

            **Reasoning**: [Brief explanation of range and central estimate with evidence]
        """)

    @staticmethod
    def get_validation_prompt(content: str, task_type: str) -> str:
        """Validation prompt for quality checking responses."""

        return clean_indents(f"""
            You are a quality assurance expert for tournament forecasting.

            # VALIDATION DIRECTIVES
            • Check every factual claim for evidence support
            • Flag any potential hallucinations or unsupported statements
            • Verify logical consistency of reasoning
            • Ensure appropriate uncertainty acknowledgment
            • Keep response concise and deterministic

            **TASK**: Validate this {task_type} response for quality and accuracy.

            **CONTENT TO VALIDATE**:
            {content}

            **VALIDATION CHECKLIST**:
            • Are all factual claims supported by evidence?
            • Is uncertainty appropriately acknowledged?
            • Is the reasoning logically consistent?
            • Are there any obvious errors or hallucinations?
            • Is the response appropriately calibrated?

            **OUTPUT**:
            • Status: [VALID/NEEDS_REVISION]
            • Issues: [List any problems found, or "None identified"]
            • Suggestions: [Brief improvement recommendations if needed]
        """)


# Global instance for easy access
anti_slop_prompts = AntiSlopPrompts()
