"""
Anti-Slop Prompt Engineering System for GPT-5 Tri-Model Router.
Implements tier-specific prompt optimizations with advanced quality guards.
"""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AntiSlopPrompts:
    """
    Advanced anti-slop prompt system with tier-specific optimizations.

    Implements Chain-of-Verification (CoVe), evidence traceability,
    and calibration techniques for GPT-5 models.
    """

    def __init__(self):
        """Initialize anti-slop prompt system."""
        self.base_directives = self._get_base_anti_slop_directives()
        self.tier_optimizations = self._get_tier_optimizations()

    def _get_base_anti_slop_directives(self) -> str:
        """Enhanced base anti-slop directives with latest prompt engineering techniques."""
        return """
# ANTI-SLOP / QUALITY GUARD - CHAIN-OF-VERIFICATION (CoVe)

## INTERNAL REASONING (CoVe Process):
1. Generate initial response internally
2. Verify each claim against evidence
3. Check for logical consistency
4. Identify uncertainty areas
5. Revise and output final response

## EVIDENCE TRACEABILITY PRE-CHECKS:
• Every factual claim MUST trace to specific, verifiable sources
• Cite sources with [Source: URL/Publication] format
• If no source available, state "Based on general knowledge" or "Uncertain"
• Flag any information gaps explicitly

## UNCERTAINTY ACKNOWLEDGMENT:
• Use calibrated confidence language: "likely", "possibly", "uncertain"
• Acknowledge limitations: "Based on available information..."
• Avoid false precision: prefer ranges over exact numbers
• State confidence level when making predictions

## STRUCTURED OUTPUT FORMATTING:
• Use bullet points (•) for clarity and scannability
• Keep responses ≤ 300 words unless complex analysis explicitly required
• Use clear section headers when appropriate
• End with explicit uncertainty statement if applicable

## CALIBRATION INSTRUCTIONS:
• Consider base rates and historical precedents
• Account for your own potential biases and overconfidence
• Provide probability ranges rather than point estimates
• Reference similar cases when available

## FINAL VERIFICATION:
• Pre-check: Does every statement trace to verifiable evidence?
• Question your reasoning: What could go wrong? What am I missing?
• Consider alternative explanations and edge cases
• Maintain helpful, human tone while being precise and honest
"""
    def _get_tier_optimizations(self) -> Dict[str, str]:
        """Tier-specific optimizations for each GPT-5 model."""
        return {
            "nano": """
## GPT-5-NANO OPTIMIZATION:
• Prioritize speed and accuracy over depth
• Focus on essential information only
• Use concise, direct language
• Validate core facts quickly
• Maximum 150 words for efficiency
• Use deterministic responses for consistency
""",
            "mini": """
## GPT-5-MINI OPTIMIZATION:
• Balance depth with efficiency
• Provide moderate detail with good reasoning
• Synthesize information effectively
• Include relevant context
• Use structured analytical approach
• Optimize for research synthesis tasks
""",
            "full": """
## GPT-5-FULL OPTIMIZATION:
• Use maximum reasoning capability
• Provide comprehensive analysis when warranted
• Consider multiple perspectives and scenarios
• Apply sophisticated reasoning patterns
• Integrate complex information with nuanced interpretation
• Apply meta-cognitive reasoning about reasoning quality
"""
        }

    def get_research_prompt(self, question_text: str, model_tier: str = "mini") -> str:
        """Generate research prompt with enhanced anti-slop directives."""
        task_directives = """
## RESEARCH TASK DIRECTIVES:
• MANDATORY: Cite every factual claim with [Source: URL/Publication, Date]
• Prioritize 48-hour news window for recent developments
• Use credible sources: major news outlets, academic papers, official reports
• Acknowledge information gaps explicitly: "No recent data available on..."
• Synthesize information without speculation or extrapolation
• Flag conflicting information from different sources
• Include source reliability assessment when relevant

## EVIDENCE REQUIREMENTS:
• Each claim needs verifiable source attribution
• Distinguish between confirmed facts and reported claims
• Note source publication dates and relevance
• Identify potential source biases or limitations
"""

        tier_optimization = self.tier_optimizations.get(model_tier, "")

        prompt = f"""{self.base_directives}

{task_directives}

{tier_optimization}

## RESEARCH QUESTION:
{question_text}

## OUTPUT REQUIREMENTS:
Provide a comprehensive research summary with:
1. Key findings with source citations
2. Information gaps and limitations
3. Source reliability assessment
4. Uncertainty acknowledgment where appropriate

Format: Use bullet points for clarity, cite all sources, acknowledge limitations.
"""
        return prompt.strip()
    def get_binary_forecast_prompt(self, question_text: str, background_info: str,
                                  resolution_criteria: str, fine_print: str,
                                  research: str, model_tier: str = "full") -> str:
        """Generate binary forecasting prompt with advanced calibration techniques."""
        task_directives = """
## BINARY FORECASTING WITH CALIBRATION:

### SCENARIO ANALYSIS REQUIRED:
• Scenario 1: Status quo continues - what probability?
• Scenario 2: Moderate change occurs - what probability?
• Scenario 3: Significant disruption happens - what probability?

### BASE RATE CONSIDERATION:
• What is the historical frequency of similar events?
• How does this specific case differ from the base rate?
• What reference class should we use for comparison?

### OVERCONFIDENCE REDUCTION:
• Consider what you might be missing or overlooking
• What could make you wrong? List 2-3 specific ways
• Are you anchoring on recent/salient information?
• Widen confidence intervals to account for unknown unknowns

### COMMUNITY PREDICTION ANCHORING:
• If community predictions available, consider their wisdom
• What might the crowd be missing that you see?
• What might you be missing that the crowd sees?
• Adjust for potential systematic biases in community
"""

        tier_optimization = self.tier_optimizations.get(model_tier, "")

        prompt = f"""{self.base_directives}

{task_directives}

{tier_optimization}

## FORECASTING QUESTION:
{question_text}

## BACKGROUND INFORMATION:
{background_info}

## RESOLUTION CRITERIA:
{resolution_criteria}

## FINE PRINT:
{fine_print}

## RESEARCH FINDINGS:
{research}

## REQUIRED OUTPUT FORMAT:
1. Base rate analysis
2. Scenario analysis (status quo, moderate change, disruption)
3. Key factors that could affect outcome
4. Uncertainty acknowledgment and potential blind spots
5. Final calibrated probability with reasoning

End with: "Probability: XX% (Confidence: Low/Medium/High)"
"""
        return prompt.strip()
    def get_multiple_choice_prompt(self, question_text: str, options: List[str],
                                  background_info: str, resolution_criteria: str,
                                  fine_print: str, research: str, model_tier: str = "full") -> str:
        """Generate multiple choice prompt with probability distribution guidance."""
        task_directives = """
## MULTIPLE CHOICE PROBABILITY DISTRIBUTION:

### PROBABILITY DISTRIBUTION GUIDANCE:
• Probabilities must sum to 100% across all options
• Avoid extreme confidence (0% or 100%) unless overwhelming evidence
• Reserve 5-15% probability for unexpected outcomes
• Consider that the "most likely" option often has <50% probability

### EVIDENCE-BASED ASSESSMENT:
• Evaluate each option against available evidence
• Consider historical patterns and precedents for each option
• Account for base rates and typical outcomes in similar situations
• Weight recent vs. historical evidence appropriately

### UNCERTAINTY CALIBRATION:
• Acknowledge which options you're most/least confident about
• Consider interaction effects between options
• Account for potential systematic biases in your reasoning
• Leave room for "black swan" or unexpected developments
"""

        tier_optimization = self.tier_optimizations.get(model_tier, "")
        options_str = "\n".join([f"• {option}" for option in options])

        prompt = f"""{self.base_directives}

{task_directives}

{tier_optimization}

## FORECASTING QUESTION:
{question_text}

## OPTIONS:
{options_str}

## BACKGROUND INFORMATION:
{background_info}

## RESOLUTION CRITERIA:
{resolution_criteria}

## FINE PRINT:
{fine_print}

## RESEARCH FINDINGS:
{research}

## REQUIRED OUTPUT FORMAT:
1. Analysis of each option with supporting evidence
2. Consideration of base rates and historical patterns
3. Uncertainty acknowledgment and potential surprises
4. Final probability distribution with reasoning

End with probabilities in format:
{chr(10).join([f'"{option}": XX%' for option in options])}
(Probabilities must sum to 100%)
"""
        return prompt.strip()
    def get_numeric_forecast_prompt(self, question_text: str, background_info: str,
                                   resolution_criteria: str, fine_print: str,
                                   research: str, unit_of_measure: Optional[str] = None,
                                   lower_bound: Optional[float] = None,
                                   upper_bound: Optional[float] = None,
                                   model_tier: str = "full") -> str:
        """Generate numeric forecasting prompt with advanced uncertainty quantification."""
        task_directives = """
## NUMERIC FORECASTING WITH UNCERTAINTY QUANTIFICATION:

### SCENARIO-BASED ESTIMATION:
• Status Quo Scenario: Current trends continue unchanged
• Moderate Change Scenario: Expected developments occur
• Disruption Scenario: Significant unexpected changes
• Assign probability weights to each scenario

### WIDE CONFIDENCE INTERVALS:
• Account for unknown unknowns with wider intervals
• Consider that most predictions are overconfident
• Use historical forecast accuracy as calibration guide
• Remember: it's better to be roughly right than precisely wrong

### HISTORICAL DATA ANCHORING:
• What do historical patterns suggest?
• How variable have similar metrics been in the past?
• Are there cyclical patterns or trends to consider?
• What expert opinions or forecasts are available?

### UNCERTAINTY ACKNOWLEDGMENT:
• Identify key factors that could drive outcomes higher/lower
• Acknowledge data limitations and measurement challenges
• Consider potential black swan events or regime changes
• Note confidence level in different parts of the distribution
"""

        tier_optimization = self.tier_optimizations.get(model_tier, "")

        bounds_info = ""
        if lower_bound is not None:
            bounds_info += f"Lower bound: {lower_bound}\n"
        if upper_bound is not None:
            bounds_info += f"Upper bound: {upper_bound}\n"
        if unit_of_measure:
            bounds_info += f"Units: {unit_of_measure}\n"

        prompt = f"""{self.base_directives}

{task_directives}

{tier_optimization}

## FORECASTING QUESTION:
{question_text}

## BACKGROUND INFORMATION:
{background_info}

## RESOLUTION CRITERIA:
{resolution_criteria}

## FINE PRINT:
{fine_print}

## CONSTRAINTS:
{bounds_info}

## RESEARCH FINDINGS:
{research}

## REQUIRED OUTPUT FORMAT:
1. Historical context and base rate analysis
2. Scenario analysis with probability weights
3. Key uncertainty factors and potential surprises
4. Confidence assessment for different parts of distribution
5. Final percentile estimates with reasoning

End with percentile estimates:
Percentile 10: XX {unit_of_measure or ""}
Percentile 20: XX {unit_of_measure or ""}
Percentile 40: XX {unit_of_measure or ""}
Percentile 60: XX {unit_of_measure or ""}
Percentile 80: XX {unit_of_measure or ""}
Percentile 90: XX {unit_of_measure or ""}

Confidence Level: Low/Medium/High
"""
        return prompt.strip()


    def get_validation_prompt(self, content: str, task_type: str, model_tier: str = "nano") -> str:
        """Generate validation prompt optimized for fast quality assurance."""
        task_directives = """
## VALIDATION TASK DIRECTIVES:
• Verify factual accuracy of claims against known information
• Check for logical consistency and coherence
• Identify potential hallucinations or unsupported claims
• Flag missing source citations where required
• Assess uncertainty acknowledgment appropriateness
• Provide binary assessment: VALID/INVALID with brief reasoning
"""

        tier_optimization = self.tier_optimizations.get(model_tier, "")

        prompt = f"""{self.base_directives}

{task_directives}

{tier_optimization}

## CONTENT TO VALIDATE:
{content}

## TASK TYPE: {task_type}

## OUTPUT REQUIREMENTS:
1. Overall assessment: VALID/INVALID
2. Specific issues found (if any)
3. Confidence in validation: Low/Medium/High

Keep response concise and focused on validation criteria.
"""
        return prompt.strip()

    def get_base_anti_slop_directives(self) -> str:
        """Public method to get base anti-slop directives."""
        return self.base_directives

    def get_chain_of_verification_prompt(self, response: str, task_type: str) -> str:
        """Generate Chain-of-Verification prompt for response validation."""
        cov_prompt = f"""{self.base_directives}

## CHAIN-OF-VERIFICATION PROTOCOL:

### STEP 1: INITIAL RESPONSE ANALYSIS
Review the following response and identify all factual claims:

{response}

### STEP 2: CLAIM VERIFICATION
For each factual claim identified:
• Verify against known information and sources
• Check for logical consistency with other claims
• Identify any potential contradictions or gaps
• Flag claims that cannot be verified

### STEP 3: EVIDENCE ASSESSMENT
• Are all claims properly sourced and cited?
• Do the sources support the specific claims made?
• Are there any unsupported assertions or speculation?
• Is the reasoning chain logically sound?

### STEP 4: UNCERTAINTY EVALUATION
• Are appropriate uncertainty qualifiers used?
• Is confidence level appropriate for the evidence?
• Are limitations and gaps acknowledged?
• Could alternative interpretations exist?

### STEP 5: REVISED OUTPUT
Based on verification analysis, provide:
1. Verification status: VERIFIED/NEEDS_REVISION/UNCERTAIN
2. Specific issues found (if any)
3. Recommended corrections or improvements
4. Confidence assessment of the original response

Task Type: {task_type}
Focus verification on task-specific quality criteria.
"""
        return cov_prompt.strip()

    def get_meta_reasoning_prompt(self, question: str, forecast: str) -> str:
        """Generate meta-reasoning prompt for forecast quality assessment."""
        meta_prompt = f"""{self.base_directives}

## META-REASONING PROTOCOL:

### ORIGINAL QUESTION:
{question}

### FORECAST TO ANALYZE:
{forecast}

### META-REASONING ANALYSIS:

#### 1. REASONING QUALITY ASSESSMENT:
• Is the reasoning chain logically sound?
• Are all steps in the analysis clearly justified?
• Are there any logical fallacies or biases present?
• Does the conclusion follow from the premises?

#### 2. EVIDENCE EVALUATION:
• Is the evidence base comprehensive and relevant?
• Are sources credible and properly cited?
• Are there significant evidence gaps?
• Is contradictory evidence acknowledged?

#### 3. CALIBRATION ANALYSIS:
• Is the confidence level appropriate for the evidence?
• Are uncertainty bounds realistic?
• Does the forecast show signs of overconfidence?
• Are base rates and historical precedents considered?

#### 4. ALTERNATIVE PERSPECTIVES:
• What alternative viewpoints might exist?
• What could make this forecast wrong?
• Are there unconsidered scenarios or factors?
• How might different experts disagree?

#### 5. IMPROVEMENT RECOMMENDATIONS:
• What additional evidence would strengthen the forecast?
• How could the reasoning be made more robust?
• What are the key uncertainty factors to monitor?
• How should this forecast be updated over time?

### OUTPUT REQUIREMENTS:
1. Overall quality score (1-10)
2. Key strengths of the forecast
3. Main weaknesses or concerns
4. Specific improvement recommendations
5. Confidence in the meta-analysis
"""
        return meta_prompt.strip()

    def adapt_prompt_for_model_capabilities(self, base_prompt: str, model_tier: str,
                                          model_name: str) -> str:
        """Dynamically adapt prompts based on specific OpenRouter model capabilities."""

        # Model-specific adaptations based on known capabilities
        if "gpt-5-nano" in model_name.lower():
            # GPT-5 Nano optimizations
            adaptations = """
## GPT-5-NANO SPECIFIC OPTIMIZATIONS:
• Use clear, structured instructions
• Focus on essential validation and parsing tasks
• Prioritize speed and accuracy over depth
• Use simple, direct language patterns
• Prefer bullet points over paragraph format
"""
        elif "gpt-5-mini" in model_name.lower():
            # GPT-5 Mini optimizations
            adaptations = """
## GPT-5-MINI SPECIFIC OPTIMIZATIONS:
• Leverage balanced reasoning capabilities for synthesis
• Use analytical frameworks and structured thinking
• Balance depth with token efficiency
• Emphasize source integration and citation
• Apply moderate complexity reasoning patterns
"""
        elif "gpt-5" in model_name.lower() and "mini" not in model_name.lower() and "nano" not in model_name.lower():
            # GPT-5 Full optimizations
            adaptations = """
## GPT-5-FULL SPECIFIC OPTIMIZATIONS:
• Utilize maximum reasoning and analysis capabilities
• Apply sophisticated analytical frameworks
• Consider multiple perspectives and meta-reasoning
• Use advanced prompt engineering techniques
• Integrate complex information with nuanced interpretation
"""
        elif "kimi" in model_name.lower() or "oss" in model_name.lower():
            # Free model optimizations
            adaptations = """
## FREE MODEL OPTIMIZATIONS:
• Keep instructions simple and clear
• Focus on essential tasks only
• Use straightforward language patterns
• Minimize complexity to ensure reliability
• Prioritize accuracy over sophistication
"""
        else:
            # Default adaptations
            adaptations = """
## GENERAL MODEL OPTIMIZATIONS:
• Use clear, structured instructions
• Balance complexity with model capabilities
• Focus on accuracy and reliability
• Adapt reasoning depth to model strengths
"""

        # Insert adaptations after base directives
        # Handle both with and without leading newline
        base_directives_clean = self.base_directives.strip()
        if base_directives_clean in base_prompt:
            adapted_prompt = base_prompt.replace(
                base_directives_clean,
                f"{base_directives_clean}\n{adaptations}"
            )
        else:
            # Fallback: just prepend adaptations
            adapted_prompt = f"{adaptations}\n\n{base_prompt}"

        return adapted_prompt

    def get_enhanced_prompt_with_model_adaptation(self, prompt_type: str, model_tier: str,
                                                 model_name: str, **kwargs) -> str:
        """Get enhanced prompt with model-specific adaptations."""

        # Get base prompt based on type
        if prompt_type == "research":
            base_prompt = self.get_research_prompt(
                question_text=kwargs.get("question_text", ""),
                model_tier=model_tier
            )
        elif prompt_type == "binary_forecast":
            base_prompt = self.get_binary_forecast_prompt(
                question_text=kwargs.get("question_text", ""),
                background_info=kwargs.get("background_info", ""),
                resolution_criteria=kwargs.get("resolution_criteria", ""),
                fine_print=kwargs.get("fine_print", ""),
                research=kwargs.get("research", ""),
                model_tier=model_tier
            )
        elif prompt_type == "multiple_choice":
            base_prompt = self.get_multiple_choice_prompt(
                question_text=kwargs.get("question_text", ""),
                options=kwargs.get("options", []),
                background_info=kwargs.get("background_info", ""),
                resolution_criteria=kwargs.get("resolution_criteria", ""),
                fine_print=kwargs.get("fine_print", ""),
                research=kwargs.get("research", ""),
                model_tier=model_tier
            )
        elif prompt_type == "numeric_forecast":
            base_prompt = self.get_numeric_forecast_prompt(
                question_text=kwargs.get("question_text", ""),
                background_info=kwargs.get("background_info", ""),
                resolution_criteria=kwargs.get("resolution_criteria", ""),
                fine_print=kwargs.get("fine_print", ""),
                research=kwargs.get("research", ""),
                unit_of_measure=kwargs.get("unit_of_measure"),
                lower_bound=kwargs.get("lower_bound"),
                upper_bound=kwargs.get("upper_bound"),
                model_tier=model_tier
            )
        elif prompt_type == "validation":
            base_prompt = self.get_validation_prompt(
                content=kwargs.get("content", ""),
                task_type=kwargs.get("task_type", ""),
                model_tier=model_tier
            )
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # Apply model-specific adaptations
        return self.adapt_prompt_for_model_capabilities(base_prompt, model_tier, model_name)


# Global instance
anti_slop_prompts = AntiSlopPrompts()
