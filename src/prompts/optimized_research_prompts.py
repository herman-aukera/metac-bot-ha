"""Optimized research prompt templates for budget-efficient forecasting.

This module provides token-efficient research prompts designed to maximize
accuracy per token spent while maintaining competitive forecasting performance.
"""

from typing import Dict

from jinja2 import Template

from ..domain.entities.question import Question


class OptimizedResearchPrompts:
    """
    Token-efficient research prompt templates optimized for budget constraints.

    These prompts are designed to:
    - Minimize token usage while maintaining quality
    - Request structured output with source citations
    - Focus on factual information relevant to forecasting
    - Adapt to different question complexity levels
    """

    def __init__(self):
        # Simple research template for low-complexity questions
        self.simple_research_template = Template(
            """
Research this question concisely:

Q: {{ question.title }}
Type: {{ question.question_type.value }}
Close: {{ question.close_time.strftime('%Y-%m-%d') }}

Find:
1. Recent news (48h)
2. Key facts
3. Expert views

Format:
RECENT: [2-3 key developments with sources]
FACTS: [3-4 relevant facts]
EXPERTS: [1-2 expert opinions with names/orgs]
SOURCES: [URLs]
"""
        )

        # Standard research template for medium-complexity questions
        self.standard_research_template = Template(
            """
Research for forecasting:

QUESTION: {{ question.title }}
DESCRIPTION: {{ question.description }}
TYPE: {{ question.question_type.value }}
CLOSE: {{ question.close_time.strftime('%Y-%m-%d') }}

Research priorities:
1. Recent developments (last 48 hours)
2. Historical precedents and base rates
3. Expert opinions and official statements
4. Key factors and trends

Output format:
{
  "recent_developments": [
    {"fact": "...", "source": "URL", "date": "YYYY-MM-DD"}
  ],
  "historical_context": [
    {"precedent": "...", "outcome": "...", "relevance": "..."}
  ],
  "expert_opinions": [
    {"expert": "Name/Org", "view": "...", "source": "URL"}
  ],
  "key_factors": ["factor1", "factor2", "factor3"],
  "base_rates": {"similar_event": 0.X}
}
"""
        )

        # Comprehensive research template for high-complexity questions
        self.comprehensive_research_template = Template(
            """
Comprehensive research for complex forecasting question:

QUESTION: {{ question.title }}
DESCRIPTION: {{ question.description }}
TYPE: {{ question.question_type.value }}
{% if question.choices %}CHOICES: {{ question.choices | join(", ") }}{% endif %}
CATEGORIES: {{ question.categories | join(", ") }}
CLOSE: {{ question.close_time.strftime('%Y-%m-%d') }}

Research framework:
1. RECENT (48h): Latest developments, announcements, data
2. TRENDS: Current patterns, momentum, leading indicators
3. PRECEDENTS: Historical cases, base rates, outcomes
4. EXPERTS: Authoritative sources, institutional forecasts
5. FACTORS: Key drivers, dependencies, risks

Required output:
{
  "executive_summary": "2-3 sentence overview",
  "recent_developments": [
    {"development": "...", "impact": "positive/negative/neutral", "source": "URL", "credibility": "high/medium/low"}
  ],
  "trend_analysis": {
    "current_direction": "...",
    "momentum": "accelerating/stable/decelerating",
    "leading_indicators": ["indicator1", "indicator2"]
  },
  "historical_precedents": [
    {"case": "...", "outcome": "...", "similarity": 0.X, "base_rate": 0.X}
  ],
  "expert_consensus": {
    "majority_view": "...",
    "confidence_level": "high/medium/low",
    "key_disagreements": ["..."]
  },
  "critical_factors": [
    {"factor": "...", "impact": "high/medium/low", "direction": "positive/negative"}
  ],
  "sources": ["URL1", "URL2", "URL3"]
}
"""
        )

        # News-focused template for time-sensitive questions
        self.news_focused_template = Template(
            """
Focus on recent news for time-sensitive question:

Q: {{ question.title }}
Close: {{ question.close_time.strftime('%Y-%m-%d') }}

Priority: Last 48 hours only

Find:
- Breaking news
- Official announcements
- Market reactions
- Expert commentary

Format:
NEWS: [3-5 recent developments, newest first]
OFFICIAL: [Government/org statements]
MARKET: [Relevant market/data reactions]
EXPERT: [Recent expert commentary]
SOURCES: [News URLs with dates]

Keep concise - focus on forecast-relevant information only.
"""
        )

        # Base rate research template for questions needing historical context
        self.base_rate_template = Template(
            """
Research historical base rates:

Q: {{ question.title }}
Type: {{ question.question_type.value }}

Find similar historical cases:
1. Identify comparable events/situations
2. Calculate success/failure rates
3. Note key differences from current case

Output:
{
  "similar_cases": [
    {"case": "...", "outcome": "success/failure", "year": YYYY, "similarity": 0.X}
  ],
  "base_rate": 0.XX,
  "sample_size": N,
  "key_differences": ["difference1", "difference2"],
  "confidence": "high/medium/low",
  "sources": ["URL1", "URL2"]
}
"""
        )

    def get_research_prompt(
        self,
        question: Question,
        complexity_level: str = "standard",
        focus_type: str = "general",
    ) -> str:
        """
        Get optimized research prompt based on question complexity and focus type.

        Args:
            question: The forecasting question
            complexity_level: "simple", "standard", or "comprehensive"
            focus_type: "general", "news", or "base_rate"

        Returns:
            Token-optimized research prompt string
        """
        if focus_type == "news":
            return self.news_focused_template.render(question=question)
        elif focus_type == "base_rate":
            return self.base_rate_template.render(question=question)
        elif complexity_level == "simple":
            return self.simple_research_template.render(question=question)
        elif complexity_level == "comprehensive":
            return self.comprehensive_research_template.render(question=question)
        else:  # standard
            return self.standard_research_template.render(question=question)

    def get_simple_research_prompt(self, question: Question) -> str:
        """Get simple research prompt for low-complexity questions."""
        return self.simple_research_template.render(question=question)

    def get_standard_research_prompt(self, question: Question) -> str:
        """Get standard research prompt for medium-complexity questions."""
        return self.standard_research_template.render(question=question)

    def get_comprehensive_research_prompt(self, question: Question) -> str:
        """Get comprehensive research prompt for high-complexity questions."""
        return self.comprehensive_research_template.render(question=question)

    def get_news_focused_prompt(self, question: Question) -> str:
        """Get news-focused prompt for time-sensitive questions."""
        return self.news_focused_template.render(question=question)

    def get_base_rate_prompt(self, question: Question) -> str:
        """Get base rate research prompt for questions needing historical context."""
        return self.base_rate_template.render(question=question)

    def estimate_token_usage(self, complexity_level: str) -> Dict[str, int]:
        """
        Estimate token usage for different prompt types.

        Returns:
            Dictionary with estimated input and expected output tokens
        """
        estimates = {
            "simple": {"input_tokens": 150, "expected_output": 300},
            "standard": {"input_tokens": 250, "expected_output": 500},
            "comprehensive": {"input_tokens": 400, "expected_output": 800},
            "news": {"input_tokens": 120, "expected_output": 250},
            "base_rate": {"input_tokens": 180, "expected_output": 350},
        }
        return estimates.get(complexity_level, estimates["standard"])


class QuestionComplexityAnalyzer:
    """
    Analyzes question complexity to determine appropriate research template.
    """

    @staticmethod
    def analyze_complexity(question: Question) -> str:
        """
        Analyze question complexity and return appropriate level.

        Args:
            question: The forecasting question to analyze

        Returns:
            Complexity level: "simple", "standard", or "comprehensive"
        """
        complexity_score = 0

        # Question length factor
        title_length = len(question.title.split())
        desc_length = len(question.description.split()) if question.description else 0

        if title_length > 15 or desc_length > 100:
            complexity_score += 2
        elif title_length > 10 or desc_length > 50:
            complexity_score += 1

        # Question type factor
        if question.question_type.value in ["NUMERIC", "DATE"]:
            complexity_score += 1
        elif (
            question.question_type.value == "MULTIPLE_CHOICE"
            and len(question.choices or []) > 4
        ):
            complexity_score += 1

        # Category factor (technical/specialized topics)
        technical_categories = [
            "science",
            "technology",
            "economics",
            "politics",
            "medicine",
        ]
        if any(cat.lower() in technical_categories for cat in question.categories):
            complexity_score += 1

        # Time horizon factor
        if question.close_time:
            from datetime import datetime, timezone

            days_until_close = (question.close_time - datetime.now(timezone.utc)).days
            if days_until_close > 365:  # Long-term questions are more complex
                complexity_score += 1

        # Determine complexity level
        if complexity_score <= 1:
            return "simple"
        elif complexity_score <= 3:
            return "standard"
        else:
            return "comprehensive"

    @staticmethod
    def determine_focus_type(question: Question) -> str:
        """
        Determine the appropriate research focus type.

        Args:
            question: The forecasting question to analyze

        Returns:
            Focus type: "general", "news", or "base_rate"
        """
        from datetime import datetime, timezone

        # Check if question is time-sensitive (closes soon)
        if question.close_time:
            days_until_close = (question.close_time - datetime.now(timezone.utc)).days
            if days_until_close <= 30:  # Closes within 30 days
                return "news"

        # Check if question benefits from historical analysis
        historical_keywords = [
            "rate",
            "frequency",
            "typically",
            "usually",
            "historically",
            "average",
            "trend",
            "pattern",
            "precedent",
        ]
        question_text = (question.title + " " + (question.description or "")).lower()

        if any(keyword in question_text for keyword in historical_keywords):
            return "base_rate"

        return "general"
