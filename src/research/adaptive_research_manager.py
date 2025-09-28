"""Adaptive research depth management for budget-efficient forecasting.

This module implements logic to determine optimal research depth based on
question complexity, budget constraints, and quality validation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..domain.entities.question import Question
from ..domain.entities.research_report import ResearchSource


class QuestionComplexityAnalyzer:
    """
    Enhanced question complexity analyzer for adaptive research depth.
    """

    @staticmethod
    def analyze_research_complexity(question: Question) -> Dict[str, Any]:
        """
        Analyze question complexity for research depth determination.

        Returns:
            Dictionary with complexity metrics and recommended research depth
        """
        complexity_factors = {}
        total_score = 0

        # 1. Question length and detail factor
        title_words = len(question.title.split())
        desc_words = len(question.description.split()) if question.description else 0

        if title_words > 20 or desc_words > 150:
            length_score = 3
        elif title_words > 12 or desc_words > 75:
            length_score = 2
        elif title_words > 8 or desc_words > 30:
            length_score = 1
        else:
            length_score = 0

        complexity_factors["length_complexity"] = length_score
        total_score += length_score
        # 2. Question type complexity
        type_complexity = {"BINARY": 1, "MULTIPLE_CHOICE": 2, "NUMERIC": 3, "DATE": 2}
        type_score = type_complexity.get(question.question_type.value, 1)

        # Multiple choice with many options is more complex
        if question.question_type.value == "MULTIPLE_CHOICE" and hasattr(
            question, "choices"
        ):
            if len(question.choices or []) > 5:
                type_score += 1

        complexity_factors["type_complexity"] = type_score
        total_score += type_score

        # 3. Domain/category complexity
        technical_domains = {
            "science": 2,
            "technology": 2,
            "medicine": 3,
            "economics": 2,
            "politics": 2,
            "finance": 2,
            "climate": 3,
            "ai": 2,
            "geopolitics": 3,
            "regulation": 2,
        }

        domain_score = 0
        for category in question.categories:
            for domain, score in technical_domains.items():
                if domain.lower() in category.lower():
                    domain_score = max(domain_score, score)

        complexity_factors["domain_complexity"] = domain_score
        total_score += domain_score

        # 4. Time horizon complexity
        time_score = 0
        if question.close_time:
            days_until_close = (question.close_time - datetime.now(timezone.utc)).days
            if days_until_close > 730:  # > 2 years
                time_score = 3
            elif days_until_close > 365:  # > 1 year
                time_score = 2
            elif days_until_close > 90:  # > 3 months
                time_score = 1

        complexity_factors["time_complexity"] = time_score
        total_score += time_score
        # 5. Interdependency complexity (keywords that suggest complex interactions)
        interdependency_keywords = [
            "depends on",
            "conditional",
            "if and only if",
            "multiple factors",
            "interaction",
            "cascade",
            "systemic",
            "network effect",
            "feedback",
        ]

        question_text = (question.title + " " + (question.description or "")).lower()
        interdependency_score = sum(
            1 for keyword in interdependency_keywords if keyword in question_text
        )
        interdependency_score = min(interdependency_score, 3)  # Cap at 3

        complexity_factors["interdependency_complexity"] = interdependency_score
        total_score += interdependency_score

        # Determine research depth level
        if total_score <= 3:
            research_depth = "shallow"
        elif total_score <= 7:
            research_depth = "standard"
        elif total_score <= 11:
            research_depth = "deep"
        else:
            research_depth = "comprehensive"

        return {
            "total_complexity_score": total_score,
            "complexity_factors": complexity_factors,
            "recommended_research_depth": research_depth,
            "research_priority_areas": QuestionComplexityAnalyzer._identify_priority_areas(
                question, complexity_factors
            ),
        }

    @staticmethod
    def _identify_priority_areas(
        question: Question, complexity_factors: Dict[str, int]
    ) -> List[str]:
        """Identify priority research areas based on complexity factors."""
        priority_areas = []

        # Always include recent developments for time-sensitive questions
        if question.close_time:
            days_until_close = (question.close_time - datetime.now(timezone.utc)).days
            if days_until_close <= 60:
                priority_areas.append("recent_developments_48h")

        # High domain complexity needs expert opinions
        if complexity_factors.get("domain_complexity", 0) >= 2:
            priority_areas.append("expert_opinions")

        # High interdependency needs factor analysis
        if complexity_factors.get("interdependency_complexity", 0) >= 2:
            priority_areas.append("factor_interactions")

        # Long time horizon needs trend analysis
        if complexity_factors.get("time_complexity", 0) >= 2:
            priority_areas.append("trend_analysis")

        # Complex question types need base rates
        if complexity_factors.get("type_complexity", 0) >= 2:
            priority_areas.append("historical_precedents")

        return priority_areas


class AdaptiveResearchManager:
    """
    Manages adaptive research depth based on question complexity and budget constraints.
    """

    def __init__(self, budget_aware: bool = True):
        self.budget_aware = budget_aware
        self.complexity_analyzer = QuestionComplexityAnalyzer()

        # Research depth configurations
        self.research_configs = {
            "shallow": {
                "max_sources": 3,
                "news_window_hours": 24,
                "research_areas": 2,
                "token_budget": 800,
                "estimated_cost": 0.05,
            },
            "standard": {
                "max_sources": 5,
                "news_window_hours": 48,
                "research_areas": 3,
                "token_budget": 1500,
                "estimated_cost": 0.12,
            },
            "deep": {
                "max_sources": 8,
                "news_window_hours": 72,
                "research_areas": 4,
                "token_budget": 2500,
                "estimated_cost": 0.25,
            },
            "comprehensive": {
                "max_sources": 12,
                "news_window_hours": 96,
                "research_areas": 5,
                "token_budget": 4000,
                "estimated_cost": 0.45,
            },
        }

    def determine_research_strategy(
        self, question: Question, budget_remaining: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Determine optimal research strategy based on complexity and budget.

        Args:
            question: The forecasting question
            budget_remaining: Remaining budget in dollars

        Returns:
            Dictionary with research strategy and configuration
        """
        # Analyze question complexity
        complexity_analysis = self.complexity_analyzer.analyze_research_complexity(
            question
        )
        recommended_depth = complexity_analysis["recommended_research_depth"]

        # Apply budget constraints if enabled
        if self.budget_aware and budget_remaining is not None:
            recommended_depth = self._apply_budget_constraints(
                recommended_depth, budget_remaining
            )

        # Get research configuration
        config = self.research_configs[recommended_depth].copy()

        # Customize based on priority areas
        priority_areas = complexity_analysis["research_priority_areas"]
        config["priority_research_areas"] = priority_areas
        config["use_asknews_48h"] = "recent_developments_48h" in priority_areas

        return {
            "research_depth": recommended_depth,
            "complexity_analysis": complexity_analysis,
            "research_config": config,
            "budget_adjusted": budget_remaining is not None
            and recommended_depth != complexity_analysis["recommended_research_depth"],
        }

    def _apply_budget_constraints(
        self, recommended_depth: str, budget_remaining: float
    ) -> str:
        """Apply budget constraints to research depth selection."""
        depth_hierarchy = ["shallow", "standard", "deep", "comprehensive"]
        current_index = depth_hierarchy.index(recommended_depth)

        # Check if we can afford the recommended depth
        estimated_cost = self.research_configs[recommended_depth]["estimated_cost"]

        if budget_remaining < estimated_cost:
            # Find the highest depth we can afford
            for i in range(current_index - 1, -1, -1):
                depth = depth_hierarchy[i]
                if budget_remaining >= self.research_configs[depth]["estimated_cost"]:
                    return depth
            return "shallow"  # Fallback to cheapest option

        return recommended_depth

    def validate_research_quality(
        self, research_sources: List[ResearchSource], research_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate research quality and detect gaps.

        Args:
            research_sources: List of research sources gathered
            research_config: Research configuration used

        Returns:
            Dictionary with quality assessment and gap detection
        """
        quality_metrics = {
            "source_count": len(research_sources),
            "avg_credibility": (
                sum(s.credibility_score for s in research_sources)
                / len(research_sources)
                if research_sources
                else 0
            ),
            "recency_score": self._calculate_recency_score(research_sources),
            "diversity_score": self._calculate_diversity_score(research_sources),
            "coverage_score": self._calculate_coverage_score(
                research_sources, research_config
            ),
        }

        # Detect gaps
        gaps = []
        if quality_metrics["source_count"] < research_config["max_sources"] * 0.6:
            gaps.append("insufficient_sources")
        if quality_metrics["avg_credibility"] < 0.7:
            gaps.append("low_credibility_sources")
        if quality_metrics["recency_score"] < 0.5:
            gaps.append("outdated_information")
        if quality_metrics["diversity_score"] < 0.4:
            gaps.append("limited_perspective_diversity")
        if quality_metrics["coverage_score"] < 0.6:
            gaps.append("incomplete_area_coverage")

        # Overall quality assessment
        overall_score = sum(quality_metrics.values()) / len(quality_metrics)

        quality_level = (
            "high"
            if overall_score >= 0.8
            else "medium"
            if overall_score >= 0.6
            else "low"
        )

        return {
            "quality_metrics": quality_metrics,
            "overall_quality": quality_level,
            "overall_score": overall_score,
            "identified_gaps": gaps,
            "recommendations": self._generate_quality_recommendations(
                gaps, research_config
            ),
        }

    def _calculate_recency_score(self, sources: List[ResearchSource]) -> float:
        """Calculate recency score based on source publication dates."""
        if not sources:
            return 0.0

        now = datetime.now(timezone.utc)
        recency_scores = []

        for source in sources:
            if source.publish_date:
                days_old = (now - source.publish_date).days
                # Score decreases with age: 1.0 for today, 0.5 for 30 days, 0.1 for 365 days
                if days_old <= 1:
                    score = 1.0
                elif days_old <= 7:
                    score = 0.9
                elif days_old <= 30:
                    score = 0.7
                elif days_old <= 90:
                    score = 0.5
                elif days_old <= 365:
                    score = 0.3
                else:
                    score = 0.1
                recency_scores.append(score)
            else:
                recency_scores.append(0.3)  # Unknown date gets medium-low score

        return sum(recency_scores) / len(recency_scores)

    def _calculate_diversity_score(self, sources: List[ResearchSource]) -> float:
        """Calculate diversity score based on source types and domains."""
        if not sources:
            return 0.0

        # Extract domains from URLs
        domains = set()
        for source in sources:
            if source.url:
                try:
                    from urllib.parse import urlparse

                    domain = urlparse(source.url).netloc.lower()
                    domains.add(domain)
                except Exception:
                    pass

        # Diversity is higher with more unique domains
        unique_domains = len(domains)
        total_sources = len(sources)

        if total_sources <= 1:
            return 0.0

        # Perfect diversity would be each source from different domain
        diversity_ratio = min(unique_domains / total_sources, 1.0)

        # Bonus for having sources from different types (news, academic, official, etc.)
        source_types = set()
        for source in sources:
            if source.url:
                url_lower = source.url.lower()
                if any(
                    news in url_lower
                    for news in ["news", "reuters", "bloomberg", "cnn", "bbc"]
                ):
                    source_types.add("news")
                elif any(
                    academic in url_lower
                    for academic in ["arxiv", "scholar", "edu", "research"]
                ):
                    source_types.add("academic")
                elif any(
                    official in url_lower for official in ["gov", "org", "official"]
                ):
                    source_types.add("official")
                else:
                    source_types.add("other")

        type_diversity = len(source_types) / 4  # 4 possible types

        return (diversity_ratio + type_diversity) / 2

    def _calculate_coverage_score(
        self, sources: List[ResearchSource], research_config: Dict[str, Any]
    ) -> float:
        """Calculate how well sources cover priority research areas."""
        priority_areas = research_config.get("priority_research_areas", [])
        if not priority_areas:
            return 1.0  # No specific areas to cover

        coverage_map = {
            "recent_developments_48h": [
                "news",
                "recent",
                "latest",
                "breaking",
                "update",
            ],
            "expert_opinions": [
                "expert",
                "analyst",
                "professor",
                "researcher",
                "opinion",
            ],
            "factor_interactions": [
                "factor",
                "cause",
                "effect",
                "interaction",
                "relationship",
            ],
            "trend_analysis": ["trend", "pattern", "growth", "decline", "trajectory"],
            "historical_precedents": [
                "history",
                "historical",
                "past",
                "precedent",
                "similar",
            ],
        }

        covered_areas = set()
        for source in sources:
            source_text = (source.title + " " + source.summary).lower()
            for area, keywords in coverage_map.items():
                if area in priority_areas and any(
                    keyword in source_text for keyword in keywords
                ):
                    covered_areas.add(area)

        return len(covered_areas) / len(priority_areas) if priority_areas else 1.0

    def _generate_quality_recommendations(
        self, gaps: List[str], research_config: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations to address research quality gaps."""
        recommendations = []

        if "insufficient_sources" in gaps:
            recommendations.append(
                f"Gather additional sources (target: {research_config['max_sources']})"
            )

        if "low_credibility_sources" in gaps:
            recommendations.append(
                "Prioritize high-credibility sources (academic, official, established news)"
            )

        if "outdated_information" in gaps:
            recommendations.append(
                f"Focus on recent sources within {research_config['news_window_hours']}h window"
            )

        if "limited_perspective_diversity" in gaps:
            recommendations.append(
                "Seek sources from different domains and perspectives"
            )

        if "incomplete_area_coverage" in gaps:
            priority_areas = research_config.get("priority_research_areas", [])
            recommendations.append(
                f"Ensure coverage of priority areas: {', '.join(priority_areas)}"
            )

        return recommendations

    def should_use_asknews_48h(
        self, question: Question, research_config: Dict[str, Any]
    ) -> bool:
        """
        Determine if AskNews 48-hour window should be used for this question.

        Args:
            question: The forecasting question
            research_config: Current research configuration

        Returns:
            Boolean indicating whether to use AskNews 48h window
        """
        # Always use if explicitly configured
        if research_config.get("use_asknews_48h", False):
            return True

        # Use for time-sensitive questions
        if question.close_time:
            days_until_close = (question.close_time - datetime.now(timezone.utc)).days
            if days_until_close <= 30:  # Closes within 30 days
                return True

        # Use for questions with recent development keywords
        recent_keywords = [
            "recent",
            "latest",
            "current",
            "today",
            "this week",
            "breaking",
        ]
        question_text = (question.title + " " + (question.description or "")).lower()

        if any(keyword in question_text for keyword in recent_keywords):
            return True

        return False
