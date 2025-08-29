"""Question categorizer service for specialized forecasting strategies."""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..entities.question import Question
from ..value_objects.tournament_strategy import (
    QuestionCategory,
    QuestionPriority,
    TournamentStrategy,
)


@dataclass
class CategoryStrategy:
    """Specialized strategy for a question category."""

    category: QuestionCategory
    research_approach: str
    confidence_adjustment: float
    resource_multiplier: float
    specialized_sources: List[str]
    reasoning_style: str
    validation_requirements: List[str]
    risk_factors: List[str]
    success_indicators: List[str]


@dataclass
class QuestionClassification:
    """Classification result for a question."""

    question_id: UUID
    primary_category: QuestionCategory
    secondary_categories: List[QuestionCategory]
    confidence_score: float
    classification_features: Dict[str, Any]
    recommended_strategy: CategoryStrategy
    resource_allocation_score: float
    complexity_indicators: List[str]


class QuestionCategorizer:
    """
    Service for question category classification and strategy mapping.

    Implements question category classification, strategy mapping,
    category-specific forecasting logic, resource allocation, and
    strategy selection based on question characteristics.
    """

    def __init__(self, llm_client=None):
        """Initialize question categorizer with specialized strategies."""
        self.llm_client = llm_client
        self._category_strategies = self._initialize_category_strategies()
        self._classification_cache: Dict[UUID, QuestionClassification] = {}
        self._keyword_patterns = self._initialize_keyword_patterns()
        self._complexity_indicators = self._initialize_complexity_indicators()

    def classify_question(
        self, question: Question, context: Optional[Dict[str, Any]] = None
    ) -> QuestionClassification:
        """
        Classify question and determine appropriate strategy.

        Args:
            question: Question to classify
            context: Additional context for classification

        Returns:
            Question classification with recommended strategy
        """
        # Check cache first
        if question.id in self._classification_cache:
            return self._classification_cache[question.id]

        # Perform multi-level classification
        primary_category, confidence = self._classify_primary_category(question)
        secondary_categories = self._identify_secondary_categories(
            question, primary_category
        )

        # Extract classification features
        features = self._extract_classification_features(question)

        # Get recommended strategy
        recommended_strategy = self._get_category_strategy(
            primary_category, question, context
        )

        # Calculate resource allocation score
        resource_score = self._calculate_resource_allocation_score(
            question, primary_category, features, context
        )

        # Identify complexity indicators
        complexity_indicators = self._identify_complexity_indicators(question, features)

        # Create classification result
        classification = QuestionClassification(
            question_id=question.id,
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            confidence_score=confidence,
            classification_features=features,
            recommended_strategy=recommended_strategy,
            resource_allocation_score=resource_score,
            complexity_indicators=complexity_indicators,
        )

        # Cache result
        self._classification_cache[question.id] = classification

        return classification

    def get_specialized_strategy(
        self,
        category: QuestionCategory,
        question: Optional[Question] = None,
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> CategoryStrategy:
        """
        Get specialized strategy for a category.

        Args:
            category: Question category
            question: Specific question for customization
            tournament_context: Tournament context for strategy adaptation

        Returns:
            Specialized category strategy
        """
        base_strategy = self._category_strategies[category]

        if question is None and tournament_context is None:
            return base_strategy

        # Customize strategy based on question and context
        return self._customize_strategy(base_strategy, question, tournament_context)

    def allocate_resources(
        self,
        questions: List[Question],
        total_resources: float = 1.0,
        tournament_strategy: Optional[TournamentStrategy] = None,
    ) -> Dict[UUID, float]:
        """
        Allocate resources across questions based on categories and priorities.

        Args:
            questions: List of questions to allocate resources for
            total_resources: Total resource budget (normalized to 1.0)
            tournament_strategy: Tournament strategy for allocation weights

        Returns:
            Resource allocation mapping question_id -> resource_amount
        """
        if not questions:
            return {}

        # Classify all questions
        classifications = [self.classify_question(q) for q in questions]

        # Calculate base allocation scores
        allocation_scores = {}
        total_score = 0.0

        for classification in classifications:
            question = next(q for q in questions if q.id == classification.question_id)

            # Base score from classification
            base_score = classification.resource_allocation_score

            # Adjust based on tournament strategy
            if tournament_strategy:
                category_weight = tournament_strategy.category_specializations.get(
                    classification.primary_category, 0.5
                )
                base_score *= category_weight

            # Adjust based on question characteristics
            scoring_potential = question.calculate_scoring_potential()
            difficulty = question.calculate_difficulty_score()

            # Higher allocation for high potential, moderate difficulty
            potential_multiplier = scoring_potential * 1.5
            difficulty_multiplier = 1.0 + (
                0.5 - abs(difficulty - 0.5)
            )  # Peak at 0.5 difficulty

            final_score = base_score * potential_multiplier * difficulty_multiplier
            allocation_scores[classification.question_id] = final_score
            total_score += final_score

        # Normalize allocations
        if total_score > 0:
            return {
                question_id: (score / total_score) * total_resources
                for question_id, score in allocation_scores.items()
            }
        else:
            # Equal allocation fallback
            equal_allocation = total_resources / len(questions)
            return {q.id: equal_allocation for q in questions}

    def select_optimal_strategy(
        self,
        question: Question,
        available_agents: List[str],
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Select optimal strategy based on question characteristics.

        Args:
            question: Question to strategize for
            available_agents: List of available reasoning agents
            tournament_context: Tournament context for strategy selection

        Returns:
            Optimal strategy configuration
        """
        classification = self.classify_question(question, tournament_context)
        category_strategy = classification.recommended_strategy

        # Select reasoning agents based on category
        recommended_agents = self._select_reasoning_agents(
            classification.primary_category, available_agents, question
        )

        # Determine research depth
        research_depth = self._determine_research_depth(
            classification, question, tournament_context
        )

        # Configure validation requirements
        validation_config = self._configure_validation(
            category_strategy, classification, tournament_context
        )

        # Set confidence thresholds
        confidence_thresholds = self._set_confidence_thresholds(
            classification, tournament_context
        )

        return {
            "primary_category": classification.primary_category.value,
            "recommended_agents": recommended_agents,
            "research_depth": research_depth,
            "validation_config": validation_config,
            "confidence_thresholds": confidence_thresholds,
            "resource_allocation": classification.resource_allocation_score,
            "specialized_sources": category_strategy.specialized_sources,
            "risk_factors": category_strategy.risk_factors,
            "success_indicators": category_strategy.success_indicators,
            "reasoning_style": category_strategy.reasoning_style,
        }

    def update_category_performance(
        self, category: QuestionCategory, performance_data: Dict[str, Any]
    ) -> None:
        """
        Update category strategy based on performance feedback.

        Args:
            category: Category to update
            performance_data: Performance metrics and feedback
        """
        if category not in self._category_strategies:
            return

        current_strategy = self._category_strategies[category]

        # Adjust confidence based on performance
        accuracy = performance_data.get("accuracy", 0.5)
        if accuracy > 0.7:
            # Good performance - slightly lower confidence adjustment (more aggressive)
            new_confidence_adjustment = max(
                -0.2, current_strategy.confidence_adjustment - 0.05
            )
        elif accuracy < 0.4:
            # Poor performance - higher confidence adjustment (more conservative)
            new_confidence_adjustment = min(
                0.2, current_strategy.confidence_adjustment + 0.05
            )
        else:
            new_confidence_adjustment = current_strategy.confidence_adjustment

        # Adjust resource multiplier based on efficiency
        efficiency = performance_data.get("efficiency", 0.5)
        if efficiency > 0.7:
            new_resource_multiplier = min(
                2.0, current_strategy.resource_multiplier + 0.1
            )
        elif efficiency < 0.4:
            new_resource_multiplier = max(
                0.5, current_strategy.resource_multiplier - 0.1
            )
        else:
            new_resource_multiplier = current_strategy.resource_multiplier

        # Update strategy
        self._category_strategies[category] = CategoryStrategy(
            category=category,
            research_approach=current_strategy.research_approach,
            confidence_adjustment=new_confidence_adjustment,
            resource_multiplier=new_resource_multiplier,
            specialized_sources=current_strategy.specialized_sources,
            reasoning_style=current_strategy.reasoning_style,
            validation_requirements=current_strategy.validation_requirements,
            risk_factors=current_strategy.risk_factors,
            success_indicators=current_strategy.success_indicators,
        )

    def get_category_insights(
        self,
        questions: List[Question],
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get insights about category distribution and opportunities.

        Args:
            questions: List of questions to analyze
            tournament_context: Tournament context for insights

        Returns:
            Category insights and recommendations
        """
        if not questions:
            return {}

        # Classify all questions
        classifications = [
            self.classify_question(q, tournament_context) for q in questions
        ]

        # Analyze category distribution
        category_counts = defaultdict(int)
        category_complexities = defaultdict(list)
        category_potentials = defaultdict(list)

        for classification in classifications:
            category = classification.primary_category
            category_counts[category] += 1

            question = next(q for q in questions if q.id == classification.question_id)
            category_complexities[category].append(
                question.calculate_difficulty_score()
            )
            category_potentials[category].append(question.calculate_scoring_potential())

        # Generate insights
        insights = {
            "total_questions": len(questions),
            "category_distribution": dict(category_counts),
            "category_analysis": {},
            "recommendations": [],
            "opportunities": [],
        }

        for category, count in category_counts.items():
            proportion = count / len(questions)
            avg_complexity = sum(category_complexities[category]) / len(
                category_complexities[category]
            )
            avg_potential = sum(category_potentials[category]) / len(
                category_potentials[category]
            )

            insights["category_analysis"][category.value] = {
                "count": count,
                "proportion": proportion,
                "average_complexity": avg_complexity,
                "average_potential": avg_potential,
                "strategy": self._category_strategies[category].research_approach,
            }

            # Generate recommendations
            if proportion > 0.3:
                insights["recommendations"].append(
                    f"High concentration in {category.value} ({proportion:.1%}) - consider specialization"
                )

            if avg_potential > 0.7 and avg_complexity < 0.6:
                insights["opportunities"].append(
                    f"{category.value} shows high potential with moderate complexity"
                )

        return insights

    def _initialize_category_strategies(
        self,
    ) -> Dict[QuestionCategory, CategoryStrategy]:
        """Initialize specialized strategies for each category."""
        strategies = {}

        strategies[QuestionCategory.TECHNOLOGY] = CategoryStrategy(
            category=QuestionCategory.TECHNOLOGY,
            research_approach="technical_analysis",
            confidence_adjustment=0.05,  # Slightly more confident
            resource_multiplier=1.2,
            specialized_sources=[
                "arxiv",
                "tech_news",
                "patent_databases",
                "github_trends",
            ],
            reasoning_style="systematic_decomposition",
            validation_requirements=[
                "technical_feasibility",
                "adoption_timeline",
                "market_readiness",
            ],
            risk_factors=[
                "rapid_technological_change",
                "regulatory_uncertainty",
                "market_volatility",
            ],
            success_indicators=[
                "clear_technical_metrics",
                "industry_consensus",
                "historical_precedent",
            ],
        )

        strategies[QuestionCategory.ECONOMICS] = CategoryStrategy(
            category=QuestionCategory.ECONOMICS,
            research_approach="quantitative_modeling",
            confidence_adjustment=0.0,  # Neutral
            resource_multiplier=1.3,
            specialized_sources=[
                "economic_indicators",
                "central_bank_data",
                "market_analysis",
                "academic_papers",
            ],
            reasoning_style="data_driven_analysis",
            validation_requirements=[
                "statistical_significance",
                "economic_theory_alignment",
                "historical_validation",
            ],
            risk_factors=[
                "market_volatility",
                "policy_changes",
                "external_shocks",
                "measurement_uncertainty",
            ],
            success_indicators=[
                "strong_data_support",
                "economic_model_consensus",
                "leading_indicator_alignment",
            ],
        )

        strategies[QuestionCategory.POLITICS] = CategoryStrategy(
            category=QuestionCategory.POLITICS,
            research_approach="multi_source_synthesis",
            confidence_adjustment=-0.1,  # More conservative due to volatility
            resource_multiplier=1.4,
            specialized_sources=[
                "polling_data",
                "political_analysis",
                "historical_elections",
                "expert_opinions",
            ],
            reasoning_style="probabilistic_reasoning",
            validation_requirements=[
                "polling_methodology_review",
                "historical_precedent_analysis",
                "expert_consensus",
            ],
            risk_factors=[
                "polling_errors",
                "unexpected_events",
                "voter_behavior_changes",
                "media_influence",
            ],
            success_indicators=[
                "consistent_polling_trends",
                "historical_pattern_match",
                "expert_agreement",
            ],
        )

        strategies[QuestionCategory.HEALTH] = CategoryStrategy(
            category=QuestionCategory.HEALTH,
            research_approach="evidence_based_medicine",
            confidence_adjustment=0.05,
            resource_multiplier=1.1,
            specialized_sources=[
                "medical_journals",
                "clinical_trials",
                "health_organizations",
                "epidemiological_data",
            ],
            reasoning_style="systematic_review",
            validation_requirements=[
                "peer_review_status",
                "sample_size_adequacy",
                "methodology_quality",
            ],
            risk_factors=[
                "study_limitations",
                "publication_bias",
                "regulatory_changes",
                "population_variability",
            ],
            success_indicators=[
                "large_sample_studies",
                "peer_reviewed_evidence",
                "regulatory_approval",
            ],
        )

        strategies[QuestionCategory.CLIMATE] = CategoryStrategy(
            category=QuestionCategory.CLIMATE,
            research_approach="scientific_consensus",
            confidence_adjustment=0.0,
            resource_multiplier=1.2,
            specialized_sources=[
                "climate_models",
                "scientific_papers",
                "ipcc_reports",
                "environmental_data",
            ],
            reasoning_style="model_ensemble",
            validation_requirements=[
                "model_validation",
                "peer_review",
                "uncertainty_quantification",
            ],
            risk_factors=[
                "model_uncertainty",
                "feedback_loops",
                "tipping_points",
                "measurement_challenges",
            ],
            success_indicators=[
                "model_consensus",
                "observational_support",
                "physical_understanding",
            ],
        )

        strategies[QuestionCategory.SCIENCE] = CategoryStrategy(
            category=QuestionCategory.SCIENCE,
            research_approach="peer_review_analysis",
            confidence_adjustment=0.1,
            resource_multiplier=1.0,
            specialized_sources=[
                "scientific_journals",
                "research_databases",
                "expert_networks",
                "conference_proceedings",
            ],
            reasoning_style="hypothesis_testing",
            validation_requirements=[
                "peer_review",
                "replication_studies",
                "statistical_power",
            ],
            risk_factors=[
                "replication_crisis",
                "publication_bias",
                "funding_bias",
                "methodological_issues",
            ],
            success_indicators=[
                "peer_reviewed_publication",
                "independent_replication",
                "expert_consensus",
            ],
        )

        strategies[QuestionCategory.GEOPOLITICS] = CategoryStrategy(
            category=QuestionCategory.GEOPOLITICS,
            research_approach="intelligence_analysis",
            confidence_adjustment=-0.15,  # Very conservative due to high uncertainty
            resource_multiplier=1.5,
            specialized_sources=[
                "intelligence_reports",
                "diplomatic_sources",
                "regional_experts",
                "historical_analysis",
            ],
            reasoning_style="scenario_planning",
            validation_requirements=[
                "multiple_source_confirmation",
                "expert_validation",
                "historical_precedent",
            ],
            risk_factors=[
                "information_uncertainty",
                "rapid_developments",
                "deception_operations",
                "cultural_misunderstanding",
            ],
            success_indicators=[
                "intelligence_consensus",
                "historical_pattern_match",
                "expert_agreement",
            ],
        )

        strategies[QuestionCategory.BUSINESS] = CategoryStrategy(
            category=QuestionCategory.BUSINESS,
            research_approach="market_analysis",
            confidence_adjustment=0.0,
            resource_multiplier=1.1,
            specialized_sources=[
                "financial_reports",
                "market_research",
                "industry_analysis",
                "expert_opinions",
            ],
            reasoning_style="competitive_analysis",
            validation_requirements=[
                "financial_data_verification",
                "market_trend_analysis",
                "competitive_positioning",
            ],
            risk_factors=[
                "market_volatility",
                "competitive_dynamics",
                "regulatory_changes",
                "economic_conditions",
            ],
            success_indicators=[
                "strong_financials",
                "market_trend_support",
                "competitive_advantage",
            ],
        )

        strategies[QuestionCategory.SPORTS] = CategoryStrategy(
            category=QuestionCategory.SPORTS,
            research_approach="statistical_modeling",
            confidence_adjustment=0.0,
            resource_multiplier=0.8,
            specialized_sources=[
                "sports_statistics",
                "performance_data",
                "injury_reports",
                "expert_analysis",
            ],
            reasoning_style="statistical_prediction",
            validation_requirements=[
                "statistical_significance",
                "performance_trend_analysis",
                "injury_status",
            ],
            risk_factors=[
                "injury_uncertainty",
                "performance_variability",
                "external_factors",
                "psychological_factors",
            ],
            success_indicators=[
                "consistent_performance",
                "statistical_significance",
                "expert_consensus",
            ],
        )

        strategies[QuestionCategory.ENTERTAINMENT] = CategoryStrategy(
            category=QuestionCategory.ENTERTAINMENT,
            research_approach="trend_analysis",
            confidence_adjustment=-0.05,
            resource_multiplier=0.9,
            specialized_sources=[
                "industry_reports",
                "social_media_trends",
                "box_office_data",
                "critic_reviews",
            ],
            reasoning_style="trend_extrapolation",
            validation_requirements=[
                "trend_consistency",
                "market_data_support",
                "industry_expert_input",
            ],
            risk_factors=[
                "taste_volatility",
                "marketing_impact",
                "competition_effects",
                "cultural_shifts",
            ],
            success_indicators=[
                "strong_trend_support",
                "industry_backing",
                "market_data_alignment",
            ],
        )

        strategies[QuestionCategory.SOCIAL] = CategoryStrategy(
            category=QuestionCategory.SOCIAL,
            research_approach="sociological_analysis",
            confidence_adjustment=-0.1,
            resource_multiplier=1.2,
            specialized_sources=[
                "social_surveys",
                "demographic_data",
                "academic_research",
                "trend_analysis",
            ],
            reasoning_style="multi_factor_analysis",
            validation_requirements=[
                "survey_methodology",
                "sample_representativeness",
                "trend_validation",
            ],
            risk_factors=[
                "social_complexity",
                "measurement_challenges",
                "cultural_factors",
                "behavioral_unpredictability",
            ],
            success_indicators=[
                "robust_survey_data",
                "consistent_trends",
                "academic_support",
            ],
        )

        strategies[QuestionCategory.OTHER] = CategoryStrategy(
            category=QuestionCategory.OTHER,
            research_approach="general_research",
            confidence_adjustment=-0.05,
            resource_multiplier=1.0,
            specialized_sources=[
                "general_search",
                "news_sources",
                "expert_opinions",
                "academic_databases",
            ],
            reasoning_style="comprehensive_analysis",
            validation_requirements=[
                "source_diversity",
                "information_quality",
                "expert_validation",
            ],
            risk_factors=[
                "information_uncertainty",
                "domain_unfamiliarity",
                "source_reliability",
            ],
            success_indicators=[
                "source_consensus",
                "information_quality",
                "expert_validation",
            ],
        )

        return strategies

    def _initialize_keyword_patterns(self) -> Dict[QuestionCategory, List[str]]:
        """Initialize keyword patterns for category classification."""
        patterns = {
            QuestionCategory.TECHNOLOGY: [
                r"\b(ai|artificial intelligence|machine learning|algorithm|software|tech|digital|internet|blockchain|cryptocurrency|robot|automation|computer|programming|data|cloud|cyber)\b",
                r"\b(startup|silicon valley|tech company|innovation|patent|app|platform|api|database|server|network|security|privacy)\b",
            ],
            QuestionCategory.ECONOMICS: [
                r"\b(economy|economic|gdp|inflation|recession|market|finance|financial|stock|price|trade|currency|dollar|euro|yen|bitcoin)\b",
                r"\b(bank|banking|federal reserve|interest rate|unemployment|employment|wage|salary|income|debt|deficit|budget|fiscal|monetary)\b",
            ],
            QuestionCategory.POLITICS: [
                r"\b(election|political|government|policy|president|congress|senate|house|vote|voting|democracy|republican|democrat|party|campaign)\b",
                r"\b(law|legislation|bill|act|supreme court|judge|justice|constitution|amendment|impeachment|scandal|approval rating)\b",
            ],
            QuestionCategory.HEALTH: [
                r"\b(health|medical|medicine|disease|pandemic|epidemic|vaccine|vaccination|hospital|doctor|patient|treatment|therapy|drug|pharmaceutical)\b",
                r"\b(covid|coronavirus|virus|bacteria|infection|outbreak|mortality|morbidity|clinical trial|fda|who|cdc|diagnosis|symptom)\b",
            ],
            QuestionCategory.CLIMATE: [
                r"\b(climate|environment|environmental|carbon|emission|temperature|global warming|greenhouse|renewable|energy|pollution|green|sustainability)\b",
                r"\b(weather|hurricane|tornado|flood|drought|wildfire|sea level|ice|glacier|arctic|antarctic|fossil fuel|solar|wind)\b",
            ],
            QuestionCategory.SCIENCE: [
                r"\b(science|scientific|research|study|experiment|discovery|theory|hypothesis|physics|chemistry|biology|astronomy|space|nasa)\b",
                r"\b(nobel prize|peer review|journal|publication|laboratory|scientist|researcher|breakthrough|innovation|technology|quantum)\b",
            ],
            QuestionCategory.GEOPOLITICS: [
                r"\b(war|conflict|international|country|nation|diplomacy|treaty|sanctions|military|peace|alliance|nato|un|united nations)\b",
                r"\b(china|russia|usa|europe|middle east|africa|asia|trade war|nuclear|terrorism|refugee|immigration|border)\b",
            ],
            QuestionCategory.BUSINESS: [
                r"\b(business|company|corporation|startup|revenue|profit|loss|merger|acquisition|ipo|ceo|cfo|stock|share|market cap)\b",
                r"\b(earnings|quarterly|annual|financial|report|investor|shareholder|dividend|bankruptcy|restructuring|layoff|hiring)\b",
            ],
            QuestionCategory.SPORTS: [
                r"\b(sport|sports|game|match|tournament|championship|league|team|player|athlete|coach|season|playoff|final)\b",
                r"\b(football|soccer|basketball|baseball|tennis|golf|olympics|world cup|nfl|nba|mlb|nhl|fifa|record|score)\b",
            ],
            QuestionCategory.ENTERTAINMENT: [
                r"\b(movie|film|tv|television|show|series|actor|actress|director|producer|studio|box office|rating|award|oscar|emmy)\b",
                r"\b(music|album|song|artist|singer|band|concert|tour|streaming|netflix|disney|hollywood|celebrity|fame)\b",
            ],
            QuestionCategory.SOCIAL: [
                r"\b(social|society|culture|cultural|demographic|population|community|public|citizen|people|human|behavior|psychology)\b",
                r"\b(survey|poll|opinion|trend|movement|protest|activism|rights|equality|diversity|inclusion|gender|race|age)\b",
            ],
        }
        return patterns

    def _initialize_complexity_indicators(self) -> Dict[str, List[str]]:
        """Initialize complexity indicators for questions."""
        return {
            "high_complexity": [
                "multiple variables",
                "long-term prediction",
                "novel situation",
                "limited historical data",
                "expert disagreement",
                "regulatory uncertainty",
                "technological disruption",
                "geopolitical instability",
                "market volatility",
            ],
            "medium_complexity": [
                "established patterns",
                "some historical data",
                "moderate expert consensus",
                "known variables",
                "medium-term prediction",
                "stable environment",
            ],
            "low_complexity": [
                "clear patterns",
                "abundant data",
                "expert consensus",
                "well-understood domain",
                "short-term prediction",
                "stable conditions",
            ],
        }

    def _classify_primary_category(
        self, question: Question
    ) -> Tuple[QuestionCategory, float]:
        """Classify the primary category of a question."""
        # Use existing categorization if available
        if question.question_category:
            return question.question_category, 0.9

        # Use built-in categorization method
        category = question.categorize_question()

        # Calculate confidence based on keyword matching
        confidence = self._calculate_classification_confidence(question, category)

        return category, confidence

    def _calculate_classification_confidence(
        self, question: Question, category: QuestionCategory
    ) -> float:
        """Calculate confidence in category classification."""
        text = f"{question.title} {question.description}".lower()

        # Check keyword pattern matches
        patterns = self._keyword_patterns.get(category, [])
        matches = 0
        total_patterns = len(patterns)

        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1

        if total_patterns == 0:
            return 0.5  # Default confidence

        # Base confidence from pattern matching
        pattern_confidence = matches / total_patterns

        # Adjust based on text length and specificity
        text_length = len(text.split())
        if text_length > 100:
            length_bonus = 0.1  # More text = more confidence
        elif text_length < 20:
            length_bonus = -0.1  # Less text = less confidence
        else:
            length_bonus = 0.0

        final_confidence = min(0.95, max(0.1, pattern_confidence + length_bonus))
        return final_confidence

    def _identify_secondary_categories(
        self, question: Question, primary_category: QuestionCategory
    ) -> List[QuestionCategory]:
        """Identify secondary categories for cross-domain questions."""
        secondary = []
        text = f"{question.title} {question.description}".lower()

        for category, patterns in self._keyword_patterns.items():
            if category == primary_category:
                continue

            matches = sum(
                1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE)
            )
            if (
                matches > 0 and matches / len(patterns) > 0.3
            ):  # Threshold for secondary category
                secondary.append(category)

        return secondary[:2]  # Limit to top 2 secondary categories

    def _extract_classification_features(self, question: Question) -> Dict[str, Any]:
        """Extract features used for classification."""
        text = f"{question.title} {question.description}".lower()

        features = {
            "text_length": len(text.split()),
            "question_type": question.question_type.value,
            "has_numeric_range": question.min_value is not None
            and question.max_value is not None,
            "has_choices": question.choices is not None
            and len(question.choices or []) > 0,
            "days_to_close": question.days_until_close(),
            "title_length": len(question.title.split()),
            "description_length": len(question.description.split()),
            "contains_numbers": bool(re.search(r"\d+", text)),
            "contains_dates": bool(
                re.search(r"\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", text)
            ),
            "contains_percentages": bool(re.search(r"\d+%|\bpercent\b", text)),
            "question_words": len(
                re.findall(
                    r"\b(what|when|where|who|why|how|will|would|could|should)\b", text
                )
            ),
            "uncertainty_words": len(
                re.findall(
                    r"\b(might|maybe|possibly|likely|unlikely|uncertain|probable)\b",
                    text,
                )
            ),
        }

        return features

    def _get_category_strategy(
        self,
        category: QuestionCategory,
        question: Question,
        context: Optional[Dict[str, Any]],
    ) -> CategoryStrategy:
        """Get category strategy, potentially customized for the specific question."""
        base_strategy = self._category_strategies[category]

        if context is None:
            return base_strategy

        return self._customize_strategy(base_strategy, question, context)

    def _customize_strategy(
        self,
        base_strategy: CategoryStrategy,
        question: Optional[Question],
        context: Optional[Dict[str, Any]],
    ) -> CategoryStrategy:
        """Customize strategy based on question and context."""
        if question is None and context is None:
            return base_strategy

        # Start with base strategy values
        confidence_adjustment = base_strategy.confidence_adjustment
        resource_multiplier = base_strategy.resource_multiplier

        # Adjust based on question characteristics
        if question:
            difficulty = question.calculate_difficulty_score()
            if difficulty > 0.8:
                confidence_adjustment -= (
                    0.1  # More conservative for difficult questions
                )
                resource_multiplier += 0.2  # More resources for difficult questions
            elif difficulty < 0.3:
                confidence_adjustment += (
                    0.05  # Slightly more confident for easy questions
                )

        # Adjust based on tournament context
        if context:
            tournament_phase = context.get("tournament_phase")
            if tournament_phase == "late":
                confidence_adjustment -= 0.05  # More conservative late in tournament
            elif tournament_phase == "early":
                confidence_adjustment += 0.05  # More aggressive early in tournament

        return CategoryStrategy(
            category=base_strategy.category,
            research_approach=base_strategy.research_approach,
            confidence_adjustment=confidence_adjustment,
            resource_multiplier=resource_multiplier,
            specialized_sources=base_strategy.specialized_sources,
            reasoning_style=base_strategy.reasoning_style,
            validation_requirements=base_strategy.validation_requirements,
            risk_factors=base_strategy.risk_factors,
            success_indicators=base_strategy.success_indicators,
        )

    def _calculate_resource_allocation_score(
        self,
        question: Question,
        category: QuestionCategory,
        features: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate resource allocation score for the question."""
        base_score = 0.5

        # Adjust based on category strategy
        strategy = self._category_strategies[category]
        base_score *= strategy.resource_multiplier

        # Adjust based on question characteristics
        difficulty = question.calculate_difficulty_score()
        scoring_potential = question.calculate_scoring_potential()

        # Higher resources for high potential questions
        base_score += scoring_potential * 0.3

        # Adjust based on difficulty (moderate difficulty gets most resources)
        if 0.4 <= difficulty <= 0.7:
            base_score += 0.2  # Sweet spot
        elif difficulty > 0.8:
            base_score += 0.1  # High difficulty needs resources but may not pay off

        # Adjust based on time to close
        days_to_close = features.get("days_to_close", 30)
        if days_to_close <= 3:
            base_score += 0.2  # Urgent questions need immediate resources
        elif days_to_close > 90:
            base_score -= 0.1  # Long-term questions can wait

        # Tournament context adjustments
        if context:
            competition_level = context.get("competition_level", 0.5)
            base_score += (
                1 - competition_level
            ) * 0.1  # More resources when less competition

        return min(1.0, max(0.1, base_score))

    def _identify_complexity_indicators(
        self, question: Question, features: Dict[str, Any]
    ) -> List[str]:
        """Identify complexity indicators for the question."""
        indicators = []

        # Check text-based indicators
        text = f"{question.title} {question.description}".lower()

        for complexity_level, keywords in self._complexity_indicators.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    indicators.append(f"{complexity_level}: {keyword}")

        # Check feature-based indicators
        if features.get("days_to_close", 30) > 365:
            indicators.append("high_complexity: long-term prediction")

        if features.get("uncertainty_words", 0) > 3:
            indicators.append("high_complexity: high uncertainty language")

        if (
            question.question_type.value == "numeric"
            and question.max_value
            and question.min_value
        ):
            range_size = question.max_value - question.min_value
            if range_size > 1000:
                indicators.append("medium_complexity: wide numeric range")

        return indicators

    def _select_reasoning_agents(
        self,
        category: QuestionCategory,
        available_agents: List[str],
        question: Question,
    ) -> List[str]:
        """Select optimal reasoning agents for the category."""
        strategy = self._category_strategies[category]
        reasoning_style = strategy.reasoning_style

        # Map reasoning styles to agent preferences
        agent_preferences = {
            "systematic_decomposition": ["tree_of_thought", "chain_of_thought"],
            "data_driven_analysis": ["chain_of_thought", "react"],
            "probabilistic_reasoning": ["ensemble", "chain_of_thought"],
            "systematic_review": ["tree_of_thought", "chain_of_thought"],
            "model_ensemble": ["ensemble", "tree_of_thought"],
            "hypothesis_testing": ["react", "chain_of_thought"],
            "scenario_planning": ["tree_of_thought", "react"],
            "competitive_analysis": ["chain_of_thought", "react"],
            "statistical_prediction": ["chain_of_thought", "ensemble"],
            "trend_extrapolation": ["react", "chain_of_thought"],
            "multi_factor_analysis": ["tree_of_thought", "ensemble"],
            "comprehensive_analysis": [
                "ensemble",
                "tree_of_thought",
                "chain_of_thought",
            ],
        }

        preferred_agents = agent_preferences.get(reasoning_style, ["chain_of_thought"])

        # Filter by available agents
        selected_agents = [
            agent for agent in preferred_agents if agent in available_agents
        ]

        # Ensure we have at least one agent
        if not selected_agents and available_agents:
            selected_agents = [available_agents[0]]

        return selected_agents

    def _determine_research_depth(
        self,
        classification: QuestionClassification,
        question: Question,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Determine appropriate research depth."""
        base_depth = "medium"

        # Adjust based on resource allocation
        if classification.resource_allocation_score > 0.8:
            base_depth = "deep"
        elif classification.resource_allocation_score < 0.3:
            base_depth = "shallow"

        # Adjust based on complexity
        high_complexity_indicators = [
            ind
            for ind in classification.complexity_indicators
            if ind.startswith("high_complexity")
        ]

        if len(high_complexity_indicators) > 2:
            base_depth = "deep"
        elif len(high_complexity_indicators) == 0:
            base_depth = "shallow"

        # Tournament context adjustments
        if context:
            time_pressure = context.get("time_pressure", "medium")
            if time_pressure == "high":
                base_depth = "shallow"
            elif time_pressure == "low" and base_depth != "deep":
                base_depth = "medium"

        return base_depth

    def _configure_validation(
        self,
        strategy: CategoryStrategy,
        classification: QuestionClassification,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Configure validation requirements."""
        config = {
            "requirements": strategy.validation_requirements.copy(),
            "minimum_sources": 3,
            "cross_validation": True,
            "expert_review": False,
        }

        # Adjust based on resource allocation
        if classification.resource_allocation_score > 0.7:
            config["minimum_sources"] = 5
            config["expert_review"] = True
        elif classification.resource_allocation_score < 0.3:
            config["minimum_sources"] = 2
            config["cross_validation"] = False

        # Tournament context adjustments
        if context:
            risk_tolerance = context.get("risk_tolerance", "medium")
            if risk_tolerance == "low":
                config["minimum_sources"] += 1
                config["expert_review"] = True
            elif risk_tolerance == "high":
                config["minimum_sources"] = max(1, config["minimum_sources"] - 1)

        return config

    def _set_confidence_thresholds(
        self, classification: QuestionClassification, context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Set confidence thresholds based on classification."""
        strategy = classification.recommended_strategy

        base_thresholds = {
            "minimum_submission": 0.6,
            "high_confidence": 0.8,
            "abstention": 0.4,
        }

        # Apply strategy adjustment
        adjustment = strategy.confidence_adjustment
        for key in base_thresholds:
            base_thresholds[key] = max(0.1, min(0.9, base_thresholds[key] + adjustment))

        # Tournament context adjustments
        if context:
            tournament_phase = context.get("tournament_phase", "middle")
            if tournament_phase == "late":
                # More conservative late in tournament
                for key in base_thresholds:
                    base_thresholds[key] = min(0.9, base_thresholds[key] + 0.05)
            elif tournament_phase == "early":
                # More aggressive early in tournament
                for key in base_thresholds:
                    base_thresholds[key] = max(0.1, base_thresholds[key] - 0.05)

        return base_thresholds
