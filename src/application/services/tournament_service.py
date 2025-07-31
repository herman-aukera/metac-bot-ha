"""Tournament Strategy Engine for optimizing tournament performance.

This module implements a comprehensive tournament strategy system that includes:
- Intelligent question categorization with ML-based classification
- Multi-criteria question prioritization
- Dynamic submission timing optimization
- Competitor analysis and market inefficiency detection
- Strategy adaptation based on tournament meta-game patterns
- Risk-adjusted strategy selection
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import numpy as np
from collections import defaultdict

from ...domain.entities.tournament import Tournament, ScoringMethod
from ...domain.entities.question import Question, QuestionType, QuestionCategory
from ...domain.value_objects.strategy_result import StrategyResult, StrategyType, StrategyOutcome
from ...domain.value_objects.confidence import Confidence


logger = logging.getLogger(__name__)


class PriorityLevel(Enum):
    """Priority levels for question processing."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MarketCondition(Enum):
    """Market conditions for timing optimization."""
    OVERSATURATED = "oversaturated"
    COMPETITIVE = "competitive"
    OPPORTUNITY = "opportunity"
    UNDEREXPLORED = "underexplored"


class RiskProfile(Enum):
    """Risk profiles for strategy selection."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


@dataclass
class QuestionPriority:
    """Represents prioritization analysis for a question."""
    question_id: int
    priority_level: PriorityLevel
    confidence_score: float
    scoring_potential: float
    deadline_urgency: float
    strategic_value: float
    overall_score: float
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class MarketAnalysis:
    """Market analysis for timing optimization."""
    question_id: int
    market_condition: MarketCondition
    competitor_density: float
    prediction_variance: float
    consensus_strength: float
    optimal_timing_window: Tuple[datetime, datetime]
    confidence: float
    reasoning: str


@dataclass
class CompetitorProfile:
    """Profile of competitor behavior and patterns."""
    participant_id: str
    accuracy_score: float
    submission_patterns: Dict[str, Any]
    category_specializations: List[QuestionCategory]
    risk_profile: RiskProfile
    recent_performance: List[float]
    prediction_style: str
    last_updated: datetime


@dataclass
class StrategyRecommendation:
    """Comprehensive strategy recommendation."""
    strategy_type: StrategyType
    confidence: Confidence
    expected_score_impact: float
    risk_level: float
    resource_requirements: Dict[str, float]
    timing_constraints: Optional[Tuple[datetime, datetime]]
    question_allocation: Dict[int, float]
    reasoning: str
    alternatives: List[Tuple[StrategyType, float]]


class TournamentService:
    """Comprehensive tournament strategy engine."""

    def __init__(self,
                 ml_classifier: Optional[Any] = None,
                 risk_tolerance: float = 0.5,
                 max_concurrent_questions: int = 10):
        """Initialize tournament service.

        Args:
            ml_classifier: Optional ML classifier for question categorization
            risk_tolerance: Risk tolerance level (0.0 = very conservative, 1.0 = very aggressive)
            max_concurrent_questions: Maximum questions to process concurrently
        """
        self.ml_classifier = ml_classifier
        self.risk_tolerance = risk_tolerance
        self.max_concurrent_questions = max_concurrent_questions

        # Strategy performance tracking
        self.strategy_history: List[StrategyResult] = []
        self.competitor_profiles: Dict[str, CompetitorProfile] = {}
        self.market_patterns: Dict[int, List[MarketAnalysis]] = defaultdict(list)

        # Configuration
        self.priority_weights = {
            'confidence': 0.3,
            'scoring_potential': 0.25,
            'deadline_urgency': 0.2,
            'strategic_value': 0.25
        }

        self.timing_factors = {
            'early_submission_bonus': 0.05,
            'late_penalty_threshold': 0.8,  # 80% of time elapsed
            'optimal_window_start': 0.3,    # 30% of time elapsed
            'optimal_window_end': 0.7       # 70% of time elapsed
        }

    async def analyze_tournament_strategy(self, tournament: Tournament) -> StrategyRecommendation:
        """Analyze tournament and recommend comprehensive strategy.

        Args:
            tournament: Tournament to analyze

        Returns:
            Comprehensive strategy recommendation
        """
        logger.info(f"Analyzing strategy for tournament {tournament.id}: {tournament.name}")

        # Parallel analysis of different aspects
        tasks = [
            self._categorize_questions(tournament.questions),
            self._prioritize_questions(tournament),
            self._analyze_market_conditions(tournament),
            self._analyze_competitors(tournament),
            self._assess_meta_game_patterns(tournament)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        categorized_questions = results[0] if not isinstance(results[0], Exception) else {}
        prioritized_questions = results[1] if not isinstance(results[1], Exception) else []
        market_analysis = results[2] if not isinstance(results[2], Exception) else {}
        competitor_analysis = results[3] if not isinstance(results[3], Exception) else {}
        meta_patterns = results[4] if not isinstance(results[4], Exception) else {}

        # Generate strategy recommendation
        strategy_rec = await self._generate_strategy_recommendation(
            tournament=tournament,
            categorized_questions=categorized_questions,
            prioritized_questions=prioritized_questions,
            market_analysis=market_analysis,
            competitor_analysis=competitor_analysis,
            meta_patterns=meta_patterns
        )

        logger.info(f"Generated {strategy_rec.strategy_type.value} strategy with "
                   f"{strategy_rec.confidence.level:.2f} confidence")

        return strategy_rec

    async def _categorize_questions(self, questions: List[Question]) -> Dict[int, Dict[str, Any]]:
        """Categorize questions using ML-based classification.

        Args:
            questions: List of questions to categorize

        Returns:
            Dictionary mapping question IDs to categorization results
        """
        categorized = {}

        for question in questions:
            # Basic rule-based categorization
            category_confidence = self._calculate_category_confidence(question)

            # ML-based enhancement if classifier available
            ml_features = {}
            if self.ml_classifier:
                ml_features = await self._extract_ml_features(question)

            categorized[question.id] = {
                'primary_category': question.category,
                'category_confidence': category_confidence,
                'complexity_score': question.get_complexity_score(),
                'specialization_required': question.requires_specialized_knowledge(),
                'ml_features': ml_features,
                'predicted_difficulty': self._predict_question_difficulty(question),
                'estimated_research_time': self._estimate_research_time(question)
            }

        return categorized

    async def _prioritize_questions(self, tournament: Tournament) -> List[QuestionPriority]:
        """Prioritize questions using multi-criteria analysis.

        Args:
            tournament: Tournament containing questions to prioritize

        Returns:
            List of prioritized questions
        """
        priorities = []
        active_questions = tournament.get_active_questions()

        for question in active_questions:
            # Calculate individual criteria scores
            confidence_score = self._calculate_confidence_score(question, tournament)
            scoring_potential = self._calculate_scoring_potential(question, tournament)
            deadline_urgency = self._calculate_deadline_urgency(question)
            strategic_value = self._calculate_strategic_value(question, tournament)

            # Calculate weighted overall score
            overall_score = (
                confidence_score * self.priority_weights['confidence'] +
                scoring_potential * self.priority_weights['scoring_potential'] +
                deadline_urgency * self.priority_weights['deadline_urgency'] +
                strategic_value * self.priority_weights['strategic_value']
            )

            # Determine priority level
            if overall_score >= 0.8:
                priority_level = PriorityLevel.CRITICAL
            elif overall_score >= 0.6:
                priority_level = PriorityLevel.HIGH
            elif overall_score >= 0.4:
                priority_level = PriorityLevel.MEDIUM
            else:
                priority_level = PriorityLevel.LOW

            reasoning = self._generate_priority_reasoning(
                question, confidence_score, scoring_potential,
                deadline_urgency, strategic_value
            )

            priority = QuestionPriority(
                question_id=question.id,
                priority_level=priority_level,
                confidence_score=confidence_score,
                scoring_potential=scoring_potential,
                deadline_urgency=deadline_urgency,
                strategic_value=strategic_value,
                overall_score=overall_score,
                reasoning=reasoning,
                metadata={
                    'question_type': question.question_type.value,
                    'category': question.category.value,
                    'scoring_weight': question.scoring_weight,
                    'time_remaining': question.time_until_deadline()
                }
            )

            priorities.append(priority)

        # Sort by overall score descending
        priorities.sort(key=lambda p: p.overall_score, reverse=True)

        return priorities

    async def _analyze_market_conditions(self, tournament: Tournament) -> Dict[int, MarketAnalysis]:
        """Analyze market conditions for timing optimization.

        Args:
            tournament: Tournament to analyze

        Returns:
            Dictionary mapping question IDs to market analysis
        """
        market_analyses = {}

        for question in tournament.get_active_questions():
            # Simulate market analysis (in real implementation, this would use actual data)
            competitor_density = self._estimate_competitor_density(question, tournament)
            prediction_variance = self._estimate_prediction_variance(question)
            consensus_strength = self._estimate_consensus_strength(question)

            # Determine market condition
            if competitor_density > 0.8 and consensus_strength > 0.7:
                condition = MarketCondition.OVERSATURATED
            elif competitor_density > 0.6:
                condition = MarketCondition.COMPETITIVE
            elif competitor_density < 0.3:
                condition = MarketCondition.UNDEREXPLORED
            else:
                condition = MarketCondition.OPPORTUNITY

            # Calculate optimal timing window
            optimal_window = self._calculate_optimal_timing_window(question, condition)

            analysis = MarketAnalysis(
                question_id=question.id,
                market_condition=condition,
                competitor_density=competitor_density,
                prediction_variance=prediction_variance,
                consensus_strength=consensus_strength,
                optimal_timing_window=optimal_window,
                confidence=0.7,  # Base confidence for market analysis
                reasoning=f"Market condition: {condition.value}, "
                         f"competitor density: {competitor_density:.2f}, "
                         f"consensus strength: {consensus_strength:.2f}"
            )

            market_analyses[question.id] = analysis

        return market_analyses

    async def _analyze_competitors(self, tournament: Tournament) -> Dict[str, CompetitorProfile]:
        """Analyze competitor behavior and patterns.

        Args:
            tournament: Tournament to analyze competitors for

        Returns:
            Dictionary mapping participant IDs to competitor profiles
        """
        competitor_profiles = {}

        # Analyze top competitors
        top_participants = tournament.get_top_participants(n=20)

        for participant_id, score in top_participants:
            # In real implementation, this would analyze historical data
            profile = CompetitorProfile(
                participant_id=participant_id,
                accuracy_score=score,
                submission_patterns=self._analyze_submission_patterns(participant_id),
                category_specializations=self._identify_specializations(participant_id),
                risk_profile=self._assess_risk_profile(participant_id),
                recent_performance=self._get_recent_performance(participant_id),
                prediction_style=self._classify_prediction_style(participant_id),
                last_updated=datetime.utcnow()
            )

            competitor_profiles[participant_id] = profile

        return competitor_profiles

    async def _assess_meta_game_patterns(self, tournament: Tournament) -> Dict[str, Any]:
        """Assess tournament meta-game patterns and trends.

        Args:
            tournament: Tournament to analyze

        Returns:
            Dictionary containing meta-game analysis
        """
        patterns = {
            'dominant_strategies': self._identify_dominant_strategies(tournament),
            'category_trends': self._analyze_category_trends(tournament),
            'timing_patterns': self._analyze_timing_patterns(tournament),
            'scoring_inefficiencies': self._identify_scoring_inefficiencies(tournament),
            'adaptation_opportunities': self._identify_adaptation_opportunities(tournament)
        }

        return patterns

    async def _generate_strategy_recommendation(self,
                                              tournament: Tournament,
                                              categorized_questions: Dict[int, Dict[str, Any]],
                                              prioritized_questions: List[QuestionPriority],
                                              market_analysis: Dict[int, MarketAnalysis],
                                              competitor_analysis: Dict[str, CompetitorProfile],
                                              meta_patterns: Dict[str, Any]) -> StrategyRecommendation:
        """Generate comprehensive strategy recommendation.

        Args:
            tournament: Tournament being analyzed
            categorized_questions: Question categorization results
            prioritized_questions: Question prioritization results
            market_analysis: Market condition analysis
            competitor_analysis: Competitor behavior analysis
            meta_patterns: Meta-game pattern analysis

        Returns:
            Comprehensive strategy recommendation
        """
        # Determine optimal strategy type based on analysis
        strategy_type = self._select_optimal_strategy(
            tournament, prioritized_questions, market_analysis,
            competitor_analysis, meta_patterns
        )

        # Calculate confidence in strategy
        confidence_level = self._calculate_strategy_confidence(
            strategy_type, tournament, prioritized_questions, market_analysis
        )

        confidence = Confidence(
            level=confidence_level,
            basis=f"Based on analysis of {len(prioritized_questions)} questions, "
                  f"market conditions, and {len(competitor_analysis)} competitors"
        )

        # Estimate expected score impact
        expected_impact = self._estimate_score_impact(
            strategy_type, tournament, prioritized_questions
        )

        # Assess risk level
        risk_level = self._assess_strategy_risk(strategy_type, tournament)

        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(
            strategy_type, prioritized_questions
        )

        # Determine timing constraints
        timing_constraints = self._determine_timing_constraints(
            strategy_type, tournament, market_analysis
        )

        # Allocate questions based on strategy
        question_allocation = self._allocate_questions(
            strategy_type, prioritized_questions, resource_requirements
        )

        # Generate reasoning
        reasoning = self._generate_strategy_reasoning(
            strategy_type, tournament, prioritized_questions,
            market_analysis, competitor_analysis
        )

        # Identify alternative strategies
        alternatives = self._identify_alternative_strategies(
            tournament, prioritized_questions, market_analysis
        )

        return StrategyRecommendation(
            strategy_type=strategy_type,
            confidence=confidence,
            expected_score_impact=expected_impact,
            risk_level=risk_level,
            resource_requirements=resource_requirements,
            timing_constraints=timing_constraints,
            question_allocation=question_allocation,
            reasoning=reasoning,
            alternatives=alternatives
        )

    def optimize_submission_timing(self,
                                 question: Question,
                                 tournament: Tournament,
                                 market_analysis: Optional[MarketAnalysis] = None) -> Tuple[datetime, str]:
        """Optimize submission timing for a specific question.

        Args:
            question: Question to optimize timing for
            tournament: Tournament context
            market_analysis: Optional market analysis for the question

        Returns:
            Tuple of (optimal_submission_time, reasoning)
        """
        if market_analysis is None:
            # Generate basic market analysis
            competitor_density = self._estimate_competitor_density(question, tournament)
            condition = MarketCondition.COMPETITIVE if competitor_density > 0.5 else MarketCondition.OPPORTUNITY
            optimal_window = self._calculate_optimal_timing_window(question, condition)
        else:
            optimal_window = market_analysis.optimal_timing_window
            condition = market_analysis.market_condition

        # Calculate optimal time within window
        window_start, window_end = optimal_window

        # Adjust based on tournament scoring rules
        if tournament.scoring_rules.bonus_for_early:
            # Bias toward earlier submission
            optimal_time = window_start + (window_end - window_start) * 0.3
            timing_reason = "Early submission bonus favors earlier timing"
        elif tournament.scoring_rules.penalty_for_late:
            # Bias toward earlier submission to avoid penalty
            optimal_time = window_start + (window_end - window_start) * 0.4
            timing_reason = "Late penalty avoidance favors earlier timing"
        else:
            # Use middle of optimal window
            optimal_time = window_start + (window_end - window_start) * 0.5
            timing_reason = "Balanced timing within optimal window"

        # Ensure we don't exceed deadline
        if optimal_time >= question.deadline:
            optimal_time = question.deadline - timedelta(hours=1)
            timing_reason += " (adjusted to avoid deadline)"

        reasoning = (f"Market condition: {condition.value}. {timing_reason}. "
                    f"Optimal window: {window_start.strftime('%H:%M')} - "
                    f"{window_end.strftime('%H:%M')}")

        return optimal_time, reasoning

    def adapt_strategy_based_on_performance(self,
                                          recent_results: List[StrategyResult]) -> StrategyType:
        """Adapt strategy based on recent performance feedback.

        Args:
            recent_results: Recent strategy execution results

        Returns:
            Adapted strategy type
        """
        if not recent_results:
            return StrategyType.BALANCED

        # Analyze performance by strategy type
        strategy_performance = defaultdict(list)
        for result in recent_results:
            if result.actual_score is not None:
                effectiveness = result.get_score_difference() or 0
                strategy_performance[result.strategy_type].append(effectiveness)

        # Calculate average effectiveness for each strategy
        strategy_averages = {}
        for strategy_type, scores in strategy_performance.items():
            strategy_averages[strategy_type] = np.mean(scores) if scores else 0

        # Select best performing strategy, with some exploration
        if strategy_averages:
            best_strategy = max(strategy_averages.keys(),
                              key=lambda s: strategy_averages[s])

            # Add some randomness for exploration
            if np.random.random() < 0.1:  # 10% exploration
                available_strategies = list(StrategyType)
                return np.random.choice(available_strategies)

            return best_strategy

        return StrategyType.BALANCED

    def select_risk_adjusted_strategy(self,
                                    tournament: Tournament,
                                    current_position: Optional[int] = None) -> StrategyType:
        """Select strategy based on risk profile and tournament position.

        Args:
            tournament: Tournament context
            current_position: Current ranking position (1 = first place)

        Returns:
            Risk-adjusted strategy type
        """
        time_remaining_ratio = tournament.time_remaining() / (
            (tournament.end_date - tournament.start_date).total_seconds() / 3600
        )

        # Determine risk profile based on position and time
        if current_position is None:
            risk_profile = RiskProfile.MODERATE
        elif current_position <= 3:
            # Leading positions - be more conservative
            if time_remaining_ratio > 0.5:
                risk_profile = RiskProfile.MODERATE
            else:
                risk_profile = RiskProfile.CONSERVATIVE
        elif current_position <= 10:
            # Good position - balanced approach
            risk_profile = RiskProfile.MODERATE
        else:
            # Need to catch up - be more aggressive
            if time_remaining_ratio > 0.3:
                risk_profile = RiskProfile.AGGRESSIVE
            else:
                risk_profile = RiskProfile.MODERATE

        # Map risk profile to strategy
        risk_strategy_mapping = {
            RiskProfile.CONSERVATIVE: StrategyType.CONSERVATIVE,
            RiskProfile.MODERATE: StrategyType.BALANCED,
            RiskProfile.AGGRESSIVE: StrategyType.AGGRESSIVE,
            RiskProfile.ADAPTIVE: StrategyType.MOMENTUM
        }

        base_strategy = risk_strategy_mapping[risk_profile]

        # Adjust based on tournament characteristics
        if tournament.scoring_rules.method == ScoringMethod.LOG_SCORE:
            # Log scoring rewards confidence, favor aggressive strategies
            if base_strategy == StrategyType.CONSERVATIVE:
                return StrategyType.BALANCED
            elif base_strategy == StrategyType.BALANCED:
                return StrategyType.AGGRESSIVE

        return base_strategy

    # Helper methods for calculations and analysis

    def _calculate_category_confidence(self, question: Question) -> float:
        """Calculate confidence in question categorization."""
        # Simple heuristic based on question characteristics
        base_confidence = 0.8

        # Adjust based on question complexity
        complexity_factor = min(question.get_complexity_score() / 2.0, 1.0)
        confidence = base_confidence * (1.0 - complexity_factor * 0.2)

        return max(0.1, min(1.0, confidence))

    async def _extract_ml_features(self, question: Question) -> Dict[str, Any]:
        """Extract ML features for question classification."""
        # Placeholder for ML feature extraction
        return {
            'text_length': len(question.text),
            'background_length': len(question.background),
            'has_numbers': any(char.isdigit() for char in question.text),
            'question_words': len(question.text.split()),
            'complexity_indicators': question.get_complexity_score()
        }

    def _predict_question_difficulty(self, question: Question) -> float:
        """Predict question difficulty score."""
        difficulty = question.get_complexity_score() / 3.0  # Normalize to 0-1 range

        # Adjust based on category
        if question.requires_specialized_knowledge():
            difficulty *= 1.2

        return min(1.0, difficulty)

    def _estimate_research_time(self, question: Question) -> float:
        """Estimate research time required in hours."""
        base_time = 2.0  # Base 2 hours

        # Adjust based on complexity
        complexity_multiplier = question.get_complexity_score()

        # Adjust based on specialization
        if question.requires_specialized_knowledge():
            complexity_multiplier *= 1.5

        return base_time * complexity_multiplier

    def _calculate_confidence_score(self, question: Question, tournament: Tournament) -> float:
        """Calculate confidence score for question prioritization."""
        # Base confidence based on question characteristics
        base_confidence = 0.5

        # Adjust based on category familiarity (simulated)
        if question.category in [QuestionCategory.AI_DEVELOPMENT, QuestionCategory.TECHNOLOGY]:
            base_confidence += 0.2

        # Adjust based on question type
        type_confidence = {
            QuestionType.BINARY: 0.8,
            QuestionType.MULTIPLE_CHOICE: 0.6,
            QuestionType.NUMERIC: 0.4,
            QuestionType.DATE: 0.5,
            QuestionType.CONDITIONAL: 0.3
        }

        confidence = base_confidence * type_confidence.get(question.question_type, 0.5)

        return max(0.1, min(1.0, confidence))

    def _calculate_scoring_potential(self, question: Question, tournament: Tournament) -> float:
        """Calculate scoring potential for question prioritization."""
        # Base on scoring weight
        potential = min(question.scoring_weight / 5.0, 1.0)  # Normalize assuming max weight of 5

        # Adjust based on tournament scoring method
        if tournament.scoring_rules.method == ScoringMethod.LOG_SCORE:
            # Log scoring rewards confident predictions
            potential *= 1.2
        elif tournament.scoring_rules.method == ScoringMethod.BRIER_SCORE:
            # Brier scoring is more forgiving
            potential *= 1.1

        return max(0.1, min(1.0, potential))

    def _calculate_deadline_urgency(self, question: Question) -> float:
        """Calculate deadline urgency for question prioritization."""
        hours_remaining = question.time_until_deadline()

        if hours_remaining <= 0:
            return 0.0
        elif hours_remaining <= 6:
            return 1.0
        elif hours_remaining <= 24:
            return 0.8
        elif hours_remaining <= 72:
            return 0.6
        elif hours_remaining <= 168:  # 1 week
            return 0.4
        else:
            return 0.2

    def _calculate_strategic_value(self, question: Question, tournament: Tournament) -> float:
        """Calculate strategic value for question prioritization."""
        value = 0.5  # Base value

        # High-value questions are more strategic
        if question.is_high_value():
            value += 0.3

        # Questions in underrepresented categories are more valuable
        category_counts = defaultdict(int)
        for q in tournament.questions:
            category_counts[q.category] += 1

        if category_counts[question.category] < len(tournament.questions) / 10:
            value += 0.2

        return max(0.1, min(1.0, value))

    def _generate_priority_reasoning(self, question: Question,
                                   confidence: float, scoring: float,
                                   urgency: float, strategic: float) -> str:
        """Generate reasoning for question prioritization."""
        reasons = []

        if confidence > 0.7:
            reasons.append("high confidence")
        elif confidence < 0.3:
            reasons.append("low confidence")

        if scoring > 0.7:
            reasons.append("high scoring potential")

        if urgency > 0.8:
            reasons.append("urgent deadline")
        elif urgency < 0.3:
            reasons.append("distant deadline")

        if strategic > 0.7:
            reasons.append("high strategic value")

        if not reasons:
            reasons.append("balanced characteristics")

        return f"Priority based on: {', '.join(reasons)}"

    def _estimate_competitor_density(self, question: Question, tournament: Tournament) -> float:
        """Estimate competitor density for market analysis."""
        # Simulate based on question characteristics
        base_density = 0.5

        # Popular categories attract more competitors
        if question.category in [QuestionCategory.AI_DEVELOPMENT, QuestionCategory.POLITICS]:
            base_density += 0.2

        # High-value questions attract more competitors
        if question.is_high_value():
            base_density += 0.1

        # Binary questions are easier, attract more competitors
        if question.question_type == QuestionType.BINARY:
            base_density += 0.1

        return max(0.1, min(1.0, base_density))

    def _estimate_prediction_variance(self, question: Question) -> float:
        """Estimate prediction variance for market analysis."""
        # Higher variance for more complex questions
        variance = question.get_complexity_score() / 3.0

        # Adjust based on question type
        if question.question_type == QuestionType.NUMERIC:
            variance *= 1.2
        elif question.question_type == QuestionType.BINARY:
            variance *= 0.8

        return max(0.1, min(1.0, variance))

    def _estimate_consensus_strength(self, question: Question) -> float:
        """Estimate consensus strength for market analysis."""
        # Inverse of complexity - simpler questions have stronger consensus
        consensus = 1.0 - (question.get_complexity_score() / 3.0)

        # Adjust based on category
        if question.requires_specialized_knowledge():
            consensus *= 0.8

        return max(0.1, min(1.0, consensus))

    def _calculate_optimal_timing_window(self, question: Question,
                                       condition: MarketCondition) -> Tuple[datetime, datetime]:
        """Calculate optimal timing window for submission."""
        total_time = question.deadline - datetime.utcnow()

        if condition == MarketCondition.UNDEREXPLORED:
            # Submit early to establish position
            start_ratio = 0.1
            end_ratio = 0.4
        elif condition == MarketCondition.OVERSATURATED:
            # Wait for market to settle
            start_ratio = 0.6
            end_ratio = 0.9
        else:
            # Standard timing
            start_ratio = self.timing_factors['optimal_window_start']
            end_ratio = self.timing_factors['optimal_window_end']

        start_time = datetime.utcnow() + total_time * start_ratio
        end_time = datetime.utcnow() + total_time * end_ratio

        return start_time, end_time

    def _analyze_submission_patterns(self, participant_id: str) -> Dict[str, Any]:
        """Analyze submission patterns for competitor profiling."""
        # Placeholder for submission pattern analysis
        return {
            'avg_submission_time_ratio': 0.6,  # 60% through question lifetime
            'early_submissions': 0.3,
            'late_submissions': 0.1,
            'revision_frequency': 0.2
        }

    def _identify_specializations(self, participant_id: str) -> List[QuestionCategory]:
        """Identify category specializations for competitor profiling."""
        # Placeholder - would analyze historical performance by category
        return [QuestionCategory.AI_DEVELOPMENT, QuestionCategory.TECHNOLOGY]

    def _assess_risk_profile(self, participant_id: str) -> RiskProfile:
        """Assess risk profile for competitor profiling."""
        # Placeholder - would analyze prediction patterns
        return RiskProfile.MODERATE

    def _get_recent_performance(self, participant_id: str) -> List[float]:
        """Get recent performance scores for competitor profiling."""
        # Placeholder - would get actual recent scores
        return [0.7, 0.8, 0.6, 0.9, 0.7]

    def _classify_prediction_style(self, participant_id: str) -> str:
        """Classify prediction style for competitor profiling."""
        # Placeholder - would analyze prediction patterns
        return "balanced"

    def _identify_dominant_strategies(self, tournament: Tournament) -> List[str]:
        """Identify dominant strategies in tournament meta-game."""
        # Placeholder for meta-game analysis
        return ["early_submission", "high_confidence_binary"]

    def _analyze_category_trends(self, tournament: Tournament) -> Dict[str, float]:
        """Analyze category performance trends."""
        # Placeholder for trend analysis
        return {
            "ai_development": 0.8,
            "technology": 0.7,
            "politics": 0.6
        }

    def _analyze_timing_patterns(self, tournament: Tournament) -> Dict[str, Any]:
        """Analyze timing patterns in tournament."""
        # Placeholder for timing analysis
        return {
            "optimal_submission_ratio": 0.6,
            "early_bonus_effectiveness": 0.05,
            "late_penalty_threshold": 0.8
        }

    def _identify_scoring_inefficiencies(self, tournament: Tournament) -> List[str]:
        """Identify scoring inefficiencies to exploit."""
        # Placeholder for inefficiency analysis
        return ["undervalued_numeric_questions", "timing_arbitrage"]

    def _identify_adaptation_opportunities(self, tournament: Tournament) -> List[str]:
        """Identify opportunities for strategy adaptation."""
        # Placeholder for adaptation analysis
        return ["category_specialization", "timing_optimization"]

    def _select_optimal_strategy(self, tournament: Tournament,
                               prioritized_questions: List[QuestionPriority],
                               market_analysis: Dict[int, MarketAnalysis],
                               competitor_analysis: Dict[str, CompetitorProfile],
                               meta_patterns: Dict[str, Any]) -> StrategyType:
        """Select optimal strategy type based on comprehensive analysis."""
        # Score each strategy type
        strategy_scores = {
            StrategyType.AGGRESSIVE: 0.0,
            StrategyType.CONSERVATIVE: 0.0,
            StrategyType.BALANCED: 0.5,  # Base score for balanced
            StrategyType.CONTRARIAN: 0.0,
            StrategyType.MOMENTUM: 0.0
        }

        # Adjust based on high-priority questions
        high_priority_count = sum(1 for p in prioritized_questions
                                if p.priority_level in [PriorityLevel.CRITICAL, PriorityLevel.HIGH])

        if high_priority_count > len(prioritized_questions) * 0.6:
            strategy_scores[StrategyType.AGGRESSIVE] += 0.3

        # Adjust based on market conditions
        opportunity_markets = sum(1 for analysis in market_analysis.values()
                                if analysis.market_condition == MarketCondition.OPPORTUNITY)

        if opportunity_markets > len(market_analysis) * 0.4:
            strategy_scores[StrategyType.AGGRESSIVE] += 0.2
            strategy_scores[StrategyType.CONTRARIAN] += 0.1

        # Adjust based on competitor density
        avg_competitor_density = np.mean([a.competitor_density for a in market_analysis.values()])
        if avg_competitor_density > 0.7:
            strategy_scores[StrategyType.CONSERVATIVE] += 0.2
            strategy_scores[StrategyType.CONTRARIAN] += 0.3

        # Adjust based on risk tolerance
        risk_adjustment = (self.risk_tolerance - 0.5) * 0.4
        strategy_scores[StrategyType.AGGRESSIVE] += risk_adjustment
        strategy_scores[StrategyType.CONSERVATIVE] -= risk_adjustment

        # Select strategy with highest score
        return max(strategy_scores.keys(), key=lambda s: strategy_scores[s])

    def _calculate_strategy_confidence(self, strategy_type: StrategyType,
                                     tournament: Tournament,
                                     prioritized_questions: List[QuestionPriority],
                                     market_analysis: Dict[int, MarketAnalysis]) -> float:
        """Calculate confidence in selected strategy."""
        base_confidence = 0.6

        # Adjust based on data quality
        if len(prioritized_questions) > 10:
            base_confidence += 0.1

        if len(market_analysis) > 5:
            base_confidence += 0.1

        # Adjust based on strategy type
        strategy_confidence_factors = {
            StrategyType.BALANCED: 0.8,
            StrategyType.CONSERVATIVE: 0.7,
            StrategyType.AGGRESSIVE: 0.6,
            StrategyType.CONTRARIAN: 0.5,
            StrategyType.MOMENTUM: 0.6
        }

        confidence = base_confidence * strategy_confidence_factors.get(strategy_type, 0.6)

        return max(0.1, min(1.0, confidence))

    def _estimate_score_impact(self, strategy_type: StrategyType,
                             tournament: Tournament,
                             prioritized_questions: List[QuestionPriority]) -> float:
        """Estimate expected score impact of strategy."""
        # Base impact estimates by strategy type
        base_impacts = {
            StrategyType.AGGRESSIVE: 0.15,
            StrategyType.CONSERVATIVE: 0.08,
            StrategyType.BALANCED: 0.10,
            StrategyType.CONTRARIAN: 0.12,
            StrategyType.MOMENTUM: 0.11
        }

        base_impact = base_impacts.get(strategy_type, 0.10)

        # Adjust based on high-priority questions
        high_priority_potential = sum(p.scoring_potential for p in prioritized_questions
                                    if p.priority_level in [PriorityLevel.CRITICAL, PriorityLevel.HIGH])

        if high_priority_potential > 0:
            base_impact *= (1.0 + high_priority_potential / len(prioritized_questions))

        return base_impact

    def _assess_strategy_risk(self, strategy_type: StrategyType, tournament: Tournament) -> float:
        """Assess risk level of strategy."""
        risk_levels = {
            StrategyType.CONSERVATIVE: 0.2,
            StrategyType.BALANCED: 0.4,
            StrategyType.AGGRESSIVE: 0.8,
            StrategyType.CONTRARIAN: 0.7,
            StrategyType.MOMENTUM: 0.6
        }

        return risk_levels.get(strategy_type, 0.5)

    def _calculate_resource_requirements(self, strategy_type: StrategyType,
                                       prioritized_questions: List[QuestionPriority]) -> Dict[str, float]:
        """Calculate resource requirements for strategy."""
        base_requirements = {
            'research_time': 2.0,  # hours per question
            'analysis_time': 1.0,  # hours per question
            'monitoring_time': 0.5  # hours per question
        }

        # Adjust based on strategy type
        if strategy_type == StrategyType.AGGRESSIVE:
            base_requirements['research_time'] *= 1.5
            base_requirements['analysis_time'] *= 1.3
        elif strategy_type == StrategyType.CONSERVATIVE:
            base_requirements['research_time'] *= 1.2
            base_requirements['analysis_time'] *= 1.1

        # Scale by number of questions
        question_count = len(prioritized_questions)
        for key in base_requirements:
            base_requirements[key] *= question_count

        return base_requirements

    def _determine_timing_constraints(self, strategy_type: StrategyType,
                                    tournament: Tournament,
                                    market_analysis: Dict[int, MarketAnalysis]) -> Optional[Tuple[datetime, datetime]]:
        """Determine timing constraints for strategy."""
        if not market_analysis:
            return None

        # Find earliest and latest optimal windows
        all_windows = [analysis.optimal_timing_window for analysis in market_analysis.values()]

        if not all_windows:
            return None

        earliest_start = min(window[0] for window in all_windows)
        latest_end = max(window[1] for window in all_windows)

        # Adjust based on strategy type
        if strategy_type == StrategyType.AGGRESSIVE:
            # Tighter timing constraints
            constraint_start = earliest_start
            constraint_end = earliest_start + (latest_end - earliest_start) * 0.7
        elif strategy_type == StrategyType.CONSERVATIVE:
            # More flexible timing
            constraint_start = earliest_start + (latest_end - earliest_start) * 0.2
            constraint_end = latest_end
        else:
            # Balanced timing
            constraint_start = earliest_start + (latest_end - earliest_start) * 0.1
            constraint_end = latest_end - (latest_end - earliest_start) * 0.1

        return constraint_start, constraint_end

    def _allocate_questions(self, strategy_type: StrategyType,
                          prioritized_questions: List[QuestionPriority],
                          resource_requirements: Dict[str, float]) -> Dict[int, float]:
        """Allocate questions based on strategy and resources."""
        allocation = {}

        # Sort questions by priority
        sorted_questions = sorted(prioritized_questions,
                                key=lambda p: p.overall_score, reverse=True)

        # Allocate based on strategy type
        if strategy_type == StrategyType.AGGRESSIVE:
            # Focus on top questions
            top_count = min(len(sorted_questions), self.max_concurrent_questions // 2)
            for i, question in enumerate(sorted_questions[:top_count]):
                allocation[question.question_id] = 1.0

            # Partial allocation for next tier
            for i, question in enumerate(sorted_questions[top_count:top_count*2]):
                allocation[question.question_id] = 0.5

        elif strategy_type == StrategyType.CONSERVATIVE:
            # Spread resources more evenly
            max_questions = min(len(sorted_questions), self.max_concurrent_questions)
            for i, question in enumerate(sorted_questions[:max_questions]):
                allocation[question.question_id] = 0.8

        else:  # BALANCED, CONTRARIAN, MOMENTUM
            # Balanced allocation
            max_questions = min(len(sorted_questions), self.max_concurrent_questions)
            for i, question in enumerate(sorted_questions[:max_questions]):
                # Higher allocation for higher priority
                priority_factor = (len(sorted_questions) - i) / len(sorted_questions)
                allocation[question.question_id] = 0.6 + 0.4 * priority_factor

        return allocation

    def _generate_strategy_reasoning(self, strategy_type: StrategyType,
                                   tournament: Tournament,
                                   prioritized_questions: List[QuestionPriority],
                                   market_analysis: Dict[int, MarketAnalysis],
                                   competitor_analysis: Dict[str, CompetitorProfile]) -> str:
        """Generate reasoning for strategy selection."""
        reasons = []

        # Tournament context
        active_questions = len(prioritized_questions)
        time_remaining = tournament.time_remaining()
        reasons.append(f"Tournament has {active_questions} active questions with {time_remaining:.1f} hours remaining")

        # Priority analysis
        high_priority = sum(1 for p in prioritized_questions
                          if p.priority_level in [PriorityLevel.CRITICAL, PriorityLevel.HIGH])
        if high_priority > 0:
            reasons.append(f"{high_priority} high-priority questions identified")

        # Market conditions
        if market_analysis:
            opportunity_count = sum(1 for a in market_analysis.values()
                                  if a.market_condition == MarketCondition.OPPORTUNITY)
            if opportunity_count > 0:
                reasons.append(f"{opportunity_count} opportunity markets detected")

        # Strategy justification
        strategy_justifications = {
            StrategyType.AGGRESSIVE: "High-reward opportunities justify aggressive approach",
            StrategyType.CONSERVATIVE: "Market saturation and competition favor conservative approach",
            StrategyType.BALANCED: "Mixed conditions support balanced strategy",
            StrategyType.CONTRARIAN: "Market inefficiencies present contrarian opportunities",
            StrategyType.MOMENTUM: "Tournament dynamics favor momentum-based approach"
        }

        reasons.append(strategy_justifications.get(strategy_type, "Strategy selected based on analysis"))

        return ". ".join(reasons) + "."

    def _identify_alternative_strategies(self, tournament: Tournament,
                                       prioritized_questions: List[QuestionPriority],
                                       market_analysis: Dict[int, MarketAnalysis]) -> List[Tuple[StrategyType, float]]:
        """Identify alternative strategies with confidence scores."""
        alternatives = []

        # Calculate scores for all strategies
        all_strategies = list(StrategyType)

        for strategy in all_strategies:
            # Simple scoring based on tournament characteristics
            score = 0.5  # Base score

            if strategy == StrategyType.AGGRESSIVE:
                if prioritized_questions:
                    high_priority_ratio = sum(1 for p in prioritized_questions
                                            if p.priority_level == PriorityLevel.CRITICAL) / len(prioritized_questions)
                    score += high_priority_ratio * 0.3

            elif strategy == StrategyType.CONSERVATIVE:
                if market_analysis:
                    competitive_ratio = sum(1 for a in market_analysis.values()
                                          if a.market_condition == MarketCondition.COMPETITIVE) / len(market_analysis)
                    score += competitive_ratio * 0.3

            elif strategy == StrategyType.CONTRARIAN:
                if market_analysis:
                    saturated_ratio = sum(1 for a in market_analysis.values()
                                        if a.market_condition == MarketCondition.OVERSATURATED) / len(market_analysis)
                    score += saturated_ratio * 0.4

            alternatives.append((strategy, score))

        # Sort by score and return top alternatives
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives[:3]  # Top 3 alternatives
