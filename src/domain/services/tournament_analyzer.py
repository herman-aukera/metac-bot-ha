"""Tournament analyzer service for dynamics analysis and competitive intelligence."""

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..entities.forecast import Forecast
from ..entities.question import Question
from ..value_objects.tournament_strategy import (
    CompetitiveIntelligence,
    QuestionCategory,
    RiskProfile,
    TournamentPhase,
    TournamentStrategy,
)


@dataclass
class TournamentPattern:
    """Detected pattern in tournament dynamics."""

    pattern_type: str
    description: str
    confidence: float
    impact_score: float
    categories_affected: List[QuestionCategory]
    time_period: Tuple[datetime, datetime]
    supporting_evidence: Dict[str, Any]


@dataclass
class MarketInefficiency:
    """Detected market inefficiency for competitive advantage."""

    inefficiency_type: str
    category: QuestionCategory
    severity: float
    opportunity_score: float
    description: str
    questions_affected: List[UUID]
    detection_time: datetime
    exploitation_strategy: str


@dataclass
class CompetitivePosition:
    """Current competitive position analysis."""

    overall_rank: Optional[int]
    category_rankings: Dict[QuestionCategory, int]
    performance_trends: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    recommended_actions: List[str]


class TournamentAnalyzer:
    """
    Service for analyzing tournament dynamics and competitive intelligence.

    Provides tournament pattern detection, meta-game analysis, competitive
    positioning analysis, and market inefficiency detection for strategic
    advantage in forecasting tournaments.
    """

    def __init__(self):
        """Initialize tournament analyzer."""
        self._pattern_cache: Dict[str, List[TournamentPattern]] = {}
        self._inefficiency_cache: Dict[str, List[MarketInefficiency]] = {}
        self._competitive_intelligence_cache: Dict[str, CompetitiveIntelligence] = {}

    def analyze_tournament_patterns(
        self,
        tournament_id: str,
        questions: List[Question],
        forecasts: List[Forecast],
        historical_data: Optional[Dict[str, Any]] = None,
    ) -> List[TournamentPattern]:
        """
        Detect patterns in tournament dynamics and meta-game analysis.

        Args:
            tournament_id: Tournament identifier
            questions: List of tournament questions
            forecasts: List of forecasts made in tournament
            historical_data: Historical tournament data for pattern recognition

        Returns:
            List of detected tournament patterns
        """
        patterns = []

        # Analyze question distribution patterns
        patterns.extend(self._analyze_question_distribution_patterns(questions))

        # Analyze scoring patterns
        patterns.extend(self._analyze_scoring_patterns(forecasts, questions))

        # Analyze temporal patterns
        patterns.extend(self._analyze_temporal_patterns(questions, forecasts))

        # Analyze category performance patterns
        patterns.extend(self._analyze_category_patterns(questions, forecasts))

        # Analyze competitive behavior patterns
        if historical_data:
            patterns.extend(
                self._analyze_competitive_patterns(historical_data, forecasts)
            )

        # Cache results
        self._pattern_cache[tournament_id] = patterns

        return patterns

    def detect_market_inefficiencies(
        self,
        tournament_id: str,
        questions: List[Question],
        market_data: Optional[Dict[str, Any]] = None,
        competitor_forecasts: Optional[List[Forecast]] = None,
    ) -> List[MarketInefficiency]:
        """
        Detect market inefficiencies for competitive advantage.

        Args:
            tournament_id: Tournament identifier
            questions: List of tournament questions
            market_data: Market prediction data
            competitor_forecasts: Competitor forecast data

        Returns:
            List of detected market inefficiencies
        """
        inefficiencies = []

        # Analyze prediction variance inefficiencies
        if market_data:
            inefficiencies.extend(
                self._detect_variance_inefficiencies(questions, market_data)
            )

        # Analyze category specialization gaps
        inefficiencies.extend(
            self._detect_specialization_gaps(questions, competitor_forecasts)
        )

        # Analyze timing inefficiencies
        inefficiencies.extend(
            self._detect_timing_inefficiencies(questions, competitor_forecasts)
        )

        # Analyze complexity-based inefficiencies
        inefficiencies.extend(self._detect_complexity_inefficiencies(questions))

        # Analyze consensus inefficiencies
        if competitor_forecasts:
            inefficiencies.extend(
                self._detect_consensus_inefficiencies(questions, competitor_forecasts)
            )

        # Cache results
        self._inefficiency_cache[tournament_id] = inefficiencies

        return inefficiencies

    def analyze_competitive_position(
        self,
        tournament_id: str,
        our_forecasts: List[Forecast],
        questions: List[Question],
        tournament_standings: Optional[Dict[str, float]] = None,
        competitor_data: Optional[Dict[str, Any]] = None,
    ) -> CompetitivePosition:
        """
        Analyze current competitive position in tournament.

        Args:
            tournament_id: Tournament identifier
            our_forecasts: Our forecasts in the tournament
            questions: Tournament questions
            tournament_standings: Current tournament standings
            competitor_data: Competitor performance data

        Returns:
            Competitive position analysis
        """
        # Calculate overall rank
        overall_rank = None
        if tournament_standings:
            our_score = tournament_standings.get("our_bot", 0.0)
            better_scores = sum(
                1 for score in tournament_standings.values() if score > our_score
            )
            overall_rank = better_scores + 1

        # Calculate category rankings
        category_rankings = self._calculate_category_rankings(
            our_forecasts, questions, competitor_data
        )

        # Calculate performance trends
        performance_trends = self._calculate_performance_trends(
            our_forecasts, questions
        )

        # Perform SWOT analysis
        strengths, weaknesses, opportunities, threats = self._perform_swot_analysis(
            our_forecasts, questions, tournament_standings, competitor_data
        )

        # Generate recommended actions
        recommended_actions = self._generate_strategic_recommendations(
            strengths, weaknesses, opportunities, threats, category_rankings
        )

        return CompetitivePosition(
            overall_rank=overall_rank,
            category_rankings=category_rankings,
            performance_trends=performance_trends,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            threats=threats,
            recommended_actions=recommended_actions,
        )

    def optimize_tournament_scoring(
        self,
        tournament_id: str,
        questions: List[Question],
        current_strategy: TournamentStrategy,
        performance_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize tournament-specific scoring strategies.

        Args:
            tournament_id: Tournament identifier
            questions: Tournament questions
            current_strategy: Current tournament strategy
            performance_data: Historical performance data

        Returns:
            Scoring optimization recommendations
        """
        optimizations = {
            "confidence_adjustments": {},
            "resource_reallocation": {},
            "timing_optimizations": {},
            "category_focus_changes": {},
            "risk_profile_adjustments": {},
        }

        # Analyze current scoring efficiency
        scoring_efficiency = self._analyze_scoring_efficiency(
            questions, current_strategy, performance_data
        )

        # Optimize confidence thresholds
        optimizations["confidence_adjustments"] = self._optimize_confidence_thresholds(
            questions, current_strategy, scoring_efficiency
        )

        # Optimize resource allocation
        optimizations["resource_reallocation"] = self._optimize_resource_allocation(
            questions, current_strategy, scoring_efficiency
        )

        # Optimize submission timing
        optimizations["timing_optimizations"] = self._optimize_submission_timing(
            questions, current_strategy
        )

        # Optimize category focus
        optimizations["category_focus_changes"] = self._optimize_category_focus(
            questions, current_strategy, scoring_efficiency
        )

        # Optimize risk profile
        optimizations["risk_profile_adjustments"] = self._optimize_risk_profile(
            questions, current_strategy, performance_data
        )

        return optimizations

    def update_competitive_intelligence(
        self, tournament_id: str, new_data: Dict[str, Any]
    ) -> CompetitiveIntelligence:
        """
        Update competitive intelligence with new tournament data.

        Args:
            tournament_id: Tournament identifier
            new_data: New competitive intelligence data

        Returns:
            Updated competitive intelligence
        """
        current_intelligence = self._competitive_intelligence_cache.get(
            tournament_id, CompetitiveIntelligence.create_empty(tournament_id)
        )

        # Update standings
        if "standings" in new_data:
            current_intelligence = CompetitiveIntelligence(
                tournament_id=current_intelligence.tournament_id,
                current_standings=new_data["standings"],
                market_inefficiencies=current_intelligence.market_inefficiencies,
                competitor_patterns=current_intelligence.competitor_patterns,
                scoring_trends=current_intelligence.scoring_trends,
                question_difficulty_distribution=current_intelligence.question_difficulty_distribution,
                timestamp=datetime.utcnow(),
            )

        # Update market inefficiencies
        if "market_inefficiencies" in new_data:
            inefficiencies = list(current_intelligence.market_inefficiencies)
            inefficiencies.extend(new_data["market_inefficiencies"])
            current_intelligence = CompetitiveIntelligence(
                tournament_id=current_intelligence.tournament_id,
                current_standings=current_intelligence.current_standings,
                market_inefficiencies=inefficiencies,
                competitor_patterns=current_intelligence.competitor_patterns,
                scoring_trends=current_intelligence.scoring_trends,
                question_difficulty_distribution=current_intelligence.question_difficulty_distribution,
                timestamp=datetime.utcnow(),
            )

        # Update competitor patterns
        if "competitor_patterns" in new_data:
            patterns = dict(current_intelligence.competitor_patterns)
            patterns.update(new_data["competitor_patterns"])
            current_intelligence = CompetitiveIntelligence(
                tournament_id=current_intelligence.tournament_id,
                current_standings=current_intelligence.current_standings,
                market_inefficiencies=current_intelligence.market_inefficiencies,
                competitor_patterns=patterns,
                scoring_trends=current_intelligence.scoring_trends,
                question_difficulty_distribution=current_intelligence.question_difficulty_distribution,
                timestamp=datetime.utcnow(),
            )

        # Cache updated intelligence
        self._competitive_intelligence_cache[tournament_id] = current_intelligence

        return current_intelligence

    def _analyze_question_distribution_patterns(
        self, questions: List[Question]
    ) -> List[TournamentPattern]:
        """Analyze patterns in question distribution."""
        patterns = []

        # Analyze category distribution
        category_counts = defaultdict(int)
        for question in questions:
            category = question.categorize_question()
            category_counts[category] += 1

        total_questions = len(questions)
        if total_questions > 0:
            # Detect category imbalances
            for category, count in category_counts.items():
                proportion = count / total_questions
                if proportion > 0.3:  # High concentration
                    patterns.append(
                        TournamentPattern(
                            pattern_type="category_concentration",
                            description=f"High concentration of {category.value} questions ({proportion:.1%})",
                            confidence=0.8,
                            impact_score=proportion,
                            categories_affected=[category],
                            time_period=(
                                datetime.utcnow() - timedelta(days=30),
                                datetime.utcnow(),
                            ),
                            supporting_evidence={
                                "proportion": proportion,
                                "count": count,
                            },
                        )
                    )

        return patterns

    def _analyze_scoring_patterns(
        self, forecasts: List[Forecast], questions: List[Question]
    ) -> List[TournamentPattern]:
        """Analyze patterns in scoring and forecast performance."""
        patterns = []

        if not forecasts:
            return patterns

        # Analyze accuracy by category
        category_accuracy = defaultdict(list)
        for forecast in forecasts:
            if forecast.accuracy_score is not None:
                question = next(
                    (q for q in questions if q.id == forecast.question_id), None
                )
                if question:
                    category = question.categorize_question()
                    category_accuracy[category].append(forecast.accuracy_score)

        # Detect performance patterns
        for category, scores in category_accuracy.items():
            if len(scores) >= 3:  # Minimum sample size
                avg_accuracy = statistics.mean(scores)
                if avg_accuracy > 0.7:  # High performance
                    patterns.append(
                        TournamentPattern(
                            pattern_type="high_category_performance",
                            description=f"Strong performance in {category.value} questions (avg: {avg_accuracy:.2f})",
                            confidence=0.7,
                            impact_score=avg_accuracy,
                            categories_affected=[category],
                            time_period=(
                                datetime.utcnow() - timedelta(days=30),
                                datetime.utcnow(),
                            ),
                            supporting_evidence={
                                "average_accuracy": avg_accuracy,
                                "sample_size": len(scores),
                            },
                        )
                    )
                elif avg_accuracy < 0.4:  # Poor performance
                    patterns.append(
                        TournamentPattern(
                            pattern_type="low_category_performance",
                            description=f"Weak performance in {category.value} questions (avg: {avg_accuracy:.2f})",
                            confidence=0.7,
                            impact_score=1.0 - avg_accuracy,
                            categories_affected=[category],
                            time_period=(
                                datetime.utcnow() - timedelta(days=30),
                                datetime.utcnow(),
                            ),
                            supporting_evidence={
                                "average_accuracy": avg_accuracy,
                                "sample_size": len(scores),
                            },
                        )
                    )

        return patterns

    def _analyze_temporal_patterns(
        self, questions: List[Question], forecasts: List[Forecast]
    ) -> List[TournamentPattern]:
        """Analyze temporal patterns in tournament dynamics."""
        patterns = []

        # Analyze submission timing patterns
        if forecasts:
            submission_times = []
            for forecast in forecasts:
                question = next(
                    (q for q in questions if q.id == forecast.question_id), None
                )
                if question and question.close_time:
                    time_to_close = (
                        question.close_time - forecast.created_at
                    ).total_seconds() / 3600  # hours
                    submission_times.append(time_to_close)

            if submission_times:
                avg_time_to_close = statistics.mean(submission_times)
                if avg_time_to_close < 24:  # Last-minute submissions
                    patterns.append(
                        TournamentPattern(
                            pattern_type="late_submission_pattern",
                            description=f"Tendency for late submissions (avg: {avg_time_to_close:.1f} hours before close)",
                            confidence=0.6,
                            impact_score=0.3,
                            categories_affected=[],
                            time_period=(
                                datetime.utcnow() - timedelta(days=30),
                                datetime.utcnow(),
                            ),
                            supporting_evidence={
                                "average_hours_before_close": avg_time_to_close
                            },
                        )
                    )

        return patterns

    def _analyze_category_patterns(
        self, questions: List[Question], forecasts: List[Forecast]
    ) -> List[TournamentPattern]:
        """Analyze category-specific patterns."""
        patterns = []

        # Analyze category difficulty patterns
        category_difficulties = defaultdict(list)
        for question in questions:
            category = question.categorize_question()
            difficulty = question.calculate_difficulty_score()
            category_difficulties[category].append(difficulty)

        for category, difficulties in category_difficulties.items():
            if len(difficulties) >= 3:
                avg_difficulty = statistics.mean(difficulties)
                if avg_difficulty > 0.7:
                    patterns.append(
                        TournamentPattern(
                            pattern_type="high_category_difficulty",
                            description=f"{category.value} questions are particularly difficult (avg: {avg_difficulty:.2f})",
                            confidence=0.7,
                            impact_score=avg_difficulty,
                            categories_affected=[category],
                            time_period=(
                                datetime.utcnow() - timedelta(days=30),
                                datetime.utcnow(),
                            ),
                            supporting_evidence={
                                "average_difficulty": avg_difficulty,
                                "sample_size": len(difficulties),
                            },
                        )
                    )

        return patterns

    def _analyze_competitive_patterns(
        self, historical_data: Dict[str, Any], forecasts: List[Forecast]
    ) -> List[TournamentPattern]:
        """Analyze competitive behavior patterns."""
        patterns = []

        # Analyze consensus patterns
        if "consensus_data" in historical_data:
            consensus_data = historical_data["consensus_data"]
            high_consensus_questions = [
                q
                for q, data in consensus_data.items()
                if data.get("variance", 1.0) < 0.1
            ]

            if len(high_consensus_questions) > len(consensus_data) * 0.3:
                patterns.append(
                    TournamentPattern(
                        pattern_type="high_consensus_environment",
                        description="Tournament shows high consensus on many questions",
                        confidence=0.6,
                        impact_score=0.4,
                        categories_affected=[],
                        time_period=(
                            datetime.utcnow() - timedelta(days=30),
                            datetime.utcnow(),
                        ),
                        supporting_evidence={
                            "high_consensus_proportion": len(high_consensus_questions)
                            / len(consensus_data)
                        },
                    )
                )

        return patterns

    def _detect_variance_inefficiencies(
        self, questions: List[Question], market_data: Dict[str, Any]
    ) -> List[MarketInefficiency]:
        """Detect inefficiencies based on prediction variance."""
        inefficiencies = []

        for question in questions:
            question_data = market_data.get(str(question.id))
            if question_data and "variance" in question_data:
                variance = question_data["variance"]
                if variance > 0.3:  # High variance indicates inefficiency
                    inefficiencies.append(
                        MarketInefficiency(
                            inefficiency_type="high_variance",
                            category=question.categorize_question(),
                            severity=min(1.0, variance),
                            opportunity_score=min(0.8, variance * 2),
                            description=f"High prediction variance ({variance:.2f}) indicates market uncertainty",
                            questions_affected=[question.id],
                            detection_time=datetime.utcnow(),
                            exploitation_strategy="confident_contrarian",
                        )
                    )

        return inefficiencies

    def _detect_specialization_gaps(
        self, questions: List[Question], competitor_forecasts: Optional[List[Forecast]]
    ) -> List[MarketInefficiency]:
        """Detect gaps in category specialization."""
        inefficiencies = []

        if not competitor_forecasts:
            return inefficiencies

        # Analyze competitor performance by category
        competitor_category_performance = defaultdict(list)
        for forecast in competitor_forecasts:
            if forecast.accuracy_score is not None:
                question = next(
                    (q for q in questions if q.id == forecast.question_id), None
                )
                if question:
                    category = question.categorize_question()
                    competitor_category_performance[category].append(
                        forecast.accuracy_score
                    )

        # Identify underperforming categories
        for category, scores in competitor_category_performance.items():
            if len(scores) >= 3:
                avg_performance = statistics.mean(scores)
                if avg_performance < 0.5:  # Poor competitor performance
                    category_questions = [
                        q.id for q in questions if q.categorize_question() == category
                    ]
                    inefficiencies.append(
                        MarketInefficiency(
                            inefficiency_type="specialization_gap",
                            category=category,
                            severity=1.0 - avg_performance,
                            opportunity_score=min(0.9, (1.0 - avg_performance) * 1.5),
                            description=f"Competitors underperforming in {category.value} (avg: {avg_performance:.2f})",
                            questions_affected=category_questions,
                            detection_time=datetime.utcnow(),
                            exploitation_strategy="category_specialization",
                        )
                    )

        return inefficiencies

    def _detect_timing_inefficiencies(
        self, questions: List[Question], competitor_forecasts: Optional[List[Forecast]]
    ) -> List[MarketInefficiency]:
        """Detect timing-based inefficiencies."""
        inefficiencies = []

        if not competitor_forecasts:
            return inefficiencies

        # Analyze submission timing patterns
        late_submissions = []
        for forecast in competitor_forecasts:
            question = next(
                (q for q in questions if q.id == forecast.question_id), None
            )
            if question and question.close_time:
                hours_before_close = (
                    question.close_time - forecast.created_at
                ).total_seconds() / 3600
                if hours_before_close < 6:  # Very late submission
                    late_submissions.append(question.id)

        if len(late_submissions) > len(competitor_forecasts) * 0.3:
            inefficiencies.append(
                MarketInefficiency(
                    inefficiency_type="timing_inefficiency",
                    category=QuestionCategory.OTHER,
                    severity=0.6,
                    opportunity_score=0.4,
                    description="Many competitors making last-minute submissions",
                    questions_affected=late_submissions,
                    detection_time=datetime.utcnow(),
                    exploitation_strategy="early_research_advantage",
                )
            )

        return inefficiencies

    def _detect_complexity_inefficiencies(
        self, questions: List[Question]
    ) -> List[MarketInefficiency]:
        """Detect inefficiencies based on question complexity."""
        inefficiencies = []

        high_complexity_questions = []
        for question in questions:
            complexity = question.calculate_research_complexity_score()
            if complexity > 0.8:
                high_complexity_questions.append(question.id)

        if high_complexity_questions:
            inefficiencies.append(
                MarketInefficiency(
                    inefficiency_type="complexity_avoidance",
                    category=QuestionCategory.OTHER,
                    severity=0.7,
                    opportunity_score=0.6,
                    description=f"{len(high_complexity_questions)} high-complexity questions may be underresearched",
                    questions_affected=high_complexity_questions,
                    detection_time=datetime.utcnow(),
                    exploitation_strategy="deep_research_advantage",
                )
            )

        return inefficiencies

    def _detect_consensus_inefficiencies(
        self, questions: List[Question], competitor_forecasts: List[Forecast]
    ) -> List[MarketInefficiency]:
        """Detect consensus-based inefficiencies."""
        inefficiencies = []

        # Group forecasts by question
        question_forecasts = defaultdict(list)
        for forecast in competitor_forecasts:
            question_forecasts[forecast.question_id].append(forecast)

        # Analyze consensus strength
        for question_id, forecasts in question_forecasts.items():
            if len(forecasts) >= 3:
                predictions = [
                    f.prediction.probability
                    for f in forecasts
                    if hasattr(f.prediction, "probability")
                ]
                if len(predictions) >= 3:
                    variance = statistics.variance(predictions)
                    if variance < 0.05:  # Very low variance = strong consensus
                        question = next(
                            (q for q in questions if q.id == question_id), None
                        )
                        if question:
                            inefficiencies.append(
                                MarketInefficiency(
                                    inefficiency_type="consensus_trap",
                                    category=question.categorize_question(),
                                    severity=0.5,
                                    opportunity_score=0.3,
                                    description=f"Strong consensus (variance: {variance:.3f}) may indicate groupthink",
                                    questions_affected=[question_id],
                                    detection_time=datetime.utcnow(),
                                    exploitation_strategy="contrarian_analysis",
                                )
                            )

        return inefficiencies

    def _calculate_category_rankings(
        self,
        our_forecasts: List[Forecast],
        questions: List[Question],
        competitor_data: Optional[Dict[str, Any]],
    ) -> Dict[QuestionCategory, int]:
        """Calculate our ranking in each category."""
        rankings = {}

        # Calculate our performance by category
        our_category_performance = defaultdict(list)
        for forecast in our_forecasts:
            if forecast.accuracy_score is not None:
                question = next(
                    (q for q in questions if q.id == forecast.question_id), None
                )
                if question:
                    category = question.categorize_question()
                    our_category_performance[category].append(forecast.accuracy_score)

        # Compare with competitors if data available
        if competitor_data and "category_performance" in competitor_data:
            competitor_performance = competitor_data["category_performance"]

            for category, our_scores in our_category_performance.items():
                if our_scores:
                    our_avg = statistics.mean(our_scores)
                    competitor_avgs = competitor_performance.get(category.value, [])

                    if competitor_avgs:
                        better_performers = sum(
                            1 for avg in competitor_avgs if avg > our_avg
                        )
                        rankings[category] = better_performers + 1
                    else:
                        rankings[category] = 1  # No competition data

        return rankings

    def _calculate_performance_trends(
        self, our_forecasts: List[Forecast], questions: List[Question]
    ) -> Dict[str, float]:
        """Calculate performance trends over time."""
        trends = {}

        if len(our_forecasts) < 5:
            return trends

        # Sort forecasts by creation time
        sorted_forecasts = sorted(our_forecasts, key=lambda f: f.created_at)

        # Calculate recent vs. early performance
        mid_point = len(sorted_forecasts) // 2
        early_forecasts = sorted_forecasts[:mid_point]
        recent_forecasts = sorted_forecasts[mid_point:]

        early_scores = [
            f.accuracy_score for f in early_forecasts if f.accuracy_score is not None
        ]
        recent_scores = [
            f.accuracy_score for f in recent_forecasts if f.accuracy_score is not None
        ]

        if early_scores and recent_scores:
            early_avg = statistics.mean(early_scores)
            recent_avg = statistics.mean(recent_scores)
            trends["accuracy_trend"] = recent_avg - early_avg
            trends["improvement_rate"] = (
                (recent_avg - early_avg) / early_avg if early_avg > 0 else 0.0
            )

        return trends

    def _perform_swot_analysis(
        self,
        our_forecasts: List[Forecast],
        questions: List[Question],
        tournament_standings: Optional[Dict[str, float]],
        competitor_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Perform SWOT analysis for competitive positioning."""
        strengths = []
        weaknesses = []
        opportunities = []
        threats = []

        # Analyze strengths
        category_performance = defaultdict(list)
        for forecast in our_forecasts:
            if forecast.accuracy_score is not None:
                question = next(
                    (q for q in questions if q.id == forecast.question_id), None
                )
                if question:
                    category = question.categorize_question()
                    category_performance[category].append(forecast.accuracy_score)

        for category, scores in category_performance.items():
            if scores:
                avg_score = statistics.mean(scores)
                if avg_score > 0.7:
                    strengths.append(
                        f"Strong performance in {category.value} questions"
                    )
                elif avg_score < 0.4:
                    weaknesses.append(f"Weak performance in {category.value} questions")

        # Analyze opportunities
        if len(questions) > len(our_forecasts):
            opportunities.append("Untapped questions available for forecasting")

        # Analyze threats
        if tournament_standings:
            our_score = tournament_standings.get("our_bot", 0.0)
            top_scores = sorted(tournament_standings.values(), reverse=True)[:3]
            if our_score not in top_scores:
                threats.append("Not in top 3 performers")

        return strengths, weaknesses, opportunities, threats

    def _generate_strategic_recommendations(
        self,
        strengths: List[str],
        weaknesses: List[str],
        opportunities: List[str],
        threats: List[str],
        category_rankings: Dict[QuestionCategory, int],
    ) -> List[str]:
        """Generate strategic recommendations based on analysis."""
        recommendations = []

        # Leverage strengths
        strong_categories = [
            cat for cat, rank in category_rankings.items() if rank <= 3
        ]
        if strong_categories:
            recommendations.append(
                f"Focus resources on strong categories: {', '.join(cat.value for cat in strong_categories)}"
            )

        # Address weaknesses
        weak_categories = [cat for cat, rank in category_rankings.items() if rank > 5]
        if weak_categories:
            recommendations.append(
                f"Improve performance in weak categories: {', '.join(cat.value for cat in weak_categories)}"
            )

        # Exploit opportunities
        if "Untapped questions" in str(opportunities):
            recommendations.append(
                "Increase forecasting coverage to capture more scoring opportunities"
            )

        # Mitigate threats
        if "Not in top 3" in str(threats):
            recommendations.append(
                "Implement aggressive strategy to improve tournament ranking"
            )

        return recommendations

    def _analyze_scoring_efficiency(
        self,
        questions: List[Question],
        current_strategy: TournamentStrategy,
        performance_data: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Analyze current scoring efficiency."""
        efficiency = {}

        # Calculate category efficiency
        for category in QuestionCategory:
            category_questions = [
                q for q in questions if q.categorize_question() == category
            ]
            if category_questions:
                specialization = current_strategy.category_specializations.get(
                    category, 0.5
                )
                threshold = current_strategy.get_category_confidence_threshold(category)

                # Simple efficiency metric
                efficiency[category.value] = specialization * (1.0 - threshold)

        return efficiency

    def _optimize_confidence_thresholds(
        self,
        questions: List[Question],
        current_strategy: TournamentStrategy,
        scoring_efficiency: Dict[str, float],
    ) -> Dict[str, float]:
        """Optimize confidence thresholds for better scoring."""
        optimizations = {}

        for category in QuestionCategory:
            current_threshold = current_strategy.get_category_confidence_threshold(
                category
            )
            efficiency = scoring_efficiency.get(category.value, 0.5)

            if efficiency < 0.3:  # Low efficiency
                # Increase threshold to be more selective
                optimizations[category.value] = min(0.9, current_threshold + 0.1)
            elif efficiency > 0.7:  # High efficiency
                # Decrease threshold to capture more opportunities
                optimizations[category.value] = max(0.1, current_threshold - 0.1)

        return optimizations

    def _optimize_resource_allocation(
        self,
        questions: List[Question],
        current_strategy: TournamentStrategy,
        scoring_efficiency: Dict[str, float],
    ) -> Dict[str, float]:
        """Optimize resource allocation across categories."""
        optimizations = {}

        # Identify high-opportunity categories
        category_opportunities = {}
        for question in questions:
            category = question.categorize_question()
            scoring_potential = question.calculate_scoring_potential()

            if category.value not in category_opportunities:
                category_opportunities[category.value] = []
            category_opportunities[category.value].append(scoring_potential)

        # Calculate average opportunity by category
        for category, potentials in category_opportunities.items():
            avg_potential = statistics.mean(potentials) if potentials else 0.5
            current_allocation = current_strategy.category_specializations.get(
                QuestionCategory(category), 0.5
            )

            # Adjust allocation based on opportunity
            if avg_potential > 0.7:
                optimizations[category] = min(1.0, current_allocation + 0.1)
            elif avg_potential < 0.3:
                optimizations[category] = max(0.1, current_allocation - 0.1)

        return optimizations

    def _optimize_submission_timing(
        self, questions: List[Question], current_strategy: TournamentStrategy
    ) -> Dict[str, Any]:
        """Optimize submission timing strategy."""
        optimizations = {}

        # Analyze question deadlines
        urgent_questions = [q for q in questions if q.days_until_close() <= 3]
        medium_term_questions = [q for q in questions if 3 < q.days_until_close() <= 14]
        long_term_questions = [q for q in questions if q.days_until_close() > 14]

        optimizations["urgent_priority"] = (
            len(urgent_questions) / len(questions) if questions else 0
        )
        optimizations["medium_term_allocation"] = 0.6 if medium_term_questions else 0.3
        optimizations["long_term_allocation"] = 0.3 if long_term_questions else 0.1

        # Recommend timing strategy
        if len(urgent_questions) > len(questions) * 0.3:
            optimizations["recommended_strategy"] = "immediate_focus"
        else:
            optimizations["recommended_strategy"] = "balanced_timing"

        return optimizations

    def _optimize_category_focus(
        self,
        questions: List[Question],
        current_strategy: TournamentStrategy,
        scoring_efficiency: Dict[str, float],
    ) -> Dict[str, float]:
        """Optimize category focus based on tournament composition."""
        optimizations = {}

        # Calculate category distribution in tournament
        category_distribution = defaultdict(int)
        for question in questions:
            category = question.categorize_question()
            category_distribution[category] += 1

        total_questions = len(questions)

        for category, count in category_distribution.items():
            proportion = count / total_questions if total_questions > 0 else 0
            current_specialization = current_strategy.category_specializations.get(
                category, 0.5
            )
            efficiency = scoring_efficiency.get(category.value, 0.5)

            # Increase focus on high-proportion, high-efficiency categories
            if proportion > 0.2 and efficiency > 0.6:
                optimizations[category.value] = min(1.0, current_specialization + 0.15)
            # Decrease focus on low-proportion, low-efficiency categories
            elif proportion < 0.1 and efficiency < 0.4:
                optimizations[category.value] = max(0.1, current_specialization - 0.1)

        return optimizations

    def _optimize_risk_profile(
        self,
        questions: List[Question],
        current_strategy: TournamentStrategy,
        performance_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Optimize risk profile based on tournament position."""
        optimizations = {}

        # Analyze tournament phase
        if performance_data and "tournament_progress" in performance_data:
            progress = performance_data["tournament_progress"]

            if progress > 0.8:  # Late tournament
                optimizations["recommended_profile"] = RiskProfile.AGGRESSIVE.value
                optimizations["rationale"] = (
                    "Late tournament phase - take calculated risks"
                )
            elif progress < 0.3:  # Early tournament
                optimizations["recommended_profile"] = RiskProfile.CONSERVATIVE.value
                optimizations["rationale"] = (
                    "Early tournament phase - build solid foundation"
                )
            else:  # Mid tournament
                optimizations["recommended_profile"] = RiskProfile.MODERATE.value
                optimizations["rationale"] = "Mid tournament phase - balanced approach"

        # Analyze current position
        if performance_data and "current_rank" in performance_data:
            rank = performance_data["current_rank"]
            total_participants = performance_data.get("total_participants", 100)

            if rank <= total_participants * 0.1:  # Top 10%
                optimizations["position_adjustment"] = "maintain_conservative"
            elif rank >= total_participants * 0.7:  # Bottom 30%
                optimizations["position_adjustment"] = "increase_aggressive"
            else:
                optimizations["position_adjustment"] = "stay_moderate"

        return optimizations
