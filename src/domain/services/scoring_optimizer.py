"""Scoring optimizer service for tournament metrics optimization."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
import statistics
from collections import defaultdict
import math

from ..entities.question import Question
from ..entities.forecast import Forecast, calculate_brier_score
from ..entities.prediction import Prediction
from ..value_objects.tournament_strategy import (
    QuestionCategory, TournamentStrategy, RiskProfile, TournamentPhase
)


@dataclass
class ScoringMetrics:
    """Tournament scoring metrics and analysis."""
    brier_score: float
    log_score: float
    calibration_score: float
    resolution_score: float
    reliability_score: float
    sharpness_score: float
    tournament_rank: Optional[int]
    category_performance: Dict[QuestionCategory, float]
    confidence_accuracy_mapping: Dict[str, float]


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation for scoring improvement."""
    recommendation_type: str
    category: Optional[QuestionCategory]
    current_value: float
    recommended_value: float
    expected_improvement: float
    confidence: float
    rationale: str
    implementation_priority: str  # "high", "medium", "low"
    risk_level: str  # "low", "medium", "high"


@dataclass
class SubmissionTiming:
    """Optimal submission timing analysis."""
    question_id: UUID
    optimal_submission_time: datetime
    current_time: datetime
    hours_until_optimal: float
    confidence_in_timing: float
    timing_strategy: str
    risk_factors: List[str]
    expected_score_improvement: float


class ScoringOptimizer:
    """
    Service for tournament-specific scoring optimization.

    Implements tournament-specific scoring optimization algorithms,
    confidence-based scoring strategies, risk adjustment, and
    submission timing optimization for maximum tournament impact.
    """

    def __init__(self):
        """Initialize scoring optimizer."""
        self._scoring_history: Dict[str, List[ScoringMetrics]] = {}
        self._optimization_cache: Dict[str, List[OptimizationRecommendation]] = {}
        self._timing_cache: Dict[UUID, SubmissionTiming] = {}

    def calculate_tournament_scoring_metrics(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        resolved_outcomes: Optional[Dict[UUID, Any]] = None,
        tournament_context: Optional[Dict[str, Any]] = None
    ) -> ScoringMetrics:
        """
        Calculate comprehensive tournament scoring metrics.

        Args:
            forecasts: List of forecasts to evaluate
            questions: List of questions for context
            resolved_outcomes: Actual outcomes for resolved questions
            tournament_context: Tournament-specific context

        Returns:
            Comprehensive scoring metrics
        """
        if not forecasts:
            return self._create_empty_metrics()

        # Calculate Brier scores
        brier_scores = self._calculate_brier_scores(forecasts, resolved_outcomes)
        avg_brier = statistics.mean(brier_scores) if brier_scores else 0.5

        # Calculate log scores
        log_scores = self._calculate_log_scores(forecasts, resolved_outcomes)
        avg_log_score = statistics.mean(log_scores) if log_scores else -0.693  # ln(0.5)

        # Calculate calibration metrics
        calibration_score = self._calculate_calibration_score(forecasts, resolved_outcomes)

        # Calculate resolution and reliability
        resolution_score = self._calculate_resolution_score(forecasts, resolved_outcomes)
        reliability_score = self._calculate_reliability_score(forecasts, resolved_outcomes)

        # Calculate sharpness (how far predictions are from 0.5)
        sharpness_score = self._calculate_sharpness_score(forecasts)

        # Calculate category performance
        category_performance = self._calculate_category_performance(
            forecasts, questions, resolved_outcomes
        )

        # Calculate confidence-accuracy mapping
        confidence_accuracy = self._calculate_confidence_accuracy_mapping(
            forecasts, resolved_outcomes
        )

        # Determine tournament rank if context available
        tournament_rank = None
        if tournament_context and "standings" in tournament_context:
            tournament_rank = self._calculate_tournament_rank(
                avg_brier, tournament_context["standings"]
            )

        return ScoringMetrics(
            brier_score=avg_brier,
            log_score=avg_log_score,
            calibration_score=calibration_score,
            resolution_score=resolution_score,
            reliability_score=reliability_score,
            sharpness_score=sharpness_score,
            tournament_rank=tournament_rank,
            category_performance=category_performance,
            confidence_accuracy_mapping=confidence_accuracy
        )

    def optimize_confidence_thresholds(
        self,
        tournament_id: str,
        current_strategy: TournamentStrategy,
        historical_performance: Dict[str, Any],
        questions: List[Question]
    ) -> List[OptimizationRecommendation]:
        """
        Optimize confidence thresholds for better scoring.

        Args:
            tournament_id: Tournament identifier
            current_strategy: Current tournament strategy
            historical_performance: Historical performance data
            questions: Current tournament questions

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Analyze current confidence threshold performance
        current_thresholds = current_strategy.confidence_thresholds

        for threshold_type, current_value in current_thresholds.items():
            # Calculate optimal threshold based on historical data
            optimal_value = self._calculate_optimal_confidence_threshold(
                threshold_type, historical_performance, questions
            )

            if abs(optimal_value - current_value) > 0.05:  # Significant difference
                expected_improvement = self._estimate_threshold_improvement(
                    threshold_type, current_value, optimal_value, historical_performance
                )

                recommendations.append(OptimizationRecommendation(
                    recommendation_type="confidence_threshold",
                    category=None,
                    current_value=current_value,
                    recommended_value=optimal_value,
                    expected_improvement=expected_improvement,
                    confidence=0.7,
                    rationale=f"Historical data suggests {threshold_type} threshold of {optimal_value:.2f} would improve scoring by {expected_improvement:.3f}",
                    implementation_priority="high" if expected_improvement > 0.05 else "medium",
                    risk_level="low" if abs(optimal_value - current_value) < 0.1 else "medium"
                ))

        return recommendations

    def optimize_risk_adjusted_scoring(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        tournament_strategy: TournamentStrategy,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Optimize risk-adjusted scoring strategies.

        Args:
            forecasts: Current forecasts
            questions: Tournament questions
            tournament_strategy: Current strategy
            market_conditions: Market condition data

        Returns:
            Risk adjustment recommendations
        """
        recommendations = []

        # Analyze current risk exposure
        risk_analysis = self._analyze_risk_exposure(forecasts, questions)

        # Optimize risk profile based on tournament position
        current_risk_profile = tournament_strategy.risk_profile
        optimal_risk_profile = self._determine_optimal_risk_profile(
            risk_analysis, tournament_strategy, market_conditions
        )

        if optimal_risk_profile != current_risk_profile:
            expected_improvement = self._estimate_risk_profile_improvement(
                current_risk_profile, optimal_risk_profile, risk_analysis
            )

            recommendations.append(OptimizationRecommendation(
                recommendation_type="risk_profile",
                category=None,
                current_value=self._risk_profile_to_numeric(current_risk_profile),
                recommended_value=self._risk_profile_to_numeric(optimal_risk_profile),
                expected_improvement=expected_improvement,
                confidence=0.6,
                rationale=f"Tournament position and market conditions suggest {optimal_risk_profile.value} risk profile",
                implementation_priority="high" if expected_improvement > 0.1 else "medium",
                risk_level="medium"
            ))

        # Optimize category-specific risk adjustments
        category_recommendations = self._optimize_category_risk_adjustments(
            forecasts, questions, tournament_strategy, risk_analysis
        )
        recommendations.extend(category_recommendations)

        return recommendations

    def optimize_submission_timing(
        self,
        question: Question,
        current_forecast: Optional[Forecast],
        tournament_context: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None
    ) -> SubmissionTiming:
        """
        Optimize submission timing for maximum impact.

        Args:
            question: Question to optimize timing for
            current_forecast: Current forecast if available
            tournament_context: Tournament context data
            market_data: Market prediction data

        Returns:
            Optimal submission timing analysis
        """
        current_time = datetime.utcnow()

        # Determine optimal timing strategy
        timing_strategy = self._determine_timing_strategy(
            question, tournament_context, market_data
        )

        # Calculate optimal submission time
        optimal_time = self._calculate_optimal_submission_time(
            question, timing_strategy, tournament_context, market_data
        )

        # Assess timing confidence
        timing_confidence = self._assess_timing_confidence(
            question, timing_strategy, market_data
        )

        # Identify risk factors
        risk_factors = self._identify_timing_risk_factors(
            question, timing_strategy, tournament_context
        )

        # Estimate score improvement from optimal timing
        score_improvement = self._estimate_timing_score_improvement(
            question, timing_strategy, current_forecast
        )

        hours_until_optimal = (optimal_time - current_time).total_seconds() / 3600

        timing = SubmissionTiming(
            question_id=question.id,
            optimal_submission_time=optimal_time,
            current_time=current_time,
            hours_until_optimal=hours_until_optimal,
            confidence_in_timing=timing_confidence,
            timing_strategy=timing_strategy,
            risk_factors=risk_factors,
            expected_score_improvement=score_improvement
        )

        # Cache result
        self._timing_cache[question.id] = timing

        return timing

    def calculate_expected_tournament_score(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        tournament_strategy: TournamentStrategy,
        optimization_recommendations: List[OptimizationRecommendation]
    ) -> Dict[str, float]:
        """
        Calculate expected tournament score with and without optimizations.

        Args:
            forecasts: Current forecasts
            questions: Tournament questions
            tournament_strategy: Current strategy
            optimization_recommendations: Proposed optimizations

        Returns:
            Expected score analysis
        """
        # Calculate current expected score
        current_score = self._calculate_current_expected_score(
            forecasts, questions, tournament_strategy
        )

        # Calculate optimized score
        optimized_score = self._calculate_optimized_expected_score(
            forecasts, questions, tournament_strategy, optimization_recommendations
        )

        # Calculate score by category
        category_scores = self._calculate_category_expected_scores(
            forecasts, questions, tournament_strategy
        )

        # Calculate confidence intervals
        score_confidence = self._calculate_score_confidence_intervals(
            forecasts, questions, tournament_strategy
        )

        return {
            "current_expected_score": current_score,
            "optimized_expected_score": optimized_score,
            "potential_improvement": optimized_score - current_score,
            "improvement_confidence": self._calculate_improvement_confidence(
                optimization_recommendations
            ),
            "category_scores": category_scores,
            "score_confidence_lower": score_confidence["lower"],
            "score_confidence_upper": score_confidence["upper"],
            "risk_adjusted_score": self._calculate_risk_adjusted_score(
                current_score, forecasts, tournament_strategy
            )
        }

    def generate_scoring_strategy_recommendations(
        self,
        tournament_id: str,
        current_performance: ScoringMetrics,
        tournament_context: Dict[str, Any],
        competitor_analysis: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Generate comprehensive scoring strategy recommendations.

        Args:
            tournament_id: Tournament identifier
            current_performance: Current performance metrics
            tournament_context: Tournament context
            competitor_analysis: Competitor performance analysis

        Returns:
            Comprehensive strategy recommendations
        """
        recommendations = []

        # Analyze calibration issues
        if current_performance.calibration_score < 0.8:
            recommendations.extend(self._generate_calibration_recommendations(
                current_performance, tournament_context
            ))

        # Analyze sharpness issues
        if current_performance.sharpness_score < 0.3:
            recommendations.extend(self._generate_sharpness_recommendations(
                current_performance, tournament_context
            ))

        # Analyze category performance issues
        recommendations.extend(self._generate_category_performance_recommendations(
            current_performance, tournament_context
        ))

        # Analyze competitive positioning
        if competitor_analysis:
            recommendations.extend(self._generate_competitive_recommendations(
                current_performance, competitor_analysis, tournament_context
            ))

        # Analyze tournament phase-specific optimizations
        tournament_phase = tournament_context.get("phase", "middle")
        recommendations.extend(self._generate_phase_specific_recommendations(
            current_performance, tournament_phase, tournament_context
        ))

        # Sort by expected improvement and priority
        recommendations.sort(
            key=lambda r: (r.expected_improvement, r.implementation_priority == "high"),
            reverse=True
        )

        return recommendations

    def _create_empty_metrics(self) -> ScoringMetrics:
        """Create empty scoring metrics for initialization."""
        return ScoringMetrics(
            brier_score=0.5,
            log_score=-0.693,
            calibration_score=0.0,
            resolution_score=0.0,
            reliability_score=0.0,
            sharpness_score=0.0,
            tournament_rank=None,
            category_performance={},
            confidence_accuracy_mapping={}
        )

    def _calculate_brier_scores(
        self,
        forecasts: List[Forecast],
        resolved_outcomes: Optional[Dict[UUID, Any]]
    ) -> List[float]:
        """Calculate Brier scores for resolved forecasts."""
        if not resolved_outcomes:
            return []

        brier_scores = []
        for forecast in forecasts:
            if forecast.question_id in resolved_outcomes:
                outcome = resolved_outcomes[forecast.question_id]
                if isinstance(outcome, (int, float)) and outcome in [0, 1]:
                    prediction = forecast.prediction  # Uses backward compatibility property
                    brier_score = calculate_brier_score(prediction, int(outcome))
                    brier_scores.append(brier_score)

        return brier_scores

    def _calculate_log_scores(
        self,
        forecasts: List[Forecast],
        resolved_outcomes: Optional[Dict[UUID, Any]]
    ) -> List[float]:
        """Calculate logarithmic scores for resolved forecasts."""
        if not resolved_outcomes:
            return []

        log_scores = []
        for forecast in forecasts:
            if forecast.question_id in resolved_outcomes:
                outcome = resolved_outcomes[forecast.question_id]
                if isinstance(outcome, (int, float)) and outcome in [0, 1]:
                    prediction = forecast.prediction
                    # Avoid log(0) by clamping predictions
                    clamped_prediction = max(0.001, min(0.999, prediction))

                    if outcome == 1:
                        log_score = math.log(clamped_prediction)
                    else:
                        log_score = math.log(1 - clamped_prediction)

                    log_scores.append(log_score)

        return log_scores

    def _calculate_calibration_score(
        self,
        forecasts: List[Forecast],
        resolved_outcomes: Optional[Dict[UUID, Any]]
    ) -> float:
        """Calculate calibration score (reliability component)."""
        if not resolved_outcomes:
            return 0.0

        # Group predictions by confidence bins
        bins = defaultdict(list)
        for forecast in forecasts:
            if forecast.question_id in resolved_outcomes:
                outcome = resolved_outcomes[forecast.question_id]
                if isinstance(outcome, (int, float)) and outcome in [0, 1]:
                    prediction = forecast.prediction
                    bin_index = int(prediction * 10)  # 10 bins
                    bins[bin_index].append((prediction, int(outcome)))

        # Calculate calibration error
        calibration_error = 0.0
        total_predictions = 0

        for bin_predictions in bins.values():
            if len(bin_predictions) > 0:
                avg_prediction = sum(p[0] for p in bin_predictions) / len(bin_predictions)
                avg_outcome = sum(p[1] for p in bin_predictions) / len(bin_predictions)

                bin_error = abs(avg_prediction - avg_outcome)
                calibration_error += bin_error * len(bin_predictions)
                total_predictions += len(bin_predictions)

        if total_predictions > 0:
            calibration_error /= total_predictions
            return 1.0 - calibration_error  # Convert error to score

        return 0.0

    def _calculate_resolution_score(
        self,
        forecasts: List[Forecast],
        resolved_outcomes: Optional[Dict[UUID, Any]]
    ) -> float:
        """Calculate resolution score (how much predictions vary from base rate)."""
        if not resolved_outcomes:
            return 0.0

        outcomes = []
        predictions = []

        for forecast in forecasts:
            if forecast.question_id in resolved_outcomes:
                outcome = resolved_outcomes[forecast.question_id]
                if isinstance(outcome, (int, float)) and outcome in [0, 1]:
                    outcomes.append(int(outcome))
                    predictions.append(forecast.prediction)

        if len(outcomes) < 2:
            return 0.0

        # Base rate (overall frequency of positive outcomes)
        base_rate = sum(outcomes) / len(outcomes)

        # Resolution is the variance of predictions weighted by frequency
        resolution = 0.0
        for prediction in predictions:
            resolution += (prediction - base_rate) ** 2

        resolution /= len(predictions)
        return resolution

    def _calculate_reliability_score(
        self,
        forecasts: List[Forecast],
        resolved_outcomes: Optional[Dict[UUID, Any]]
    ) -> float:
        """Calculate reliability score (inverse of calibration error)."""
        calibration_score = self._calculate_calibration_score(forecasts, resolved_outcomes)
        return calibration_score  # Already converted from error to score

    def _calculate_sharpness_score(self, forecasts: List[Forecast]) -> float:
        """Calculate sharpness score (how far predictions are from 0.5)."""
        if not forecasts:
            return 0.0

        sharpness_values = []
        for forecast in forecasts:
            prediction = forecast.prediction
            sharpness = abs(prediction - 0.5) * 2  # Scale to 0-1
            sharpness_values.append(sharpness)

        return statistics.mean(sharpness_values)

    def _calculate_category_performance(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        resolved_outcomes: Optional[Dict[UUID, Any]]
    ) -> Dict[QuestionCategory, float]:
        """Calculate performance by question category."""
        if not resolved_outcomes:
            return {}

        category_scores = defaultdict(list)

        for forecast in forecasts:
            if forecast.question_id in resolved_outcomes:
                question = next((q for q in questions if q.id == forecast.question_id), None)
                if question:
                    outcome = resolved_outcomes[forecast.question_id]
                    if isinstance(outcome, (int, float)) and outcome in [0, 1]:
                        category = question.categorize_question()
                        brier_score = calculate_brier_score(forecast.prediction, int(outcome))
                        category_scores[category].append(1.0 - brier_score)  # Convert to accuracy

        # Calculate average performance by category
        category_performance = {}
        for category, scores in category_scores.items():
            if scores:
                category_performance[category] = statistics.mean(scores)

        return category_performance

    def _calculate_confidence_accuracy_mapping(
        self,
        forecasts: List[Forecast],
        resolved_outcomes: Optional[Dict[UUID, Any]]
    ) -> Dict[str, float]:
        """Calculate accuracy by confidence level."""
        if not resolved_outcomes:
            return {}

        confidence_accuracy = defaultdict(list)

        for forecast in forecasts:
            if forecast.question_id in resolved_outcomes:
                outcome = resolved_outcomes[forecast.question_id]
                if isinstance(outcome, (int, float)) and outcome in [0, 1]:
                    confidence_level = forecast.confidence  # Uses backward compatibility property
                    prediction = forecast.prediction

                    # Simple accuracy: 1 if prediction > 0.5 and outcome = 1, or prediction <= 0.5 and outcome = 0
                    correct = (prediction > 0.5 and outcome == 1) or (prediction <= 0.5 and outcome == 0)
                    confidence_accuracy[str(confidence_level)].append(1.0 if correct else 0.0)

        # Calculate average accuracy by confidence level
        mapping = {}
        for confidence_level, accuracies in confidence_accuracy.items():
            if accuracies:
                mapping[confidence_level] = statistics.mean(accuracies)

        return mapping

    def _calculate_tournament_rank(
        self,
        our_brier_score: float,
        standings: Dict[str, float]
    ) -> int:
        """Calculate tournament rank based on Brier score."""
        our_score = 1.0 - our_brier_score  # Convert Brier to accuracy-like score
        better_scores = sum(1 for score in standings.values() if score > our_score)
        return better_scores + 1

    def _calculate_optimal_confidence_threshold(
        self,
        threshold_type: str,
        historical_performance: Dict[str, Any],
        questions: List[Question]
    ) -> float:
        """Calculate optimal confidence threshold based on historical data."""
        # Default thresholds
        defaults = {
            "minimum_submission": 0.6,
            "high_confidence": 0.8,
            "abstention": 0.4
        }

        base_threshold = defaults.get(threshold_type, 0.6)

        # Adjust based on historical performance
        if "accuracy_by_confidence" in historical_performance:
            accuracy_data = historical_performance["accuracy_by_confidence"]

            # Find threshold that maximizes expected score
            best_threshold = base_threshold
            best_score = 0.0

            for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                expected_score = self._estimate_score_at_threshold(
                    threshold, accuracy_data, questions
                )
                if expected_score > best_score:
                    best_score = expected_score
                    best_threshold = threshold

            return best_threshold

        return base_threshold

    def _estimate_threshold_improvement(
        self,
        threshold_type: str,
        current_value: float,
        optimal_value: float,
        historical_performance: Dict[str, Any]
    ) -> float:
        """Estimate improvement from threshold change."""
        # Simple heuristic: larger changes in critical thresholds have more impact
        change_magnitude = abs(optimal_value - current_value)

        if threshold_type == "minimum_submission":
            return change_magnitude * 0.1  # High impact threshold
        elif threshold_type == "high_confidence":
            return change_magnitude * 0.05  # Medium impact
        else:
            return change_magnitude * 0.02  # Lower impact

    def _estimate_score_at_threshold(
        self,
        threshold: float,
        accuracy_data: Dict[str, float],
        questions: List[Question]
    ) -> float:
        """Estimate expected score at given confidence threshold."""
        # Simplified estimation - in practice would use more sophisticated modeling
        base_score = 0.5

        # Higher thresholds generally improve accuracy but reduce coverage
        if threshold > 0.7:
            base_score += 0.1  # Accuracy bonus
            base_score -= (threshold - 0.7) * 0.2  # Coverage penalty
        elif threshold < 0.5:
            base_score -= 0.1  # Accuracy penalty
            base_score += (0.5 - threshold) * 0.1  # Coverage bonus

        return max(0.0, min(1.0, base_score))

    def _analyze_risk_exposure(
        self,
        forecasts: List[Forecast],
        questions: List[Question]
    ) -> Dict[str, float]:
        """Analyze current risk exposure across forecasts."""
        risk_analysis = {
            "prediction_variance": 0.0,
            "confidence_distribution": 0.0,
            "category_concentration": 0.0,
            "timing_risk": 0.0,
            "overall_risk": 0.0
        }

        if not forecasts:
            return risk_analysis

        # Calculate prediction variance
        predictions = [f.prediction for f in forecasts]
        if len(predictions) > 1:
            risk_analysis["prediction_variance"] = statistics.variance(predictions)

        # Calculate confidence distribution risk
        confidence_scores = [f.confidence for f in forecasts]
        if confidence_scores:
            # Risk is higher when all predictions have similar confidence
            unique_confidences = len(set(confidence_scores))
            risk_analysis["confidence_distribution"] = 1.0 - (unique_confidences / len(confidence_scores))

        # Calculate category concentration risk
        categories = []
        for forecast in forecasts:
            question = next((q for q in questions if q.id == forecast.question_id), None)
            if question:
                categories.append(question.categorize_question())

        if categories:
            category_counts = defaultdict(int)
            for category in categories:
                category_counts[category] += 1

            max_concentration = max(category_counts.values()) / len(categories)
            risk_analysis["category_concentration"] = max_concentration

        # Calculate timing risk (simplified)
        current_time = datetime.utcnow()
        timing_risks = []
        for forecast in forecasts:
            question = next((q for q in questions if q.id == forecast.question_id), None)
            if question and question.close_time:
                hours_to_close = (question.close_time - current_time).total_seconds() / 3600
                if hours_to_close < 6:
                    timing_risks.append(1.0)  # High risk
                elif hours_to_close < 24:
                    timing_risks.append(0.5)  # Medium risk
                else:
                    timing_risks.append(0.1)  # Low risk

        if timing_risks:
            risk_analysis["timing_risk"] = statistics.mean(timing_risks)

        # Calculate overall risk
        risk_analysis["overall_risk"] = statistics.mean([
            risk_analysis["prediction_variance"],
            risk_analysis["confidence_distribution"],
            risk_analysis["category_concentration"],
            risk_analysis["timing_risk"]
        ])

        return risk_analysis

    def _determine_optimal_risk_profile(
        self,
        risk_analysis: Dict[str, float],
        tournament_strategy: TournamentStrategy,
        market_conditions: Optional[Dict[str, Any]]
    ) -> RiskProfile:
        """Determine optimal risk profile based on analysis."""
        current_risk = risk_analysis["overall_risk"]

        # Consider tournament phase
        if tournament_strategy.phase == TournamentPhase.EARLY:
            if current_risk > 0.7:
                return RiskProfile.CONSERVATIVE
            else:
                return RiskProfile.MODERATE
        elif tournament_strategy.phase == TournamentPhase.LATE:
            if current_risk < 0.3:
                return RiskProfile.AGGRESSIVE
            else:
                return RiskProfile.MODERATE
        else:  # Middle phase
            if current_risk > 0.6:
                return RiskProfile.CONSERVATIVE
            elif current_risk < 0.4:
                return RiskProfile.MODERATE
            else:
                return RiskProfile.ADAPTIVE

        return RiskProfile.MODERATE  # Default

    def _estimate_risk_profile_improvement(
        self,
        current_profile: RiskProfile,
        optimal_profile: RiskProfile,
        risk_analysis: Dict[str, float]
    ) -> float:
        """Estimate improvement from risk profile change."""
        profile_scores = {
            RiskProfile.CONSERVATIVE: 0.3,
            RiskProfile.MODERATE: 0.5,
            RiskProfile.AGGRESSIVE: 0.7,
            RiskProfile.ADAPTIVE: 0.6
        }

        current_score = profile_scores[current_profile]
        optimal_score = profile_scores[optimal_profile]

        # Adjust based on current risk level
        risk_adjustment = risk_analysis["overall_risk"] * 0.2

        return (optimal_score - current_score) * (1 + risk_adjustment)

    def _risk_profile_to_numeric(self, profile: RiskProfile) -> float:
        """Convert risk profile to numeric value."""
        mapping = {
            RiskProfile.CONSERVATIVE: 0.3,
            RiskProfile.MODERATE: 0.5,
            RiskProfile.AGGRESSIVE: 0.7,
            RiskProfile.ADAPTIVE: 0.6
        }
        return mapping[profile]

    def _optimize_category_risk_adjustments(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        tournament_strategy: TournamentStrategy,
        risk_analysis: Dict[str, float]
    ) -> List[OptimizationRecommendation]:
        """Optimize category-specific risk adjustments."""
        recommendations = []

        # Analyze category risk exposure
        category_risks = defaultdict(list)
        for forecast in forecasts:
            question = next((q for q in questions if q.id == forecast.question_id), None)
            if question:
                category = question.categorize_question()
                prediction_risk = abs(forecast.prediction - 0.5) * 2  # Sharpness as risk proxy
                category_risks[category].append(prediction_risk)

        # Generate recommendations for high-risk categories
        for category, risks in category_risks.items():
            if risks:
                avg_risk = statistics.mean(risks)
                current_specialization = tournament_strategy.category_specializations.get(category, 0.5)

                if avg_risk > 0.8:  # High risk category
                    recommended_specialization = max(0.3, current_specialization - 0.1)

                    recommendations.append(OptimizationRecommendation(
                        recommendation_type="category_risk_adjustment",
                        category=category,
                        current_value=current_specialization,
                        recommended_value=recommended_specialization,
                        expected_improvement=0.05,
                        confidence=0.6,
                        rationale=f"High risk exposure in {category.value} category suggests reducing specialization",
                        implementation_priority="medium",
                        risk_level="low"
                    ))

        return recommendations

    def _determine_timing_strategy(
        self,
        question: Question,
        tournament_context: Dict[str, Any],
        market_data: Optional[Dict[str, Any]]
    ) -> str:
        """Determine optimal timing strategy for question."""
        # Consider question characteristics
        difficulty = question.calculate_difficulty_score()
        days_to_close = question.days_until_close()

        # Consider tournament context
        tournament_phase = tournament_context.get("phase", "middle")
        competition_level = tournament_context.get("competition_level", 0.5)

        # Consider market data
        market_volatility = 0.5
        if market_data and "volatility" in market_data:
            market_volatility = market_data["volatility"]

        # Decision logic
        if tournament_phase == "late" and competition_level > 0.7:
            return "immediate_submission"  # High competition, submit quickly
        elif difficulty > 0.8 and days_to_close > 7:
            return "extended_research"  # Complex question, take time
        elif market_volatility > 0.7:
            return "wait_for_stability"  # Volatile market, wait
        elif days_to_close <= 3:
            return "immediate_submission"  # Deadline pressure
        else:
            return "optimal_window"  # Standard timing

    def _calculate_optimal_submission_time(
        self,
        question: Question,
        timing_strategy: str,
        tournament_context: Dict[str, Any],
        market_data: Optional[Dict[str, Any]]
    ) -> datetime:
        """Calculate optimal submission time based on strategy."""
        current_time = datetime.utcnow()
        close_time = question.close_time

        if timing_strategy == "immediate_submission":
            return current_time + timedelta(hours=1)  # Submit soon
        elif timing_strategy == "extended_research":
            # Submit with 25% of time remaining
            time_remaining = close_time - current_time
            return close_time - timedelta(seconds=time_remaining.total_seconds() * 0.25)
        elif timing_strategy == "wait_for_stability":
            # Wait for market to stabilize, but not too late
            return current_time + timedelta(hours=12)
        elif timing_strategy == "optimal_window":
            # Submit in the middle 50% of available time
            time_remaining = close_time - current_time
            return current_time + timedelta(seconds=time_remaining.total_seconds() * 0.5)
        else:
            return current_time + timedelta(hours=6)  # Default

    def _assess_timing_confidence(
        self,
        question: Question,
        timing_strategy: str,
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Assess confidence in timing recommendation."""
        base_confidence = 0.6

        # Higher confidence for simpler strategies
        if timing_strategy == "immediate_submission":
            base_confidence += 0.2
        elif timing_strategy == "extended_research":
            base_confidence += 0.1

        # Adjust based on market data availability
        if market_data:
            base_confidence += 0.1

        # Adjust based on question characteristics
        difficulty = question.calculate_difficulty_score()
        if difficulty < 0.5:  # Easier questions have more predictable timing
            base_confidence += 0.1

        return min(0.9, base_confidence)

    def _identify_timing_risk_factors(
        self,
        question: Question,
        timing_strategy: str,
        tournament_context: Dict[str, Any]
    ) -> List[str]:
        """Identify risk factors for timing strategy."""
        risk_factors = []

        days_to_close = question.days_until_close()

        if days_to_close <= 1:
            risk_factors.append("Very close to deadline")

        if timing_strategy == "wait_for_stability":
            risk_factors.append("Market volatility may persist")

        if timing_strategy == "extended_research":
            risk_factors.append("May run out of time for thorough research")

        competition_level = tournament_context.get("competition_level", 0.5)
        if competition_level > 0.8:
            risk_factors.append("High competition may require faster submission")

        if question.calculate_difficulty_score() > 0.8:
            risk_factors.append("High question complexity increases timing uncertainty")

        return risk_factors

    def _estimate_timing_score_improvement(
        self,
        question: Question,
        timing_strategy: str,
        current_forecast: Optional[Forecast]
    ) -> float:
        """Estimate score improvement from optimal timing."""
        base_improvement = 0.02  # Small but meaningful improvement

        # Higher improvement for better strategies
        if timing_strategy == "optimal_window":
            base_improvement += 0.01
        elif timing_strategy == "extended_research":
            base_improvement += 0.03  # More research time helps

        # Adjust based on question difficulty
        difficulty = question.calculate_difficulty_score()
        if difficulty > 0.7:
            base_improvement += 0.02  # More benefit for difficult questions

        # Adjust based on current forecast quality
        if current_forecast:
            confidence = current_forecast.confidence
            if confidence < 0.6:  # Low confidence forecast
                base_improvement += 0.02  # More benefit from better timing

        return base_improvement

    def _calculate_current_expected_score(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        tournament_strategy: TournamentStrategy
    ) -> float:
        """Calculate current expected tournament score."""
        if not forecasts:
            return 0.5

        # Simple expected score based on confidence and sharpness
        total_score = 0.0
        for forecast in forecasts:
            confidence = forecast.confidence
            sharpness = abs(forecast.prediction - 0.5) * 2

            # Expected score combines confidence and sharpness
            expected_score = (confidence * 0.7) + (sharpness * 0.3)
            total_score += expected_score

        return total_score / len(forecasts)

    def _calculate_optimized_expected_score(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        tournament_strategy: TournamentStrategy,
        optimization_recommendations: List[OptimizationRecommendation]
    ) -> float:
        """Calculate expected score with optimizations applied."""
        current_score = self._calculate_current_expected_score(
            forecasts, questions, tournament_strategy
        )

        # Apply improvements from recommendations
        total_improvement = sum(rec.expected_improvement for rec in optimization_recommendations)

        # Apply diminishing returns
        improvement_factor = 1.0 - math.exp(-total_improvement * 2)

        return min(1.0, current_score + (total_improvement * improvement_factor))

    def _calculate_category_expected_scores(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        tournament_strategy: TournamentStrategy
    ) -> Dict[str, float]:
        """Calculate expected scores by category."""
        category_scores = defaultdict(list)

        for forecast in forecasts:
            question = next((q for q in questions if q.id == forecast.question_id), None)
            if question:
                category = question.categorize_question()
                confidence = forecast.confidence
                sharpness = abs(forecast.prediction - 0.5) * 2

                expected_score = (confidence * 0.7) + (sharpness * 0.3)
                category_scores[category.value].append(expected_score)

        # Calculate averages
        return {
            category: statistics.mean(scores)
            for category, scores in category_scores.items()
            if scores
        }

    def _calculate_score_confidence_intervals(
        self,
        forecasts: List[Forecast],
        questions: List[Question],
        tournament_strategy: TournamentStrategy
    ) -> Dict[str, float]:
        """Calculate confidence intervals for score estimates."""
        current_score = self._calculate_current_expected_score(
            forecasts, questions, tournament_strategy
        )

        # Simple confidence interval based on forecast variance
        if len(forecasts) > 1:
            forecast_scores = []
            for forecast in forecasts:
                confidence = forecast.confidence
                sharpness = abs(forecast.prediction - 0.5) * 2
                score = (confidence * 0.7) + (sharpness * 0.3)
                forecast_scores.append(score)

            score_std = statistics.stdev(forecast_scores)
            margin = 1.96 * score_std / math.sqrt(len(forecasts))  # 95% CI

            return {
                "lower": max(0.0, current_score - margin),
                "upper": min(1.0, current_score + margin)
            }

        return {"lower": current_score * 0.9, "upper": current_score * 1.1}

    def _calculate_improvement_confidence(
        self,
        optimization_recommendations: List[OptimizationRecommendation]
    ) -> float:
        """Calculate confidence in improvement estimates."""
        if not optimization_recommendations:
            return 0.0

        # Average confidence weighted by expected improvement
        total_weighted_confidence = 0.0
        total_weight = 0.0

        for rec in optimization_recommendations:
            weight = rec.expected_improvement
            total_weighted_confidence += rec.confidence * weight
            total_weight += weight

        if total_weight > 0:
            return total_weighted_confidence / total_weight

        return statistics.mean(rec.confidence for rec in optimization_recommendations)

    def _calculate_risk_adjusted_score(
        self,
        base_score: float,
        forecasts: List[Forecast],
        tournament_strategy: TournamentStrategy
    ) -> float:
        """Calculate risk-adjusted score."""
        if not forecasts:
            return base_score

        # Calculate risk penalty
        prediction_variance = statistics.variance([f.prediction for f in forecasts]) if len(forecasts) > 1 else 0.0
        risk_penalty = prediction_variance * 0.1  # Small penalty for high variance

        # Adjust based on risk profile
        if tournament_strategy.risk_profile == RiskProfile.CONSERVATIVE:
            risk_penalty *= 0.5  # Lower penalty for conservative strategy
        elif tournament_strategy.risk_profile == RiskProfile.AGGRESSIVE:
            risk_penalty *= 1.5  # Higher penalty for aggressive strategy

        return max(0.0, base_score - risk_penalty)

    def _generate_calibration_recommendations(
        self,
        current_performance: ScoringMetrics,
        tournament_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations to improve calibration."""
        recommendations = []

        calibration_gap = 0.9 - current_performance.calibration_score
        if calibration_gap > 0.1:
            recommendations.append(OptimizationRecommendation(
                recommendation_type="calibration_improvement",
                category=None,
                current_value=current_performance.calibration_score,
                recommended_value=0.9,
                expected_improvement=calibration_gap * 0.5,
                confidence=0.7,
                rationale="Poor calibration detected - implement confidence adjustment mechanisms",
                implementation_priority="high",
                risk_level="low"
            ))

        return recommendations

    def _generate_sharpness_recommendations(
        self,
        current_performance: ScoringMetrics,
        tournament_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations to improve sharpness."""
        recommendations = []

        if current_performance.sharpness_score < 0.4:
            recommendations.append(OptimizationRecommendation(
                recommendation_type="sharpness_improvement",
                category=None,
                current_value=current_performance.sharpness_score,
                recommended_value=0.5,
                expected_improvement=0.03,
                confidence=0.6,
                rationale="Low sharpness - predictions too close to 0.5, increase confidence in strong predictions",
                implementation_priority="medium",
                risk_level="medium"
            ))

        return recommendations

    def _generate_category_performance_recommendations(
        self,
        current_performance: ScoringMetrics,
        tournament_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate category-specific performance recommendations."""
        recommendations = []

        for category, performance in current_performance.category_performance.items():
            if performance < 0.6:  # Poor performance threshold
                recommendations.append(OptimizationRecommendation(
                    recommendation_type="category_improvement",
                    category=category,
                    current_value=performance,
                    recommended_value=0.7,
                    expected_improvement=0.05,
                    confidence=0.6,
                    rationale=f"Poor performance in {category.value} category - increase research depth or reduce focus",
                    implementation_priority="medium",
                    risk_level="low"
                ))

        return recommendations

    def _generate_competitive_recommendations(
        self,
        current_performance: ScoringMetrics,
        competitor_analysis: Dict[str, Any],
        tournament_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate competitive positioning recommendations."""
        recommendations = []

        if current_performance.tournament_rank and current_performance.tournament_rank > 10:
            recommendations.append(OptimizationRecommendation(
                recommendation_type="competitive_positioning",
                category=None,
                current_value=float(current_performance.tournament_rank),
                recommended_value=5.0,
                expected_improvement=0.1,
                confidence=0.5,
                rationale="Low tournament ranking - implement more aggressive strategy",
                implementation_priority="high",
                risk_level="high"
            ))

        return recommendations

    def _generate_phase_specific_recommendations(
        self,
        current_performance: ScoringMetrics,
        tournament_phase: str,
        tournament_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate tournament phase-specific recommendations."""
        recommendations = []

        if tournament_phase == "late":
            recommendations.append(OptimizationRecommendation(
                recommendation_type="late_phase_optimization",
                category=None,
                current_value=0.5,
                recommended_value=0.7,
                expected_improvement=0.08,
                confidence=0.6,
                rationale="Late tournament phase - increase risk tolerance and submission speed",
                implementation_priority="high",
                risk_level="medium"
            ))
        elif tournament_phase == "early":
            recommendations.append(OptimizationRecommendation(
                recommendation_type="early_phase_optimization",
                category=None,
                current_value=0.5,
                recommended_value=0.4,
                expected_improvement=0.04,
                confidence=0.7,
                rationale="Early tournament phase - focus on calibration and research quality",
                implementation_priority="medium",
                risk_level="low"
            ))

        return recommendations
