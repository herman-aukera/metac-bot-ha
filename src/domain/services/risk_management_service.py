"""Risk management service for tournament forecasting."""

import statistics
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

from ..entities.forecast import Forecast
from ..entities.question import Question
from ..value_objects.tournament_strategy import (
    RiskProfile,
    TournamentStrategy,
)

logger = structlog.get_logger(__name__)


class RiskLevel(Enum):
    """Risk levels for forecasting decisions."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RiskType(Enum):
    """Types of risks in forecasting."""

    CALIBRATION_DRIFT = "calibration_drift"
    HIGH_UNCERTAINTY = "high_uncertainty"
    TIME_PRESSURE = "time_pressure"
    INSUFFICIENT_RESEARCH = "insufficient_research"
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement"
    COMPETITIVE_PRESSURE = "competitive_pressure"
    OVERCONFIDENCE = "overconfidence"
    CATEGORY_UNFAMILIARITY = "category_unfamiliarity"


class RiskManagementService:
    """
    Domain service for managing forecasting risks in tournament settings.

    Assesses various risk factors, provides risk mitigation strategies,
    and helps optimize risk-adjusted decision making.
    """

    def __init__(self):
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: 0.2,
            RiskLevel.LOW: 0.4,
            RiskLevel.MODERATE: 0.6,
            RiskLevel.HIGH: 0.8,
            RiskLevel.VERY_HIGH: 1.0,
        }
        self.risk_history: List[Dict[str, Any]] = []

    def assess_forecast_risk(
        self,
        forecast: Forecast,
        question: Question,
        tournament_strategy: Optional[TournamentStrategy] = None,
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive risk assessment for a forecast.

        Args:
            forecast: The forecast to assess
            question: The question being forecasted
            tournament_strategy: Current tournament strategy
            tournament_context: Additional tournament context

        Returns:
            Comprehensive risk assessment
        """
        risk_factors = {}

        # Assess individual risk factors
        risk_factors[RiskType.HIGH_UNCERTAINTY] = self._assess_uncertainty_risk(
            forecast
        )
        risk_factors[RiskType.TIME_PRESSURE] = self._assess_time_pressure_risk(
            question, tournament_context
        )
        risk_factors[RiskType.INSUFFICIENT_RESEARCH] = (
            self._assess_research_quality_risk(forecast)
        )
        risk_factors[RiskType.ENSEMBLE_DISAGREEMENT] = (
            self._assess_ensemble_disagreement_risk(forecast)
        )
        risk_factors[RiskType.OVERCONFIDENCE] = self._assess_overconfidence_risk(
            forecast
        )
        risk_factors[RiskType.CATEGORY_UNFAMILIARITY] = self._assess_category_risk(
            question, tournament_strategy
        )

        if tournament_context:
            risk_factors[RiskType.COMPETITIVE_PRESSURE] = (
                self._assess_competitive_pressure_risk(tournament_context)
            )

        # Calculate overall risk score
        risk_scores = list(risk_factors.values())
        overall_risk_score = statistics.mean(risk_scores)
        overall_risk_level = self._score_to_risk_level(overall_risk_score)

        # Generate risk mitigation recommendations
        mitigation_strategies = self._generate_mitigation_strategies(
            risk_factors, tournament_strategy
        )

        # Determine submission recommendation
        submission_recommendation = self._get_submission_recommendation(
            overall_risk_score, risk_factors, tournament_strategy
        )

        risk_assessment = {
            "overall_risk_score": overall_risk_score,
            "overall_risk_level": overall_risk_level,
            "risk_factors": {
                risk_type.value: score for risk_type, score in risk_factors.items()
            },
            "highest_risk_factors": self._get_highest_risk_factors(risk_factors),
            "mitigation_strategies": mitigation_strategies,
            "submission_recommendation": submission_recommendation,
            "confidence_adjustment": self._calculate_confidence_adjustment(
                risk_factors
            ),
            "timestamp": datetime.utcnow(),
        }

        # Store in risk history
        self.risk_history.append(
            {
                "forecast_id": str(forecast.id),
                "question_id": str(forecast.question_id),
                "assessment": risk_assessment,
                "timestamp": datetime.utcnow(),
            }
        )

        logger.info(
            "Completed risk assessment",
            forecast_id=str(forecast.id),
            overall_risk=overall_risk_level.value,
            highest_risks=[
                rf.value for rf in self._get_highest_risk_factors(risk_factors)
            ],
        )

        return risk_assessment

    def _assess_uncertainty_risk(self, forecast: Forecast) -> float:
        """Assess risk from high uncertainty in predictions."""
        # Check prediction variance
        variance = forecast.calculate_prediction_variance()
        variance_risk = min(1.0, variance * 10)  # Scale variance to risk

        # Check confidence levels
        if forecast.predictions:
            confidence_scores = [p.get_confidence_score() for p in forecast.predictions]
            avg_confidence = statistics.mean(confidence_scores)
            confidence_risk = 1.0 - avg_confidence
        else:
            confidence_risk = 0.8  # High risk with no predictions

        # Check uncertainty quantification
        uncertainty_risk = 0.5  # Default
        if forecast.final_prediction and hasattr(
            forecast.final_prediction, "uncertainty_quantification"
        ):
            uncertainty_data = forecast.final_prediction.uncertainty_quantification
            if uncertainty_data:
                # Higher uncertainty sources = higher risk
                uncertainty_risk = min(1.0, len(uncertainty_data) * 0.1)

        return statistics.mean([variance_risk, confidence_risk, uncertainty_risk])

    def _assess_time_pressure_risk(
        self, question: Question, tournament_context: Optional[Dict[str, Any]]
    ) -> float:
        """Assess risk from time pressure."""
        if not question.close_time:
            return 0.5  # Default moderate risk

        now = datetime.utcnow()
        close_time = question.close_time

        # Handle timezone-aware vs naive datetime comparison
        if close_time.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elif close_time.tzinfo is None and now.tzinfo is not None:
            close_time = close_time.replace(tzinfo=timezone.utc)

        hours_to_close = (close_time - now).total_seconds() / 3600

        if hours_to_close < 2:
            return 0.95  # Very high risk
        elif hours_to_close < 12:
            return 0.8  # High risk
        elif hours_to_close < 48:
            return 0.4  # Moderate risk
        elif hours_to_close < 168:  # 1 week
            return 0.2  # Low risk
        else:
            return 0.1  # Very low risk

    def _assess_research_quality_risk(self, forecast: Forecast) -> float:
        """Assess risk from insufficient or low-quality research."""
        if not forecast.research_reports:
            return 0.9  # Very high risk with no research

        # Assess research quality
        quality_scores = []
        for report in forecast.research_reports:
            if report.quality.value == "high":
                quality_scores.append(0.8)
            elif report.quality.value == "medium":
                quality_scores.append(0.5)
            else:
                quality_scores.append(0.2)

        avg_quality = statistics.mean(quality_scores)
        quality_risk = 1.0 - avg_quality

        # Assess research quantity
        report_count = len(forecast.research_reports)
        if report_count >= 3:
            quantity_risk = 0.1
        elif report_count >= 2:
            quantity_risk = 0.3
        else:
            quantity_risk = 0.7

        # Assess source diversity
        source_count = sum(len(report.sources) for report in forecast.research_reports)
        if source_count >= 10:
            diversity_risk = 0.1
        elif source_count >= 5:
            diversity_risk = 0.3
        else:
            diversity_risk = 0.6

        return statistics.mean([quality_risk, quantity_risk, diversity_risk])

    def _assess_ensemble_disagreement_risk(self, forecast: Forecast) -> float:
        """Assess risk from high disagreement among ensemble predictions."""
        if len(forecast.predictions) < 2:
            return 0.5  # Moderate risk with single prediction

        variance = forecast.calculate_prediction_variance()

        if variance > 0.15:
            return 0.9  # Very high disagreement
        elif variance > 0.1:
            return 0.7  # High disagreement
        elif variance > 0.05:
            return 0.4  # Moderate disagreement
        else:
            return 0.2  # Low disagreement

    def _assess_overconfidence_risk(self, forecast: Forecast) -> float:
        """Assess risk from overconfidence in predictions."""
        if not forecast.predictions:
            return 0.5

        confidence_scores = [p.get_confidence_score() for p in forecast.predictions]
        avg_confidence = statistics.mean(confidence_scores)

        # Check for overconfidence patterns
        high_confidence_count = sum(1 for score in confidence_scores if score > 0.8)
        high_confidence_ratio = high_confidence_count / len(confidence_scores)

        # Risk increases with very high confidence, especially if uniform
        if avg_confidence > 0.9:
            base_risk = 0.8
        elif avg_confidence > 0.8:
            base_risk = 0.6
        elif avg_confidence > 0.7:
            base_risk = 0.4
        else:
            base_risk = 0.2

        # Additional risk if most predictions are high confidence
        uniformity_risk = high_confidence_ratio * 0.3

        return min(1.0, base_risk + uniformity_risk)

    def _assess_category_risk(
        self, question: Question, tournament_strategy: Optional[TournamentStrategy]
    ) -> float:
        """Assess risk from unfamiliarity with question category."""
        category = question.categorize_question()

        if not tournament_strategy:
            return 0.5  # Default moderate risk

        # Check category specialization
        specialization = tournament_strategy.category_specializations.get(category, 0.5)

        # Lower specialization = higher risk
        return 1.0 - specialization

    def _assess_competitive_pressure_risk(
        self, tournament_context: Dict[str, Any]
    ) -> float:
        """Assess risk from competitive pressure in tournament."""
        base_risk = 0.3

        # High competition increases risk
        competition_level = tournament_context.get("competition_level", 0.5)
        competition_risk = competition_level * 0.4

        # Late tournament phase increases pressure
        phase = tournament_context.get("phase", "early")
        if phase == "final":
            phase_risk = 0.3
        elif phase == "late":
            phase_risk = 0.2
        else:
            phase_risk = 0.0

        return min(1.0, base_risk + competition_risk + phase_risk)

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert risk score to risk level enum."""
        if score <= self.risk_thresholds[RiskLevel.VERY_LOW]:
            return RiskLevel.VERY_LOW
        elif score <= self.risk_thresholds[RiskLevel.LOW]:
            return RiskLevel.LOW
        elif score <= self.risk_thresholds[RiskLevel.MODERATE]:
            return RiskLevel.MODERATE
        elif score <= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _get_highest_risk_factors(
        self, risk_factors: Dict[RiskType, float], top_n: int = 3
    ) -> List[RiskType]:
        """Get the highest risk factors."""
        sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
        return [risk_type for risk_type, _ in sorted_risks[:top_n]]

    def _generate_mitigation_strategies(
        self,
        risk_factors: Dict[RiskType, float],
        tournament_strategy: Optional[TournamentStrategy],
    ) -> List[str]:
        """Generate risk mitigation strategies."""
        strategies = []

        # Address highest risk factors
        for risk_type, score in risk_factors.items():
            if score > 0.6:  # High risk threshold
                strategies.extend(self._get_mitigation_for_risk_type(risk_type, score))

        # Strategy-specific mitigations
        if (
            tournament_strategy
            and tournament_strategy.risk_profile == RiskProfile.CONSERVATIVE
        ):
            strategies.append("Apply conservative confidence adjustments")
            strategies.append("Require additional validation before submission")

        return list(set(strategies))  # Remove duplicates

    def _get_mitigation_for_risk_type(
        self, risk_type: RiskType, score: float
    ) -> List[str]:
        """Get specific mitigation strategies for a risk type."""
        strategies = []

        if risk_type == RiskType.HIGH_UNCERTAINTY:
            strategies.extend(
                [
                    "Gather additional research from diverse sources",
                    "Increase ensemble diversity",
                    "Apply uncertainty-adjusted confidence scaling",
                ]
            )

        elif risk_type == RiskType.TIME_PRESSURE:
            strategies.extend(
                [
                    "Prioritize high-confidence predictions",
                    "Use rapid validation techniques",
                    "Consider abstaining from low-confidence predictions",
                ]
            )

        elif risk_type == RiskType.INSUFFICIENT_RESEARCH:
            strategies.extend(
                [
                    "Conduct additional targeted research",
                    "Seek expert opinions",
                    "Validate findings with multiple sources",
                ]
            )

        elif risk_type == RiskType.ENSEMBLE_DISAGREEMENT:
            strategies.extend(
                [
                    "Investigate sources of disagreement",
                    "Add more diverse reasoning approaches",
                    "Consider median aggregation instead of mean",
                ]
            )

        elif risk_type == RiskType.OVERCONFIDENCE:
            strategies.extend(
                [
                    "Apply calibration-based confidence adjustment",
                    "Seek disconfirming evidence",
                    "Use conservative confidence scaling",
                ]
            )

        elif risk_type == RiskType.CATEGORY_UNFAMILIARITY:
            strategies.extend(
                [
                    "Seek domain expert consultation",
                    "Increase research depth for this category",
                    "Apply category-specific confidence penalties",
                ]
            )

        elif risk_type == RiskType.COMPETITIVE_PRESSURE:
            strategies.extend(
                [
                    "Focus on high-confidence predictions",
                    "Avoid rushed decisions",
                    "Maintain systematic approach despite pressure",
                ]
            )

        return strategies

    def _get_submission_recommendation(
        self,
        overall_risk_score: float,
        risk_factors: Dict[RiskType, float],
        tournament_strategy: Optional[TournamentStrategy],
    ) -> Dict[str, Any]:
        """Get recommendation for forecast submission."""

        # Base recommendation on overall risk
        if overall_risk_score > 0.8:
            base_recommendation = "do_not_submit"
            reason = "Overall risk too high"
        elif overall_risk_score > 0.6:
            base_recommendation = "submit_with_caution"
            reason = "Moderate to high risk"
        else:
            base_recommendation = "submit"
            reason = "Acceptable risk level"

        # Adjust based on strategy
        if tournament_strategy:
            risk_tolerance = self._get_risk_tolerance(tournament_strategy.risk_profile)

            if overall_risk_score > risk_tolerance:
                base_recommendation = "do_not_submit"
                reason = (
                    f"Risk exceeds {tournament_strategy.risk_profile.value} tolerance"
                )

        # Special cases for specific high-risk factors
        if risk_factors.get(RiskType.TIME_PRESSURE, 0) > 0.9:
            if base_recommendation == "do_not_submit":
                base_recommendation = "submit_with_caution"
                reason = "High time pressure overrides risk concerns"

        return {
            "recommendation": base_recommendation,
            "reason": reason,
            "confidence": (
                "high"
                if overall_risk_score < 0.3 or overall_risk_score > 0.8
                else "medium"
            ),
        }

    def _calculate_confidence_adjustment(
        self, risk_factors: Dict[RiskType, float]
    ) -> float:
        """Calculate confidence adjustment based on risk factors."""
        base_adjustment = 0.0

        # Overconfidence risk adjustment
        overconfidence_risk = risk_factors.get(RiskType.OVERCONFIDENCE, 0)
        if overconfidence_risk > 0.6:
            base_adjustment -= 0.1

        # High uncertainty adjustment
        uncertainty_risk = risk_factors.get(RiskType.HIGH_UNCERTAINTY, 0)
        if uncertainty_risk > 0.6:
            base_adjustment -= 0.05

        # Category unfamiliarity adjustment
        category_risk = risk_factors.get(RiskType.CATEGORY_UNFAMILIARITY, 0)
        if category_risk > 0.6:
            base_adjustment -= 0.05

        return max(-0.2, min(0.1, base_adjustment))  # Limit adjustment range

    def _get_risk_tolerance(self, risk_profile: RiskProfile) -> float:
        """Get risk tolerance threshold for a risk profile."""
        tolerance_map = {
            RiskProfile.CONSERVATIVE: 0.4,
            RiskProfile.MODERATE: 0.6,
            RiskProfile.AGGRESSIVE: 0.8,
            RiskProfile.ADAPTIVE: 0.6,  # Default to moderate
        }
        return tolerance_map.get(risk_profile, 0.6)

    def analyze_risk_trends(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze risk trends over time."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        recent_assessments = [
            record for record in self.risk_history if record["timestamp"] >= cutoff_date
        ]

        if len(recent_assessments) < 5:
            return {
                "trend_analysis": "insufficient_data",
                "sample_count": len(recent_assessments),
            }

        # Analyze overall risk trends
        risk_scores = [
            record["assessment"]["overall_risk_score"] for record in recent_assessments
        ]

        # Split into early and recent periods
        split_point = len(risk_scores) // 2
        early_scores = risk_scores[:split_point]
        recent_scores = risk_scores[split_point:]

        early_avg = statistics.mean(early_scores)
        recent_avg = statistics.mean(recent_scores)

        trend = "improving" if recent_avg < early_avg else "deteriorating"

        # Analyze risk factor frequencies
        risk_factor_counts = {}
        for record in recent_assessments:
            for risk_factor in record["assessment"]["highest_risk_factors"]:
                risk_factor_counts[risk_factor] = (
                    risk_factor_counts.get(risk_factor, 0) + 1
                )

        return {
            "trend_analysis": trend,
            "early_avg_risk": early_avg,
            "recent_avg_risk": recent_avg,
            "risk_change": recent_avg - early_avg,
            "sample_count": len(recent_assessments),
            "most_common_risks": sorted(
                risk_factor_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "risk_distribution": {
                "low_risk_fraction": sum(1 for score in risk_scores if score < 0.4)
                / len(risk_scores),
                "high_risk_fraction": sum(1 for score in risk_scores if score > 0.7)
                / len(risk_scores),
            },
        }

    def get_risk_management_summary(self) -> Dict[str, Any]:
        """Get summary of risk management service state."""
        return {
            "total_assessments": len(self.risk_history),
            "recent_trends": (
                self.analyze_risk_trends() if len(self.risk_history) >= 5 else None
            ),
            "risk_thresholds": {
                level.value: threshold
                for level, threshold in self.risk_thresholds.items()
            },
            "supported_risk_types": [risk_type.value for risk_type in RiskType],
        }
