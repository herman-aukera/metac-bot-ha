"""Conservative strategy engine for risk management and tournament optimization."""

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..entities.forecast import Forecast
from ..entities.prediction import Prediction
from .calibration_service import CalibrationMetrics, CalibrationTracker
from .uncertainty_quantifier import UncertaintyAssessment, UncertaintyQuantifier


class RiskLevel(Enum):
    """Risk levels for predictions."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ConservativeAction(Enum):
    """Conservative actions that can be taken."""

    SUBMIT = "submit"
    ABSTAIN = "abstain"
    DEFER = "defer"
    REDUCE_CONFIDENCE = "reduce_confidence"
    REQUEST_RESEARCH = "request_research"


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for a prediction."""

    overall_risk: RiskLevel
    risk_factors: Dict[str, float]
    uncertainty_score: float
    calibration_risk: float
    tournament_risk: float
    time_pressure_risk: float

    # Risk mitigation recommendations
    recommended_action: ConservativeAction
    confidence_adjustment: Optional[float]
    reasoning: str

    # Tournament-specific considerations
    abstention_penalty: float
    scoring_opportunity_cost: float
    competitive_impact: float

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of risk assessment."""
        return {
            "overall_risk": self.overall_risk.value,
            "uncertainty_score": self.uncertainty_score,
            "calibration_risk": self.calibration_risk,
            "tournament_risk": self.tournament_risk,
            "recommended_action": self.recommended_action.value,
            "confidence_adjustment": self.confidence_adjustment,
            "reasoning": self.reasoning,
            "abstention_penalty": self.abstention_penalty,
        }


@dataclass
class ConservativeStrategyConfig:
    """Configuration for conservative strategy engine."""

    # Risk thresholds
    high_uncertainty_threshold: float = 0.7
    poor_calibration_threshold: float = 0.15
    time_pressure_threshold_hours: float = 6.0

    # Conservative adjustments
    confidence_reduction_factor: float = 0.8
    uncertainty_penalty_factor: float = 0.5
    calibration_penalty_factor: float = 0.3

    # Tournament considerations
    abstention_penalty_weight: float = 0.2
    scoring_opportunity_weight: float = 0.3
    competitive_pressure_weight: float = 0.1

    # Abstention thresholds
    abstention_uncertainty_threshold: float = 0.8
    abstention_calibration_threshold: float = 0.2
    abstention_risk_threshold: float = 0.75

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.confidence_reduction_factor <= 1.0:
            raise ValueError("Confidence reduction factor must be between 0 and 1")
        if not 0.0 <= self.high_uncertainty_threshold <= 1.0:
            raise ValueError("High uncertainty threshold must be between 0 and 1")


class ConservativeStrategyEngine:
    """Engine for implementing conservative strategies and risk management."""

    def __init__(
        self,
        config: Optional[ConservativeStrategyConfig] = None,
        uncertainty_quantifier: Optional[UncertaintyQuantifier] = None,
        calibration_tracker: Optional[CalibrationTracker] = None,
    ):
        """Initialize conservative strategy engine."""
        self.config = config or ConservativeStrategyConfig()
        self.config.validate_config()

        self.uncertainty_quantifier = uncertainty_quantifier or UncertaintyQuantifier()
        self.calibration_tracker = calibration_tracker or CalibrationTracker()

        # Historical performance tracking
        self.strategy_performance_history: List[Dict[str, Any]] = []

        # Tournament context
        self.current_tournament_context: Optional[Dict[str, Any]] = None

    def assess_prediction_risk(
        self,
        prediction: Prediction,
        uncertainty_assessment: Optional[UncertaintyAssessment] = None,
        calibration_metrics: Optional[CalibrationMetrics] = None,
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """Assess risk for a prediction and recommend conservative actions."""
        # Get uncertainty assessment if not provided
        if uncertainty_assessment is None:
            uncertainty_assessment = (
                self.uncertainty_quantifier.assess_prediction_uncertainty(prediction)
            )

        # Calculate risk factors
        risk_factors = self._calculate_risk_factors(
            prediction, uncertainty_assessment, calibration_metrics, tournament_context
        )

        # Determine overall risk level
        overall_risk = self._determine_overall_risk_level(risk_factors)

        # Calculate tournament-specific risks
        tournament_risk = self._calculate_tournament_risk(
            prediction, tournament_context
        )
        time_pressure_risk = self._calculate_time_pressure_risk(tournament_context)

        # Determine recommended action
        recommended_action, confidence_adjustment = self._determine_conservative_action(
            prediction, risk_factors, overall_risk, tournament_context
        )

        # Calculate tournament penalties and costs
        abstention_penalty = self._calculate_abstention_penalty(tournament_context)
        scoring_opportunity_cost = self._calculate_scoring_opportunity_cost(
            prediction, tournament_context
        )
        competitive_impact = self._calculate_competitive_impact(
            recommended_action, tournament_context
        )

        # Generate reasoning
        reasoning = self._generate_risk_reasoning(
            risk_factors, overall_risk, recommended_action, tournament_context
        )

        return RiskAssessment(
            overall_risk=overall_risk,
            risk_factors=risk_factors,
            uncertainty_score=uncertainty_assessment.total_uncertainty,
            calibration_risk=risk_factors.get("calibration", 0.0),
            tournament_risk=tournament_risk,
            time_pressure_risk=time_pressure_risk,
            recommended_action=recommended_action,
            confidence_adjustment=confidence_adjustment,
            reasoning=reasoning,
            abstention_penalty=abstention_penalty,
            scoring_opportunity_cost=scoring_opportunity_cost,
            competitive_impact=competitive_impact,
        )

    def apply_conservative_strategy(
        self, forecast: Forecast, tournament_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Forecast, Dict[str, Any]]:
        """Apply conservative strategy to a forecast."""
        # Assess risk for the forecast
        uncertainty_assessment = (
            self.uncertainty_quantifier.assess_forecast_uncertainty(forecast)
        )

        # Get calibration metrics if available
        calibration_metrics = None
        if self.calibration_tracker.calibration_history:
            calibration_metrics = self.calibration_tracker.calibration_history[-1]

        risk_assessment = self.assess_prediction_risk(
            forecast.final_prediction,
            uncertainty_assessment,
            calibration_metrics,
            tournament_context,
        )

        # Apply conservative adjustments based on risk assessment
        adjusted_forecast = self._apply_risk_adjustments(forecast, risk_assessment)

        # Generate strategy report
        strategy_report = self._generate_strategy_report(
            forecast, adjusted_forecast, risk_assessment, tournament_context
        )

        # Track strategy performance
        self._track_strategy_application(
            forecast, adjusted_forecast, risk_assessment, tournament_context
        )

        return adjusted_forecast, strategy_report

    def should_abstain_from_prediction(
        self,
        prediction: Prediction,
        uncertainty_assessment: Optional[UncertaintyAssessment] = None,
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Determine if prediction should be abstained based on conservative strategy."""
        risk_assessment = self.assess_prediction_risk(
            prediction, uncertainty_assessment, tournament_context=tournament_context
        )

        # Check abstention criteria
        should_abstain = (
            risk_assessment.overall_risk in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
            or risk_assessment.uncertainty_score
            > self.config.abstention_uncertainty_threshold
            or risk_assessment.calibration_risk
            > self.config.abstention_calibration_threshold
            or risk_assessment.recommended_action == ConservativeAction.ABSTAIN
        )

        # Consider tournament penalties
        if tournament_context:
            abstention_penalty = risk_assessment.abstention_penalty
            scoring_opportunity = risk_assessment.scoring_opportunity_cost

            # Adjust abstention decision based on tournament context
            if abstention_penalty > scoring_opportunity * 1.5:
                should_abstain = False  # Penalty too high, submit anyway

        return {
            "should_abstain": should_abstain,
            "risk_assessment": risk_assessment,
            "abstention_reason": risk_assessment.reasoning if should_abstain else None,
            "tournament_considerations": {
                "abstention_penalty": risk_assessment.abstention_penalty,
                "scoring_opportunity_cost": risk_assessment.scoring_opportunity_cost,
                "competitive_impact": risk_assessment.competitive_impact,
            },
        }

    def optimize_tournament_scoring(
        self, predictions: List[Prediction], tournament_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize predictions for tournament scoring with risk management."""
        optimization_results = {
            "original_predictions": len(predictions),
            "submitted_predictions": 0,
            "abstained_predictions": 0,
            "deferred_predictions": 0,
            "confidence_adjustments": 0,
            "expected_score_impact": 0.0,
            "risk_mitigation_actions": [],
        }

        optimized_predictions = []

        for prediction in predictions:
            # Assess risk for each prediction
            risk_assessment = self.assess_prediction_risk(
                prediction, tournament_context=tournament_context
            )

            # Apply optimization based on risk assessment
            if risk_assessment.recommended_action == ConservativeAction.SUBMIT:
                optimized_predictions.append(prediction)
                optimization_results["submitted_predictions"] += 1

            elif risk_assessment.recommended_action == ConservativeAction.ABSTAIN:
                optimization_results["abstained_predictions"] += 1
                optimization_results["risk_mitigation_actions"].append(
                    {
                        "prediction_id": str(prediction.id),
                        "action": "abstain",
                        "reason": risk_assessment.reasoning,
                    }
                )

            elif risk_assessment.recommended_action == ConservativeAction.DEFER:
                optimization_results["deferred_predictions"] += 1
                optimization_results["risk_mitigation_actions"].append(
                    {
                        "prediction_id": str(prediction.id),
                        "action": "defer",
                        "reason": risk_assessment.reasoning,
                    }
                )

            elif (
                risk_assessment.recommended_action
                == ConservativeAction.REDUCE_CONFIDENCE
            ):
                # Apply confidence reduction
                if risk_assessment.confidence_adjustment:
                    adjusted_prediction = self._apply_confidence_adjustment(
                        prediction, risk_assessment.confidence_adjustment
                    )
                    optimized_predictions.append(adjusted_prediction)
                    optimization_results["confidence_adjustments"] += 1
                    optimization_results["risk_mitigation_actions"].append(
                        {
                            "prediction_id": str(prediction.id),
                            "action": "reduce_confidence",
                            "adjustment": risk_assessment.confidence_adjustment,
                            "reason": risk_assessment.reasoning,
                        }
                    )
                else:
                    optimized_predictions.append(prediction)
                    optimization_results["submitted_predictions"] += 1

            # Calculate expected score impact
            optimization_results["expected_score_impact"] += (
                self._calculate_score_impact(
                    prediction, risk_assessment, tournament_context
                )
            )

        optimization_results["optimized_predictions"] = optimized_predictions

        return optimization_results

    def update_strategy_based_on_performance(
        self, performance_data: Dict[str, float]
    ) -> None:
        """Update conservative strategy based on performance feedback."""
        # Analyze recent performance
        if len(self.strategy_performance_history) < 5:
            return  # Need more data

        recent_performance = self.strategy_performance_history[-10:]

        # Calculate performance metrics
        abstention_rate = statistics.mean(
            [p.get("abstained", 0) for p in recent_performance]
        )

        accuracy_when_submitted = (
            statistics.mean(
                [
                    p.get("accuracy", 0.5)
                    for p in recent_performance
                    if p.get("submitted", False)
                ]
            )
            if any(p.get("submitted", False) for p in recent_performance)
            else 0.5
        )

        # Adjust thresholds based on performance
        if abstention_rate > 0.3 and accuracy_when_submitted > 0.7:
            # Too conservative, reduce thresholds
            self.config.abstention_uncertainty_threshold = min(
                0.9, self.config.abstention_uncertainty_threshold * 1.1
            )
            self.config.abstention_risk_threshold = min(
                0.9, self.config.abstention_risk_threshold * 1.05
            )

        elif abstention_rate < 0.1 and accuracy_when_submitted < 0.6:
            # Not conservative enough, increase thresholds
            self.config.abstention_uncertainty_threshold = max(
                0.5, self.config.abstention_uncertainty_threshold * 0.9
            )
            self.config.abstention_risk_threshold = max(
                0.5, self.config.abstention_risk_threshold * 0.95
            )

    def get_conservative_strategy_report(
        self, time_window_days: int = 30
    ) -> Dict[str, Any]:
        """Generate comprehensive conservative strategy report."""
        if not self.strategy_performance_history:
            return {"error": "No strategy performance data available"}

        # Filter recent data
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
        recent_data = [
            entry
            for entry in self.strategy_performance_history
            if entry.get("timestamp", datetime.min) >= cutoff_date
        ]

        if not recent_data:
            return {"error": "No recent strategy data available"}

        # Calculate summary statistics
        summary = self._calculate_strategy_summary(recent_data)

        # Analyze risk patterns
        risk_analysis = self._analyze_risk_patterns(recent_data)

        # Tournament performance analysis
        tournament_analysis = self._analyze_tournament_performance(recent_data)

        # Generate recommendations
        recommendations = self._generate_strategy_recommendations(
            summary, risk_analysis, tournament_analysis
        )

        return {
            "summary": summary,
            "risk_analysis": risk_analysis,
            "tournament_analysis": tournament_analysis,
            "recommendations": recommendations,
            "current_config": {
                "abstention_uncertainty_threshold": self.config.abstention_uncertainty_threshold,
                "abstention_risk_threshold": self.config.abstention_risk_threshold,
                "confidence_reduction_factor": self.config.confidence_reduction_factor,
            },
            "report_timestamp": datetime.utcnow().isoformat(),
            "time_window_days": time_window_days,
        }

    def _calculate_risk_factors(
        self,
        prediction: Prediction,
        uncertainty_assessment: UncertaintyAssessment,
        calibration_metrics: Optional[CalibrationMetrics],
        tournament_context: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate various risk factors for a prediction."""
        risk_factors = {}

        # Uncertainty risk
        risk_factors["uncertainty"] = uncertainty_assessment.total_uncertainty

        # Calibration risk
        if calibration_metrics:
            risk_factors["calibration"] = calibration_metrics.calibration_error
        else:
            risk_factors["calibration"] = 0.1  # Default moderate risk

        # Confidence risk (overconfidence)
        predicted_confidence = prediction.get_confidence_score()
        assessed_confidence = uncertainty_assessment.confidence_level
        confidence_gap = abs(predicted_confidence - assessed_confidence)
        risk_factors["confidence_mismatch"] = confidence_gap

        # Evidence quality risk
        if prediction.evidence_strength:
            risk_factors["evidence_quality"] = 1.0 - prediction.evidence_strength
        else:
            risk_factors["evidence_quality"] = 0.5

        # Method risk (some methods are riskier)
        method_risks = {
            "chain_of_thought": 0.3,
            "tree_of_thought": 0.2,
            "react": 0.4,
            "auto_cot": 0.35,
            "ensemble": 0.15,
        }
        risk_factors["method"] = method_risks.get(prediction.method.value, 0.4)

        # Tournament-specific risks
        if tournament_context:
            risk_factors["tournament"] = self._calculate_tournament_risk(
                prediction, tournament_context
            )
        else:
            risk_factors["tournament"] = 0.3

        return risk_factors

    def _determine_overall_risk_level(
        self, risk_factors: Dict[str, float]
    ) -> RiskLevel:
        """Determine overall risk level from risk factors."""
        # Weighted average of risk factors
        weights = {
            "uncertainty": 0.3,
            "calibration": 0.25,
            "confidence_mismatch": 0.2,
            "evidence_quality": 0.15,
            "method": 0.1,
        }

        weighted_risk = sum(
            risk_factors.get(factor, 0.0) * weight for factor, weight in weights.items()
        )

        # Classify risk level
        if weighted_risk >= 0.8:
            return RiskLevel.VERY_HIGH
        elif weighted_risk >= 0.6:
            return RiskLevel.HIGH
        elif weighted_risk >= 0.4:
            return RiskLevel.MODERATE
        elif weighted_risk >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _calculate_tournament_risk(
        self, prediction: Prediction, tournament_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate tournament-specific risk factors."""
        if not tournament_context:
            return 0.3  # Default moderate risk

        risk = 0.0

        # Time pressure risk
        hours_to_deadline = tournament_context.get("hours_to_deadline", 24)
        if hours_to_deadline < self.config.time_pressure_threshold_hours:
            risk += 0.3

        # Competition intensity risk
        competition_level = tournament_context.get("competition_level", "medium")
        competition_risks = {"low": 0.1, "medium": 0.2, "high": 0.4, "very_high": 0.5}
        risk += competition_risks.get(competition_level, 0.2)

        # Question difficulty risk
        difficulty = tournament_context.get("question_difficulty", "medium")
        difficulty_risks = {"easy": 0.1, "medium": 0.2, "hard": 0.4, "very_hard": 0.5}
        risk += difficulty_risks.get(difficulty, 0.2)

        return min(1.0, risk)

    def _calculate_time_pressure_risk(
        self, tournament_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate risk from time pressure."""
        if not tournament_context:
            return 0.2

        hours_to_deadline = tournament_context.get("hours_to_deadline", 24)

        if hours_to_deadline < 1:
            return 0.9
        elif hours_to_deadline < 6:
            return 0.7
        elif hours_to_deadline < 24:
            return 0.4
        else:
            return 0.1

    def _determine_conservative_action(
        self,
        prediction: Prediction,
        risk_factors: Dict[str, float],
        overall_risk: RiskLevel,
        tournament_context: Optional[Dict[str, Any]],
    ) -> Tuple[ConservativeAction, Optional[float]]:
        """Determine conservative action and confidence adjustment."""
        # High-risk scenarios
        if overall_risk == RiskLevel.VERY_HIGH:
            return ConservativeAction.ABSTAIN, None

        if overall_risk == RiskLevel.HIGH:
            # Check if abstention penalty is too high
            if tournament_context:
                abstention_penalty = self._calculate_abstention_penalty(
                    tournament_context
                )
                if abstention_penalty > 0.3:
                    # Penalty too high, reduce confidence instead
                    return (
                        ConservativeAction.REDUCE_CONFIDENCE,
                        self.config.confidence_reduction_factor,
                    )
            return ConservativeAction.ABSTAIN, None

        # Moderate risk scenarios
        if overall_risk == RiskLevel.MODERATE:
            # Check specific risk factors
            if (
                risk_factors.get("uncertainty", 0)
                > self.config.high_uncertainty_threshold
            ):
                return ConservativeAction.REQUEST_RESEARCH, None
            elif risk_factors.get("confidence_mismatch", 0) > 0.3:
                return ConservativeAction.REDUCE_CONFIDENCE, 0.9
            else:
                return ConservativeAction.SUBMIT, None

        # Low risk scenarios
        return ConservativeAction.SUBMIT, None

    def _calculate_abstention_penalty(
        self, tournament_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate penalty for abstaining from prediction."""
        if not tournament_context:
            return 0.1  # Default low penalty

        # Base penalty from tournament rules
        base_penalty = tournament_context.get("abstention_penalty", 0.1)

        # Adjust based on tournament phase
        tournament_phase = tournament_context.get("phase", "middle")
        phase_multipliers = {"early": 0.8, "middle": 1.0, "late": 1.2, "final": 1.5}

        return base_penalty * phase_multipliers.get(tournament_phase, 1.0)

    def _calculate_scoring_opportunity_cost(
        self, prediction: Prediction, tournament_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate opportunity cost of not submitting prediction."""
        if not tournament_context:
            return 0.2  # Default moderate opportunity cost

        # Base scoring potential
        confidence = prediction.get_confidence_score()
        base_opportunity = confidence * 0.5  # Rough estimate

        # Adjust based on question importance
        question_weight = tournament_context.get("question_weight", 1.0)

        return base_opportunity * question_weight

    def _calculate_competitive_impact(
        self, action: ConservativeAction, tournament_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate competitive impact of the action."""
        if not tournament_context:
            return 0.0

        # Impact varies by action
        action_impacts = {
            ConservativeAction.SUBMIT: 0.0,
            ConservativeAction.ABSTAIN: -0.2,
            ConservativeAction.DEFER: -0.1,
            ConservativeAction.REDUCE_CONFIDENCE: -0.05,
            ConservativeAction.REQUEST_RESEARCH: 0.1,
        }

        base_impact = action_impacts.get(action, 0.0)

        # Adjust based on competition level
        competition_level = tournament_context.get("competition_level", "medium")
        competition_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
            "very_high": 2.0,
        }

        return base_impact * competition_multipliers.get(competition_level, 1.0)

    def _generate_risk_reasoning(
        self,
        risk_factors: Dict[str, float],
        overall_risk: RiskLevel,
        recommended_action: ConservativeAction,
        tournament_context: Optional[Dict[str, Any]],
    ) -> str:
        """Generate human-readable reasoning for risk assessment."""
        reasoning_parts = []

        # Overall risk assessment
        reasoning_parts.append(f"Overall risk level: {overall_risk.value}")

        # Dominant risk factors
        sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
        top_risks = sorted_risks[:2]

        if top_risks:
            risk_descriptions = {
                "uncertainty": "high prediction uncertainty",
                "calibration": "poor historical calibration",
                "confidence_mismatch": "confidence-uncertainty mismatch",
                "evidence_quality": "low evidence quality",
                "method": "prediction method limitations",
                "tournament": "tournament-specific risks",
            }

            risk_reasons = [
                f"{risk_descriptions.get(risk_name, risk_name)}: {risk_value:.2f}"
                for risk_name, risk_value in top_risks
                if risk_value > 0.3
            ]

            if risk_reasons:
                reasoning_parts.append(f"Primary concerns: {', '.join(risk_reasons)}")

        # Action reasoning
        action_reasons = {
            ConservativeAction.SUBMIT: "Risk acceptable for submission",
            ConservativeAction.ABSTAIN: "Risk too high, abstention recommended",
            ConservativeAction.DEFER: "Defer pending additional information",
            ConservativeAction.REDUCE_CONFIDENCE: "Reduce confidence due to uncertainty",
            ConservativeAction.REQUEST_RESEARCH: "Additional research needed",
        }

        reasoning_parts.append(
            f"Recommendation: {action_reasons.get(recommended_action, 'Unknown action')}"
        )

        # Tournament considerations
        if tournament_context:
            hours_to_deadline = tournament_context.get("hours_to_deadline", 24)
            if hours_to_deadline < 6:
                reasoning_parts.append("Time pressure factor considered")

        return ". ".join(reasoning_parts)

    def _apply_risk_adjustments(
        self, forecast: Forecast, risk_assessment: RiskAssessment
    ) -> Forecast:
        """Apply risk-based adjustments to forecast."""
        if risk_assessment.recommended_action == ConservativeAction.REDUCE_CONFIDENCE:
            if risk_assessment.confidence_adjustment:
                # Apply confidence adjustment to final prediction
                adjusted_prediction = self._apply_confidence_adjustment(
                    forecast.final_prediction, risk_assessment.confidence_adjustment
                )

                # Create new forecast with adjusted prediction
                adjusted_forecast = Forecast.create_new(
                    question_id=forecast.question_id,
                    research_reports=forecast.research_reports,
                    predictions=forecast.predictions + [adjusted_prediction],
                    final_prediction=adjusted_prediction,
                    reasoning_summary=f"Conservative adjustment applied: {forecast.reasoning_summary}",
                    ensemble_method=f"conservative_{forecast.ensemble_method}",
                    weight_distribution=forecast.weight_distribution,
                    consensus_strength=forecast.consensus_strength
                    * risk_assessment.confidence_adjustment,
                    metadata={
                        **forecast.metadata,
                        "conservative_adjustment_applied": True,
                        "original_confidence": forecast.confidence_score,
                        "risk_assessment": risk_assessment.get_risk_summary(),
                    },
                )

                return adjusted_forecast

        return forecast  # No adjustments needed

    def _apply_confidence_adjustment(
        self, prediction: Prediction, adjustment_factor: float
    ) -> Prediction:
        """Apply confidence adjustment to a prediction."""
        if prediction.result.binary_probability is None:
            return prediction  # Can't adjust non-binary predictions

        # Adjust probability towards 0.5 (less confident)
        original_prob = prediction.result.binary_probability
        adjusted_prob = 0.5 + (original_prob - 0.5) * adjustment_factor

        # Create adjusted prediction
        adjusted_prediction = Prediction.create_binary_prediction(
            question_id=prediction.question_id,
            research_report_id=prediction.research_report_id,
            probability=adjusted_prob,
            confidence=prediction.confidence,
            method=prediction.method,
            reasoning=f"Conservative adjustment applied: {prediction.reasoning}",
            created_by=f"{prediction.created_by}_conservative",
            reasoning_steps=prediction.reasoning_steps
            + [f"Applied conservative adjustment factor: {adjustment_factor}"],
            method_metadata={
                **prediction.method_metadata,
                "conservative_adjustment_applied": True,
                "original_probability": original_prob,
                "adjustment_factor": adjustment_factor,
            },
        )

        return adjusted_prediction

    def _calculate_score_impact(
        self,
        prediction: Prediction,
        risk_assessment: RiskAssessment,
        tournament_context: Dict[str, Any],
    ) -> float:
        """Calculate expected score impact of risk management action."""
        base_score = prediction.get_confidence_score() * 0.5  # Rough estimate

        if risk_assessment.recommended_action == ConservativeAction.ABSTAIN:
            return -risk_assessment.abstention_penalty
        elif risk_assessment.recommended_action == ConservativeAction.REDUCE_CONFIDENCE:
            if risk_assessment.confidence_adjustment:
                return base_score * (risk_assessment.confidence_adjustment - 1.0)
        elif risk_assessment.recommended_action == ConservativeAction.DEFER:
            return -0.1  # Small penalty for deferring

        return 0.0  # No impact for submit or request research

    def _generate_strategy_report(
        self,
        original_forecast: Forecast,
        adjusted_forecast: Forecast,
        risk_assessment: RiskAssessment,
        tournament_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate strategy application report."""
        return {
            "original_forecast_id": str(original_forecast.id),
            "adjusted_forecast_id": str(adjusted_forecast.id),
            "risk_assessment": risk_assessment.get_risk_summary(),
            "adjustments_applied": original_forecast.id != adjusted_forecast.id,
            "confidence_change": adjusted_forecast.confidence_score
            - original_forecast.confidence_score,
            "tournament_context": tournament_context,
            "strategy_timestamp": datetime.utcnow().isoformat(),
        }

    def _track_strategy_application(
        self,
        original_forecast: Forecast,
        adjusted_forecast: Forecast,
        risk_assessment: RiskAssessment,
        tournament_context: Optional[Dict[str, Any]],
    ) -> None:
        """Track strategy application for performance analysis."""
        performance_entry = {
            "timestamp": datetime.utcnow(),
            "original_confidence": original_forecast.confidence_score,
            "adjusted_confidence": adjusted_forecast.confidence_score,
            "risk_level": risk_assessment.overall_risk.value,
            "action_taken": risk_assessment.recommended_action.value,
            "abstained": risk_assessment.recommended_action
            == ConservativeAction.ABSTAIN,
            "submitted": risk_assessment.recommended_action
            in [ConservativeAction.SUBMIT, ConservativeAction.REDUCE_CONFIDENCE],
            "tournament_context": tournament_context,
        }

        self.strategy_performance_history.append(performance_entry)

        # Keep only recent history (last 100 entries)
        if len(self.strategy_performance_history) > 100:
            self.strategy_performance_history = self.strategy_performance_history[-100:]

    def _calculate_strategy_summary(
        self, recent_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for strategy performance."""
        if not recent_data:
            return {}

        return {
            "total_predictions": len(recent_data),
            "abstention_rate": statistics.mean(
                [1.0 if entry.get("abstained", False) else 0.0 for entry in recent_data]
            ),
            "submission_rate": statistics.mean(
                [1.0 if entry.get("submitted", False) else 0.0 for entry in recent_data]
            ),
            "average_confidence_adjustment": statistics.mean(
                [
                    entry.get("adjusted_confidence", 0.5)
                    - entry.get("original_confidence", 0.5)
                    for entry in recent_data
                ]
            ),
            "risk_level_distribution": self._calculate_risk_distribution(recent_data),
        }

    def _calculate_risk_distribution(
        self, recent_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate distribution of risk levels."""
        risk_counts = {}
        for entry in recent_data:
            risk_level = entry.get("risk_level", "moderate")
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1

        total = len(recent_data)
        return {risk_level: count / total for risk_level, count in risk_counts.items()}

    def _analyze_risk_patterns(
        self, recent_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns in risk assessment."""
        if not recent_data:
            return {}

        # Analyze risk trends over time
        risk_levels = [entry.get("risk_level", "moderate") for entry in recent_data]

        # Count high-risk predictions
        high_risk_count = sum(
            1 for risk in risk_levels if risk in ["high", "very_high"]
        )

        return {
            "high_risk_percentage": high_risk_count / len(recent_data) * 100,
            "most_common_risk_level": max(set(risk_levels), key=risk_levels.count),
            "risk_trend": self._analyze_risk_trend(recent_data),
        }

    def _analyze_risk_trend(self, recent_data: List[Dict[str, Any]]) -> str:
        """Analyze trend in risk levels over time."""
        if len(recent_data) < 5:
            return "insufficient_data"

        # Map risk levels to numeric values
        risk_values = []
        risk_mapping = {
            "very_low": 1,
            "low": 2,
            "moderate": 3,
            "high": 4,
            "very_high": 5,
        }

        for entry in recent_data:
            risk_level = entry.get("risk_level", "moderate")
            risk_values.append(risk_mapping.get(risk_level, 3))

        # Simple trend analysis
        first_half = risk_values[: len(risk_values) // 2]
        second_half = risk_values[len(risk_values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg + 0.5:
            return "increasing_risk"
        elif second_avg < first_avg - 0.5:
            return "decreasing_risk"
        else:
            return "stable_risk"

    def _analyze_tournament_performance(
        self, recent_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze tournament-specific performance."""
        tournament_entries = [
            entry
            for entry in recent_data
            if entry.get("tournament_context") is not None
        ]

        if not tournament_entries:
            return {"error": "No tournament data available"}

        return {
            "tournament_predictions": len(tournament_entries),
            "tournament_abstention_rate": statistics.mean(
                [
                    1.0 if entry.get("abstained", False) else 0.0
                    for entry in tournament_entries
                ]
            ),
            "average_time_pressure": statistics.mean(
                [
                    entry.get("tournament_context", {}).get("hours_to_deadline", 24)
                    for entry in tournament_entries
                ]
            ),
        }

    def _generate_strategy_recommendations(
        self,
        summary: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        tournament_analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations for strategy improvement."""
        recommendations = []

        # Abstention rate analysis
        abstention_rate = summary.get("abstention_rate", 0.0)
        if abstention_rate > 0.4:
            recommendations.append(
                "Abstention rate is high - consider relaxing risk thresholds"
            )
        elif abstention_rate < 0.05:
            recommendations.append(
                "Abstention rate is low - consider tightening risk thresholds"
            )

        # Risk trend analysis
        risk_trend = risk_analysis.get("risk_trend", "stable_risk")
        if risk_trend == "increasing_risk":
            recommendations.append(
                "Risk levels are increasing - review prediction methodology"
            )
        elif risk_trend == "decreasing_risk":
            recommendations.append(
                "Risk levels are decreasing - consider more aggressive strategies"
            )

        # High risk percentage
        high_risk_pct = risk_analysis.get("high_risk_percentage", 0.0)
        if high_risk_pct > 30:
            recommendations.append(
                "High percentage of high-risk predictions - improve risk assessment"
            )

        # Tournament performance
        if "error" not in tournament_analysis:
            tournament_abstention = tournament_analysis.get(
                "tournament_abstention_rate", 0.0
            )
            if tournament_abstention > 0.3:
                recommendations.append(
                    "High tournament abstention rate - balance risk vs. opportunity"
                )

        return recommendations
