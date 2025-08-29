"""Strategy adaptation engine for dynamic optimization and competitive positioning."""

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import structlog

from ..entities.forecast import Forecast, ForecastStatus
from ..entities.prediction import Prediction, PredictionConfidence, PredictionMethod
from ..entities.question import Question, QuestionType
from ..value_objects.tournament_strategy import QuestionPriority, TournamentStrategy
from .pattern_detector import AdaptationRecommendation, DetectedPattern, PatternDetector
from .performance_analyzer import ImprovementOpportunity, PerformanceAnalyzer

logger = structlog.get_logger(__name__)


class AdaptationTrigger(Enum):
    """Types of triggers that can initiate strategy adaptation."""

    PERFORMANCE_DECLINE = "performance_decline"
    PATTERN_DETECTION = "pattern_detection"
    COMPETITIVE_PRESSURE = "competitive_pressure"
    TOURNAMENT_PHASE_CHANGE = "tournament_phase_change"
    RESOURCE_CONSTRAINT = "resource_constraint"
    MARKET_OPPORTUNITY = "market_opportunity"
    CALIBRATION_DRIFT = "calibration_drift"
    METHOD_INEFFICIENCY = "method_inefficiency"
    SCHEDULED_REVIEW = "scheduled_review"
    MANUAL_OVERRIDE = "manual_override"


class OptimizationObjective(Enum):
    """Optimization objectives for strategy adaptation."""

    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_BRIER_SCORE = "minimize_brier_score"
    IMPROVE_CALIBRATION = "improve_calibration"
    INCREASE_TOURNAMENT_RANKING = "increase_tournament_ranking"
    OPTIMIZE_RESOURCE_EFFICIENCY = "optimize_resource_efficiency"
    ENHANCE_COMPETITIVE_ADVANTAGE = "enhance_competitive_advantage"
    BALANCE_RISK_REWARD = "balance_risk_reward"
    MAXIMIZE_SCORING_POTENTIAL = "maximize_scoring_potential"


@dataclass
class AdaptationContext:
    """Context information for strategy adaptation decisions."""

    trigger: AdaptationTrigger
    trigger_data: Dict[str, Any]
    current_performance: Dict[str, float]
    tournament_context: Optional[Dict[str, Any]]
    resource_constraints: Dict[str, Any]
    competitive_landscape: Dict[str, Any]
    time_constraints: Dict[str, Any]
    historical_adaptations: List[Dict[str, Any]]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyAdjustment:
    """Represents a specific strategy adjustment."""

    adjustment_type: str
    target_component: str  # What part of the strategy to adjust
    current_value: Any
    proposed_value: Any
    rationale: str
    expected_impact: float
    confidence: float
    implementation_priority: float
    rollback_plan: Optional[str] = None
    success_metrics: List[str] = field(default_factory=list)
    monitoring_period_days: int = 7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationPlan:
    """Comprehensive adaptation plan with multiple adjustments."""

    plan_id: str
    objective: OptimizationObjective
    context: AdaptationContext
    adjustments: List[StrategyAdjustment]
    implementation_sequence: List[str]  # Order of adjustment implementation
    total_expected_impact: float
    plan_confidence: float
    estimated_implementation_time: timedelta
    resource_requirements: Dict[str, Any]
    risk_assessment: Dict[str, float]
    success_criteria: List[str]
    monitoring_schedule: Dict[str, Any]
    created_at: datetime
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """Result of implementing an adaptation plan."""

    plan_id: str
    implementation_status: str
    adjustments_applied: List[str]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    actual_impact: float
    success_rate: float
    lessons_learned: List[str]
    rollback_actions: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyAdaptationEngine:
    """
    Engine for dynamic strategy optimization and competitive positioning.

    Implements strategy refinement based on performance feedback, resource allocation
    adjustment, competitive positioning, and tournament-specific adaptation.
    """

    def __init__(
        self,
        performance_analyzer: PerformanceAnalyzer,
        pattern_detector: PatternDetector,
    ):
        self.performance_analyzer = performance_analyzer
        self.pattern_detector = pattern_detector

        # Adaptation history and state
        self.adaptation_history: List[AdaptationResult] = []
        self.active_plans: List[AdaptationPlan] = []
        self.current_strategy: Optional[TournamentStrategy] = None

        # Configuration
        self.adaptation_threshold = (
            0.05  # Minimum performance change to trigger adaptation
        )
        self.confidence_threshold = 0.7  # Minimum confidence for adaptation decisions
        self.max_concurrent_adaptations = 3
        self.adaptation_cooldown_hours = 24

        # Optimization objectives and weights
        self.objective_weights = {
            OptimizationObjective.MAXIMIZE_ACCURACY: 0.3,
            OptimizationObjective.MINIMIZE_BRIER_SCORE: 0.25,
            OptimizationObjective.IMPROVE_CALIBRATION: 0.2,
            OptimizationObjective.INCREASE_TOURNAMENT_RANKING: 0.15,
            OptimizationObjective.OPTIMIZE_RESOURCE_EFFICIENCY: 0.1,
        }

        # Strategy components that can be adapted
        self.adaptable_components = [
            "method_preferences",
            "ensemble_weights",
            "confidence_calibration",
            "resource_allocation",
            "question_prioritization",
            "submission_timing",
            "research_depth",
            "risk_tolerance",
            "competitive_positioning",
        ]

    def evaluate_adaptation_need(
        self,
        recent_forecasts: List[Forecast],
        ground_truth: Optional[List[bool]] = None,
        tournament_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate whether strategy adaptation is needed based on current performance.

        Args:
            recent_forecasts: Recent forecasts for evaluation
            ground_truth: Ground truth for resolved forecasts
            tournament_context: Tournament-specific context

        Returns:
            Evaluation results with adaptation recommendations
        """
        logger.info(
            "Evaluating adaptation need",
            forecast_count=len(recent_forecasts),
            has_ground_truth=ground_truth is not None,
            evaluation_timestamp=datetime.utcnow(),
        )

        # Analyze current performance
        if ground_truth:
            performance_analysis = (
                self.performance_analyzer.analyze_resolved_predictions(
                    recent_forecasts, ground_truth
                )
            )
        else:
            performance_analysis = {
                "overall_metrics": {"accuracy": 0.5, "brier_score": 0.25}
            }

        # Detect patterns that might indicate adaptation needs
        questions = []  # Would need to be provided or retrieved
        pattern_analysis = self.pattern_detector.detect_patterns(
            recent_forecasts, questions, ground_truth, tournament_context
        )

        # Identify adaptation triggers
        triggers = self._identify_adaptation_triggers(
            performance_analysis, pattern_analysis, tournament_context
        )

        # Assess adaptation urgency
        urgency_score = self._calculate_adaptation_urgency(
            triggers, performance_analysis
        )

        # Generate adaptation recommendations
        recommendations = self._generate_adaptation_recommendations(
            triggers, performance_analysis, pattern_analysis, tournament_context
        )

        evaluation_results = {
            "evaluation_timestamp": datetime.utcnow(),
            "adaptation_needed": urgency_score > 0.5,
            "urgency_score": urgency_score,
            "identified_triggers": [self._serialize_trigger(t) for t in triggers],
            "performance_summary": performance_analysis.get("overall_metrics", {}),
            "pattern_summary": {
                "significant_patterns": pattern_analysis.get("significant_patterns", 0),
                "high_impact_patterns": len(
                    [
                        p
                        for p in pattern_analysis.get("detected_patterns", [])
                        if p.get("strength", 0) > 0.2
                    ]
                ),
            },
            "adaptation_recommendations": recommendations,
            "next_evaluation_recommended": datetime.utcnow()
            + timedelta(hours=self.adaptation_cooldown_hours),
        }

        logger.info(
            "Adaptation evaluation completed",
            adaptation_needed=evaluation_results["adaptation_needed"],
            urgency_score=urgency_score,
            triggers_count=len(triggers),
        )

        return evaluation_results

    def create_adaptation_plan(
        self,
        objective: OptimizationObjective,
        context: AdaptationContext,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> AdaptationPlan:
        """
        Create a comprehensive adaptation plan for strategy optimization.

        Args:
            objective: Primary optimization objective
            context: Adaptation context with trigger information
            constraints: Optional constraints on adaptation

        Returns:
            Detailed adaptation plan
        """
        logger.info(
            "Creating adaptation plan",
            objective=objective.value,
            trigger=context.trigger.value,
            plan_timestamp=datetime.utcnow(),
        )

        plan_id = f"adaptation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Generate strategy adjustments based on objective and context
        adjustments = self._generate_strategy_adjustments(
            objective, context, constraints
        )

        # Optimize adjustment sequence
        implementation_sequence = self._optimize_implementation_sequence(adjustments)

        # Calculate plan metrics
        total_expected_impact = sum(adj.expected_impact for adj in adjustments)
        plan_confidence = self._calculate_plan_confidence(adjustments, context)

        # Estimate implementation requirements
        implementation_time = self._estimate_implementation_time(adjustments)
        resource_requirements = self._calculate_resource_requirements(adjustments)
        risk_assessment = self._assess_adaptation_risks(adjustments, context)

        # Define success criteria and monitoring
        success_criteria = self._define_success_criteria(objective, adjustments)
        monitoring_schedule = self._create_monitoring_schedule(adjustments)

        adaptation_plan = AdaptationPlan(
            plan_id=plan_id,
            objective=objective,
            context=context,
            adjustments=adjustments,
            implementation_sequence=implementation_sequence,
            total_expected_impact=total_expected_impact,
            plan_confidence=plan_confidence,
            estimated_implementation_time=implementation_time,
            resource_requirements=resource_requirements,
            risk_assessment=risk_assessment,
            success_criteria=success_criteria,
            monitoring_schedule=monitoring_schedule,
            created_at=datetime.utcnow(),
        )

        logger.info(
            "Adaptation plan created",
            plan_id=plan_id,
            adjustments_count=len(adjustments),
            expected_impact=total_expected_impact,
            plan_confidence=plan_confidence,
        )

        return adaptation_plan

    def implement_adaptation_plan(
        self,
        plan: AdaptationPlan,
        current_strategy: TournamentStrategy,
        dry_run: bool = False,
    ) -> AdaptationResult:
        """
        Implement an adaptation plan to modify the current strategy.

        Args:
            plan: Adaptation plan to implement
            current_strategy: Current tournament strategy
            dry_run: If True, simulate implementation without making changes

        Returns:
            Results of the adaptation implementation
        """
        logger.info(
            "Implementing adaptation plan",
            plan_id=plan.plan_id,
            dry_run=dry_run,
            adjustments_count=len(plan.adjustments),
            implementation_timestamp=datetime.utcnow(),
        )

        # Record performance before adaptation
        performance_before = self._capture_current_performance(current_strategy)

        # Track implementation status
        adjustments_applied = []
        rollback_actions = []
        implementation_errors = []

        # Implement adjustments in sequence
        for adjustment_id in plan.implementation_sequence:
            adjustment = next(
                (
                    adj
                    for adj in plan.adjustments
                    if adj.target_component == adjustment_id
                ),
                None,
            )
            if not adjustment:
                continue

            try:
                if not dry_run:
                    # Apply the adjustment
                    success = self._apply_strategy_adjustment(
                        adjustment, current_strategy
                    )
                    if success:
                        adjustments_applied.append(adjustment_id)
                        if adjustment.rollback_plan:
                            rollback_actions.append(adjustment.rollback_plan)
                    else:
                        implementation_errors.append(f"Failed to apply {adjustment_id}")
                else:
                    # Simulate application
                    adjustments_applied.append(adjustment_id)
                    logger.info(f"Simulated application of {adjustment_id}")

            except Exception as e:
                implementation_errors.append(
                    f"Error applying {adjustment_id}: {str(e)}"
                )
                logger.error(
                    "Adjustment implementation failed",
                    adjustment_id=adjustment_id,
                    error=str(e),
                )

        # Calculate implementation results
        performance_after = (
            self._capture_current_performance(current_strategy)
            if not dry_run
            else performance_before
        )
        actual_impact = self._calculate_actual_impact(
            performance_before, performance_after
        )
        success_rate = (
            len(adjustments_applied) / len(plan.adjustments)
            if plan.adjustments
            else 0.0
        )

        # Generate lessons learned
        lessons_learned = self._extract_lessons_learned(
            plan, adjustments_applied, implementation_errors, actual_impact
        )

        # Determine implementation status
        if success_rate >= 0.8:
            status = "successful"
        elif success_rate >= 0.5:
            status = "partial"
        else:
            status = "failed"

        adaptation_result = AdaptationResult(
            plan_id=plan.plan_id,
            implementation_status=status,
            adjustments_applied=adjustments_applied,
            performance_before=performance_before,
            performance_after=performance_after,
            actual_impact=actual_impact,
            success_rate=success_rate,
            lessons_learned=lessons_learned,
            rollback_actions=rollback_actions,
            timestamp=datetime.utcnow(),
            metadata={
                "dry_run": dry_run,
                "implementation_errors": implementation_errors,
                "expected_impact": plan.total_expected_impact,
            },
        )

        # Store result in history
        if not dry_run:
            self.adaptation_history.append(adaptation_result)
            self.current_strategy = current_strategy

        logger.info(
            "Adaptation plan implementation completed",
            plan_id=plan.plan_id,
            status=status,
            success_rate=success_rate,
            actual_impact=actual_impact,
        )

        return adaptation_result

    def optimize_tournament_positioning(
        self,
        tournament_context: Dict[str, Any],
        current_strategy: TournamentStrategy,
        competitive_intelligence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize strategy for competitive tournament positioning.

        Args:
            tournament_context: Current tournament state and context
            current_strategy: Current tournament strategy
            competitive_intelligence: Intelligence about competitors

        Returns:
            Optimization recommendations and adjustments
        """
        logger.info(
            "Optimizing tournament positioning",
            tournament_id=tournament_context.get("tournament_id", "unknown"),
            current_ranking=tournament_context.get("current_ranking"),
            optimization_timestamp=datetime.utcnow(),
        )

        # Analyze current competitive position
        competitive_analysis = self._analyze_competitive_position(
            tournament_context, current_strategy, competitive_intelligence
        )

        # Identify positioning opportunities
        opportunities = self._identify_positioning_opportunities(
            competitive_analysis, tournament_context
        )

        # Generate positioning adjustments
        positioning_adjustments = self._generate_positioning_adjustments(
            opportunities, current_strategy, tournament_context
        )

        # Calculate optimal resource allocation
        resource_optimization = self._optimize_tournament_resources(
            tournament_context, current_strategy, opportunities
        )

        # Assess timing strategies
        timing_optimization = self._optimize_submission_timing(
            tournament_context, competitive_analysis
        )

        optimization_results = {
            "optimization_timestamp": datetime.utcnow(),
            "competitive_analysis": competitive_analysis,
            "positioning_opportunities": opportunities,
            "recommended_adjustments": positioning_adjustments,
            "resource_optimization": resource_optimization,
            "timing_optimization": timing_optimization,
            "expected_ranking_improvement": self._estimate_ranking_improvement(
                positioning_adjustments, competitive_analysis
            ),
            "implementation_priority": self._calculate_implementation_priority(
                positioning_adjustments, tournament_context
            ),
        }

        logger.info(
            "Tournament positioning optimization completed",
            opportunities_count=len(opportunities),
            adjustments_count=len(positioning_adjustments),
            expected_improvement=optimization_results["expected_ranking_improvement"],
        )

        return optimization_results

    def _identify_adaptation_triggers(
        self,
        performance_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[AdaptationTrigger]:
        """Identify triggers that indicate adaptation is needed."""
        triggers = []

        # Performance-based triggers
        overall_metrics = performance_analysis.get("overall_metrics", {})
        if overall_metrics.get("brier_score", 0.25) > 0.3:
            triggers.append(AdaptationTrigger.PERFORMANCE_DECLINE)

        # Pattern-based triggers
        if pattern_analysis.get("significant_patterns", 0) > 2:
            triggers.append(AdaptationTrigger.PATTERN_DETECTION)

        # Calibration-based triggers
        calibration = performance_analysis.get("calibration_analysis", {})
        if calibration.get("expected_calibration_error", 0.1) > 0.15:
            triggers.append(AdaptationTrigger.CALIBRATION_DRIFT)

        # Tournament-based triggers
        if tournament_context:
            if tournament_context.get("phase_change", False):
                triggers.append(AdaptationTrigger.TOURNAMENT_PHASE_CHANGE)

            if tournament_context.get("competitive_pressure", 0.5) > 0.7:
                triggers.append(AdaptationTrigger.COMPETITIVE_PRESSURE)

        return triggers

    def _calculate_adaptation_urgency(
        self, triggers: List[AdaptationTrigger], performance_analysis: Dict[str, Any]
    ) -> float:
        """Calculate urgency score for adaptation."""
        base_urgency = 0.0

        # Trigger-based urgency
        trigger_weights = {
            AdaptationTrigger.PERFORMANCE_DECLINE: 0.3,
            AdaptationTrigger.CALIBRATION_DRIFT: 0.25,
            AdaptationTrigger.COMPETITIVE_PRESSURE: 0.2,
            AdaptationTrigger.PATTERN_DETECTION: 0.15,
            AdaptationTrigger.TOURNAMENT_PHASE_CHANGE: 0.1,
        }

        for trigger in triggers:
            base_urgency += trigger_weights.get(trigger, 0.05)

        # Performance-based urgency adjustment
        overall_metrics = performance_analysis.get("overall_metrics", {})
        brier_score = overall_metrics.get("brier_score", 0.25)
        if brier_score > 0.35:
            base_urgency += 0.2

        accuracy = overall_metrics.get("accuracy", 0.5)
        if accuracy < 0.4:
            base_urgency += 0.15

        return min(1.0, base_urgency)

    def _generate_adaptation_recommendations(
        self,
        triggers: List[AdaptationTrigger],
        performance_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        tournament_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate specific adaptation recommendations."""
        recommendations = []

        for trigger in triggers:
            if trigger == AdaptationTrigger.PERFORMANCE_DECLINE:
                recommendations.append(
                    {
                        "type": "performance_improvement",
                        "priority": "high",
                        "actions": [
                            "Review and optimize method selection",
                            "Increase ensemble diversity",
                            "Enhance research quality",
                        ],
                        "expected_impact": 0.1,
                    }
                )

            elif trigger == AdaptationTrigger.CALIBRATION_DRIFT:
                recommendations.append(
                    {
                        "type": "calibration_adjustment",
                        "priority": "medium",
                        "actions": [
                            "Recalibrate confidence levels",
                            "Implement temperature scaling",
                            "Add calibration feedback loops",
                        ],
                        "expected_impact": 0.08,
                    }
                )

            elif trigger == AdaptationTrigger.COMPETITIVE_PRESSURE:
                recommendations.append(
                    {
                        "type": "competitive_positioning",
                        "priority": "high",
                        "actions": [
                            "Adjust question prioritization",
                            "Optimize submission timing",
                            "Exploit identified market gaps",
                        ],
                        "expected_impact": 0.12,
                    }
                )

        return recommendations

    def _generate_strategy_adjustments(
        self,
        objective: OptimizationObjective,
        context: AdaptationContext,
        constraints: Optional[Dict[str, Any]],
    ) -> List[StrategyAdjustment]:
        """Generate specific strategy adjustments based on objective and context."""
        adjustments = []

        # Method preference adjustments
        if objective in [
            OptimizationObjective.MAXIMIZE_ACCURACY,
            OptimizationObjective.MINIMIZE_BRIER_SCORE,
        ]:
            method_adjustment = self._create_method_adjustment(context)
            if method_adjustment:
                adjustments.append(method_adjustment)

        # Ensemble weight adjustments
        if objective == OptimizationObjective.MINIMIZE_BRIER_SCORE:
            ensemble_adjustment = self._create_ensemble_adjustment(context)
            if ensemble_adjustment:
                adjustments.append(ensemble_adjustment)

        # Calibration adjustments
        if objective == OptimizationObjective.IMPROVE_CALIBRATION:
            calibration_adjustment = self._create_calibration_adjustment(context)
            if calibration_adjustment:
                adjustments.append(calibration_adjustment)

        # Resource allocation adjustments
        if objective == OptimizationObjective.OPTIMIZE_RESOURCE_EFFICIENCY:
            resource_adjustment = self._create_resource_adjustment(context)
            if resource_adjustment:
                adjustments.append(resource_adjustment)

        # Competitive positioning adjustments
        if objective == OptimizationObjective.INCREASE_TOURNAMENT_RANKING:
            positioning_adjustment = self._create_positioning_adjustment(context)
            if positioning_adjustment:
                adjustments.append(positioning_adjustment)

        return adjustments

    def _create_method_adjustment(
        self, context: AdaptationContext
    ) -> Optional[StrategyAdjustment]:
        """Create method preference adjustment."""
        # Analyze method performance from context
        performance_data = context.current_performance

        # Simple heuristic: if accuracy is low, suggest ensemble methods
        if performance_data.get("accuracy", 0.5) < 0.6:
            return StrategyAdjustment(
                adjustment_type="method_preference",
                target_component="method_preferences",
                current_value={
                    "ensemble": 0.3,
                    "chain_of_thought": 0.4,
                    "tree_of_thought": 0.3,
                },
                proposed_value={
                    "ensemble": 0.5,
                    "chain_of_thought": 0.3,
                    "tree_of_thought": 0.2,
                },
                rationale="Increase ensemble usage to improve accuracy",
                expected_impact=0.08,
                confidence=0.7,
                implementation_priority=0.8,
                success_metrics=["accuracy_improvement", "brier_score_reduction"],
            )

        return None

    def _create_ensemble_adjustment(
        self, context: AdaptationContext
    ) -> Optional[StrategyAdjustment]:
        """Create ensemble weight adjustment."""
        return StrategyAdjustment(
            adjustment_type="ensemble_weights",
            target_component="ensemble_weights",
            current_value={"agent_1": 0.33, "agent_2": 0.33, "agent_3": 0.34},
            proposed_value={"agent_1": 0.4, "agent_2": 0.35, "agent_3": 0.25},
            rationale="Reweight ensemble based on recent performance",
            expected_impact=0.05,
            confidence=0.6,
            implementation_priority=0.6,
            success_metrics=["ensemble_performance_improvement"],
        )

    def _create_calibration_adjustment(
        self, context: AdaptationContext
    ) -> Optional[StrategyAdjustment]:
        """Create calibration adjustment."""
        return StrategyAdjustment(
            adjustment_type="confidence_calibration",
            target_component="confidence_calibration",
            current_value={"temperature": 1.0, "bias_correction": 0.0},
            proposed_value={"temperature": 1.2, "bias_correction": -0.05},
            rationale="Adjust calibration to reduce overconfidence",
            expected_impact=0.06,
            confidence=0.8,
            implementation_priority=0.7,
            success_metrics=["calibration_error_reduction"],
        )

    def _create_resource_adjustment(
        self, context: AdaptationContext
    ) -> Optional[StrategyAdjustment]:
        """Create resource allocation adjustment."""
        return StrategyAdjustment(
            adjustment_type="resource_allocation",
            target_component="resource_allocation",
            current_value={
                "research_time": 0.4,
                "analysis_time": 0.3,
                "validation_time": 0.3,
            },
            proposed_value={
                "research_time": 0.5,
                "analysis_time": 0.3,
                "validation_time": 0.2,
            },
            rationale="Increase research time to improve prediction quality",
            expected_impact=0.07,
            confidence=0.6,
            implementation_priority=0.5,
            success_metrics=["research_quality_improvement"],
        )

    def _create_positioning_adjustment(
        self, context: AdaptationContext
    ) -> Optional[StrategyAdjustment]:
        """Create competitive positioning adjustment."""
        return StrategyAdjustment(
            adjustment_type="competitive_positioning",
            target_component="competitive_positioning",
            current_value={"question_focus": "balanced", "timing_strategy": "early"},
            proposed_value={
                "question_focus": "specialized",
                "timing_strategy": "optimal",
            },
            rationale="Focus on specialized questions for competitive advantage",
            expected_impact=0.1,
            confidence=0.7,
            implementation_priority=0.9,
            success_metrics=["tournament_ranking_improvement"],
        )

    def _optimize_implementation_sequence(
        self, adjustments: List[StrategyAdjustment]
    ) -> List[str]:
        """Optimize the sequence of implementing adjustments."""
        # Sort by implementation priority
        sorted_adjustments = sorted(
            adjustments, key=lambda x: x.implementation_priority, reverse=True
        )
        return [adj.target_component for adj in sorted_adjustments]

    def _calculate_plan_confidence(
        self, adjustments: List[StrategyAdjustment], context: AdaptationContext
    ) -> float:
        """Calculate overall confidence in the adaptation plan."""
        if not adjustments:
            return 0.0

        # Weight by expected impact
        weighted_confidence = sum(
            adj.confidence * adj.expected_impact for adj in adjustments
        )
        total_impact = sum(adj.expected_impact for adj in adjustments)

        return weighted_confidence / total_impact if total_impact > 0 else 0.0

    def _estimate_implementation_time(
        self, adjustments: List[StrategyAdjustment]
    ) -> timedelta:
        """Estimate time required to implement all adjustments."""
        # Simple heuristic: each adjustment takes 1-4 hours based on complexity
        base_hours = len(adjustments) * 2  # 2 hours per adjustment on average
        return timedelta(hours=base_hours)

    def _calculate_resource_requirements(
        self, adjustments: List[StrategyAdjustment]
    ) -> Dict[str, Any]:
        """Calculate resource requirements for implementing adjustments."""
        return {
            "computational_resources": "medium",
            "human_oversight_hours": len(adjustments) * 0.5,
            "testing_time_hours": len(adjustments) * 1.0,
            "rollback_preparation_hours": len(adjustments) * 0.25,
        }

    def _assess_adaptation_risks(
        self, adjustments: List[StrategyAdjustment], context: AdaptationContext
    ) -> Dict[str, float]:
        """Assess risks associated with the adaptation plan."""
        return {
            "performance_degradation_risk": 0.2,
            "implementation_failure_risk": 0.1,
            "unintended_consequences_risk": 0.15,
            "rollback_difficulty_risk": 0.1,
            "competitive_disadvantage_risk": 0.05,
        }

    def _define_success_criteria(
        self, objective: OptimizationObjective, adjustments: List[StrategyAdjustment]
    ) -> List[str]:
        """Define success criteria for the adaptation plan."""
        criteria = []

        if objective == OptimizationObjective.MAXIMIZE_ACCURACY:
            criteria.append("Accuracy improvement of at least 5%")
        elif objective == OptimizationObjective.MINIMIZE_BRIER_SCORE:
            criteria.append("Brier score reduction of at least 0.02")
        elif objective == OptimizationObjective.IMPROVE_CALIBRATION:
            criteria.append("Calibration error reduction of at least 0.05")

        # Add adjustment-specific criteria
        for adj in adjustments:
            criteria.extend(adj.success_metrics)

        return list(set(criteria))  # Remove duplicates

    def _create_monitoring_schedule(
        self, adjustments: List[StrategyAdjustment]
    ) -> Dict[str, Any]:
        """Create monitoring schedule for tracking adaptation success."""
        return {
            "immediate_check": "24 hours after implementation",
            "short_term_review": "1 week after implementation",
            "medium_term_review": "1 month after implementation",
            "performance_metrics_frequency": "daily",
            "rollback_decision_point": "72 hours after implementation",
        }

    def _apply_strategy_adjustment(
        self, adjustment: StrategyAdjustment, strategy: TournamentStrategy
    ) -> bool:
        """Apply a specific strategy adjustment."""
        try:
            # This would implement the actual strategy modification
            # For now, we'll simulate successful application
            logger.info(
                "Applied strategy adjustment",
                adjustment_type=adjustment.adjustment_type,
                target_component=adjustment.target_component,
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to apply strategy adjustment",
                adjustment_type=adjustment.adjustment_type,
                error=str(e),
            )
            return False

    def _capture_current_performance(
        self, strategy: TournamentStrategy
    ) -> Dict[str, float]:
        """Capture current performance metrics."""
        # This would capture actual performance metrics
        return {
            "accuracy": 0.65,
            "brier_score": 0.22,
            "calibration_error": 0.08,
            "tournament_ranking": 15.0,
        }

    def _calculate_actual_impact(
        self, performance_before: Dict[str, float], performance_after: Dict[str, float]
    ) -> float:
        """Calculate actual impact of adaptation."""
        # Simple metric: improvement in accuracy
        accuracy_before = performance_before.get("accuracy", 0.5)
        accuracy_after = performance_after.get("accuracy", 0.5)
        return accuracy_after - accuracy_before

    def _extract_lessons_learned(
        self,
        plan: AdaptationPlan,
        adjustments_applied: List[str],
        implementation_errors: List[str],
        actual_impact: float,
    ) -> List[str]:
        """Extract lessons learned from adaptation implementation."""
        lessons = []

        if actual_impact > plan.total_expected_impact:
            lessons.append("Adaptation exceeded expected impact")
        elif actual_impact < plan.total_expected_impact * 0.5:
            lessons.append("Adaptation underperformed expectations")

        if implementation_errors:
            lessons.append(
                f"Implementation challenges: {len(implementation_errors)} errors encountered"
            )

        if len(adjustments_applied) == len(plan.adjustments):
            lessons.append("All planned adjustments successfully implemented")

        return lessons

    def _analyze_competitive_position(
        self,
        tournament_context: Dict[str, Any],
        current_strategy: TournamentStrategy,
        competitive_intelligence: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze current competitive position in tournament."""
        return {
            "current_ranking": tournament_context.get("current_ranking", 50),
            "ranking_trend": "stable",
            "competitive_gaps": ["question_type_specialization", "timing_optimization"],
            "competitive_advantages": ["ensemble_methods", "calibration_quality"],
            "market_position": "middle_tier",
        }

    def _identify_positioning_opportunities(
        self, competitive_analysis: Dict[str, Any], tournament_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify opportunities for better competitive positioning."""
        opportunities = []

        # Example opportunities based on competitive gaps
        gaps = competitive_analysis.get("competitive_gaps", [])
        if "question_type_specialization" in gaps:
            opportunities.append(
                {
                    "type": "specialization",
                    "description": "Focus on specific question types where we have advantage",
                    "potential_impact": 0.15,
                    "implementation_difficulty": 0.6,
                }
            )

        if "timing_optimization" in gaps:
            opportunities.append(
                {
                    "type": "timing",
                    "description": "Optimize submission timing for competitive advantage",
                    "potential_impact": 0.08,
                    "implementation_difficulty": 0.3,
                }
            )

        return opportunities

    def _generate_positioning_adjustments(
        self,
        opportunities: List[Dict[str, Any]],
        current_strategy: TournamentStrategy,
        tournament_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate specific positioning adjustments."""
        adjustments = []

        for opportunity in opportunities:
            if opportunity["type"] == "specialization":
                adjustments.append(
                    {
                        "type": "question_focus",
                        "description": "Increase focus on binary questions",
                        "current_allocation": 0.33,
                        "proposed_allocation": 0.5,
                        "expected_impact": opportunity["potential_impact"],
                    }
                )

            elif opportunity["type"] == "timing":
                adjustments.append(
                    {
                        "type": "submission_timing",
                        "description": "Shift to optimal timing window",
                        "current_strategy": "early_submission",
                        "proposed_strategy": "optimal_window",
                        "expected_impact": opportunity["potential_impact"],
                    }
                )

        return adjustments

    def _optimize_tournament_resources(
        self,
        tournament_context: Dict[str, Any],
        current_strategy: TournamentStrategy,
        opportunities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Optimize resource allocation for tournament performance."""
        return {
            "research_allocation": {
                "high_value_questions": 0.6,
                "medium_value_questions": 0.3,
                "low_value_questions": 0.1,
            },
            "method_allocation": {"ensemble_methods": 0.5, "individual_methods": 0.5},
            "time_allocation": {
                "research": 0.4,
                "analysis": 0.3,
                "validation": 0.2,
                "submission": 0.1,
            },
        }

    def _optimize_submission_timing(
        self, tournament_context: Dict[str, Any], competitive_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize submission timing strategy."""
        return {
            "recommended_strategy": "adaptive_timing",
            "early_submission_threshold": 0.8,  # Submit early if confidence > 0.8
            "optimal_window_start": "48_hours_before_deadline",
            "optimal_window_end": "12_hours_before_deadline",
            "last_minute_threshold": 0.9,  # Only submit last minute if confidence > 0.9
        }

    def _estimate_ranking_improvement(
        self,
        positioning_adjustments: List[Dict[str, Any]],
        competitive_analysis: Dict[str, Any],
    ) -> float:
        """Estimate expected ranking improvement from positioning adjustments."""
        total_impact = sum(
            adj.get("expected_impact", 0) for adj in positioning_adjustments
        )
        current_ranking = competitive_analysis.get("current_ranking", 50)

        # Simple heuristic: each 0.1 impact improves ranking by 5 positions
        ranking_improvement = total_impact * 50
        return min(
            ranking_improvement, current_ranking - 1
        )  # Can't improve beyond rank 1

    def _calculate_implementation_priority(
        self,
        positioning_adjustments: List[Dict[str, Any]],
        tournament_context: Dict[str, Any],
    ) -> str:
        """Calculate implementation priority for positioning adjustments."""
        total_impact = sum(
            adj.get("expected_impact", 0) for adj in positioning_adjustments
        )

        if total_impact > 0.2:
            return "high"
        elif total_impact > 0.1:
            return "medium"
        else:
            return "low"

    def _serialize_trigger(self, trigger: AdaptationTrigger) -> Dict[str, Any]:
        """Serialize adaptation trigger for JSON output."""
        return {
            "type": trigger.value,
            "description": f"Adaptation trigger: {trigger.value}",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_adaptation_history(self, days: int = 30) -> Dict[str, Any]:
        """Get adaptation history for the last N days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        recent_adaptations = [
            result
            for result in self.adaptation_history
            if result.timestamp >= cutoff_date
        ]

        return {
            "period_days": days,
            "total_adaptations": len(recent_adaptations),
            "successful_adaptations": len(
                [
                    r
                    for r in recent_adaptations
                    if r.implementation_status == "successful"
                ]
            ),
            "average_impact": (
                statistics.mean([r.actual_impact for r in recent_adaptations])
                if recent_adaptations
                else 0.0
            ),
            "adaptation_frequency": len(recent_adaptations)
            / max(1, days)
            * 7,  # Per week
            "most_common_adjustments": self._get_most_common_adjustments(
                recent_adaptations
            ),
        }

    def _get_most_common_adjustments(
        self, adaptations: List[AdaptationResult]
    ) -> Dict[str, int]:
        """Get most common types of adjustments from adaptation history."""
        adjustment_counts = defaultdict(int)

        for adaptation in adaptations:
            for adjustment in adaptation.adjustments_applied:
                adjustment_counts[adjustment] += 1

        return dict(adjustment_counts)

    def get_current_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and recent adaptations."""
        return {
            "strategy_last_updated": (
                self.current_strategy.created_at if self.current_strategy else None
            ),
            "active_adaptations": len(self.active_plans),
            "recent_adaptation_count": len(
                [
                    r
                    for r in self.adaptation_history
                    if r.timestamp >= datetime.utcnow() - timedelta(days=7)
                ]
            ),
            "adaptation_cooldown_remaining": (
                max(
                    0,
                    (
                        self.adaptation_history[-1].timestamp
                        + timedelta(hours=self.adaptation_cooldown_hours)
                        - datetime.utcnow()
                    ).total_seconds()
                    / 3600,
                )
                if self.adaptation_history
                else 0
            ),
            "strategy_performance_trend": self._calculate_strategy_performance_trend(),
        }

    def _calculate_strategy_performance_trend(self) -> str:
        """Calculate recent performance trend of the strategy."""
        if len(self.adaptation_history) < 2:
            return "insufficient_data"

        recent_impacts = [r.actual_impact for r in self.adaptation_history[-5:]]
        avg_impact = statistics.mean(recent_impacts)

        if avg_impact > 0.05:
            return "improving"
        elif avg_impact < -0.05:
            return "declining"
        else:
            return "stable"
