"""Tests for StrategyAdaptationEngine service."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from src.domain.entities.forecast import Forecast, ForecastStatus
from src.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
    PredictionResult,
)
from src.domain.services.pattern_detector import PatternDetector
from src.domain.services.performance_analyzer import PerformanceAnalyzer
from src.domain.services.strategy_adaptation_engine import (
    AdaptationContext,
    AdaptationPlan,
    AdaptationResult,
    AdaptationTrigger,
    OptimizationObjective,
    StrategyAdaptationEngine,
    StrategyAdjustment,
)
from src.domain.value_objects.tournament_strategy import TournamentStrategy


class TestStrategyAdaptationEngine:
    """Test cases for StrategyAdaptationEngine."""

    @pytest.fixture
    def performance_analyzer(self):
        """Create mock PerformanceAnalyzer."""
        return Mock(spec=PerformanceAnalyzer)

    @pytest.fixture
    def pattern_detector(self):
        """Create mock PatternDetector."""
        return Mock(spec=PatternDetector)

    @pytest.fixture
    def adaptation_engine(self, performance_analyzer, pattern_detector):
        """Create StrategyAdaptationEngine instance."""
        return StrategyAdaptationEngine(performance_analyzer, pattern_detector)

    @pytest.fixture
    def sample_forecasts(self):
        """Create sample forecasts for testing."""
        forecasts = []
        question_id = uuid4()
        research_report_id = uuid4()

        for i in range(5):
            prediction = Prediction.create_binary_prediction(
                question_id=question_id,
                research_report_id=research_report_id,
                probability=0.6 + i * 0.05,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.ENSEMBLE,
                reasoning=f"Test reasoning {i}",
                created_by="test_agent",
            )

            forecast = Forecast.create_new(
                question_id=question_id,
                research_reports=[],
                predictions=[prediction],
                final_prediction=prediction,
            )
            forecast.status = ForecastStatus.RESOLVED
            forecasts.append(forecast)

        return forecasts

    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth."""
        return [True, False, True, True, False]

    @pytest.fixture
    def tournament_context(self):
        """Create sample tournament context."""
        return {
            "tournament_id": "test_tournament_2024",
            "current_ranking": 25,
            "total_participants": 100,
            "phase": "active",
            "time_remaining": timedelta(days=30),
            "competitive_pressure": 0.6,
        }

    @pytest.fixture
    def sample_tournament_strategy(self):
        """Create sample tournament strategy."""
        return TournamentStrategy.create_default(tournament_id="test_tournament")

    def test_evaluate_adaptation_need_basic(
        self,
        adaptation_engine,
        sample_forecasts,
        sample_ground_truth,
        tournament_context,
    ):
        """Test basic adaptation need evaluation."""
        # Mock analyzer responses
        adaptation_engine.performance_analyzer.analyze_resolved_predictions.return_value = {
            "overall_metrics": {
                "accuracy": 0.6,
                "brier_score": 0.25,
                "calibration_error": 0.1,
            },
            "calibration_analysis": {"expected_calibration_error": 0.12},
        }

        adaptation_engine.pattern_detector.detect_patterns.return_value = {
            "significant_patterns": 2,
            "detected_patterns": [
                {"strength": 0.3, "type": "test_pattern"},
                {"strength": 0.15, "type": "another_pattern"},
            ],
        }

        results = adaptation_engine.evaluate_adaptation_need(
            sample_forecasts, sample_ground_truth, tournament_context
        )

        assert "evaluation_timestamp" in results
        assert "adaptation_needed" in results
        assert "urgency_score" in results
        assert "identified_triggers" in results
        assert "adaptation_recommendations" in results

        assert isinstance(results["adaptation_needed"], bool)
        assert 0 <= results["urgency_score"] <= 1

    def test_identify_adaptation_triggers(self, adaptation_engine):
        """Test adaptation trigger identification."""
        performance_analysis = {
            "overall_metrics": {
                "brier_score": 0.35,  # High brier score
                "accuracy": 0.45,
            },
            "calibration_analysis": {
                "expected_calibration_error": 0.18  # High calibration error
            },
        }

        pattern_analysis = {"significant_patterns": 3}  # Many patterns

        tournament_context = {"phase_change": True, "competitive_pressure": 0.8}

        triggers = adaptation_engine._identify_adaptation_triggers(
            performance_analysis, pattern_analysis, tournament_context
        )

        assert isinstance(triggers, list)
        assert len(triggers) > 0

        # Should identify multiple triggers based on the data
        trigger_types = [trigger.value for trigger in triggers]
        assert "performance_decline" in trigger_types
        assert "calibration_drift" in trigger_types
        assert "pattern_detection" in trigger_types

    def test_calculate_adaptation_urgency(self, adaptation_engine):
        """Test adaptation urgency calculation."""
        triggers = [
            AdaptationTrigger.PERFORMANCE_DECLINE,
            AdaptationTrigger.CALIBRATION_DRIFT,
        ]

        performance_analysis = {
            "overall_metrics": {
                "brier_score": 0.4,  # Very high
                "accuracy": 0.35,  # Very low
            }
        }

        urgency = adaptation_engine._calculate_adaptation_urgency(
            triggers, performance_analysis
        )

        assert 0 <= urgency <= 1
        assert urgency > 0.5  # Should be high urgency given the poor performance

    def test_generate_adaptation_recommendations(self, adaptation_engine):
        """Test adaptation recommendation generation."""
        triggers = [
            AdaptationTrigger.PERFORMANCE_DECLINE,
            AdaptationTrigger.CALIBRATION_DRIFT,
            AdaptationTrigger.COMPETITIVE_PRESSURE,
        ]

        performance_analysis = {"overall_metrics": {"accuracy": 0.5}}
        pattern_analysis = {"significant_patterns": 1}
        tournament_context = {"tournament_id": "test"}

        recommendations = adaptation_engine._generate_adaptation_recommendations(
            triggers, performance_analysis, pattern_analysis, tournament_context
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) == len(triggers)  # One recommendation per trigger

        for rec in recommendations:
            assert "type" in rec
            assert "priority" in rec
            assert "actions" in rec
            assert "expected_impact" in rec

    def test_create_adaptation_plan(self, adaptation_engine):
        """Test adaptation plan creation."""
        context = AdaptationContext(
            trigger=AdaptationTrigger.PERFORMANCE_DECLINE,
            trigger_data={"performance_drop": 0.1},
            current_performance={"accuracy": 0.5, "brier_score": 0.3},
            tournament_context={"tournament_id": "test"},
            resource_constraints={"time_limit": 24},
            competitive_landscape={"ranking": 25},
            time_constraints={"deadline": datetime.utcnow() + timedelta(days=7)},
            historical_adaptations=[],
            timestamp=datetime.utcnow(),
        )

        plan = adaptation_engine.create_adaptation_plan(
            OptimizationObjective.MAXIMIZE_ACCURACY, context
        )

        assert isinstance(plan, AdaptationPlan)
        assert plan.objective == OptimizationObjective.MAXIMIZE_ACCURACY
        assert plan.context == context
        assert isinstance(plan.adjustments, list)
        assert isinstance(plan.implementation_sequence, list)
        assert 0 <= plan.plan_confidence <= 1
        assert plan.total_expected_impact >= 0
        assert isinstance(plan.estimated_implementation_time, timedelta)

    def test_generate_strategy_adjustments(self, adaptation_engine):
        """Test strategy adjustment generation."""
        context = AdaptationContext(
            trigger=AdaptationTrigger.PERFORMANCE_DECLINE,
            trigger_data={},
            current_performance={"accuracy": 0.5},
            tournament_context=None,
            resource_constraints={},
            competitive_landscape={},
            time_constraints={},
            historical_adaptations=[],
            timestamp=datetime.utcnow(),
        )

        adjustments = adaptation_engine._generate_strategy_adjustments(
            OptimizationObjective.MAXIMIZE_ACCURACY, context, None
        )

        assert isinstance(adjustments, list)

        for adjustment in adjustments:
            assert isinstance(adjustment, StrategyAdjustment)
            assert adjustment.adjustment_type is not None
            assert adjustment.target_component is not None
            assert adjustment.rationale is not None
            assert 0 <= adjustment.confidence <= 1
            assert 0 <= adjustment.implementation_priority <= 1

    def test_create_method_adjustment(self, adaptation_engine):
        """Test method preference adjustment creation."""
        context = AdaptationContext(
            trigger=AdaptationTrigger.PERFORMANCE_DECLINE,
            trigger_data={},
            current_performance={
                "accuracy": 0.5
            },  # Low accuracy should trigger adjustment
            tournament_context=None,
            resource_constraints={},
            competitive_landscape={},
            time_constraints={},
            historical_adaptations=[],
            timestamp=datetime.utcnow(),
        )

        adjustment = adaptation_engine._create_method_adjustment(context)

        assert adjustment is not None
        assert isinstance(adjustment, StrategyAdjustment)
        assert adjustment.adjustment_type == "method_preference"
        assert adjustment.target_component == "method_preferences"
        assert "ensemble" in str(adjustment.proposed_value)

    def test_create_ensemble_adjustment(self, adaptation_engine):
        """Test ensemble weight adjustment creation."""
        context = AdaptationContext(
            trigger=AdaptationTrigger.PERFORMANCE_DECLINE,
            trigger_data={},
            current_performance={},
            tournament_context=None,
            resource_constraints={},
            competitive_landscape={},
            time_constraints={},
            historical_adaptations=[],
            timestamp=datetime.utcnow(),
        )

        adjustment = adaptation_engine._create_ensemble_adjustment(context)

        assert adjustment is not None
        assert isinstance(adjustment, StrategyAdjustment)
        assert adjustment.adjustment_type == "ensemble_weights"
        assert adjustment.target_component == "ensemble_weights"

    def test_create_calibration_adjustment(self, adaptation_engine):
        """Test calibration adjustment creation."""
        context = AdaptationContext(
            trigger=AdaptationTrigger.CALIBRATION_DRIFT,
            trigger_data={},
            current_performance={},
            tournament_context=None,
            resource_constraints={},
            competitive_landscape={},
            time_constraints={},
            historical_adaptations=[],
            timestamp=datetime.utcnow(),
        )

        adjustment = adaptation_engine._create_calibration_adjustment(context)

        assert adjustment is not None
        assert isinstance(adjustment, StrategyAdjustment)
        assert adjustment.adjustment_type == "confidence_calibration"
        assert adjustment.target_component == "confidence_calibration"

    def test_optimize_implementation_sequence(self, adaptation_engine):
        """Test implementation sequence optimization."""
        adjustments = [
            StrategyAdjustment(
                adjustment_type="test1",
                target_component="component1",
                current_value={},
                proposed_value={},
                rationale="test",
                expected_impact=0.1,
                confidence=0.8,
                implementation_priority=0.3,  # Low priority
                success_metrics=[],
            ),
            StrategyAdjustment(
                adjustment_type="test2",
                target_component="component2",
                current_value={},
                proposed_value={},
                rationale="test",
                expected_impact=0.1,
                confidence=0.8,
                implementation_priority=0.9,  # High priority
                success_metrics=[],
            ),
        ]

        sequence = adaptation_engine._optimize_implementation_sequence(adjustments)

        assert isinstance(sequence, list)
        assert len(sequence) == 2
        assert sequence[0] == "component2"  # High priority first
        assert sequence[1] == "component1"  # Low priority second

    def test_calculate_plan_confidence(self, adaptation_engine):
        """Test plan confidence calculation."""
        adjustments = [
            StrategyAdjustment(
                adjustment_type="test1",
                target_component="component1",
                current_value={},
                proposed_value={},
                rationale="test",
                expected_impact=0.1,
                confidence=0.8,
                implementation_priority=0.5,
                success_metrics=[],
            ),
            StrategyAdjustment(
                adjustment_type="test2",
                target_component="component2",
                current_value={},
                proposed_value={},
                rationale="test",
                expected_impact=0.2,
                confidence=0.6,
                implementation_priority=0.5,
                success_metrics=[],
            ),
        ]

        context = Mock()
        confidence = adaptation_engine._calculate_plan_confidence(adjustments, context)

        assert 0 <= confidence <= 1
        # Should be weighted average: (0.8*0.1 + 0.6*0.2) / (0.1+0.2) = 0.2/0.3 = 0.667
        assert abs(confidence - 0.667) < 0.01

    def test_implement_adaptation_plan_dry_run(
        self, adaptation_engine, sample_tournament_strategy
    ):
        """Test adaptation plan implementation in dry run mode."""
        adjustments = [
            StrategyAdjustment(
                adjustment_type="test_adjustment",
                target_component="test_component",
                current_value={"value": 1},
                proposed_value={"value": 2},
                rationale="test rationale",
                expected_impact=0.1,
                confidence=0.8,
                implementation_priority=0.7,
                success_metrics=["test_metric"],
            )
        ]

        plan = AdaptationPlan(
            plan_id="test_plan",
            objective=OptimizationObjective.MAXIMIZE_ACCURACY,
            context=Mock(),
            adjustments=adjustments,
            implementation_sequence=["test_component"],
            total_expected_impact=0.1,
            plan_confidence=0.8,
            estimated_implementation_time=timedelta(hours=2),
            resource_requirements={},
            risk_assessment={},
            success_criteria=["test_criterion"],
            monitoring_schedule={},
            created_at=datetime.utcnow(),
        )

        result = adaptation_engine.implement_adaptation_plan(
            plan, sample_tournament_strategy, dry_run=True
        )

        assert isinstance(result, AdaptationResult)
        assert result.plan_id == "test_plan"
        assert result.implementation_status in ["successful", "partial", "failed"]
        assert 0 <= result.success_rate <= 1
        assert isinstance(result.lessons_learned, list)
        assert result.metadata["dry_run"] is True

    def test_implement_adaptation_plan_actual(
        self, adaptation_engine, sample_tournament_strategy
    ):
        """Test actual adaptation plan implementation."""
        adjustments = [
            StrategyAdjustment(
                adjustment_type="test_adjustment",
                target_component="test_component",
                current_value={"value": 1},
                proposed_value={"value": 2},
                rationale="test rationale",
                expected_impact=0.1,
                confidence=0.8,
                implementation_priority=0.7,
                success_metrics=["test_metric"],
            )
        ]

        plan = AdaptationPlan(
            plan_id="test_plan_actual",
            objective=OptimizationObjective.MAXIMIZE_ACCURACY,
            context=Mock(),
            adjustments=adjustments,
            implementation_sequence=["test_component"],
            total_expected_impact=0.1,
            plan_confidence=0.8,
            estimated_implementation_time=timedelta(hours=2),
            resource_requirements={},
            risk_assessment={},
            success_criteria=["test_criterion"],
            monitoring_schedule={},
            created_at=datetime.utcnow(),
        )

        result = adaptation_engine.implement_adaptation_plan(
            plan, sample_tournament_strategy, dry_run=False
        )

        assert isinstance(result, AdaptationResult)
        assert result.plan_id == "test_plan_actual"
        assert result.metadata["dry_run"] is False

        # Should be stored in history
        assert len(adaptation_engine.adaptation_history) > 0
        assert adaptation_engine.adaptation_history[-1].plan_id == "test_plan_actual"

    def test_optimize_tournament_positioning(
        self, adaptation_engine, tournament_context, sample_tournament_strategy
    ):
        """Test tournament positioning optimization."""
        competitive_intelligence = {
            "competitor_weaknesses": ["timing", "calibration"],
            "market_gaps": ["binary_questions", "short_term_predictions"],
        }

        results = adaptation_engine.optimize_tournament_positioning(
            tournament_context, sample_tournament_strategy, competitive_intelligence
        )

        assert "optimization_timestamp" in results
        assert "competitive_analysis" in results
        assert "positioning_opportunities" in results
        assert "recommended_adjustments" in results
        assert "resource_optimization" in results
        assert "timing_optimization" in results
        assert "expected_ranking_improvement" in results
        assert "implementation_priority" in results

        assert isinstance(results["positioning_opportunities"], list)
        assert isinstance(results["recommended_adjustments"], list)

    def test_analyze_competitive_position(
        self, adaptation_engine, tournament_context, sample_tournament_strategy
    ):
        """Test competitive position analysis."""
        competitive_intelligence = {"test": "data"}

        analysis = adaptation_engine._analyze_competitive_position(
            tournament_context, sample_tournament_strategy, competitive_intelligence
        )

        assert isinstance(analysis, dict)
        assert "current_ranking" in analysis
        assert "ranking_trend" in analysis
        assert "competitive_gaps" in analysis
        assert "competitive_advantages" in analysis
        assert "market_position" in analysis

    def test_identify_positioning_opportunities(self, adaptation_engine):
        """Test positioning opportunity identification."""
        competitive_analysis = {
            "competitive_gaps": ["question_type_specialization", "timing_optimization"],
            "current_ranking": 25,
        }

        tournament_context = {"tournament_id": "test"}

        opportunities = adaptation_engine._identify_positioning_opportunities(
            competitive_analysis, tournament_context
        )

        assert isinstance(opportunities, list)
        assert len(opportunities) > 0

        for opportunity in opportunities:
            assert "type" in opportunity
            assert "description" in opportunity
            assert "potential_impact" in opportunity
            assert "implementation_difficulty" in opportunity

    def test_generate_positioning_adjustments(
        self, adaptation_engine, sample_tournament_strategy, tournament_context
    ):
        """Test positioning adjustment generation."""
        opportunities = [
            {
                "type": "specialization",
                "description": "Focus on binary questions",
                "potential_impact": 0.15,
                "implementation_difficulty": 0.6,
            },
            {
                "type": "timing",
                "description": "Optimize submission timing",
                "potential_impact": 0.08,
                "implementation_difficulty": 0.3,
            },
        ]

        adjustments = adaptation_engine._generate_positioning_adjustments(
            opportunities, sample_tournament_strategy, tournament_context
        )

        assert isinstance(adjustments, list)
        assert len(adjustments) == len(opportunities)

        for adjustment in adjustments:
            assert "type" in adjustment
            assert "description" in adjustment
            assert "expected_impact" in adjustment

    def test_estimate_ranking_improvement(self, adaptation_engine):
        """Test ranking improvement estimation."""
        positioning_adjustments = [{"expected_impact": 0.1}, {"expected_impact": 0.05}]

        competitive_analysis = {"current_ranking": 30}

        improvement = adaptation_engine._estimate_ranking_improvement(
            positioning_adjustments, competitive_analysis
        )

        assert isinstance(improvement, (int, float))
        assert improvement >= 0
        assert improvement < 30  # Can't improve beyond rank 1

    def test_calculate_implementation_priority(
        self, adaptation_engine, tournament_context
    ):
        """Test implementation priority calculation."""
        # High impact adjustments
        high_impact_adjustments = [{"expected_impact": 0.15}, {"expected_impact": 0.1}]

        priority = adaptation_engine._calculate_implementation_priority(
            high_impact_adjustments, tournament_context
        )

        assert priority == "high"

        # Low impact adjustments
        low_impact_adjustments = [{"expected_impact": 0.02}, {"expected_impact": 0.03}]

        priority = adaptation_engine._calculate_implementation_priority(
            low_impact_adjustments, tournament_context
        )

        assert priority == "low"

    def test_get_adaptation_history(self, adaptation_engine):
        """Test adaptation history retrieval."""
        # Add some test adaptations
        test_adaptations = [
            AdaptationResult(
                plan_id="test_1",
                implementation_status="successful",
                adjustments_applied=["method_preferences"],
                performance_before={"accuracy": 0.6},
                performance_after={"accuracy": 0.65},
                actual_impact=0.05,
                success_rate=1.0,
                lessons_learned=["Test lesson"],
                rollback_actions=[],
                timestamp=datetime.utcnow() - timedelta(days=5),
            ),
            AdaptationResult(
                plan_id="test_2",
                implementation_status="partial",
                adjustments_applied=["ensemble_weights"],
                performance_before={"accuracy": 0.65},
                performance_after={"accuracy": 0.67},
                actual_impact=0.02,
                success_rate=0.5,
                lessons_learned=["Another lesson"],
                rollback_actions=[],
                timestamp=datetime.utcnow() - timedelta(days=2),
            ),
        ]

        adaptation_engine.adaptation_history.extend(test_adaptations)

        history = adaptation_engine.get_adaptation_history(days=7)

        assert "period_days" in history
        assert "total_adaptations" in history
        assert "successful_adaptations" in history
        assert "average_impact" in history
        assert "adaptation_frequency" in history
        assert "most_common_adjustments" in history

        assert history["period_days"] == 7
        assert history["total_adaptations"] == 2
        assert history["successful_adaptations"] == 1

    def test_get_current_strategy_status(
        self, adaptation_engine, sample_tournament_strategy
    ):
        """Test current strategy status retrieval."""
        adaptation_engine.current_strategy = sample_tournament_strategy

        status = adaptation_engine.get_current_strategy_status()

        assert "strategy_last_updated" in status
        assert "active_adaptations" in status
        assert "recent_adaptation_count" in status
        assert "adaptation_cooldown_remaining" in status
        assert "strategy_performance_trend" in status

    def test_calculate_strategy_performance_trend(self, adaptation_engine):
        """Test strategy performance trend calculation."""
        # Test improving trend
        improving_adaptations = [
            AdaptationResult(
                plan_id=f"test_{i}",
                implementation_status="successful",
                adjustments_applied=[],
                performance_before={},
                performance_after={},
                actual_impact=0.08,  # Positive impact
                success_rate=1.0,
                lessons_learned=[],
                rollback_actions=[],
                timestamp=datetime.utcnow() - timedelta(days=i),
            )
            for i in range(3)
        ]

        adaptation_engine.adaptation_history.extend(improving_adaptations)

        trend = adaptation_engine._calculate_strategy_performance_trend()
        assert trend == "improving"

        # Clear history and test declining trend
        adaptation_engine.adaptation_history.clear()

        declining_adaptations = [
            AdaptationResult(
                plan_id=f"test_{i}",
                implementation_status="failed",
                adjustments_applied=[],
                performance_before={},
                performance_after={},
                actual_impact=-0.08,  # Negative impact
                success_rate=0.0,
                lessons_learned=[],
                rollback_actions=[],
                timestamp=datetime.utcnow() - timedelta(days=i),
            )
            for i in range(3)
        ]

        adaptation_engine.adaptation_history.extend(declining_adaptations)

        trend = adaptation_engine._calculate_strategy_performance_trend()
        assert trend == "declining"

    def test_adaptation_trigger_enumeration(self):
        """Test adaptation trigger enumeration completeness."""
        expected_triggers = [
            "PERFORMANCE_DECLINE",
            "PATTERN_DETECTION",
            "COMPETITIVE_PRESSURE",
            "TOURNAMENT_PHASE_CHANGE",
            "RESOURCE_CONSTRAINT",
            "MARKET_OPPORTUNITY",
            "CALIBRATION_DRIFT",
            "METHOD_INEFFICIENCY",
            "SCHEDULED_REVIEW",
            "MANUAL_OVERRIDE",
        ]

        for expected_trigger in expected_triggers:
            assert hasattr(AdaptationTrigger, expected_trigger)

    def test_optimization_objective_enumeration(self):
        """Test optimization objective enumeration completeness."""
        expected_objectives = [
            "MAXIMIZE_ACCURACY",
            "MINIMIZE_BRIER_SCORE",
            "IMPROVE_CALIBRATION",
            "INCREASE_TOURNAMENT_RANKING",
            "OPTIMIZE_RESOURCE_EFFICIENCY",
            "ENHANCE_COMPETITIVE_ADVANTAGE",
            "BALANCE_RISK_REWARD",
            "MAXIMIZE_SCORING_POTENTIAL",
        ]

        for expected_objective in expected_objectives:
            assert hasattr(OptimizationObjective, expected_objective)

    def test_empty_forecasts_handling(self, adaptation_engine, tournament_context):
        """Test handling of empty forecast lists."""
        # Mock empty responses
        adaptation_engine.performance_analyzer.analyze_resolved_predictions.return_value = {
            "overall_metrics": {"accuracy": 0.5, "brier_score": 0.25}
        }

        adaptation_engine.pattern_detector.detect_patterns.return_value = {
            "significant_patterns": 0,
            "detected_patterns": [],
        }

        results = adaptation_engine.evaluate_adaptation_need([], [], tournament_context)

        # Should still work with empty data
        assert "adaptation_needed" in results
        assert "urgency_score" in results

    def test_no_tournament_context_handling(
        self, adaptation_engine, sample_forecasts, sample_ground_truth
    ):
        """Test handling when no tournament context is provided."""
        # Mock responses
        adaptation_engine.performance_analyzer.analyze_resolved_predictions.return_value = {
            "overall_metrics": {"accuracy": 0.6, "brier_score": 0.25},
            "calibration_analysis": {"expected_calibration_error": 0.1},
        }

        adaptation_engine.pattern_detector.detect_patterns.return_value = {
            "significant_patterns": 1,
            "detected_patterns": [],
        }

        results = adaptation_engine.evaluate_adaptation_need(
            sample_forecasts, sample_ground_truth, None
        )

        # Should work without tournament context
        assert "adaptation_needed" in results
        assert isinstance(results["identified_triggers"], list)

    def test_adaptation_context_creation(self):
        """Test AdaptationContext creation and validation."""
        context = AdaptationContext(
            trigger=AdaptationTrigger.PERFORMANCE_DECLINE,
            trigger_data={"test": "data"},
            current_performance={"accuracy": 0.6},
            tournament_context={"tournament_id": "test"},
            resource_constraints={"time": 24},
            competitive_landscape={"ranking": 25},
            time_constraints={"deadline": datetime.utcnow()},
            historical_adaptations=[],
            timestamp=datetime.utcnow(),
        )

        assert context.trigger == AdaptationTrigger.PERFORMANCE_DECLINE
        assert context.trigger_data["test"] == "data"
        assert context.current_performance["accuracy"] == 0.6

    def test_strategy_adjustment_creation(self):
        """Test StrategyAdjustment creation and validation."""
        adjustment = StrategyAdjustment(
            adjustment_type="test_type",
            target_component="test_component",
            current_value={"old": "value"},
            proposed_value={"new": "value"},
            rationale="Test rationale",
            expected_impact=0.1,
            confidence=0.8,
            implementation_priority=0.7,
            success_metrics=["metric1", "metric2"],
        )

        assert adjustment.adjustment_type == "test_type"
        assert adjustment.expected_impact == 0.1
        assert adjustment.confidence == 0.8
        assert len(adjustment.success_metrics) == 2

    def test_comprehensive_adaptation_workflow(
        self,
        adaptation_engine,
        sample_forecasts,
        sample_ground_truth,
        tournament_context,
        sample_tournament_strategy,
    ):
        """Test the complete adaptation workflow."""
        # Mock analyzer responses
        adaptation_engine.performance_analyzer.analyze_resolved_predictions.return_value = {
            "overall_metrics": {
                "accuracy": 0.5,  # Low accuracy to trigger adaptation
                "brier_score": 0.35,
                "calibration_error": 0.15,
            },
            "calibration_analysis": {"expected_calibration_error": 0.18},
        }

        adaptation_engine.pattern_detector.detect_patterns.return_value = {
            "significant_patterns": 3,
            "detected_patterns": [
                {"strength": 0.3, "type": "performance_pattern"},
                {"strength": 0.25, "type": "calibration_pattern"},
            ],
        }

        # Step 1: Evaluate adaptation need
        evaluation = adaptation_engine.evaluate_adaptation_need(
            sample_forecasts, sample_ground_truth, tournament_context
        )

        assert evaluation["adaptation_needed"] is True
        assert evaluation["urgency_score"] > 0.5

        # Step 2: Create adaptation plan
        context = AdaptationContext(
            trigger=AdaptationTrigger.PERFORMANCE_DECLINE,
            trigger_data={"performance_drop": 0.1},
            current_performance={"accuracy": 0.5},
            tournament_context=tournament_context,
            resource_constraints={},
            competitive_landscape={},
            time_constraints={},
            historical_adaptations=[],
            timestamp=datetime.utcnow(),
        )

        plan = adaptation_engine.create_adaptation_plan(
            OptimizationObjective.MAXIMIZE_ACCURACY, context
        )

        assert plan.objective == OptimizationObjective.MAXIMIZE_ACCURACY
        assert len(plan.adjustments) > 0
        assert plan.plan_confidence > 0

        # Step 3: Implement plan (dry run)
        result = adaptation_engine.implement_adaptation_plan(
            plan, sample_tournament_strategy, dry_run=True
        )

        assert result.plan_id == plan.plan_id
        assert result.implementation_status in ["successful", "partial", "failed"]

        # Step 4: Optimize tournament positioning
        positioning = adaptation_engine.optimize_tournament_positioning(
            tournament_context, sample_tournament_strategy
        )

        assert "competitive_analysis" in positioning
        assert "positioning_opportunities" in positioning
        assert "expected_ranking_improvement" in positioning
