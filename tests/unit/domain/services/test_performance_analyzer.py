"""Tests for PerformanceAnalyzer service."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.domain.entities.forecast import Forecast, ForecastStatus
from src.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
)
from src.domain.services.performance_analyzer import (
    ImprovementOpportunity,
    ImprovementOpportunityType,
    LearningInsight,
    PerformanceAnalyzer,
    PerformanceMetric,
    PerformanceMetricType,
    PerformancePattern,
)


class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create PerformanceAnalyzer instance."""
        return PerformanceAnalyzer()

    @pytest.fixture
    def sample_forecasts(self):
        """Create sample forecasts for testing."""
        forecasts = []
        question_id = uuid4()
        research_report_id = uuid4()

        # Create diverse forecasts with different predictions and confidence levels
        forecast_data = [
            (
                0.7,
                PredictionConfidence.HIGH,
                "agent_1",
                PredictionMethod.CHAIN_OF_THOUGHT,
            ),
            (
                0.3,
                PredictionConfidence.MEDIUM,
                "agent_2",
                PredictionMethod.TREE_OF_THOUGHT,
            ),
            (0.8, PredictionConfidence.VERY_HIGH, "agent_1", PredictionMethod.REACT),
            (0.2, PredictionConfidence.LOW, "agent_3", PredictionMethod.AUTO_COT),
            (0.6, PredictionConfidence.MEDIUM, "agent_2", PredictionMethod.ENSEMBLE),
            (
                0.9,
                PredictionConfidence.VERY_HIGH,
                "agent_1",
                PredictionMethod.CHAIN_OF_THOUGHT,
            ),
            (
                0.1,
                PredictionConfidence.HIGH,
                "agent_3",
                PredictionMethod.TREE_OF_THOUGHT,
            ),  # Overconfident
            (
                0.4,
                PredictionConfidence.LOW,
                "agent_2",
                PredictionMethod.REACT,
            ),  # Underconfident
        ]

        for i, (prob, conf, agent, method) in enumerate(forecast_data):
            prediction = Prediction.create_binary_prediction(
                question_id=question_id,
                research_report_id=research_report_id,
                probability=prob,
                confidence=conf,
                method=method,
                reasoning=f"Test reasoning for prediction {i}",
                created_by=agent,
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
        """Create sample ground truth values."""
        # Ground truth that creates interesting patterns
        return [True, False, True, False, True, True, False, False]

    def test_analyze_resolved_predictions_basic(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test basic resolved prediction analysis."""
        results = analyzer.analyze_resolved_predictions(
            sample_forecasts, sample_ground_truth
        )

        assert "analysis_timestamp" in results
        assert "sample_size" in results
        assert results["sample_size"] == len(sample_forecasts)

        assert "overall_metrics" in results
        overall = results["overall_metrics"]
        assert "brier_score" in overall
        assert "accuracy" in overall
        assert "log_score" in overall
        assert 0 <= overall["accuracy"] <= 1
        assert overall["brier_score"] >= 0

    def test_calculate_overall_metrics(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test overall metrics calculation."""
        metrics = analyzer._calculate_overall_metrics(
            sample_forecasts, sample_ground_truth
        )

        # Check all required metrics are present
        required_metrics = [
            "brier_score",
            "log_score",
            "accuracy",
            "resolution",
            "reliability",
            "sharpness",
            "discrimination",
            "base_rate",
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

        # Check metric ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["brier_score"] >= 0
        assert 0 <= metrics["base_rate"] <= 1
        assert 0 <= metrics["discrimination"] <= 1

    def test_calculate_agent_metrics(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test agent-specific metrics calculation."""
        metrics = analyzer._calculate_agent_metrics(
            sample_forecasts, sample_ground_truth
        )

        # Should have metrics for agents with sufficient samples
        assert len(metrics) > 0

        for agent_id, agent_metrics in metrics.items():
            assert "brier_score" in agent_metrics
            assert "accuracy" in agent_metrics
            assert "prediction_count" in agent_metrics
            assert "confidence_correlation" in agent_metrics
            assert agent_metrics["prediction_count"] >= 3

    def test_calculate_method_metrics(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test method-specific metrics calculation."""
        metrics = analyzer._calculate_method_metrics(
            sample_forecasts, sample_ground_truth
        )

        # Should have metrics for methods with sufficient samples
        for method, method_metrics in metrics.items():
            assert "brier_score" in method_metrics
            assert "accuracy" in method_metrics
            assert "prediction_count" in method_metrics

    def test_analyze_calibration(self, analyzer, sample_forecasts, sample_ground_truth):
        """Test calibration analysis."""
        calibration = analyzer._analyze_calibration(
            sample_forecasts, sample_ground_truth
        )

        assert "expected_calibration_error" in calibration
        assert "maximum_calibration_error" in calibration
        assert "calibration_bins" in calibration
        assert "is_well_calibrated" in calibration
        assert "overconfidence_detected" in calibration
        assert "underconfidence_detected" in calibration

        assert 0 <= calibration["expected_calibration_error"] <= 1
        assert 0 <= calibration["maximum_calibration_error"] <= 1
        assert isinstance(calibration["is_well_calibrated"], bool)

    def test_identify_improvement_opportunities(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test improvement opportunity identification."""
        overall_metrics = analyzer._calculate_overall_metrics(
            sample_forecasts, sample_ground_truth
        )
        opportunities = analyzer._identify_improvement_opportunities(
            sample_forecasts, sample_ground_truth, overall_metrics
        )

        assert isinstance(opportunities, list)

        for opp in opportunities:
            assert isinstance(opp, ImprovementOpportunity)
            assert opp.opportunity_type in ImprovementOpportunityType
            assert 0 <= opp.severity <= 1
            assert 0 <= opp.potential_impact <= 1
            assert 0 <= opp.implementation_difficulty <= 1
            assert len(opp.recommended_actions) > 0

    def test_detect_performance_patterns(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test performance pattern detection."""
        patterns = analyzer._detect_performance_patterns(
            sample_forecasts, sample_ground_truth
        )

        assert isinstance(patterns, list)

        for pattern in patterns:
            assert isinstance(pattern, PerformancePattern)
            assert pattern.pattern_type is not None
            assert 0 <= pattern.frequency <= 1
            assert 0 <= pattern.confidence <= 1
            assert isinstance(pattern.affected_contexts, list)

    def test_detect_confidence_accuracy_pattern(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test confidence-accuracy pattern detection."""
        pattern = analyzer._detect_confidence_accuracy_pattern(
            sample_forecasts, sample_ground_truth
        )

        # With our sample data, we should detect overconfidence pattern
        if pattern:
            assert pattern.pattern_type == "confidence_accuracy_mismatch"
            assert pattern.confidence > 0
            assert len(pattern.examples) > 0

    def test_detect_method_performance_pattern(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test method performance pattern detection."""
        pattern = analyzer._detect_method_performance_pattern(
            sample_forecasts, sample_ground_truth
        )

        if pattern:
            assert pattern.pattern_type == "method_performance_difference"
            assert "method_scores" in pattern.metadata
            assert pattern.performance_impact != 0

    def test_generate_learning_insights(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test learning insight generation."""
        overall_metrics = analyzer._calculate_overall_metrics(
            sample_forecasts, sample_ground_truth
        )
        agent_metrics = analyzer._calculate_agent_metrics(
            sample_forecasts, sample_ground_truth
        )
        method_metrics = analyzer._calculate_method_metrics(
            sample_forecasts, sample_ground_truth
        )
        opportunities = analyzer._identify_improvement_opportunities(
            sample_forecasts, sample_ground_truth, overall_metrics
        )
        patterns = analyzer._detect_performance_patterns(
            sample_forecasts, sample_ground_truth
        )

        insights = analyzer._generate_learning_insights(
            overall_metrics, agent_metrics, method_metrics, opportunities, patterns
        )

        assert isinstance(insights, list)

        for insight in insights:
            assert isinstance(insight, LearningInsight)
            assert insight.insight_type is not None
            assert insight.title is not None
            assert 0 <= insight.confidence <= 1
            assert 0 <= insight.priority <= 1
            assert len(insight.actionable_recommendations) > 0

    def test_store_performance_metrics(self, analyzer):
        """Test performance metrics storage."""
        overall_metrics = {"brier_score": 0.2, "accuracy": 0.8, "log_score": 0.5}
        agent_metrics = {"agent_1": {"brier_score": 0.15, "accuracy": 0.85}}
        method_metrics = {"ensemble": {"brier_score": 0.18, "accuracy": 0.82}}

        initial_count = len(analyzer.performance_history)

        analyzer._store_performance_metrics(
            overall_metrics, agent_metrics, method_metrics
        )

        # Should have added metrics to history
        assert len(analyzer.performance_history) > initial_count

        # Check agent performance tracking
        assert "agent_1" in analyzer.agent_performance
        assert len(analyzer.agent_performance["agent_1"]) > 0

        # Check method performance tracking
        assert "ensemble" in analyzer.method_performance
        assert len(analyzer.method_performance["ensemble"]) > 0

    def test_calculate_brier_score_components(self, analyzer):
        """Test Brier score decomposition components."""
        predictions = [0.1, 0.3, 0.5, 0.7, 0.9]
        ground_truth = [False, False, True, True, True]
        base_rate = 0.6

        resolution = analyzer._calculate_resolution(
            predictions, ground_truth, base_rate
        )
        reliability = analyzer._calculate_reliability(predictions, ground_truth)

        assert resolution >= 0
        assert reliability >= 0

    def test_calculate_discrimination(self, analyzer):
        """Test discrimination calculation (AUC approximation)."""
        # Perfect discrimination case
        predictions = [0.1, 0.2, 0.8, 0.9]
        ground_truth = [False, False, True, True]

        discrimination = analyzer._calculate_discrimination(predictions, ground_truth)
        assert discrimination == 1.0  # Perfect discrimination

        # Random discrimination case
        predictions = [0.5, 0.5, 0.5, 0.5]
        ground_truth = [False, True, False, True]

        discrimination = analyzer._calculate_discrimination(predictions, ground_truth)
        assert discrimination == 0.5  # Random discrimination

    def test_confidence_correlation(self, analyzer):
        """Test confidence-accuracy correlation calculation."""
        # Create forecasts with perfect correlation
        forecasts = []
        ground_truth = []

        for i, (conf, pred, truth) in enumerate(
            [
                (0.9, 0.9, True),  # High confidence, correct
                (0.8, 0.8, True),  # High confidence, correct
                (0.3, 0.3, False),  # Low confidence, correct
                (0.2, 0.2, False),  # Low confidence, correct
            ]
        ):
            question_id = uuid4()
            research_report_id = uuid4()

            prediction = Prediction.create_binary_prediction(
                question_id=question_id,
                research_report_id=research_report_id,
                probability=pred,
                confidence=(
                    PredictionConfidence.HIGH
                    if conf > 0.7
                    else PredictionConfidence.LOW
                ),
                method=PredictionMethod.ENSEMBLE,
                reasoning="Test reasoning",
                created_by="test_agent",
            )

            forecast = Forecast.create_new(
                question_id=question_id,
                research_reports=[],
                predictions=[prediction],
                final_prediction=prediction,
            )
            # Set confidence manually for testing
            forecast.confidence_score = conf

            forecasts.append(forecast)
            ground_truth.append(truth)

        correlation = analyzer._calculate_confidence_correlation(
            forecasts, ground_truth
        )
        assert (
            correlation >= 0
        )  # Should be non-negative correlation (can be 0 if no variance)

    def test_get_performance_summary(self, analyzer):
        """Test performance summary generation."""
        # Add some test metrics
        test_metrics = [
            PerformanceMetric(
                metric_type=PerformanceMetricType.ACCURACY,
                value=0.8,
                timestamp=datetime.utcnow() - timedelta(days=1),
            ),
            PerformanceMetric(
                metric_type=PerformanceMetricType.BRIER_SCORE,
                value=0.2,
                timestamp=datetime.utcnow() - timedelta(days=2),
            ),
        ]

        analyzer.performance_history.extend(test_metrics)

        summary = analyzer.get_performance_summary(days=7)

        assert "period_days" in summary
        assert "total_metrics" in summary
        assert "metric_types" in summary
        assert summary["period_days"] == 7

    def test_get_improvement_tracking(self, analyzer):
        """Test improvement opportunity tracking."""
        # Add test opportunities
        test_opportunity = ImprovementOpportunity(
            opportunity_type=ImprovementOpportunityType.OVERCONFIDENCE,
            description="Test overconfidence",
            severity=0.8,
            affected_questions=[uuid4()],
            affected_agents=["test_agent"],
            recommended_actions=["Test action"],
            potential_impact=0.1,
            implementation_difficulty=0.5,
            timestamp=datetime.utcnow(),
        )

        analyzer.improvement_opportunities.append(test_opportunity)

        tracking = analyzer.get_improvement_tracking()

        assert "active_opportunities" in tracking
        assert "opportunities_by_type" in tracking
        assert "high_severity_count" in tracking
        assert "total_potential_impact" in tracking
        assert tracking["active_opportunities"] >= 1

    def test_empty_forecasts_handling(self, analyzer):
        """Test handling of empty forecast lists."""
        with pytest.raises(ValueError, match="Cannot analyze empty forecast list"):
            analyzer.analyze_resolved_predictions([], [])

    def test_mismatched_lengths(self, analyzer, sample_forecasts):
        """Test handling of mismatched forecast and ground truth lengths."""
        with pytest.raises(
            ValueError, match="Forecasts and ground truth must have same length"
        ):
            analyzer.analyze_resolved_predictions(sample_forecasts, [True, False])

    def test_insufficient_samples_handling(self, analyzer):
        """Test handling of insufficient samples for analysis."""
        # Create minimal forecasts
        question_id = uuid4()
        research_report_id = uuid4()

        prediction = Prediction.create_binary_prediction(
            question_id=question_id,
            research_report_id=research_report_id,
            probability=0.7,
            confidence=PredictionConfidence.MEDIUM,
            method=PredictionMethod.ENSEMBLE,
            reasoning="Test reasoning",
            created_by="test_agent",
        )

        forecast = Forecast.create_new(
            question_id=question_id,
            research_reports=[],
            predictions=[prediction],
            final_prediction=prediction,
        )

        forecasts = [forecast]
        ground_truth = [True]

        results = analyzer.analyze_resolved_predictions(forecasts, ground_truth)

        # Should still work with minimal data
        assert results["sample_size"] == 1
        assert "overall_metrics" in results

    def test_extreme_predictions_handling(self, analyzer):
        """Test handling of extreme predictions (0.0 and 1.0)."""
        question_id = uuid4()
        research_report_id = uuid4()

        forecasts = []
        for prob in [0.0, 1.0, 0.001, 0.999]:
            prediction = Prediction.create_binary_prediction(
                question_id=question_id,
                research_report_id=research_report_id,
                probability=prob,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.ENSEMBLE,
                reasoning="Test reasoning",
                created_by="test_agent",
            )

            forecast = Forecast.create_new(
                question_id=question_id,
                research_reports=[],
                predictions=[prediction],
                final_prediction=prediction,
            )
            forecasts.append(forecast)

        ground_truth = [False, True, False, True]

        results = analyzer.analyze_resolved_predictions(forecasts, ground_truth)

        # Should handle extreme values without errors
        assert "overall_metrics" in results
        assert results["overall_metrics"]["log_score"] < float("inf")

    def test_serialization_methods(self, analyzer):
        """Test serialization of analysis objects."""
        # Test opportunity serialization
        opportunity = ImprovementOpportunity(
            opportunity_type=ImprovementOpportunityType.POOR_CALIBRATION,
            description="Test description",
            severity=0.7,
            affected_questions=[uuid4(), uuid4()],
            affected_agents=["agent1", "agent2"],
            recommended_actions=["action1", "action2"],
            potential_impact=0.15,
            implementation_difficulty=0.6,
            timestamp=datetime.utcnow(),
        )

        serialized = analyzer._serialize_opportunity(opportunity)

        assert serialized["type"] == "poor_calibration"
        assert serialized["description"] == "Test description"
        assert serialized["severity"] == 0.7
        assert serialized["affected_questions_count"] == 2
        assert len(serialized["affected_agents"]) == 2

        # Test pattern serialization
        pattern = PerformancePattern(
            pattern_type="test_pattern",
            description="Test pattern description",
            frequency=0.5,
            confidence=0.8,
            affected_contexts=["context1"],
            performance_impact=-0.1,
            first_observed=datetime.utcnow() - timedelta(days=5),
            last_observed=datetime.utcnow(),
            examples=[uuid4(), uuid4()],
        )

        serialized_pattern = analyzer._serialize_pattern(pattern)

        assert serialized_pattern["type"] == "test_pattern"
        assert serialized_pattern["frequency"] == 0.5
        assert serialized_pattern["examples_count"] == 2

        # Test insight serialization
        insight = LearningInsight(
            insight_type="test_insight",
            title="Test Insight",
            description="Test insight description",
            evidence=["evidence1", "evidence2"],
            confidence=0.9,
            actionable_recommendations=["rec1", "rec2"],
            expected_improvement=0.05,
            priority=0.8,
            timestamp=datetime.utcnow(),
        )

        serialized_insight = analyzer._serialize_insight(insight)

        assert serialized_insight["type"] == "test_insight"
        assert serialized_insight["title"] == "Test Insight"
        assert len(serialized_insight["evidence"]) == 2
        assert len(serialized_insight["recommendations"]) == 2

    def test_performance_metric_types(self):
        """Test performance metric type enumeration."""
        # Ensure all expected metric types are available
        expected_types = [
            "ACCURACY",
            "CALIBRATION",
            "BRIER_SCORE",
            "LOG_SCORE",
            "RESOLUTION",
            "RELIABILITY",
            "SHARPNESS",
            "DISCRIMINATION",
        ]

        for expected_type in expected_types:
            assert hasattr(PerformanceMetricType, expected_type)

    def test_improvement_opportunity_types(self):
        """Test improvement opportunity type enumeration."""
        # Ensure all expected opportunity types are available
        expected_types = [
            "OVERCONFIDENCE",
            "UNDERCONFIDENCE",
            "POOR_CALIBRATION",
            "LOW_RESOLUTION",
            "INCONSISTENT_REASONING",
            "INSUFFICIENT_RESEARCH",
            "BIAS_DETECTION",
            "METHOD_SELECTION",
            "ENSEMBLE_WEIGHTING",
            "TIMING_OPTIMIZATION",
        ]

        for expected_type in expected_types:
            assert hasattr(ImprovementOpportunityType, expected_type)

    def test_comprehensive_analysis_workflow(
        self, analyzer, sample_forecasts, sample_ground_truth
    ):
        """Test the complete analysis workflow."""
        # Run full analysis
        results = analyzer.analyze_resolved_predictions(
            sample_forecasts, sample_ground_truth
        )

        # Verify all expected sections are present
        expected_sections = [
            "analysis_timestamp",
            "sample_size",
            "overall_metrics",
            "agent_performance",
            "method_performance",
            "calibration_analysis",
            "improvement_opportunities",
            "performance_patterns",
            "learning_insights",
            "recommendations",
        ]

        for section in expected_sections:
            assert section in results, f"Missing section: {section}"

        # Verify recommendations are prioritized
        recommendations = results["recommendations"]
        if len(recommendations) > 1:
            # Check that recommendations are sorted by priority
            priorities = [rec["priority"] for rec in recommendations]
            assert priorities == sorted(priorities, reverse=True)

        # Verify data consistency
        assert results["sample_size"] == len(sample_forecasts)
        assert isinstance(results["overall_metrics"]["brier_score"], (int, float))
        assert isinstance(
            results["calibration_analysis"]["expected_calibration_error"], (int, float)
        )
