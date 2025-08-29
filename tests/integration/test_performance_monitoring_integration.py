"""
Integration tests for performance monitoring and analytics components.
Tests the complete monitoring pipeline and analytics functionality.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.infrastructure.monitoring.integrated_monitoring_service import (
    IntegratedMonitoringService,
)
from src.infrastructure.monitoring.model_performance_tracker import (
    ModelPerformanceTracker,
    ModelSelectionRecord,
)
from src.infrastructure.monitoring.optimization_analytics import OptimizationAnalytics
from src.infrastructure.monitoring.performance_tracker import PerformanceTracker


class TestModelPerformanceTracker:
    """Test model performance tracking functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.tracker = ModelPerformanceTracker()
        self.tracker.selection_records = []  # Clear any existing records

    def test_record_model_selection(self):
        """Test recording model selection decisions."""
        record = self.tracker.record_model_selection(
            question_id="test-123",
            task_type="forecast",
            selected_model="openai/gpt-5",
            selected_tier="full",
            routing_rationale="Complex forecasting task requires maximum reasoning",
            estimated_cost=0.05,
            operation_mode="normal",
            budget_remaining=75.0,
        )

        assert record.question_id == "test-123"
        assert record.task_type == "forecast"
        assert record.selected_model == "openai/gpt-5"
        assert record.selected_tier == "full"
        assert record.estimated_cost == 0.05
        assert record.operation_mode == "normal"
        assert record.budget_remaining == 75.0

        assert len(self.tracker.selection_records) == 1

    def test_update_selection_outcome(self):
        """Test updating selection records with actual outcomes."""
        # First record a selection
        self.tracker.record_model_selection(
            question_id="test-456",
            task_type="research",
            selected_model="openai/gpt-5-mini",
            selected_tier="mini",
            routing_rationale="Research synthesis task",
            estimated_cost=0.02,
        )

        # Then update with outcome
        success = self.tracker.update_selection_outcome(
            question_id="test-456",
            actual_cost=0.025,
            execution_time=45.5,
            quality_score=0.85,
            success=True,
            fallback_used=False,
        )

        assert success is True

        # Check the record was updated
        record = self.tracker.selection_records[0]
        assert record.actual_cost == 0.025
        assert record.execution_time == 45.5
        assert record.quality_score == 0.85
        assert record.success is True
        assert record.fallback_used is False

    def test_cost_breakdown_calculation(self):
        """Test cost breakdown calculation."""
        # Add multiple records with different tiers and tasks
        test_records = [
            ("q1", "forecast", "openai/gpt-5", "full", 0.05, 0.055, 0.8),
            ("q2", "research", "openai/gpt-5-mini", "mini", 0.02, 0.022, 0.75),
            ("q3", "validation", "openai/gpt-5-nano", "nano", 0.005, 0.006, 0.7),
            ("q4", "forecast", "openai/gpt-5", "full", 0.05, 0.048, 0.85),
        ]

        for qid, task, model, tier, est_cost, actual_cost, quality in test_records:
            self.tracker.record_model_selection(
                qid, task, model, tier, "test", est_cost
            )
            self.tracker.update_selection_outcome(
                qid, actual_cost, 30.0, quality, True, False
            )

        breakdown = self.tracker.get_cost_breakdown(24)

        assert breakdown.question_count == 4
        assert breakdown.total_cost == pytest.approx(0.131, rel=1e-3)
        assert breakdown.avg_cost_per_question == pytest.approx(0.03275, rel=1e-3)

        # Check tier breakdown
        assert "full" in breakdown.by_tier
        assert "mini" in breakdown.by_tier
        assert "nano" in breakdown.by_tier

        assert breakdown.by_tier["full"]["count"] == 2
        assert breakdown.by_tier["mini"]["count"] == 1
        assert breakdown.by_tier["nano"]["count"] == 1

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        # Add records with varying quality scores
        test_records = [
            ("q1", 0.05, 30.0, 0.8, True, False),
            ("q2", 0.02, 25.0, 0.75, True, False),
            ("q3", 0.01, 20.0, 0.9, True, True),  # Fallback used
            ("q4", 0.03, 35.0, None, False, False),  # Failed
        ]

        for i, (qid, cost, time, quality, success, fallback) in enumerate(test_records):
            self.tracker.record_model_selection(
                qid, "test", "test-model", "mini", "test", cost
            )
            self.tracker.update_selection_outcome(
                qid, cost, time, quality, success, fallback
            )

        metrics = self.tracker.get_quality_metrics(24)

        assert metrics.avg_quality_score == pytest.approx(
            0.816667, rel=1e-3
        )  # (0.8+0.75+0.9)/3
        assert metrics.success_rate == 0.75  # 3/4
        assert metrics.fallback_rate == 0.25  # 1/4
        assert metrics.avg_execution_time == 27.5  # (30+25+20+35)/4

    def test_tournament_competitiveness_indicators(self):
        """Test tournament competitiveness calculation."""
        # Add records for competitiveness analysis
        for i in range(10):
            qid = f"comp-test-{i}"
            self.tracker.record_model_selection(
                qid, "forecast", "test-model", "mini", "test", 0.02
            )
            self.tracker.update_selection_outcome(qid, 0.02, 30.0, 0.8, True, False)

        indicators = self.tracker.get_tournament_competitiveness_indicators(100.0, 24)

        assert indicators.cost_efficiency_score == 50.0  # 10 questions / 0.2 total cost
        assert (
            indicators.quality_efficiency_score == 40.0
        )  # 0.8 quality / 0.02 avg cost
        assert indicators.budget_utilization_rate == 0.2  # 0.2 / 100.0
        assert indicators.projected_questions_remaining == 4990  # (100-0.2) / 0.02
        assert indicators.competitiveness_level in [
            "excellent",
            "good",
            "concerning",
            "critical",
        ]


class TestOptimizationAnalytics:
    """Test optimization analytics functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.tracker = ModelPerformanceTracker()
        self.tracker.selection_records = []
        self.analytics = OptimizationAnalytics(self.tracker)

    def test_cost_effectiveness_analysis(self):
        """Test cost-effectiveness analysis."""
        # Add records with different tier efficiencies
        test_data = [
            ("nano", 0.005, 5),  # Very efficient
            ("mini", 0.02, 3),  # Moderately efficient
            ("full", 0.05, 2),  # Less efficient but higher quality
        ]

        for tier, cost, count in test_data:
            for i in range(count):
                qid = f"{tier}-{i}"
                self.tracker.record_model_selection(
                    qid, "test", f"model-{tier}", tier, "test", cost
                )
                self.tracker.update_selection_outcome(qid, cost, 30.0, 0.8, True, False)

        analysis = self.analytics.analyze_cost_effectiveness(24)

        assert analysis.overall_efficiency > 0
        assert "nano" in analysis.tier_efficiency
        assert "mini" in analysis.tier_efficiency
        assert "full" in analysis.tier_efficiency

        # Nano should be most efficient
        assert analysis.tier_efficiency["nano"] > analysis.tier_efficiency["mini"]
        assert analysis.tier_efficiency["mini"] > analysis.tier_efficiency["full"]

    def test_performance_correlation_analysis(self):
        """Test performance correlation analysis."""
        # Add records with cost-quality correlation
        test_records = [
            (0.005, 0.6),  # Low cost, low quality
            (0.02, 0.75),  # Medium cost, medium quality
            (0.05, 0.9),  # High cost, high quality
            (0.08, 0.95),  # Very high cost, very high quality
        ]

        for i, (cost, quality) in enumerate(test_records):
            qid = f"corr-{i}"
            self.tracker.record_model_selection(
                qid, "test", "test-model", "test", "test", cost
            )
            self.tracker.update_selection_outcome(qid, cost, 30.0, quality, True, False)

        # Add more records to meet minimum sample requirement
        for i in range(20):
            qid = f"extra-{i}"
            cost = 0.02 + (i * 0.001)
            quality = 0.7 + (i * 0.01)
            self.tracker.record_model_selection(
                qid, "test", "test-model", "test", "test", cost
            )
            self.tracker.update_selection_outcome(qid, cost, 30.0, quality, True, False)

        analysis = self.analytics.analyze_performance_correlations(24)

        # Should detect positive correlation between cost and quality
        assert analysis.cost_quality_correlation > 0.3

    def test_tournament_phase_strategy_generation(self):
        """Test tournament phase strategy generation."""
        # Test early phase strategy
        early_strategy = self.analytics.generate_tournament_phase_strategy(15.0, 100.0)
        assert early_strategy.phase == "early"
        assert early_strategy.risk_tolerance == "aggressive"
        assert early_strategy.budget_allocation_strategy["full"] > 30.0

        # Test middle phase strategy
        middle_strategy = self.analytics.generate_tournament_phase_strategy(45.0, 100.0)
        assert middle_strategy.phase == "middle"
        assert middle_strategy.risk_tolerance == "balanced"

        # Test late phase strategy
        late_strategy = self.analytics.generate_tournament_phase_strategy(75.0, 100.0)
        assert late_strategy.phase == "late"
        assert late_strategy.risk_tolerance == "conservative"

        # Test final phase strategy
        final_strategy = self.analytics.generate_tournament_phase_strategy(95.0, 100.0)
        assert final_strategy.phase == "final"
        assert final_strategy.risk_tolerance == "conservative"
        assert final_strategy.budget_allocation_strategy["nano"] > 50.0

    def test_budget_optimization_suggestions(self):
        """Test budget optimization suggestions."""
        # Add records with suboptimal allocation
        inefficient_records = [
            ("full", 0.08, 2),  # Expensive tier used frequently
            ("mini", 0.02, 1),  # Efficient tier used less
            ("nano", 0.005, 1),  # Most efficient tier used least
        ]

        for tier, cost, count in inefficient_records:
            for i in range(count):
                qid = f"opt-{tier}-{i}"
                self.tracker.record_model_selection(
                    qid, "test", f"model-{tier}", tier, "test", cost
                )
                self.tracker.update_selection_outcome(qid, cost, 30.0, 0.8, True, False)

        suggestions = self.analytics.generate_budget_optimization_suggestions(100.0, 24)

        assert suggestions.potential_savings >= 0
        assert suggestions.additional_questions_possible >= 0
        assert len(suggestions.implementation_steps) > 0
        assert suggestions.risk_assessment in ["low", "medium", "high"]


class TestIntegratedMonitoringService:
    """Test integrated monitoring service functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Create fresh instances for each test
        model_tracker = ModelPerformanceTracker()
        model_tracker.selection_records = []
        analytics = OptimizationAnalytics(model_tracker)
        perf_tracker = PerformanceTracker()
        perf_tracker.forecast_records = []

        self.monitoring = IntegratedMonitoringService(
            model_tracker=model_tracker,
            analytics=analytics,
            perf_tracker=perf_tracker,
            cost_monitor=None,
        )

    def test_record_model_usage(self):
        """Test recording model usage."""
        self.monitoring.record_model_usage(
            question_id="integrated-test-1",
            task_type="forecast",
            selected_model="openai/gpt-5",
            selected_tier="full",
            routing_rationale="Complex analysis required",
            estimated_cost=0.05,
            operation_mode="normal",
            budget_remaining=80.0,
        )

        # Check that the record was created
        records = self.monitoring.model_tracker.selection_records
        assert len(records) == 1
        assert records[0].question_id == "integrated-test-1"

    def test_record_execution_outcome(self):
        """Test recording execution outcomes."""
        # First record usage
        self.monitoring.record_model_usage(
            question_id="integrated-test-2",
            task_type="research",
            selected_model="openai/gpt-5-mini",
            selected_tier="mini",
            routing_rationale="Research synthesis",
            estimated_cost=0.02,
        )

        # Then record outcome
        self.monitoring.record_execution_outcome(
            question_id="integrated-test-2",
            actual_cost=0.022,
            execution_time=35.0,
            quality_score=0.85,
            success=True,
            fallback_used=False,
            forecast_value=0.75,
            confidence=0.8,
        )

        # Check that both trackers were updated
        # Find the record for integrated-test-2
        model_record = None
        for record in self.monitoring.model_tracker.selection_records:
            if record.question_id == "integrated-test-2":
                model_record = record
                break

        assert model_record is not None
        assert model_record.actual_cost == 0.022
        assert model_record.quality_score == 0.85

    def test_comprehensive_status_generation(self):
        """Test comprehensive status generation."""
        # Add some test data
        for i in range(5):
            qid = f"status-test-{i}"
            self.monitoring.record_model_usage(
                question_id=qid,
                task_type="forecast",
                selected_model="openai/gpt-5-mini",
                selected_tier="mini",
                routing_rationale="Test",
                estimated_cost=0.02,
            )
            self.monitoring.record_execution_outcome(
                question_id=qid,
                actual_cost=0.02,
                execution_time=30.0,
                quality_score=0.8,
                success=True,
            )

        status = self.monitoring.get_comprehensive_status(100.0)

        assert status.overall_health in ["excellent", "good", "concerning", "critical"]
        assert "budget" in status.budget_status
        assert "cost_breakdown" in status.performance_metrics
        assert isinstance(status.optimization_recommendations, list)
        assert isinstance(status.active_alerts, list)

    def test_strategic_recommendations_generation(self):
        """Test strategic recommendations generation."""
        recommendations = self.monitoring.generate_strategic_recommendations(
            budget_used_percentage=45.0, total_budget=100.0
        )

        assert "tournament_phase_strategy" in recommendations
        assert "budget_optimization" in recommendations
        assert "effectiveness_trends" in recommendations
        assert "implementation_priority" in recommendations

        # Check that phase strategy is appropriate for middle phase
        phase_strategy = recommendations["tournament_phase_strategy"]
        assert phase_strategy["phase"] == "middle"

    def test_alert_checking(self):
        """Test alert checking functionality."""
        # Add data that should trigger alerts
        for i in range(10):
            qid = f"alert-test-{i}"
            self.monitoring.record_model_usage(
                question_id=qid,
                task_type="forecast",
                selected_model="openai/gpt-5",
                selected_tier="full",
                routing_rationale="Expensive test",
                estimated_cost=0.1,  # High cost
            )
            self.monitoring.record_execution_outcome(
                question_id=qid,
                actual_cost=0.1,
                execution_time=60.0,
                quality_score=0.5,  # Low quality
                success=True,
            )

        alerts = self.monitoring.check_alerts_and_thresholds()

        # Should have some alerts due to low efficiency
        assert isinstance(alerts, list)

    @patch("threading.Thread")
    def test_monitoring_service_lifecycle(self, mock_thread):
        """Test monitoring service start/stop lifecycle."""
        # Test starting monitoring
        self.monitoring.start_monitoring()
        assert self.monitoring._is_running is True
        mock_thread.assert_called_once()

        # Test stopping monitoring
        self.monitoring.stop_monitoring()
        assert self.monitoring._is_running is False

    def test_export_monitoring_data(self):
        """Test monitoring data export."""
        # Add some test data
        self.monitoring.record_model_usage(
            question_id="export-test",
            task_type="forecast",
            selected_model="openai/gpt-5-mini",
            selected_tier="mini",
            routing_rationale="Export test",
            estimated_cost=0.02,
        )
        self.monitoring.record_execution_outcome(
            question_id="export-test",
            actual_cost=0.02,
            execution_time=30.0,
            quality_score=0.8,
            success=True,
        )

        export_data = self.monitoring.export_monitoring_data(24)

        assert "comprehensive_status" in export_data
        assert "model_effectiveness_trends" in export_data
        assert "cost_breakdown" in export_data
        assert "quality_metrics" in export_data
        assert "optimization_analysis" in export_data
        assert "alert_history" in export_data
        assert "export_timestamp" in export_data


@pytest.mark.asyncio
async def test_monitoring_integration_workflow():
    """Test complete monitoring workflow integration."""
    # Create fresh instances for this test
    model_tracker = ModelPerformanceTracker()
    model_tracker.selection_records = []
    analytics = OptimizationAnalytics(model_tracker)
    perf_tracker = PerformanceTracker()
    perf_tracker.forecast_records = []

    monitoring = IntegratedMonitoringService(
        model_tracker=model_tracker,
        analytics=analytics,
        perf_tracker=perf_tracker,
        cost_monitor=None,
    )

    # Simulate a complete question processing workflow
    question_id = "workflow-test"

    # 1. Record model selection
    monitoring.record_model_usage(
        question_id=question_id,
        task_type="forecast",
        selected_model="openai/gpt-5",
        selected_tier="full",
        routing_rationale="Complex forecasting requires maximum reasoning capability",
        estimated_cost=0.05,
        operation_mode="normal",
        budget_remaining=85.0,
    )

    # 2. Simulate processing time
    await asyncio.sleep(0.1)

    # 3. Record execution outcome
    monitoring.record_execution_outcome(
        question_id=question_id,
        actual_cost=0.048,
        execution_time=45.5,
        quality_score=0.92,
        success=True,
        fallback_used=False,
        forecast_value=0.75,
        confidence=0.85,
    )

    # 4. Get comprehensive status
    status = monitoring.get_comprehensive_status(100.0)

    # 5. Generate strategic recommendations
    recommendations = monitoring.generate_strategic_recommendations(15.0, 100.0)

    # 6. Check for alerts
    alerts = monitoring.check_alerts_and_thresholds()

    # Verify the workflow completed successfully
    assert len(monitoring.model_tracker.selection_records) == 1
    assert status.overall_health in ["excellent", "good", "concerning", "critical"]
    assert recommendations["tournament_phase_strategy"]["phase"] == "early"
    assert isinstance(alerts, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
