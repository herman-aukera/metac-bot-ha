"""Tests for the performance tracking service."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
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
from src.domain.entities.research_report import (
    ResearchQuality,
    ResearchReport,
    ResearchSource,
)
from src.domain.services.performance_tracking_service import (
    AlertLevel,
    MetricType,
    PerformanceAlert,
    PerformanceMetric,
    PerformanceTrackingService,
    TournamentAnalytics,
)
from src.domain.value_objects.reasoning_trace import (
    ReasoningStep,
    ReasoningStepType,
    ReasoningTrace,
)


class TestPerformanceTrackingService:
    """Test cases for PerformanceTrackingService."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def performance_service(self, temp_storage_path):
        """Create performance tracking service for testing."""
        return PerformanceTrackingService(
            metrics_storage_path=temp_storage_path, enable_real_time_monitoring=True
        )

    @pytest.fixture
    def sample_forecast(self):
        """Create sample forecast for testing."""
        question_id = uuid4()
        research_report_id = uuid4()

        # Create research sources
        sources = [
            ResearchSource(
                url="https://example.com/source1",
                title="Source 1",
                summary="Summary 1",
                credibility_score=0.8,
            ),
            ResearchSource(
                url="https://example.com/source2",
                title="Source 2",
                summary="Summary 2",
                credibility_score=0.9,
            ),
        ]

        # Create research report
        research_report = ResearchReport.create_new(
            question_id=question_id,
            title="Sample Research Report",
            executive_summary="Sample executive summary",
            detailed_analysis="Sample detailed analysis",
            sources=sources,
            created_by="test_agent",
            quality=ResearchQuality.HIGH,
        )

        # Create predictions
        predictions = []
        for i in range(3):
            prediction = Prediction.create_binary_prediction(
                question_id=question_id,
                research_report_id=research_report_id,
                probability=0.6 + i * 0.1,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning=f"Test reasoning {i}",
                created_by=f"agent_{i}",
                reasoning_steps=[f"Step 1 for agent {i}", f"Step 2 for agent {i}"],
            )
            predictions.append(prediction)

        # Create final prediction
        final_prediction = Prediction.create_binary_prediction(
            question_id=question_id,
            research_report_id=research_report_id,
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.ENSEMBLE,
            reasoning="Ensemble prediction",
            created_by="ensemble",
        )

        # Create forecast
        forecast = Forecast.create_new(
            question_id=question_id,
            research_reports=[research_report],
            predictions=predictions,
            final_prediction=final_prediction,
            reasoning_summary="Test forecast reasoning",
            ensemble_method="confidence_weighted",
            weight_distribution={"agent_0": 0.4, "agent_1": 0.3, "agent_2": 0.3},
            consensus_strength=0.8,
        )

        return forecast

    def test_initialization(self, temp_storage_path):
        """Test service initialization."""
        service = PerformanceTrackingService(
            metrics_storage_path=temp_storage_path, enable_real_time_monitoring=True
        )

        assert service.metrics_storage_path == temp_storage_path
        assert service.enable_real_time_monitoring is True
        assert len(service.metrics_buffer) == 0
        assert len(service.alerts_buffer) == 0
        assert len(service.tournament_analytics) == 0

    def test_track_forecast_performance_basic(
        self, performance_service, sample_forecast
    ):
        """Test basic forecast performance tracking."""
        processing_time = 120.5  # seconds
        resource_usage = {"cpu_percent": 45.2, "memory_mb": 512.0}

        performance_service.track_forecast_performance(
            forecast=sample_forecast,
            processing_time=processing_time,
            resource_usage=resource_usage,
        )

        # Check that metrics were stored
        assert len(performance_service.metrics_buffer) > 0

        # Check for different metric types
        metric_types = [
            metric.metric_type for metric in performance_service.metrics_buffer
        ]
        assert MetricType.CONFIDENCE in metric_types
        assert MetricType.REASONING_QUALITY in metric_types
        assert MetricType.RESPONSE_TIME in metric_types
        assert MetricType.RESOURCE_USAGE in metric_types

    def test_track_forecast_performance_with_reasoning_traces(
        self, performance_service, sample_forecast
    ):
        """Test forecast tracking with reasoning traces."""
        # Add reasoning traces to predictions
        for prediction in sample_forecast.predictions:
            reasoning_steps = [
                ReasoningStep.create(
                    ReasoningStepType.OBSERVATION, "Test observation", confidence=0.8
                ),
                ReasoningStep.create(
                    ReasoningStepType.ANALYSIS, "Test analysis", confidence=0.7
                ),
            ]

            reasoning_trace = ReasoningTrace.create(
                question_id=sample_forecast.question_id,
                agent_id=prediction.created_by,
                reasoning_method=prediction.method.value,
                steps=reasoning_steps,
                final_conclusion="Test conclusion",
                overall_confidence=0.75,
            )

            prediction.add_reasoning_trace(reasoning_trace)

        performance_service.track_forecast_performance(sample_forecast)

        # Check that reasoning quality metrics were tracked
        quality_metrics = [
            metric
            for metric in performance_service.metrics_buffer
            if metric.metric_type == MetricType.REASONING_QUALITY
        ]
        assert len(quality_metrics) > 0

        # Quality should be higher due to reasoning traces
        quality_metric = quality_metrics[0]
        assert quality_metric.value > 0.5

    def test_track_resolved_prediction(self, performance_service, sample_forecast):
        """Test tracking resolved prediction performance."""
        actual_outcome = 1  # Question resolved to True
        resolution_timestamp = datetime.utcnow()

        metrics = performance_service.track_resolved_prediction(
            forecast=sample_forecast,
            actual_outcome=actual_outcome,
            resolution_timestamp=resolution_timestamp,
        )

        # Check returned metrics
        assert "brier_score" in metrics
        assert "log_score" in metrics
        assert "accuracy" in metrics
        assert "calibration_bin" in metrics

        # Check Brier score calculation
        prediction_prob = sample_forecast.prediction
        expected_brier = (prediction_prob - actual_outcome) ** 2
        assert abs(metrics["brier_score"] - expected_brier) < 0.001

        # Check accuracy calculation
        expected_accuracy = (
            1.0 if (prediction_prob > 0.5) == (actual_outcome == 1) else 0.0
        )
        assert metrics["accuracy"] == expected_accuracy

        # Check that metrics were stored
        brier_metrics = [
            metric
            for metric in performance_service.metrics_buffer
            if metric.metric_type == MetricType.BRIER_SCORE
        ]
        assert len(brier_metrics) > 0

    def test_performance_alerts(self, performance_service, sample_forecast):
        """Test performance alert generation."""
        # Create forecast with very low confidence to trigger alert
        sample_forecast.confidence_score = 0.1  # Very low confidence

        performance_service.track_forecast_performance(sample_forecast)

        # Check that alert was generated
        assert len(performance_service.alerts_buffer) > 0

        alert = performance_service.alerts_buffer[0]
        assert alert.level == AlertLevel.CRITICAL
        assert alert.metric_type == MetricType.CONFIDENCE
        assert "low confidence" in alert.message.lower()

    def test_performance_dashboard_data(self, performance_service, sample_forecast):
        """Test performance dashboard data generation."""
        # Track some performance data
        performance_service.track_forecast_performance(sample_forecast)

        # Add some resolved predictions
        performance_service.track_resolved_prediction(sample_forecast, actual_outcome=1)

        dashboard_data = performance_service.get_performance_dashboard_data()

        # Check dashboard structure
        assert "summary" in dashboard_data
        assert "agent_performance" in dashboard_data
        assert "recent_alerts" in dashboard_data
        assert "tournament_analytics" in dashboard_data
        assert "real_time_metrics" in dashboard_data
        assert "timestamp" in dashboard_data

        # Check summary data
        summary = dashboard_data["summary"]
        assert isinstance(summary, dict)

        # Check real-time metrics
        real_time = dashboard_data["real_time_metrics"]
        assert "metrics_count_last_hour" in real_time

    def test_tournament_analytics_update(self, performance_service):
        """Test tournament analytics update."""
        tournament_id = 12345
        ranking = 15
        total_participants = 100
        brier_scores = [0.15, 0.18, 0.12, 0.20]

        performance_service.update_tournament_analytics(
            tournament_id=tournament_id,
            ranking=ranking,
            total_participants=total_participants,
            brier_scores=brier_scores,
            questions_answered=25,
            questions_resolved=20,
            calibration_score=0.85,
        )

        # Check that analytics were stored
        assert tournament_id in performance_service.tournament_analytics

        analytics = performance_service.tournament_analytics[tournament_id]
        assert analytics.tournament_id == tournament_id
        assert analytics.current_ranking == ranking
        assert analytics.total_participants == total_participants
        assert analytics.questions_answered == 25
        assert analytics.questions_resolved == 20
        assert analytics.calibration_score == 0.85
        assert (
            abs(analytics.average_brier_score - sum(brier_scores) / len(brier_scores))
            < 0.001
        )
        assert (
            abs(
                analytics.competitive_position_percentile
                - (ranking / total_participants)
            )
            < 0.001
        )

    def test_performance_report_generation(self, performance_service, sample_forecast):
        """Test performance report generation."""
        # Track some performance data
        performance_service.track_forecast_performance(sample_forecast)
        performance_service.track_resolved_prediction(sample_forecast, actual_outcome=1)

        # Generate report
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        report = performance_service.get_performance_report(
            start_date=start_date, end_date=end_date
        )

        # Check report structure
        assert "report_period" in report
        assert "summary" in report
        assert "detailed_analysis" in report
        assert "recommendations" in report
        assert "generated_at" in report

        # Check report period
        period = report["report_period"]
        assert "start_date" in period
        assert "end_date" in period
        assert "duration_days" in period

        # Check detailed analysis
        analysis = report["detailed_analysis"]
        assert "total_metrics" in analysis
        assert "unique_questions" in analysis
        assert "unique_agents" in analysis

        # Check recommendations
        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_metrics_persistence(self, performance_service, sample_forecast):
        """Test metrics persistence to disk."""
        # Track enough performance data to trigger persistence
        for _ in range(10):
            performance_service.track_forecast_performance(sample_forecast)

        # Manually trigger persistence
        performance_service._persist_metrics()

        # Check that metrics files were created
        metrics_files = list(
            performance_service.metrics_storage_path.glob("metrics_*.json")
        )
        assert len(metrics_files) > 0

        # Check file content
        with open(metrics_files[0], "r") as f:
            metrics_data = json.load(f)

        assert isinstance(metrics_data, list)
        assert len(metrics_data) > 0

        # Check metric structure
        metric = metrics_data[0]
        assert "id" in metric
        assert "metric_type" in metric
        assert "value" in metric
        assert "timestamp" in metric

    def test_cleanup_old_data(self, performance_service, sample_forecast):
        """Test cleanup of old performance data."""
        # Add some metrics
        performance_service.track_forecast_performance(sample_forecast)

        # Manually add old metrics
        old_metric = PerformanceMetric.create(
            MetricType.CONFIDENCE, 0.5, question_id=sample_forecast.question_id
        )
        old_metric.timestamp = datetime.utcnow() - timedelta(days=35)  # 35 days old
        performance_service.metrics_buffer.append(old_metric)

        initial_count = len(performance_service.metrics_buffer)

        # Cleanup data older than 30 days
        cleanup_stats = performance_service.cleanup_old_data(days_to_keep=30)

        # Check cleanup results
        assert "metrics_removed" in cleanup_stats
        assert cleanup_stats["metrics_removed"] >= 1
        assert len(performance_service.metrics_buffer) < initial_count

    def test_agent_performance_aggregation(self, performance_service, sample_forecast):
        """Test agent performance aggregation."""
        # Track performance for multiple forecasts
        for i in range(5):
            performance_service.track_forecast_performance(sample_forecast)

        # Check agent performance aggregation
        agent_performance = performance_service.agent_performance

        # Should have data for agents from the forecast
        for prediction in sample_forecast.predictions:
            agent_id = prediction.created_by
            assert agent_id in agent_performance
            assert "confidence" in agent_performance[agent_id]
            assert "predictions" in agent_performance[agent_id]
            assert len(agent_performance[agent_id]["confidence"]) > 0

    def test_real_time_monitoring_disabled(self, temp_storage_path, sample_forecast):
        """Test service with real-time monitoring disabled."""
        service = PerformanceTrackingService(
            metrics_storage_path=temp_storage_path, enable_real_time_monitoring=False
        )

        # Track performance with very low confidence (would normally trigger alert)
        sample_forecast.confidence_score = 0.1
        service.track_forecast_performance(sample_forecast)

        # Should not generate alerts when monitoring is disabled
        assert len(service.alerts_buffer) == 0

    def test_custom_alert_thresholds(self, temp_storage_path, sample_forecast):
        """Test service with custom alert thresholds."""
        custom_thresholds = {
            MetricType.CONFIDENCE: {"warning": 0.8, "error": 0.6, "critical": 0.4}
        }

        service = PerformanceTrackingService(
            metrics_storage_path=temp_storage_path,
            enable_real_time_monitoring=True,
            alert_thresholds=custom_thresholds,
        )

        # Track performance with confidence that would trigger custom threshold
        sample_forecast.confidence_score = 0.5  # Below custom critical threshold of 0.4
        service.track_forecast_performance(sample_forecast)

        # Should use custom thresholds
        assert service.alert_thresholds[MetricType.CONFIDENCE]["critical"] == 0.4

    def test_error_handling(self, performance_service):
        """Test error handling in performance tracking."""
        # Test with invalid forecast (None)
        performance_service.track_forecast_performance(None)

        # Should not crash and should log error
        # Metrics buffer should remain empty or unchanged
        initial_count = len(performance_service.metrics_buffer)

        # Test with invalid resolved prediction data
        metrics = performance_service.track_resolved_prediction(
            forecast=None, actual_outcome=1
        )

        # Should return empty dict on error
        assert metrics == {}

        # Metrics buffer should not have increased significantly
        assert len(performance_service.metrics_buffer) <= initial_count + 1

    def test_performance_metric_creation(self):
        """Test PerformanceMetric creation and validation."""
        question_id = uuid4()

        metric = PerformanceMetric.create(
            metric_type=MetricType.BRIER_SCORE,
            value=0.15,
            question_id=question_id,
            agent_id="test_agent",
            metadata={"test_key": "test_value"},
        )

        assert metric.metric_type == MetricType.BRIER_SCORE
        assert metric.value == 0.15
        assert metric.question_id == question_id
        assert metric.agent_id == "test_agent"
        assert metric.metadata["test_key"] == "test_value"
        assert isinstance(metric.timestamp, datetime)
        assert isinstance(metric.id, type(uuid4()))

    def test_performance_alert_creation(self):
        """Test PerformanceAlert creation."""
        alert = PerformanceAlert.create(
            level=AlertLevel.WARNING,
            message="Test alert message",
            metric_type=MetricType.CONFIDENCE,
            threshold_value=0.5,
            actual_value=0.3,
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test alert message"
        assert alert.metric_type == MetricType.CONFIDENCE
        assert alert.threshold_value == 0.5
        assert alert.actual_value == 0.3
        assert alert.resolved is False
        assert isinstance(alert.timestamp, datetime)
        assert isinstance(alert.id, type(uuid4()))

    def test_reasoning_trace_preservation(self, temp_storage_path, sample_forecast):
        """Test reasoning trace preservation."""
        with patch(
            "src.domain.services.performance_tracking_service.get_reasoning_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Create service after mock is set up
            performance_service = PerformanceTrackingService(
                metrics_storage_path=temp_storage_path, enable_real_time_monitoring=True
            )

            performance_service.track_forecast_performance(sample_forecast)

            # Should have called reasoning logger for predictions and ensemble
            assert mock_logger.log_reasoning_trace.call_count >= len(
                sample_forecast.predictions
            )

    def test_trend_calculation(self, performance_service):
        """Test trend calculation for agent performance."""
        # Test improving trend
        improving_values = [0.5, 0.55, 0.6, 0.65, 0.7]
        trend = performance_service._calculate_trend(improving_values)
        assert trend == "improving"

        # Test declining trend
        declining_values = [0.7, 0.65, 0.6, 0.55, 0.5]
        trend = performance_service._calculate_trend(declining_values)
        assert trend == "declining"

        # Test stable trend
        stable_values = [0.6, 0.61, 0.59, 0.6, 0.6]
        trend = performance_service._calculate_trend(stable_values)
        assert trend == "stable"

        # Test insufficient data
        insufficient_values = [0.5, 0.6]
        trend = performance_service._calculate_trend(insufficient_values)
        assert trend == "insufficient_data"
