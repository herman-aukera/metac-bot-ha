"""
Integration tests for the complete monitoring pipeline and alert workflows.

Tests the integration between metrics collection, health monitoring,
distributed tracing, dashboards, and alerting systems.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.infrastructure.monitoring.metrics_collector import (
    MetricsCollector, configure_metrics
)
from src.infrastructure.monitoring.health_check_manager import (
    HealthCheckManager, HealthCheck, HealthStatus, HealthCheckResult,
    configure_health_monitoring
)
from src.infrastructure.monitoring.distributed_tracing import (
    DistributedTracer, SpanKind, SpanStatus, configure_tracing
)
from src.infrastructure.monitoring.dashboard import (
    PerformanceDashboard, AlertManager, AlertSeverity, configure_dashboard
)
from src.infrastructure.monitoring.reasoning_trace import (
    ReasoningTraceManager, ReasoningStepType, configure_reasoning_traces
)
from src.infrastructure.monitoring.performance_benchmarking import (
    PerformanceBenchmarkManager, BenchmarkType, configure_benchmarking
)


class TestHealthCheck(HealthCheck):
    """Test health check implementation."""

    def __init__(self, component_name: str, should_fail: bool = False):
        super().__init__(component_name)
        self.should_fail = should_fail

    async def check_health(self) -> HealthCheckResult:
        """Mock health check."""
        if self.should_fail:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.UNHEALTHY,
                message="Test failure",
                details={"test": True}
            )
        else:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.HEALTHY,
                message="Test success",
                details={"test": True}
            )


@pytest.fixture
def metrics_collector():
    """Create metrics collector for testing."""
    return configure_metrics(enable_prometheus=False)


@pytest.fixture
def health_manager():
    """Create health check manager for testing."""
    return configure_health_monitoring(check_interval=1.0, auto_start=False)


@pytest.fixture
def tracer():
    """Create distributed tracer for testing."""
    return configure_tracing(max_traces=100, trace_retention_hours=1)


@pytest.fixture
def dashboard():
    """Create performance dashboard for testing."""
    return configure_dashboard(auto_start_monitoring=False)


@pytest.fixture
def reasoning_trace_manager():
    """Create reasoning trace manager for testing."""
    return configure_reasoning_traces(max_traces=100, retention_days=1)


@pytest.fixture
def benchmark_manager():
    """Create benchmark manager for testing."""
    return configure_benchmarking(regression_threshold=5.0)


class TestMetricsIntegration:
    """Test metrics collection integration."""

    def test_metrics_collection_basic(self, metrics_collector):
        """Test basic metrics collection."""
        # Test counter
        metrics_collector.increment_counter("test_counter", 1.0, {"type": "test"})

        # Test gauge
        metrics_collector.set_gauge("test_gauge", 42.0, {"component": "test"})

        # Test histogram
        metrics_collector.observe_histogram("test_histogram", 1.5, {"operation": "test"})

        # Verify metrics are recorded
        summary = metrics_collector.get_metric_summary("test_counter")
        assert "test_counter{type=test}" in summary

        summary = metrics_collector.get_metric_summary("test_gauge")
        assert "test_gauge{component=test}" in summary

        summary = metrics_collector.get_metric_summary("test_histogram")
        assert "test_histogram{operation=test}" in summary

    def test_forecasting_metrics_recording(self, metrics_collector):
        """Test forecasting-specific metrics recording."""
        metrics_collector.record_forecasting_request(
            question_type="binary",
            agent_type="cot",
            status="success",
            duration=2.5,
            accuracy=0.85,
            confidence=0.9
        )

        # Verify all metrics are recorded
        summary = metrics_collector.get_metric_summary("forecasting_requests_total")
        assert summary

        summary = metrics_collector.get_metric_summary("forecasting_duration_seconds")
        assert summary

        summary = metrics_collector.get_metric_summary("prediction_accuracy")
        assert summary

        summary = metrics_collector.get_metric_summary("prediction_confidence")
        assert summary

    def test_api_metrics_recording(self, metrics_collector):
        """Test API metrics recording."""
        metrics_collector.record_api_request(
            service="asknews",
            endpoint="/search",
            status="success",
            duration=0.5
        )

        summary = metrics_collector.get_metric_summary("api_requests_total")
        assert summary

        summary = metrics_collector.get_metric_summary("api_request_duration_seconds")
        assert summary

    def test_tournament_metrics_recording(self, metrics_collector):
        """Test tournament metrics recording."""
        metrics_collector.record_tournament_metrics(
            tournament_id="test_tournament",
            ranking_position=5,
            score=85.5,
            questions_processed=10
        )

        summary = metrics_collector.get_metric_summary("tournament_ranking_position")
        assert summary

        summary = metrics_collector.get_metric_summary("tournament_score")
        assert summary

        summary = metrics_collector.get_metric_summary("tournament_questions_processed")
        assert summary


class TestHealthMonitoringIntegration:
    """Test health monitoring integration."""

    @pytest.mark.asyncio
    async def test_health_check_registration_and_execution(self, health_manager):
        """Test health check registration and execution."""
        # Register test health checks
        healthy_check = TestHealthCheck("test_healthy", should_fail=False)
        unhealthy_check = TestHealthCheck("test_unhealthy", should_fail=True)

        health_manager.register_health_check(healthy_check)
        health_manager.register_health_check(unhealthy_check)

        # Run health checks
        summary = await health_manager.check_all_health()

        # Verify results
        assert summary.healthy_components >= 1
        assert summary.unhealthy_components >= 1
        assert len(summary.component_results) >= 2

        # Check individual results
        healthy_result = health_manager.get_component_status("test_healthy")
        assert healthy_result.status == HealthStatus.HEALTHY

        unhealthy_result = health_manager.get_component_status("test_unhealthy")
        assert unhealthy_result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_monitoring_background_task(self, health_manager):
        """Test background health monitoring."""
        # Register a test health check
        test_check = TestHealthCheck("background_test")
        health_manager.register_health_check(test_check)

        # Start monitoring
        await health_manager.start_monitoring()

        # Wait for at least one monitoring cycle
        await asyncio.sleep(1.5)

        # Check that health check was executed
        result = health_manager.get_component_status("background_test")
        assert result is not None

        # Stop monitoring
        await health_manager.stop_monitoring()


class TestDistributedTracingIntegration:
    """Test distributed tracing integration."""

    @pytest.mark.asyncio
    async def test_trace_creation_and_spans(self, tracer):
        """Test trace creation and span management."""
        # Start a trace
        root_span = tracer.start_trace("test_operation", tags={"test": "true"})

        # Add child spans
        child_span1 = tracer.start_span("child_operation_1", kind=SpanKind.INTERNAL)
        child_span1.set_tag("step", "1")
        child_span1.log("Processing step 1")
        tracer.finish_span(child_span1, SpanStatus.OK)

        child_span2 = tracer.start_span("child_operation_2", kind=SpanKind.CLIENT)
        child_span2.set_tag("step", "2")
        child_span2.add_event("API call started")
        tracer.finish_span(child_span2, SpanStatus.OK)

        # Finish root span
        tracer.finish_span(root_span, SpanStatus.OK)

        # Verify trace structure
        trace = tracer.get_trace(root_span.trace_id)
        assert trace is not None
        assert len(trace.spans) == 3
        assert trace.root_span == root_span

        # Verify span relationships
        child_spans = trace.get_child_spans(root_span.span_id)
        assert len(child_spans) == 2

    @pytest.mark.asyncio
    async def test_trace_search_functionality(self, tracer):
        """Test trace search capabilities."""
        # Create multiple traces
        for i in range(5):
            span = tracer.start_trace(f"operation_{i}", tags={"batch": "test", "index": str(i)})
            tracer.finish_span(span, SpanStatus.OK)

        # Search by operation name
        results = tracer.search_traces(operation_name="operation_2", limit=10)
        assert len(results) == 1
        assert results[0].root_span.operation_name == "operation_2"

        # Search by tags
        results = tracer.search_traces(tags={"batch": "test"}, limit=10)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_trace_context_manager(self, tracer):
        """Test trace context manager functionality."""
        with tracer.trace_operation("context_test", tags={"method": "context_manager"}) as span:
            span.log("Inside context manager")
            span.set_tag("processed", True)

        # Verify span was created and finished
        trace = tracer.get_trace(span.trace_id)
        assert trace is not None
        assert span.end_time is not None
        assert span.status == SpanStatus.OK


class TestDashboardIntegration:
    """Test dashboard and alerting integration."""

    def test_dashboard_data_collection(self, dashboard, metrics_collector):
        """Test dashboard data collection from metrics."""
        # Generate some test metrics
        metrics_collector.record_forecasting_request(
            question_type="binary",
            agent_type="cot",
            status="success",
            duration=1.5,
            accuracy=0.8,
            confidence=0.9
        )

        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data("forecasting_performance")

        assert dashboard_data["id"] == "forecasting_performance"
        assert "widgets" in dashboard_data
        assert len(dashboard_data["widgets"]) > 0

        # Check widget data
        for widget in dashboard_data["widgets"]:
            assert "data" in widget
            assert "timestamp" in widget

    def test_alert_rule_evaluation(self, dashboard):
        """Test alert rule evaluation."""
        alert_manager = dashboard.alert_manager

        # Mock metrics that should trigger alerts
        with patch.object(alert_manager.metrics_collector, 'get_metric_summary') as mock_summary:
            mock_summary.return_value = {
                "forecasting_duration_seconds": {
                    "mean": 65.0  # Above 60s threshold
                }
            }

            # Evaluate rules
            alert_manager.evaluate_rules()

            # Check if alert was triggered
            active_alerts = alert_manager.get_active_alerts()
            slow_forecasting_alerts = [a for a in active_alerts if "Slow Forecasting" in a.name]
            assert len(slow_forecasting_alerts) > 0

    def test_alert_lifecycle(self, dashboard):
        """Test complete alert lifecycle."""
        alert_manager = dashboard.alert_manager

        # Create a test alert rule
        from src.infrastructure.monitoring.dashboard import AlertRule

        test_rule = AlertRule(
            name="Test Alert",
            metric_name="test_metric",
            condition="greater_than",
            threshold=10.0,
            severity=AlertSeverity.HIGH,
            description="Test alert for integration testing"
        )

        alert_manager.add_alert_rule(test_rule)

        # Mock metrics to trigger alert
        with patch.object(alert_manager.metrics_collector, 'get_metric_summary') as mock_summary:
            mock_summary.return_value = {
                "test_metric": {
                    "current_value": 15.0  # Above threshold
                }
            }

            # Trigger alert
            alert_manager.evaluate_rules()
            active_alerts = alert_manager.get_active_alerts()
            assert len(active_alerts) > 0

            test_alert = next((a for a in active_alerts if a.name == "Test Alert"), None)
            assert test_alert is not None
            assert test_alert.current_value == 15.0

            # Acknowledge alert
            alert_manager.acknowledge_alert(test_alert.id)
            updated_alert = alert_manager.active_alerts[test_alert.id]
            assert updated_alert.acknowledged_at is not None

            # Resolve alert (mock metrics below threshold)
            mock_summary.return_value = {
                "test_metric": {
                    "current_value": 5.0  # Below threshold
                }
            }

            alert_manager.evaluate_rules()
            active_alerts = alert_manager.get_active_alerts()
            test_alert_active = next((a for a in active_alerts if a.name == "Test Alert"), None)
            assert test_alert_active is None  # Should be resolved


class TestReasoningTraceIntegration:
    """Test reasoning trace integration."""

    def test_reasoning_trace_lifecycle(self, reasoning_trace_manager):
        """Test complete reasoning trace lifecycle."""
        # Start trace
        trace_id = reasoning_trace_manager.start_trace(
            question_id="test_question",
            agent_id="test_agent",
            agent_type="cot",
            operation_type="forecast",
            metadata={"test": True}
        )

        # Add reasoning steps
        step1_id = reasoning_trace_manager.add_reasoning_step(
            trace_id=trace_id,
            step_type=ReasoningStepType.PROBLEM_ANALYSIS,
            description="Analyzing the problem",
            input_data={"question": "test question"},
            output_data={"analysis": "problem breakdown"},
            confidence_level=0.8,
            confidence_basis="Clear problem statement",
            duration_ms=100.0
        )

        step2_id = reasoning_trace_manager.add_reasoning_step(
            trace_id=trace_id,
            step_type=ReasoningStepType.PREDICTION_GENERATION,
            description="Generating prediction",
            input_data={"analysis": "problem breakdown"},
            output_data={"prediction": 0.75},
            confidence_level=0.9,
            confidence_basis="Strong evidence",
            duration_ms=200.0
        )

        # Finish trace
        reasoning_trace_manager.finish_trace(
            trace_id=trace_id,
            success=True,
            final_result={"prediction": 0.75, "confidence": 0.9}
        )

        # Verify trace
        trace = reasoning_trace_manager.get_trace(trace_id)
        assert trace is not None
        assert len(trace.steps) == 2
        assert trace.success is True
        assert trace.final_result["prediction"] == 0.75

    def test_reasoning_trace_search(self, reasoning_trace_manager):
        """Test reasoning trace search functionality."""
        # Create multiple traces
        for i in range(3):
            trace_id = reasoning_trace_manager.start_trace(
                question_id=f"question_{i}",
                agent_id=f"agent_{i}",
                agent_type="cot" if i % 2 == 0 else "react",
                operation_type="forecast"
            )

            reasoning_trace_manager.add_reasoning_step(
                trace_id=trace_id,
                step_type=ReasoningStepType.PROBLEM_ANALYSIS,
                description=f"Analysis for question {i}",
                input_data={"question": f"question_{i}"},
                output_data={"analysis": f"analysis_{i}"},
                confidence_level=0.7 + (i * 0.1),
                confidence_basis="Test basis",
                duration_ms=100.0
            )

            reasoning_trace_manager.finish_trace(trace_id, success=True)

        # Search by agent type
        cot_traces = reasoning_trace_manager.search_traces(agent_type="cot")
        assert len(cot_traces) == 2

        # Search by confidence
        high_confidence_traces = reasoning_trace_manager.search_traces(min_confidence=0.8)
        assert len(high_confidence_traces) >= 1

        # Search by question
        specific_traces = reasoning_trace_manager.search_traces(question_id="question_1")
        assert len(specific_traces) == 1

    def test_reasoning_patterns_analysis(self, reasoning_trace_manager):
        """Test reasoning patterns analysis."""
        # Create traces with different patterns
        for agent_type in ["cot", "react", "tot"]:
            for i in range(2):
                trace_id = reasoning_trace_manager.start_trace(
                    question_id=f"question_{agent_type}_{i}",
                    agent_id=f"agent_{agent_type}",
                    agent_type=agent_type,
                    operation_type="forecast"
                )

                reasoning_trace_manager.add_reasoning_step(
                    trace_id=trace_id,
                    step_type=ReasoningStepType.PROBLEM_ANALYSIS,
                    description="Analysis step",
                    input_data={},
                    output_data={},
                    confidence_level=0.8,
                    confidence_basis="Test",
                    duration_ms=100.0
                )

                reasoning_trace_manager.finish_trace(trace_id, success=True)

        # Analyze patterns
        patterns = reasoning_trace_manager.get_reasoning_patterns()
        assert patterns["total_traces"] == 6
        assert patterns["success_rate"] == 1.0
        assert "step_type_distribution" in patterns

        # Analyze patterns by agent type
        cot_patterns = reasoning_trace_manager.get_reasoning_patterns(agent_type="cot")
        assert cot_patterns["total_traces"] == 2


class TestBenchmarkingIntegration:
    """Test performance benchmarking integration."""

    @pytest.mark.asyncio
    async def test_resource_usage_benchmark(self, benchmark_manager):
        """Test resource usage benchmark."""
        result = await benchmark_manager.run_benchmark("resource_usage", duration_seconds=1)

        assert result is not None
        assert result.benchmark_name == "resource_usage"
        assert result.benchmark_type == BenchmarkType.RESOURCE_USAGE
        assert result.value >= 0.0
        assert "duration_seconds" in result.metadata

    def test_baseline_management(self, benchmark_manager):
        """Test benchmark baseline management."""
        # Set a baseline
        benchmark_manager.set_baseline("resource_usage", 25.0)

        # Verify baseline is set
        assert benchmark_manager.baselines["resource_usage"] == 25.0

        # Check benchmark has baseline
        benchmark = benchmark_manager.benchmarks["resource_usage"]
        assert benchmark.baseline_value == 25.0

    @pytest.mark.asyncio
    async def test_regression_detection(self, benchmark_manager):
        """Test performance regression detection."""
        # Set a low baseline
        benchmark_manager.set_baseline("resource_usage", 10.0)

        # Mock high resource usage to trigger regression
        with patch('psutil.cpu_percent', return_value=80.0), \
             patch('psutil.virtual_memory') as mock_memory:

            mock_memory.return_value.percent = 90.0

            result = await benchmark_manager.run_benchmark("resource_usage", duration_seconds=1)

            # Should detect regression
            assert result.regression_percentage > benchmark_manager.regression_threshold

            # Check if regression was recorded
            active_regressions = benchmark_manager.active_regressions
            resource_regressions = [r for r in active_regressions.values()
                                  if r.benchmark_name == "resource_usage"]
            assert len(resource_regressions) > 0

    def test_performance_trends(self, benchmark_manager):
        """Test performance trend analysis."""
        # Add some mock results to history
        from src.infrastructure.monitoring.performance_benchmarking import BenchmarkResult

        benchmark_name = "resource_usage"

        # Create mock results with trend
        for i in range(10):
            result = BenchmarkResult(
                id=f"test_{i}",
                benchmark_name=benchmark_name,
                benchmark_type=BenchmarkType.RESOURCE_USAGE,
                value=10.0 + i,  # Increasing trend
                unit="percentage",
                timestamp=datetime.utcnow() - timedelta(hours=i)
            )
            benchmark_manager.results_history[benchmark_name].append(result)

        # Get trends
        trends = benchmark_manager.get_benchmark_trends(benchmark_name)

        assert trends["benchmark_name"] == benchmark_name
        assert trends["sample_count"] == 10
        assert trends["trend_direction"] == "increasing"
        assert trends["min_value"] == 10.0
        assert trends["max_value"] == 19.0


class TestEndToEndMonitoringPipeline:
    """Test complete end-to-end monitoring pipeline."""

    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(
        self,
        metrics_collector,
        health_manager,
        tracer,
        dashboard,
        reasoning_trace_manager
    ):
        """Test complete monitoring workflow integration."""

        # 1. Start a distributed trace for a forecasting operation
        root_span = tracer.start_trace("forecast_operation", tags={"test": "e2e"})

        # 2. Start reasoning trace
        reasoning_trace_id = reasoning_trace_manager.start_trace(
            question_id="e2e_question",
            agent_id="e2e_agent",
            agent_type="cot",
            operation_type="forecast"
        )

        # 3. Simulate forecasting steps with metrics and tracing
        research_span = tracer.start_span("research_phase", kind=SpanKind.INTERNAL)

        # Record research step in reasoning trace
        reasoning_trace_manager.add_reasoning_step(
            trace_id=reasoning_trace_id,
            step_type=ReasoningStepType.EVIDENCE_GATHERING,
            description="Gathering evidence from multiple sources",
            input_data={"query": "test query"},
            output_data={"sources": ["source1", "source2"]},
            confidence_level=0.8,
            confidence_basis="Multiple reliable sources",
            duration_ms=500.0
        )

        # Record API metrics
        metrics_collector.record_api_request("asknews", "/search", "success", 0.3)
        metrics_collector.record_api_request("perplexity", "/search", "success", 0.4)

        tracer.finish_span(research_span, SpanStatus.OK)

        # 4. Prediction phase
        prediction_span = tracer.start_span("prediction_phase", kind=SpanKind.INTERNAL)

        reasoning_trace_manager.add_reasoning_step(
            trace_id=reasoning_trace_id,
            step_type=ReasoningStepType.PREDICTION_GENERATION,
            description="Generating final prediction",
            input_data={"evidence": "compiled evidence"},
            output_data={"prediction": 0.75, "confidence": 0.85},
            confidence_level=0.85,
            confidence_basis="Strong evidence convergence",
            duration_ms=300.0
        )

        tracer.finish_span(prediction_span, SpanStatus.OK)

        # 5. Record final forecasting metrics
        metrics_collector.record_forecasting_request(
            question_type="binary",
            agent_type="cot",
            status="success",
            duration=2.5,
            accuracy=0.85,
            confidence=0.85
        )

        # 6. Finish traces
        tracer.finish_span(root_span, SpanStatus.OK)
        reasoning_trace_manager.finish_trace(
            reasoning_trace_id,
            success=True,
            final_result={"prediction": 0.75, "confidence": 0.85}
        )

        # 7. Run health checks
        test_health_check = TestHealthCheck("e2e_component")
        health_manager.register_health_check(test_health_check)
        health_summary = await health_manager.check_all_health()

        # 8. Verify all monitoring data is collected

        # Check metrics
        forecasting_summary = metrics_collector.get_metric_summary("forecasting_requests_total")
        assert forecasting_summary

        api_summary = metrics_collector.get_metric_summary("api_requests_total")
        assert api_summary

        # Check distributed trace
        trace = tracer.get_trace(root_span.trace_id)
        assert trace is not None
        assert len(trace.spans) == 3  # root + research + prediction

        # Check reasoning trace
        reasoning_trace = reasoning_trace_manager.get_trace(reasoning_trace_id)
        assert reasoning_trace is not None
        assert len(reasoning_trace.steps) == 2
        assert reasoning_trace.success is True

        # Check health monitoring
        assert health_summary.healthy_components >= 1

        # Check dashboard data integration
        dashboard_data = dashboard.get_dashboard_data("forecasting_performance")
        assert dashboard_data
        assert len(dashboard_data["widgets"]) > 0

        # Verify cross-system correlation
        # All systems should have recorded data for the same operation
        assert root_span.trace_id  # Distributed trace ID
        assert reasoning_trace_id  # Reasoning trace ID
        # In a real system, these would be correlated via correlation IDs

    @pytest.mark.asyncio
    async def test_monitoring_under_failure_conditions(
        self,
        metrics_collector,
        health_manager,
        tracer,
        dashboard
    ):
        """Test monitoring system behavior under failure conditions."""

        # 1. Simulate failing health check
        failing_check = TestHealthCheck("failing_component", should_fail=True)
        health_manager.register_health_check(failing_check)

        # 2. Start trace for failed operation
        failed_span = tracer.start_trace("failed_operation")

        # 3. Simulate error in span
        try:
            raise ValueError("Simulated error")
        except ValueError as e:
            failed_span.set_error(e)

        tracer.finish_span(failed_span, SpanStatus.ERROR)

        # 4. Record failed metrics
        metrics_collector.record_forecasting_request(
            question_type="binary",
            agent_type="cot",
            status="error",
            duration=5.0,
            accuracy=None,
            confidence=None
        )

        # 5. Run health checks
        health_summary = await health_manager.check_all_health()

        # 6. Verify failure handling

        # Health check should show unhealthy component
        assert health_summary.unhealthy_components >= 1
        failing_result = health_manager.get_component_status("failing_component")
        assert failing_result.status == HealthStatus.UNHEALTHY

        # Trace should show error
        trace = tracer.get_trace(failed_span.trace_id)
        assert trace is not None
        assert failed_span.status == SpanStatus.ERROR
        assert "error" in failed_span.tags

        # Metrics should record the failure
        forecasting_summary = metrics_collector.get_metric_summary("forecasting_requests_total")
        assert forecasting_summary

        # Dashboard should reflect the issues
        dashboard_data = dashboard.get_dashboard_data("system_health")
        assert dashboard_data

        # Alert system should potentially trigger alerts
        # (depending on thresholds and evaluation timing)
        alerts = dashboard.get_alerts()
        assert "active_alerts" in alerts
        assert "alert_history" in alerts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
