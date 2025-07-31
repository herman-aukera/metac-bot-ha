"""
Comprehensive monitoring and observability infrastructure.

This module provides a complete monitoring solution including:
- Structured logging with correlation IDs
- Metrics collection with Prometheus integration
- Health monitoring with component-level checks
- Distributed tracing for request flow visibility
- Performance dashboards with real-time data
- Automated alerting for anomalies and failures
- Reasoning trace preservation for analysis
- Performance benchmarking and regression detection
"""

from src.infrastructure.logging.structured_logger import (
    StructuredLogger,
    LogLevel,
    get_logger,
    configure_logging,
    system_logger,
    domain_logger,
    application_logger,
    infrastructure_logger
)

from src.infrastructure.monitoring.metrics_collector import (
    MetricsCollector,
    MetricType,
    get_metrics_collector,
    configure_metrics,
    metrics_collector
)

from src.infrastructure.monitoring.health_check_manager import (
    HealthCheckManager,
    HealthCheck,
    HealthStatus,
    HealthCheckResult,
    SystemHealthSummary,
    DatabaseHealthCheck,
    ExternalAPIHealthCheck,
    MemoryHealthCheck,
    DiskHealthCheck,
    CustomHealthCheck,
    get_health_check_manager,
    configure_health_monitoring,
    health_check_manager
)

from src.infrastructure.monitoring.distributed_tracing import (
    DistributedTracer,
    Span,
    Trace,
    SpanKind,
    SpanStatus,
    SpanEvent,
    get_tracer,
    configure_tracing,
    start_trace,
    start_span,
    finish_span,
    get_current_span,
    trace_operation,
    distributed_tracer
)

from src.infrastructure.monitoring.dashboard import (
    PerformanceDashboard,
    AlertManager,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    Dashboard,
    DashboardWidget,
    get_dashboard,
    configure_dashboard,
    performance_dashboard
)

from src.infrastructure.monitoring.reasoning_trace import (
    ReasoningTraceManager,
    ReasoningTrace,
    ReasoningStep,
    ReasoningStepType,
    get_reasoning_trace_manager,
    configure_reasoning_traces,
    start_reasoning_trace,
    add_reasoning_step,
    finish_reasoning_trace,
    reasoning_trace_manager
)

from src.infrastructure.monitoring.performance_benchmarking import (
    PerformanceBenchmarkManager,
    PerformanceBenchmark,
    BenchmarkResult,
    BenchmarkType,
    RegressionSeverity,
    PerformanceRegression,
    ForecastingLatencyBenchmark,
    ResearchThroughputBenchmark,
    PredictionAccuracyBenchmark,
    ResourceUsageBenchmark,
    get_benchmark_manager,
    configure_benchmarking,
    benchmark_manager
)

# Convenience functions for easy monitoring setup
def initialize_monitoring(
    service_name: str = "tournament_optimization",
    log_level: LogLevel = LogLevel.INFO,
    enable_prometheus: bool = True,
    prometheus_port: int = 8000,
    health_check_interval: float = 30.0,
    max_traces: int = 1000,
    trace_retention_hours: int = 24,
    max_reasoning_traces: int = 10000,
    reasoning_trace_retention_days: int = 30,
    regression_threshold: float = 10.0,
    auto_start_monitoring: bool = True
):
    """
    Initialize complete monitoring infrastructure with sensible defaults.

    Args:
        service_name: Name of the service for tracing and metrics
        log_level: Logging level
        enable_prometheus: Enable Prometheus metrics exposition
        prometheus_port: Port for Prometheus metrics server
        health_check_interval: Interval between health checks in seconds
        max_traces: Maximum number of distributed traces to keep
        trace_retention_hours: Hours to retain traces
        max_reasoning_traces: Maximum number of reasoning traces to keep
        reasoning_trace_retention_days: Days to retain reasoning traces
        regression_threshold: Threshold percentage for regression detection
        auto_start_monitoring: Whether to start background monitoring tasks

    Returns:
        Dictionary containing all monitoring components
    """

    # Configure logging
    configure_logging(level=log_level, enable_console=True)
    logger = get_logger("monitoring_init")
    logger.info("Initializing comprehensive monitoring infrastructure")

    # Configure metrics collection
    metrics = configure_metrics(
        enable_prometheus=enable_prometheus,
        prometheus_port=prometheus_port if enable_prometheus else None
    )
    logger.info(f"Metrics collection configured (Prometheus: {enable_prometheus})")

    # Configure health monitoring
    health_manager = configure_health_monitoring(
        check_interval=health_check_interval,
        auto_start=auto_start_monitoring
    )
    logger.info(f"Health monitoring configured (interval: {health_check_interval}s)")

    # Configure distributed tracing
    tracer = configure_tracing(
        service_name=service_name,
        max_traces=max_traces,
        trace_retention_hours=trace_retention_hours
    )
    logger.info(f"Distributed tracing configured (max traces: {max_traces})")

    # Configure reasoning traces
    reasoning_traces = configure_reasoning_traces(
        max_traces=max_reasoning_traces,
        retention_days=reasoning_trace_retention_days,
        auto_log=True
    )
    logger.info(f"Reasoning traces configured (max traces: {max_reasoning_traces})")

    # Configure performance benchmarking
    benchmarks = configure_benchmarking(
        regression_threshold=regression_threshold
    )
    logger.info(f"Performance benchmarking configured (threshold: {regression_threshold}%)")

    # Configure dashboard and alerting
    dashboard = configure_dashboard(
        auto_start_monitoring=auto_start_monitoring
    )
    logger.info("Performance dashboard and alerting configured")

    logger.info("Monitoring infrastructure initialization complete")

    return {
        "logger": logger,
        "metrics": metrics,
        "health_manager": health_manager,
        "tracer": tracer,
        "reasoning_traces": reasoning_traces,
        "benchmarks": benchmarks,
        "dashboard": dashboard
    }


def get_monitoring_status():
    """
    Get current status of all monitoring components.

    Returns:
        Dictionary with status information for all monitoring components
    """
    status = {
        "timestamp": system_logger.logger.handlers[0].formatter.formatTime(
            system_logger.logger.makeRecord(
                "status", 20, "", 0, "", (), None
            )
        ) if system_logger.logger.handlers else None,
        "components": {}
    }

    # Metrics status
    try:
        metrics_summary = metrics_collector.get_metric_summary("system_health_score")
        status["components"]["metrics"] = {
            "status": "healthy",
            "metrics_count": len(metrics_summary) if metrics_summary else 0
        }
    except Exception as e:
        status["components"]["metrics"] = {
            "status": "error",
            "error": str(e)
        }

    # Health monitoring status
    try:
        health_summary = health_check_manager.get_latest_health_status()
        status["components"]["health_monitoring"] = {
            "status": health_summary.overall_status.value,
            "healthy_components": health_summary.healthy_components,
            "total_components": len(health_summary.component_results)
        }
    except Exception as e:
        status["components"]["health_monitoring"] = {
            "status": "error",
            "error": str(e)
        }

    # Distributed tracing status
    try:
        all_traces = distributed_tracer.get_all_traces()
        status["components"]["distributed_tracing"] = {
            "status": "healthy",
            "total_traces": len(all_traces),
            "active_spans": len(distributed_tracer.active_spans)
        }
    except Exception as e:
        status["components"]["distributed_tracing"] = {
            "status": "error",
            "error": str(e)
        }

    # Reasoning traces status
    try:
        trace_stats = reasoning_trace_manager.get_trace_statistics()
        status["components"]["reasoning_traces"] = {
            "status": "healthy",
            "total_traces": trace_stats.get("total_traces", 0),
            "success_rate": trace_stats.get("success_rate", 0.0)
        }
    except Exception as e:
        status["components"]["reasoning_traces"] = {
            "status": "error",
            "error": str(e)
        }

    # Performance benchmarking status
    try:
        benchmark_summary = benchmark_manager.get_performance_summary()
        status["components"]["benchmarking"] = {
            "status": "healthy",
            "total_benchmarks": benchmark_summary.get("total_benchmarks", 0),
            "active_regressions": benchmark_summary.get("active_regressions", 0)
        }
    except Exception as e:
        status["components"]["benchmarking"] = {
            "status": "error",
            "error": str(e)
        }

    # Dashboard status
    try:
        alerts = performance_dashboard.get_alerts()
        status["components"]["dashboard"] = {
            "status": "healthy",
            "active_alerts": alerts.get("summary", {}).get("total_active", 0),
            "dashboards": len(performance_dashboard.dashboards)
        }
    except Exception as e:
        status["components"]["dashboard"] = {
            "status": "error",
            "error": str(e)
        }

    return status


__all__ = [
    # Structured Logging
    "StructuredLogger",
    "LogLevel",
    "get_logger",
    "configure_logging",
    "system_logger",
    "domain_logger",
    "application_logger",
    "infrastructure_logger",

    # Metrics Collection
    "MetricsCollector",
    "MetricType",
    "get_metrics_collector",
    "configure_metrics",
    "metrics_collector",

    # Health Monitoring
    "HealthCheckManager",
    "HealthCheck",
    "HealthStatus",
    "HealthCheckResult",
    "SystemHealthSummary",
    "DatabaseHealthCheck",
    "ExternalAPIHealthCheck",
    "MemoryHealthCheck",
    "DiskHealthCheck",
    "CustomHealthCheck",
    "get_health_check_manager",
    "configure_health_monitoring",
    "health_check_manager",

    # Distributed Tracing
    "DistributedTracer",
    "Span",
    "Trace",
    "SpanKind",
    "SpanStatus",
    "SpanEvent",
    "get_tracer",
    "configure_tracing",
    "start_trace",
    "start_span",
    "finish_span",
    "get_current_span",
    "trace_operation",
    "distributed_tracer",

    # Dashboard and Alerting
    "PerformanceDashboard",
    "AlertManager",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "Dashboard",
    "DashboardWidget",
    "get_dashboard",
    "configure_dashboard",
    "performance_dashboard",

    # Reasoning Traces
    "ReasoningTraceManager",
    "ReasoningTrace",
    "ReasoningStep",
    "ReasoningStepType",
    "get_reasoning_trace_manager",
    "configure_reasoning_traces",
    "start_reasoning_trace",
    "add_reasoning_step",
    "finish_reasoning_trace",
    "reasoning_trace_manager",

    # Performance Benchmarking
    "PerformanceBenchmarkManager",
    "PerformanceBenchmark",
    "BenchmarkResult",
    "BenchmarkType",
    "RegressionSeverity",
    "PerformanceRegression",
    "ForecastingLatencyBenchmark",
    "ResearchThroughputBenchmark",
    "PredictionAccuracyBenchmark",
    "ResourceUsageBenchmark",
    "get_benchmark_manager",
    "configure_benchmarking",
    "benchmark_manager",

    # Convenience Functions
    "initialize_monitoring",
    "get_monitoring_status"
]
