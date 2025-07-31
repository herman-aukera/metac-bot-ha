"""
Performance benchmarking and regression detection systems.

Provides comprehensive performance benchmarking capabilities with
automated regression detection and performance trend analysis.
"""

import time
import statistics
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import json

from src.infrastructure.logging.structured_logger import get_logger
from src.infrastructure.monitoring.metrics_collector import get_metrics_collector


class BenchmarkType(Enum):
    """Types of benchmarks."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    RESOURCE_USAGE = "resource_usage"
    CUSTOM = "custom"


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    id: str
    benchmark_name: str
    benchmark_type: BenchmarkType
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    baseline_value: Optional[float] = None
    regression_percentage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "benchmark_name": self.benchmark_name,
            "benchmark_type": self.benchmark_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "baseline_value": self.baseline_value,
            "regression_percentage": self.regression_percentage
        }


@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks."""
    name: str
    description: str
    benchmarks: List[str]
    schedule: Optional[str] = None  # Cron-like schedule
    enabled: bool = True


@dataclass
class PerformanceRegression:
    """Detected performance regression."""
    id: str
    benchmark_name: str
    severity: RegressionSeverity
    current_value: float
    baseline_value: float
    regression_percentage: float
    detected_at: datetime
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "benchmark_name": self.benchmark_name,
            "severity": self.severity.value,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "regression_percentage": self.regression_percentage,
            "detected_at": self.detected_at.isoformat(),
            "description": self.description,
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class PerformanceBenchmark:
    """Individual performance benchmark."""

    def __init__(
        self,
        name: str,
        benchmark_type: BenchmarkType,
        unit: str,
        description: str = "",
        baseline_value: Optional[float] = None
    ):
        self.name = name
        self.benchmark_type = benchmark_type
        self.unit = unit
        self.description = description
        self.baseline_value = baseline_value
        self.logger = get_logger(f"benchmark.{name}")

    async def run(self, **kwargs) -> BenchmarkResult:
        """Run the benchmark - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run method")

    def _create_result(self, value: float, metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        """Create a benchmark result."""
        result_id = f"{self.name}_{int(time.time())}"

        result = BenchmarkResult(
            id=result_id,
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            value=value,
            unit=self.unit,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            baseline_value=self.baseline_value
        )

        # Calculate regression if baseline exists
        if self.baseline_value and self.baseline_value > 0:
            if self.benchmark_type in [BenchmarkType.LATENCY, BenchmarkType.RESOURCE_USAGE]:
                # For latency and resource usage, higher is worse
                result.regression_percentage = ((value - self.baseline_value) / self.baseline_value) * 100
            else:
                # For throughput and accuracy, lower is worse
                result.regression_percentage = ((self.baseline_value - value) / self.baseline_value) * 100

        return result


class ForecastingLatencyBenchmark(PerformanceBenchmark):
    """Benchmark for forecasting operation latency."""

    def __init__(self, forecasting_service, baseline_value: Optional[float] = None):
        super().__init__(
            name="forecasting_latency",
            benchmark_type=BenchmarkType.LATENCY,
            unit="milliseconds",
            description="Time to complete a forecasting operation",
            baseline_value=baseline_value
        )
        self.forecasting_service = forecasting_service

    async def run(self, question_data: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        """Run forecasting latency benchmark."""
        # Use sample question if none provided
        if not question_data:
            question_data = {
                "id": "benchmark_question",
                "title": "Benchmark Question",
                "description": "Sample question for benchmarking",
                "type": "binary"
            }

        start_time = time.time()

        try:
            # Run forecasting operation
            result = await self.forecasting_service.process_question(question_data)

            duration_ms = (time.time() - start_time) * 1000

            metadata = {
                "question_id": question_data.get("id"),
                "question_type": question_data.get("type"),
                "success": True,
                "result_summary": {
                    "prediction_count": len(result.get("predictions", [])),
                    "confidence": result.get("confidence", 0.0)
                }
            }

            return self._create_result(duration_ms, metadata)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            metadata = {
                "question_id": question_data.get("id"),
                "question_type": question_data.get("type"),
                "success": False,
                "error": str(e)
            }

            # Return high latency for failed operations
            return self._create_result(duration_ms * 10, metadata)


class ResearchThroughputBenchmark(PerformanceBenchmark):
    """Benchmark for research operation throughput."""

    def __init__(self, research_service, baseline_value: Optional[float] = None):
        super().__init__(
            name="research_throughput",
            benchmark_type=BenchmarkType.THROUGHPUT,
            unit="operations_per_second",
            description="Number of research operations completed per second",
            baseline_value=baseline_value
        )
        self.research_service = research_service

    async def run(self, duration_seconds: int = 60, concurrent_operations: int = 5) -> BenchmarkResult:
        """Run research throughput benchmark."""
        import asyncio

        completed_operations = 0
        start_time = time.time()
        end_time = start_time + duration_seconds

        async def research_operation():
            nonlocal completed_operations
            while time.time() < end_time:
                try:
                    await self.research_service.conduct_research("sample query")
                    completed_operations += 1
                except Exception:
                    pass  # Count failures as well

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)

        # Run concurrent operations
        tasks = [research_operation() for _ in range(concurrent_operations)]
        await asyncio.gather(*tasks, return_exceptions=True)

        actual_duration = time.time() - start_time
        throughput = completed_operations / actual_duration

        metadata = {
            "duration_seconds": actual_duration,
            "completed_operations": completed_operations,
            "concurrent_operations": concurrent_operations,
            "avg_operations_per_worker": completed_operations / concurrent_operations
        }

        return self._create_result(throughput, metadata)


class PredictionAccuracyBenchmark(PerformanceBenchmark):
    """Benchmark for prediction accuracy."""

    def __init__(self, forecasting_service, test_questions: List[Dict[str, Any]], baseline_value: Optional[float] = None):
        super().__init__(
            name="prediction_accuracy",
            benchmark_type=BenchmarkType.ACCURACY,
            unit="accuracy_score",
            description="Accuracy of predictions on test dataset",
            baseline_value=baseline_value
        )
        self.forecasting_service = forecasting_service
        self.test_questions = test_questions

    async def run(self) -> BenchmarkResult:
        """Run prediction accuracy benchmark."""
        correct_predictions = 0
        total_predictions = 0

        for question in self.test_questions:
            if "expected_answer" not in question:
                continue

            try:
                result = await self.forecasting_service.process_question(question)
                prediction = result.get("prediction")
                expected = question["expected_answer"]

                # Simple accuracy check (can be made more sophisticated)
                if abs(prediction - expected) < 0.1:  # Within 10%
                    correct_predictions += 1

                total_predictions += 1

            except Exception:
                total_predictions += 1  # Count failures as incorrect

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        metadata = {
            "total_questions": len(self.test_questions),
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy_percentage": accuracy * 100
        }

        return self._create_result(accuracy, metadata)


class ResourceUsageBenchmark(PerformanceBenchmark):
    """Benchmark for resource usage."""

    def __init__(self, baseline_value: Optional[float] = None):
        super().__init__(
            name="resource_usage",
            benchmark_type=BenchmarkType.RESOURCE_USAGE,
            unit="percentage",
            description="System resource usage during operations",
            baseline_value=baseline_value
        )

    async def run(self, duration_seconds: int = 60) -> BenchmarkResult:
        """Run resource usage benchmark."""
        try:
            import psutil

            # Monitor resource usage
            cpu_samples = []
            memory_samples = []

            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent

                cpu_samples.append(cpu_percent)
                memory_samples.append(memory_percent)

            avg_cpu = statistics.mean(cpu_samples)
            avg_memory = statistics.mean(memory_samples)
            max_cpu = max(cpu_samples)
            max_memory = max(memory_samples)

            # Use combined resource score
            resource_score = (avg_cpu + avg_memory) / 2

            metadata = {
                "duration_seconds": duration_seconds,
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "max_cpu_percent": max_cpu,
                "max_memory_percent": max_memory,
                "sample_count": len(cpu_samples)
            }

            return self._create_result(resource_score, metadata)

        except ImportError:
            # Fallback if psutil not available
            metadata = {"error": "psutil not available"}
            return self._create_result(0.0, metadata)


class PerformanceBenchmarkManager:
    """
    Comprehensive performance benchmarking and regression detection system.

    Manages performance benchmarks, detects regressions, and provides
    trend analysis for system performance monitoring.
    """

    def __init__(self, regression_threshold: float = 10.0):
        self.logger = get_logger("performance_benchmark_manager")
        self.metrics_collector = get_metrics_collector()
        self.regression_threshold = regression_threshold

        # Benchmark storage
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.benchmark_suites: Dict[str, BenchmarkSuite] = {}
        self.results_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Regression tracking
        self.active_regressions: Dict[str, PerformanceRegression] = {}
        self.regression_history: List[PerformanceRegression] = []

        # Baseline management
        self.baselines: Dict[str, float] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Initialize default benchmarks
        self._initialize_default_benchmarks()

    def _initialize_default_benchmarks(self):
        """Initialize default performance benchmarks."""
        # Resource usage benchmark (always available)
        resource_benchmark = ResourceUsageBenchmark(baseline_value=50.0)  # 50% resource usage baseline
        self.register_benchmark(resource_benchmark)

        # Create default benchmark suite
        default_suite = BenchmarkSuite(
            name="system_performance",
            description="Core system performance benchmarks",
            benchmarks=["resource_usage"],
            schedule="0 */6 * * *",  # Every 6 hours
            enabled=True
        )
        self.register_benchmark_suite(default_suite)

    def register_benchmark(self, benchmark: PerformanceBenchmark):
        """Register a performance benchmark."""
        with self._lock:
            self.benchmarks[benchmark.name] = benchmark

            # Set baseline if provided
            if benchmark.baseline_value:
                self.baselines[benchmark.name] = benchmark.baseline_value

        self.logger.info(f"Registered benchmark: {benchmark.name}")

    def register_benchmark_suite(self, suite: BenchmarkSuite):
        """Register a benchmark suite."""
        with self._lock:
            self.benchmark_suites[suite.name] = suite

        self.logger.info(f"Registered benchmark suite: {suite.name}")

    async def run_benchmark(self, benchmark_name: str, **kwargs) -> Optional[BenchmarkResult]:
        """Run a specific benchmark."""
        benchmark = self.benchmarks.get(benchmark_name)
        if not benchmark:
            self.logger.warning(f"Benchmark not found: {benchmark_name}")
            return None

        try:
            self.logger.info(f"Running benchmark: {benchmark_name}")
            result = await benchmark.run(**kwargs)

            # Store result
            with self._lock:
                self.results_history[benchmark_name].append(result)

            # Check for regressions
            self._check_for_regression(result)

            # Record metrics
            self.metrics_collector.observe_histogram(
                "benchmark_duration_seconds",
                result.value / 1000 if result.unit == "milliseconds" else result.value,
                labels={
                    "benchmark_name": benchmark_name,
                    "benchmark_type": result.benchmark_type.value
                }
            )

            self.logger.info(
                f"Benchmark completed: {benchmark_name}",
                extra={
                    "benchmark_result": result.to_dict(),
                    "event_type": "benchmark_completed"
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"Benchmark failed: {benchmark_name}", exception=e)
            return None

    async def run_benchmark_suite(self, suite_name: str, **kwargs) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in a suite."""
        suite = self.benchmark_suites.get(suite_name)
        if not suite or not suite.enabled:
            self.logger.warning(f"Benchmark suite not found or disabled: {suite_name}")
            return {}

        results = {}

        self.logger.info(f"Running benchmark suite: {suite_name}")

        for benchmark_name in suite.benchmarks:
            result = await self.run_benchmark(benchmark_name, **kwargs)
            if result:
                results[benchmark_name] = result

        self.logger.info(f"Benchmark suite completed: {suite_name}")
        return results

    def set_baseline(self, benchmark_name: str, value: Optional[float] = None):
        """Set baseline value for a benchmark."""
        if value is None:
            # Use latest result as baseline
            with self._lock:
                history = self.results_history.get(benchmark_name, [])
                if history:
                    value = history[-1].value
                else:
                    self.logger.warning(f"No results available for baseline: {benchmark_name}")
                    return

        with self._lock:
            self.baselines[benchmark_name] = value

            # Update benchmark baseline
            if benchmark_name in self.benchmarks:
                self.benchmarks[benchmark_name].baseline_value = value

        self.logger.info(f"Set baseline for {benchmark_name}: {value}")

    def _check_for_regression(self, result: BenchmarkResult):
        """Check if a benchmark result indicates a performance regression."""
        if result.regression_percentage is None:
            return

        # Determine severity based on regression percentage
        severity = None
        if abs(result.regression_percentage) >= 50:
            severity = RegressionSeverity.CRITICAL
        elif abs(result.regression_percentage) >= 25:
            severity = RegressionSeverity.MAJOR
        elif abs(result.regression_percentage) >= 15:
            severity = RegressionSeverity.MODERATE
        elif abs(result.regression_percentage) >= self.regression_threshold:
            severity = RegressionSeverity.MINOR

        if severity:
            self._create_regression(result, severity)

    def _create_regression(self, result: BenchmarkResult, severity: RegressionSeverity):
        """Create a performance regression record."""
        regression_id = f"{result.benchmark_name}_{int(time.time())}"

        regression = PerformanceRegression(
            id=regression_id,
            benchmark_name=result.benchmark_name,
            severity=severity,
            current_value=result.value,
            baseline_value=result.baseline_value,
            regression_percentage=result.regression_percentage,
            detected_at=result.timestamp,
            description=f"Performance regression detected in {result.benchmark_name}: {result.regression_percentage:.1f}% degradation",
            metadata=result.metadata
        )

        with self._lock:
            self.active_regressions[regression_id] = regression
            self.regression_history.append(regression)

        # Log regression
        self.logger.warning(
            f"Performance regression detected: {result.benchmark_name}",
            extra={
                "regression": regression.to_dict(),
                "event_type": "performance_regression"
            }
        )

        # Record regression metric
        self.metrics_collector.increment_counter(
            "performance_regressions_total",
            labels={
                "benchmark_name": result.benchmark_name,
                "severity": severity.value
            }
        )

    def resolve_regression(self, regression_id: str):
        """Mark a regression as resolved."""
        with self._lock:
            if regression_id in self.active_regressions:
                regression = self.active_regressions[regression_id]
                regression.resolved = True
                regression.resolved_at = datetime.utcnow()

                del self.active_regressions[regression_id]

                self.logger.info(f"Regression resolved: {regression.benchmark_name}")

    def get_benchmark_trends(
        self,
        benchmark_name: str,
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Get performance trends for a benchmark."""
        with self._lock:
            history = list(self.results_history.get(benchmark_name, []))

        if not history:
            return {"error": "No data available"}

        # Filter by time window
        cutoff_time = datetime.utcnow() - time_window
        recent_results = [r for r in history if r.timestamp >= cutoff_time]

        if not recent_results:
            return {"error": "No recent data available"}

        values = [r.value for r in recent_results]

        # Calculate trend statistics
        trends = {
            "benchmark_name": benchmark_name,
            "time_window": str(time_window),
            "sample_count": len(recent_results),
            "latest_value": values[-1],
            "min_value": min(values),
            "max_value": max(values),
            "mean_value": statistics.mean(values),
            "median_value": statistics.median(values),
            "std_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "trend_direction": self._calculate_trend_direction(values),
            "baseline_value": self.baselines.get(benchmark_name),
            "recent_results": [r.to_dict() for r in recent_results[-10:]]  # Last 10 results
        }

        return trends

    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "stable"

        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))

        # Calculate slope
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Determine trend
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        with self._lock:
            active_regressions = list(self.active_regressions.values())
            total_benchmarks = len(self.benchmarks)
            total_suites = len(self.benchmark_suites)

        # Count regressions by severity
        regression_counts = {
            "critical": len([r for r in active_regressions if r.severity == RegressionSeverity.CRITICAL]),
            "major": len([r for r in active_regressions if r.severity == RegressionSeverity.MAJOR]),
            "moderate": len([r for r in active_regressions if r.severity == RegressionSeverity.MODERATE]),
            "minor": len([r for r in active_regressions if r.severity == RegressionSeverity.MINOR])
        }

        return {
            "total_benchmarks": total_benchmarks,
            "total_benchmark_suites": total_suites,
            "active_regressions": len(active_regressions),
            "regression_counts": regression_counts,
            "total_regression_history": len(self.regression_history),
            "benchmarks": list(self.benchmarks.keys()),
            "benchmark_suites": list(self.benchmark_suites.keys()),
            "baselines": dict(self.baselines)
        }

    def export_results(
        self,
        benchmark_name: Optional[str] = None,
        format: str = "json"
    ) -> str:
        """Export benchmark results."""
        with self._lock:
            if benchmark_name:
                results = list(self.results_history.get(benchmark_name, []))
            else:
                results = []
                for history in self.results_history.values():
                    results.extend(history)

        if format == "json":
            return json.dumps([r.to_dict() for r in results], indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global benchmark manager instance
benchmark_manager = PerformanceBenchmarkManager()


def get_benchmark_manager() -> PerformanceBenchmarkManager:
    """Get the global benchmark manager instance."""
    return benchmark_manager


def configure_benchmarking(
    regression_threshold: float = 10.0
) -> PerformanceBenchmarkManager:
    """Configure and return a benchmark manager."""
    global benchmark_manager
    benchmark_manager = PerformanceBenchmarkManager(
        regression_threshold=regression_threshold
    )
    return benchmark_manager
