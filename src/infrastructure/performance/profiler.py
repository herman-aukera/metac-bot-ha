"""
Performance profiling and bottleneck identification tools.
"""

import time
import cProfile
import pstats
import io
import logging
import asyncio
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import threading
from collections import defaultdict, deque
import psutil
import tracemalloc

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ProfileResult:
    """Performance profiling result."""
    function_name: str
    execution_time: float
    cpu_time: float
    memory_usage: int
    call_count: int
    timestamp: datetime
    stack_trace: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckInfo:
    """Information about performance bottleneck."""
    function_name: str
    total_time: float
    avg_time: float
    call_count: int
    percentage_of_total: float
    memory_impact: int
    severity: str  # 'low', 'medium', 'high', 'critical'


class PerformanceProfiler:
    """Comprehensive performance profiler with bottleneck identification."""

    def __init__(self, enable_memory_profiling: bool = True):
        self.enable_memory_profiling = enable_memory_profiling
        self.profile_results: deque[ProfileResult] = deque(maxlen=10000)
        self.function_stats: Dict[str, List[float]] = defaultdict(list)
        self.bottlenecks: List[BottleneckInfo] = []

        # Profiling state
        self.is_profiling = False
        self.profiler: Optional[cProfile.Profile] = None
        self.memory_snapshots: List[Any] = []

        # Thread-local storage for nested profiling
        self.local = threading.local()

        # Initialize memory profiling
        if self.enable_memory_profiling:
            tracemalloc.start()

    @contextmanager
    def profile_context(self, name: str, custom_metrics: Optional[Dict[str, Any]] = None):
        """Context manager for profiling code blocks."""
        start_time = time.time()
        start_cpu_time = time.process_time()

        # Memory snapshot before
        memory_before = 0
        if self.enable_memory_profiling:
            try:
                current, peak = tracemalloc.get_traced_memory()
                memory_before = current
            except Exception:
                pass

        # Start CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            yield
        finally:
            # Stop CPU profiling
            profiler.disable()

            # Calculate metrics
            end_time = time.time()
            end_cpu_time = time.process_time()
            execution_time = end_time - start_time
            cpu_time = end_cpu_time - start_cpu_time

            # Memory usage
            memory_after = 0
            if self.enable_memory_profiling:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    memory_after = current
                except Exception:
                    pass

            memory_usage = memory_after - memory_before

            # Create profile result
            result = ProfileResult(
                function_name=name,
                execution_time=execution_time,
                cpu_time=cpu_time,
                memory_usage=memory_usage,
                call_count=1,
                timestamp=datetime.now(),
                custom_metrics=custom_metrics or {}
            )

            self.profile_results.append(result)
            self.function_stats[name].append(execution_time)

            # Log slow operations
            if execution_time > 1.0:  # Log operations taking more than 1 second
                logger.warning(f"Slow operation '{name}': {execution_time:.3f}s")

    @asynccontextmanager
    async def async_profile_context(self, name: str, custom_metrics: Optional[Dict[str, Any]] = None):
        """Async context manager for profiling async code blocks."""
        start_time = time.time()

        # Memory snapshot before
        memory_before = 0
        if self.enable_memory_profiling:
            try:
                current, peak = tracemalloc.get_traced_memory()
                memory_before = current
            except Exception:
                pass

        try:
            yield
        finally:
            # Calculate metrics
            end_time = time.time()
            execution_time = end_time - start_time

            # Memory usage
            memory_after = 0
            if self.enable_memory_profiling:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    memory_after = current
                except Exception:
                    pass

            memory_usage = memory_after - memory_before

            # Create profile result
            result = ProfileResult(
                function_name=name,
                execution_time=execution_time,
                cpu_time=0.0,  # CPU time not available for async
                memory_usage=memory_usage,
                call_count=1,
                timestamp=datetime.now(),
                custom_metrics=custom_metrics or {}
            )

            self.profile_results.append(result)
            self.function_stats[name].append(execution_time)

            # Log slow operations
            if execution_time > 1.0:
                logger.warning(f"Slow async operation '{name}': {execution_time:.3f}s")

    def profile_function(self, name: Optional[str] = None):
        """Decorator for profiling functions."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            func_name = name or f"{func.__module__}.{func.__name__}"

            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.async_profile_context(func_name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.profile_context(func_name):
                        return func(*args, **kwargs)
                return sync_wrapper

        return decorator

    def start_continuous_profiling(self) -> None:
        """Start continuous profiling."""
        if self.is_profiling:
            return

        self.is_profiling = True
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        logger.info("Started continuous profiling")

    def stop_continuous_profiling(self) -> Optional[pstats.Stats]:
        """Stop continuous profiling and return stats."""
        if not self.is_profiling or not self.profiler:
            return None

        self.profiler.disable()
        self.is_profiling = False

        # Create stats object
        stats_stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)

        logger.info("Stopped continuous profiling")
        return stats

    def analyze_bottlenecks(self, min_calls: int = 10, min_time: float = 0.1) -> List[BottleneckInfo]:
        """Analyze performance bottlenecks from collected data."""
        bottlenecks = []

        # Calculate total execution time
        total_time = sum(
            sum(times) for times in self.function_stats.values()
        )

        if total_time == 0:
            return bottlenecks

        # Analyze each function
        for func_name, times in self.function_stats.items():
            if len(times) < min_calls:
                continue

            func_total_time = sum(times)
            if func_total_time < min_time:
                continue

            avg_time = func_total_time / len(times)
            percentage = (func_total_time / total_time) * 100

            # Calculate memory impact
            memory_impact = 0
            for result in self.profile_results:
                if result.function_name == func_name:
                    memory_impact += result.memory_usage

            # Determine severity
            if percentage > 20:
                severity = 'critical'
            elif percentage > 10:
                severity = 'high'
            elif percentage > 5:
                severity = 'medium'
            else:
                severity = 'low'

            bottleneck = BottleneckInfo(
                function_name=func_name,
                total_time=func_total_time,
                avg_time=avg_time,
                call_count=len(times),
                percentage_of_total=percentage,
                memory_impact=memory_impact,
                severity=severity
            )

            bottlenecks.append(bottleneck)

        # Sort by total time descending
        bottlenecks.sort(key=lambda x: x.total_time, reverse=True)
        self.bottlenecks = bottlenecks

        return bottlenecks

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        # Analyze bottlenecks
        bottlenecks = self.analyze_bottlenecks()

        # Calculate summary statistics
        if self.profile_results:
            total_operations = len(self.profile_results)
            total_time = sum(r.execution_time for r in self.profile_results)
            avg_time = total_time / total_operations

            # Find slowest operations
            slowest = sorted(
                self.profile_results,
                key=lambda x: x.execution_time,
                reverse=True
            )[:10]

            # Memory statistics
            total_memory = sum(r.memory_usage for r in self.profile_results)

        else:
            total_operations = 0
            total_time = 0.0
            avg_time = 0.0
            slowest = []
            total_memory = 0

        return {
            'summary': {
                'total_operations': total_operations,
                'total_time': total_time,
                'average_time': avg_time,
                'total_memory_usage': total_memory
            },
            'bottlenecks': [
                {
                    'function_name': b.function_name,
                    'total_time': b.total_time,
                    'avg_time': b.avg_time,
                    'call_count': b.call_count,
                    'percentage': b.percentage_of_total,
                    'memory_impact': b.memory_impact,
                    'severity': b.severity
                }
                for b in bottlenecks[:10]
            ],
            'slowest_operations': [
                {
                    'function_name': r.function_name,
                    'execution_time': r.execution_time,
                    'memory_usage': r.memory_usage,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in slowest
            ],
            'function_stats': {
                name: {
                    'call_count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
                for name, times in self.function_stats.items()
            }
        }

    def clear_stats(self) -> None:
        """Clear all profiling statistics."""
        self.profile_results.clear()
        self.function_stats.clear()
        self.bottlenecks.clear()
        logger.info("Cleared profiling statistics")

    def export_profile_data(self, filename: str) -> None:
        """Export profile data to file."""
        try:
            report = self.get_performance_report()

            import json
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Exported profile data to {filename}")

        except Exception as e:
            logger.error(f"Error exporting profile data: {e}")


class ProfilerContext:
    """Context manager for easy profiling."""

    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        return self.profiler.profile_context(self.name).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.profiler.profile_context(self.name).__exit__(exc_type, exc_val, exc_tb)


class SystemProfiler:
    """System-wide performance profiler."""

    def __init__(self, monitoring_interval: int = 5):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # System metrics history
        self.cpu_history: deque[float] = deque(maxlen=1000)
        self.memory_history: deque[float] = deque(maxlen=1000)
        self.disk_io_history: deque[Tuple[int, int]] = deque(maxlen=1000)
        self.network_io_history: deque[Tuple[int, int]] = deque(maxlen=1000)

        # Process-specific metrics
        self.process = psutil.Process()

    async def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started system profiling")

    async def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped system profiling")

    async def _monitoring_loop(self) -> None:
        """System monitoring loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.monitoring_interval)

                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()

                # Store metrics
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory.percent)

                if disk_io:
                    self.disk_io_history.append((disk_io.read_bytes, disk_io.write_bytes))

                if network_io:
                    self.network_io_history.append((network_io.bytes_sent, network_io.bytes_recv))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")

    def get_system_report(self) -> Dict[str, Any]:
        """Get system performance report."""
        if not self.cpu_history:
            return {'error': 'No monitoring data available'}

        # Calculate statistics
        import statistics

        cpu_stats = {
            'avg': statistics.mean(self.cpu_history),
            'max': max(self.cpu_history),
            'min': min(self.cpu_history),
            'current': self.cpu_history[-1] if self.cpu_history else 0
        }

        memory_stats = {
            'avg': statistics.mean(self.memory_history),
            'max': max(self.memory_history),
            'min': min(self.memory_history),
            'current': self.memory_history[-1] if self.memory_history else 0
        }

        # Process-specific stats
        try:
            process_info = {
                'cpu_percent': self.process.cpu_percent(),
                'memory_info': self.process.memory_info()._asdict(),
                'num_threads': self.process.num_threads(),
                'open_files': len(self.process.open_files()),
                'connections': len(self.process.connections())
            }
        except Exception as e:
            process_info = {'error': str(e)}

        return {
            'cpu': cpu_stats,
            'memory': memory_stats,
            'process': process_info,
            'monitoring_duration': len(self.cpu_history) * self.monitoring_interval,
            'data_points': len(self.cpu_history)
        }


# Global profiler instances
performance_profiler = PerformanceProfiler()
system_profiler = SystemProfiler()
