"""
Performance optimization and scalability infrastructure.
"""

from .memory_optimizer import MemoryOptimizer, MemoryPool, ObjectPool
from .auto_scaler import AutoScaler, ScalingPolicy, ResourceMonitor
from .profiler import PerformanceProfiler, ProfilerContext
from .query_optimizer import QueryOptimizer, IndexManager

__all__ = [
    'MemoryOptimizer',
    'MemoryPool',
    'ObjectPool',
    'AutoScaler',
    'ScalingPolicy',
    'ResourceMonitor',
    'PerformanceProfiler',
    'ProfilerContext',
    'QueryOptimizer',
    'IndexManager'
]
