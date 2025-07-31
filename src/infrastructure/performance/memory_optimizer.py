"""
Memory optimization for large tournament datasets and historical analysis.
"""

import gc
import sys
import weakref
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import asyncio
from contextlib import contextmanager
import psutil

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    gc_collections: Dict[int, int]
    object_counts: Dict[str, int]

    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.used_memory / (1024 * 1024)


class MemoryOptimizer:
    """Memory optimization and monitoring system."""

    def __init__(
        self,
        memory_threshold: float = 0.8,  # 80% memory usage threshold
        gc_threshold: int = 1000,       # Objects before triggering GC
        monitoring_interval: int = 30    # Seconds between memory checks
    ):
        self.memory_threshold = memory_threshold
        self.gc_threshold = gc_threshold
        self.monitoring_interval = monitoring_interval

        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Memory tracking
        self.memory_history: deque[MemoryStats] = deque(maxlen=100)
        self.object_pools: Dict[str, 'ObjectPool'] = {}
        self.weak_references: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)

        # Callbacks for memory pressure
        self.memory_pressure_callbacks: List[Callable[[], None]] = []

        # Configure garbage collection
        self._configure_gc()

    def _configure_gc(self) -> None:
        """Configure garbage collection for optimal performance."""
        # Set GC thresholds for better performance with large datasets
        gc.set_threshold(self.gc_threshold, self.gc_threshold // 2, self.gc_threshold // 4)

        # Enable automatic garbage collection
        gc.enable()

        logger.info("Configured garbage collection for memory optimization")

    async def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started memory monitoring")

    async def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped memory monitoring")

    async def _monitoring_loop(self) -> None:
        """Memory monitoring loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.monitoring_interval)

                # Collect memory statistics
                stats = self.get_memory_stats()
                self.memory_history.append(stats)

                # Check for memory pressure
                if stats.memory_percent > self.memory_threshold:
                    logger.warning(f"Memory pressure detected: {stats.memory_percent:.1f}%")
                    await self._handle_memory_pressure()

                # Log memory stats periodically
                if len(self.memory_history) % 10 == 0:
                    logger.info(f"Memory usage: {stats.memory_usage_mb:.1f}MB ({stats.memory_percent:.1f}%)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        memory = psutil.virtual_memory()

        # Garbage collection stats
        gc_stats = {}
        for i in range(3):
            gc_stats[i] = gc.get_count()[i]

        # Object counts by type
        object_counts = defaultdict(int)
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] += 1

        return MemoryStats(
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percent=memory.percent,
            gc_collections=gc_stats,
            object_counts=dict(object_counts)
        )

    async def _handle_memory_pressure(self) -> None:
        """Handle memory pressure situation."""
        logger.info("Handling memory pressure...")

        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")

        # Clear object pools
        for pool_name, pool in self.object_pools.items():
            cleared = pool.clear_unused()
            logger.info(f"Cleared {cleared} unused objects from pool '{pool_name}'")

        # Execute memory pressure callbacks
        for callback in self.memory_pressure_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in memory pressure callback: {e}")

        # Log memory stats after cleanup
        stats = self.get_memory_stats()
        logger.info(f"Memory after cleanup: {stats.memory_usage_mb:.1f}MB ({stats.memory_percent:.1f}%)")

    def add_memory_pressure_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to execute during memory pressure."""
        self.memory_pressure_callbacks.append(callback)

    def create_object_pool(self, name: str, factory: Callable[[], T], max_size: int = 100) -> 'ObjectPool[T]':
        """Create an object pool for memory optimization."""
        pool = ObjectPool(factory, max_size)
        self.object_pools[name] = pool
        return pool

    def track_objects(self, category: str, objects: List[Any]) -> None:
        """Track objects for memory monitoring."""
        weak_set = self.weak_references[category]
        for obj in objects:
            weak_set.add(obj)

    def get_tracked_object_count(self, category: str) -> int:
        """Get count of tracked objects in category."""
        return len(self.weak_references[category])

    @contextmanager
    def memory_limit_context(self, limit_mb: int):
        """Context manager for memory-limited operations."""
        initial_stats = self.get_memory_stats()
        initial_memory_mb = initial_stats.memory_usage_mb

        try:
            yield
        finally:
            final_stats = self.get_memory_stats()
            final_memory_mb = final_stats.memory_usage_mb
            memory_used = final_memory_mb - initial_memory_mb

            if memory_used > limit_mb:
                logger.warning(f"Memory limit exceeded: {memory_used:.1f}MB > {limit_mb}MB")
                # Force cleanup
                gc.collect()

    def optimize_for_large_datasets(self) -> None:
        """Optimize memory settings for large dataset processing."""
        # Increase GC thresholds for large datasets
        gc.set_threshold(self.gc_threshold * 2, self.gc_threshold, self.gc_threshold // 2)

        # Disable automatic GC during processing (manual control)
        gc.disable()

        logger.info("Optimized memory settings for large dataset processing")

    def restore_normal_settings(self) -> None:
        """Restore normal memory settings."""
        self._configure_gc()
        logger.info("Restored normal memory settings")


class ObjectPool(Generic[T]):
    """Thread-safe object pool for memory optimization."""

    def __init__(self, factory: Callable[[], T], max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool: deque[T] = deque()
        self.lock = threading.Lock()
        self.created_count = 0
        self.reused_count = 0

    def get(self) -> T:
        """Get object from pool or create new one."""
        with self.lock:
            if self.pool:
                obj = self.pool.popleft()
                self.reused_count += 1
                return obj
            else:
                obj = self.factory()
                self.created_count += 1
                return obj

    def put(self, obj: T) -> None:
        """Return object to pool."""
        with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)

    def clear_unused(self) -> int:
        """Clear unused objects from pool."""
        with self.lock:
            cleared = len(self.pool)
            self.pool.clear()
            return cleared

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'max_size': self.max_size,
                'created_count': self.created_count,
                'reused_count': self.reused_count,
                'reuse_rate': (self.reused_count / max(1, self.created_count + self.reused_count)) * 100
            }


class MemoryPool:
    """Memory pool for efficient allocation of fixed-size objects."""

    def __init__(self, object_size: int, pool_size: int = 1000):
        self.object_size = object_size
        self.pool_size = pool_size
        self.pool: List[bytearray] = []
        self.free_indices: deque[int] = deque()
        self.lock = threading.Lock()

        # Pre-allocate memory pool
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize the memory pool."""
        with self.lock:
            for i in range(self.pool_size):
                self.pool.append(bytearray(self.object_size))
                self.free_indices.append(i)

        logger.info(f"Initialized memory pool: {self.pool_size} objects of {self.object_size} bytes")

    def allocate(self) -> Optional[bytearray]:
        """Allocate memory from pool."""
        with self.lock:
            if self.free_indices:
                index = self.free_indices.popleft()
                return self.pool[index]
            return None

    def deallocate(self, memory: bytearray) -> None:
        """Return memory to pool."""
        with self.lock:
            # Find index of memory block
            try:
                index = self.pool.index(memory)
                # Clear memory
                memory[:] = b'\x00' * len(memory)
                self.free_indices.append(index)
            except ValueError:
                logger.warning("Attempted to deallocate memory not from this pool")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            return {
                'pool_size': self.pool_size,
                'object_size': self.object_size,
                'free_objects': len(self.free_indices),
                'used_objects': self.pool_size - len(self.free_indices),
                'total_memory_mb': (self.pool_size * self.object_size) / (1024 * 1024)
            }


class DatasetOptimizer:
    """Optimizer for large tournament datasets."""

    def __init__(self, memory_optimizer: MemoryOptimizer):
        self.memory_optimizer = memory_optimizer
        self.chunk_size = 1000  # Process data in chunks
        self.compression_enabled = True

    async def process_large_dataset(
        self,
        data: List[Any],
        processor: Callable[[List[Any]], List[Any]],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Process large dataset in memory-efficient chunks."""
        chunk_size = chunk_size or self.chunk_size
        results = []

        # Optimize memory for large dataset processing
        self.memory_optimizer.optimize_for_large_datasets()

        try:
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]

                # Process chunk
                with self.memory_optimizer.memory_limit_context(100):  # 100MB limit per chunk
                    chunk_results = await asyncio.get_event_loop().run_in_executor(
                        None, processor, chunk
                    )
                    results.extend(chunk_results)

                # Force garbage collection between chunks
                if i % (chunk_size * 5) == 0:
                    gc.collect()

                # Check memory pressure
                stats = self.memory_optimizer.get_memory_stats()
                if stats.memory_percent > 0.9:
                    logger.warning("High memory usage during dataset processing")
                    await self.memory_optimizer._handle_memory_pressure()

        finally:
            # Restore normal memory settings
            self.memory_optimizer.restore_normal_settings()

        return results

    def compress_historical_data(self, data: List[Dict[str, Any]]) -> bytes:
        """Compress historical data for storage."""
        if not self.compression_enabled:
            return str(data).encode()

        try:
            import pickle
            import gzip

            # Serialize and compress
            pickled_data = pickle.dumps(data)
            compressed_data = gzip.compress(pickled_data)

            compression_ratio = len(compressed_data) / len(pickled_data)
            logger.debug(f"Compressed data: {compression_ratio:.2f} ratio")

            return compressed_data

        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            return str(data).encode()

    def decompress_historical_data(self, compressed_data: bytes) -> List[Dict[str, Any]]:
        """Decompress historical data."""
        if not self.compression_enabled:
            return eval(compressed_data.decode())

        try:
            import pickle
            import gzip

            # Decompress and deserialize
            pickled_data = gzip.decompress(compressed_data)
            data = pickle.loads(pickled_data)

            return data

        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            return []


# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()
