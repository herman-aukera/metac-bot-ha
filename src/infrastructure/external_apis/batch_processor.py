"""
Intelligent request batching and queue management for API efficiency.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchStrategy(Enum):
    """Batching strategies."""
    SIZE_BASED = "size_based"
    TIME_BASED = "time_based"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"


@dataclass
class BatchRequest(Generic[T]):
    """Individual request in a batch."""
    id: str
    data: T
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None
    callback: Optional[Callable[[Any], None]] = None
    future: asyncio.Future = field(default_factory=asyncio.Future)


@dataclass
class RequestBatch(Generic[T]):
    """Collection of requests to be processed together."""
    id: str
    requests: List[BatchRequest[T]]
    created_at: datetime = field(default_factory=datetime.now)
    strategy: BatchStrategy = BatchStrategy.SIZE_BASED

    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.requests)

    @property
    def age(self) -> float:
        """Get batch age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def add_request(self, request: BatchRequest[T]) -> None:
        """Add request to batch."""
        self.requests.append(request)

    def get_data(self) -> List[T]:
        """Get all request data."""
        return [req.data for req in self.requests]


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 10
    max_wait_time: float = 1.0  # seconds
    min_batch_size: int = 1
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    priority_levels: int = 3
    adaptive_threshold: float = 0.8  # CPU/memory threshold for adaptive batching
    queue_size_limit: int = 1000


class BatchProcessor(Generic[T, R]):
    """Intelligent batch processor for API requests."""

    def __init__(
        self,
        config: BatchConfig,
        process_batch_func: Callable[[List[T]], List[R]],
        name: str = "BatchProcessor"
    ):
        self.config = config
        self.process_batch_func = process_batch_func
        self.name = name

        # Request queues by priority
        self.request_queues: Dict[int, deque[BatchRequest[T]]] = defaultdict(deque)
        self.current_batch: Optional[RequestBatch[T]] = None

        # Processing state
        self.is_running = False
        self.processor_task: Optional[asyncio.Task] = None
        self.batch_counter = 0

        # Metrics
        self.metrics = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0,
            'queue_sizes': defaultdict(int),
            'timeouts': 0,
            'errors': 0
        }

        # Adaptive batching state
        self.recent_processing_times: deque[float] = deque(maxlen=100)
        self.system_load_history: deque[float] = deque(maxlen=50)

    async def start(self) -> None:
        """Start the batch processor."""
        if self.is_running:
            return

        self.is_running = True
        self.processor_task = asyncio.create_task(self._processing_loop())
        logger.info(f"Started batch processor: {self.name}")

    async def stop(self) -> None:
        """Stop the batch processor."""
        if not self.is_running:
            return

        self.is_running = False

        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

        # Process remaining requests
        await self._flush_all_queues()

        logger.info(f"Stopped batch processor: {self.name}")

    async def submit_request(
        self,
        data: T,
        priority: int = 0,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[R], None]] = None
    ) -> R:
        """Submit a request for batch processing."""
        if not self.is_running:
            raise RuntimeError("Batch processor is not running")

        # Check queue size limits
        total_queue_size = sum(len(queue) for queue in self.request_queues.values())
        if total_queue_size >= self.config.queue_size_limit:
            raise RuntimeError("Batch processor queue is full")

        # Create request
        request_id = f"{self.name}_{self.batch_counter}_{len(self.request_queues[priority])}"
        request = BatchRequest(
            id=request_id,
            data=data,
            priority=priority,
            timeout=timeout,
            callback=callback
        )

        # Add to appropriate priority queue
        self.request_queues[priority].append(request)
        self.metrics['total_requests'] += 1
        self.metrics['queue_sizes'][priority] += 1

        logger.debug(f"Submitted request {request_id} with priority {priority}")

        # Wait for result
        try:
            if timeout:
                result = await asyncio.wait_for(request.future, timeout=timeout)
            else:
                result = await request.future

            # Execute callback if provided
            if callback:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback error for request {request_id}: {e}")

            return result

        except asyncio.TimeoutError:
            self.metrics['timeouts'] += 1
            logger.warning(f"Request {request_id} timed out")
            raise
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Request {request_id} failed: {e}")
            raise

    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while self.is_running:
            try:
                # Determine optimal batch configuration
                batch_config = await self._get_adaptive_batch_config()

                # Create batch from queues
                batch = await self._create_batch(batch_config)

                if batch and batch.size > 0:
                    await self._process_batch(batch)
                else:
                    # No requests to process, wait briefly
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    async def _get_adaptive_batch_config(self) -> Dict[str, Any]:
        """Get adaptive batch configuration based on current conditions."""
        if self.config.strategy != BatchStrategy.ADAPTIVE:
            return {
                'max_size': self.config.max_batch_size,
                'max_wait': self.config.max_wait_time
            }

        # Calculate system load
        system_load = await self._estimate_system_load()
        self.system_load_history.append(system_load)

        # Adjust batch size based on load and recent performance
        if system_load > self.config.adaptive_threshold:
            # High load - smaller batches, faster processing
            max_size = max(1, self.config.max_batch_size // 2)
            max_wait = self.config.max_wait_time * 0.5
        elif system_load < 0.3:
            # Low load - larger batches for efficiency
            max_size = min(self.config.max_batch_size * 2, 50)
            max_wait = self.config.max_wait_time * 1.5
        else:
            # Normal load - default configuration
            max_size = self.config.max_batch_size
            max_wait = self.config.max_wait_time

        # Adjust based on recent processing times
        if self.recent_processing_times:
            avg_processing_time = sum(self.recent_processing_times) / len(self.recent_processing_times)
            if avg_processing_time > 2.0:  # Slow processing
                max_size = max(1, max_size // 2)

        return {
            'max_size': int(max_size),
            'max_wait': max_wait
        }

    async def _estimate_system_load(self) -> float:
        """Estimate current system load."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            return max(cpu_percent, memory_percent) / 100.0
        except ImportError:
            # Fallback to queue-based estimation
            total_queue_size = sum(len(queue) for queue in self.request_queues.values())
            return min(1.0, total_queue_size / self.config.queue_size_limit)

    async def _create_batch(self, batch_config: Dict[str, Any]) -> Optional[RequestBatch[T]]:
        """Create a batch from queued requests."""
        max_size = batch_config['max_size']
        max_wait = batch_config['max_wait']

        # Check if current batch should be processed
        if self.current_batch:
            should_process = (
                self.current_batch.size >= max_size or
                self.current_batch.age >= max_wait or
                self.current_batch.size >= self.config.min_batch_size
            )

            if should_process:
                batch = self.current_batch
                self.current_batch = None
                return batch

        # Create new batch if we have requests
        if not any(self.request_queues.values()):
            return None

        # Start new batch
        batch_id = f"{self.name}_batch_{self.batch_counter}"
        self.batch_counter += 1

        batch = RequestBatch(
            id=batch_id,
            requests=[],
            strategy=self.config.strategy
        )

        # Add requests by priority (higher priority first)
        added_count = 0
        for priority in sorted(self.request_queues.keys(), reverse=True):
            queue = self.request_queues[priority]

            while queue and added_count < max_size:
                request = queue.popleft()
                batch.add_request(request)
                added_count += 1
                self.metrics['queue_sizes'][priority] -= 1

        if batch.size > 0:
            self.current_batch = batch

            # Wait for more requests or timeout
            await asyncio.sleep(min(0.1, max_wait))

            return None  # Will be processed in next iteration

        return None

    async def _process_batch(self, batch: RequestBatch[T]) -> None:
        """Process a batch of requests."""
        start_time = time.time()

        try:
            logger.debug(f"Processing batch {batch.id} with {batch.size} requests")

            # Extract data for processing
            batch_data = batch.get_data()

            # Process batch
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.process_batch_func, batch_data
            )

            # Distribute results to futures
            if len(results) != len(batch.requests):
                raise ValueError(f"Result count mismatch: {len(results)} != {len(batch.requests)}")

            for request, result in zip(batch.requests, results):
                if not request.future.done():
                    request.future.set_result(result)

            # Update metrics
            processing_time = time.time() - start_time
            self.recent_processing_times.append(processing_time)

            self.metrics['total_batches'] += 1
            self.metrics['avg_batch_size'] = (
                (self.metrics['avg_batch_size'] * (self.metrics['total_batches'] - 1) + batch.size) /
                self.metrics['total_batches']
            )
            self.metrics['avg_processing_time'] = (
                (self.metrics['avg_processing_time'] * (self.metrics['total_batches'] - 1) + processing_time) /
                self.metrics['total_batches']
            )

            logger.debug(f"Processed batch {batch.id} in {processing_time:.3f}s")

        except Exception as e:
            logger.error(f"Error processing batch {batch.id}: {e}")

            # Set exception for all requests in batch
            for request in batch.requests:
                if not request.future.done():
                    request.future.set_exception(e)

            self.metrics['errors'] += 1

    async def _flush_all_queues(self) -> None:
        """Process all remaining requests in queues."""
        while any(self.request_queues.values()) or self.current_batch:
            batch_config = {
                'max_size': self.config.max_batch_size,
                'max_wait': 0.0  # No waiting during shutdown
            }

            batch = await self._create_batch(batch_config)
            if batch:
                await self._process_batch(batch)
            else:
                break

    def get_metrics(self) -> Dict[str, Any]:
        """Get batch processor metrics."""
        return {
            **self.metrics,
            'current_queue_sizes': {
                priority: len(queue)
                for priority, queue in self.request_queues.items()
            },
            'current_batch_size': self.current_batch.size if self.current_batch else 0,
            'is_running': self.is_running
        }

    async def health_check(self) -> bool:
        """Check if batch processor is healthy."""
        if not self.is_running:
            return False

        # Check if processor task is running
        if self.processor_task and self.processor_task.done():
            return False

        # Check queue sizes
        total_queue_size = sum(len(queue) for queue in self.request_queues.values())
        if total_queue_size >= self.config.queue_size_limit * 0.9:
            return False

        return True


class BatchProcessorManager:
    """Manages multiple batch processors."""

    def __init__(self):
        self.processors: Dict[str, BatchProcessor] = {}

    def add_processor(self, name: str, processor: BatchProcessor) -> None:
        """Add a batch processor."""
        self.processors[name] = processor

    async def start_all(self) -> None:
        """Start all batch processors."""
        for name, processor in self.processors.items():
            await processor.start()
            logger.info(f"Started batch processor: {name}")

    async def stop_all(self) -> None:
        """Stop all batch processors."""
        for name, processor in self.processors.items():
            await processor.stop()
            logger.info(f"Stopped batch processor: {name}")

    def get_processor(self, name: str) -> Optional[BatchProcessor]:
        """Get batch processor by name."""
        return self.processors.get(name)

    async def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all processors."""
        metrics = {}
        for name, processor in self.processors.items():
            metrics[name] = processor.get_metrics()
        return metrics

    async def health_check_all(self) -> Dict[str, bool]:
        """Health check for all processors."""
        health = {}
        for name, processor in self.processors.items():
            health[name] = await processor.health_check()
        return health


# Global batch processor manager
batch_manager = BatchProcessorManager()
