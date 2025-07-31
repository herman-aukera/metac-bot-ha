"""
Performance tests for concurrent question processing.
"""

import pytest
import asyncio
import time
import random
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

from src.infrastructure.external_apis.batch_processor import BatchProcessor, BatchConfig, BatchStrategy
from src.infrastructure.performance.memory_optimizer import MemoryOptimizer
from src.infrastructure.performance.auto_scaler import AutoScaler, ResourceMonitor, ScalingPolicy, ResourceType


class MockQuestion:
    """Mock question for testing."""

    def __init__(self, question_id: int, complexity: str = "medium"):
        self.id = question_id
        self.title = f"Test Question {question_id}"
        self.complexity = complexity
        self.processing_time = {
            "simple": 0.1,
            "medium": 0.5,
            "complex": 2.0
        }.get(complexity, 0.5)


class MockForecastingPipeline:
    """Mock forecasting pipeline for testing."""

    def __init__(self, base_processing_time: float = 0.5):
        self.base_processing_time = base_processing_time
        self.processed_questions = []

    async def process_question(self, question: MockQuestion) -> Dict[str, Any]:
        """Mock question processing."""
        # Simulate processing time
        processing_time = question.processing_time + random.uniform(-0.1, 0.1)
        await asyncio.sleep(processing_time)

        result = {
            'question_id': question.id,
            'prediction': random.uniform(0.1, 0.9),
            'confidence': random.uniform(0.6, 0.95),
            'processing_time': processing_time
        }

        self.processed_questions.append(result)
        return result


@pytest.mark.performance
class TestConcurrentProcessing:
    """Performance tests for concurrent question processing."""

    @pytest.fixture
    def forecasting_pipeline(self):
        """Create mock forecasting pipeline."""
        return MockForecastingPipeline()

    @pytest.fixture
    def memory_optimizer(self):
        """Create memory optimizer."""
        return MemoryOptimizer(memory_threshold=0.9)

    async def test_concurrent_question_processing(self, forecasting_pipeline):
        """Test processing 100+ concurrent questions."""
        num_questions = 100
        questions = [MockQuestion(i) for i in range(num_questions)]

        start_time = time.time()

        # Process questions concurrently
        tasks = [
            forecasting_pipeline.process_question(question)
            for question in questions
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Verify results
        assert len(results) == num_questions
        assert len(forecasting_pipeline.processed_questions) == num_questions

        # Performance assertions
        assert total_time < 10.0, f"Processing {num_questions} questions took {total_time:.2f}s, expected < 10s"

        # Calculate throughput
        throughput = num_questions / total_time
        assert throughput > 20, f"Throughput {throughput:.1f} questions/sec below target (20 q/s)"

        # Check individual processing times
        processing_times = [result['processing_time'] for result in results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)

        assert avg_processing_time < 1.0, f"Average processing time {avg_processing_time:.2f}s exceeds target (1s)"
        assert max_processing_time < 3.0, f"Max processing time {max_processing_time:.2f}s exceeds target (3s)"

        print(f"Concurrent processing performance:")
        print(f"  Questions: {num_questions}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} questions/sec")
        print(f"  Avg processing time: {avg_processing_time:.2f}s")
        print(f"  Max processing time: {max_processing_time:.2f}s")

    async def test_large_tournament_processing(self, forecasting_pipeline):
        """Test processing 1000+ questions for large tournament."""
        num_questions = 1000

        # Create questions with varying complexity
        questions = []
        for i in range(num_questions):
            complexity = random.choice(["simple", "medium", "complex"])
            questions.append(MockQuestion(i, complexity))

        # Process in batches to avoid overwhelming the system
        batch_size = 50
        batches = [questions[i:i + batch_size] for i in range(0, num_questions, batch_size)]

        start_time = time.time()
        all_results = []

        for batch in batches:
            batch_tasks = [
                forecasting_pipeline.process_question(question)
                for question in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)

            # Small delay between batches to prevent resource exhaustion
            await asyncio.sleep(0.1)

        total_time = time.time() - start_time

        # Verify results
        assert len(all_results) == num_questions

        # Performance assertions for large tournament
        assert total_time < 300.0, f"Processing {num_questions} questions took {total_time:.2f}s, expected < 300s"

        throughput = num_questions / total_time
        assert throughput > 5, f"Large tournament throughput {throughput:.1f} q/s below target (5 q/s)"

        # Analyze performance by complexity
        complexity_stats = {}
        for result in all_results:
            question = questions[result['question_id']]
            complexity = question.complexity

            if complexity not in complexity_stats:
                complexity_stats[complexity] = []
            complexity_stats[complexity].append(result['processing_time'])

        print(f"Large tournament processing performance:")
        print(f"  Questions: {num_questions}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} questions/sec")

        for complexity, times in complexity_stats.items():
            avg_time = sum(times) / len(times)
            print(f"  {complexity.capitalize()} questions: {len(times)} processed, avg time: {avg_time:.2f}s")

    async def test_batch_processing_performance(self):
        """Test batch processor performance."""

        def process_batch(items: List[int]) -> List[int]:
            """Mock batch processing function."""
            # Simulate processing time
            time.sleep(0.01 * len(items))  # 10ms per item
            return [item * 2 for item in items]

        # Configure batch processor
        config = BatchConfig(
            max_batch_size=20,
            max_wait_time=0.5,
            strategy=BatchStrategy.ADAPTIVE
        )

        processor = BatchProcessor(config, process_batch, "test_processor")

        try:
            await processor.start()

            # Submit many requests concurrently
            num_requests = 200
            start_time = time.time()

            tasks = [
                processor.submit_request(i, priority=random.randint(0, 2))
                for i in range(num_requests)
            ]

            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            # Verify results
            assert len(results) == num_requests
            for i, result in enumerate(results):
                assert result == i * 2, f"Incorrect result for item {i}: {result} != {i * 2}"

            # Performance assertions
            throughput = num_requests / total_time
            assert throughput > 50, f"Batch processing throughput {throughput:.1f} req/s below target (50 req/s)"

            # Get batch processor metrics
            metrics = processor.get_metrics()

            print(f"Batch processing performance:")
            print(f"  Requests: {num_requests}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.1f} requests/sec")
            print(f"  Total batches: {metrics['total_batches']}")
            print(f"  Avg batch size: {metrics['avg_batch_size']:.1f}")
            print(f"  Avg processing time: {metrics['avg_processing_time']:.3f}s")

            # Assert batch efficiency
            assert metrics['avg_batch_size'] > 5, f"Average batch size {metrics['avg_batch_size']:.1f} too small"

        finally:
            await processor.stop()

    async def test_memory_optimization_under_load(self, memory_optimizer):
        """Test memory optimization during high load processing."""

        # Start memory monitoring
        await memory_optimizer.start_monitoring()

        try:
            # Create large dataset for processing
            large_dataset = []
            for i in range(10000):
                large_dataset.append({
                    'id': i,
                    'data': f"Large data item {i} with substantial content " * 10,
                    'metadata': {
                        'timestamp': time.time(),
                        'category': f"category_{i % 10}",
                        'tags': [f"tag_{j}" for j in range(5)]
                    }
                })

            # Process dataset in chunks with memory monitoring
            def process_chunk(chunk):
                """Process a chunk of data."""
                processed = []
                for item in chunk:
                    # Simulate processing
                    processed_item = {
                        'id': item['id'],
                        'processed_data': item['data'].upper(),
                        'score': len(item['data'])
                    }
                    processed.append(processed_item)
                return processed

            # Use dataset optimizer
            from src.infrastructure.performance.memory_optimizer import DatasetOptimizer
            dataset_optimizer = DatasetOptimizer(memory_optimizer)

            start_time = time.time()
            results = await dataset_optimizer.process_large_dataset(
                large_dataset,
                process_chunk,
                chunk_size=500
            )
            processing_time = time.time() - start_time

            # Verify results
            assert len(results) == len(large_dataset)

            # Get memory statistics
            final_stats = memory_optimizer.get_memory_stats()

            # Performance assertions
            assert processing_time < 30.0, f"Large dataset processing took {processing_time:.2f}s, expected < 30s"
            assert final_stats.memory_percent < 95.0, f"Memory usage {final_stats.memory_percent:.1f}% too high"

            print(f"Memory optimization performance:")
            print(f"  Dataset size: {len(large_dataset)} items")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Memory usage: {final_stats.memory_usage_mb:.1f}MB ({final_stats.memory_percent:.1f}%)")
            print(f"  Throughput: {len(large_dataset) / processing_time:.1f} items/sec")

        finally:
            await memory_optimizer.stop_monitoring()

    async def test_auto_scaling_performance(self):
        """Test auto-scaling system performance."""

        # Create resource monitor and auto-scaler
        resource_monitor = ResourceMonitor(monitoring_interval=1)
        auto_scaler = AutoScaler(resource_monitor)

        # Add custom metric for testing
        request_count = 0

        def get_request_rate():
            nonlocal request_count
            return request_count / 10.0  # requests per second

        resource_monitor.add_custom_metric_collector('request_rate', get_request_rate)

        # Create scaling policy
        policy = ScalingPolicy(
            name='request_scaling',
            resource_type=ResourceType.CUSTOM,
            scale_up_threshold=50.0,  # 50 requests/sec
            scale_down_threshold=10.0,  # 10 requests/sec
            min_instances=1,
            max_instances=5,
            cooldown_period=5,  # 5 seconds for testing
            evaluation_period=3
        )

        auto_scaler.add_policy(policy)

        # Mock scaling callbacks
        current_instances = 1
        scaling_events = []

        def scale_up_callback(instances):
            nonlocal current_instances
            current_instances = instances
            scaling_events.append(('up', instances))

        def scale_down_callback(instances):
            nonlocal current_instances
            current_instances = instances
            scaling_events.append(('down', instances))

        auto_scaler.set_scale_callbacks('request_scaling', scale_up_callback, scale_down_callback)

        try:
            # Start monitoring and auto-scaling
            await resource_monitor.start_monitoring()
            await auto_scaler.start_auto_scaling()

            # Simulate load increase
            print("Simulating load increase...")
            for i in range(10):
                request_count += 10  # Increase load
                await asyncio.sleep(1)

            # Wait for scaling decision
            await asyncio.sleep(8)

            # Simulate load decrease
            print("Simulating load decrease...")
            request_count = 5  # Low load
            await asyncio.sleep(8)

            # Get scaling statistics
            stats = auto_scaler.get_scaling_stats()

            print(f"Auto-scaling performance:")
            print(f"  Scaling events: {len(scaling_events)}")
            print(f"  Final instances: {current_instances}")
            print(f"  Policy enabled: {stats['policies']['request_scaling']['enabled']}")

            # Verify scaling occurred
            assert len(scaling_events) > 0, "No scaling events occurred"

            # Verify scale-up happened
            scale_up_events = [e for e in scaling_events if e[0] == 'up']
            assert len(scale_up_events) > 0, "No scale-up events occurred"

        finally:
            await auto_scaler.stop_auto_scaling()
            await resource_monitor.stop_monitoring()


@pytest.mark.performance
class TestSystemLimits:
    """Tests for system performance limits and boundaries."""

    async def test_maximum_concurrent_connections(self):
        """Test maximum concurrent connections handling."""
        max_connections = 200
        connection_tasks = []

        async def mock_connection():
            """Mock connection that holds for a short time."""
            await asyncio.sleep(0.5)
            return "connected"

        start_time = time.time()

        # Create maximum concurrent connections
        for i in range(max_connections):
            task = asyncio.create_task(mock_connection())
            connection_tasks.append(task)

        # Wait for all connections to complete
        results = await asyncio.gather(*connection_tasks)
        total_time = time.time() - start_time

        # Verify all connections succeeded
        assert len(results) == max_connections
        assert all(result == "connected" for result in results)

        # Performance assertion
        assert total_time < 2.0, f"Max connections test took {total_time:.2f}s, expected < 2s"

        print(f"Maximum concurrent connections test:")
        print(f"  Connections: {max_connections}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Success rate: 100%")

    async def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""

        # Create memory-intensive objects
        memory_objects = []
        object_size = 1024 * 1024  # 1MB per object
        max_objects = 100

        try:
            start_time = time.time()

            for i in range(max_objects):
                # Create large object
                large_object = {
                    'id': i,
                    'data': 'x' * object_size,
                    'timestamp': time.time()
                }
                memory_objects.append(large_object)

                # Check if we should stop due to memory pressure
                if i % 10 == 0:
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 90:
                        print(f"Stopping at {i} objects due to memory pressure ({memory_percent:.1f}%)")
                        break

            creation_time = time.time() - start_time

            # Verify objects were created
            assert len(memory_objects) > 0, "No objects were created"

            # Clean up
            start_cleanup = time.time()
            memory_objects.clear()
            import gc
            gc.collect()
            cleanup_time = time.time() - start_cleanup

            print(f"Memory pressure handling:")
            print(f"  Objects created: {len(memory_objects)}")
            print(f"  Creation time: {creation_time:.2f}s")
            print(f"  Cleanup time: {cleanup_time:.2f}s")

            # Performance assertions
            assert creation_time < 10.0, f"Object creation took {creation_time:.2f}s, expected < 10s"
            assert cleanup_time < 5.0, f"Cleanup took {cleanup_time:.2f}s, expected < 5s"

        except MemoryError:
            # This is expected under extreme memory pressure
            print("MemoryError encountered - system handled memory pressure correctly")
            assert True  # Test passes if we handle memory pressure gracefully
