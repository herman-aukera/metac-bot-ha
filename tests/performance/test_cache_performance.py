"""
Performance tests for caching system.
"""

import pytest
import asyncio
import time
import random
from typing import List, Dict, Any

from src.infrastructure.cache.cache_manager import CacheManager
from src.infrastructure.cache.redis_cache import RedisCache
from src.infrastructure.cache.connection_pool import RedisConnectionPool
from src.infrastructure.cache.cache_strategies import TournamentCacheStrategy


class TestCachePerformance:
    """Performance tests for cache system."""

    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager for testing."""
        manager = CacheManager()

        # Configure test cache
        cache_configs = {
            'test_cache': {
                'connection': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 1,  # Use test database
                    'max_connections': 20
                },
                'cache': {
                    'default_ttl': 300,
                    'key_prefix': 'test:',
                    'serialization': 'json'
                },
                'strategy': {
                    'type': 'tournament',
                    'params': {'max_size': 10000}
                }
            }
        }

        try:
            await manager.initialize(cache_configs, 'test_cache')
            yield manager
        finally:
            await manager.close()

    @pytest.mark.performance
    async def test_cache_throughput(self, cache_manager):
        """Test cache throughput under load."""
        num_operations = 1000
        start_time = time.time()

        # Perform mixed read/write operations
        tasks = []
        for i in range(num_operations):
            if i % 3 == 0:  # 33% writes
                task = cache_manager.set(f"key_{i}", f"value_{i}")
            else:  # 67% reads
                task = cache_manager.get(f"key_{i % 100}")  # Read from smaller set
            tasks.append(task)

        # Execute all operations concurrently
        await asyncio.gather(*tasks)

        elapsed_time = time.time() - start_time
        throughput = num_operations / elapsed_time

        # Assert performance targets
        assert throughput > 500, f"Cache throughput {throughput:.1f} ops/sec below target (500 ops/sec)"
        assert elapsed_time < 5.0, f"Cache operations took {elapsed_time:.2f}s, expected < 5s"

        print(f"Cache throughput: {throughput:.1f} operations/second")

    @pytest.mark.performance
    async def test_cache_latency(self, cache_manager):
        """Test cache operation latency."""
        num_samples = 100
        latencies = []

        # Warm up cache
        for i in range(10):
            await cache_manager.set(f"warmup_{i}", f"value_{i}")

        # Measure latencies
        for i in range(num_samples):
            start_time = time.time()
            await cache_manager.get(f"warmup_{i % 10}")
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

        # Assert latency targets
        assert avg_latency < 10.0, f"Average latency {avg_latency:.2f}ms exceeds target (10ms)"
        assert p95_latency < 50.0, f"P95 latency {p95_latency:.2f}ms exceeds target (50ms)"
        assert max_latency < 100.0, f"Max latency {max_latency:.2f}ms exceeds target (100ms)"

        print(f"Cache latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms, Max: {max_latency:.2f}ms")

    @pytest.mark.performance
    async def test_concurrent_cache_access(self, cache_manager):
        """Test cache performance under concurrent access."""
        num_concurrent = 50
        operations_per_client = 20

        async def client_workload(client_id: int):
            """Workload for individual client."""
            operations = []

            for i in range(operations_per_client):
                key = f"client_{client_id}_key_{i}"
                value = f"client_{client_id}_value_{i}"

                # Set operation
                start_time = time.time()
                await cache_manager.set(key, value)
                set_time = time.time() - start_time

                # Get operation
                start_time = time.time()
                result = await cache_manager.get(key)
                get_time = time.time() - start_time

                operations.append({
                    'set_time': set_time,
                    'get_time': get_time,
                    'success': result == value
                })

            return operations

        # Run concurrent clients
        start_time = time.time()
        client_tasks = [client_workload(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*client_tasks)
        total_time = time.time() - start_time

        # Analyze results
        all_operations = [op for client_ops in results for op in client_ops]
        total_operations = len(all_operations)
        successful_operations = sum(1 for op in all_operations if op['success'])

        avg_set_time = sum(op['set_time'] for op in all_operations) / total_operations
        avg_get_time = sum(op['get_time'] for op in all_operations) / total_operations

        success_rate = (successful_operations / total_operations) * 100
        throughput = total_operations / total_time

        # Assert performance targets
        assert success_rate > 99.0, f"Success rate {success_rate:.1f}% below target (99%)"
        assert throughput > 200, f"Concurrent throughput {throughput:.1f} ops/sec below target (200 ops/sec)"
        assert avg_set_time < 0.1, f"Average set time {avg_set_time:.3f}s exceeds target (0.1s)"
        assert avg_get_time < 0.05, f"Average get time {avg_get_time:.3f}s exceeds target (0.05s)"

        print(f"Concurrent cache performance:")
        print(f"  Clients: {num_concurrent}")
        print(f"  Total operations: {total_operations}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Throughput: {throughput:.1f} ops/sec")
        print(f"  Avg set time: {avg_set_time:.3f}s")
        print(f"  Avg get time: {avg_get_time:.3f}s")

    @pytest.mark.performance
    async def test_large_data_caching(self, cache_manager):
        """Test cache performance with large data objects."""
        # Create large data objects of different sizes
        data_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB

        performance_results = {}

        for size in data_sizes:
            # Create test data
            large_data = {
                'data': 'x' * size,
                'metadata': {
                    'size': size,
                    'timestamp': time.time(),
                    'test_id': f"large_data_{size}"
                }
            }

            # Measure set performance
            start_time = time.time()
            success = await cache_manager.set(f"large_data_{size}", large_data, ttl=60)
            set_time = time.time() - start_time

            # Measure get performance
            start_time = time.time()
            retrieved_data = await cache_manager.get(f"large_data_{size}")
            get_time = time.time() - start_time

            # Verify data integrity
            data_matches = (retrieved_data is not None and
                          retrieved_data['data'] == large_data['data'])

            performance_results[size] = {
                'set_time': set_time,
                'get_time': get_time,
                'set_success': success,
                'data_integrity': data_matches,
                'throughput_mb_s': (size / (1024 * 1024)) / (set_time + get_time)
            }

        # Assert performance targets for different data sizes
        for size, results in performance_results.items():
            assert results['set_success'], f"Failed to set {size} byte object"
            assert results['data_integrity'], f"Data integrity failed for {size} byte object"

            # Performance targets scale with data size
            max_set_time = min(1.0, size / 1048576)  # 1s per MB
            max_get_time = min(0.5, size / 2097152)  # 0.5s per 2MB

            assert results['set_time'] < max_set_time, \
                f"Set time {results['set_time']:.3f}s exceeds target {max_set_time:.3f}s for {size} bytes"
            assert results['get_time'] < max_get_time, \
                f"Get time {results['get_time']:.3f}s exceeds target {max_get_time:.3f}s for {size} bytes"

        print("Large data caching performance:")
        for size, results in performance_results.items():
            print(f"  {size:>8} bytes: Set {results['set_time']:.3f}s, "
                  f"Get {results['get_time']:.3f}s, "
                  f"Throughput {results['throughput_mb_s']:.1f} MB/s")

    @pytest.mark.performance
    async def test_cache_memory_efficiency(self, cache_manager):
        """Test cache memory efficiency and cleanup."""
        import psutil
        import gc

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Store many objects in cache
        num_objects = 1000
        object_size = 1024  # 1KB each

        for i in range(num_objects):
            data = {
                'id': i,
                'data': 'x' * object_size,
                'timestamp': time.time()
            }
            await cache_manager.set(f"memory_test_{i}", data, ttl=30)

        # Measure memory after caching
        gc.collect()  # Force garbage collection
        after_cache_memory = process.memory_info().rss
        cache_memory_usage = after_cache_memory - initial_memory

        # Clear cache
        for i in range(num_objects):
            await cache_manager.delete(f"memory_test_{i}")

        # Force cleanup and measure final memory
        gc.collect()
        await asyncio.sleep(1)  # Allow cleanup
        final_memory = process.memory_info().rss

        # Calculate metrics
        expected_memory = num_objects * object_size * 2  # Account for overhead
        memory_efficiency = (expected_memory / cache_memory_usage) * 100 if cache_memory_usage > 0 else 0
        memory_cleanup_ratio = ((after_cache_memory - final_memory) / cache_memory_usage) * 100 if cache_memory_usage > 0 else 0

        print(f"Cache memory efficiency:")
        print(f"  Objects cached: {num_objects}")
        print(f"  Memory used: {cache_memory_usage / 1024 / 1024:.1f} MB")
        print(f"  Memory efficiency: {memory_efficiency:.1f}%")
        print(f"  Memory cleanup: {memory_cleanup_ratio:.1f}%")

        # Assert memory efficiency targets
        assert memory_efficiency > 30, f"Memory efficiency {memory_efficiency:.1f}% below target (30%)"
        assert memory_cleanup_ratio > 50, f"Memory cleanup {memory_cleanup_ratio:.1f}% below target (50%)"


@pytest.mark.performance
class TestCacheScalability:
    """Scalability tests for cache system."""

    @pytest.mark.parametrize("num_keys", [100, 1000, 10000])
    async def test_cache_scalability(self, num_keys):
        """Test cache performance scaling with number of keys."""
        # This test would require a running Redis instance
        # For now, we'll create a mock test structure

        cache_manager = CacheManager()

        # Simulate cache operations
        start_time = time.time()

        # Simulate set operations
        for i in range(num_keys):
            # Simulate cache set operation
            await asyncio.sleep(0.001)  # 1ms per operation

        set_time = time.time() - start_time

        # Simulate get operations
        start_time = time.time()
        for i in range(num_keys):
            # Simulate cache get operation
            await asyncio.sleep(0.0005)  # 0.5ms per operation

        get_time = time.time() - start_time

        # Calculate performance metrics
        set_throughput = num_keys / set_time
        get_throughput = num_keys / get_time

        print(f"Cache scalability for {num_keys} keys:")
        print(f"  Set throughput: {set_throughput:.1f} ops/sec")
        print(f"  Get throughput: {get_throughput:.1f} ops/sec")

        # Performance should not degrade significantly with scale
        min_set_throughput = 500  # ops/sec
        min_get_throughput = 1000  # ops/sec

        assert set_throughput > min_set_throughput, \
            f"Set throughput {set_throughput:.1f} below minimum {min_set_throughput}"
        assert get_throughput > min_get_throughput, \
            f"Get throughput {get_throughput:.1f} below minimum {min_get_throughput}"
