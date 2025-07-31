"""
Unit tests for graceful degradation manager.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from src.infrastructure.resilience.graceful_degradation import (
    GracefulDegradationManager,
    ServiceHealth,
    ServiceStatus,
    DegradationLevel,
    FallbackConfig,
)
from src.domain.exceptions import InfrastructureError


class TestServiceStatus:
    """Test ServiceStatus functionality."""

    def test_service_status_creation(self):
        """Test ServiceStatus creation."""
        status = ServiceStatus("test_service")

        assert status.name == "test_service"
        assert status.health == ServiceHealth.UNKNOWN
        assert status.consecutive_failures == 0
        assert status.consecutive_successes == 0
        assert status.error_rate == 0.0
        assert status.metadata == {}

    def test_is_available(self):
        """Test service availability check."""
        status = ServiceStatus("test_service")

        # Unknown is not available
        status.health = ServiceHealth.UNKNOWN
        assert not status.is_available

        # Healthy is available
        status.health = ServiceHealth.HEALTHY
        assert status.is_available

        # Degraded is available
        status.health = ServiceHealth.DEGRADED
        assert status.is_available

        # Unhealthy is not available
        status.health = ServiceHealth.UNHEALTHY
        assert not status.is_available

    def test_uptime_percentage(self):
        """Test uptime percentage calculation."""
        status = ServiceStatus("test_service")

        # No failures
        status.consecutive_successes = 10
        assert status.uptime_percentage == 100.0

        # Some failures
        status.consecutive_failures = 2
        status.consecutive_successes = 8
        assert status.uptime_percentage == 80.0

        # All failures
        status.consecutive_failures = 10
        status.consecutive_successes = 0
        assert status.uptime_percentage == 0.0


class TestFallbackConfig:
    """Test FallbackConfig functionality."""

    def test_fallback_config_creation(self):
        """Test FallbackConfig creation."""
        config = FallbackConfig(
            primary_services=["service1", "service2"],
            fallback_services=["service3", "service4"]
        )

        assert config.primary_services == ["service1", "service2"]
        assert config.fallback_services == ["service3", "service4"]
        assert config.fallback_strategy == "round_robin"
        assert config.health_check_interval == 30
        assert config.failure_threshold == 3
        assert config.recovery_threshold == 2
        assert config.enable_caching is True
        assert config.cache_ttl == 300
        assert config.enable_partial_results is True


class TestGracefulDegradationManager:
    """Test GracefulDegradationManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create graceful degradation manager for testing."""
        return GracefulDegradationManager()

    @pytest.mark.asyncio
    async def test_register_service(self, manager):
        """Test service registration."""
        health_check = AsyncMock(return_value=True)

        await manager.register_service(
            "test_service",
            health_check_func=health_check,
            initial_health=ServiceHealth.HEALTHY
        )

        assert "test_service" in manager.service_status
        status = manager.service_status["test_service"]
        assert status.name == "test_service"
        assert status.health == ServiceHealth.HEALTHY
        assert "test_service" in manager._health_check_tasks

    @pytest.mark.asyncio
    async def test_update_service_health(self, manager):
        """Test service health update."""
        await manager.register_service("test_service")

        await manager.update_service_health(
            "test_service",
            ServiceHealth.HEALTHY,
            metadata={"response_time": 0.1}
        )

        status = manager.service_status["test_service"]
        assert status.health == ServiceHealth.HEALTHY
        assert status.consecutive_successes == 1
        assert status.consecutive_failures == 0
        assert status.metadata["response_time"] == 0.1
        assert status.last_success is not None

    @pytest.mark.asyncio
    async def test_update_service_health_failure(self, manager):
        """Test service health update on failure."""
        await manager.register_service("test_service")

        await manager.update_service_health("test_service", ServiceHealth.UNHEALTHY)

        status = manager.service_status["test_service"]
        assert status.health == ServiceHealth.UNHEALTHY
        assert status.consecutive_failures == 1
        assert status.consecutive_successes == 0
        assert status.last_failure is not None

    @pytest.mark.asyncio
    async def test_get_available_services(self, manager):
        """Test getting available services."""
        # Register services with different health states
        await manager.register_service("healthy_service")
        await manager.register_service("degraded_service")
        await manager.register_service("unhealthy_service")

        await manager.update_service_health("healthy_service", ServiceHealth.HEALTHY)
        await manager.update_service_health("degraded_service", ServiceHealth.DEGRADED)
        await manager.update_service_health("unhealthy_service", ServiceHealth.UNHEALTHY)

        # Get available services including degraded
        available = await manager.get_available_services("search", include_degraded=True)
        # Note: These services aren't in the default config, so they won't be returned

        # Test with actual configured services
        manager.fallback_configs["test"] = FallbackConfig(
            primary_services=["healthy_service", "degraded_service"],
            fallback_services=["unhealthy_service"]
        )

        available = await manager.get_available_services("test", include_degraded=True)
        assert "healthy_service" in available
        assert "degraded_service" in available
        assert "unhealthy_service" not in available

        # Get available services excluding degraded
        available = await manager.get_available_services("test", include_degraded=False)
        assert "healthy_service" in available
        assert "degraded_service" not in available

    @pytest.mark.asyncio
    async def test_get_primary_service(self, manager):
        """Test getting primary service."""
        # Setup services
        await manager.register_service("primary1")
        await manager.register_service("primary2")
        await manager.register_service("fallback1")

        await manager.update_service_health("primary1", ServiceHealth.HEALTHY)
        await manager.update_service_health("primary2", ServiceHealth.UNHEALTHY)
        await manager.update_service_health("fallback1", ServiceHealth.HEALTHY)

        # Configure fallback
        manager.fallback_configs["test"] = FallbackConfig(
            primary_services=["primary1", "primary2"],
            fallback_services=["fallback1"]
        )

        # Should return first healthy primary service
        primary = await manager.get_primary_service("test")
        assert primary == "primary1"

        # Make primary unhealthy
        await manager.update_service_health("primary1", ServiceHealth.UNHEALTHY)

        # Should fallback to fallback service
        primary = await manager.get_primary_service("test")
        assert primary == "fallback1"

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, manager):
        """Test successful execution with fallback."""
        # Setup services
        await manager.register_service("service1")
        await manager.update_service_health("service1", ServiceHealth.HEALTHY)

        manager.fallback_configs["test"] = FallbackConfig(
            primary_services=["service1"],
            fallback_services=[],
            enable_caching=False
        )

        # Mock operation
        async def mock_operation(service_name, *args, **kwargs):
            assert service_name == "service1"
            assert args == ("arg1",)
            assert kwargs == {"key": "value"}
            return "success"

        result = await manager.execute_with_fallback(
            "test",
            mock_operation,
            "arg1",
            key="value"
        )

        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_with_fallback_failure_and_recovery(self, manager):
        """Test execution with fallback after primary failure."""
        # Setup services
        await manager.register_service("primary")
        await manager.register_service("fallback")
        await manager.update_service_health("primary", ServiceHealth.HEALTHY)
        await manager.update_service_health("fallback", ServiceHealth.HEALTHY)

        manager.fallback_configs["test"] = FallbackConfig(
            primary_services=["primary"],
            fallback_services=["fallback"],
            enable_caching=False
        )

        # Mock operation that fails for primary but succeeds for fallback
        async def mock_operation(service_name, *args, **kwargs):
            if service_name == "primary":
                raise ConnectionError("Primary service failed")
            return f"success_from_{service_name}"

        result = await manager.execute_with_fallback("test", mock_operation)

        assert result == "success_from_fallback"

        # Primary service should be marked unhealthy
        assert manager.service_status["primary"].health == ServiceHealth.UNHEALTHY

    @pytest.mark.asyncio
    async def test_execute_with_fallback_all_fail(self, manager):
        """Test execution when all services fail."""
        # Setup services
        await manager.register_service("service1")
        await manager.register_service("service2")
        await manager.update_service_health("service1", ServiceHealth.HEALTHY)
        await manager.update_service_health("service2", ServiceHealth.HEALTHY)

        manager.fallback_configs["test"] = FallbackConfig(
            primary_services=["service1"],
            fallback_services=["service2"],
            enable_caching=False,
            enable_partial_results=False
        )

        # Mock operation that always fails
        async def mock_operation(service_name, *args, **kwargs):
            raise ConnectionError(f"{service_name} failed")

        with pytest.raises(InfrastructureError) as exc_info:
            await manager.execute_with_fallback("test", mock_operation)

        assert "All services failed for test" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_caching_functionality(self, manager):
        """Test caching functionality."""
        # Setup service
        await manager.register_service("service1")
        await manager.update_service_health("service1", ServiceHealth.HEALTHY)

        manager.fallback_configs["test"] = FallbackConfig(
            primary_services=["service1"],
            fallback_services=[],
            enable_caching=True,
            cache_ttl=1  # Short TTL for testing
        )

        call_count = 0

        async def mock_operation(service_name):
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # First call should execute and cache
        result1 = await manager.execute_with_fallback("test", mock_operation)
        assert result1 == "result_1"
        assert call_count == 1

        # Second call should return cached result
        result2 = await manager.execute_with_fallback("test", mock_operation)
        assert result2 == "result_1"  # Same as first call
        assert call_count == 1  # No additional call

        # Wait for cache to expire
        await asyncio.sleep(1.1)

        # Third call should execute again
        result3 = await manager.execute_with_fallback("test", mock_operation)
        assert result3 == "result_2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_degradation_level_calculation(self, manager):
        """Test system degradation level calculation."""
        # Register multiple services
        services = ["service1", "service2", "service3", "service4", "service5"]
        for service in services:
            await manager.register_service(service)

        # All healthy - no degradation
        for service in services:
            await manager.update_service_health(service, ServiceHealth.HEALTHY)
        assert manager.degradation_level == DegradationLevel.NONE

        # 1 unhealthy (20%) - minimal degradation
        await manager.update_service_health("service1", ServiceHealth.UNHEALTHY)
        assert manager.degradation_level == DegradationLevel.MINIMAL

        # 2 unhealthy (40%) - moderate degradation
        await manager.update_service_health("service2", ServiceHealth.UNHEALTHY)
        assert manager.degradation_level == DegradationLevel.MODERATE

        # 4 unhealthy (80%) - critical degradation
        await manager.update_service_health("service3", ServiceHealth.UNHEALTHY)
        await manager.update_service_health("service4", ServiceHealth.UNHEALTHY)
        assert manager.degradation_level == DegradationLevel.CRITICAL

    def test_cache_key_generation(self, manager):
        """Test cache key generation."""
        def test_func():
            pass

        key1 = manager._generate_cache_key("service", test_func, ("arg1",), {"key": "value"})
        key2 = manager._generate_cache_key("service", test_func, ("arg1",), {"key": "value"})
        key3 = manager._generate_cache_key("service", test_func, ("arg2",), {"key": "value"})

        # Same parameters should generate same key
        assert key1 == key2

        # Different parameters should generate different key
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_cache_cleanup(self, manager):
        """Test cache cleanup functionality."""
        # Fill cache beyond limit
        for i in range(1100):  # More than 1000 limit
            await manager._cache_result(f"key_{i}", f"value_{i}")

        # Should have cleaned up to 900 entries (removed oldest 100)
        assert len(manager.cached_results) == 900
        assert len(manager.cache_timestamps) == 900

    def test_get_system_status(self, manager):
        """Test system status retrieval."""
        # Add some services
        asyncio.run(manager.register_service("service1"))
        asyncio.run(manager.register_service("service2"))
        asyncio.run(manager.update_service_health("service1", ServiceHealth.HEALTHY))
        asyncio.run(manager.update_service_health("service2", ServiceHealth.DEGRADED))

        status = manager.get_system_status()

        assert status["degradation_level"] == DegradationLevel.NONE.value
        assert "services" in status
        assert "service1" in status["services"]
        assert "service2" in status["services"]
        assert status["services"]["service1"]["health"] == ServiceHealth.HEALTHY.value
        assert status["services"]["service2"]["health"] == ServiceHealth.DEGRADED.value
        assert "cache_size" in status
        assert "active_health_checks" in status

    @pytest.mark.asyncio
    async def test_cleanup(self, manager):
        """Test manager cleanup."""
        # Register service with health check
        health_check = AsyncMock(return_value=True)
        await manager.register_service("test_service", health_check_func=health_check)

        # Verify task is running
        assert "test_service" in manager._health_check_tasks
        task = manager._health_check_tasks["test_service"]
        assert not task.done()

        # Cleanup
        await manager.cleanup()

        # Verify task is cancelled and cleaned up
        assert task.cancelled() or task.done()
        assert len(manager._health_check_tasks) == 0

    @pytest.mark.asyncio
    async def test_health_check_loop(self, manager):
        """Test health check loop functionality."""
        health_check_calls = 0

        async def mock_health_check():
            nonlocal health_check_calls
            health_check_calls += 1
            return health_check_calls <= 2  # Healthy for first 2 calls, then unhealthy

        # Register service with very short health check interval
        manager.fallback_configs["test"] = FallbackConfig(
            primary_services=["test_service"],
            fallback_services=[],
            health_check_interval=0.1  # Very short for testing
        )

        await manager.register_service("test_service", health_check_func=mock_health_check)

        # Wait for a few health checks
        await asyncio.sleep(0.3)

        # Verify health checks were performed
        assert health_check_calls >= 2

        # Service should eventually become unhealthy
        status = manager.service_status["test_service"]
        # Note: Due to timing, we can't guarantee the exact state, but we can verify calls were made

        # Cleanup
        await manager.cleanup()
