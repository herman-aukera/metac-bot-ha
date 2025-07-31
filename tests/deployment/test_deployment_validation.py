#!/usr/bin/env python3
"""
Tests for deployment validation and smoke testing functionality.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import subprocess

# Import the scripts we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))

from deployment_validation import DeploymentValidator
from smoke_tests import SmokeTestRunner
from deployment_health_monitor import DeploymentHealthMonitor


class TestDeploymentValidator:
    """Test deployment validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create a deployment validator instance."""
        return DeploymentValidator('staging', strict=False)

    @pytest.mark.asyncio
    async def test_validate_kubernetes_deployment_success(self, validator):
        """Test successful Kubernetes deployment validation."""
        mock_deployment = {
            'status': {
                'replicas': 3,
                'readyReplicas': 3,
                'availableReplicas': 3,
                'conditions': [
                    {'type': 'Available', 'status': 'True'}
                ]
            }
        }

        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = json.dumps(mock_deployment)
            mock_run.return_value.returncode = 0

            result = await validator.validate_kubernetes_deployment()

            assert result['status'] == 'passed'
            assert result['ready_replicas'] == 3
            assert result['desired_replicas'] == 3
            assert result['available_replicas'] == 3

    @pytest.mark.asyncio
    async def test_validate_kubernetes_deployment_failure(self, validator):
        """Test failed Kubernetes deployment validation."""
        mock_deployment = {
            'status': {
                'replicas': 3,
                'readyReplicas': 1,
                'availableReplicas': 1,
                'conditions': [
                    {'type': 'Available', 'status': 'False'}
                ]
            }
        }

        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = json.dumps(mock_deployment)
            mock_run.return_value.returncode = 0

            result = await validator.validate_kubernetes_deployment()

            assert result['status'] == 'failed'
            assert result['ready_replicas'] == 1
            assert result['desired_replicas'] == 3

    @pytest.mark.asyncio
    async def test_validate_service_endpoints(self, validator):
        """Test service endpoints validation."""
        mock_service = {
            'spec': {
                'ports': [
                    {'port': 8000, 'name': 'http'},
                    {'port': 9090, 'name': 'metrics'}
                ]
            }
        }

        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = json.dumps(mock_service)
            mock_run.return_value.returncode = 0

            result = await validator.validate_service_endpoints()

            assert result['status'] == 'passed'
            assert 8000 in result['actual_ports']
            assert 9090 in result['actual_ports']
            assert len(result['missing_ports']) == 0

    @pytest.mark.asyncio
    async def test_validate_configuration_success(self, validator):
        """Test successful configuration validation."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'valid': True,
            'missing_configs': [],
            'invalid_configs': [],
            'environment': 'staging'
        })

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await validator.validate_configuration()

            assert result['status'] == 'passed'
            assert result['environment'] == 'staging'
            assert len(result['missing_configs']) == 0

    @pytest.mark.asyncio
    async def test_validate_feature_flags(self, validator):
        """Test feature flags validation."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'flags': {
                'circuit_breaker_protection': {'enabled': True},
                'performance_monitoring': {'enabled': True},
                'blue_green_deployment': {'enabled': True},
                'other_flag': {'enabled': False}
            },
            'environment_config': {'default_rollout_percentage': 50.0}
        })

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await validator.validate_feature_flags()

            assert result['status'] == 'passed'
            assert result['total_flags'] == 4
            assert result['enabled_flags'] == 3
            assert len(result['missing_critical']) == 0

    @pytest.mark.asyncio
    async def test_validate_performance_metrics(self, validator):
        """Test performance metrics validation."""
        mock_metrics = """
        tournament_optimization_response_time_seconds 15.5
        tournament_optimization_memory_usage_bytes 1073741824
        tournament_optimization_error_rate 0.02
        tournament_optimization_requests_total 1000
        """

        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=mock_metrics)

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await validator.validate_performance_metrics()

            assert result['status'] == 'passed'
            assert result['response_time'] == 15.5
            assert result['memory_usage_gb'] == 1.0
            assert result['error_rate'] == 0.02
            assert len(result['validation_errors']) == 0

    @pytest.mark.asyncio
    async def test_validate_security_headers(self, validator):
        """Test security headers validation."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await validator.validate_security_headers()

            assert result['status'] == 'passed'
            assert len(result['missing_headers']) == 0
            assert len(result['present_headers']) == 4

    @pytest.mark.asyncio
    async def test_run_all_validations(self, validator):
        """Test running all validations together."""
        # Mock all validation methods to return success
        validator.validate_kubernetes_deployment = AsyncMock(return_value={'status': 'passed'})
        validator.validate_service_endpoints = AsyncMock(return_value={'status': 'passed'})
        validator.validate_configuration = AsyncMock(return_value={'status': 'passed'})
        validator.validate_feature_flags = AsyncMock(return_value={'status': 'passed'})
        validator.validate_performance_metrics = AsyncMock(return_value={'status': 'passed'})
        validator.validate_security_headers = AsyncMock(return_value={'status': 'passed'})
        validator.validate_database_migrations = AsyncMock(return_value={'status': 'passed'})

        results = await validator.run_all_validations()

        assert len(results) == 7
        assert all(result['status'] == 'passed' for result in results)


class TestSmokeTestRunner:
    """Test smoke testing functionality."""

    @pytest.fixture
    def smoke_runner(self):
        """Create a smoke test runner instance."""
        return SmokeTestRunner('https://test.example.com', timeout=60)

    @pytest.mark.asyncio
    async def test_health_endpoint_success(self, smoke_runner):
        """Test successful health endpoint check."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'status': 'healthy'})
        mock_response.headers = {'X-Response-Time': '150ms'}

        smoke_runner.session = Mock()
        smoke_runner.session.get.return_value.__aenter__.return_value = mock_response

        result = await smoke_runner.test_health_endpoint()

        assert result['status'] == 'passed'
        assert result['response_time'] == '150ms'
        assert result['data']['status'] == 'healthy'

    @pytest.mark.asyncio
    async def test_health_endpoint_failure(self, smoke_runner):
        """Test failed health endpoint check."""
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value='Internal Server Error')

        smoke_runner.session = Mock()
        smoke_runner.session.get.return_value.__aenter__.return_value = mock_response

        result = await smoke_runner.test_health_endpoint()

        assert result['status'] == 'failed'
        assert 'HTTP 500' in result['error']

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, smoke_runner):
        """Test metrics endpoint check."""
        mock_metrics = """
        # HELP tournament_optimization_requests_total Total requests
        tournament_optimization_requests_total 1000
        tournament_optimization_response_time_seconds 0.5
        """

        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=mock_metrics)

        smoke_runner.session = Mock()
        smoke_runner.session.get.return_value.__aenter__.return_value = mock_response

        result = await smoke_runner.test_metrics_endpoint()

        assert result['status'] == 'passed'
        assert result['metrics_count'] == 2

    @pytest.mark.asyncio
    async def test_api_endpoints(self, smoke_runner):
        """Test API endpoints check."""
        mock_response = Mock()
        mock_response.status = 200

        smoke_runner.session = Mock()
        smoke_runner.session.post.return_value.__aenter__.return_value = mock_response

        results = await smoke_runner.test_api_endpoints()

        assert len(results) == 1
        assert results[0]['name'] == 'forecast_api'
        assert results[0]['status'] == 'passed'

    @pytest.mark.asyncio
    async def test_wait_for_deployment_success(self, smoke_runner):
        """Test successful deployment wait."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'status': 'healthy'})

        smoke_runner.session = Mock()
        smoke_runner.session.get.return_value.__aenter__.return_value = mock_response

        # Mock time to avoid actual waiting
        with patch('time.time', side_effect=[0, 1]):  # Simulate immediate success
            result = await smoke_runner.wait_for_deployment()

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_deployment_timeout(self, smoke_runner):
        """Test deployment wait timeout."""
        smoke_runner.timeout = 1  # Very short timeout

        # Mock session to always fail
        smoke_runner.session = Mock()
        smoke_runner.session.get.side_effect = Exception("Connection failed")

        with patch('asyncio.sleep'):  # Mock sleep to avoid actual waiting
            result = await smoke_runner.wait_for_deployment()

        assert result is False


class TestDeploymentHealthMonitor:
    """Test deployment health monitoring functionality."""

    @pytest.fixture
    def health_monitor(self):
        """Create a deployment health monitor instance."""
        return DeploymentHealthMonitor('staging', duration=60)

    @pytest.mark.asyncio
    async def test_collect_health_metrics_success(self, health_monitor):
        """Test successful health metrics collection."""
        mock_health_response = Mock()
        mock_health_response.status = 200
        mock_health_response.json = AsyncMock(return_value={'status': 'healthy'})

        mock_metrics_response = Mock()
        mock_metrics_response.status = 200
        mock_metrics_response.text = AsyncMock(return_value="""
        tournament_optimization_response_time_seconds 1.5
        tournament_optimization_error_rate 0.01
        tournament_optimization_memory_usage_ratio 0.6
        tournament_optimization_cpu_usage_ratio 0.4
        tournament_optimization_active_connections 50
        tournament_optimization_queue_length 10
        """)

        health_monitor.session = Mock()
        health_monitor.session.get.side_effect = [
            Mock(__aenter__=AsyncMock(return_value=mock_health_response)),
            Mock(__aenter__=AsyncMock(return_value=mock_metrics_response))
        ]

        metrics = await health_monitor.collect_health_metrics()

        assert metrics is not None
        assert metrics.response_time == 1.5
        assert metrics.error_rate == 0.01
        assert metrics.memory_usage == 0.6
        assert metrics.cpu_usage == 0.4
        assert metrics.active_connections == 50
        assert metrics.queue_length == 10

    def test_check_health_thresholds_pass(self, health_monitor):
        """Test health threshold checking with passing metrics."""
        from deployment_health_monitor import HealthMetrics

        metrics = HealthMetrics(
            timestamp=datetime.now(),
            response_time=5.0,
            error_rate=0.02,
            memory_usage=0.7,
            cpu_usage=0.6,
            active_connections=100,
            queue_length=20
        )

        violations = health_monitor.check_health_thresholds(metrics)

        assert len(violations) == 0

    def test_check_health_thresholds_violations(self, health_monitor):
        """Test health threshold checking with violations."""
        from deployment_health_monitor import HealthMetrics

        metrics = HealthMetrics(
            timestamp=datetime.now(),
            response_time=35.0,  # Above 30s threshold
            error_rate=0.08,     # Above 5% threshold
            memory_usage=0.9,    # Above 85% threshold
            cpu_usage=0.85,      # Above 80% threshold
            active_connections=100,
            queue_length=20
        )

        violations = health_monitor.check_health_thresholds(metrics)

        assert len(violations) == 4
        assert any('Response time too high' in v for v in violations)
        assert any('Error rate too high' in v for v in violations)
        assert any('Memory usage too high' in v for v in violations)
        assert any('CPU usage too high' in v for v in violations)

    @pytest.mark.asyncio
    async def test_trigger_rollback_success(self, health_monitor):
        """Test successful rollback trigger."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Rollback completed successfully"

            # Mock successful health check after rollback
            health_monitor.collect_health_metrics = AsyncMock(return_value=Mock(
                response_time=5.0,
                error_rate=0.01,
                memory_usage=0.6,
                cpu_usage=0.5
            ))
            health_monitor.check_health_thresholds = Mock(return_value=[])

            result = await health_monitor.trigger_rollback("Test rollback")

            assert result is True
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_rollback_failure(self, health_monitor):
        """Test failed rollback trigger."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'deploy.sh', stderr="Rollback failed")

            result = await health_monitor.trigger_rollback("Test rollback")

            assert result is False

    def test_analyze_trends_degrading(self, health_monitor):
        """Test trend analysis detecting degradation."""
        from deployment_health_monitor import HealthMetrics

        # Create metrics history showing degradation
        base_time = datetime.now()

        # Older metrics (good performance)
        for i in range(5):
            metrics = HealthMetrics(
                timestamp=base_time,
                response_time=2.0,
                error_rate=0.01,
                memory_usage=0.5,
                cpu_usage=0.4,
                active_connections=50,
                queue_length=10
            )
            health_monitor.metrics_history.append(metrics)

        # Recent metrics (degraded performance)
        for i in range(5):
            metrics = HealthMetrics(
                timestamp=base_time,
                response_time=8.0,  # Much higher
                error_rate=0.05,    # Much higher
                memory_usage=0.8,
                cpu_usage=0.7,
                active_connections=100,
                queue_length=30
            )
            health_monitor.metrics_history.append(metrics)

        trends = health_monitor.analyze_trends()

        assert trends['degrading'] is True
        assert trends['response_time_trend'] > 0
        assert trends['error_rate_trend'] > 0


@pytest.mark.integration
class TestDeploymentIntegration:
    """Integration tests for deployment validation."""

    @pytest.mark.asyncio
    async def test_full_deployment_validation_pipeline(self):
        """Test the complete deployment validation pipeline."""
        # This would be a more comprehensive test that actually
        # validates against a test deployment
        pass

    @pytest.mark.asyncio
    async def test_smoke_test_pipeline(self):
        """Test the complete smoke test pipeline."""
        # This would test the full smoke test suite against
        # a test environment
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
