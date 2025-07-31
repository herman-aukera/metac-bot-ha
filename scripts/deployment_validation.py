#!/usr/bin/env python3
"""
Deployment validation script for tournament optimization system.
Validates deployment health, configuration, and performance metrics.
"""

import asyncio
import aiohttp
import argparse
import sys
import json
import time
from typing import Dict, List, Any, Optional
import subprocess


class DeploymentValidator:
    """Validates deployment health and configuration."""

    def __init__(self, environment: str, strict: bool = False):
        self.environment = environment
        self.strict = strict
        self.base_urls = {
            'staging': 'https://tournament-optimization-staging.example.com',
            'production': 'https://tournament-optimization.example.com'
        }
        self.base_url = self.base_urls[environment]
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def validate_kubernetes_deployment(self) -> Dict[str, Any]:
        """Validate Kubernetes deployment status."""
        try:
            # Check deployment status
            result = subprocess.run([
                'kubectl', 'get', 'deployment', 'tournament-optimization',
                '-n', f'tournament-optimization-{self.environment}',
                '-o', 'json'
            ], capture_output=True, text=True, check=True)

            deployment = json.loads(result.stdout)
            status = deployment.get('status', {})

            ready_replicas = status.get('readyReplicas', 0)
            desired_replicas = status.get('replicas', 0)
            available_replicas = status.get('availableReplicas', 0)

            is_healthy = (ready_replicas == desired_replicas and
                         available_replicas == desired_replicas and
                         desired_replicas > 0)

            return {
                'name': 'kubernetes_deployment',
                'status': 'passed' if is_healthy else 'failed',
                'ready_replicas': ready_replicas,
                'desired_replicas': desired_replicas,
                'available_replicas': available_replicas,
                'conditions': status.get('conditions', [])
            }

        except subprocess.CalledProcessError as e:
            return {
                'name': 'kubernetes_deployment',
                'status': 'failed',
                'error': f'kubectl error: {e.stderr}'
            }
        except Exception as e:
            return {
                'name': 'kubernetes_deployment',
                'status': 'failed',
                'error': str(e)
            }

    async def validate_service_endpoints(self) -> Dict[str, Any]:
        """Validate service endpoints are accessible."""
        try:
            result = subprocess.run([
                'kubectl', 'get', 'service', 'tournament-optimization',
                '-n', f'tournament-optimization-{self.environment}',
                '-o', 'json'
            ], capture_output=True, text=True, check=True)

            service = json.loads(result.stdout)
            spec = service.get('spec', {})
            ports = spec.get('ports', [])

            expected_ports = [8000, 9090]  # HTTP and metrics
            actual_ports = [port.get('port') for port in ports]

            missing_ports = set(expected_ports) - set(actual_ports)

            return {
                'name': 'service_endpoints',
                'status': 'passed' if not missing_ports else 'failed',
                'expected_ports': expected_ports,
                'actual_ports': actual_ports,
                'missing_ports': list(missing_ports)
            }

        except Exception as e:
            return {
                'name': 'service_endpoints',
                'status': 'failed',
                'error': str(e)
            }

    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate application configuration."""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/config/validate") as response:
                if response.status == 200:
                    data = await response.json()

                    config_valid = data.get('valid', False)
                    missing_configs = data.get('missing_configs', [])
                    invalid_configs = data.get('invalid_configs', [])

                    return {
                        'name': 'configuration_validation',
                        'status': 'passed' if config_valid else 'failed',
                        'missing_configs': missing_configs,
                        'invalid_configs': invalid_configs,
                        'environment': data.get('environment')
                    }
                else:
                    return {
                        'name': 'configuration_validation',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'name': 'configuration_validation',
                'status': 'failed',
                'error': str(e)
            }

    async def validate_feature_flags(self) -> Dict[str, Any]:
        """Validate feature flags configuration."""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/feature-flags/status") as response:
                if response.status == 200:
                    data = await response.json()

                    flags = data.get('flags', {})
                    environment_flags = data.get('environment_config', {})

                    # Check critical flags are properly configured
                    critical_flags = [
                        'circuit_breaker_protection',
                        'performance_monitoring',
                        'blue_green_deployment'
                    ]

                    missing_critical = []
                    for flag in critical_flags:
                        if flag not in flags or not flags[flag].get('enabled'):
                            missing_critical.append(flag)

                    return {
                        'name': 'feature_flags_validation',
                        'status': 'passed' if not missing_critical else 'failed',
                        'total_flags': len(flags),
                        'enabled_flags': sum(1 for f in flags.values() if f.get('enabled')),
                        'missing_critical': missing_critical,
                        'environment_config': environment_flags
                    }
                else:
                    return {
                        'name': 'feature_flags_validation',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'name': 'feature_flags_validation',
                'status': 'failed',
                'error': str(e)
            }

    async def validate_performance_metrics(self) -> Dict[str, Any]:
        """Validate performance metrics are within acceptable ranges."""
        try:
            async with self.session.get(f"{self.base_url}:9090/metrics") as response:
                if response.status == 200:
                    metrics_text = await response.text()

                    # Parse key metrics
                    metrics = {}
                    for line in metrics_text.split('\n'):
                        if line.startswith('tournament_optimization_'):
                            parts = line.split(' ')
                            if len(parts) >= 2:
                                metric_name = parts[0]
                                metric_value = float(parts[1])
                                metrics[metric_name] = metric_value

                    # Validate critical metrics
                    validations = []

                    # Response time should be under 30 seconds
                    response_time = metrics.get('tournament_optimization_response_time_seconds', 0)
                    if response_time > 30:
                        validations.append(f"Response time too high: {response_time}s")

                    # Memory usage should be under 2GB
                    memory_usage = metrics.get('tournament_optimization_memory_usage_bytes', 0)
                    if memory_usage > 2 * 1024 * 1024 * 1024:  # 2GB
                        validations.append(f"Memory usage too high: {memory_usage / 1024 / 1024 / 1024:.2f}GB")

                    # Error rate should be under 5%
                    error_rate = metrics.get('tournament_optimization_error_rate', 0)
                    if error_rate > 0.05:
                        validations.append(f"Error rate too high: {error_rate * 100:.2f}%")

                    return {
                        'name': 'performance_metrics',
                        'status': 'passed' if not validations else 'failed',
                        'response_time': response_time,
                        'memory_usage_gb': memory_usage / 1024 / 1024 / 1024,
                        'error_rate': error_rate,
                        'validation_errors': validations,
                        'total_metrics': len(metrics)
                    }
                else:
                    return {
                        'name': 'performance_metrics',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'name': 'performance_metrics',
                'status': 'failed',
                'error': str(e)
            }

    async def validate_security_headers(self) -> Dict[str, Any]:
        """Validate security headers are present."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                headers = response.headers

                required_headers = [
                    'X-Content-Type-Options',
                    'X-Frame-Options',
                    'X-XSS-Protection',
                    'Strict-Transport-Security'
                ]

                missing_headers = []
                present_headers = {}

                for header in required_headers:
                    if header in headers:
                        present_headers[header] = headers[header]
                    else:
                        missing_headers.append(header)

                return {
                    'name': 'security_headers',
                    'status': 'passed' if not missing_headers else 'failed',
                    'present_headers': present_headers,
                    'missing_headers': missing_headers
                }
        except Exception as e:
            return {
                'name': 'security_headers',
                'status': 'failed',
                'error': str(e)
            }

    async def validate_database_migrations(self) -> Dict[str, Any]:
        """Validate database migrations are up to date."""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/database/migration-status") as response:
                if response.status == 200:
                    data = await response.json()

                    migrations_applied = data.get('applied_migrations', [])
                    pending_migrations = data.get('pending_migrations', [])
                    migration_errors = data.get('errors', [])

                    return {
                        'name': 'database_migrations',
                        'status': 'passed' if not pending_migrations and not migration_errors else 'failed',
                        'applied_count': len(migrations_applied),
                        'pending_count': len(pending_migrations),
                        'errors': migration_errors,
                        'latest_migration': migrations_applied[-1] if migrations_applied else None
                    }
                else:
                    return {
                        'name': 'database_migrations',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'name': 'database_migrations',
                'status': 'failed',
                'error': str(e)
            }

    async def run_all_validations(self) -> List[Dict[str, Any]]:
        """Run all deployment validations."""
        validations = []

        # Kubernetes validations
        validations.append(await self.validate_kubernetes_deployment())
        validations.append(await self.validate_service_endpoints())

        # Application validations
        validations.append(await self.validate_configuration())
        validations.append(await self.validate_feature_flags())
        validations.append(await self.validate_performance_metrics())
        validations.append(await self.validate_security_headers())
        validations.append(await self.validate_database_migrations())

        return validations

    def print_results(self, results: List[Dict[str, Any]]) -> bool:
        """Print validation results."""
        print("\n" + "="*60)
        print(f"DEPLOYMENT VALIDATION RESULTS - {self.environment.upper()}")
        print("="*60)

        passed_count = 0
        failed_count = 0
        warnings = []

        for result in results:
            name = result['name']
            status = result['status']

            if status == 'passed':
                passed_count += 1
                print(f"✅ {name}: PASSED")

                # Print additional info for passed tests
                if name == 'kubernetes_deployment':
                    print(f"   Replicas: {result.get('ready_replicas')}/{result.get('desired_replicas')}")
                elif name == 'performance_metrics':
                    print(f"   Response time: {result.get('response_time', 0):.2f}s")
                    print(f"   Memory usage: {result.get('memory_usage_gb', 0):.2f}GB")
                    print(f"   Error rate: {result.get('error_rate', 0) * 100:.2f}%")
                elif name == 'feature_flags_validation':
                    print(f"   Enabled flags: {result.get('enabled_flags')}/{result.get('total_flags')}")

            else:
                failed_count += 1
                print(f"❌ {name}: FAILED")
                if 'error' in result:
                    print(f"   Error: {result['error']}")

                # Print specific failure details
                if name == 'configuration_validation':
                    if result.get('missing_configs'):
                        print(f"   Missing configs: {result['missing_configs']}")
                    if result.get('invalid_configs'):
                        print(f"   Invalid configs: {result['invalid_configs']}")
                elif name == 'feature_flags_validation':
                    if result.get('missing_critical'):
                        print(f"   Missing critical flags: {result['missing_critical']}")
                elif name == 'performance_metrics':
                    if result.get('validation_errors'):
                        for error in result['validation_errors']:
                            print(f"   {error}")
                elif name == 'security_headers':
                    if result.get('missing_headers'):
                        print(f"   Missing headers: {result['missing_headers']}")

        print(f"\nSUMMARY:")
        print(f"  Passed: {passed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(results)}")

        success = failed_count == 0

        if success:
            print(f"\n🎉 DEPLOYMENT VALIDATION PASSED")
            print("All validation checks passed successfully.")
        else:
            print(f"\n🚨 DEPLOYMENT VALIDATION FAILED")
            print(f"{failed_count} validation checks failed.")

            if self.strict:
                print("Running in strict mode - deployment should be rolled back.")
            else:
                print("Running in non-strict mode - warnings logged but deployment continues.")

        print("="*60)

        return success if self.strict else True


async def main():
    parser = argparse.ArgumentParser(description='Validate deployment health and configuration')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Target environment')
    parser.add_argument('--strict', action='store_true',
                       help='Fail on any validation error (recommended for production)')

    args = parser.parse_args()

    async with DeploymentValidator(args.environment, args.strict) as validator:
        print(f"🔍 Running deployment validation for {args.environment} environment")
        if args.strict:
            print("⚠️  Running in strict mode - any failure will fail the validation")

        results = await validator.run_all_validations()
        success = validator.print_results(results)

        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
