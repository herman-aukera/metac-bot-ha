#!/usr/bin/env python3
"""
Pre-deployment checks for tournament optimization system.
Validates environment readiness before deployment.
"""

import asyncio
import aiohttp
import argparse
import sys
import json
import subprocess
import boto3
from typing import Dict, List, Any, Optional
from datetime import datetime


class PreDeploymentChecker:
    """Performs comprehensive pre-deployment checks."""

    def __init__(self, environment: str):
        self.environment = environment
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []

    async def check_infrastructure_readiness(self) -> Dict[str, Any]:
        """Check infrastructure components are ready."""
        print("🔍 Checking infrastructure readiness...")

        checks = []

        # Check EKS cluster
        try:
            result = subprocess.run([
                'kubectl', 'cluster-info'
            ], capture_output=True, text=True, check=True)

            checks.append({
                'name': 'eks_cluster_connectivity',
                'status': 'passed',
                'message': 'EKS cluster is accessible'
            })
        except subprocess.CalledProcessError as e:
            checks.append({
                'name': 'eks_cluster_connectivity',
                'status': 'failed',
                'message': f'Cannot connect to EKS cluster: {e.stderr}'
            })

        # Check namespace exists
        try:
            namespace = f'tournament-optimization-{self.environment}'
            result = subprocess.run([
                'kubectl', 'get', 'namespace', namespace
            ], capture_output=True, text=True, check=True)

            checks.append({
                'name': 'namespace_exists',
                'status': 'passed',
                'message': f'Namespace {namespace} exists'
            })
        except subprocess.CalledProcessError:
            checks.append({
                'name': 'namespace_exists',
                'status': 'failed',
                'message': f'Namespace {namespace} does not exist'
            })

        # Check node capacity
        try:
            result = subprocess.run([
                'kubectl', 'top', 'nodes'
            ], capture_output=True, text=True, check=True)

            # Parse node resource usage
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            high_usage_nodes = []

            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    cpu_usage = parts[1].replace('%', '')
                    memory_usage = parts[2].replace('%', '')

                    if int(cpu_usage) > 80 or int(memory_usage) > 80:
                        high_usage_nodes.append(parts[0])

            if high_usage_nodes:
                checks.append({
                    'name': 'node_capacity',
                    'status': 'warning',
                    'message': f'High resource usage on nodes: {", ".join(high_usage_nodes)}'
                })
                self.warnings.append(f'High resource usage detected on {len(high_usage_nodes)} nodes')
            else:
                checks.append({
                    'name': 'node_capacity',
                    'status': 'passed',
                    'message': 'Node resource usage is within acceptable limits'
                })

        except subprocess.CalledProcessError as e:
            checks.append({
                'name': 'node_capacity',
                'status': 'warning',
                'message': f'Could not check node capacity: {e.stderr}'
            })

        return {
            'category': 'infrastructure',
            'checks': checks
        }

    async def check_database_readiness(self) -> Dict[str, Any]:
        """Check database connectivity and health."""
        print("🔍 Checking database readiness...")

        checks = []

        # Check RDS instance status
        try:
            rds_client = boto3.client('rds')
            db_instance_id = f'tournament-optimization-{self.environment}-db'

            response = rds_client.describe_db_instances(
                DBInstanceIdentifier=db_instance_id
            )

            db_instance = response['DBInstances'][0]
            status = db_instance['DBInstanceStatus']

            if status == 'available':
                checks.append({
                    'name': 'rds_instance_status',
                    'status': 'passed',
                    'message': f'RDS instance {db_instance_id} is available'
                })
            else:
                checks.append({
                    'name': 'rds_instance_status',
                    'status': 'failed',
                    'message': f'RDS instance {db_instance_id} status: {status}'
                })

        except Exception as e:
            checks.append({
                'name': 'rds_instance_status',
                'status': 'failed',
                'message': f'Failed to check RDS status: {str(e)}'
            })

        # Check database connectivity from cluster
        try:
            namespace = f'tournament-optimization-{self.environment}'
            result = subprocess.run([
                'kubectl', 'run', 'db-test', '--rm', '-i', '--restart=Never',
                '--image=postgres:15', '-n', namespace,
                '--', 'pg_isready', '-h', 'tournament-optimization-db'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                checks.append({
                    'name': 'database_connectivity',
                    'status': 'passed',
                    'message': 'Database is reachable from cluster'
                })
            else:
                checks.append({
                    'name': 'database_connectivity',
                    'status': 'failed',
                    'message': 'Database is not reachable from cluster'
                })

        except subprocess.TimeoutExpired:
            checks.append({
                'name': 'database_connectivity',
                'status': 'failed',
                'message': 'Database connectivity check timed out'
            })
        except Exception as e:
            checks.append({
                'name': 'database_connectivity',
                'status': 'failed',
                'message': f'Database connectivity check failed: {str(e)}'
            })

        return {
            'category': 'database',
            'checks': checks
        }

    async def check_external_dependencies(self) -> Dict[str, Any]:
        """Check external service dependencies."""
        print("🔍 Checking external dependencies...")

        checks = []

        # External APIs to check
        external_apis = [
            {'name': 'metaculus', 'url': 'https://www.metaculus.com/api2/questions/'},
            {'name': 'openai', 'url': 'https://api.openai.com/v1/models'},
        ]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for api in external_apis:
                try:
                    async with session.get(api['url']) as response:
                        if response.status < 500:  # Accept 4xx as API is reachable
                            checks.append({
                                'name': f'{api["name"]}_api_reachable',
                                'status': 'passed',
                                'message': f'{api["name"]} API is reachable (status: {response.status})'
                            })
                        else:
                            checks.append({
                                'name': f'{api["name"]}_api_reachable',
                                'status': 'warning',
                                'message': f'{api["name"]} API returned {response.status}'
                            })
                            self.warnings.append(f'{api["name"]} API may be experiencing issues')

                except asyncio.TimeoutError:
                    checks.append({
                        'name': f'{api["name"]}_api_reachable',
                        'status': 'warning',
                        'message': f'{api["name"]} API request timed out'
                    })
                    self.warnings.append(f'{api["name"]} API timeout - may affect functionality')

                except Exception as e:
                    checks.append({
                        'name': f'{api["name"]}_api_reachable',
                        'status': 'warning',
                        'message': f'{api["name"]} API check failed: {str(e)}'
                    })

        return {
            'category': 'external_dependencies',
            'checks': checks
        }

    async def check_secrets_and_config(self) -> Dict[str, Any]:
        """Check secrets and configuration are present."""
        print("🔍 Checking secrets and configuration...")

        checks = []
        namespace = f'tournament-optimization-{self.environment}'

        # Check required secrets
        required_secrets = [
            'tournament-optimization-secrets'
        ]

        for secret_name in required_secrets:
            try:
                result = subprocess.run([
                    'kubectl', 'get', 'secret', secret_name, '-n', namespace
                ], capture_output=True, text=True, check=True)

                checks.append({
                    'name': f'secret_{secret_name}',
                    'status': 'passed',
                    'message': f'Secret {secret_name} exists'
                })

            except subprocess.CalledProcessError:
                checks.append({
                    'name': f'secret_{secret_name}',
                    'status': 'failed',
                    'message': f'Secret {secret_name} is missing'
                })

        # Check required configmaps
        required_configmaps = [
            'tournament-optimization-config',
            'tournament-optimization-feature-flags'
        ]

        for configmap_name in required_configmaps:
            try:
                result = subprocess.run([
                    'kubectl', 'get', 'configmap', configmap_name, '-n', namespace
                ], capture_output=True, text=True, check=True)

                checks.append({
                    'name': f'configmap_{configmap_name}',
                    'status': 'passed',
                    'message': f'ConfigMap {configmap_name} exists'
                })

            except subprocess.CalledProcessError:
                checks.append({
                    'name': f'configmap_{configmap_name}',
                    'status': 'failed',
                    'message': f'ConfigMap {configmap_name} is missing'
                })

        return {
            'category': 'secrets_and_config',
            'checks': checks
        }

    async def check_container_registry(self) -> Dict[str, Any]:
        """Check container registry accessibility."""
        print("🔍 Checking container registry...")

        checks = []

        # Check if we can pull from the registry
        try:
            registry_url = 'ghcr.io/tournament-optimization/tournament-optimization'

            result = subprocess.run([
                'docker', 'manifest', 'inspect', f'{registry_url}:latest'
            ], capture_output=True, text=True, check=True)

            checks.append({
                'name': 'container_registry_access',
                'status': 'passed',
                'message': 'Container registry is accessible'
            })

        except subprocess.CalledProcessError as e:
            checks.append({
                'name': 'container_registry_access',
                'status': 'failed',
                'message': f'Cannot access container registry: {e.stderr}'
            })

        return {
            'category': 'container_registry',
            'checks': checks
        }

    async def check_monitoring_and_logging(self) -> Dict[str, Any]:
        """Check monitoring and logging infrastructure."""
        print("🔍 Checking monitoring and logging...")

        checks = []

        # Check if Prometheus is accessible (if deployed)
        try:
            result = subprocess.run([
                'kubectl', 'get', 'service', 'prometheus-server', '-n', 'monitoring'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                checks.append({
                    'name': 'prometheus_service',
                    'status': 'passed',
                    'message': 'Prometheus service is available'
                })
            else:
                checks.append({
                    'name': 'prometheus_service',
                    'status': 'warning',
                    'message': 'Prometheus service not found - metrics may not be collected'
                })
                self.warnings.append('Prometheus not available - consider deploying monitoring stack')

        except Exception as e:
            checks.append({
                'name': 'prometheus_service',
                'status': 'warning',
                'message': f'Could not check Prometheus: {str(e)}'
            })

        # Check CloudWatch log groups
        try:
            logs_client = boto3.client('logs')
            log_group_name = f'/aws/eks/tournament-optimization-{self.environment}/application'

            logs_client.describe_log_groups(logGroupNamePrefix=log_group_name)

            checks.append({
                'name': 'cloudwatch_logs',
                'status': 'passed',
                'message': 'CloudWatch log groups are configured'
            })

        except Exception as e:
            checks.append({
                'name': 'cloudwatch_logs',
                'status': 'warning',
                'message': f'CloudWatch logs check failed: {str(e)}'
            })

        return {
            'category': 'monitoring_and_logging',
            'checks': checks
        }

    async def check_backup_readiness(self) -> Dict[str, Any]:
        """Check backup infrastructure is ready."""
        print("🔍 Checking backup readiness...")

        checks = []

        # Check S3 backup bucket
        try:
            s3_client = boto3.client('s3')
            bucket_name = f'tournament-optimization-{self.environment}-backups'

            s3_client.head_bucket(Bucket=bucket_name)

            checks.append({
                'name': 'backup_s3_bucket',
                'status': 'passed',
                'message': f'Backup S3 bucket {bucket_name} is accessible'
            })

        except Exception as e:
            checks.append({
                'name': 'backup_s3_bucket',
                'status': 'failed',
                'message': f'Backup S3 bucket check failed: {str(e)}'
            })

        # Check recent backups exist
        try:
            s3_client = boto3.client('s3')
            bucket_name = f'tournament-optimization-{self.environment}-backups'

            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix='manifests/',
                MaxKeys=5
            )

            if response.get('Contents'):
                latest_backup = max(response['Contents'], key=lambda x: x['LastModified'])
                age_hours = (datetime.now().replace(tzinfo=None) - latest_backup['LastModified'].replace(tzinfo=None)).total_seconds() / 3600

                if age_hours < 24:
                    checks.append({
                        'name': 'recent_backup_exists',
                        'status': 'passed',
                        'message': f'Recent backup found (age: {age_hours:.1f} hours)'
                    })
                else:
                    checks.append({
                        'name': 'recent_backup_exists',
                        'status': 'warning',
                        'message': f'Latest backup is {age_hours:.1f} hours old'
                    })
                    self.warnings.append('Backup may be stale - consider running fresh backup')
            else:
                checks.append({
                    'name': 'recent_backup_exists',
                    'status': 'warning',
                    'message': 'No backup manifests found'
                })
                self.warnings.append('No backups found - ensure backup system is working')

        except Exception as e:
            checks.append({
                'name': 'recent_backup_exists',
                'status': 'warning',
                'message': f'Could not check recent backups: {str(e)}'
            })

        return {
            'category': 'backup_readiness',
            'checks': checks
        }

    def print_results(self, results: List[Dict[str, Any]]) -> bool:
        """Print check results and return overall success."""
        print("\n" + "="*60)
        print(f"PRE-DEPLOYMENT CHECKS - {self.environment.upper()}")
        print("="*60)

        all_passed = True

        for category_result in results:
            category = category_result['category']
            checks = category_result['checks']

            print(f"\n📋 {category.replace('_', ' ').title()}")
            print("-" * 40)

            for check in checks:
                name = check['name']
                status = check['status']
                message = check['message']

                if status == 'passed':
                    print(f"  ✅ {name}: {message}")
                    self.checks_passed += 1
                elif status == 'warning':
                    print(f"  ⚠️  {name}: {message}")
                    self.checks_passed += 1  # Warnings don't fail deployment
                else:
                    print(f"  ❌ {name}: {message}")
                    self.checks_failed += 1
                    all_passed = False

        # Print summary
        print(f"\n📊 SUMMARY:")
        print(f"  Passed: {self.checks_passed}")
        print(f"  Failed: {self.checks_failed}")
        print(f"  Warnings: {len(self.warnings)}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if all_passed:
            print(f"\n🎉 PRE-DEPLOYMENT CHECKS PASSED")
            print("Environment is ready for deployment")
        else:
            print(f"\n🚨 PRE-DEPLOYMENT CHECKS FAILED")
            print(f"{self.checks_failed} critical issues must be resolved before deployment")

        print("="*60)
        return all_passed

    async def run_all_checks(self) -> bool:
        """Run all pre-deployment checks."""
        print(f"🚀 Running pre-deployment checks for {self.environment} environment")

        check_functions = [
            self.check_infrastructure_readiness,
            self.check_database_readiness,
            self.check_external_dependencies,
            self.check_secrets_and_config,
            self.check_container_registry,
            self.check_monitoring_and_logging,
            self.check_backup_readiness
        ]

        results = []
        for check_func in check_functions:
            try:
                result = await check_func()
                results.append(result)
            except Exception as e:
                print(f"❌ Check {check_func.__name__} failed with error: {e}")
                results.append({
                    'category': check_func.__name__.replace('check_', ''),
                    'checks': [{
                        'name': 'check_execution',
                        'status': 'failed',
                        'message': f'Check failed to execute: {str(e)}'
                    }]
                })

        return self.print_results(results)


async def main():
    parser = argparse.ArgumentParser(description='Run pre-deployment checks')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Target environment')

    args = parser.parse_args()

    checker = PreDeploymentChecker(args.environment)
    success = await checker.run_all_checks()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
