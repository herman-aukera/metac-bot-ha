#!/usr/bin/env python3
"""
Smoke tests for deployment validation.
Performs basic functionality checks after deployment.
"""

import asyncio
import aiohttp
import argparse
import sys
import time
from typing import Dict, List, Any
import json


class SmokeTestRunner:
    """Runs smoke tests against deployed application."""

    def __init__(self, base_url: str, timeout: int = 300, comprehensive: bool = False):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.comprehensive = comprehensive
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test basic health endpoint."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'name': 'health_endpoint',
                        'status': 'passed',
                        'response_time': response.headers.get('X-Response-Time', 'unknown'),
                        'data': data
                    }
                else:
                    return {
                        'name': 'health_endpoint',
                        'status': 'failed',
                        'error': f'HTTP {response.status}',
                        'response': await response.text()
                    }
        except Exception as e:
            return {
                'name': 'health_endpoint',
                'status': 'failed',
                'error': str(e)
            }

    async def test_readiness_endpoint(self) -> Dict[str, Any]:
        """Test readiness endpoint."""
        try:
            async with self.session.get(f"{self.base_url}/ready") as response:
                if response.status == 200:
                    return {
                        'name': 'readiness_endpoint',
                        'status': 'passed',
                        'response_time': response.headers.get('X-Response-Time', 'unknown')
                    }
                else:
                    return {
                        'name': 'readiness_endpoint',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'name': 'readiness_endpoint',
                'status': 'failed',
                'error': str(e)
            }

    async def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test metrics endpoint."""
        try:
            async with self.session.get(f"{self.base_url}:9090/metrics") as response:
                if response.status == 200:
                    metrics_text = await response.text()
                    # Basic validation that it's Prometheus format
                    if 'tournament_optimization' in metrics_text:
                        return {
                            'name': 'metrics_endpoint',
                            'status': 'passed',
                            'metrics_count': len([line for line in metrics_text.split('\n') if line and not line.startswith('#')])
                        }
                    else:
                        return {
                            'name': 'metrics_endpoint',
                            'status': 'failed',
                            'error': 'Invalid metrics format'
                        }
                else:
                    return {
                        'name': 'metrics_endpoint',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'name': 'metrics_endpoint',
                'status': 'failed',
                'error': str(e)
            }

    async def test_api_endpoints(self) -> List[Dict[str, Any]]:
        """Test core API endpoints."""
        results = []

        # Test forecast endpoint
        try:
            test_question = {
                "id": "test-question-1",
                "title": "Test Question",
                "description": "This is a test question for smoke testing",
                "question_type": "binary"
            }

            async with self.session.post(
                f"{self.base_url}/api/v1/forecast",
                json=test_question
            ) as response:
                if response.status in [200, 202]:  # Accept both sync and async responses
                    results.append({
                        'name': 'forecast_api',
                        'status': 'passed',
                        'response_status': response.status
                    })
                else:
                    results.append({
                        'name': 'forecast_api',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    })
        except Exception as e:
            results.append({
                'name': 'forecast_api',
                'status': 'failed',
                'error': str(e)
            })

        return results

    async def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity through API."""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/status/database") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'healthy':
                        return {
                            'name': 'database_connectivity',
                            'status': 'passed',
                            'connection_pool': data.get('connection_pool', {})
                        }
                    else:
                        return {
                            'name': 'database_connectivity',
                            'status': 'failed',
                            'error': 'Database unhealthy'
                        }
                else:
                    return {
                        'name': 'database_connectivity',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'name': 'database_connectivity',
                'status': 'failed',
                'error': str(e)
            }

    async def test_cache_connectivity(self) -> Dict[str, Any]:
        """Test cache connectivity through API."""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/status/cache") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'healthy':
                        return {
                            'name': 'cache_connectivity',
                            'status': 'passed',
                            'cache_info': data.get('cache_info', {})
                        }
                    else:
                        return {
                            'name': 'cache_connectivity',
                            'status': 'failed',
                            'error': 'Cache unhealthy'
                        }
                else:
                    return {
                        'name': 'cache_connectivity',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'name': 'cache_connectivity',
                'status': 'failed',
                'error': str(e)
            }

    async def test_external_apis(self) -> List[Dict[str, Any]]:
        """Test external API connectivity."""
        results = []

        try:
            async with self.session.get(f"{self.base_url}/api/v1/status/external-apis") as response:
                if response.status == 200:
                    data = await response.json()
                    for api_name, api_status in data.get('apis', {}).items():
                        results.append({
                            'name': f'external_api_{api_name}',
                            'status': 'passed' if api_status.get('healthy') else 'failed',
                            'response_time': api_status.get('response_time'),
                            'error': api_status.get('error')
                        })
                else:
                    results.append({
                        'name': 'external_apis_check',
                        'status': 'failed',
                        'error': f'HTTP {response.status}'
                    })
        except Exception as e:
            results.append({
                'name': 'external_apis_check',
                'status': 'failed',
                'error': str(e)
            })

        return results

    async def run_comprehensive_tests(self) -> List[Dict[str, Any]]:
        """Run comprehensive smoke tests."""
        all_results = []

        # Basic health checks
        all_results.append(await self.test_health_endpoint())
        all_results.append(await self.test_readiness_endpoint())
        all_results.append(await self.test_metrics_endpoint())

        # Infrastructure tests
        all_results.append(await self.test_database_connectivity())
        all_results.append(await self.test_cache_connectivity())

        # API tests
        api_results = await self.test_api_endpoints()
        all_results.extend(api_results)

        # External API tests (if comprehensive)
        if self.comprehensive:
            external_api_results = await self.test_external_apis()
            all_results.extend(external_api_results)

        return all_results

    async def wait_for_deployment(self) -> bool:
        """Wait for deployment to be ready."""
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            try:
                async with self.session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'healthy':
                            print(f"✅ Deployment is ready after {time.time() - start_time:.1f}s")
                            return True
            except Exception:
                pass

            print(f"⏳ Waiting for deployment... ({time.time() - start_time:.1f}s)")
            await asyncio.sleep(10)

        print(f"❌ Deployment not ready after {self.timeout}s")
        return False

    def print_results(self, results: List[Dict[str, Any]]) -> bool:
        """Print test results and return overall success."""
        print("\n" + "="*60)
        print("SMOKE TEST RESULTS")
        print("="*60)

        passed_count = 0
        failed_count = 0

        for result in results:
            name = result['name']
            status = result['status']

            if status == 'passed':
                passed_count += 1
                print(f"✅ {name}: PASSED")
                if 'response_time' in result:
                    print(f"   Response time: {result['response_time']}")
            else:
                failed_count += 1
                print(f"❌ {name}: FAILED")
                if 'error' in result:
                    print(f"   Error: {result['error']}")

        print(f"\nSUMMARY:")
        print(f"  Passed: {passed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(results)}")

        success = failed_count == 0
        if success:
            print(f"\n🎉 ALL SMOKE TESTS PASSED")
        else:
            print(f"\n🚨 {failed_count} SMOKE TESTS FAILED")

        print("="*60)
        return success


async def main():
    parser = argparse.ArgumentParser(description='Run smoke tests against deployed application')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Target environment')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout for waiting for deployment (seconds)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive tests including external APIs')

    args = parser.parse_args()

    # Environment-specific URLs
    base_urls = {
        'staging': 'https://tournament-optimization-staging.example.com',
        'production': 'https://tournament-optimization.example.com'
    }

    base_url = base_urls[args.environment]

    async with SmokeTestRunner(base_url, args.timeout, args.comprehensive) as runner:
        print(f"🚀 Running smoke tests against {args.environment} environment")
        print(f"Base URL: {base_url}")

        # Wait for deployment to be ready
        if not await runner.wait_for_deployment():
            print("❌ Deployment not ready, aborting smoke tests")
            return 1

        # Run tests
        results = await runner.run_comprehensive_tests()

        # Print results and determine exit code
        success = runner.print_results(results)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
