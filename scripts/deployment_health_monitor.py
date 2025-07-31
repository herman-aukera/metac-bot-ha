#!/usr/bin/env python3
"""
Deployment health monitoring script with automatic rollback capabilities.
Monitors deployment health and triggers rollback if issues are detected.
"""

import asyncio
import aiohttp
import argparse
import sys
import time
import json
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class HealthMetrics:
    """Health metrics for deployment monitoring."""
    timestamp: datetime
    response_time: float
    error_rate: float
    memory_usage: float
    cpu_usage: float
    active_connections: int
    queue_length: int


class DeploymentHealthMonitor:
    """Monitors deployment health and triggers rollback if needed."""

    def __init__(self, environment: str, duration: int = 300, auto_rollback: bool = False):
        self.environment = environment
        self.duration = duration
        self.auto_rollback = auto_rollback

        self.base_urls = {
            'staging': 'https://tournament-optimization-staging.example.com',
            'production': 'https://tournament-optimization.example.com'
        }
        self.base_url = self.base_urls[environment]

        # Health thresholds
        self.thresholds = {
            'response_time_max': 30.0,  # seconds
            'error_rate_max': 0.05,     # 5%
            'memory_usage_max': 0.85,   # 85% of limit
            'cpu_usage_max': 0.80,      # 80% of limit
            'consecutive_failures_max': 5
        }

        self.session = None
        self.metrics_history: List[HealthMetrics] = []
        self.consecutive_failures = 0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def collect_health_metrics(self) -> Optional[HealthMetrics]:
        """Collect current health metrics."""
        try:
            # Get basic health status
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status != 200:
                    return None

                health_data = await response.json()

            # Get detailed metrics
            async with self.session.get(f"{self.base_url}:9090/metrics") as response:
                if response.status != 200:
                    return None

                metrics_text = await response.text()

            # Parse metrics
            metrics = self._parse_prometheus_metrics(metrics_text)

            return HealthMetrics(
                timestamp=datetime.now(),
                response_time=metrics.get('tournament_optimization_response_time_seconds', 0),
                error_rate=metrics.get('tournament_optimization_error_rate', 0),
                memory_usage=metrics.get('tournament_optimization_memory_usage_ratio', 0),
                cpu_usage=metrics.get('tournament_optimization_cpu_usage_ratio', 0),
                active_connections=int(metrics.get('tournament_optimization_active_connections', 0)),
                queue_length=int(metrics.get('tournament_optimization_queue_length', 0))
            )

        except Exception as e:
            print(f"❌ Failed to collect metrics: {e}")
            return None

    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus metrics text."""
        metrics = {}

        for line in metrics_text.split('\n'):
            if line.startswith('tournament_optimization_') and ' ' in line:
                parts = line.split(' ')
                if len(parts) >= 2:
                    metric_name = parts[0]
                    try:
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
                    except ValueError:
                        continue

        return metrics

    def check_health_thresholds(self, metrics: HealthMetrics) -> List[str]:
        """Check if metrics exceed health thresholds."""
        violations = []

        if metrics.response_time > self.thresholds['response_time_max']:
            violations.append(f"Response time too high: {metrics.response_time:.2f}s")

        if metrics.error_rate > self.thresholds['error_rate_max']:
            violations.append(f"Error rate too high: {metrics.error_rate * 100:.2f}%")

        if metrics.memory_usage > self.thresholds['memory_usage_max']:
            violations.append(f"Memory usage too high: {metrics.memory_usage * 100:.1f}%")

        if metrics.cpu_usage > self.thresholds['cpu_usage_max']:
            violations.append(f"CPU usage too high: {metrics.cpu_usage * 100:.1f}%")

        return violations

    async def trigger_rollback(self, reason: str) -> bool:
        """Trigger deployment rollback."""
        print(f"🚨 TRIGGERING ROLLBACK: {reason}")

        try:
            # Execute rollback script
            result = subprocess.run([
                'bash', 'scripts/deploy.sh',
                '--environment', self.environment,
                '--rollback'
            ], capture_output=True, text=True, check=True)

            print(f"✅ Rollback completed successfully")
            print(f"Rollback output: {result.stdout}")

            # Wait for rollback to take effect
            await asyncio.sleep(60)

            # Verify rollback success
            rollback_metrics = await self.collect_health_metrics()
            if rollback_metrics:
                violations = self.check_health_thresholds(rollback_metrics)
                if not violations:
                    print(f"✅ Rollback verification successful")
                    return True
                else:
                    print(f"❌ Rollback verification failed: {violations}")
                    return False
            else:
                print(f"❌ Could not verify rollback")
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Rollback failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Rollback error: {e}")
            return False

    async def send_alert(self, message: str, severity: str = 'warning'):
        """Send alert notification."""
        alert_data = {
            'environment': self.environment,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'deployment_url': self.base_url
        }

        print(f"🚨 ALERT [{severity.upper()}]: {message}")

        # In a real implementation, this would send to Slack, PagerDuty, etc.
        # For now, we'll just log the alert
        try:
            with open(f'/tmp/deployment_alerts_{self.environment}.json', 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
        except Exception as e:
            print(f"Failed to log alert: {e}")

    def print_metrics_summary(self, metrics: HealthMetrics):
        """Print current metrics summary."""
        print(f"\n📊 Health Metrics ({metrics.timestamp.strftime('%H:%M:%S')})")
        print(f"   Response Time: {metrics.response_time:.2f}s")
        print(f"   Error Rate: {metrics.error_rate * 100:.2f}%")
        print(f"   Memory Usage: {metrics.memory_usage * 100:.1f}%")
        print(f"   CPU Usage: {metrics.cpu_usage * 100:.1f}%")
        print(f"   Active Connections: {metrics.active_connections}")
        print(f"   Queue Length: {metrics.queue_length}")

    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze health trends over time."""
        if len(self.metrics_history) < 2:
            return {}

        recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
        older_metrics = self.metrics_history[-10:-5] if len(self.metrics_history) >= 10 else []

        if not older_metrics:
            return {}

        # Calculate trends
        recent_avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        older_avg_response_time = sum(m.response_time for m in older_metrics) / len(older_metrics)

        recent_avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        older_avg_error_rate = sum(m.error_rate for m in older_metrics) / len(older_metrics)

        return {
            'response_time_trend': recent_avg_response_time - older_avg_response_time,
            'error_rate_trend': recent_avg_error_rate - older_avg_error_rate,
            'degrading': (recent_avg_response_time > older_avg_response_time * 1.2 or
                         recent_avg_error_rate > older_avg_error_rate * 1.5)
        }

    async def monitor_deployment(self) -> bool:
        """Monitor deployment health for specified duration."""
        print(f"🔍 Starting deployment health monitoring")
        print(f"Environment: {self.environment}")
        print(f"Duration: {self.duration}s")
        print(f"Auto-rollback: {'enabled' if self.auto_rollback else 'disabled'}")
        print(f"Base URL: {self.base_url}")

        start_time = time.time()
        check_interval = 30  # Check every 30 seconds

        while time.time() - start_time < self.duration:
            # Collect metrics
            metrics = await self.collect_health_metrics()

            if metrics is None:
                self.consecutive_failures += 1
                print(f"❌ Failed to collect metrics (consecutive failures: {self.consecutive_failures})")

                if self.consecutive_failures >= self.thresholds['consecutive_failures_max']:
                    await self.send_alert(
                        f"Failed to collect metrics {self.consecutive_failures} times consecutively",
                        'critical'
                    )

                    if self.auto_rollback:
                        success = await self.trigger_rollback("Consecutive metric collection failures")
                        return success
                    else:
                        print("❌ Auto-rollback disabled, manual intervention required")
                        return False
            else:
                self.consecutive_failures = 0
                self.metrics_history.append(metrics)
                self.print_metrics_summary(metrics)

                # Check thresholds
                violations = self.check_health_thresholds(metrics)

                if violations:
                    violation_msg = "; ".join(violations)
                    await self.send_alert(f"Health threshold violations: {violation_msg}", 'critical')

                    if self.auto_rollback:
                        success = await self.trigger_rollback(f"Health threshold violations: {violation_msg}")
                        return success
                    else:
                        print("❌ Auto-rollback disabled, manual intervention required")
                        return False

                # Analyze trends
                trends = self.analyze_trends()
                if trends.get('degrading'):
                    await self.send_alert("Performance degradation detected", 'warning')

            # Wait for next check
            await asyncio.sleep(check_interval)

        print(f"✅ Monitoring completed successfully")
        print(f"Total metrics collected: {len(self.metrics_history)}")

        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            final_violations = self.check_health_thresholds(final_metrics)

            if not final_violations:
                print(f"🎉 Deployment is healthy")
                return True
            else:
                print(f"⚠️  Deployment has issues: {'; '.join(final_violations)}")
                return False
        else:
            print(f"❌ No metrics collected during monitoring period")
            return False


async def main():
    parser = argparse.ArgumentParser(description='Monitor deployment health with auto-rollback')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Target environment')
    parser.add_argument('--duration', type=int, default=300,
                       help='Monitoring duration in seconds')
    parser.add_argument('--auto-rollback', action='store_true',
                       help='Enable automatic rollback on health issues')

    args = parser.parse_args()

    async with DeploymentHealthMonitor(args.environment, args.duration, args.auto_rollback) as monitor:
        success = await monitor.monitor_deployment()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
