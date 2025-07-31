#!/usr/bin/env python3
"""
Performance regression check for tournament optimization system.
Compares current performance metrics against baseline to detect regressions.
"""

import json
import sys
import argparse
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class PerformanceRegressionChecker:
    """Checks for performance regressions in benchmark results."""

    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_file = baseline_file
        self.regression_thresholds = {
            'response_time': 1.2,  # 20% increase is a regression
            'memory_usage': 1.15,  # 15% increase is a regression
            'cpu_usage': 1.25,     # 25% increase is a regression
            'throughput': 0.85,    # 15% decrease is a regression
            'error_rate': 2.0      # 100% increase is a regression
        }

    def load_benchmark_results(self, file_path: str) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Benchmark file not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in benchmark file: {e}")
            return {}

    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline performance metrics."""
        if not self.baseline_file:
            return None

        return self.load_benchmark_results(self.baseline_file)

    def extract_metrics(self, benchmark_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key performance metrics from benchmark data."""
        metrics = {}

        # Extract pytest-benchmark metrics
        if 'benchmarks' in benchmark_data:
            for benchmark in benchmark_data['benchmarks']:
                name = benchmark.get('name', 'unknown')
                stats = benchmark.get('stats', {})

                # Response time metrics
                if 'mean' in stats:
                    metrics[f'{name}_mean_time'] = stats['mean']
                if 'median' in stats:
                    metrics[f'{name}_median_time'] = stats['median']
                if 'max' in stats:
                    metrics[f'{name}_max_time'] = stats['max']

                # Memory metrics (if available)
                if 'memory' in benchmark:
                    metrics[f'{name}_memory_usage'] = benchmark['memory']

        # Extract custom metrics
        if 'custom_metrics' in benchmark_data:
            custom = benchmark_data['custom_metrics']

            if 'response_times' in custom:
                response_times = custom['response_times']
                metrics['avg_response_time'] = statistics.mean(response_times)
                metrics['p95_response_time'] = self._percentile(response_times, 95)
                metrics['p99_response_time'] = self._percentile(response_times, 99)

            if 'memory_usage' in custom:
                metrics['memory_usage'] = custom['memory_usage']

            if 'cpu_usage' in custom:
                metrics['cpu_usage'] = custom['cpu_usage']

            if 'throughput' in custom:
                metrics['throughput'] = custom['throughput']

            if 'error_rate' in custom:
                metrics['error_rate'] = custom['error_rate']

        return metrics

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a list of numbers."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def compare_metrics(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compare current metrics against baseline."""

        comparisons = {}
        regressions = []
        improvements = []

        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline_metrics:
                continue

            baseline_value = baseline_metrics[metric_name]

            if baseline_value == 0:
                continue  # Avoid division by zero

            ratio = current_value / baseline_value
            percentage_change = (ratio - 1) * 100

            # Determine if this is a regression
            is_regression = False
            threshold_key = None

            # Map metric names to threshold categories
            if 'time' in metric_name.lower() or 'response' in metric_name.lower():
                threshold_key = 'response_time'
                is_regression = ratio > self.regression_thresholds[threshold_key]
            elif 'memory' in metric_name.lower():
                threshold_key = 'memory_usage'
                is_regression = ratio > self.regression_thresholds[threshold_key]
            elif 'cpu' in metric_name.lower():
                threshold_key = 'cpu_usage'
                is_regression = ratio > self.regression_thresholds[threshold_key]
            elif 'throughput' in metric_name.lower():
                threshold_key = 'throughput'
                is_regression = ratio < self.regression_thresholds[threshold_key]
            elif 'error' in metric_name.lower():
                threshold_key = 'error_rate'
                is_regression = ratio > self.regression_thresholds[threshold_key]

            comparison = {
                'metric': metric_name,
                'current_value': current_value,
                'baseline_value': baseline_value,
                'ratio': ratio,
                'percentage_change': percentage_change,
                'is_regression': is_regression,
                'threshold_key': threshold_key
            }

            comparisons[metric_name] = comparison

            if is_regression:
                regressions.append(comparison)
            elif percentage_change < -5:  # 5% improvement
                improvements.append(comparison)

        return {
            'comparisons': comparisons,
            'regressions': regressions,
            'improvements': improvements,
            'total_metrics': len(comparisons),
            'regression_count': len(regressions)
        }

    def generate_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate a human-readable performance report."""

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PERFORMANCE REGRESSION ANALYSIS")
        report_lines.append("=" * 60)

        total_metrics = comparison_results['total_metrics']
        regression_count = comparison_results['regression_count']
        improvement_count = len(comparison_results['improvements'])

        report_lines.append(f"Total metrics compared: {total_metrics}")
        report_lines.append(f"Performance regressions: {regression_count}")
        report_lines.append(f"Performance improvements: {improvement_count}")
        report_lines.append("")

        # Report regressions
        if comparison_results['regressions']:
            report_lines.append("🚨 PERFORMANCE REGRESSIONS DETECTED:")
            report_lines.append("-" * 40)

            for regression in comparison_results['regressions']:
                metric = regression['metric']
                change = regression['percentage_change']
                current = regression['current_value']
                baseline = regression['baseline_value']
                threshold_key = regression['threshold_key']

                report_lines.append(f"❌ {metric}")
                report_lines.append(f"   Current: {current:.4f}")
                report_lines.append(f"   Baseline: {baseline:.4f}")
                report_lines.append(f"   Change: {change:+.2f}%")
                report_lines.append(f"   Threshold: {threshold_key}")
                report_lines.append("")

        # Report improvements
        if comparison_results['improvements']:
            report_lines.append("✅ PERFORMANCE IMPROVEMENTS:")
            report_lines.append("-" * 40)

            for improvement in comparison_results['improvements']:
                metric = improvement['metric']
                change = improvement['percentage_change']
                current = improvement['current_value']
                baseline = improvement['baseline_value']

                report_lines.append(f"✅ {metric}")
                report_lines.append(f"   Current: {current:.4f}")
                report_lines.append(f"   Baseline: {baseline:.4f}")
                report_lines.append(f"   Change: {change:+.2f}%")
                report_lines.append("")

        # Summary
        if regression_count == 0:
            report_lines.append("🎉 NO PERFORMANCE REGRESSIONS DETECTED")
            report_lines.append("All metrics are within acceptable thresholds.")
        else:
            report_lines.append(f"⚠️  {regression_count} PERFORMANCE REGRESSIONS DETECTED")
            report_lines.append("Review the regressions above and consider optimizations.")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def save_baseline(self, benchmark_file: str, baseline_output: str):
        """Save current benchmark results as new baseline."""
        try:
            benchmark_data = self.load_benchmark_results(benchmark_file)

            if not benchmark_data:
                print("❌ No benchmark data to save as baseline")
                return False

            # Add metadata
            baseline_data = {
                'created_at': datetime.utcnow().isoformat(),
                'source_file': benchmark_file,
                'metrics': self.extract_metrics(benchmark_data),
                'raw_data': benchmark_data
            }

            with open(baseline_output, 'w') as f:
                json.dump(baseline_data, f, indent=2)

            print(f"✅ Saved baseline to {baseline_output}")
            return True

        except Exception as e:
            print(f"❌ Failed to save baseline: {e}")
            return False

    def check_regression(self, benchmark_file: str) -> bool:
        """Check for performance regressions against baseline."""

        # Load current benchmark results
        current_data = self.load_benchmark_results(benchmark_file)
        if not current_data:
            print("❌ Could not load current benchmark results")
            return False

        current_metrics = self.extract_metrics(current_data)
        if not current_metrics:
            print("❌ No metrics found in current benchmark results")
            return False

        # Load baseline
        baseline_data = self.load_baseline()
        if not baseline_data:
            print("⚠️  No baseline found - saving current results as baseline")
            baseline_file = f"baseline-{datetime.now().strftime('%Y%m%d')}.json"
            self.save_baseline(benchmark_file, baseline_file)
            return True

        baseline_metrics = baseline_data.get('metrics', {})
        if not baseline_metrics:
            print("❌ No metrics found in baseline data")
            return False

        # Compare metrics
        comparison_results = self.compare_metrics(current_metrics, baseline_metrics)

        # Generate and print report
        report = self.generate_report(comparison_results)
        print(report)

        # Return True if no regressions, False if regressions found
        return comparison_results['regression_count'] == 0


def main():
    parser = argparse.ArgumentParser(description='Check for performance regressions')
    parser.add_argument('benchmark_file', help='Path to benchmark results JSON file')
    parser.add_argument('--baseline', help='Path to baseline benchmark file')
    parser.add_argument('--save-baseline', help='Save current results as baseline to specified file')
    parser.add_argument('--threshold-response-time', type=float, default=1.2,
                       help='Regression threshold for response time (default: 1.2)')
    parser.add_argument('--threshold-memory', type=float, default=1.15,
                       help='Regression threshold for memory usage (default: 1.15)')
    parser.add_argument('--threshold-cpu', type=float, default=1.25,
                       help='Regression threshold for CPU usage (default: 1.25)')

    args = parser.parse_args()

    checker = PerformanceRegressionChecker(args.baseline)

    # Update thresholds if provided
    if args.threshold_response_time:
        checker.regression_thresholds['response_time'] = args.threshold_response_time
    if args.threshold_memory:
        checker.regression_thresholds['memory_usage'] = args.threshold_memory
    if args.threshold_cpu:
        checker.regression_thresholds['cpu_usage'] = args.threshold_cpu

    # Save baseline if requested
    if args.save_baseline:
        success = checker.save_baseline(args.benchmark_file, args.save_baseline)
        return 0 if success else 1

    # Check for regressions
    no_regressions = checker.check_regression(args.benchmark_file)
    return 0 if no_regressions else 1


if __name__ == "__main__":
    sys.exit(main())
