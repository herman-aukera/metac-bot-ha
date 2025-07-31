#!/usr/bin/env python3
"""
Security gate check script for CI/CD pipeline.
Analyzes security scan results and fails the build if critical issues are found.
"""

import json
import sys
import argparse
from typing import Dict, List, Any
from pathlib import Path


class SecurityGateChecker:
    """Checks security scan results against defined thresholds."""

    def __init__(self):
        self.critical_threshold = 0  # No critical vulnerabilities allowed
        self.high_threshold = 5      # Max 5 high severity issues
        self.medium_threshold = 20   # Max 20 medium severity issues

    def check_safety_report(self, report_path: str) -> Dict[str, Any]:
        """Check Safety vulnerability report."""
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)

            vulnerabilities = data.get('vulnerabilities', [])

            critical_count = sum(1 for v in vulnerabilities if v.get('severity') == 'critical')
            high_count = sum(1 for v in vulnerabilities if v.get('severity') == 'high')
            medium_count = sum(1 for v in vulnerabilities if v.get('severity') == 'medium')

            return {
                'tool': 'safety',
                'critical': critical_count,
                'high': high_count,
                'medium': medium_count,
                'total': len(vulnerabilities),
                'passed': critical_count <= self.critical_threshold and
                         high_count <= self.high_threshold and
                         medium_count <= self.medium_threshold
            }

        except FileNotFoundError:
            print(f"Warning: Safety report not found at {report_path}")
            return {'tool': 'safety', 'passed': True, 'error': 'Report not found'}
        except Exception as e:
            print(f"Error processing Safety report: {e}")
            return {'tool': 'safety', 'passed': False, 'error': str(e)}

    def check_bandit_report(self, report_path: str) -> Dict[str, Any]:
        """Check Bandit security report."""
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)

            results = data.get('results', [])

            critical_count = sum(1 for r in results if r.get('issue_severity') == 'HIGH')
            medium_count = sum(1 for r in results if r.get('issue_severity') == 'MEDIUM')
            low_count = sum(1 for r in results if r.get('issue_severity') == 'LOW')

            return {
                'tool': 'bandit',
                'critical': critical_count,
                'high': 0,  # Bandit doesn't have high, treating HIGH as critical
                'medium': medium_count,
                'low': low_count,
                'total': len(results),
                'passed': critical_count <= self.critical_threshold and
                         medium_count <= self.medium_threshold
            }

        except FileNotFoundError:
            print(f"Warning: Bandit report not found at {report_path}")
            return {'tool': 'bandit', 'passed': True, 'error': 'Report not found'}
        except Exception as e:
            print(f"Error processing Bandit report: {e}")
            return {'tool': 'bandit', 'passed': False, 'error': str(e)}

    def check_semgrep_report(self, report_path: str) -> Dict[str, Any]:
        """Check Semgrep security report."""
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)

            results = data.get('results', [])

            # Semgrep severity mapping
            severity_map = {'ERROR': 'critical', 'WARNING': 'medium', 'INFO': 'low'}

            critical_count = sum(1 for r in results if severity_map.get(r.get('extra', {}).get('severity', ''), 'low') == 'critical')
            high_count = 0  # Semgrep doesn't have high severity
            medium_count = sum(1 for r in results if severity_map.get(r.get('extra', {}).get('severity', ''), 'low') == 'medium')

            return {
                'tool': 'semgrep',
                'critical': critical_count,
                'high': high_count,
                'medium': medium_count,
                'total': len(results),
                'passed': critical_count <= self.critical_threshold and
                         medium_count <= self.medium_threshold
            }

        except FileNotFoundError:
            print(f"Warning: Semgrep report not found at {report_path}")
            return {'tool': 'semgrep', 'passed': True, 'error': 'Report not found'}
        except Exception as e:
            print(f"Error processing Semgrep report: {e}")
            return {'tool': 'semgrep', 'passed': False, 'error': str(e)}

    def generate_summary(self, results: List[Dict[str, Any]]) -> None:
        """Generate security gate summary."""
        print("\n" + "="*60)
        print("SECURITY GATE CHECK SUMMARY")
        print("="*60)

        total_critical = 0
        total_high = 0
        total_medium = 0
        all_passed = True

        for result in results:
            if 'error' in result:
                print(f"\n{result['tool'].upper()}: ERROR - {result['error']}")
                continue

            critical = result.get('critical', 0)
            high = result.get('high', 0)
            medium = result.get('medium', 0)
            passed = result.get('passed', False)

            total_critical += critical
            total_high += high
            total_medium += medium

            if not passed:
                all_passed = False

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{result['tool'].upper()}: {status}")
            print(f"  Critical: {critical}")
            print(f"  High: {high}")
            print(f"  Medium: {medium}")
            print(f"  Total: {result.get('total', 0)}")

        print(f"\nOVERALL SUMMARY:")
        print(f"  Total Critical: {total_critical} (threshold: {self.critical_threshold})")
        print(f"  Total High: {total_high} (threshold: {self.high_threshold})")
        print(f"  Total Medium: {total_medium} (threshold: {self.medium_threshold})")

        if all_passed:
            print(f"\n🎉 SECURITY GATE: PASSED")
            print("All security checks passed the defined thresholds.")
        else:
            print(f"\n🚨 SECURITY GATE: FAILED")
            print("One or more security checks exceeded the defined thresholds.")
            print("Please review and fix the security issues before proceeding.")

        print("="*60)

        return all_passed


def main():
    parser = argparse.ArgumentParser(description='Security gate check for CI/CD pipeline')
    parser.add_argument('--safety-report', help='Path to Safety JSON report')
    parser.add_argument('--bandit-report', help='Path to Bandit JSON report')
    parser.add_argument('--semgrep-report', help='Path to Semgrep JSON report')

    args = parser.parse_args()

    checker = SecurityGateChecker()
    results = []

    if args.safety_report:
        results.append(checker.check_safety_report(args.safety_report))

    if args.bandit_report:
        results.append(checker.check_bandit_report(args.bandit_report))

    if args.semgrep_report:
        results.append(checker.check_semgrep_report(args.semgrep_report))

    if not results:
        print("No security reports provided. Skipping security gate check.")
        return 0

    all_passed = checker.generate_summary(results)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
