#!/usr/bin/env python3
"""
Security scanning integration for CI/CD pipeline.

This script performs security scans on the codebase and can be integrated
into CI/CD pipelines to ensure security standards are maintained.
"""

import os
import sys
import subprocess
import json
import argparse
from typing import Dict, List, Any
from pathlib import Path


class SecurityScanner:
    """Security scanner for CI/CD integration."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            "security_tests": {"passed": 0, "failed": 0, "errors": []},
            "dependency_scan": {"vulnerabilities": [], "high_risk": 0, "medium_risk": 0, "low_risk": 0},
            "code_scan": {"issues": [], "critical": 0, "high": 0, "medium": 0, "low": 0},
            "configuration_scan": {"issues": [], "errors": []},
            "overall_status": "unknown"
        }

    def run_security_tests(self) -> bool:
        """Run security-focused unit and integration tests."""
        print("🔒 Running security tests...")

        try:
            # Run security unit tests
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/unit/infrastructure/security/",
                "-v", "--tb=short", "--json-report", "--json-report-file=security_test_results.json"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                self.results["security_tests"]["passed"] += 1
                print("✅ Security unit tests passed")
            else:
                self.results["security_tests"]["failed"] += 1
                self.results["security_tests"]["errors"].append(result.stderr)
                print("❌ Security unit tests failed")

            # Run penetration tests
            pen_result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/integration/test_security_penetration.py",
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=self.project_root)

            if pen_result.returncode == 0:
                self.results["security_tests"]["passed"] += 1
                print("✅ Penetration tests passed")
            else:
                self.results["security_tests"]["failed"] += 1
                self.results["security_tests"]["errors"].append(pen_result.stderr)
                print("❌ Penetration tests failed")

            return result.returncode == 0 and pen_result.returncode == 0

        except Exception as e:
            self.results["security_tests"]["errors"].append(str(e))
            print(f"❌ Error running security tests: {e}")
            return False

    def scan_dependencies(self) -> bool:
        """Scan dependencies for known vulnerabilities."""
        print("🔍 Scanning dependencies for vulnerabilities...")

        try:
            # Check if safety is installed
            safety_result = subprocess.run([
                sys.executable, "-m", "pip", "show", "safety"
            ], capture_output=True, text=True)

            if safety_result.returncode != 0:
                print("⚠️  Safety not installed, skipping dependency scan")
                print("   Install with: pip install safety")
                return True

            # Run safety check
            result = subprocess.run([
                sys.executable, "-m", "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                print("✅ No known vulnerabilities found in dependencies")
                return True
            else:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    self.results["dependency_scan"]["vulnerabilities"] = vulnerabilities

                    for vuln in vulnerabilities:
                        severity = vuln.get("severity", "unknown").lower()
                        if severity == "high":
                            self.results["dependency_scan"]["high_risk"] += 1
                        elif severity == "medium":
                            self.results["dependency_scan"]["medium_risk"] += 1
                        else:
                            self.results["dependency_scan"]["low_risk"] += 1

                    print(f"⚠️  Found {len(vulnerabilities)} vulnerabilities in dependencies")
                    return len(vulnerabilities) == 0

                except json.JSONDecodeError:
                    print("❌ Error parsing safety output")
                    return False

        except Exception as e:
            print(f"❌ Error scanning dependencies: {e}")
            return False

    def scan_code_security(self) -> bool:
        """Scan code for security issues using bandit."""
        print("🔍 Scanning code for security issues...")

        try:
            # Check if bandit is installed
            bandit_result = subprocess.run([
                sys.executable, "-m", "pip", "show", "bandit"
            ], capture_output=True, text=True)

            if bandit_result.returncode != 0:
                print("⚠️  Bandit not installed, skipping code security scan")
                print("   Install with: pip install bandit")
                return True

            # Run bandit scan
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", "src/", "-f", "json"
            ], capture_output=True, text=True, cwd=self.project_root)

            try:
                bandit_output = json.loads(result.stdout)
                issues = bandit_output.get("results", [])
                self.results["code_scan"]["issues"] = issues

                for issue in issues:
                    severity = issue.get("issue_severity", "LOW").upper()
                    if severity == "CRITICAL":
                        self.results["code_scan"]["critical"] += 1
                    elif severity == "HIGH":
                        self.results["code_scan"]["high"] += 1
                    elif severity == "MEDIUM":
                        self.results["code_scan"]["medium"] += 1
                    else:
                        self.results["code_scan"]["low"] += 1

                critical_high = self.results["code_scan"]["critical"] + self.results["code_scan"]["high"]

                if critical_high == 0:
                    print("✅ No critical or high severity security issues found")
                    return True
                else:
                    print(f"⚠️  Found {critical_high} critical/high severity security issues")
                    return False

            except json.JSONDecodeError:
                print("❌ Error parsing bandit output")
                return False

        except Exception as e:
            print(f"❌ Error scanning code security: {e}")
            return False

    def scan_configuration(self) -> bool:
        """Scan configuration for security issues."""
        print("🔍 Scanning configuration for security issues...")

        issues = []

        # Check for hardcoded secrets in config files
        config_files = [
            ".env.example",
            ".env.template",
            "pyproject.toml"
        ]

        dangerous_patterns = [
            "password=",
            "secret=",
            "api_key=",
            "token=",
            "private_key="
        ]

        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    content = config_path.read_text().lower()
                    for pattern in dangerous_patterns:
                        if pattern in content and "example" not in pattern and "template" not in pattern:
                            issues.append(f"Potential hardcoded secret in {config_file}: {pattern}")
                except Exception as e:
                    issues.append(f"Error reading {config_file}: {e}")

        # Check for insecure permissions on sensitive files
        sensitive_files = [
            ".env",
            "credentials.json",
            "private_key.pem"
        ]

        for sensitive_file in sensitive_files:
            file_path = self.project_root / sensitive_file
            if file_path.exists():
                try:
                    stat = file_path.stat()
                    # Check if file is readable by others (octal 044)
                    if stat.st_mode & 0o044:
                        issues.append(f"Insecure permissions on {sensitive_file}")
                except Exception as e:
                    issues.append(f"Error checking permissions on {sensitive_file}: {e}")

        self.results["configuration_scan"]["issues"] = issues

        if len(issues) == 0:
            print("✅ No configuration security issues found")
            return True
        else:
            print(f"⚠️  Found {len(issues)} configuration security issues")
            for issue in issues:
                print(f"   - {issue}")
            return False

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        # Determine overall status
        all_passed = (
            self.results["security_tests"]["failed"] == 0 and
            self.results["dependency_scan"]["high_risk"] == 0 and
            self.results["code_scan"]["critical"] == 0 and
            self.results["code_scan"]["high"] == 0 and
            len(self.results["configuration_scan"]["issues"]) == 0
        )

        self.results["overall_status"] = "PASS" if all_passed else "FAIL"

        return self.results

    def save_report(self, filename: str = "security_report.json"):
        """Save security report to file."""
        report_path = self.project_root / filename
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Security report saved to {report_path}")

    def print_summary(self):
        """Print security scan summary."""
        print("\n" + "="*60)
        print("🔒 SECURITY SCAN SUMMARY")
        print("="*60)

        print(f"Overall Status: {'✅ PASS' if self.results['overall_status'] == 'PASS' else '❌ FAIL'}")
        print()

        print("Security Tests:")
        print(f"  ✅ Passed: {self.results['security_tests']['passed']}")
        print(f"  ❌ Failed: {self.results['security_tests']['failed']}")
        print()

        print("Dependency Vulnerabilities:")
        print(f"  🔴 High Risk: {self.results['dependency_scan']['high_risk']}")
        print(f"  🟡 Medium Risk: {self.results['dependency_scan']['medium_risk']}")
        print(f"  🟢 Low Risk: {self.results['dependency_scan']['low_risk']}")
        print()

        print("Code Security Issues:")
        print(f"  🔴 Critical: {self.results['code_scan']['critical']}")
        print(f"  🟠 High: {self.results['code_scan']['high']}")
        print(f"  🟡 Medium: {self.results['code_scan']['medium']}")
        print(f"  🟢 Low: {self.results['code_scan']['low']}")
        print()

        print("Configuration Issues:")
        print(f"  📋 Issues Found: {len(self.results['configuration_scan']['issues'])}")
        print()


def main():
    """Main entry point for security scanner."""
    parser = argparse.ArgumentParser(description="Security scanner for CI/CD integration")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", default="security_report.json", help="Output report file")
    parser.add_argument("--fail-on-issues", action="store_true", help="Exit with error code if issues found")
    parser.add_argument("--skip-tests", action="store_true", help="Skip security tests")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency scan")
    parser.add_argument("--skip-code", action="store_true", help="Skip code security scan")
    parser.add_argument("--skip-config", action="store_true", help="Skip configuration scan")

    args = parser.parse_args()

    scanner = SecurityScanner(args.project_root)

    print("🚀 Starting security scan...")
    print()

    all_passed = True

    if not args.skip_tests:
        if not scanner.run_security_tests():
            all_passed = False

    if not args.skip_deps:
        if not scanner.scan_dependencies():
            all_passed = False

    if not args.skip_code:
        if not scanner.scan_code_security():
            all_passed = False

    if not args.skip_config:
        if not scanner.scan_configuration():
            all_passed = False

    # Generate and save report
    report = scanner.generate_report()
    scanner.save_report(args.output)
    scanner.print_summary()

    # Exit with appropriate code
    if args.fail_on_issues and not all_passed:
        print("\n❌ Security scan failed - exiting with error code 1")
        sys.exit(1)
    else:
        print("\n✅ Security scan completed")
        sys.exit(0)


if __name__ == "__main__":
    main()
