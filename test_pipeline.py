#!/usr/bin/env python3
"""
Pipeline Test Script
Simulates the CI/CD pipeline to test if all components work correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔍 {description}")
    print(f"Running: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run pipeline tests."""
    print("🚀 Starting Pipeline Tests")
    print("=" * 50)

    tests = [
        # Poetry and dependency tests
        ("poetry check", "Poetry Configuration Check"),
        ("poetry run python --version", "Python Environment Check"),

        # Code quality tests
        ("poetry run black --check --diff .", "Code Formatting Check (Black)"),
        ("poetry run isort --check-only .", "Import Sorting Check (isort)"),

        # Basic linting (if files exist)
        ("poetry run flake8 --version", "Flake8 Availability Check"),

        # Test framework check
        ("poetry run pytest --collect-only -q", "Test Discovery"),

        # Security tools check
        ("poetry run bandit --version", "Security Scanner Check"),

        # Type checking
        ("poetry run mypy --version", "Type Checker Check"),
    ]

    passed = 0
    failed = 0

    for cmd, description in tests:
        if run_command(cmd, description):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print("📊 PIPELINE TEST RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 All pipeline tests passed! The CI/CD pipeline should work correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
