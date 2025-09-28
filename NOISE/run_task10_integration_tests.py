#!/usr/bin/env python3
"""
Task 10.2 Integration Test Runner
Executes comprehensive integration tests for complete workflow scenarios.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_integration_test_suite():
    """Run the complete integration test suite for Task 10.2."""

    print("🔗 Task 10.2: Running Integration Tests")
    print("=" * 60)

    # Test files to run
    test_files = [
        "tests/integration/test_complete_workflow.py",
        "tests/integration/test_budget_operation_integration.py"
    ]

    # Check if test files exist
    missing_files = []
    for test_file in test_files:
        if not Path(test_file).exists():
            missing_files.append(test_file)

    if missing_files:
        print("❌ Missing test files:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    # Run each test file
    all_passed = True
    results = {}

    for test_file in test_files:
        print(f"\n🔗 Running: {test_file}")
        print("-" * 40)

        try:
            # Run pytest with verbose output
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--no-header",
                "-x"  # Stop on first failure for integration tests
            ], capture_output=True, text=True, timeout=180)  # Longer timeout for integration tests

            if result.returncode == 0:
                print("✅ PASSED")
                results[test_file] = "PASSED"
            else:
                print("❌ FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                results[test_file] = "FAILED"
                all_passed = False

        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT")
            results[test_file] = "TIMEOUT"
            all_passed = False
        except Exception as e:
            print(f"💥 ERROR: {e}")
            results[test_file] = f"ERROR: {e}"
            all_passed = False

    # Print summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)

    for test_file, status in results.items():
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"{status_icon} {Path(test_file).name}: {status}")

    if all_passed:
        print("\n🎉 ALL INTEGRATION TESTS PASSED! Task 10.2 completed successfully.")
        return True
    else:
        print("\n⚠️  Some integration tests failed. Please review the output above.")
        return False


def check_integration_test_environment():
    """Check if integration test environment is properly configured."""

    print("🔍 Checking integration test environment...")

    # Check required environment variables
    required_env_vars = [
        "OPENROUTER_API_KEY",
        "METACULUS_TOKEN",
        "ASKNEWS_CLIENT_ID",
        "ASKNEWS_SECRET"
    ]

    missing_env_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_env_vars.append(var)

    if missing_env_vars:
        print("⚠️  Missing environment variables (will use test defaults):")
        for var in missing_env_vars:
            print(f"   - {var}")

        # Set test defaults
        os.environ.update({
            "OPENROUTER_API_KEY": "test_openrouter_key",
            "METACULUS_TOKEN": "test_metaculus_token",
            "ASKNEWS_CLIENT_ID": "test_asknews_client",
            "ASKNEWS_SECRET": "test_asknews_secret",
            "APP_ENV": "test",
            "DRY_RUN": "true"
        })
        print("✅ Test environment variables set")
    else:
        print("✅ All environment variables available")

    # Check test directories exist
    test_dirs = ["tests", "tests/integration"]
    for test_dir in test_dirs:
        if not Path(test_dir).exists():
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created test directory: {test_dir}")

    return True


def main():
    """Main integration test execution function."""

    print("🚀 Task 10.2: Enhanced Tri-Model System Integration Tests")
    print("Testing: Complete Workflow, Budget Operations, Error Recovery")
    print()

    # Check integration test environment
    if not check_integration_test_environment():
        sys.exit(1)

    # Run the integration test suite
    success = run_integration_test_suite()

    if success:
        print("\n✨ Task 10.2 completed successfully!")
        print("Ready to proceed to Task 10.3 (Performance Tests)")
        sys.exit(0)
    else:
        print("\n❌ Task 10.2 failed. Please fix issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
