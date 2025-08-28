#!/usr/bin/env python3
"""
Task 10.1 Unit Test Runner
Executes comprehensive unit tests for enhanced tri-model system components.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_test_suite():
    """Run the complete unit test suite for Task 10.1."""

    print("🧪 Task 10.1: Running Comprehensive Unit Tests")
    print("=" * 60)

    # Test files to run
    test_files = [
        "tests/unit/infrastructure/test_enhanced_tri_model_router.py",
        "tests/unit/domain/test_multi_stage_validation_pipeline.py",
        "tests/unit/infrastructure/test_budget_aware_operation_manager.py"
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
        print(f"\n📋 Running: {test_file}")
        print("-" * 40)

        try:
            # Run pytest with verbose output
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--no-header"
            ], capture_output=True, text=True, timeout=120)

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
    print("📊 TEST SUMMARY")
    print("=" * 60)

    for test_file, status in results.items():
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"{status_icon} {Path(test_file).name}: {status}")

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Task 10.1 unit tests completed successfully.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return False


def check_dependencies():
    """Check if required dependencies are available."""

    print("🔍 Checking dependencies...")

    required_modules = [
        "pytest",
        "asyncio",
        "unittest.mock"
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing_modules.append(module)

    if missing_modules:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_modules)}")
        print("Install with: pip install pytest")
        return False

    print("✅ All dependencies available")
    return True


def main():
    """Main test execution function."""

    print("🚀 Task 10.1: Enhanced Tri-Model System Unit Tests")
    print("Testing: Router, Pipeline, Budget Manager, Anti-Slop Prompts")
    print()

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Run the test suite
    success = run_test_suite()

    if success:
        print("\n✨ Task 10.1 completed successfully!")
        print("Ready to proceed to Task 10.2 (Integration Tests)")
        sys.exit(0)
    else:
        print("\n❌ Task 10.1 failed. Please fix issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
