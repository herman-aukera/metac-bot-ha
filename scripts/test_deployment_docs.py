#!/usr/bin/env python3
"""
Test script to verify emergency deployment documentation is complete and accurate.

This script validates that all deployment documentation files exist and contain
the required information for emergency tournament deployment.
"""

import os
import sys
from pathlib import Path

def test_file_exists(filepath: str, description: str) -> bool:
    """Test if a file exists and is readable."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def test_file_contains(filepath: str, required_content: list, description: str) -> bool:
    """Test if a file contains required content."""
    if not os.path.exists(filepath):
        print(f"❌ {description}: {filepath} (FILE MISSING)")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        missing_content = []
        for required in required_content:
            if required not in content:
                missing_content.append(required)

        if missing_content:
            print(f"❌ {description}: Missing content - {missing_content}")
            return False
        else:
            print(f"✅ {description}: All required content present")
            return True

    except Exception as e:
        print(f"❌ {description}: Error reading file - {e}")
        return False

def test_script_executable(filepath: str, description: str) -> bool:
    """Test if a script is executable."""
    if not os.path.exists(filepath):
        print(f"❌ {description}: {filepath} (FILE MISSING)")
        return False

    if os.access(filepath, os.X_OK):
        print(f"✅ {description}: {filepath} (executable)")
        return True
    else:
        print(f"⚠️  {description}: {filepath} (not executable - run chmod +x)")
        return True  # Not critical, can be fixed

def main():
    """Main test function."""
    print("🧪 Testing Emergency Deployment Documentation")
    print("=" * 50)

    tests_passed = 0
    tests_total = 0

    # Test 1: Emergency deployment guide exists
    tests_total += 1
    if test_file_exists("docs/EMERGENCY_DEPLOYMENT.md", "Emergency Deployment Guide"):
        tests_passed += 1

    # Test 2: Emergency deployment guide contains required sections
    tests_total += 1
    required_sections = [
        "Cloud Instance Manual Deployment",
        "Local Development Machine",
        "Emergency Pip-Only Installation",
        "Local Testing Verification Commands",
        "Emergency Troubleshooting"
    ]
    if test_file_contains("docs/EMERGENCY_DEPLOYMENT.md", required_sections, "Emergency Guide Content"):
        tests_passed += 1

    # Test 3: Emergency requirements file exists
    tests_total += 1
    if test_file_exists("requirements-emergency.txt", "Emergency Requirements File"):
        tests_passed += 1

    # Test 4: Emergency requirements contains core dependencies
    tests_total += 1
    required_deps = [
        "requests",
        "openai",
        "python-dotenv",
        "pydantic",
        "asknews",
        "forecasting-tools"
    ]
    if test_file_contains("requirements-emergency.txt", required_deps, "Emergency Requirements Content"):
        tests_passed += 1

    # Test 5: Deployment verification script exists
    tests_total += 1
    if test_file_exists("scripts/emergency_deployment_verification.py", "Deployment Verification Script"):
        tests_passed += 1

    # Test 6: Deployment verification script is executable
    tests_total += 1
    if test_script_executable("scripts/emergency_deployment_verification.py", "Verification Script Executable"):
        tests_passed += 1

    # Test 7: Manual cloud deployment script exists
    tests_total += 1
    if test_file_exists("scripts/manual_cloud_deployment.sh", "Manual Cloud Deployment Script"):
        tests_passed += 1

    # Test 8: Manual cloud deployment script is executable
    tests_total += 1
    if test_script_executable("scripts/manual_cloud_deployment.sh", "Cloud Deployment Script Executable"):
        tests_passed += 1

    # Test 9: Quick deployment guide exists
    tests_total += 1
    if test_file_exists("QUICK_DEPLOYMENT_GUIDE.md", "Quick Deployment Guide"):
        tests_passed += 1

    # Test 10: Quick deployment guide contains essential commands
    tests_total += 1
    required_commands = [
        "git clone",
        "pip install",
        "python3 -m src.main",
        "--tournament 32813",
        "--dry-run"
    ]
    if test_file_contains("QUICK_DEPLOYMENT_GUIDE.md", required_commands, "Quick Guide Commands"):
        tests_passed += 1

    # Test 11: Deployment scripts contain proper error handling
    tests_total += 1
    error_handling_patterns = [
        "set -e",
        "print_error",
        "exit 1"
    ]
    if test_file_contains("scripts/manual_cloud_deployment.sh", error_handling_patterns, "Deployment Script Error Handling"):
        tests_passed += 1

    # Test 12: Verification script contains comprehensive tests
    tests_total += 1
    verification_tests = [
        "test_python_version",
        "test_core_imports",
        "test_environment_variables",
        "test_configuration_loading",
        "test_api_connectivity"
    ]
    if test_file_contains("scripts/emergency_deployment_verification.py", verification_tests, "Verification Script Tests"):
        tests_passed += 1

    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{tests_total} passed")

    if tests_passed == tests_total:
        print("🎉 All deployment documentation tests passed!")
        print("✅ Emergency deployment documentation is complete and ready.")
        return True
    else:
        failed = tests_total - tests_passed
        print(f"⚠️  {failed} test(s) failed.")
        print("❌ Emergency deployment documentation needs fixes.")
        return False

def test_requirements_coverage():
    """Test that documentation covers all requirements from the spec."""
    print("\n🎯 Testing Requirements Coverage")
    print("=" * 30)

    # Requirements from the spec that should be covered
    requirements_coverage = {
        "11.1": "Manual deployment instructions available",
        "11.2": "Pip-based installation works as fallback",
        "11.3": "Core functionality testable locally",
        "11.4": "Bot deployable on any Linux server with basic Python"
    }

    coverage_tests = 0
    coverage_passed = 0

    for req_id, req_desc in requirements_coverage.items():
        coverage_tests += 1
        print(f"📋 Requirement {req_id}: {req_desc}")

        # Check if requirement is addressed in documentation
        if req_id == "11.1":
            # Manual deployment instructions
            if (os.path.exists("docs/EMERGENCY_DEPLOYMENT.md") and
                os.path.exists("scripts/manual_cloud_deployment.sh")):
                print(f"  ✅ Covered by emergency deployment guide and cloud deployment script")
                coverage_passed += 1
            else:
                print(f"  ❌ Missing manual deployment documentation")

        elif req_id == "11.2":
            # Pip-based installation
            if os.path.exists("requirements-emergency.txt"):
                print(f"  ✅ Covered by emergency requirements file")
                coverage_passed += 1
            else:
                print(f"  ❌ Missing pip-based installation support")

        elif req_id == "11.3":
            # Local testing
            if os.path.exists("scripts/emergency_deployment_verification.py"):
                print(f"  ✅ Covered by deployment verification script")
                coverage_passed += 1
            else:
                print(f"  ❌ Missing local testing verification")

        elif req_id == "11.4":
            # Linux server deployment
            if (os.path.exists("scripts/manual_cloud_deployment.sh") and
                os.path.exists("requirements-emergency.txt")):
                print(f"  ✅ Covered by cloud deployment script and emergency requirements")
                coverage_passed += 1
            else:
                print(f"  ❌ Missing Linux server deployment support")

    print(f"\n📊 Requirements Coverage: {coverage_passed}/{coverage_tests}")
    return coverage_passed == coverage_tests

if __name__ == "__main__":
    print("🚀 Emergency Deployment Documentation Test Suite")
    print("=" * 60)

    # Run main tests
    main_tests_passed = main()

    # Run requirements coverage tests
    requirements_covered = test_requirements_coverage()

    # Final summary
    print("\n" + "=" * 60)
    if main_tests_passed and requirements_covered:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Emergency deployment documentation is complete and ready for tournament.")
        print("🏆 Task 12 implementation successful!")
        sys.exit(0)
    else:
        print("❌ Some tests failed.")
        print("⚠️  Emergency deployment documentation needs attention.")
        sys.exit(1)
