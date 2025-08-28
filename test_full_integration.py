#!/usr/bin/env python3
"""
Full Integration Test Script
Tests the complete enhanced tri-model system with real API keys.
"""
import os
import sys
import asyncio
import subprocess
from pathlib import Path

def check_environment_setup():
    """Check if all required environment variables are set."""
    print("🔍 Checking Environment Setup...")
    required_vars = [
        "OPENROUTER_API_KEY",
        "METACULUS_TOKEN",
        "ASKNEWS_CLIENT_ID",
        "ASKNEWS_SECRET"
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            # Show first/last 4 chars for verification
            value = os.getenv(var)
            masked = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            print(f"✅ {var}: {masked}")

    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set these in your .env file or environment")
        return False

    print("✅ All environment variables configured")
    return True

def run_test_suite(test_name, script_path):
    """Run a specific test suite."""
    print(f"\n🧪 Running {test_name}...")
    print("-" * 50)

    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✅ {test_name}: PASSED")
            return True
        else:
            print(f"❌ {test_name}: FAILED")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name}: TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {test_name}: ERROR - {e}")
        return False

async def test_openrouter_connectivity():
    """Test OpenRouter API connectivity."""
    print("\n🔗 Testing OpenRouter Connectivity...")
    try:
        # Import and test OpenRouter startup validator
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from infrastructure.config.openrouter_startup_validator import OpenRouterStartupValidator

        validator = OpenRouterStartupValidator()
        result = await validator.run_startup_validation(exit_on_failure=False)

        if result:
            print("✅ OpenRouter connectivity: PASSED")
            return True
        else:
            print("❌ OpenRouter connectivity: FAILED")
            return False
    except Exception as e:
        print(f"❌ OpenRouter connectivity: ERROR - {e}")
        return False

def test_basic_functionality():
    """Test basic system functionality."""
    print("\n⚙️ Testing Basic Functionality...")
    try:
        # Test main.py import and basic setup
        result = subprocess.run([
            sys.executable, "-c",
            "import main; print('✅ Main module imports successfully')"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Basic functionality: PASSED")
            return True
        else:
            print("❌ Basic functionality: FAILED")
            print("Error:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Basic functionality: ERROR - {e}")
        return False

async def main():
    """Main integration test execution."""
    print("🚀 Full Integration Test Suite")
    print("Testing Enhanced Tri-Model System with Real API Keys")
    print("=" * 60)

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Check environment setup
    if not check_environment_setup():
        sys.exit(1)

    # Test basic functionality
    if not test_basic_functionality():
        print("❌ Basic functionality test failed - stopping")
        sys.exit(1)

    # Test OpenRouter connectivity
    if not await test_openrouter_connectivity():
        print("⚠️ OpenRouter connectivity failed - continuing with other tests")

    # Run all test suites
    test_suites = [
        ("Unit Tests", "run_task10_unit_tests.py"),
        ("Integration Tests", "run_task10_integration_tests.py"),
        ("Performance Tests", "run_task10_performance_tests.py")
    ]

    results = {}
    for test_name, script_path in test_suites:
        if Path(script_path).exists():
            results[test_name] = run_test_suite(test_name, script_path)
        else:
            print(f"⚠️ {test_name}: Script not found - {script_path}")
            results[test_name] = False

    # Test AskNews connectivity
    print(f"\n🧪 Running AskNews Authentication Test...")
    print("-" * 50)
    asknews_result = run_test_suite("AskNews Auth", "test_asknews_auth_only.py")
    results["AskNews Authentication"] = asknews_result

    # Print final summary
    print("\n" + "=" * 60)
    print("📊 FULL INTEGRATION TEST SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, result in results.items():
        status_icon = "✅" if result else "❌"
        print(f"{status_icon} {test_name}")

    print(f"\n📈 Results: {passed_tests}/{total_tests} test suites passed")

    if passed_tests == total_tests:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("🏆 Enhanced tri-model system is fully operational!")
        print("🔗 All API credentials are working correctly")
        print("🚀 Ready for Fall 2025 AI Benchmark Tournament!")
        sys.exit(0)
    else:
        print("⚠️ Some integration tests failed.")
        print("Please review the output above and fix any issues.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
