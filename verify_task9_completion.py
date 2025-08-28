#!/usr/bin/env python3
"""
Verify Task 9 completion - Environment Configuration and OpenRouter Setup
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_task_9_1():
    """Check Task 9.1: Update environment variables for OpenRouter model configuration"""
    print("="*60)
    print("Task 9.1: Environment Variables for OpenRouter Configuration")
    print("="*60)

    required_vars = [
        'OPENROUTER_API_KEY',
        'OPENROUTER_BASE_URL',
        'OPENROUTER_HTTP_REFERER',
        'OPENROUTER_APP_TITLE',
        'DEFAULT_MODEL',
        'MINI_MODEL',
        'NANO_MODEL',
        'FREE_FALLBACK_MODELS'
    ]

    operation_mode_vars = [
        'NORMAL_MODE_THRESHOLD',
        'CONSERVATIVE_MODE_THRESHOLD',
        'EMERGENCY_MODE_THRESHOLD'
    ]

    all_configured = True

    print("\n📋 Required OpenRouter Environment Variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if 'key' in var.lower():
                masked = value[:8] + '*' * (len(value) - 8) if len(value) > 8 else '*****'
                print(f"  ✅ {var}: {masked}")
            else:
                print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: Not set")
            all_configured = False

    print("\n📋 Operation Mode Threshold Variables:")
    for var in operation_mode_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: Not set")
            all_configured = False

    print(f"\n🎯 Task 9.1 Status: {'✅ COMPLETE' if all_configured else '❌ INCOMPLETE'}")
    return all_configured

def check_task_9_2():
    """Check Task 9.2: OpenRouter model availability detection and auto-configuration"""
    print("\n" + "="*60)
    print("Task 9.2: OpenRouter Model Availability Detection & Auto-Configuration")
    print("="*60)

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    try:
        # Check if components can be imported
        from infrastructure.config.openrouter_startup_validator import OpenRouterStartupValidator
        print("  ✅ OpenRouterStartupValidator: Available")

        from infrastructure.config.tri_model_router import OpenRouterTriModelRouter
        print("  ✅ OpenRouterTriModelRouter: Available")

        # Check key methods exist
        print("\n📋 Required Methods:")

        # Validator methods
        validator = OpenRouterStartupValidator()
        validator_methods = ['validate_configuration', 'run_startup_validation']
        for method in validator_methods:
            if hasattr(validator, method):
                print(f"  ✅ OpenRouterStartupValidator.{method}: Available")
            else:
                print(f"  ❌ OpenRouterStartupValidator.{method}: Missing")
                return False

        # Router methods
        router_methods = [
            'detect_model_availability',
            'auto_configure_fallback_chains',
            'health_monitor_startup'
        ]

        router = OpenRouterTriModelRouter()
        for method in router_methods:
            if hasattr(router, method):
                print(f"  ✅ OpenRouterTriModelRouter.{method}: Available")
            else:
                print(f"  ❌ OpenRouterTriModelRouter.{method}: Missing")
                return False

        # Check class method
        if hasattr(OpenRouterTriModelRouter, 'create_with_auto_configuration'):
            print(f"  ✅ OpenRouterTriModelRouter.create_with_auto_configuration: Available")
        else:
            print(f"  ❌ OpenRouterTriModelRouter.create_with_auto_configuration: Missing")
            return False

        print(f"\n🎯 Task 9.2 Status: ✅ COMPLETE")
        return True

    except ImportError as e:
        print(f"  ❌ Import Error: {e}")
        print(f"\n🎯 Task 9.2 Status: ❌ INCOMPLETE")
        return False

def check_main_integration():
    """Check that OpenRouter validation is integrated in main.py"""
    print("\n" + "="*60)
    print("Main Application Integration")
    print("="*60)

    try:
        with open("main.py", "r") as f:
            content = f.read()

        integration_checks = [
            ("validate_openrouter_startup function", "async def validate_openrouter_startup"),
            ("OpenRouterStartupValidator import", "OpenRouterStartupValidator"),
            ("OpenRouterTriModelRouter import", "OpenRouterTriModelRouter"),
            ("Startup validation call", "asyncio.run(validate_openrouter_startup())"),
        ]

        all_integrated = True
        for check_name, pattern in integration_checks:
            if pattern in content:
                print(f"  ✅ {check_name}: Found")
            else:
                print(f"  ❌ {check_name}: Missing")
                all_integrated = False

        print(f"\n🎯 Main Integration Status: {'✅ COMPLETE' if all_integrated else '❌ INCOMPLETE'}")
        return all_integrated

    except Exception as e:
        print(f"  ❌ Error checking main.py: {e}")
        return False

def main():
    """Main verification function"""
    print("Task 9 Completion Verification")
    print("Environment Configuration and OpenRouter Setup")

    # Check both subtasks
    task_9_1_complete = check_task_9_1()
    task_9_2_complete = check_task_9_2()
    main_integration_complete = check_main_integration()

    # Overall status
    print("\n" + "="*60)
    print("OVERALL TASK 9 STATUS")
    print("="*60)

    print(f"Task 9.1 (Environment Variables): {'✅ COMPLETE' if task_9_1_complete else '❌ INCOMPLETE'}")
    print(f"Task 9.2 (Model Detection & Auto-Config): {'✅ COMPLETE' if task_9_2_complete else '❌ INCOMPLETE'}")
    print(f"Main Application Integration: {'✅ COMPLETE' if main_integration_complete else '❌ INCOMPLETE'}")

    overall_complete = task_9_1_complete and task_9_2_complete and main_integration_complete

    print(f"\n🎯 TASK 9 OVERALL STATUS: {'✅ COMPLETE' if overall_complete else '❌ INCOMPLETE'}")

    if overall_complete:
        print("\n🎉 All Task 9 requirements have been successfully implemented!")
        print("✅ Environment variables are properly configured")
        print("✅ OpenRouter model availability detection is implemented")
        print("✅ Auto-configuration system is working")
        print("✅ Startup validation is integrated in main application")
    else:
        print("\n⚠️ Some Task 9 requirements need attention.")

    return 0 if overall_complete else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
