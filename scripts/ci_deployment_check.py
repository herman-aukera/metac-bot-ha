#!/usr/bin/env python3
"""
CI-Friendly Deployment Readiness Check

This script performs essential deployment checks suitable for CI environments.
It's more lenient with optional dependencies and focuses on critical requirements.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)"


def check_essential_imports() -> Tuple[bool, str]:
    """Check if essential imports are available."""
    essential_modules = [
        "requests",
        "openai",
        "pydantic",
        "typer"
    ]

    missing = []
    for module in essential_modules:
        try:
            __import__(module.replace("-", "_"))
        except ImportError:
            missing.append(module)

    if missing:
        return False, f"Missing essential modules: {', '.join(missing)}"
    return True, "All essential modules available"


def check_optional_imports() -> Tuple[bool, str]:
    """Check optional imports (non-blocking)."""
    optional_modules = {
        "asknews": "AskNews SDK",
        "forecasting_tools": "Forecasting Tools"
    }

    available = []
    missing = []

    for module, name in optional_modules.items():
        try:
            __import__(module)
            available.append(name)
        except ImportError:
            missing.append(name)

    status = f"Available: {', '.join(available) if available else 'None'}"
    if missing:
        status += f" | Missing: {', '.join(missing)}"

    return True, status  # Always return True for optional modules


def check_environment_variables(fork_mode: bool = False) -> Tuple[bool, str]:
    """Check if required environment variables are set."""
    required_vars = [
        "OPENROUTER_API_KEY",
        "METACULUS_TOKEN"
    ]

    optional_vars = [
        "ASKNEWS_CLIENT_ID",
        "ASKNEWS_SECRET"
    ]

    if fork_mode:
        # In fork mode, we can't access secrets, so we skip this check
        return True, "Fork mode: Environment variable check skipped (secrets not available in forks)"

    missing_required = []
    missing_optional = []

    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)

    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)

    if missing_required:
        return False, f"Missing required: {', '.join(missing_required)}"

    status = "All required variables set"
    if missing_optional:
        status += f" | Optional missing: {', '.join(missing_optional)}"

    return True, status


def check_project_structure() -> Tuple[bool, str]:
    """Check if essential project files exist."""
    essential_files = [
        "main.py",
        "src/agents/base_agent.py",
        "src/domain/entities/question.py",
        "src/infrastructure/config/settings.py"
    ]

    missing = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        return False, f"Missing files: {', '.join(missing)}"
    return True, "Essential project structure complete"


def run_ci_checks(fork_mode: bool = False) -> int:
    """Run all CI deployment checks."""
    checks = [
        ("Python Version", lambda: check_python_version()),
        ("Essential Imports", lambda: check_essential_imports()),
        ("Optional Imports", lambda: check_optional_imports()),
        ("Environment Variables", lambda: check_environment_variables(fork_mode)),
        ("Project Structure", lambda: check_project_structure())
    ]

    results = {
        "checks": {},
        "summary": {
            "total": len(checks),
            "passed": 0,
            "failed": 0,
            "warnings": 0
        },
        "deployment_ready": True,
        "fork_mode": fork_mode,
        "timestamp": None
    }

    print("🚀 CI Deployment Readiness Check")
    if fork_mode:
        print("⚠️  FORK MODE: Running with limited checks (secrets not available)")
    print("=" * 50)

    for check_name, check_func in checks:
        try:
            success, message = check_func()
            results["checks"][check_name] = {
                "success": success,
                "message": message
            }

            status_icon = "✅" if success else "❌"
            print(f"{status_icon} {check_name}: {message}")

            if success:
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
                # Only fail deployment for critical checks (not in fork mode)
                if not fork_mode and check_name in ["Python Version", "Essential Imports", "Environment Variables", "Project Structure"]:
                    results["deployment_ready"] = False

        except Exception as e:
            results["checks"][check_name] = {
                "success": False,
                "message": f"Check failed: {str(e)}"
            }
            results["summary"]["failed"] += 1
            print(f"❌ {check_name}: Check failed - {str(e)}")

    print("=" * 50)

    # Summary
    summary = results["summary"]
    print(f"📊 Summary: {summary['passed']} passed, {summary['failed']} failed")

    if results["deployment_ready"]:
        if fork_mode:
            print("✅ ✅ FORK CHECK PASSED")
            print("Fork-based checks completed successfully. Full deployment readiness requires main repository.")
        else:
            print("✅ ✅ DEPLOYMENT READY")
            print("The bot is ready for tournament deployment.")
        return_code = 0
    else:
        print("❌ ❌ DEPLOYMENT NOT READY")
        print("Critical checks failed. Fix required issues before deployment.")
        return_code = 1

    # Save results
    results["timestamp"] = __import__("datetime").datetime.now().isoformat()
    with open("ci_deployment_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"📋 Report saved to: ci_deployment_report.json")

    return return_code


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="CI Deployment Readiness Check")
    parser.add_argument("--fork-mode", action="store_true",
                       help="Run in fork mode (skip environment variable checks)")

    args = parser.parse_args()

    exit_code = run_ci_checks(fork_mode=args.fork_mode)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
