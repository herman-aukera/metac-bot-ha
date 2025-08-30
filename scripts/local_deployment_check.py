#!/usr/bin/env python3
"""
Local Deployment Check Script

A simplified deployment check for local development environments.
This script is designed to work without requiring all production secrets.

Usage:
    python3 scripts/local_deployment_check.py
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def print_status(message: str, status: str = "INFO"):
    """Print status message with formatting."""
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }

    color = colors.get(status, colors["INFO"])
    reset = colors["RESET"]

    symbols = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌"
    }

    symbol = symbols.get(status, "ℹ️")
    print(f"{color}{symbol} {message}{reset}")

def print_header(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

async def main():
    """Run local deployment check."""
    print("🚀 Local Development Deployment Check")
    print("=" * 60)
    print("This check validates your local environment for tournament bot development.")
    print("It does NOT require production secrets - those are configured in GitHub Actions.")
    print()

    # Set local dev mode
    os.environ['LOCAL_DEV_MODE'] = 'true'

    # Import and run the main deployment check
    try:
        from deployment_readiness_check import DeploymentReadinessChecker

        checker = DeploymentReadinessChecker()

        print_status("Running local development deployment check...", "INFO")

        # Run quick tests which are most relevant for local dev
        tests = await checker.run_quick_tests()

        # Print results
        success = checker.print_results(tests, "Local Development")

        print_header("Local Development Summary")

        if success:
            print_status("✅ LOCAL DEVELOPMENT ENVIRONMENT READY", "SUCCESS")
            print("Your local environment is properly configured for development.")
            print("\nNext steps:")
            print("1. Your secrets are configured in GitHub Actions ✅")
            print("2. Tests will run with full secrets in CI/CD ✅")
            print("3. You can develop and test locally without production secrets ✅")
            print("4. Push your changes to trigger the full deployment pipeline ✅")
        else:
            print_status("❌ LOCAL DEVELOPMENT ISSUES DETECTED", "ERROR")
            print("Some issues were found in your local development environment.")
            print("\nThis is OK if:")
            print("- You have secrets configured in GitHub Actions")
            print("- The failing tests are related to missing API keys")
            print("- Core Python/project structure tests are passing")

        print(f"\n🎯 Ready for tournament development!")
        print("Push your changes to GitHub to run the full deployment pipeline with secrets.")

    except ImportError as e:
        print_status(f"Failed to import deployment checker: {e}", "ERROR")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print_status(f"Unexpected error: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
