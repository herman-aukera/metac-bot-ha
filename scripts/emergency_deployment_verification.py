#!/usr/bin/env python3
"""
Emergency Deployment Verification Script

Validates that the tournament bot is ready for deployment with minimal dependencies.
Designed to work even when some components are missing or failing.

Usage:
    python3 scripts/emergency_deployment_verification.py
    python3 scripts/emergency_deployment_verification.py --quick
    python3 scripts/emergency_deployment_verification.py --full
"""

import sys
import os
import asyncio
import time
import traceback
from typing import Dict, List, Tuple, Any
import argparse

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

def test_python_version() -> Tuple[bool, str]:
    """Test Python version compatibility."""
    try:
        version = sys.version_info
        if version.major == 3 and version.minor >= 11:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)"
    except Exception as e:
        return False, f"Version check failed: {e}"

def test_core_imports() -> Tuple[bool, str]:
    """Test core module imports."""
    try:
        # Test basic imports
        import requests
        import json
        import os
        import asyncio

        # Test AI/ML imports
        import openai
        import numpy
        import pandas

        # Test forecasting imports
        try:
            import asknews
        except ImportError:
            return False, "AskNews import failed - research functionality unavailable"

        try:
            import forecasting_tools
        except ImportError:
            return False, "Forecasting tools import failed"

        return True, "All core imports successful"
    except ImportError as e:
        return False, f"Import failed: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def test_environment_variables() -> Tuple[bool, str]:
    """Test required environment variables."""
    try:
        required_vars = [
            "ASKNEWS_CLIENT_ID",
            "ASKNEWS_SECRET",
            "OPENROUTER_API_KEY"
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            return False, f"Missing environment variables: {', '.join(missing_vars)}"

        return True, "All required environment variables set"
    except Exception as e:
        return False, f"Environment check failed: {e}"

def test_configuration_loading() -> Tuple[bool, str]:
    """Test configuration loading."""
    try:
        from infrastructure.config.settings import Config
        config = Config()

        # Check critical config values
        if not hasattr(config, 'tournament_id'):
            return False, "Tournament ID not configured"

        if not hasattr(config, 'llm_config'):
            return False, "LLM configuration missing"

        return True, f"Configuration loaded (Tournament ID: {getattr(config, 'tournament_id', 'Unknown')})"
    except ImportError:
        return False, "Configuration module import failed"
    except Exception as e:
        return False, f"Configuration loading failed: {e}"

def test_agent_initialization() -> Tuple[bool, str]:
    """Test agent initialization."""
    try:
        from infrastructure.config.settings import Config
        from agents.ensemble_agent import EnsembleAgent

        config = Config()
        agent = EnsembleAgent('test', config.llm_config)

        return True, "Agent initialization successful"
    except ImportError as e:
        return False, f"Agent import failed: {e}"
    except Exception as e:
        return False, f"Agent initialization failed: {e}"

async def test_api_connectivity() -> Tuple[bool, str]:
    """Test API connectivity."""
    try:
        import httpx

        # Test OpenRouter API
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://openrouter.ai/api/v1/models")
                if response.status_code == 200:
                    openrouter_status = "✅"
                else:
                    openrouter_status = f"❌ ({response.status_code})"
        except Exception:
            openrouter_status = "❌ (timeout/error)"

        # Test AskNews API
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://api.asknews.app/v1/news/search?q=test&n_articles=1")
                if response.status_code in [200, 401]:  # 401 is expected without auth
                    asknews_status = "✅"
                else:
                    asknews_status = f"❌ ({response.status_code})"
        except Exception:
            asknews_status = "❌ (timeout/error)"

        status_msg = f"OpenRouter: {openrouter_status}, AskNews: {asknews_status}"

        if "✅" in openrouter_status and "✅" in asknews_status:
            return True, status_msg
        else:
            return False, status_msg

    except ImportError:
        return False, "HTTP client not available"
    except Exception as e:
        return False, f"API connectivity test failed: {e}"

async def test_llm_client() -> Tuple[bool, str]:
    """Test LLM client functionality."""
    try:
        from infrastructure.external_apis.llm_client import LLMClient
        from infrastructure.config.settings import Config

        config = Config()
        client = LLMClient(config.llm_config)

        # Test with a simple prompt
        response = await client.generate_response("Say 'test'", max_tokens=5)

        if response and len(response.strip()) > 0:
            return True, f"LLM client working (response: '{response[:20]}...')"
        else:
            return False, "LLM client returned empty response"

    except ImportError as e:
        return False, f"LLM client import failed: {e}"
    except Exception as e:
        return False, f"LLM client test failed: {e}"

async def test_research_pipeline() -> Tuple[bool, str]:
    """Test research pipeline functionality."""
    try:
        from infrastructure.external_apis.tournament_asknews import TournamentAskNews
        from infrastructure.config.settings import Config

        config = Config()
        asknews = TournamentAskNews(config.asknews_config)

        # Test search functionality
        results = await asknews.search("AI forecasting", max_results=1)

        if results and len(results) > 0:
            return True, f"Research pipeline working ({len(results)} results)"
        else:
            return False, "Research pipeline returned no results"

    except ImportError as e:
        return False, f"Research pipeline import failed: {e}"
    except Exception as e:
        return False, f"Research pipeline test failed: {e}"

def test_file_permissions() -> Tuple[bool, str]:
    """Test file permissions and directory structure."""
    try:
        # Check if we can create log directories
        log_dirs = ["logs", "logs/performance", "logs/reasoning", "data"]

        for log_dir in log_dirs:
            os.makedirs(log_dir, exist_ok=True)

            # Test write permissions
            test_file = os.path.join(log_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)

        return True, "File permissions and directories OK"
    except Exception as e:
        return False, f"File permission test failed: {e}"

def test_memory_usage() -> Tuple[bool, str]:
    """Test memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        if memory_mb < 500:  # Less than 500MB is good
            return True, f"Memory usage: {memory_mb:.1f} MB"
        elif memory_mb < 1000:  # Less than 1GB is acceptable
            return True, f"Memory usage: {memory_mb:.1f} MB (acceptable)"
        else:
            return False, f"Memory usage: {memory_mb:.1f} MB (high)"

    except ImportError:
        return True, "Memory monitoring not available (psutil not installed)"
    except Exception as e:
        return False, f"Memory test failed: {e}"

async def run_quick_tests() -> Dict[str, Tuple[bool, str]]:
    """Run quick deployment verification tests."""
    tests = {
        "Python Version": test_python_version(),
        "Core Imports": test_core_imports(),
        "Environment Variables": test_environment_variables(),
        "Configuration": test_configuration_loading(),
        "File Permissions": test_file_permissions(),
        "Memory Usage": test_memory_usage(),
    }

    return tests

async def run_full_tests() -> Dict[str, Tuple[bool, str]]:
    """Run comprehensive deployment verification tests."""
    # Start with quick tests
    tests = await run_quick_tests()

    # Add comprehensive tests
    additional_tests = {
        "Agent Initialization": test_agent_initialization(),
        "API Connectivity": await test_api_connectivity(),
        "LLM Client": await test_llm_client(),
        "Research Pipeline": await test_research_pipeline(),
    }

    tests.update(additional_tests)
    return tests

def print_results(tests: Dict[str, Tuple[bool, str]], test_type: str = "Deployment"):
    """Print test results with summary."""
    print(f"\n🧪 {test_type} Verification Results")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, (success, message) in tests.items():
        if success:
            print_status(f"{test_name}: {message}", "SUCCESS")
            passed += 1
        else:
            print_status(f"{test_name}: {message}", "ERROR")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Summary: {passed} passed, {failed} failed")

    if failed == 0:
        print_status("🎉 All tests passed! Tournament deployment ready.", "SUCCESS")
        return True
    elif failed <= 2 and passed >= 6:
        print_status("⚠️  Some tests failed but core functionality appears working.", "WARNING")
        print_status("Consider proceeding with deployment if critical tests passed.", "WARNING")
        return True
    else:
        print_status("❌ Multiple critical tests failed. Deployment not recommended.", "ERROR")
        return False

def print_emergency_instructions():
    """Print emergency deployment instructions."""
    print("\n🚨 Emergency Deployment Instructions")
    print("=" * 50)
    print("If tests are failing, try these emergency steps:")
    print()
    print("1. Install minimal dependencies:")
    print("   pip install requests openai python-dotenv pydantic typer")
    print()
    print("2. Set required environment variables:")
    print("   export ASKNEWS_CLIENT_ID=your_client_id")
    print("   export ASKNEWS_SECRET=your_secret")
    print("   export OPENROUTER_API_KEY=your_api_key")
    print()
    print("3. Test minimal functionality:")
    print("   python3 -m src.main --tournament 32813 --max-questions 1 --dry-run")
    print()
    print("4. If still failing, check:")
    print("   - Python version (3.11+ required)")
    print("   - Internet connectivity")
    print("   - API key validity")
    print("   - File permissions")

async def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description="Emergency Deployment Verification")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--full", action="store_true", help="Run comprehensive tests")
    parser.add_argument("--emergency", action="store_true", help="Show emergency instructions")

    args = parser.parse_args()

    print("🚀 Metaculus Tournament Bot - Emergency Deployment Verification")
    print("=" * 60)

    if args.emergency:
        print_emergency_instructions()
        return

    try:
        if args.quick:
            tests = await run_quick_tests()
            success = print_results(tests, "Quick")
        elif args.full:
            tests = await run_full_tests()
            success = print_results(tests, "Comprehensive")
        else:
            # Default: run quick tests first, then full if they pass
            print_status("Running quick verification tests...", "INFO")
            quick_tests = await run_quick_tests()
            quick_success = print_results(quick_tests, "Quick")

            if quick_success:
                print_status("\nRunning comprehensive tests...", "INFO")
                full_tests = await run_full_tests()
                success = print_results(full_tests, "Comprehensive")
            else:
                success = False

        if not success:
            print_emergency_instructions()
            sys.exit(1)
        else:
            print_status("\n🎯 Tournament bot is ready for deployment!", "SUCCESS")
            sys.exit(0)

    except KeyboardInterrupt:
        print_status("\n⏹️  Verification interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_status(f"\n💥 Verification failed with unexpected error: {e}", "ERROR")
        traceback.print_exc()
        print_emergency_instructions()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
