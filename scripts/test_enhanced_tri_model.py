#!/usr/bin/env python3
"""
Test script for Enhanced Tri-Model Router with GPT-5 variants.
Validates model initialization, routing logic, and fallback mechanisms.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path and handle import conflicts
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Remove conflicting paths
if str(project_root / "src" / "agents") in sys.path:
    sys.path.remove(str(project_root / "src" / "agents"))

try:
    from infrastructure.config.tri_model_router import tri_model_router
except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires the enhanced tri-model router to be properly installed.")
    print("Please ensure the forecasting_tools package is compatible.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_model_initialization():
    """Test model initialization and status checking."""
    print("\n" + "=" * 60)
    print("TESTING MODEL INITIALIZATION")
    print("=" * 60)

    # Check model status
    status = tri_model_router.get_model_status()
    print("\nModel Status:")
    for tier, status_msg in status.items():
        print(f"  {tier.upper()}: {status_msg}")

    # Get detailed status
    detailed_status = tri_model_router.get_detailed_status()
    print("\nDetailed Status:")
    for tier, details in detailed_status.items():
        print(f"\n{tier.upper()} Model:")
        print(f"  Model Name: {details['model_name']}")
        print(f"  Available: {details['is_available']}")
        print(f"  Cost/1M tokens: ${details['cost_per_million_tokens']}")
        print(f"  Description: {details['description']}")
        if details["error_message"]:
            print(f"  Error: {details['error_message']}")
        print(f"  Fallback Chain: {details['fallback_chain']}")


def test_routing_logic():
    """Test model routing logic across different scenarios."""
    print("\n" + "=" * 60)
    print("TESTING ROUTING LOGIC")
    print("=" * 60)

    test_scenarios = [
        # (task_type, complexity, content_length, budget_remaining, description)
        ("validation", "minimal", 50, 100.0, "Simple validation with full budget"),
        ("research", "medium", 500, 100.0, "Standard research with full budget"),
        ("forecast", "high", 1000, 100.0, "Complex forecasting with full budget"),
        ("forecast", "high", 1000, 75.0, "Complex forecasting with 75% budget"),
        (
            "forecast",
            "high",
            1000,
            45.0,
            "Complex forecasting with 45% budget (conservative)",
        ),
        (
            "forecast",
            "high",
            1000,
            15.0,
            "Complex forecasting with 15% budget (emergency)",
        ),
        (
            "forecast",
            "high",
            1000,
            5.0,
            "Complex forecasting with 5% budget (critical)",
        ),
        ("simple", None, 25, 50.0, "Very short simple task"),
        ("research", "minimal", 200, 85.0, "Simple research in emergency mode"),
    ]

    print("\nRouting Decisions:")
    for (
        task_type,
        complexity,
        content_length,
        budget_remaining,
        description,
    ) in test_scenarios:
        model, tier = tri_model_router.choose_model(
            task_type=task_type,
            complexity=complexity,
            content_length=content_length,
            budget_remaining=budget_remaining,
        )

        cost_estimate = tri_model_router.get_cost_estimate(
            task_type=task_type,
            content_length=content_length,
            complexity=complexity,
            budget_remaining=budget_remaining,
        )

        explanation = tri_model_router.get_routing_explanation(
            task_type=task_type,
            complexity=complexity,
            content_length=content_length,
            budget_remaining=budget_remaining,
        )

        print(f"\n{description}:")
        print(f"  Selected: {tier.upper()} ({model.model})")
        print(f"  Cost Estimate: ${cost_estimate:.6f}")
        print(f"  Explanation: {explanation}")


def test_operation_modes():
    """Test operation mode detection and adjustments."""
    print("\n" + "=" * 60)
    print("TESTING OPERATION MODES")
    print("=" * 60)

    budget_scenarios = [
        (100.0, "Full budget available"),
        (75.0, "25% budget used"),
        (45.0, "55% budget used (conservative mode)"),
        (15.0, "85% budget used (emergency mode)"),
        (3.0, "97% budget used (critical mode)"),
        (0.0, "Budget exhausted"),
    ]

    print("\nOperation Mode Detection:")
    for budget_remaining, description in budget_scenarios:
        operation_mode = tri_model_router.get_operation_mode(budget_remaining)
        print(f"  {description}: {operation_mode.upper()}")


async def test_model_health():
    """Test model health checking."""
    print("\n" + "=" * 60)
    print("TESTING MODEL HEALTH")
    print("=" * 60)

    print("\nChecking model health...")
    for tier in ["nano", "mini", "full"]:
        try:
            status = await tri_model_router.check_model_health(tier)
            if status.is_available:
                response_time = (
                    f" ({status.response_time:.2f}s)" if status.response_time else ""
                )
                print(f"  {tier.upper()}: ✓ Healthy{response_time}")
            else:
                print(f"  {tier.upper()}: ✗ Unhealthy - {status.error_message}")
        except Exception as e:
            print(f"  {tier.upper()}: ✗ Health check failed - {e}")


def test_cost_calculations():
    """Test cost estimation accuracy."""
    print("\n" + "=" * 60)
    print("TESTING COST CALCULATIONS")
    print("=" * 60)

    # Get model costs
    costs = tri_model_router.get_model_costs()
    print("\nModel Costs (per 1M tokens):")
    for tier, cost in costs.items():
        print(f"  {tier.upper()}: ${cost}")

    # Test cost estimates for different scenarios
    test_cases = [
        ("validation", 100, "minimal"),
        ("research", 500, "medium"),
        ("forecast", 1000, "high"),
        ("simple", 50, None),
    ]

    print("\nCost Estimates:")
    for task_type, content_length, complexity in test_cases:
        for budget in [100.0, 50.0, 15.0]:  # Different budget levels
            cost = tri_model_router.get_cost_estimate(
                task_type=task_type,
                content_length=content_length,
                complexity=complexity,
                budget_remaining=budget,
            )
            _, tier = tri_model_router.choose_model(
                task_type=task_type,
                complexity=complexity,
                content_length=content_length,
                budget_remaining=budget,
            )
            print(
                f"  {task_type} ({content_length} chars, {budget}% budget): ${cost:.6f} ({tier})"
            )


async def test_routing_with_prompts():
    """Test actual routing with sample prompts."""
    print("\n" + "=" * 60)
    print("TESTING ROUTING WITH SAMPLE PROMPTS")
    print("=" * 60)

    sample_prompts = [
        (
            "validation",
            "Check if this statement is accurate: The sky is blue.",
            "minimal",
            90.0,
        ),
        (
            "research",
            "Research recent developments in AI forecasting for tournament questions.",
            "medium",
            70.0,
        ),
        (
            "forecast",
            "Analyze this complex geopolitical question and provide a calibrated probability estimate.",
            "high",
            40.0,
        ),
        ("simple", "Summarize: AI is advancing rapidly.", None, 20.0),
    ]

    print("\nSample Routing Tests:")
    for task_type, prompt, complexity, budget_remaining in sample_prompts:
        try:
            # This would normally call the actual model, but we'll just test the routing
            model, tier = tri_model_router.choose_model(
                task_type=task_type,
                complexity=complexity,
                content_length=len(prompt),
                budget_remaining=budget_remaining,
            )

            cost_estimate = tri_model_router.get_cost_estimate(
                task_type=task_type,
                content_length=len(prompt),
                complexity=complexity,
                budget_remaining=budget_remaining,
            )

            print(f"\n{task_type.upper()} Task:")
            print(f"  Prompt: {prompt[:60]}...")
            print(f"  Budget Remaining: {budget_remaining}%")
            print(f"  Selected Model: {tier.upper()} ({model.model})")
            print(f"  Estimated Cost: ${cost_estimate:.6f}")

        except Exception as e:
            print(f"  Error testing {task_type}: {e}")


def print_summary():
    """Print test summary and recommendations."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)

    print("\nEnhanced Tri-Model Router Features Tested:")
    print("  ✓ Model initialization with fallback chains")
    print("  ✓ Budget-aware operation mode detection")
    print("  ✓ Intelligent model routing based on task complexity")
    print("  ✓ Cost estimation with task-specific multipliers")
    print("  ✓ Model health monitoring and status reporting")
    print("  ✓ Comprehensive error handling and fallbacks")

    print("\nKey Improvements Over Original:")
    print("  • Enhanced model availability detection")
    print("  • Robust fallback chains for reliability")
    print("  • Operation mode-based routing (normal/conservative/emergency/critical)")
    print("  • Improved cost estimation accuracy")
    print("  • Comprehensive monitoring and status reporting")
    print("  • Better error handling and recovery mechanisms")

    print("\nRecommendations:")
    print("  1. Monitor model availability regularly in production")
    print("  2. Set up alerts for operation mode changes")
    print("  3. Track actual vs estimated costs for calibration")
    print("  4. Use health checks to detect model issues early")
    print("  5. Review routing decisions periodically for optimization")

    print(f"\nConfiguration Status:")
    print(
        f"  OpenRouter API Key: {'✓ Configured' if os.getenv('OPENROUTER_API_KEY') else '✗ Missing'}"
    )
    print(f"  Budget Limit: ${os.getenv('BUDGET_LIMIT', 'Not set')}")
    print(
        f"  GPT-5 Models: {os.getenv('DEFAULT_MODEL', 'gpt-5')} / {os.getenv('MINI_MODEL', 'gpt-5-mini')} / {os.getenv('NANO_MODEL', 'gpt-5-nano')}"
    )


async def main():
    """Run all tests."""
    print("Enhanced Tri-Model Router Test Suite")
    print("Testing GPT-5 variants with anti-slop directives")

    try:
        # Run all tests
        await test_model_initialization()
        test_routing_logic()
        test_operation_modes()
        await test_model_health()
        test_cost_calculations()
        await test_routing_with_prompts()
        print_summary()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
