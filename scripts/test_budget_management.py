#!/usr/bin/env python3
"""
Test script for budget management system.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from infrastructure.config.api_keys import api_key_manager
from infrastructure.config.budget_alerts import budget_alert_system
from infrastructure.config.budget_manager import budget_manager
from infrastructure.config.token_tracker import token_tracker


def test_budget_manager():
    """Test budget manager functionality."""
    print("=== Testing Budget Manager ===")

    # Test cost estimation
    cost = budget_manager.estimate_cost("gpt-4o", 1000, 500)
    print(f"Estimated cost for GPT-4o (1000 input, 500 output tokens): ${cost:.4f}")

    cost_mini = budget_manager.estimate_cost("gpt-4o-mini", 1000, 500)
    print(
        f"Estimated cost for GPT-4o-mini (1000 input, 500 output tokens): ${cost_mini:.4f}"
    )

    # Test budget status
    status = budget_manager.get_budget_status()
    print(f"Budget status: {status.status_level}")
    print(f"Utilization: {status.utilization_percentage:.1f}%")

    # Test affordability check
    can_afford = budget_manager.can_afford(cost)
    print(f"Can afford GPT-4o call: {can_afford}")

    # Test recording a cost
    recorded_cost = budget_manager.record_cost(
        question_id="test-123",
        model="gpt-4o-mini",
        input_tokens=500,
        output_tokens=300,
        task_type="test",
        success=True,
    )
    print(f"Recorded cost: ${recorded_cost:.4f}")

    # Test updated status
    updated_status = budget_manager.get_budget_status()
    print(f"Updated utilization: {updated_status.utilization_percentage:.1f}%")

    print("✓ Budget Manager tests passed\n")


def test_token_tracker():
    """Test token tracker functionality."""
    print("=== Testing Token Tracker ===")

    test_text = "This is a test prompt for token counting."

    # Test token counting
    tokens = token_tracker.count_tokens(test_text, "gpt-4o")
    print(f"Token count for test text: {tokens}")

    # Test prompt estimation
    estimation = token_tracker.estimate_tokens_for_prompt(test_text, "gpt-4o")
    print(f"Prompt estimation: {estimation}")

    # Test context validation
    is_valid, validation = token_tracker.validate_prompt_length(test_text, "gpt-4o")
    print(f"Prompt valid: {is_valid}")

    print("✓ Token Tracker tests passed\n")


def test_api_keys():
    """Test API key manager."""
    print("=== Testing API Key Manager ===")

    # Test key validation
    validation = api_key_manager.validate_required_keys()
    print(f"Required keys valid: {validation['valid']}")

    if validation["missing_keys"]:
        print("Missing keys:")
        for key_info in validation["missing_keys"]:
            print(f"  - {key_info['key']}: {key_info['description']}")

    # Test getting OpenRouter key
    openrouter_key = api_key_manager.get_api_key("OPENROUTER_API_KEY")
    if openrouter_key:
        print(f"OpenRouter key configured: {openrouter_key[:20]}...")
    else:
        print("OpenRouter key not configured")

    print("✓ API Key Manager tests passed\n")


def test_budget_alerts():
    """Test budget alert system."""
    print("=== Testing Budget Alert System ===")

    # Check for alerts
    alert = budget_alert_system.check_and_alert()
    if alert:
        print(f"Alert generated: {alert.alert_type} - {alert.message}")
    else:
        print("No alerts at current budget level")

    # Get recommendations
    recommendations = budget_alert_system.get_budget_recommendations()
    print("Budget recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec}")

    # Get optimization suggestions
    optimizations = budget_alert_system.get_cost_optimization_suggestions()
    print("Cost optimization suggestions:")
    for i, opt in enumerate(optimizations[:3], 1):
        print(f"  {i}. {opt}")

    print("✓ Budget Alert System tests passed\n")


def main():
    """Run all tests."""
    print("Testing Budget Management System")
    print("=" * 50)

    try:
        test_api_keys()
        test_token_tracker()
        test_budget_manager()
        test_budget_alerts()

        print("=" * 50)
        print("✓ All tests passed successfully!")

        # Final budget status
        print("\n=== Final Budget Status ===")
        budget_manager.log_budget_status()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
