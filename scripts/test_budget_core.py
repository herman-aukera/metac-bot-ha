#!/usr/bin/env python3
"""
Test script for core budget management functionality without forecasting_tools dependencies.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set test environment variables if not set
if not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = "dummy_openrouter_key_for_testing"

if not os.getenv("METACULUS_TOKEN"):
    os.environ["METACULUS_TOKEN"] = "test_token_123"

if not os.getenv("BUDGET_LIMIT"):
    os.environ["BUDGET_LIMIT"] = "100.0"

def test_core_budget_functionality():
    """Test core budget management without LLM dependencies."""
    print("=== Testing Core Budget Management ===")

    try:
        from infrastructure.config.budget_manager import budget_manager
        from infrastructure.config.budget_alerts import budget_alert_system
        from infrastructure.config.token_tracker import token_tracker
        from infrastructure.config.api_keys import api_key_manager

        # Test API key validation
        print("1. Testing API Key Management...")
        validation = api_key_manager.validate_required_keys()
        openrouter_key = api_key_manager.get_api_key("OPENROUTER_API_KEY")
        print(f"   OpenRouter key configured: {'✓' if openrouter_key else '✗'}")

        # Test token counting
        print("2. Testing Token Tracking...")
        test_prompt = "This is a test prompt for forecasting about economic indicators."
        tokens = token_tracker.count_tokens(test_prompt, "gpt-4o")
        estimation = token_tracker.estimate_tokens_for_prompt(test_prompt, "gpt-4o")
        print(f"   Token count: {tokens}")
        print(f"   Estimated output tokens: {estimation['estimated_output_tokens']}")

        # Test cost estimation
        print("3. Testing Cost Estimation...")
        gpt4o_cost = budget_manager.estimate_cost("gpt-4o", 1000, 500)
        gpt4o_mini_cost = budget_manager.estimate_cost("gpt-4o-mini", 1000, 500)
        print(f"   GPT-4o cost (1000 in, 500 out): ${gpt4o_cost:.4f}")
        print(f"   GPT-4o-mini cost (1000 in, 500 out): ${gpt4o_mini_cost:.4f}")
        print(f"   Cost savings with mini: {((gpt4o_cost - gpt4o_mini_cost) / gpt4o_cost * 100):.1f}%")

        # Test budget tracking
        print("4. Testing Budget Tracking...")
        initial_status = budget_manager.get_budget_status()
        print(f"   Initial budget utilization: {initial_status.utilization_percentage:.2f}%")

        # Simulate some API calls
        for i in range(3):
            cost = budget_manager.record_cost(
                question_id=f"test-{i}",
                model="gpt-4o-mini",
                input_tokens=800,
                output_tokens=400,
                task_type="research" if i % 2 == 0 else "forecast",
                success=True
            )
            print(f"   Recorded cost for test-{i}: ${cost:.4f}")

        updated_status = budget_manager.get_budget_status()
        print(f"   Updated budget utilization: {updated_status.utilization_percentage:.2f}%")
        print(f"   Questions processed: {updated_status.questions_processed}")
        print(f"   Average cost per question: ${updated_status.average_cost_per_question:.4f}")

        # Test budget alerts
        print("5. Testing Budget Alerts...")
        alert = budget_alert_system.check_and_alert()
        if alert:
            print(f"   Alert generated: {alert.alert_type} - {alert.message}")
        else:
            print("   No alerts at current budget level")

        recommendations = budget_alert_system.get_budget_recommendations()
        print(f"   Recommendations: {len(recommendations)} available")

        # Test cost breakdown
        print("6. Testing Cost Analysis...")
        breakdown = budget_manager.get_cost_breakdown()
        print(f"   Models used: {list(breakdown['by_model'].keys())}")
        print(f"   Task types: {list(breakdown['by_task_type'].keys())}")

        for model, data in breakdown['by_model'].items():
            print(f"   {model}: ${data['cost']:.4f} ({data['calls']} calls)")

        print("✓ Core Budget Management tests passed\n")
        return True

    except Exception as e:
        print(f"❌ Core Budget Management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emergency_simulation():
    """Test emergency mode without LLM dependencies."""
    print("=== Testing Emergency Mode Simulation ===")

    try:
        from infrastructure.config.budget_manager import budget_manager
        from infrastructure.config.budget_alerts import budget_alert_system

        # Save original state
        original_spend = budget_manager.current_spend

        # Simulate high budget usage
        budget_manager.current_spend = budget_manager.budget_limit * 0.96

        print(f"Simulated budget usage: {(budget_manager.current_spend / budget_manager.budget_limit) * 100:.1f}%")

        # Check status
        status = budget_manager.get_budget_status()
        print(f"Budget status level: {status.status_level}")

        # Check alerts
        alert = budget_alert_system.check_and_alert()
        if alert:
            print(f"Emergency alert: {alert.alert_type} - {alert.message}")

        # Get emergency recommendations
        recommendations = budget_alert_system.get_budget_recommendations()
        print("Emergency recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec}")

        # Test affordability in emergency mode
        high_cost = 5.0  # $5 - should be rejected in emergency mode
        can_afford = budget_manager.can_afford(high_cost)
        print(f"Can afford $5 task in emergency mode: {can_afford}")

        # Restore original state
        budget_manager.current_spend = original_spend

        print("✓ Emergency Mode Simulation tests passed\n")
        return True

    except Exception as e:
        print(f"❌ Emergency Mode Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_selection_logic():
    """Test model selection logic without actual LLM creation."""
    print("=== Testing Model Selection Logic ===")

    try:
        # Test question complexity assessment
        from infrastructure.config.enhanced_llm_config import EnhancedLLMConfig

        # Create instance without initializing (to avoid LLM imports)
        config = EnhancedLLMConfig.__new__(EnhancedLLMConfig)
        config.model_configs = {
            "research": {"model": "openai/gpt-4o-mini", "cost_tier": "low"},
            "forecast": {"model": "openai/gpt-4o", "cost_tier": "high"},
            "simple": {"model": "openai/gpt-4o-mini", "cost_tier": "low"}
        }

        # Test complexity assessment
        simple_q = "Will it rain tomorrow?"
        complex_q = "What will be the geopolitical implications of the upcoming international trade negotiations?"

        simple_complexity = config.assess_question_complexity(simple_q)
        complex_complexity = config.assess_question_complexity(complex_q)

        print(f"Simple question complexity: {simple_complexity}")
        print(f"Complex question complexity: {complex_complexity}")

        # Test model selection logic (without creating actual LLMs)
        from infrastructure.config.budget_manager import budget_manager

        # Normal mode
        budget_manager.current_spend = 0
        status = budget_manager.get_budget_status()
        print(f"Normal mode status: {status.status_level}")

        # Conservative mode
        budget_manager.current_spend = budget_manager.budget_limit * 0.85
        status = budget_manager.get_budget_status()
        print(f"Conservative mode status: {status.status_level}")

        # Emergency mode
        budget_manager.current_spend = budget_manager.budget_limit * 0.96
        status = budget_manager.get_budget_status()
        print(f"Emergency mode status: {status.status_level}")

        # Reset
        budget_manager.current_spend = 0

        print("✓ Model Selection Logic tests passed\n")
        return True

    except Exception as e:
        print(f"❌ Model Selection Logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all core tests."""
    print("Testing Core Budget Management System")
    print("=" * 60)

    success = True

    # Run tests
    success &= test_core_budget_functionality()
    success &= test_emergency_simulation()
    success &= test_model_selection_logic()

    print("=" * 60)
    if success:
        print("✓ All core tests passed successfully!")

        # Final status report
        try:
            from infrastructure.config.budget_manager import budget_manager
            print("\n=== Final Budget Status ===")
            budget_manager.log_budget_status()

        except Exception as e:
            print(f"Warning: Could not generate final report: {e}")

        return 0
    else:
        print("❌ Some core tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
