#!/usr/bin/env python3
"""
Test script for budget management integration with main application.
"""
import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set test environment variables if not set
if not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = "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"

if not os.getenv("METACULUS_TOKEN"):
    os.environ["METACULUS_TOKEN"] = "test_token_123"

if not os.getenv("BUDGET_LIMIT"):
    os.environ["BUDGET_LIMIT"] = "100.0"

def test_enhanced_llm_config():
    """Test enhanced LLM configuration."""
    print("=== Testing Enhanced LLM Configuration ===")

    try:
        from infrastructure.config.enhanced_llm_config import enhanced_llm_config

        # Test model selection for different tasks
        research_llm = enhanced_llm_config.get_llm_for_task("research", "simple")
        print(f"Research LLM model: {research_llm.model}")

        forecast_llm = enhanced_llm_config.get_llm_for_task("forecast", "complex")
        print(f"Forecast LLM model: {forecast_llm.model}")

        # Test cost estimation
        test_prompt = "This is a test prompt for cost estimation."
        cost, details = enhanced_llm_config.estimate_task_cost(test_prompt, "research")
        print(f"Estimated cost for research task: ${cost:.4f}")
        print(f"Details: {details}")

        # Test affordability check
        can_afford, afford_details = enhanced_llm_config.can_afford_task(test_prompt, "forecast")
        print(f"Can afford forecast task: {can_afford}")

        # Test question complexity assessment
        simple_question = "Will it rain tomorrow?"
        complex_question = "What will be the geopolitical implications of the upcoming international trade negotiations on global economic stability?"

        simple_complexity = enhanced_llm_config.assess_question_complexity(simple_question)
        complex_complexity = enhanced_llm_config.assess_question_complexity(complex_question)

        print(f"Simple question complexity: {simple_complexity}")
        print(f"Complex question complexity: {complex_complexity}")

        # Log configuration status
        enhanced_llm_config.log_configuration_status()

        print("✓ Enhanced LLM Configuration tests passed\n")

    except Exception as e:
        print(f"❌ Enhanced LLM Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_budget_workflow():
    """Test complete budget workflow."""
    print("=== Testing Budget Workflow ===")

    try:
        from infrastructure.config.budget_manager import budget_manager
        from infrastructure.config.budget_alerts import budget_alert_system
        from infrastructure.config.enhanced_llm_config import enhanced_llm_config

        # Simulate a research task
        research_prompt = """
        You are a research assistant. Please provide information about:
        Will the Federal Reserve raise interest rates in the next quarter?

        Please provide recent news and expert opinions.
        """

        # Check if we can afford the task
        can_afford, details = enhanced_llm_config.can_afford_task(research_prompt, "research")
        print(f"Can afford research task: {can_afford}")
        print(f"Estimated cost: ${details['estimated_cost']:.4f}")

        if can_afford:
            # Simulate task completion
            mock_response = "Based on recent Federal Reserve statements and economic indicators..."

            # Record the task completion
            actual_cost = enhanced_llm_config.record_task_completion(
                question_id="test-workflow-123",
                prompt=research_prompt,
                response=mock_response,
                task_type="research",
                model_used=details["model"],
                success=True
            )

            print(f"Recorded actual cost: ${actual_cost:.4f}")

        # Check for budget alerts
        alert = budget_alert_system.check_and_alert()
        if alert:
            print(f"Budget alert: {alert.alert_type} - {alert.message}")

        # Get current budget status
        status = budget_manager.get_budget_status()
        print(f"Current budget utilization: {status.utilization_percentage:.2f}%")
        print(f"Questions processed: {status.questions_processed}")

        print("✓ Budget Workflow tests passed\n")

    except Exception as e:
        print(f"❌ Budget Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_emergency_mode_simulation():
    """Test emergency mode behavior."""
    print("=== Testing Emergency Mode Simulation ===")

    try:
        from infrastructure.config.budget_manager import budget_manager
        from infrastructure.config.enhanced_llm_config import enhanced_llm_config

        # Save original budget
        original_budget = budget_manager.budget_limit
        original_spend = budget_manager.current_spend

        # Simulate high budget usage (96% used)
        budget_manager.current_spend = budget_manager.budget_limit * 0.96

        print(f"Simulated budget usage: {(budget_manager.current_spend / budget_manager.budget_limit) * 100:.1f}%")

        # Test model selection in emergency mode
        status = budget_manager.get_budget_status()
        print(f"Budget status level: {status.status_level}")

        # Get LLM for forecast task - should use cheapest model
        emergency_llm = enhanced_llm_config.get_llm_for_task("forecast", "complex")
        print(f"Emergency mode forecast model: {emergency_llm.model}")

        # Test alert generation
        from infrastructure.config.budget_alerts import budget_alert_system
        alert = budget_alert_system.check_and_alert()
        if alert:
            print(f"Emergency alert: {alert.alert_type} - {alert.message}")

        # Get recommendations
        recommendations = budget_alert_system.get_budget_recommendations()
        print("Emergency recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec}")

        # Restore original budget state
        budget_manager.current_spend = original_spend

        print("✓ Emergency Mode Simulation tests passed\n")

    except Exception as e:
        print(f"❌ Emergency Mode Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def main():
    """Run all integration tests."""
    print("Testing Budget Management Integration")
    print("=" * 60)

    success = True

    # Run tests
    success &= test_enhanced_llm_config()
    success &= test_budget_workflow()
    success &= test_emergency_mode_simulation()

    print("=" * 60)
    if success:
        print("✓ All integration tests passed successfully!")

        # Final status report
        try:
            from infrastructure.config.budget_manager import budget_manager
            from infrastructure.config.budget_alerts import budget_alert_system

            print("\n=== Final Integration Status ===")
            budget_manager.log_budget_status()

            # Generate comprehensive report
            report = budget_alert_system.generate_budget_report()
            print(f"Total cost records: {len(budget_manager.cost_records)}")
            print(f"Alert history: {len(budget_alert_system.alert_history)} alerts")

        except Exception as e:
            print(f"Warning: Could not generate final report: {e}")

        return 0
    else:
        print("❌ Some integration tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
