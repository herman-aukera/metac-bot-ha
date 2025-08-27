#!/usr/bin/env python3
"""
Demo script showing how to use the enhanced token counting and cost tracking system.
This demonstrates the integration of TokenTracker, BudgetManager, and CostMonitor.
"""
import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from infrastructure.config.token_tracker import TokenTracker
from infrastructure.config.budget_manager import BudgetManager
from infrastructure.config.cost_monitor import CostMonitor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_basic_token_tracking():
    """Demonstrate basic token tracking functionality."""
    print("\n=== Basic Token Tracking Demo ===")

    tracker = TokenTracker()

    # Example prompts and responses
    research_prompt = """
    Please research the following question: Will AI achieve AGI by 2030?

    Look for:
    1. Recent developments in AI capabilities
    2. Expert opinions and predictions
    3. Technical milestones and benchmarks
    4. Funding and investment trends

    Provide a concise summary with sources.
    """

    research_response = """
    Based on recent research and expert opinions:

    Recent Developments:
    - Large language models have shown emergent capabilities
    - Multimodal AI systems are advancing rapidly
    - Robotics integration is improving

    Expert Opinions:
    - Geoffrey Hinton: 50% chance by 2030
    - Yann LeCun: More skeptical, estimates 2040+
    - Demis Hassabis: Possible but uncertain timeline

    Key challenges remain in reasoning, planning, and generalization.
    """

    # Track the API call
    result = tracker.track_actual_usage(
        prompt=research_prompt,
        response=research_response,
        model="gpt-4o-mini",
        question_id="agi-2030-research",
        task_type="research"
    )

    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Estimated cost: ${result['estimated_cost']:.4f}")

    # Show usage summary
    tracker.log_usage_summary()


def demo_cost_monitoring():
    """Demonstrate comprehensive cost monitoring."""
    print("\n=== Cost Monitoring Demo ===")

    # Create components with small budget for demo
    budget_manager = BudgetManager(budget_limit=1.0)  # $1 budget for demo
    token_tracker = TokenTracker()
    cost_monitor = CostMonitor(token_tracker, budget_manager)

    # Simulate several API calls
    test_scenarios = [
        {
            "question_id": "climate-2030",
            "model": "gpt-4o-mini",
            "task_type": "research",
            "prompt": "Research climate change impacts by 2030" * 10,
            "response": "Climate research shows significant impacts expected" * 15
        },
        {
            "question_id": "climate-2030",
            "model": "gpt-4o",
            "task_type": "forecast",
            "prompt": "Based on research, forecast probability of 2°C warming by 2030" * 5,
            "response": "Considering current trends and policies, I estimate 25% probability" * 8
        },
        {
            "question_id": "tech-adoption",
            "model": "gpt-4o-mini",
            "task_type": "research",
            "prompt": "Research EV adoption rates globally" * 8,
            "response": "EV adoption is accelerating with government incentives" * 12
        },
        {
            "question_id": "tech-adoption",
            "model": "gpt-4o",
            "task_type": "forecast",
            "prompt": "Forecast EV market share by 2030" * 6,
            "response": "Based on current trends, I estimate 40% market share by 2030" * 10
        }
    ]

    print("Simulating API calls...")
    for scenario in test_scenarios:
        result = cost_monitor.track_api_call_with_monitoring(
            question_id=scenario["question_id"],
            model=scenario["model"],
            task_type=scenario["task_type"],
            prompt=scenario["prompt"],
            response=scenario["response"],
            success=True
        )

        print(f"  {scenario['task_type']} call: {result['total_tokens']} tokens, "
              f"${result['estimated_cost']:.4f}")

    # Show comprehensive status
    cost_monitor.log_comprehensive_status()

    # Show optimization recommendations
    recommendations = cost_monitor.get_optimization_recommendations()
    if recommendations:
        print("\n--- Optimization Recommendations ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("\nNo optimization recommendations at current usage level.")


def demo_budget_alerts():
    """Demonstrate budget alert system."""
    print("\n=== Budget Alerts Demo ===")

    # Create system with very small budget to trigger alerts
    budget_manager = BudgetManager(budget_limit=0.10)  # 10 cents for demo
    token_tracker = TokenTracker()
    cost_monitor = CostMonitor(token_tracker, budget_manager)

    # Simulate expensive calls to trigger alerts
    expensive_prompt = "Very detailed analysis required " * 50
    expensive_response = "Comprehensive analysis with detailed reasoning " * 50

    print("Simulating expensive API calls to trigger budget alerts...")

    for i in range(5):
        cost_monitor.track_api_call_with_monitoring(
            question_id=f"expensive-analysis-{i}",
            model="gpt-4o",  # More expensive model
            task_type="forecast",
            prompt=expensive_prompt,
            response=expensive_response,
            success=True
        )

        # Check for new alerts
        recent_alerts = [a for a in cost_monitor.alerts
                        if cost_monitor._is_alert_recent(a, hours=1)]

        if recent_alerts:
            latest_alert = recent_alerts[-1]
            print(f"  ALERT: {latest_alert.severity.upper()} - {latest_alert.message}")
            print(f"    Recommendation: {latest_alert.recommendation}")

    # Final status
    status = cost_monitor.get_comprehensive_status()
    budget = status["budget"]
    print(f"\nFinal Budget Status:")
    print(f"  Spent: ${budget['spent']:.4f} / ${budget['total']:.2f}")
    print(f"  Utilization: {budget['utilization_percent']:.1f}%")
    print(f"  Status: {budget['status_level'].upper()}")


def demo_model_comparison():
    """Demonstrate cost comparison between different models."""
    print("\n=== Model Cost Comparison Demo ===")

    tracker = TokenTracker()

    # Same prompt/response for different models
    prompt = "Analyze the probability of this forecasting question" * 20
    response = "Based on analysis, the probability is estimated at" * 15

    models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "claude-3-haiku"]

    print("Cost comparison for same content across models:")
    print(f"{'Model':<20} {'Tokens':<10} {'Cost':<10} {'Cost/Token':<12}")
    print("-" * 55)

    for model in models:
        result = tracker.track_actual_usage(
            prompt=prompt,
            response=response,
            model=model,
            question_id=f"comparison-{model}",
            task_type="forecast"
        )

        cost_per_token = result['estimated_cost'] / result['total_tokens']

        print(f"{model:<20} {result['total_tokens']:<10} "
              f"${result['estimated_cost']:<9.4f} ${cost_per_token:<11.6f}")

    # Show efficiency metrics
    print("\n--- Efficiency Analysis ---")
    metrics = tracker.get_cost_efficiency_metrics()

    if "model_efficiency" in metrics:
        for model, efficiency in metrics["model_efficiency"].items():
            print(f"{model}: {efficiency['tokens_per_call']:.0f} tokens/call, "
                  f"${efficiency['cost_per_token']:.6f}/token")


def main():
    """Run all demos."""
    print("Token Counting and Cost Tracking System Demo")
    print("=" * 50)

    try:
        demo_basic_token_tracking()
        demo_cost_monitoring()
        demo_budget_alerts()
        demo_model_comparison()

        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Real-time token counting and cost calculation")
        print("✓ Comprehensive usage tracking and monitoring")
        print("✓ Budget threshold alerts and recommendations")
        print("✓ Model cost comparison and efficiency analysis")
        print("✓ Integration between TokenTracker, BudgetManager, and CostMonitor")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
