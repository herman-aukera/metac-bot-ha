#!/usr/bin/env python3
"""
Demonstration of the Task Complexity Analyzer for intelligent model selection.

This script shows how the complexity analyzer assesses different types of forecasting
questions and recommends appropriate models based on complexity and budget constraints.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from infrastructure.config.task_complexity_analyzer import (
    TaskComplexityAnalyzer
)


def demo_complexity_analysis():
    """Demonstrate complexity analysis on various question types."""
    print("=" * 80)
    print("TASK COMPLEXITY ANALYZER DEMONSTRATION")
    print("=" * 80)
    print()

    analyzer = TaskComplexityAnalyzer()

    # Test questions of varying complexity
    test_questions = [
        {
            "name": "Simple Binary Question",
            "question": "Will the next iPhone be released before December 31, 2024?",
            "background": "Apple typically releases new iPhones in September each year. The iPhone 15 was released in September 2023.",
            "resolution_criteria": "This question resolves Yes if Apple officially releases a new iPhone model before December 31, 2024.",
            "fine_print": "Pre-orders and announcements do not count, only the actual release date."
        },
        {
            "name": "Medium Market Question",
            "question": "Will the S&P 500 close above 5000 by the end of 2024?",
            "background": "The S&P 500 is currently trading around 4800. Market conditions have been volatile due to inflation concerns and geopolitical tensions.",
            "resolution_criteria": "This question resolves Yes if the S&P 500 closes above 5000 on any trading day before December 31, 2024.",
            "fine_print": "Intraday highs do not count, only closing prices."
        },
        {
            "name": "Complex Geopolitical Question",
            "question": "Will there be a major international conflict involving at least three nations before 2025?",
            "background": """This question involves complex geopolitical dynamics across multiple regions.
            Current tensions exist in various areas including Eastern Europe, the South China Sea, and the Middle East.
            The outcome depends on numerous interdependent factors including diplomatic relations, economic conditions,
            military capabilities, and domestic political pressures in multiple countries. Historical precedents show
            that such conflicts often emerge from cascading events and miscalculations rather than deliberate planning.""",
            "resolution_criteria": """This question resolves Yes if there is an armed conflict involving military forces
            from at least three sovereign nations, with sustained combat operations lasting more than 72 hours.""",
            "fine_print": """Proxy conflicts, cyber warfare, and economic sanctions do not count unless accompanied
            by direct military engagement. Peacekeeping operations and humanitarian interventions do not count."""
        },
        {
            "name": "Technical/Scientific Question",
            "question": "Will a quantum computer achieve 1000+ logical qubits before 2026?",
            "background": """Current quantum computers have achieved hundreds of physical qubits, but logical qubits
            (error-corrected) are much more challenging. IBM, Google, and other companies are racing to achieve
            quantum advantage in practical applications.""",
            "resolution_criteria": "This resolves Yes if a quantum computer demonstrates 1000 or more logical qubits in a peer-reviewed publication.",
            "fine_print": "Physical qubits do not count, only error-corrected logical qubits."
        }
    ]

    for i, question_data in enumerate(test_questions, 1):
        print(f"{i}. {question_data['name']}")
        print("-" * 60)
        print(f"Question: {question_data['question']}")
        print()

        # Perform complexity analysis
        assessment = analyzer.assess_question_complexity(
            question_data['question'],
            question_data['background'],
            question_data['resolution_criteria'],
            question_data['fine_print']
        )

        print("COMPLEXITY ASSESSMENT:")
        print(f"  Level: {assessment.level.value.upper()}")
        print(f"  Score: {assessment.score:.2f}")
        print(f"  Reasoning: {assessment.reasoning}")
        print()

        print("DETAILED FACTORS:")
        for factor, score in assessment.factors.items():
            if score > 0:
                print(f"  {factor}: {score:.2f}")
        print()

        # Show model recommendations for different budget states
        print("MODEL RECOMMENDATIONS:")
        budget_states = ["normal", "conservative", "emergency"]
        for budget_state in budget_states:
            research_model = analyzer.get_model_for_task("research", assessment, budget_state)
            forecast_model = analyzer.get_model_for_task("forecast", assessment, budget_state)
            print(f"  {budget_state.capitalize()} Budget:")
            print(f"    Research: {research_model}")
            print(f"    Forecast: {forecast_model}")
        print()

        # Show cost estimates
        print("COST ESTIMATES (Normal Budget):")
        research_cost = analyzer.estimate_cost_per_task(assessment, "research", "normal")
        forecast_cost = analyzer.estimate_cost_per_task(assessment, "forecast", "normal")
        total_cost = research_cost["estimated_cost"] + forecast_cost["estimated_cost"]

        print(f"  Research: ${research_cost['estimated_cost']:.4f} ({research_cost['model']})")
        print(f"  Forecast: ${forecast_cost['estimated_cost']:.4f} ({forecast_cost['model']})")
        print(f"  Total: ${total_cost:.4f}")
        print()
        print("=" * 80)
        print()


def demo_budget_impact():
    """Demonstrate how budget status affects model selection."""
    print("BUDGET IMPACT DEMONSTRATION")
    print("=" * 80)
    print()

    analyzer = TaskComplexityAnalyzer()

    # Use a complex question to show the impact
    complex_question = """Will there be a systemic global financial crisis involving
    multiple interconnected factors including sovereign debt, banking sector instability,
    and geopolitical tensions before 2025?"""

    background = """This involves complex macroeconomic relationships, international
    financial markets, central bank policies, and geopolitical risks that could cascade
    into a global crisis."""

    assessment = analyzer.assess_question_complexity(complex_question, background)

    print(f"Question Complexity: {assessment.level.value.upper()} (Score: {assessment.score:.2f})")
    print()

    budget_scenarios = [
        ("normal", "Full budget available, can use premium models"),
        ("conservative", "80%+ budget used, prefer cost-effective models"),
        ("emergency", "95%+ budget used, use only cheapest models")
    ]

    print("MODEL SELECTION BY BUDGET STATUS:")
    print("-" * 50)

    for budget_status, description in budget_scenarios:
        research_model = analyzer.get_model_for_task("research", assessment, budget_status)
        forecast_model = analyzer.get_model_for_task("forecast", assessment, budget_status)

        research_cost = analyzer.estimate_cost_per_task(assessment, "research", budget_status)
        forecast_cost = analyzer.estimate_cost_per_task(assessment, "forecast", budget_status)
        total_cost = research_cost["estimated_cost"] + forecast_cost["estimated_cost"]

        print(f"{budget_status.upper()} Budget ({description}):")
        print(f"  Research: {research_model} (${research_cost['estimated_cost']:.4f})")
        print(f"  Forecast: {forecast_model} (${forecast_cost['estimated_cost']:.4f})")
        print(f"  Total Cost: ${total_cost:.4f}")
        print()


def demo_model_selection_strategy():
    """Demonstrate the model selection strategy."""
    print("MODEL SELECTION STRATEGY DEMONSTRATION")
    print("=" * 80)
    print()

    analyzer = TaskComplexityAnalyzer()

    print("STRATEGY OVERVIEW:")
    print("- Simple questions: Use GPT-4o-mini for all tasks (cost-effective)")
    print("- Medium questions: Use GPT-4o-mini for research, GPT-4o-mini for forecasts")
    print("- Complex questions: Use GPT-4o-mini for research, GPT-4o for forecasts (when budget allows)")
    print("- Budget constraints override complexity-based selection")
    print()

    # Test different complexity levels
    test_cases = [
        ("Simple", "Will X be announced by date Y?", "Official announcement expected."),
        ("Medium", "Will market index reach target by date?", "Market conditions are uncertain."),
        ("Complex", "Will geopolitical crisis involving multiple factors occur?",
         "Complex interdependent international dynamics with uncertain outcomes.")
    ]

    print("EXAMPLES:")
    print("-" * 40)

    for complexity_name, question, background in test_cases:
        assessment = analyzer.assess_question_complexity(question, background)

        print(f"{complexity_name} Question:")
        print(f"  Assessed Level: {assessment.level.value}")
        print(f"  Normal Budget - Research: {analyzer.get_model_for_task('research', assessment, 'normal')}")
        print(f"  Normal Budget - Forecast: {analyzer.get_model_for_task('forecast', assessment, 'normal')}")

        cost_estimate = analyzer.estimate_cost_per_task(assessment, "forecast", "normal")
        print(f"  Estimated Cost: ${cost_estimate['estimated_cost']:.4f}")
        print()


if __name__ == "__main__":
    print("Task Complexity Analyzer Demo")
    print("This demonstrates intelligent model selection for tournament forecasting")
    print()

    try:
        demo_complexity_analysis()
        demo_budget_impact()
        demo_model_selection_strategy()

        print("Demo completed successfully!")
        print()
        print("Key Benefits:")
        print("✓ Intelligent model selection based on question complexity")
        print("✓ Budget-aware operation with automatic cost optimization")
        print("✓ Detailed cost estimation and tracking")
        print("✓ Maintains forecast quality while optimizing costs")
        print("✓ Supports tournament requirements within $100 budget")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
