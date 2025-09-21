#!/usr/bin/env python3
"""
Demo script for optimized research prompts.

This script demonstrates the usage of the new optimized research prompt templates
designed for budget-efficient forecasting.
"""

import sys
import os
from datetime import datetime, timezone, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from domain.entities.question import Question, QuestionType
from prompts.optimized_research_prompts import OptimizedResearchPrompts, QuestionComplexityAnalyzer
from prompts.research_prompt_manager import ResearchPromptManager


def create_sample_questions():
    """Create sample questions for testing different complexity levels."""

    # Simple binary question
    simple_question = Question(
        id="simple_001",
        title="Will it rain tomorrow in San Francisco?",
        description="Weather forecast question for next day.",
        question_type=QuestionType.BINARY,
        categories=["weather"],
        close_time=datetime.now(timezone.utc) + timedelta(days=1)
    )

    # Standard complexity question
    standard_question = Question(
        id="standard_001",
        title="Will the Federal Reserve raise interest rates at their next meeting?",
        description="The Federal Reserve Open Market Committee meets regularly to set monetary policy. This question asks about their next scheduled meeting.",
        question_type=QuestionType.BINARY,
        categories=["economics", "politics"],
        close_time=datetime.now(timezone.utc) + timedelta(days=45)
    )

    # Complex question
    complex_question = Question(
        id="complex_001",
        title="Will a new COVID-19 variant cause WHO to declare a Public Health Emergency of International Concern before 2026?",
        description="The World Health Organization has declared several Public Health Emergencies of International Concern (PHEIC) related to COVID-19. This question asks whether a new variant will trigger another such declaration before the end of 2025, considering factors like transmissibility, severity, immune escape, and global health impact.",
        question_type=QuestionType.BINARY,
        categories=["science", "medicine", "politics", "global-health"],
        close_time=datetime.now(timezone.utc) + timedelta(days=365)
    )

    # Time-sensitive question
    time_sensitive_question = Question(
        id="time_001",
        title="Will the stock market close higher today?",
        description="Will the S&P 500 close higher than yesterday's close?",
        question_type=QuestionType.BINARY,
        categories=["economics", "finance"],
        close_time=datetime.now(timezone.utc) + timedelta(hours=6)
    )

    return [simple_question, standard_question, complex_question, time_sensitive_question]


def demo_complexity_analysis():
    """Demonstrate question complexity analysis."""
    print("=== Question Complexity Analysis Demo ===\n")

    questions = create_sample_questions()
    analyzer = QuestionComplexityAnalyzer()

    for question in questions:
        complexity = analyzer.analyze_complexity(question)
        focus_type = analyzer.determine_focus_type(question)

        print(f"Question: {question.title}")
        print(f"Complexity: {complexity}")
        print(f"Focus Type: {focus_type}")
        print(f"Categories: {', '.join(question.categories)}")
        print("-" * 50)


def demo_optimized_prompts():
    """Demonstrate optimized research prompts."""
    print("\n=== Optimized Research Prompts Demo ===\n")

    questions = create_sample_questions()
    prompts = OptimizedResearchPrompts()

    # Demo different prompt types
    question = questions[1]  # Standard complexity question

    print("SIMPLE RESEARCH PROMPT:")
    print(prompts.get_simple_research_prompt(question))
    print("\n" + "="*80 + "\n")

    print("STANDARD RESEARCH PROMPT:")
    print(prompts.get_standard_research_prompt(question))
    print("\n" + "="*80 + "\n")

    print("NEWS-FOCUSED PROMPT:")
    print(prompts.get_news_focused_prompt(questions[3]))  # Time-sensitive question
    print("\n" + "="*80 + "\n")


def demo_prompt_manager():
    """Demonstrate the research prompt manager."""
    print("\n=== Research Prompt Manager Demo ===\n")

    questions = create_sample_questions()
    manager = ResearchPromptManager(budget_aware=True)

    for i, question in enumerate(questions):
        print(f"Question {i+1}: {question.title}")

        # Get optimal prompt with different budget scenarios
        budget_scenarios = [100, 25, 5]  # High, medium, low budget

        for budget in budget_scenarios:
            result = manager.get_optimal_research_prompt(
                question=question,
                budget_remaining=budget
            )

            print(f"\nBudget: ${budget}")
            print(f"Complexity: {result['complexity_level']}")
            print(f"Focus: {result['focus_type']}")
            print(f"Recommended Model: {result['recommended_model']}")
            print(f"Estimated Cost (GPT-4o-mini): ${result['cost_estimates']['gpt-4o-mini']['total_cost']:.4f}")
            print(f"Estimated Cost (GPT-4o): ${result['cost_estimates']['gpt-4o']['total_cost']:.4f}")

        print("-" * 60)


def demo_efficiency_metrics():
    """Demonstrate prompt efficiency metrics."""
    print("\n=== Prompt Efficiency Metrics Demo ===\n")

    manager = ResearchPromptManager()
    metrics = manager.get_prompt_efficiency_metrics()

    print("EFFICIENCY METRICS BY PROMPT TYPE:")
    print("-" * 40)

    for prompt_type, data in metrics["prompt_metrics"].items():
        print(f"\n{prompt_type.upper()} PROMPT:")
        print(f"  Tokens per dollar (GPT-4o-mini): {data['tokens_per_dollar']['gpt-4o-mini']:.0f}")
        print(f"  Tokens per dollar (GPT-4o): {data['tokens_per_dollar']['gpt-4o']:.0f}")
        print(f"  Cost per question (GPT-4o-mini): ${data['cost_per_question']['gpt-4o-mini']['total_cost']:.4f}")
        print(f"  Cost per question (GPT-4o): ${data['cost_per_question']['gpt-4o']['total_cost']:.4f}")

    print("\nRECOMMENDATIONS:")
    for use_case, prompt_type in metrics["recommendations"].items():
        print(f"  {use_case.replace('_', ' ').title()}: {prompt_type}")


def main():
    """Run all demos."""
    print("Optimized Research Prompts Demo")
    print("=" * 50)

    try:
        demo_complexity_analysis()
        demo_optimized_prompts()
        demo_prompt_manager()
        demo_efficiency_metrics()

        print("\n✅ All demos completed successfully!")
        print("\nKey Benefits of Optimized Research Prompts:")
        print("- Token-efficient templates reduce API costs")
        print("- Structured output formats improve parsing")
        print("- Complexity-aware selection optimizes quality/cost ratio")
        print("- Source citation requirements improve transparency")
        print("- Budget-aware operation prevents overspending")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
