"""
Demonstration of the integrated performance monitoring and analytics system.
Shows real-time cost tracking, model effectiveness analysis, and optimization recommendations.
"""

import asyncio
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.monitoring.integrated_monitoring_service import IntegratedMonitoringService


async def simulate_tournament_questions(monitoring: IntegratedMonitoringService, num_questions: int = 50):
    """Simulate processing tournament questions with various models and outcomes."""

    # Model configurations for simulation
    models = [
        ("openai/gpt-5", "full", 0.05, "Complex forecasting with maximum reasoning"),
        ("openai/gpt-5-mini", "mini", 0.02, "Research synthesis and intermediate analysis"),
        ("openai/gpt-5-nano", "nano", 0.005, "Fast validation and simple tasks"),
        ("moonshotai/kimi-k2:free", "nano", 0.0, "Free model fallback"),
        ("openai/gpt-oss-20b:free", "nano", 0.0, "Free model emergency fallback")
    ]

    task_types = ["forecast", "research", "validation", "simple"]
    operation_modes = ["normal", "conservative", "emergency", "critical"]

    print(f"🚀 Starting simulation of {num_questions} tournament questions...")
    print("=" * 60)

    total_budget = 100.0
    budget_used = 0.0

    for i in range(num_questions):
        question_id = f"sim-question-{i+1:03d}"

        # Determine operation mode based on budget usage
        budget_used_percentage = (budget_used / total_budget) * 100
        if budget_used_percentage < 25:
            operation_mode = "normal"
            model_weights = [0.4, 0.4, 0.15, 0.03, 0.02]  # Prefer premium models
        elif budget_used_percentage < 60:
            operation_mode = "conservative"
            model_weights = [0.2, 0.5, 0.25, 0.03, 0.02]  # Prefer mini/nano
        elif budget_used_percentage < 85:
            operation_mode = "emergency"
            model_weights = [0.1, 0.3, 0.4, 0.1, 0.1]  # Prefer nano/free
        else:
            operation_mode = "critical"
            model_weights = [0.05, 0.15, 0.3, 0.25, 0.25]  # Prefer free models

        # Select model based on operation mode
        model_name, tier, base_cost, rationale = random.choices(models, weights=model_weights)[0]
        task_type = random.choice(task_types)

        # Add some cost variation
        estimated_cost = base_cost * random.uniform(0.8, 1.2) if base_cost > 0 else 0.0
        budget_remaining = total_budget - budget_used

        # Record model selection
        monitoring.record_model_usage(
            question_id=question_id,
            task_type=task_type,
            selected_model=model_name,
            selected_tier=tier,
            routing_rationale=rationale,
            estimated_cost=estimated_cost,
            operation_mode=operation_mode,
            budget_remaining=budget_remaining
        )

        # Simulate processing time
        processing_time = random.uniform(15.0, 90.0)
        await asyncio.sleep(0.01)  # Small delay for realism

        # Simulate execution outcome
        actual_cost = estimated_cost * random.uniform(0.9, 1.1) if estimated_cost > 0 else 0.0
        execution_time = processing_time * random.uniform(0.8, 1.2)

        # Quality score based on model tier and cost
        if tier == "full":
            base_quality = 0.85
        elif tier == "mini":
            base_quality = 0.75
        else:  # nano or free
            base_quality = 0.65

        quality_score = max(0.3, min(1.0, base_quality + random.uniform(-0.15, 0.15)))

        # Success rate based on model and operation mode
        success_probability = 0.95 if operation_mode in ["normal", "conservative"] else 0.90
        success = random.random() < success_probability

        # Fallback usage probability
        fallback_probability = 0.05 if operation_mode == "normal" else 0.15
        fallback_used = random.random() < fallback_probability

        # Generate forecast values for demonstration
        forecast_value = random.uniform(0.1, 0.9)
        confidence = random.uniform(0.6, 0.95)

        # Record execution outcome
        monitoring.record_execution_outcome(
            question_id=question_id,
            actual_cost=actual_cost,
            execution_time=execution_time,
            quality_score=quality_score,
            success=success,
            fallback_used=fallback_used,
            forecast_value=forecast_value,
            confidence=confidence
        )

        budget_used += actual_cost

        # Print progress every 10 questions
        if (i + 1) % 10 == 0:
            print(f"📊 Processed {i+1}/{num_questions} questions | "
                  f"Budget: ${budget_used:.3f}/${total_budget:.0f} ({budget_used_percentage:.1f}%) | "
                  f"Mode: {operation_mode}")

    print(f"✅ Simulation completed! Total budget used: ${budget_used:.3f}")
    return budget_used


def demonstrate_real_time_monitoring(monitoring: IntegratedMonitoringService):
    """Demonstrate real-time monitoring capabilities."""
    print("\n🔍 REAL-TIME MONITORING ANALYSIS")
    print("=" * 60)

    # Get comprehensive status
    status = monitoring.get_comprehensive_status(100.0)

    print(f"Overall Health: {status.overall_health.upper()}")
    print(f"Timestamp: {status.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Budget status
    budget = status.budget_status.get("budget", {})
    print("\n💰 Budget Status:")
    print(f"  Used: ${budget.get('spent', 0):.3f} / ${budget.get('total', 100):.0f}")
    print(f"  Remaining: ${budget.get('remaining', 0):.3f}")
    print(f"  Utilization: {budget.get('utilization_percent', 0):.1f}%")
    print(f"  Questions Processed: {budget.get('questions_processed', 0)}")
    print(f"  Avg Cost per Question: ${budget.get('avg_cost_per_question', 0):.4f}")

    # Performance metrics
    perf = status.performance_metrics.get("quality_metrics", {})
    print("\n📈 Performance Metrics:")
    print(f"  Avg Quality Score: {perf.get('avg_quality_score', 0):.3f}")
    print(f"  Success Rate: {perf.get('success_rate', 0):.1%}")
    print(f"  Fallback Rate: {perf.get('fallback_rate', 0):.1%}")
    print(f"  Avg Execution Time: {perf.get('avg_execution_time', 0):.1f}s")

    # Tournament competitiveness
    tournament = status.tournament_competitiveness
    print("\n🏆 Tournament Competitiveness:")
    print(f"  Level: {tournament.get('competitiveness_level', 'unknown').upper()}")
    print(f"  Cost Efficiency: {tournament.get('cost_efficiency_score', 0):.1f} questions/$")
    print(f"  Quality Efficiency: {tournament.get('quality_efficiency_score', 0):.1f} quality/$")
    print(f"  Projected Questions Remaining: {tournament.get('projected_questions_remaining', 0)}")

    # Active alerts
    if status.active_alerts:
        print(f"\n⚠️  Active Alerts ({len(status.active_alerts)}):")
        for alert in status.active_alerts[:5]:  # Show top 5
            severity_icon = "🔴" if alert.severity == "critical" else "🟡" if alert.severity == "warning" else "🔵"
            print(f"  {severity_icon} {alert.message}")

    # Top recommendations
    if status.optimization_recommendations:
        print("\n💡 Top Optimization Recommendations:")
        for i, rec in enumerate(status.optimization_recommendations[:5], 1):
            print(f"  {i}. {rec}")


def demonstrate_cost_effectiveness_analysis(monitoring: IntegratedMonitoringService):
    """Demonstrate cost-effectiveness analysis."""
    print("\n💹 COST-EFFECTIVENESS ANALYSIS")
    print("=" * 60)

    # Get cost breakdown
    cost_breakdown = monitoring.model_tracker.get_cost_breakdown(24)

    print(f"Total Questions: {cost_breakdown.question_count}")
    print(f"Total Cost: ${cost_breakdown.total_cost:.4f}")
    print(f"Avg Cost per Question: ${cost_breakdown.avg_cost_per_question:.4f}")

    # Tier breakdown
    if cost_breakdown.by_tier:
        print("\n📊 Cost by Model Tier:")
        for tier, data in cost_breakdown.by_tier.items():
            efficiency = data['count'] / data['cost'] if data['cost'] > 0 else 0
            print(f"  {tier.upper()}: {data['count']} questions, ${data['cost']:.4f} "
                  f"(${data['avg_cost']:.4f}/q, {efficiency:.1f} q/$)")

    # Task type breakdown
    if cost_breakdown.by_task_type:
        print("\n📋 Cost by Task Type:")
        for task, data in cost_breakdown.by_task_type.items():
            efficiency = data['count'] / data['cost'] if data['cost'] > 0 else 0
            print(f"  {task}: {data['count']} questions, ${data['cost']:.4f} "
                  f"(${data['avg_cost']:.4f}/q, {efficiency:.1f} q/$)")

    # Operation mode breakdown
    if cost_breakdown.by_operation_mode:
        print("\n⚙️  Cost by Operation Mode:")
        for mode, data in cost_breakdown.by_operation_mode.items():
            efficiency = data['count'] / data['cost'] if data['cost'] > 0 else 0
            print(f"  {mode}: {data['count']} questions, ${data['cost']:.4f} "
                  f"(${data['avg_cost']:.4f}/q, {efficiency:.1f} q/$)")


def demonstrate_strategic_recommendations(monitoring: IntegratedMonitoringService, budget_used_percentage: float):
    """Demonstrate strategic recommendations."""
    print(f"\n🎯 STRATEGIC RECOMMENDATIONS (Budget Used: {budget_used_percentage:.1f}%)")
    print("=" * 60)

    recommendations = monitoring.generate_strategic_recommendations(budget_used_percentage, 100.0)

    # Tournament phase strategy
    phase_strategy = recommendations["tournament_phase_strategy"]
    print(f"Tournament Phase: {phase_strategy['phase'].upper()}")
    print(f"Risk Tolerance: {phase_strategy['risk_tolerance']}")

    print("\n📊 Recommended Budget Allocation:")
    for tier, percentage in phase_strategy["budget_allocation_strategy"].items():
        print(f"  {tier.upper()}: {percentage:.1f}%")

    print("\n🔧 Routing Adjustments:")
    for i, adjustment in enumerate(phase_strategy["routing_adjustments"], 1):
        print(f"  {i}. {adjustment}")

    # Budget optimization
    budget_opt = recommendations["budget_optimization"]
    print("\n💰 Budget Optimization:")
    print(f"  Potential Savings: ${budget_opt['potential_savings']:.4f}")
    print(f"  Additional Questions Possible: {budget_opt['additional_questions_possible']}")
    print(f"  Risk Assessment: {budget_opt['risk_assessment'].upper()}")

    if budget_opt["implementation_steps"]:
        print("\n📋 Implementation Steps:")
        for i, step in enumerate(budget_opt["implementation_steps"], 1):
            print(f"  {i}. {step}")

    # Implementation priorities
    priorities = recommendations["implementation_priority"]
    if priorities:
        print("\n🎯 Implementation Priorities:")
        for i, priority in enumerate(priorities, 1):
            print(f"  {i}. {priority}")


def demonstrate_trend_analysis(monitoring: IntegratedMonitoringService):
    """Demonstrate trend analysis capabilities."""
    print("\n📈 TREND ANALYSIS")
    print("=" * 60)

    trends = monitoring.model_tracker.get_model_effectiveness_trends(7)

    if trends.get("insufficient_data"):
        print("⚠️  Insufficient data for trend analysis (need more historical data)")
        return

    # Daily trends
    daily_trends = trends.get("daily_trends", {})
    if daily_trends:
        print("📅 Daily Performance Trends:")
        for date, metrics in sorted(daily_trends.items())[-7:]:  # Last 7 days
            print(f"  {date}: {metrics['question_count']} questions, "
                  f"${metrics['avg_cost_per_question']:.4f}/q, "
                  f"quality {metrics['avg_quality_score']:.3f}, "
                  f"{metrics['cost_efficiency']:.1f} q/$")

    # Trend analysis
    trend_analysis = trends.get("trend_analysis", {})
    if trend_analysis:
        print("\n📊 Trend Analysis:")
        print(f"  Cost Trend: {trend_analysis.get('cost_trend', 'unknown').upper()}")
        print(f"  Quality Trend: {trend_analysis.get('quality_trend', 'unknown').upper()}")
        print(f"  Efficiency Trend: {trend_analysis.get('efficiency_trend', 'unknown').upper()}")


async def main():
    """Main demonstration function."""
    print("🎯 PERFORMANCE MONITORING & ANALYTICS DEMONSTRATION")
    print("=" * 80)
    print("This demo shows the integrated monitoring system for tournament optimization.")
    print("It simulates question processing and demonstrates real-time analytics.\n")

    # Initialize monitoring service
    monitoring = IntegratedMonitoringService()

    # Start monitoring service
    print("🔧 Starting monitoring service...")
    monitoring.start_monitoring()

    try:
        # Simulate tournament questions
        budget_used = await simulate_tournament_questions(monitoring, 50)
        budget_used_percentage = (budget_used / 100.0) * 100

        # Wait a moment for monitoring to process
        await asyncio.sleep(1)

        # Demonstrate monitoring capabilities
        demonstrate_real_time_monitoring(monitoring)
        demonstrate_cost_effectiveness_analysis(monitoring)
        demonstrate_strategic_recommendations(monitoring, budget_used_percentage)
        demonstrate_trend_analysis(monitoring)

        # Check for alerts
        print("\n🚨 ALERT CHECKING")
        print("=" * 60)
        alerts = monitoring.check_alerts_and_thresholds()

        if alerts:
            print(f"Found {len(alerts)} alerts:")
            for alert in alerts:
                severity_icon = "🔴" if alert.severity == "critical" else "🟡" if alert.severity == "warning" else "🔵"
                print(f"  {severity_icon} [{alert.alert_type.upper()}] {alert.message}")
                if alert.recommendations:
                    for rec in alert.recommendations[:2]:  # Show top 2 recommendations
                        print(f"    💡 {rec}")
        else:
            print("✅ No alerts detected - system operating normally")

        # Export monitoring data
        print("\n📤 DATA EXPORT")
        print("=" * 60)
        export_data = monitoring.export_monitoring_data(24)
        print("Exported comprehensive monitoring data:")
        print(f"  - Comprehensive status with {len(export_data['comprehensive_status'])} metrics")
        print(f"  - Cost breakdown for {export_data['cost_breakdown']['question_count']} questions")
        print("  - Quality metrics and trends")
        print("  - Optimization analysis and recommendations")
        print(f"  - Alert history with {len(export_data['alert_history'])} alerts")

        print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("📊 Final Stats:")
        print(f"  - Questions Processed: {export_data['cost_breakdown']['question_count']}")
        print(f"  - Total Cost: ${export_data['cost_breakdown']['total_cost']:.4f}")
        print(f"  - Budget Utilization: {budget_used_percentage:.1f}%")
        print(f"  - Cost Efficiency: {export_data['cost_breakdown']['question_count'] / max(export_data['cost_breakdown']['total_cost'], 0.001):.1f} questions/$")

    finally:
        # Stop monitoring service
        print("\n🔧 Stopping monitoring service...")
        monitoring.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
