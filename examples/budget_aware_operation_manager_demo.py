"""
Budget-Aware Operation Manager Demo.
Demonstrates dynamic operation mode detection, switching, and cost optimization strategies.
"""
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.infrastructure.config.budget_aware_operation_manager import (
    budget_aware_operation_manager, EmergencyProtocol
)
from src.infrastructure.config.operation_modes import OperationMode
from src.domain.services.cost_optimization_service import (
    cost_optimization_service, TaskPriority, TaskComplexity
)


async def demonstrate_budget_monitoring():
    """Demonstrate budget utilization monitoring and threshold detection."""
    print("\n" + "="*60)
    print("BUDGET MONITORING AND THRESHOLD DETECTION")
    print("="*60)

    # Monitor current budget utilization
    monitoring_result = budget_aware_operation_manager.monitor_budget_utilization()

    print(f"Current Budget Utilization: {monitoring_result['current_utilization']:.1f}%")
    print(f"Emergency Protocol: {monitoring_result['emergency_protocol']}")

    if monitoring_result['threshold_alerts']:
        print("\nThreshold Alerts:")
        for alert in monitoring_result['threshold_alerts']:
            print(f"  - {alert['threshold_name']}: {alert['percentage']}% threshold crossed")
            print(f"    Recommended mode: {alert['recommended_mode']}")
            print(f"    Actions: {', '.join(alert['actions'])}")
    else:
        print("No threshold alerts - operating within normal parameters")


async def demonstrate_operation_mode_switching():
    """Demonstrate automatic operation mode detection and switching."""
    print("\n" + "="*60)
    print("OPERATION MODE DETECTION AND SWITCHING")
    print("="*60)

    # Check and potentially switch operation modes
    mode_changed, transition_log = budget_aware_operation_manager.detect_and_switch_operation_mode()

    if mode_changed and transition_log:
        print(f"Mode transition detected!")
        print(f"  From: {transition_log.from_mode.value}")
        print(f"  To: {transition_log.to_mode.value}")
        print(f"  Reason: {transition_log.trigger_reason}")
        print(f"  Budget Utilization: {transition_log.budget_utilization:.1f}%")
        print(f"  Emergency Protocol: {transition_log.emergency_protocol.value}")
        print(f"  Estimated Cost Savings: ${transition_log.cost_savings_estimate:.4f}")

        # Show performance impact
        perf_impact = transition_log.performance_impact
        print(f"  Performance Impact Score: {perf_impact.get('performance_score_change', 0):.2f}")

        if 'feature_changes' in perf_impact:
            print("  Feature Changes:")
            for feature, change in perf_impact['feature_changes'].items():
                print(f"    {feature}: {change['from']} → {change['to']}")
    else:
        print("No mode transition needed - current mode is appropriate")

    # Show current operation mode status
    budget_aware_operation_manager.log_operation_mode_performance()


async def demonstrate_cost_optimization_strategies():
    """Demonstrate cost optimization strategies for different operation modes."""
    print("\n" + "="*60)
    print("COST OPTIMIZATION STRATEGIES")
    print("="*60)

    # Test model selection optimization across different modes
    test_scenarios = [
        ("research", "openai/gpt-4o", OperationMode.NORMAL, TaskComplexity.MEDIUM),
        ("forecast", "claude-3-5-sonnet", OperationMode.CONSERVATIVE, TaskComplexity.HIGH),
        ("validation", "openai/gpt-4o", OperationMode.EMERGENCY, TaskComplexity.MINIMAL)
    ]

    for task_type, original_model, mode, complexity in test_scenarios:
        print(f"\nScenario: {task_type.upper()} task in {mode.value.upper()} mode")
        print(f"Original model: {original_model}")
        print(f"Task complexity: {complexity.value}")

        # Get optimized model selection
        model_result = cost_optimization_service.optimize_model_selection(
            task_type=task_type,
            original_model=original_model,
            operation_mode=mode,
            task_complexity=complexity
        )

        print(f"Optimized model: {model_result.selected_model}")
        print(f"Cost reduction: {model_result.cost_reduction:.1%}")
        print(f"Performance impact: {model_result.performance_impact:.1%}")
        print(f"Rationale: {model_result.rationale}")


async def demonstrate_task_prioritization():
    """Demonstrate task prioritization algorithms for budget conservation."""
    print("\n" + "="*60)
    print("TASK PRIORITIZATION FOR BUDGET CONSERVATION")
    print("="*60)

    # Test different task priorities across operation modes
    test_tasks = [
        ("Critical forecasting task", TaskPriority.CRITICAL, TaskComplexity.HIGH),
        ("Normal research task", TaskPriority.NORMAL, TaskComplexity.MEDIUM),
        ("Low priority validation", TaskPriority.LOW, TaskComplexity.MINIMAL)
    ]

    current_mode = budget_aware_operation_manager.operation_mode_manager.get_current_mode()

    print(f"Current operation mode: {current_mode.value.upper()}")
    print("\nTask Prioritization Results:")

    for task_desc, priority, complexity in test_tasks:
        result = cost_optimization_service.prioritize_task(
            task_description=task_desc,
            task_priority=priority,
            task_complexity=complexity,
            operation_mode=current_mode,
            estimated_tokens=1200
        )

        status = "✓ APPROVED" if result.should_process else "✗ REJECTED"
        print(f"\n{status}: {task_desc}")
        print(f"  Priority: {priority.value} (score: {result.priority_score:.2f})")
        print(f"  Estimated cost: ${result.estimated_cost:.4f}")
        print(f"  Reason: {result.reason}")

        if result.should_process:
            allocation = result.resource_allocation
            print(f"  Resource allocation: {allocation['cpu_priority']} CPU, "
                  f"{allocation['memory_limit_mb']}MB RAM, "
                  f"{allocation['timeout_seconds']}s timeout")


async def demonstrate_research_depth_adaptation():
    """Demonstrate research depth adaptation based on budget constraints."""
    print("\n" + "="*60)
    print("RESEARCH DEPTH ADAPTATION")
    print("="*60)

    base_config = {
        "max_sources": 10,
        "max_depth": 3,
        "max_iterations": 5
    }

    current_mode = budget_aware_operation_manager.operation_mode_manager.get_current_mode()
    budget_status = budget_aware_operation_manager.budget_manager.get_budget_status()
    budget_remaining = budget_status.remaining / budget_status.total_budget

    print(f"Base research configuration: {base_config}")
    print(f"Current mode: {current_mode.value}")
    print(f"Budget remaining: {budget_remaining:.1%}")

    # Test adaptation for different complexity levels
    complexities = [TaskComplexity.MINIMAL, TaskComplexity.MEDIUM, TaskComplexity.HIGH]

    for complexity in complexities:
        adapted_config = cost_optimization_service.adapt_research_depth(
            base_config=base_config,
            operation_mode=current_mode,
            task_complexity=complexity,
            budget_remaining=budget_remaining
        )

        print(f"\n{complexity.value.upper()} complexity adaptation:")
        print(f"  Max sources: {base_config['max_sources']} → {adapted_config.max_sources}")
        print(f"  Max depth: {base_config['max_depth']} → {adapted_config.max_depth}")
        print(f"  Max iterations: {base_config['max_iterations']} → {adapted_config.max_iterations}")
        print(f"  Deep analysis: {adapted_config.enable_deep_analysis}")
        print(f"  Time limit: {adapted_config.time_limit_seconds}s")


async def demonstrate_graceful_degradation():
    """Demonstrate graceful feature degradation for emergency modes."""
    print("\n" + "="*60)
    print("GRACEFUL FEATURE DEGRADATION")
    print("="*60)

    current_mode = budget_aware_operation_manager.operation_mode_manager.get_current_mode()
    budget_status = budget_aware_operation_manager.budget_manager.get_budget_status()
    budget_remaining = budget_status.remaining / budget_status.total_budget

    # Get degradation strategy
    degradation_strategy = cost_optimization_service.get_graceful_degradation_strategy(
        operation_mode=current_mode,
        budget_remaining=budget_remaining
    )

    print(f"Current mode: {current_mode.value}")
    print(f"Budget remaining: {budget_remaining:.1%}")
    print("\nFeature Degradation Strategy:")

    for feature, enabled in degradation_strategy.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"  {feature}: {status}")

    # Show what features are degraded compared to normal mode
    normal_strategy = cost_optimization_service.get_graceful_degradation_strategy(
        operation_mode=OperationMode.NORMAL,
        budget_remaining=1.0
    )

    print("\nFeatures degraded from normal mode:")
    degraded_features = []
    for feature, enabled in degradation_strategy.items():
        if normal_strategy[feature] and not enabled:
            degraded_features.append(feature)

    if degraded_features:
        for feature in degraded_features:
            print(f"  - {feature}")
    else:
        print("  No features degraded")


async def demonstrate_emergency_protocols():
    """Demonstrate emergency protocol activation and management."""
    print("\n" + "="*60)
    print("EMERGENCY PROTOCOL DEMONSTRATION")
    print("="*60)

    current_protocol = budget_aware_operation_manager.current_emergency_protocol
    print(f"Current emergency protocol: {current_protocol.value}")

    if current_protocol != EmergencyProtocol.NONE:
        activation_time = budget_aware_operation_manager.emergency_activation_time
        print(f"Activated at: {activation_time}")

    # Show what would happen with different emergency protocols
    print("\nEmergency Protocol Effects:")

    protocols = [EmergencyProtocol.BUDGET_WARNING, EmergencyProtocol.BUDGET_CRITICAL]

    for protocol in protocols:
        print(f"\n{protocol.value.upper()}:")

        if protocol == EmergencyProtocol.BUDGET_WARNING:
            print("  - Switch to conservative operation mode")
            print("  - Reduce batch processing sizes")
            print("  - Skip low-priority questions")
            print("  - Enable cost monitoring alerts")
        elif protocol == EmergencyProtocol.BUDGET_CRITICAL:
            print("  - Switch to emergency operation mode")
            print("  - Process critical questions only")
            print("  - Use cheapest models exclusively")
            print("  - Disable non-essential features")
            print("  - Implement strict cost limits")


async def main():
    """Run the complete budget-aware operation manager demonstration."""
    print("Budget-Aware Operation Manager Demonstration")
    print("=" * 80)

    try:
        # Run all demonstrations
        await demonstrate_budget_monitoring()
        await demonstrate_operation_mode_switching()
        await demonstrate_cost_optimization_strategies()
        await demonstrate_task_prioritization()
        await demonstrate_research_depth_adaptation()
        await demonstrate_graceful_degradation()
        await demonstrate_emergency_protocols()

        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("The Budget-Aware Operation Manager successfully demonstrated:")
        print("✓ Budget utilization monitoring and threshold detection")
        print("✓ Automatic operation mode transitions")
        print("✓ Operation mode logging and performance impact tracking")
        print("✓ Emergency protocol activation and management")
        print("✓ Model selection adjustments per operation mode")
        print("✓ Task prioritization algorithms for budget conservation")
        print("✓ Research depth adaptation based on budget constraints")
        print("✓ Graceful feature degradation for emergency modes")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
