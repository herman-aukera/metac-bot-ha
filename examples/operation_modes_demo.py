#!/usr/bin/env python3
"""
Demo script showing budget-aware operation modes functionality.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.config.operation_modes import operation_mode_manager, OperationMode
from src.infrastructure.config.budget_manager import budget_manager


def demo_operation_modes():
    """Demonstrate operation modes functionality."""
    print("=== Budget-Aware Operation Modes Demo ===\n")

    # Show initial status
    print("1. Initial Configuration:")
    operation_mode_manager.log_mode_status()
    print()

    # Show question processing in different modes
    print("2. Question Processing Tests:")

    test_questions = [
        ("high", "Critical market prediction question"),
        ("normal", "Standard forecasting question"),
        ("low", "Low priority research question")
    ]

    for priority, description in test_questions:
        can_process, reason = operation_mode_manager.can_process_question(priority)
        print(f"  {priority.upper()} priority: {'✓' if can_process else '✗'} - {reason}")
    print()

    # Demonstrate mode transitions
    print("3. Mode Transition Simulation:")

    # Force conservative mode
    print("  Switching to CONSERVATIVE mode...")
    transition = operation_mode_manager.force_mode_transition(OperationMode.CONSERVATIVE, "demo")
    print(f"  Transition: {transition.from_mode.value} → {transition.to_mode.value}")

    # Test processing in conservative mode
    print("  Question processing in conservative mode:")
    for priority, description in test_questions:
        can_process, reason = operation_mode_manager.can_process_question(priority)
        print(f"    {priority.upper()}: {'✓' if can_process else '✗'} - {reason}")

    # Force emergency mode
    print("\n  Switching to EMERGENCY mode...")
    transition = operation_mode_manager.force_mode_transition(OperationMode.EMERGENCY, "demo")
    print(f"  Transition: {transition.from_mode.value} → {transition.to_mode.value}")

    # Test processing in emergency mode
    print("  Question processing in emergency mode:")
    for priority, description in test_questions:
        can_process, reason = operation_mode_manager.can_process_question(priority)
        print(f"    {priority.upper()}: {'✓' if can_process else '✗'} - {reason}")
    print()

    # Show model selection in different modes
    print("4. Model Selection by Mode:")

    modes = [OperationMode.NORMAL, OperationMode.CONSERVATIVE, OperationMode.EMERGENCY]
    tasks = ["research", "forecast"]

    for mode in modes:
        operation_mode_manager.force_mode_transition(mode, "demo")
        print(f"  {mode.value.upper()} mode:")

        for task in tasks:
            model = operation_mode_manager.get_model_for_task(task)
            print(f"    {task}: {model}")
        print()

    # Show processing limits
    print("5. Processing Limits by Mode:")

    for mode in modes:
        operation_mode_manager.force_mode_transition(mode, "demo")
        limits = operation_mode_manager.get_processing_limits()

        print(f"  {mode.value.upper()} mode:")
        print(f"    Max questions/batch: {limits['max_questions_per_batch']}")
        print(f"    Max retries: {limits['max_retries']}")
        print(f"    Timeout: {limits['timeout_seconds']}s")
        print(f"    Complexity analysis: {'Enabled' if limits['enable_complexity_analysis'] else 'Disabled'}")
        print(f"    Skip low priority: {'Yes' if limits['skip_low_priority_questions'] else 'No'}")
        print()

    # Show graceful degradation strategy
    print("6. Graceful Degradation Strategies:")

    # Simulate different budget utilization levels
    budget_levels = [0.5, 0.85, 0.97]

    for level in budget_levels:
        # This is a simplified demo - in real usage, budget status comes from actual usage
        print(f"  Budget utilization: {level:.0%}")

        if level >= 0.95:
            mode = OperationMode.EMERGENCY
        elif level >= 0.80:
            mode = OperationMode.CONSERVATIVE
        else:
            mode = OperationMode.NORMAL

        operation_mode_manager.force_mode_transition(mode, "demo")
        strategy = operation_mode_manager.get_graceful_degradation_strategy()

        print(f"    Mode: {strategy['current_mode']}")
        print(f"    Actions:")
        for action in strategy['actions']:
            print(f"      - {action}")
        print()

    # Show transition history
    print("7. Mode Transition History:")
    history = operation_mode_manager.get_mode_history()

    print(f"  Total transitions: {len(history)}")
    if history:
        print("  Recent transitions:")
        for transition in history[-5:]:  # Show last 5
            print(f"    {transition.timestamp.strftime('%H:%M:%S')}: "
                  f"{transition.from_mode.value} → {transition.to_mode.value} "
                  f"({transition.trigger_reason})")
    print()

    # Reset to normal mode
    operation_mode_manager.force_mode_transition(OperationMode.NORMAL, "demo_cleanup")
    print("Demo completed. Reset to NORMAL mode.")


def demo_enhanced_llm_integration():
    """Demonstrate integration with enhanced LLM configuration."""
    print("\n=== Enhanced LLM Configuration Integration ===\n")

    # Show configuration status
    print("1. Configuration Status:")
    try:
        from src.infrastructure.config.enhanced_llm_config import enhanced_llm_config
        enhanced_llm_config.log_configuration_status()
    except Exception as e:
        print(f"Note: Enhanced LLM config requires API keys. Error: {e}")
        print("Skipping enhanced LLM integration demo.")
        return
    print()

    # Test question processing check
    print("2. Question Processing Integration:")

    test_priorities = ["high", "normal", "low"]

    for priority in test_priorities:
        try:
            can_process, reason = enhanced_llm_config.can_process_question(priority)
            print(f"  {priority.upper()} priority: {'✓' if can_process else '✗'} - {reason}")
        except Exception as e:
            print(f"  {priority.upper()} priority: Error - {e}")
    print()


if __name__ == "__main__":
    try:
        demo_operation_modes()
        demo_enhanced_llm_integration()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()

    print("\nDemo finished.")
