#!/usr/bin/env python3
"""
Tournament Scheduling Configuration Demo

This script demonstrates the new tournament scheduling features including:
- Configurable base scheduling frequency (default 4 hours)
- Deadline-aware scheduling with different frequencies for critical periods
- Environment variable configuration
- Manual scheduling control
"""

import os
import sys
sys.path.append('.')
from src.infrastructure.config.tournament_config import TournamentConfig, TournamentMode


def demo_basic_scheduling():
    """Demonstrate basic scheduling configuration."""
    print("🕐 Basic Scheduling Configuration")
    print("=" * 50)

    config = TournamentConfig()

    print(f"Default scheduling frequency: {config.scheduling_interval_hours} hours")
    print(f"Deadline-aware scheduling: {config.deadline_aware_scheduling}")
    print(f"Critical period frequency: {config.critical_period_frequency_hours} hours")
    print(f"Final 24h frequency: {config.final_24h_frequency_hours} hours")
    print(f"Tournament scope: {config.tournament_scope}")
    print(f"Cron schedule: {config.get_cron_schedule()}")
    print()


def demo_deadline_aware_scheduling():
    """Demonstrate deadline-aware scheduling logic."""
    print("⏰ Deadline-Aware Scheduling")
    print("=" * 50)

    config = TournamentConfig()

    test_scenarios = [
        (200, "Normal period (8+ days)"),
        (100, "Normal period (4+ days)"),
        (72, "Critical period boundary (3 days)"),
        (48, "Critical period (2 days)"),
        (25, "Critical period (1+ day)"),
        (24, "Final 24h boundary"),
        (12, "Final 12 hours"),
        (1, "Final hour")
    ]

    for hours_left, description in test_scenarios:
        frequency = config.get_deadline_aware_frequency(hours_left)
        print(f"{description:25} | {hours_left:3}h left → Run every {frequency}h")

    print()


def demo_should_run_logic():
    """Demonstrate the should_run_now decision logic."""
    print("🤔 Should Run Now Logic")
    print("=" * 50)

    config = TournamentConfig()

    test_cases = [
        (100, 4, "Normal period, 4h since last run"),
        (100, 3, "Normal period, 3h since last run"),
        (48, 2, "Critical period, 2h since last run"),
        (48, 1, "Critical period, 1h since last run"),
        (12, 1, "Final 24h, 1h since last run"),
        (12, 0.5, "Final 24h, 30min since last run"),
    ]

    for hours_left, hours_since_last, description in test_cases:
        should_run = config.should_run_now(hours_left, hours_since_last)
        status = "✅ RUN" if should_run else "⏸️ WAIT"
        print(f"{description:35} → {status}")

    print()


def demo_environment_configuration():
    """Demonstrate configuration from environment variables."""
    print("🌍 Environment Variable Configuration")
    print("=" * 50)

    # Save original environment
    original_env = dict(os.environ)

    try:
        # Set custom environment variables
        os.environ.update({
            "SCHEDULING_FREQUENCY_HOURS": "6",
            "DEADLINE_AWARE_SCHEDULING": "false",
            "CRITICAL_PERIOD_FREQUENCY_HOURS": "3",
            "FINAL_24H_FREQUENCY_HOURS": "2",
            "TOURNAMENT_SCOPE": "daily"
        })

        config = TournamentConfig.from_environment()

        print("Custom configuration from environment:")
        print(f"  Base frequency: {config.scheduling_interval_hours} hours")
        print(f"  Deadline-aware: {config.deadline_aware_scheduling}")
        print(f"  Critical period: {config.critical_period_frequency_hours} hours")
        print(f"  Final 24h: {config.final_24h_frequency_hours} hours")
        print(f"  Tournament scope: {config.tournament_scope}")

        # Test with deadline-aware disabled
        print("\nWith deadline-aware disabled:")
        print(f"  100h left → {config.get_deadline_aware_frequency(100)}h frequency")
        print(f"  12h left → {config.get_deadline_aware_frequency(12)}h frequency")

    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    print()


def demo_scheduling_strategy():
    """Demonstrate the scheduling strategy output."""
    print("📋 Scheduling Strategy Configuration")
    print("=" * 50)

    config = TournamentConfig()
    strategy = config.get_scheduling_strategy()

    print("Current scheduling strategy:")
    for key, value in strategy.items():
        print(f"  {key}: {value}")

    print()


def demo_tournament_modes():
    """Demonstrate different tournament modes and their scheduling."""
    print("🏆 Tournament Mode Scheduling")
    print("=" * 50)

    modes = [
        (TournamentMode.TOURNAMENT, "Tournament Mode"),
        (TournamentMode.DEVELOPMENT, "Development Mode"),
        (TournamentMode.QUARTERLY_CUP, "Quarterly Cup Mode"),
    ]

    for mode, description in modes:
        config = TournamentConfig(mode=mode, scheduling_interval_hours=4)
        cron = config.get_cron_schedule()
        print(f"{description:20} → {cron}")

    print()


def main():
    """Run all scheduling demos."""
    print("🚀 Tournament Scheduling Configuration Demo")
    print("=" * 60)
    print()

    demo_basic_scheduling()
    demo_deadline_aware_scheduling()
    demo_should_run_logic()
    demo_environment_configuration()
    demo_scheduling_strategy()
    demo_tournament_modes()

    print("✅ Demo completed!")
    print()
    print("💡 Key Benefits:")
    print("  • Reduced from 30min to 4h frequency saves ~90% of API costs")
    print("  • Deadline-aware scheduling ensures timely final submissions")
    print("  • Configurable through environment variables")
    print("  • Manual control through GitHub Actions workflow dispatch")
    print("  • Seasonal tournament scope (50-100 questions total)")


if __name__ == "__main__":
    main()
