#!/usr/bin/env python3
"""
Tournament Scope Management Demo

This script demonstrates the tournament scope management functionality
that updates question volume expectations from daily to seasonal scope.

Key Features:
- Tournament duration and question count estimation
- Sustainable forecasting rate calculations
- Progress tracking and validation
- Seasonal scope validation (50-100 questions total, not per day)
"""

import sys
sys.path.append('src')

from infrastructure.config.tournament_config import (
    get_tournament_config,
    get_tournament_scope_manager
)

def demo_tournament_scope_management():
    """Demonstrate tournament scope management features."""

    print("🏆 Tournament Scope Management Demo")
    print("=" * 50)

    # Get tournament configuration
    config = get_tournament_config()
    scope_manager = get_tournament_scope_manager()

    # Display basic tournament info
    print(f"Tournament: {config.tournament_name}")
    print(f"Tournament ID: {config.tournament_id}")
    print(f"Scope: {config.tournament_scope}")
    print(f"Mode: {config.mode.value}")
    print()

    # Show tournament duration
    duration = config.get_tournament_duration_days()
    if duration:
        print(f"📅 Tournament Duration: {duration} days")
        print(f"Start Date: {config.tournament_start_date}")
        print(f"End Date: {config.tournament_end_date}")
    else:
        print("📅 Tournament Duration: Using default estimate (120 days)")
    print()

    # Show question expectations (seasonal, not daily)
    print("📊 Question Volume Expectations (SEASONAL SCOPE)")
    print(f"Expected Total Questions: {config.expected_total_questions}")
    print(f"Range: {config.min_expected_questions}-{config.max_expected_questions}")
    print(f"Currently Processed: {config.questions_processed}")
    print()

    # Show sustainable forecasting rates
    print("⚡ Sustainable Forecasting Rates")
    rates = config.calculate_sustainable_forecasting_rate()
    print(f"Questions per Day: {rates['questions_per_day']:.2f}")
    print(f"Questions per Week: {rates['questions_per_week']:.1f}")
    print(f"Questions per Month: {rates['questions_per_month']:.1f}")
    print(f"Total Target: {rates['total_target']} questions")
    print()

    # Show progress tracking
    print("📈 Tournament Progress")
    progress = config.get_tournament_progress()
    print(f"Progress: {progress['progress_percentage']:.1f}%")
    print(f"Remaining Questions: {progress['remaining_questions']}")
    print(f"On Track: {'✅ Yes' if progress['is_on_track'] else '❌ No'}")
    print()

    # Show recommended scheduling
    print("⏰ Recommended Scheduling")
    recommended_freq = config.get_recommended_scheduling_frequency()
    print(f"Recommended Frequency: Every {recommended_freq} hours")
    print(f"Current Frequency: Every {config.scheduling_interval_hours} hours")

    # Check if throttling is needed
    current_rate = 2.0  # Example current rate
    should_throttle = config.should_throttle_forecasting(current_rate)
    print(f"Should Throttle (at {current_rate} q/day): {'⚠️ Yes' if should_throttle else '✅ No'}")
    print()

    # Validate seasonal scope
    print("🔍 Seasonal Scope Validation")
    validation = scope_manager.validate_seasonal_scope()
    print(f"Valid Configuration: {'✅ Yes' if validation['is_valid'] else '❌ No'}")

    if validation['issues']:
        print("Issues Found:")
        for issue in validation['issues']:
            print(f"  ❌ {issue}")

    if validation['recommendations']:
        print("Recommendations:")
        for rec in validation['recommendations']:
            print(f"  💡 {rec}")
    print()

    # Show comprehensive summary
    print("📋 Comprehensive Scope Summary")
    summary = scope_manager.get_scope_summary()

    print("Tournament Info:")
    for key, value in summary['tournament_info'].items():
        print(f"  {key}: {value}")

    print("Question Expectations:")
    for key, value in summary['question_expectations'].items():
        print(f"  {key}: {value}")

    print()
    print("🎯 Key Insights:")
    print("• This is a SEASONAL tournament, not daily")
    print(f"• Target: {config.expected_total_questions} questions over {duration or 120} days")
    print(f"• Sustainable rate: ~{rates['questions_per_day']:.1f} questions per day")
    print(f"• Recommended scheduling: Every {recommended_freq} hours")
    print(f"• Current progress: {progress['progress_percentage']:.1f}% complete")

def demo_scope_comparison():
    """Compare daily vs seasonal scope expectations."""

    print("\n" + "=" * 50)
    print("📊 Daily vs Seasonal Scope Comparison")
    print("=" * 50)

    config = get_tournament_config()
    duration = config.get_tournament_duration_days() or 120

    # Daily scope (old expectation)
    daily_questions = 75  # 50-100 questions per day
    daily_total = daily_questions * duration

    # Seasonal scope (new expectation)
    seasonal_total = config.expected_total_questions
    seasonal_daily = seasonal_total / duration

    print("❌ OLD (Daily Scope):")
    print(f"  Expected: {daily_questions} questions PER DAY")
    print(f"  Total over {duration} days: {daily_total:,} questions")
    print("  Completely unsustainable! 💸")
    print()

    print("✅ NEW (Seasonal Scope):")
    print(f"  Expected: {seasonal_total} questions TOTAL")
    print(f"  Daily rate: {seasonal_daily:.2f} questions per day")
    print("  Sustainable and budget-friendly! 💰")
    print()

    print(f"💡 Savings: {daily_total - seasonal_total:,} fewer questions")
    print(f"💡 Rate reduction: {daily_questions / seasonal_daily:.0f}x less frequent")

if __name__ == "__main__":
    demo_tournament_scope_management()
    demo_scope_comparison()
