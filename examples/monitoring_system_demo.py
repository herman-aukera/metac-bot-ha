#!/usr/bin/env python3
"""
Demonstration of the comprehensive monitoring and performance tracking system.
Shows how to integrate monitoring into tournament forecasting workflow.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.infrastructure.monitoring.monitoring_integration import monitoring_integration
from src.infrastructure.monitoring.comprehensive_monitor import comprehensive_monitor
from src.infrastructure.monitoring.alert_system import alert_system
import time
import random

def demo_monitoring_system():
    """Demonstrate the monitoring system capabilities."""
    print("=== Tournament API Optimization Monitoring System Demo ===\n")

    # 1. Track some sample API calls
    print("1. Tracking sample API calls...")

    sample_questions = [
        ("q1", "Will the S&P 500 close above 4500 by end of month?"),
        ("q2", "Will unemployment rate decrease next quarter?"),
        ("q3", "Will Bitcoin price exceed $50,000 by year end?")
    ]

    for question_id, question_text in sample_questions:
        # Simulate research API call
        research_result = monitoring_integration.track_api_call(
            question_id=question_id,
            model="gpt-4o-mini",
            task_type="research",
            prompt=f"Research this question: {question_text}",
            response="Based on economic indicators and market trends...",
            success=True
        )

        # Simulate forecast API call
        forecast_value = random.uniform(0.3, 0.7)
        confidence = random.uniform(0.6, 0.9)

        forecast_result = monitoring_integration.track_forecast(
            question_id=question_id,
            forecast_value=forecast_value,
            confidence=confidence,
            model="gpt-4o"
        )

        print(f"  Tracked {question_id}: forecast={forecast_value:.3f}, confidence={confidence:.3f}")
        time.sleep(0.1)  # Small delay to simulate real usage

    print()

    # 2. Show budget status
    print("2. Current Budget Status:")
    budget_status = monitoring_integration.get_budget_status()
    budget_info = budget_status["budget"]

    print(f"  Total Budget: ${budget_info['total']:.2f}")
    print(f"  Spent: ${budget_info['spent']:.4f} ({budget_info['utilization_percent']:.1f}%)")
    print(f"  Remaining: ${budget_info['remaining']:.4f}")
    print(f"  Questions Processed: {budget_info['questions_processed']}")
    print(f"  Status Level: {budget_info['status_level'].upper()}")
    print()

    # 3. Show performance metrics
    print("3. Performance Metrics:")
    performance_metrics = monitoring_integration.get_performance_metrics()

    print(f"  Total Forecasts: {performance_metrics['total_forecasts']}")
    print(f"  Resolved Forecasts: {performance_metrics['resolved_forecasts']}")
    print(f"  Overall Brier Score: {performance_metrics['overall_brier_score']:.4f}")
    print(f"  Calibration Error: {performance_metrics['calibration_error']:.4f}")
    print(f"  Performance Trend: {performance_metrics['performance_trend']}")
    print()

    # 4. Show comprehensive dashboard
    print("4. Comprehensive Dashboard Summary:")
    dashboard = monitoring_integration.get_dashboard_data()
    summary = dashboard["summary"]

    print(f"  Budget Utilization: {summary['budget_utilization']:.1f}%")
    print(f"  Questions Processed: {summary['questions_processed']}")
    print(f"  API Success Rate: {summary['api_success_rate']:.1%}")
    print(f"  Active Alerts: {summary['active_alerts']}")
    print()

    # 5. Show recommendations
    if dashboard.get("recommendations"):
        print("5. Optimization Recommendations:")
        for i, rec in enumerate(dashboard["recommendations"], 1):
            print(f"  {i}. {rec}")
        print()

    # 6. Demonstrate alert system
    print("6. Alert System Status:")
    alert_summary = alert_system.get_alert_summary()

    print(f"  Active Alerts: {alert_summary['active_alerts']}")
    print(f"  Critical: {alert_summary['active_by_severity']['critical']}")
    print(f"  Warning: {alert_summary['active_by_severity']['warning']}")
    print(f"  Info: {alert_summary['active_by_severity']['info']}")
    print(f"  Alerts in Last 24h: {alert_summary['alerts_last_24h']}")
    print()

    # 7. Simulate updating forecast outcomes
    print("7. Simulating forecast outcome updates...")
    for question_id, _ in sample_questions:
        # Simulate random outcome
        actual_outcome = random.choice([0.0, 1.0])  # Binary outcome
        success = monitoring_integration.update_forecast_outcome(question_id, actual_outcome)
        print(f"  Updated {question_id} outcome: {actual_outcome} (success: {success})")

    print()

    # 8. Show final status
    print("8. Final System Health Check:")
    health_check = comprehensive_monitor.get_health_check()

    print(f"  Overall Status: {health_check['status'].upper()}")
    print(f"  Monitoring Active: {health_check['monitoring_active']}")

    if health_check.get("issues"):
        print("  Issues Detected:")
        for issue in health_check["issues"]:
            print(f"    - {issue}")
    else:
        print("  No issues detected")

    print("\n=== Demo Complete ===")
    print("\nThe monitoring system is now tracking your tournament forecasting activity.")
    print("Check the logs/ directory for detailed monitoring data and alerts.")


if __name__ == "__main__":
    demo_monitoring_system()
