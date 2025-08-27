#!/usr/bin/env python3
"""
Deployment Cost Monitor Script

This script provides comprehensive monitoring and reporting for deployment costs,
budget utilization, and workflow management in the tournament forecasting system.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.infrastructure.config.budget_manager import BudgetManager
from src.infrastructure.monitoring.budget_dashboard import BudgetDashboard
from src.infrastructure.monitoring.alert_system import AlertSystem
from src.infrastructure.monitoring.performance_tracker import PerformanceTracker


class DeploymentCostMonitor:
    """Monitor deployment costs and manage budget-related workflow decisions."""

    def __init__(self, budget_limit: float = None):
        """Initialize the deployment cost monitor."""
        self.budget_limit = budget_limit or float(os.getenv("BUDGET_LIMIT", "100"))
        self.budget_manager = BudgetManager(budget_limit=self.budget_limit)
        self.dashboard = BudgetDashboard(self.budget_manager)
        self.alert_system = AlertSystem()
        self.performance_tracker = PerformanceTracker()

    def check_budget_status(self) -> dict:
        """Check current budget status and return comprehensive report."""
        status = self.budget_manager.get_budget_status()
        recommendations = self.dashboard.get_budget_recommendations()

        # Determine action requirements
        should_suspend = status['utilization'] >= 0.98
        should_alert = status['utilization'] >= 0.80
        should_conserve = status['utilization'] >= 0.80

        return {
            'budget_status': status,
            'recommendations': recommendations,
            'actions': {
                'should_suspend_workflows': should_suspend,
                'should_send_alerts': should_alert,
                'should_enable_conservative_mode': should_conserve,
                'manual_oversight_required': status['utilization'] >= 0.90
            },
            'alert_level': (
                'critical' if should_suspend else
                'warning' if should_alert else
                'normal'
            )
        }

    def generate_cost_report(self, output_file: str = None) -> dict:
        """Generate comprehensive cost report."""
        report_data = self.check_budget_status()

        # Add timestamp and additional metadata
        report_data.update({
            'timestamp': datetime.now().isoformat(),
            'report_type': 'deployment_cost_monitoring',
            'budget_limit': self.budget_limit,
            'environment': os.getenv('APP_ENV', 'development')
        })

        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"📄 Cost report saved to {output_file}")

        return report_data

    def analyze_cost_per_run(self) -> dict:
        """Analyze cost efficiency per workflow run."""
        status = self.budget_manager.get_budget_status()

        analysis = {
            'total_spent': status['spent'],
            'questions_processed': status.get('questions_processed', 0),
            'average_cost_per_question': status.get('average_cost_per_question', 0),
            'estimated_questions_remaining': status.get('estimated_questions_remaining', 0)
        }

        # Calculate efficiency metrics
        if analysis['questions_processed'] > 0:
            analysis['cost_efficiency'] = analysis['total_spent'] / analysis['questions_processed']

            # Estimate remaining capacity
            remaining_budget = self.budget_limit - status['spent']
            if analysis['cost_efficiency'] > 0:
                analysis['estimated_remaining_questions'] = int(
                    remaining_budget / analysis['cost_efficiency']
                )
            else:
                analysis['estimated_remaining_questions'] = 0

            # Calculate recommended frequency
            days_remaining = 30  # Assume 30 days left in tournament
            if analysis['estimated_remaining_questions'] > 0:
                questions_per_day = analysis['estimated_remaining_questions'] / days_remaining
                analysis['recommended_frequency_hours'] = max(
                    1, int(24 / questions_per_day)
                ) if questions_per_day > 0 else 24
            else:
                analysis['recommended_frequency_hours'] = 24
        else:
            analysis.update({
                'cost_efficiency': 0,
                'estimated_remaining_questions': 0,
                'recommended_frequency_hours': 4
            })

        return analysis

    def check_workflow_suspension_needed(self) -> bool:
        """Check if workflows should be suspended due to budget constraints."""
        status = self.budget_manager.get_budget_status()
        return status['utilization'] >= 0.98

    def generate_workflow_recommendations(self) -> dict:
        """Generate specific workflow management recommendations."""
        status = self.budget_manager.get_budget_status()
        cost_analysis = self.analyze_cost_per_run()

        recommendations = {
            'current_status': status['status'],
            'utilization': status['utilization'],
            'actions': []
        }

        if status['utilization'] >= 0.98:
            recommendations['actions'].extend([
                'CRITICAL: Suspend all automated workflows immediately',
                'Switch to manual forecasting for critical deadlines only',
                'Review cost optimization opportunities',
                'Consider emergency budget increase if tournament is critical'
            ])
        elif status['utilization'] >= 0.95:
            recommendations['actions'].extend([
                'WARNING: Enable emergency mode with minimal forecasting',
                'Reduce workflow frequency to maximum intervals',
                'Monitor budget utilization hourly',
                'Prepare for potential workflow suspension'
            ])
        elif status['utilization'] >= 0.80:
            recommendations['actions'].extend([
                'CAUTION: Enable conservative mode',
                'Use GPT-4o-mini for all tasks where possible',
                'Reduce forecasting frequency if needed',
                'Monitor budget utilization closely'
            ])
        else:
            recommendations['actions'].extend([
                'NORMAL: Continue current operation',
                'Monitor budget utilization regularly',
                'Optimize prompts for token efficiency'
            ])

        # Add frequency recommendations
        if cost_analysis['recommended_frequency_hours'] != 4:
            recommendations['actions'].append(
                f"Consider adjusting frequency to every {cost_analysis['recommended_frequency_hours']} hours"
            )

        return recommendations

    def print_status_report(self):
        """Print a comprehensive status report to console."""
        report = self.check_budget_status()
        cost_analysis = self.analyze_cost_per_run()
        workflow_recs = self.generate_workflow_recommendations()

        print("💰 DEPLOYMENT COST MONITORING REPORT")
        print("=" * 60)
        print(f"🕐 Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"💰 Total Budget: ${self.budget_limit:.2f}")
        print(f"💸 Amount Spent: ${report['budget_status']['spent']:.2f}")
        print(f"💵 Remaining: ${report['budget_status']['remaining']:.2f}")
        print(f"📊 Utilization: {report['budget_status']['utilization']:.1%}")
        print(f"🎯 Status Level: {report['budget_status']['status']}")
        print(f"🚨 Alert Level: {report['alert_level']}")
        print()

        print("📊 COST EFFICIENCY ANALYSIS")
        print("-" * 40)
        print(f"📈 Questions Processed: {cost_analysis['questions_processed']}")
        print(f"💲 Avg Cost/Question: ${cost_analysis['average_cost_per_question']:.3f}")
        print(f"🔮 Est. Questions Remaining: {cost_analysis['estimated_remaining_questions']}")
        print(f"⏰ Recommended Frequency: Every {cost_analysis['recommended_frequency_hours']} hours")
        print()

        print("🔧 WORKFLOW RECOMMENDATIONS")
        print("-" * 40)
        for action in workflow_recs['actions']:
            print(f"  • {action}")
        print()

        if report['actions']['should_suspend_workflows']:
            print("🚨 CRITICAL ACTION REQUIRED:")
            print("  Workflows should be suspended immediately!")
            print("  Run: python scripts/deployment_cost_monitor.py --suspend-workflows")
        elif report['actions']['should_send_alerts']:
            print("⚠️ WARNING:")
            print("  High budget utilization detected - monitor closely")

        print()
        print("📋 DETAILED RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Monitor deployment costs and budget utilization")
    parser.add_argument('--budget-limit', type=float, help='Budget limit override')
    parser.add_argument('--output-file', help='Save report to JSON file')
    parser.add_argument('--check-suspension', action='store_true',
                       help='Check if workflows should be suspended')
    parser.add_argument('--cost-analysis', action='store_true',
                       help='Show detailed cost analysis')
    parser.add_argument('--workflow-recommendations', action='store_true',
                       help='Show workflow management recommendations')
    parser.add_argument('--json-output', action='store_true',
                       help='Output results in JSON format')

    args = parser.parse_args()

    # Initialize monitor
    monitor = DeploymentCostMonitor(budget_limit=args.budget_limit)

    if args.check_suspension:
        should_suspend = monitor.check_workflow_suspension_needed()
        if args.json_output:
            print(json.dumps({'should_suspend_workflows': should_suspend}))
        else:
            print(f"Should suspend workflows: {should_suspend}")
        sys.exit(1 if should_suspend else 0)

    elif args.cost_analysis:
        analysis = monitor.analyze_cost_per_run()
        if args.json_output:
            print(json.dumps(analysis, indent=2))
        else:
            print("📊 COST ANALYSIS")
            print(f"Total Spent: ${analysis['total_spent']:.2f}")
            print(f"Questions Processed: {analysis['questions_processed']}")
            print(f"Avg Cost/Question: ${analysis['average_cost_per_question']:.3f}")
            print(f"Est. Questions Remaining: {analysis['estimated_remaining_questions']}")
            print(f"Recommended Frequency: Every {analysis['recommended_frequency_hours']} hours")

    elif args.workflow_recommendations:
        recs = monitor.generate_workflow_recommendations()
        if args.json_output:
            print(json.dumps(recs, indent=2))
        else:
            print("🔧 WORKFLOW RECOMMENDATIONS")
            print(f"Status: {recs['current_status']}")
            print(f"Utilization: {recs['utilization']:.1%}")
            print("Actions:")
            for action in recs['actions']:
                print(f"  • {action}")

    else:
        # Default: show full status report
        if args.json_output:
            report = monitor.generate_cost_report(args.output_file)
            print(json.dumps(report, indent=2))
        else:
            monitor.print_status_report()
            if args.output_file:
                monitor.generate_cost_report(args.output_file)


if __name__ == "__main__":
    main()
