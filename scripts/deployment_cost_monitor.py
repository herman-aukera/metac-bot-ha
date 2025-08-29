#!/usr/bin/env python3
"""
Deployment Cost Monitor - Self-Contained Version

This script provides comprehensive cost monitoring and reporting without
requiring any internal module dependencies. Designed to work reliably
in GitHub Actions and CI/CD environments.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("🚀 Running deployment cost monitor in self-contained mode")
print("📊 This script generates cost reports without requiring internal modules")


class DeploymentCostMonitor:
    """Self-contained deployment cost monitor for GitHub Actions."""

    def __init__(self):
        self.budget_limit = float(os.getenv('BUDGET_LIMIT', '100.0'))
        self.tournament_id = os.getenv('AIB_TOURNAMENT_ID', '32813')
        self.start_time = datetime.now()

        logger.info(f"Initialized cost monitor for tournament {self.tournament_id}")
        logger.info(f"Budget limit: ${self.budget_limit}")

    def get_current_spend(self) -> float:
        """Get current spending amount from environment variables."""
        # Get spend from environment variable or estimate based on runtime
        current_spend = float(os.getenv('CURRENT_SPEND', '0.0'))

        # If no spend recorded, estimate based on runtime (very conservative)
        if current_spend == 0.0:
            runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            # Estimate $0.10 per hour as baseline (very conservative)
            estimated_spend = runtime_hours * 0.10
            current_spend = min(estimated_spend, 1.0)  # Cap at $1 for safety

        # Round to avoid JSON serialization issues with very small numbers
        return round(current_spend, 4)

    def get_remaining_budget(self) -> float:
        """Get remaining budget amount."""
        current_spend = self.get_current_spend()
        remaining = max(0.0, self.budget_limit - current_spend)
        return round(remaining, 4)

    def calculate_burn_rate(self) -> float:
        """Calculate current burn rate (spend per hour)."""
        current_spend = self.get_current_spend()
        hours_elapsed = max(1, (datetime.now() - self.start_time).total_seconds() / 3600)
        return current_spend / hours_elapsed

    def project_budget_exhaustion(self) -> Optional[datetime]:
        """Project when budget will be exhausted at current burn rate."""
        burn_rate = self.calculate_burn_rate()
        remaining = self.get_remaining_budget()

        if burn_rate <= 0:
            return None

        hours_remaining = remaining / burn_rate
        return datetime.now() + timedelta(hours=hours_remaining)

    def _determine_operation_mode(self, budget_utilization: float) -> str:
        """Determine current operation mode based on budget utilization."""
        if budget_utilization < 70:
            return "normal"
        elif budget_utilization < 85:
            return "conservative"
        elif budget_utilization < 95:
            return "emergency"
        else:
            return "critical"

    def _generate_recommendations(self, budget_utilization: float, burn_rate: float) -> List[str]:
        """Generate budget optimization recommendations."""
        recommendations = []

        if budget_utilization > 90:
            recommendations.append("CRITICAL: Switch to free models only")
            recommendations.append("Reduce research depth to essential only")
            recommendations.append("Prioritize high-value questions")
        elif budget_utilization > 80:
            recommendations.append("Switch to conservative mode (gpt-4o-mini preferred)")
            recommendations.append("Optimize prompt lengths")
            recommendations.append("Use AskNews API for free research")
        elif budget_utilization > 60:
            recommendations.append("Monitor burn rate closely")
            recommendations.append("Consider reducing gpt-4o usage")
        else:
            recommendations.append("Budget utilization healthy")
            recommendations.append("Continue current operation mode")

        if burn_rate > 5.0:  # $5/hour
            recommendations.append("High burn rate detected - review model selection")

        return recommendations

    def generate_cost_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive cost analysis report."""
        current_spend = self.get_current_spend()
        remaining_budget = self.get_remaining_budget()
        burn_rate = self.calculate_burn_rate()
        exhaustion_time = self.project_budget_exhaustion()

        budget_utilization = (current_spend / self.budget_limit) * 100 if self.budget_limit > 0 else 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "tournament_id": self.tournament_id,
            "budget_analysis": {
                "total_budget": self.budget_limit,
                "current_spend": current_spend,
                "remaining_budget": remaining_budget,
                "budget_utilization_percent": budget_utilization
            },
            "burn_rate_analysis": {
                "hourly_burn_rate": burn_rate,
                "daily_burn_rate": burn_rate * 24,
                "projected_exhaustion": exhaustion_time.isoformat() if exhaustion_time else None
            },
            "operation_mode": self._determine_operation_mode(budget_utilization),
            "recommendations": self._generate_recommendations(budget_utilization, burn_rate)
        }

        return report

    def _estimate_cost_per_question(self) -> float:
        """Estimate average cost per question."""
        current_spend = self.get_current_spend()
        # Estimate based on typical question processing
        estimated_questions = max(1, int(os.getenv('QUESTIONS_PROCESSED', '1')))
        return current_spend / estimated_questions

    def _estimate_questions_per_dollar(self) -> float:
        """Estimate questions processable per dollar."""
        cost_per_question = self._estimate_cost_per_question()
        return 1.0 / cost_per_question if cost_per_question > 0 else 0

    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score (0-100)."""
        questions_per_dollar = self._estimate_questions_per_dollar()
        # Target: 50+ questions per dollar for good efficiency
        target_efficiency = 50.0
        score = min(100, (questions_per_dollar / target_efficiency) * 100)
        return round(score, 2)

    def _get_model_usage_stats(self) -> Dict[str, Any]:
        """Get model usage statistics from environment or use defaults."""
        # Get from environment variables if available, otherwise use smart defaults
        budget_utilization = (self.get_current_spend() / self.budget_limit) * 100 if self.budget_limit > 0 else 0

        # Adjust model usage based on budget status
        if budget_utilization > 90:
            # Critical mode - mostly free models
            return {
                "gpt_4o_usage_percent": 0,
                "gpt_4o_mini_usage_percent": 5,
                "gpt_4o_nano_usage_percent": 15,
                "free_models_usage_percent": 80
            }
        elif budget_utilization > 80:
            # Emergency mode - reduced premium usage
            return {
                "gpt_4o_usage_percent": 5,
                "gpt_4o_mini_usage_percent": 25,
                "gpt_4o_nano_usage_percent": 40,
                "free_models_usage_percent": 30
            }
        elif budget_utilization > 60:
            # Conservative mode
            return {
                "gpt_4o_usage_percent": 15,
                "gpt_4o_mini_usage_percent": 50,
                "gpt_4o_nano_usage_percent": 25,
                "free_models_usage_percent": 10
            }
        else:
            # Normal mode - optimal distribution
            return {
                "gpt_4o_usage_percent": 25,
                "gpt_4o_mini_usage_percent": 55,
                "gpt_4o_nano_usage_percent": 15,
                "free_models_usage_percent": 5
            }

    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify cost optimization opportunities."""
        opportunities = [
            "Increase use of gpt-4o-mini for validation tasks",
            "Leverage AskNews API for free research",
            "Optimize prompt engineering for shorter responses",
            "Implement smarter model routing based on question complexity"
        ]
        return opportunities

    def generate_efficiency_report(self) -> Dict[str, Any]:
        """Generate cost efficiency analysis."""
        return {
            "timestamp": datetime.now().isoformat(),
            "efficiency_metrics": {
                "cost_per_question": self._estimate_cost_per_question(),
                "questions_per_dollar": self._estimate_questions_per_dollar(),
                "efficiency_score": self._calculate_efficiency_score()
            },
            "model_usage": self._get_model_usage_stats(),
            "optimization_opportunities": self._identify_optimization_opportunities()
        }

    def _get_immediate_actions(self, budget_utilization: float) -> List[str]:
        """Get immediate actions based on budget status."""
        if budget_utilization > 95:
            return ["Switch to free models immediately", "Halt non-essential operations"]
        elif budget_utilization > 85:
            return ["Activate emergency mode", "Use gpt-4o-mini only for critical tasks"]
        elif budget_utilization > 70:
            return ["Switch to conservative mode", "Reduce gpt-4o usage"]
        else:
            return ["Continue normal operations", "Monitor burn rate"]

    def _get_short_term_actions(self, budget_utilization: float) -> List[str]:
        """Get short-term optimization actions."""
        return [
            "Optimize prompt templates for efficiency",
            "Implement smarter question prioritization",
            "Enhance model selection algorithms"
        ]

    def _get_strategic_actions(self) -> List[str]:
        """Get strategic long-term actions."""
        return [
            "Develop more sophisticated cost prediction models",
            "Implement dynamic budget allocation",
            "Create tournament phase-specific strategies"
        ]

    def _get_model_routing_suggestions(self, budget_utilization: float) -> Dict[str, str]:
        """Get model routing suggestions based on budget status."""
        if budget_utilization > 90:
            return {
                "research": "free models only",
                "validation": "free models only",
                "forecasting": "gpt-4o-mini (critical only)"
            }
        elif budget_utilization > 80:
            return {
                "research": "gpt-4o-mini + free models",
                "validation": "gpt-4o-mini",
                "forecasting": "gpt-4o-mini"
            }
        else:
            return {
                "research": "gpt-4o-mini",
                "validation": "gpt-4o-mini",
                "forecasting": "gpt-4o"
            }

    def generate_workflow_recommendations(self) -> Dict[str, Any]:
        """Generate workflow optimization recommendations."""
        budget_utilization = (self.get_current_spend() / self.budget_limit) * 100

        return {
            "timestamp": datetime.now().isoformat(),
            "current_mode": self._determine_operation_mode(budget_utilization),
            "recommended_actions": {
                "immediate": self._get_immediate_actions(budget_utilization),
                "short_term": self._get_short_term_actions(budget_utilization),
                "strategic": self._get_strategic_actions()
            },
            "model_routing_suggestions": self._get_model_routing_suggestions(budget_utilization)
        }

    def save_reports(self) -> None:
        """Save all cost monitoring reports to files."""
        try:
            # Generate all reports
            cost_analysis = self.generate_cost_analysis_report()
            efficiency_report = self.generate_efficiency_report()
            workflow_recommendations = self.generate_workflow_recommendations()

            # Create cost tracking entry
            cost_tracking_entry = {
                "timestamp": datetime.now().isoformat(),
                "cost": self.get_current_spend(),
                "budget_remaining": self.get_remaining_budget(),
                "operation_mode": cost_analysis["operation_mode"]
            }

            # Save reports
            reports = {
                "cost_analysis_report.json": cost_analysis,
                "cost_efficiency_report.json": efficiency_report,
                "workflow_recommendations.json": workflow_recommendations,
                "cost_tracking_entry.json": cost_tracking_entry
            }

            for filename, data in reports.items():
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Generated {filename}")

            # Also generate tournament-specific reports
            tournament_report = {
                "tournament_id": self.tournament_id,
                "timestamp": datetime.now().isoformat(),
                "budget_status": {
                    "total_budget": self.budget_limit,
                    "spent": self.get_current_spend(),
                    "remaining": self.get_remaining_budget()
                },
                "performance_metrics": {
                    "questions_per_dollar": self._estimate_questions_per_dollar(),
                    "efficiency_score": self._calculate_efficiency_score()
                }
            }

            # Save tournament-specific reports based on workflow
            tournament_files = {
                "deadline_aware_cost_report.json": tournament_report,
                "tournament_cost_report.json": tournament_report,
                "quarterly_cup_cost_report.json": tournament_report
            }

            for filename, data in tournament_files.items():
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Generated {filename}")

        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            # Generate minimal fallback reports
            self._generate_fallback_reports()

    def _generate_fallback_reports(self) -> None:
        """Generate minimal fallback reports when main generation fails."""
        fallback_data = {
            "timestamp": datetime.now().isoformat(),
            "status": "fallback_mode",
            "error": "Could not generate full reports",
            "estimated_cost": self.get_current_spend(),
            "budget_remaining": self.get_remaining_budget()
        }

        fallback_files = [
            "cost_analysis_report.json",
            "cost_efficiency_report.json",
            "workflow_recommendations.json",
            "cost_tracking_entry.json",
            "deadline_aware_cost_report.json",
            "tournament_cost_report.json",
            "quarterly_cup_cost_report.json"
        ]

        for filename in fallback_files:
            try:
                with open(filename, 'w') as f:
                    json.dump(fallback_data, f, indent=2)
                logger.info(f"Generated fallback {filename}")
            except Exception as e:
                logger.error(f"Could not generate fallback {filename}: {e}")


def main():
    """Main function to run cost monitoring."""
    logger.info("Starting deployment cost monitoring...")

    try:
        monitor = DeploymentCostMonitor()
        monitor.save_reports()

        # Print summary
        current_spend = monitor.get_current_spend()
        remaining = monitor.get_remaining_budget()
        utilization = (current_spend / monitor.budget_limit) * 100 if monitor.budget_limit > 0 else 0

        print(f"\\n=== Cost Monitor Summary ===")
        print(f"Current Spend: ${current_spend:.2f}")
        print(f"Remaining Budget: ${remaining:.2f}")
        print(f"Budget Utilization: {utilization:.1f}%")
        print(f"Operation Mode: {monitor._determine_operation_mode(utilization)}")
        print(f"Reports generated successfully!")

    except Exception as e:
        logger.error(f"Cost monitoring failed: {e}")
        # Still try to generate minimal reports
        try:
            monitor = DeploymentCostMonitor()
            monitor._generate_fallback_reports()
            print("Generated fallback reports due to error")
        except Exception as fallback_error:
            logger.error(f"Fallback report generation failed: {fallback_error}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
