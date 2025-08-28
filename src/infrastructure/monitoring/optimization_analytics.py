"""
Optimization analytics and strategic recommendations for tournament performance.
Analyzes cost-effectiveness, performance correlations, and provides actionable insights.
"""

import logging
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

from .model_performance_tracker import ModelPerformanceTracker, model_performance_tracker

logger = logging.getLogger(__name__)


@dataclass
class CostEffectivenessAnalysis:
    """Analysis of cost-effectiveness for model routing decisions."""
    overall_efficiency: float  # Questions per dollar
    tier_efficiency: Dict[str, float]  # Efficiency by model tier
    task_efficiency: Dict[str, float]  # Efficiency by task type
    mode_efficiency: Dict[str, float]  # Efficiency by operation mode
    optimal_routing_suggestions: List[str]
    cost_savings_potential: float  # Potential savings in dollars


@dataclass
class PerformanceCorrelationAnalysis:
    """Analysis of correlations between cost and quality metrics."""
    cost_quality_correlation: float  # Correlation between cost and quality
    tier_quality_analysis: Dict[str, Dict[str, float]]  # Quality analysis by tier
    diminishing_returns_threshold: Optional[float]  # Cost point where returns diminish
    sweet_spot_recommendations: List[str]
    quality_cost_tradeoffs: Dict[str, Any]


@dataclass
class TournamentPhaseStrategy:
    """Strategic recommendations based on tournament phase."""
    phase: str  # "early", "middle", "late", "final"
    budget_allocation_strategy: Dict[str, float]  # Percentage allocation by tier
    routing_adjustments: List[str]
    risk_tolerance: str  # "aggressive", "balanced", "conservative"
    priority_tasks: List[str]
    emergency_thresholds: Dict[str, float]


@dataclass
class BudgetOptimizationSuggestion:
    """Budget allocation optimization suggestions."""
    current_allocation: Dict[str, float]  # Current spending by tier
    optimal_allocation: Dict[str, float]  # Recommended spending by tier
    potential_savings: float
    additional_questions_possible: int
    implementation_steps: List[str]
    risk_assessment: str


class OptimizationAnalytics:
    """Advanced analytics for tournament optimization and strategic recommendations."""

    def __init__(self, performance_tracker: ModelPerformanceTracker = None):
        """Initialize optimization analytics."""
        self.performance_tracker = performance_tracker or model_performance_tracker

        # Analysis parameters
        self.min_samples_for_analysis = 20
        self.correlation_significance_threshold = 0.3
        self.efficiency_improvement_threshold = 0.1  # 10% improvement threshold

        # Tournament phase detection parameters
        self.early_phase_threshold = 0.25  # 0-25% budget used
        self.middle_phase_threshold = 0.60  # 25-60% budget used
        self.late_phase_threshold = 0.85   # 60-85% budget used
        # final phase is 85-100%

    def analyze_cost_effectiveness(self, hours: int = 24) -> CostEffectivenessAnalysis:
        """Analyze cost-effectiveness of model routing decisions."""
        cost_breakdown = self.performance_tracker.get_cost_breakdown(hours)

        if cost_breakdown.question_count == 0:
            return CostEffectivenessAnalysis(
                overall_efficiency=0.0,
                tier_efficiency={},
                task_efficiency={},
                mode_efficiency={},
                optimal_routing_suggestions=[],
                cost_savings_potential=0.0
            )

        # Overall efficiency (questions per dollar)
        overall_efficiency = cost_breakdown.question_count / max(cost_breakdown.total_cost, 0.001)

        # Efficiency by tier
        tier_efficiency = {}
        for tier, data in cost_breakdown.by_tier.items():
            if data['cost'] > 0:
                tier_efficiency[tier] = data['count'] / data['cost']

        # Efficiency by task type
        task_efficiency = {}
        for task, data in cost_breakdown.by_task_type.items():
            if data['cost'] > 0:
                task_efficiency[task] = data['count'] / data['cost']

        # Efficiency by operation mode
        mode_efficiency = {}
        for mode, data in cost_breakdown.by_operation_mode.items():
            if data['cost'] > 0:
                mode_efficiency[mode] = data['count'] / data['cost']

        # Generate optimization suggestions
        optimal_routing_suggestions = self._generate_routing_optimization_suggestions(
            tier_efficiency, task_efficiency, mode_efficiency
        )

        # Calculate potential cost savings
        cost_savings_potential = self._calculate_cost_savings_potential(
            cost_breakdown, tier_efficiency, task_efficiency
        )

        return CostEffectivenessAnalysis(
            overall_efficiency=overall_efficiency,
            tier_efficiency=tier_efficiency,
            task_efficiency=task_efficiency,
            mode_efficiency=mode_efficiency,
            optimal_routing_suggestions=optimal_routing_suggestions,
            cost_savings_potential=cost_savings_potential
        )

    def analyze_performance_correlations(self, hours: int = 24) -> PerformanceCorrelationAnalysis:
        """Analyze correlations between cost and quality metrics."""
        # Get recent records with both cost and quality data
        cutoff_time = datetime.now() - timedelta(hours=hours)
        records = [
            r for r in self.performance_tracker.selection_records
            if (r.timestamp >= cutoff_time and
                r.actual_cost is not None and
                r.quality_score is not None)
        ]

        if len(records) < self.min_samples_for_analysis:
            return PerformanceCorrelationAnalysis(
                cost_quality_correlation=0.0,
                tier_quality_analysis={},
                diminishing_returns_threshold=None,
                sweet_spot_recommendations=[],
                quality_cost_tradeoffs={}
            )

        # Calculate cost-quality correlation
        costs = [r.actual_cost for r in records]
        qualities = [r.quality_score for r in records]
        cost_quality_correlation = self._calculate_correlation(costs, qualities)

        # Analyze quality by tier
        tier_quality_analysis = self._analyze_tier_quality_relationships(records)

        # Find diminishing returns threshold
        diminishing_returns_threshold = self._find_diminishing_returns_threshold(records)

        # Generate sweet spot recommendations
        sweet_spot_recommendations = self._generate_sweet_spot_recommendations(
            records, tier_quality_analysis, diminishing_returns_threshold
        )

        # Analyze quality-cost tradeoffs
        quality_cost_tradeoffs = self._analyze_quality_cost_tradeoffs(records)

        return PerformanceCorrelationAnalysis(
            cost_quality_correlation=cost_quality_correlation,
            tier_quality_analysis=tier_quality_analysis,
            diminishing_returns_threshold=diminishing_returns_threshold,
            sweet_spot_recommendations=sweet_spot_recommendations,
            quality_cost_tradeoffs=quality_cost_tradeoffs
        )

    def generate_tournament_phase_strategy(
        self,
        budget_used_percentage: float,
        total_budget: float = 100.0,
        questions_processed: int = 0
    ) -> TournamentPhaseStrategy:
        """Generate strategic recommendations based on tournament phase."""
        # Determine tournament phase
        phase = self._determine_tournament_phase(budget_used_percentage)

        # Get phase-specific strategy
        if phase == "early":
            return self._generate_early_phase_strategy(budget_used_percentage, total_budget)
        elif phase == "middle":
            return self._generate_middle_phase_strategy(budget_used_percentage, total_budget)
        elif phase == "late":
            return self._generate_late_phase_strategy(budget_used_percentage, total_budget)
        else:  # final phase
            return self._generate_final_phase_strategy(budget_used_percentage, total_budget)

    def generate_budget_optimization_suggestions(
        self,
        total_budget: float = 100.0,
        hours: int = 24
    ) -> BudgetOptimizationSuggestion:
        """Generate budget allocation optimization suggestions."""
        cost_breakdown = self.performance_tracker.get_cost_breakdown(hours)
        cost_effectiveness = self.analyze_cost_effectiveness(hours)

        # Calculate current allocation percentages
        current_allocation = {}
        if cost_breakdown.total_cost > 0:
            for tier, data in cost_breakdown.by_tier.items():
                current_allocation[tier] = (data['cost'] / cost_breakdown.total_cost) * 100

        # Calculate optimal allocation based on efficiency
        optimal_allocation = self._calculate_optimal_allocation(
            cost_effectiveness.tier_efficiency, current_allocation
        )

        # Calculate potential savings and additional questions
        potential_savings = self._calculate_potential_savings(
            current_allocation, optimal_allocation, cost_breakdown.total_cost
        )

        additional_questions = self._calculate_additional_questions_possible(
            potential_savings, cost_breakdown.avg_cost_per_question
        )

        # Generate implementation steps
        implementation_steps = self._generate_implementation_steps(
            current_allocation, optimal_allocation
        )

        # Assess risk
        risk_assessment = self._assess_optimization_risk(
            current_allocation, optimal_allocation
        )

        return BudgetOptimizationSuggestion(
            current_allocation=current_allocation,
            optimal_allocation=optimal_allocation,
            potential_savings=potential_savings,
            additional_questions_possible=additional_questions,
            implementation_steps=implementation_steps,
            risk_assessment=risk_assessment
        )

    def _generate_routing_optimization_suggestions(
        self,
        tier_efficiency: Dict[str, float],
        task_efficiency: Dict[str, float],
        mode_efficiency: Dict[str, float]
    ) -> List[str]:
        """Generate routing optimization suggestions."""
        suggestions = []

        # Tier efficiency suggestions
        if tier_efficiency:
            most_efficient_tier = max(tier_efficiency.items(), key=lambda x: x[1])
            least_efficient_tier = min(tier_efficiency.items(), key=lambda x: x[1])

            efficiency_ratio = most_efficient_tier[1] / max(least_efficient_tier[1], 0.001)

            if efficiency_ratio > 2.0:  # Significant efficiency difference
                suggestions.append(
                    f"Increase usage of {most_efficient_tier[0]} tier "
                    f"({most_efficient_tier[1]:.1f} questions/$) vs {least_efficient_tier[0]} "
                    f"({least_efficient_tier[1]:.1f} questions/$)"
                )

        # Task efficiency suggestions
        if task_efficiency:
            sorted_tasks = sorted(task_efficiency.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_tasks) >= 2:
                best_task = sorted_tasks[0]
                worst_task = sorted_tasks[-1]

                if best_task[1] > worst_task[1] * 1.5:
                    suggestions.append(
                        f"Optimize {worst_task[0]} tasks - currently least efficient "
                        f"({worst_task[1]:.1f} questions/$)"
                    )

        # Operation mode suggestions
        if mode_efficiency:
            most_efficient_mode = max(mode_efficiency.items(), key=lambda x: x[1])
            suggestions.append(
                f"Most efficient operation mode: {most_efficient_mode[0]} "
                f"({most_efficient_mode[1]:.1f} questions/$)"
            )

        return suggestions

    def _calculate_cost_savings_potential(
        self,
        cost_breakdown,
        tier_efficiency: Dict[str, float],
        task_efficiency: Dict[str, float]
    ) -> float:
        """Calculate potential cost savings from optimization."""
        if not tier_efficiency or cost_breakdown.total_cost == 0:
            return 0.0

        # Find most efficient tier
        most_efficient_tier = max(tier_efficiency.items(), key=lambda x: x[1])[0]
        most_efficient_rate = tier_efficiency[most_efficient_tier]

        # Calculate savings if all questions used most efficient tier
        total_questions = cost_breakdown.question_count
        optimal_cost = total_questions / most_efficient_rate
        potential_savings = max(0, cost_breakdown.total_cost - optimal_cost)

        return potential_savings

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        try:
            return float(np.corrcoef(x, y)[0, 1])
        except:
            # Fallback calculation if numpy fails
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)
            sum_y2 = sum(yi * yi for yi in y)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5

            return numerator / denominator if denominator != 0 else 0.0

    def _analyze_tier_quality_relationships(self, records) -> Dict[str, Dict[str, float]]:
        """Analyze quality relationships by tier."""
        tier_analysis = {}

        # Group records by tier
        tier_groups = defaultdict(list)
        for record in records:
            tier_groups[record.selected_tier].append(record)

        for tier, tier_records in tier_groups.items():
            if len(tier_records) < 5:  # Need minimum samples
                continue

            costs = [r.actual_cost for r in tier_records]
            qualities = [r.quality_score for r in tier_records]

            tier_analysis[tier] = {
                'avg_cost': statistics.mean(costs),
                'avg_quality': statistics.mean(qualities),
                'cost_quality_correlation': self._calculate_correlation(costs, qualities),
                'quality_per_dollar': statistics.mean(qualities) / max(statistics.mean(costs), 0.001),
                'sample_count': len(tier_records)
            }

        return tier_analysis

    def _find_diminishing_returns_threshold(self, records) -> Optional[float]:
        """Find the cost threshold where quality returns diminish."""
        if len(records) < 20:
            return None

        # Sort records by cost
        sorted_records = sorted(records, key=lambda r: r.actual_cost)

        # Analyze quality improvement vs cost increase
        cost_buckets = []
        quality_buckets = []

        bucket_size = len(sorted_records) // 5  # 5 buckets

        for i in range(0, len(sorted_records), bucket_size):
            bucket = sorted_records[i:i + bucket_size]
            if len(bucket) >= 3:
                avg_cost = statistics.mean([r.actual_cost for r in bucket])
                avg_quality = statistics.mean([r.quality_score for r in bucket])
                cost_buckets.append(avg_cost)
                quality_buckets.append(avg_quality)

        # Find where quality improvement per cost unit drops significantly
        if len(cost_buckets) >= 3:
            improvements = []
            for i in range(1, len(cost_buckets)):
                cost_increase = cost_buckets[i] - cost_buckets[i-1]
                quality_increase = quality_buckets[i] - quality_buckets[i-1]

                if cost_increase > 0:
                    improvement_rate = quality_increase / cost_increase
                    improvements.append((cost_buckets[i], improvement_rate))

            # Find where improvement rate drops significantly
            if len(improvements) >= 2:
                for i in range(1, len(improvements)):
                    current_rate = improvements[i][1]
                    previous_rate = improvements[i-1][1]

                    if previous_rate > 0 and current_rate / previous_rate < 0.5:
                        return improvements[i][0]

        return None

    def _generate_sweet_spot_recommendations(
        self,
        records,
        tier_analysis: Dict[str, Dict[str, float]],
        diminishing_returns_threshold: Optional[float]
    ) -> List[str]:
        """Generate sweet spot recommendations."""
        recommendations = []

        # Find best quality per dollar tier
        if tier_analysis:
            best_tier = max(
                tier_analysis.items(),
                key=lambda x: x[1]['quality_per_dollar']
            )

            recommendations.append(
                f"Best quality per dollar: {best_tier[0]} tier "
                f"({best_tier[1]['quality_per_dollar']:.3f} quality/$)"
            )

        # Diminishing returns recommendation
        if diminishing_returns_threshold:
            recommendations.append(
                f"Diminishing returns threshold: ${diminishing_returns_threshold:.4f} per question"
            )

        # Correlation-based recommendations
        if tier_analysis:
            high_correlation_tiers = [
                tier for tier, data in tier_analysis.items()
                if data['cost_quality_correlation'] > self.correlation_significance_threshold
            ]

            if high_correlation_tiers:
                recommendations.append(
                    f"Strong cost-quality correlation in: {', '.join(high_correlation_tiers)}"
                )

        return recommendations

    def _analyze_quality_cost_tradeoffs(self, records) -> Dict[str, Any]:
        """Analyze quality-cost tradeoffs."""
        if len(records) < 10:
            return {"insufficient_data": True}

        # Categorize by cost levels
        costs = [r.actual_cost for r in records]
        cost_percentiles = {
            'low': np.percentile(costs, 33),
            'medium': np.percentile(costs, 67),
            'high': np.percentile(costs, 100)
        }

        cost_categories = {
            'low': [r for r in records if r.actual_cost <= cost_percentiles['low']],
            'medium': [r for r in records if cost_percentiles['low'] < r.actual_cost <= cost_percentiles['medium']],
            'high': [r for r in records if r.actual_cost > cost_percentiles['medium']]
        }

        tradeoff_analysis = {}
        for category, category_records in cost_categories.items():
            if category_records:
                avg_quality = statistics.mean([r.quality_score for r in category_records])
                avg_cost = statistics.mean([r.actual_cost for r in category_records])

                tradeoff_analysis[category] = {
                    'avg_cost': avg_cost,
                    'avg_quality': avg_quality,
                    'sample_count': len(category_records),
                    'quality_per_dollar': avg_quality / max(avg_cost, 0.001)
                }

        return tradeoff_analysis

    def _determine_tournament_phase(self, budget_used_percentage: float) -> str:
        """Determine current tournament phase."""
        if budget_used_percentage <= self.early_phase_threshold * 100:
            return "early"
        elif budget_used_percentage <= self.middle_phase_threshold * 100:
            return "middle"
        elif budget_used_percentage <= self.late_phase_threshold * 100:
            return "late"
        else:
            return "final"

    def _generate_early_phase_strategy(self, budget_used: float, total_budget: float) -> TournamentPhaseStrategy:
        """Generate strategy for early tournament phase."""
        return TournamentPhaseStrategy(
            phase="early",
            budget_allocation_strategy={
                "full": 40.0,  # Use premium models for quality establishment
                "mini": 45.0,  # Balanced approach
                "nano": 15.0   # Minimal usage
            },
            routing_adjustments=[
                "Prioritize quality over cost in early phase",
                "Use full tier for complex forecasting tasks",
                "Establish baseline performance metrics"
            ],
            risk_tolerance="aggressive",
            priority_tasks=["forecast", "research"],
            emergency_thresholds={"budget_utilization": 30.0}
        )

    def _generate_middle_phase_strategy(self, budget_used: float, total_budget: float) -> TournamentPhaseStrategy:
        """Generate strategy for middle tournament phase."""
        return TournamentPhaseStrategy(
            phase="middle",
            budget_allocation_strategy={
                "full": 30.0,  # Reduce premium usage
                "mini": 50.0,  # Increase balanced tier
                "nano": 20.0   # Moderate usage
            },
            routing_adjustments=[
                "Balance quality and cost efficiency",
                "Optimize based on performance data",
                "Increase mini tier usage for research"
            ],
            risk_tolerance="balanced",
            priority_tasks=["research", "forecast"],
            emergency_thresholds={"budget_utilization": 65.0}
        )

    def _generate_late_phase_strategy(self, budget_used: float, total_budget: float) -> TournamentPhaseStrategy:
        """Generate strategy for late tournament phase."""
        return TournamentPhaseStrategy(
            phase="late",
            budget_allocation_strategy={
                "full": 20.0,  # Minimal premium usage
                "mini": 50.0,  # Maintain balanced tier
                "nano": 30.0   # Increase efficient tier
            },
            routing_adjustments=[
                "Prioritize cost efficiency",
                "Reserve full tier for critical forecasts only",
                "Increase nano tier for validation tasks"
            ],
            risk_tolerance="conservative",
            priority_tasks=["forecast"],
            emergency_thresholds={"budget_utilization": 80.0}
        )

    def _generate_final_phase_strategy(self, budget_used: float, total_budget: float) -> TournamentPhaseStrategy:
        """Generate strategy for final tournament phase."""
        return TournamentPhaseStrategy(
            phase="final",
            budget_allocation_strategy={
                "full": 10.0,  # Emergency use only
                "mini": 30.0,  # Reduced usage
                "nano": 60.0   # Maximum efficiency
            },
            routing_adjustments=[
                "Maximum cost efficiency mode",
                "Use free models when possible",
                "Reserve budget for critical forecasts only"
            ],
            risk_tolerance="conservative",
            priority_tasks=["forecast"],
            emergency_thresholds={"budget_utilization": 95.0}
        )

    def _calculate_optimal_allocation(
        self,
        tier_efficiency: Dict[str, float],
        current_allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate optimal budget allocation based on efficiency."""
        if not tier_efficiency:
            return current_allocation

        # Weight allocation by efficiency
        total_efficiency = sum(tier_efficiency.values())
        optimal_allocation = {}

        for tier, efficiency in tier_efficiency.items():
            # Base allocation on efficiency, but cap extremes
            efficiency_weight = efficiency / total_efficiency
            optimal_percentage = efficiency_weight * 100

            # Apply constraints to prevent extreme allocations
            optimal_percentage = max(10.0, min(60.0, optimal_percentage))
            optimal_allocation[tier] = optimal_percentage

        # Normalize to 100%
        total_optimal = sum(optimal_allocation.values())
        if total_optimal > 0:
            for tier in optimal_allocation:
                optimal_allocation[tier] = (optimal_allocation[tier] / total_optimal) * 100

        return optimal_allocation

    def _calculate_potential_savings(
        self,
        current_allocation: Dict[str, float],
        optimal_allocation: Dict[str, float],
        total_cost: float
    ) -> float:
        """Calculate potential savings from optimization."""
        # This is a simplified calculation
        # In practice, would need more sophisticated modeling
        savings_factor = 0.0

        for tier in current_allocation:
            if tier in optimal_allocation:
                allocation_diff = optimal_allocation[tier] - current_allocation[tier]
                # Assume higher efficiency tiers save money
                if tier == "nano":
                    savings_factor += allocation_diff * 0.02  # 2% savings per % shift to nano
                elif tier == "mini":
                    savings_factor += allocation_diff * 0.01  # 1% savings per % shift to mini

        return max(0, total_cost * (savings_factor / 100))

    def _calculate_additional_questions_possible(
        self,
        potential_savings: float,
        avg_cost_per_question: float
    ) -> int:
        """Calculate additional questions possible with savings."""
        if avg_cost_per_question <= 0:
            return 0

        return int(potential_savings / avg_cost_per_question)

    def _generate_implementation_steps(
        self,
        current_allocation: Dict[str, float],
        optimal_allocation: Dict[str, float]
    ) -> List[str]:
        """Generate implementation steps for optimization."""
        steps = []

        for tier in optimal_allocation:
            if tier in current_allocation:
                current = current_allocation[tier]
                optimal = optimal_allocation[tier]
                diff = optimal - current

                if abs(diff) > 5.0:  # Significant change
                    if diff > 0:
                        steps.append(f"Increase {tier} tier usage by {diff:.1f}%")
                    else:
                        steps.append(f"Decrease {tier} tier usage by {abs(diff):.1f}%")

        if not steps:
            steps.append("Current allocation is near optimal")

        return steps

    def _assess_optimization_risk(
        self,
        current_allocation: Dict[str, float],
        optimal_allocation: Dict[str, float]
    ) -> str:
        """Assess risk of implementing optimization."""
        total_change = sum(
            abs(optimal_allocation.get(tier, 0) - current_allocation.get(tier, 0))
            for tier in set(list(current_allocation.keys()) + list(optimal_allocation.keys()))
        )

        if total_change < 20:
            return "low"
        elif total_change < 50:
            return "medium"
        else:
            return "high"

    def log_optimization_analysis(self):
        """Log comprehensive optimization analysis."""
        cost_effectiveness = self.analyze_cost_effectiveness(24)
        performance_correlations = self.analyze_performance_correlations(24)
        budget_optimization = self.generate_budget_optimization_suggestions()

        logger.info("=== Optimization Analysis (24h) ===")
        logger.info(f"Overall Efficiency: {cost_effectiveness.overall_efficiency:.1f} questions/$")

        if cost_effectiveness.tier_efficiency:
            logger.info("--- Tier Efficiency ---")
            for tier, efficiency in cost_effectiveness.tier_efficiency.items():
                logger.info(f"{tier.upper()}: {efficiency:.1f} questions/$")

        logger.info("--- Performance Correlations ---")
        logger.info(f"Cost-Quality Correlation: {performance_correlations.cost_quality_correlation:.3f}")

        if performance_correlations.diminishing_returns_threshold:
            logger.info(f"Diminishing Returns Threshold: ${performance_correlations.diminishing_returns_threshold:.4f}")

        logger.info("--- Budget Optimization ---")
        logger.info(f"Potential Savings: ${budget_optimization.potential_savings:.4f}")
        logger.info(f"Additional Questions Possible: {budget_optimization.additional_questions_possible}")

        if cost_effectiveness.optimal_routing_suggestions:
            logger.info("--- Optimization Suggestions ---")
            for i, suggestion in enumerate(cost_effectiveness.optimal_routing_suggestions, 1):
                logger.info(f"{i}. {suggestion}")


# Global instance
optimization_analytics = OptimizationAnalytics()
