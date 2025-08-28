"""
Real-time model selection effectiveness monitoring and cost tracking.
Tracks model routing decisions, cost per question, and quality metrics.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ModelSelectionRecord:
    """Record of a model selection decision and its outcome."""
    timestamp: datetime
    question_id: str
    task_type: str
    selected_model: str
    selected_tier: str
    routing_rationale: str
    estimated_cost: float
    actual_cost: Optional[float] = None
    execution_time: Optional[float] = None
    quality_score: Optional[float] = None
    success: bool = True
    fallback_used: bool = False
    operation_mode: str = "normal"
    budget_remaining: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelSelectionRecord':
        """Create from dictionary for JSON deserialization."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class CostBreakdown:
    """Cost breakdown by model tier and task type."""
    total_cost: float
    question_count: int
    avg_cost_per_question: float
    by_tier: Dict[str, Dict[str, float]]
    by_task_type: Dict[str, Dict[str, float]]
    by_operation_mode: Dict[str, Dict[str, float]]


@dataclass
class QualityMetrics:
    """Quality metrics for model performance."""
    avg_quality_score: float
    success_rate: float
    fallback_rate: float
    avg_execution_time: float
    quality_by_tier: Dict[str, float]
    quality_by_task: Dict[str, float]


@dataclass
class TournamentCompetitivenessIndicator:
    """Tournament competitiveness indicators and alerts."""
    cost_efficiency_score: float  # Questions per dollar
    quality_efficiency_score: float  # Quality per dollar
    budget_utilization_rate: float
    projected_questions_remaining: int
    competitiveness_level: str  # "excellent", "good", "concerning", "critical"
    recommendations: List[str]


class ModelPerformanceTracker:
    """Tracks model selection effectiveness and cost performance."""

    def __init__(self):
        """Initialize model performance tracker."""
        self.selection_records: List[ModelSelectionRecord] = []
        self.cost_history = deque(maxlen=1000)  # Last 1000 cost records

        # Data persistence
        self.data_file = Path("logs/model_performance.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        # Performance thresholds
        self.quality_threshold = 0.7  # Minimum acceptable quality score
        self.cost_efficiency_threshold = 50  # Questions per dollar
        self.fallback_rate_threshold = 0.1  # 10% fallback rate threshold

        self._load_existing_data()
        logger.info(f"Model performance tracker initialized with {len(self.selection_records)} records")

    def record_model_selection(
        self,
        question_id: str,
        task_type: str,
        selected_model: str,
        selected_tier: str,
        routing_rationale: str,
        estimated_cost: float,
        operation_mode: str = "normal",
        budget_remaining: Optional[float] = None
    ) -> ModelSelectionRecord:
        """Record a model selection decision."""
        record = ModelSelectionRecord(
            timestamp=datetime.now(),
            question_id=question_id,
            task_type=task_type,
            selected_model=selected_model,
            selected_tier=selected_tier,
            routing_rationale=routing_rationale,
            estimated_cost=estimated_cost,
            operation_mode=operation_mode,
            budget_remaining=budget_remaining
        )

        self.selection_records.append(record)

        # Save data periodically
        if len(self.selection_records) % 10 == 0:
            self._save_data()

        logger.debug(f"Recorded model selection for {question_id}: {selected_model} ({selected_tier})")
        return record

    def update_selection_outcome(
        self,
        question_id: str,
        actual_cost: float,
        execution_time: float,
        quality_score: Optional[float] = None,
        success: bool = True,
        fallback_used: bool = False
    ) -> bool:
        """Update model selection record with actual outcome."""
        # Find the most recent record for this question
        record = None
        for r in reversed(self.selection_records):
            if r.question_id == question_id and r.actual_cost is None:
                record = r
                break

        if not record:
            logger.warning(f"No pending model selection record found for question {question_id}")
            return False

        # Update with actual outcome
        record.actual_cost = actual_cost
        record.execution_time = execution_time
        record.quality_score = quality_score
        record.success = success
        record.fallback_used = fallback_used

        # Add to cost history for trend analysis
        self.cost_history.append({
            'timestamp': datetime.now().isoformat(),
            'question_id': question_id,
            'cost': actual_cost,
            'tier': record.selected_tier,
            'task_type': record.task_type,
            'operation_mode': record.operation_mode
        })

        self._save_data()

        logger.debug(f"Updated selection outcome for {question_id}: "
                    f"cost=${actual_cost:.4f}, time={execution_time:.2f}s, success={success}")
        return True

    def get_cost_breakdown(self, hours: int = 24) -> CostBreakdown:
        """Get detailed cost breakdown for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter recent records with actual costs
        recent_records = [
            r for r in self.selection_records
            if r.timestamp >= cutoff_time and r.actual_cost is not None
        ]

        if not recent_records:
            return CostBreakdown(
                total_cost=0.0,
                question_count=0,
                avg_cost_per_question=0.0,
                by_tier={},
                by_task_type={},
                by_operation_mode={}
            )

        total_cost = sum(r.actual_cost for r in recent_records)
        question_count = len(recent_records)
        avg_cost = total_cost / question_count if question_count > 0 else 0.0

        # Breakdown by tier
        by_tier = defaultdict(lambda: {'cost': 0.0, 'count': 0})
        for record in recent_records:
            by_tier[record.selected_tier]['cost'] += record.actual_cost
            by_tier[record.selected_tier]['count'] += 1

        # Add averages
        for tier_data in by_tier.values():
            tier_data['avg_cost'] = tier_data['cost'] / tier_data['count']

        # Breakdown by task type
        by_task_type = defaultdict(lambda: {'cost': 0.0, 'count': 0})
        for record in recent_records:
            by_task_type[record.task_type]['cost'] += record.actual_cost
            by_task_type[record.task_type]['count'] += 1

        # Add averages
        for task_data in by_task_type.values():
            task_data['avg_cost'] = task_data['cost'] / task_data['count']

        # Breakdown by operation mode
        by_operation_mode = defaultdict(lambda: {'cost': 0.0, 'count': 0})
        for record in recent_records:
            by_operation_mode[record.operation_mode]['cost'] += record.actual_cost
            by_operation_mode[record.operation_mode]['count'] += 1

        # Add averages
        for mode_data in by_operation_mode.values():
            mode_data['avg_cost'] = mode_data['cost'] / mode_data['count']

        return CostBreakdown(
            total_cost=total_cost,
            question_count=question_count,
            avg_cost_per_question=avg_cost,
            by_tier=dict(by_tier),
            by_task_type=dict(by_task_type),
            by_operation_mode=dict(by_operation_mode)
        )

    def get_quality_metrics(self, hours: int = 24) -> QualityMetrics:
        """Get quality metrics for model performance."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter recent records with quality scores
        recent_records = [
            r for r in self.selection_records
            if r.timestamp >= cutoff_time and r.actual_cost is not None
        ]

        if not recent_records:
            return QualityMetrics(
                avg_quality_score=0.0,
                success_rate=0.0,
                fallback_rate=0.0,
                avg_execution_time=0.0,
                quality_by_tier={},
                quality_by_task={}
            )

        # Calculate overall metrics
        quality_scores = [r.quality_score for r in recent_records if r.quality_score is not None]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0

        success_rate = sum(1 for r in recent_records if r.success) / len(recent_records)
        fallback_rate = sum(1 for r in recent_records if r.fallback_used) / len(recent_records)

        execution_times = [r.execution_time for r in recent_records if r.execution_time is not None]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0

        # Quality by tier
        quality_by_tier = {}
        tier_groups = defaultdict(list)
        for record in recent_records:
            if record.quality_score is not None:
                tier_groups[record.selected_tier].append(record.quality_score)

        for tier, scores in tier_groups.items():
            quality_by_tier[tier] = statistics.mean(scores)

        # Quality by task type
        quality_by_task = {}
        task_groups = defaultdict(list)
        for record in recent_records:
            if record.quality_score is not None:
                task_groups[record.task_type].append(record.quality_score)

        for task, scores in task_groups.items():
            quality_by_task[task] = statistics.mean(scores)

        return QualityMetrics(
            avg_quality_score=avg_quality,
            success_rate=success_rate,
            fallback_rate=fallback_rate,
            avg_execution_time=avg_execution_time,
            quality_by_tier=quality_by_tier,
            quality_by_task=quality_by_task
        )

    def get_tournament_competitiveness_indicators(
        self,
        total_budget: float = 100.0,
        hours: int = 24
    ) -> TournamentCompetitivenessIndicator:
        """Get tournament competitiveness indicators and alerts."""
        cost_breakdown = self.get_cost_breakdown(hours)
        quality_metrics = self.get_quality_metrics(hours)

        # Calculate cost efficiency (questions per dollar)
        cost_efficiency = (cost_breakdown.question_count / max(cost_breakdown.total_cost, 0.001))

        # Calculate quality efficiency (quality per dollar)
        quality_efficiency = (quality_metrics.avg_quality_score / max(cost_breakdown.avg_cost_per_question, 0.001))

        # Calculate budget utilization rate
        budget_used = cost_breakdown.total_cost
        budget_utilization_rate = (budget_used / total_budget) * 100

        # Project remaining questions based on current rate
        if cost_breakdown.avg_cost_per_question > 0:
            remaining_budget = total_budget - budget_used
            projected_questions = int(remaining_budget / cost_breakdown.avg_cost_per_question)
        else:
            projected_questions = 0

        # Determine competitiveness level
        competitiveness_level = self._assess_competitiveness_level(
            cost_efficiency, quality_efficiency, budget_utilization_rate, quality_metrics
        )

        # Generate recommendations
        recommendations = self._generate_competitiveness_recommendations(
            cost_efficiency, quality_efficiency, budget_utilization_rate, quality_metrics
        )

        return TournamentCompetitivenessIndicator(
            cost_efficiency_score=cost_efficiency,
            quality_efficiency_score=quality_efficiency,
            budget_utilization_rate=budget_utilization_rate,
            projected_questions_remaining=projected_questions,
            competitiveness_level=competitiveness_level,
            recommendations=recommendations
        )

    def _assess_competitiveness_level(
        self,
        cost_efficiency: float,
        quality_efficiency: float,
        budget_utilization_rate: float,
        quality_metrics: QualityMetrics
    ) -> str:
        """Assess overall competitiveness level."""
        score = 0

        # Cost efficiency scoring
        if cost_efficiency >= 100:  # 100+ questions per dollar
            score += 3
        elif cost_efficiency >= 50:  # 50+ questions per dollar
            score += 2
        elif cost_efficiency >= 20:  # 20+ questions per dollar
            score += 1

        # Quality scoring
        if quality_metrics.avg_quality_score >= 0.8:
            score += 3
        elif quality_metrics.avg_quality_score >= 0.7:
            score += 2
        elif quality_metrics.avg_quality_score >= 0.6:
            score += 1

        # Success rate scoring
        if quality_metrics.success_rate >= 0.95:
            score += 2
        elif quality_metrics.success_rate >= 0.9:
            score += 1

        # Budget utilization penalty
        if budget_utilization_rate > 90:
            score -= 2
        elif budget_utilization_rate > 80:
            score -= 1

        # Determine level
        if score >= 7:
            return "excellent"
        elif score >= 5:
            return "good"
        elif score >= 3:
            return "concerning"
        else:
            return "critical"

    def _generate_competitiveness_recommendations(
        self,
        cost_efficiency: float,
        quality_efficiency: float,
        budget_utilization_rate: float,
        quality_metrics: QualityMetrics
    ) -> List[str]:
        """Generate actionable recommendations for improving competitiveness."""
        recommendations = []

        # Cost efficiency recommendations
        if cost_efficiency < 50:
            recommendations.append("Switch to more cost-efficient models (GPT-5 nano/mini) for non-critical tasks")

        if cost_efficiency < 20:
            recommendations.append("URGENT: Enable emergency mode to preserve budget")

        # Quality recommendations
        if quality_metrics.avg_quality_score < 0.7:
            recommendations.append("Review prompt engineering and anti-slop directives")

        if quality_metrics.fallback_rate > 0.2:
            recommendations.append("High fallback rate detected - check model availability")

        # Budget recommendations
        if budget_utilization_rate > 85:
            recommendations.append("Budget critical - switch to conservative/emergency mode")
        elif budget_utilization_rate > 75:
            recommendations.append("Budget warning - consider conservative mode")

        # Success rate recommendations
        if quality_metrics.success_rate < 0.9:
            recommendations.append("Low success rate - investigate API failures and error handling")

        # Execution time recommendations
        if quality_metrics.avg_execution_time > 60:
            recommendations.append("High execution times - consider timeout optimization")

        return recommendations

    def get_model_effectiveness_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get model effectiveness trends over time."""
        cutoff_time = datetime.now() - timedelta(days=days)

        # Filter records for trend analysis
        records = [
            r for r in self.selection_records
            if r.timestamp >= cutoff_time and r.actual_cost is not None
        ]

        if len(records) < 10:
            return {"insufficient_data": True}

        # Group by day for trend analysis
        daily_metrics = defaultdict(lambda: {
            'cost': 0.0, 'count': 0, 'quality_scores': [], 'success_count': 0
        })

        for record in records:
            day_key = record.timestamp.date().isoformat()
            daily_metrics[day_key]['cost'] += record.actual_cost
            daily_metrics[day_key]['count'] += 1
            if record.quality_score is not None:
                daily_metrics[day_key]['quality_scores'].append(record.quality_score)
            if record.success:
                daily_metrics[day_key]['success_count'] += 1

        # Calculate daily averages
        trend_data = {}
        for day, metrics in daily_metrics.items():
            avg_cost = metrics['cost'] / metrics['count']
            avg_quality = statistics.mean(metrics['quality_scores']) if metrics['quality_scores'] else 0.0
            success_rate = metrics['success_count'] / metrics['count']

            trend_data[day] = {
                'avg_cost_per_question': avg_cost,
                'avg_quality_score': avg_quality,
                'success_rate': success_rate,
                'question_count': metrics['count'],
                'cost_efficiency': metrics['count'] / metrics['cost'] if metrics['cost'] > 0 else 0
            }

        return {
            "daily_trends": trend_data,
            "trend_analysis": self._analyze_trends(trend_data)
        }

    def _analyze_trends(self, trend_data: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Analyze trends in the data."""
        if len(trend_data) < 3:
            return {"insufficient_data": "Need at least 3 days of data for trend analysis"}

        # Sort by date
        sorted_days = sorted(trend_data.keys())

        # Analyze cost trend
        costs = [trend_data[day]['avg_cost_per_question'] for day in sorted_days]
        cost_trend = "stable"
        if len(costs) >= 3:
            recent_avg = statistics.mean(costs[-3:])
            older_avg = statistics.mean(costs[:3])
            if recent_avg > older_avg * 1.1:
                cost_trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                cost_trend = "decreasing"

        # Analyze quality trend
        qualities = [trend_data[day]['avg_quality_score'] for day in sorted_days if trend_data[day]['avg_quality_score'] > 0]
        quality_trend = "stable"
        if len(qualities) >= 3:
            recent_avg = statistics.mean(qualities[-3:])
            older_avg = statistics.mean(qualities[:3])
            if recent_avg > older_avg * 1.05:
                quality_trend = "improving"
            elif recent_avg < older_avg * 0.95:
                quality_trend = "declining"

        # Analyze efficiency trend
        efficiencies = [trend_data[day]['cost_efficiency'] for day in sorted_days]
        efficiency_trend = "stable"
        if len(efficiencies) >= 3:
            recent_avg = statistics.mean(efficiencies[-3:])
            older_avg = statistics.mean(efficiencies[:3])
            if recent_avg > older_avg * 1.1:
                efficiency_trend = "improving"
            elif recent_avg < older_avg * 0.9:
                efficiency_trend = "declining"

        return {
            "cost_trend": cost_trend,
            "quality_trend": quality_trend,
            "efficiency_trend": efficiency_trend
        }

    def _save_data(self):
        """Save performance tracking data to file."""
        try:
            data = {
                "selection_records": [record.to_dict() for record in self.selection_records],
                "cost_history": list(self.cost_history),
                "last_updated": datetime.now().isoformat()
            }

            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save model performance data: {e}")

    def _load_existing_data(self):
        """Load existing performance data if available."""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)

                # Load selection records
                records_data = data.get("selection_records", [])
                self.selection_records = [
                    ModelSelectionRecord.from_dict(record) for record in records_data
                ]

                # Load cost history
                cost_history = data.get("cost_history", [])
                self.cost_history.extend(cost_history)

                logger.info(f"Loaded {len(self.selection_records)} model selection records")

        except Exception as e:
            logger.warning(f"Failed to load existing model performance data: {e}")

    def log_performance_summary(self):
        """Log comprehensive performance summary."""
        cost_breakdown = self.get_cost_breakdown(24)
        quality_metrics = self.get_quality_metrics(24)
        competitiveness = self.get_tournament_competitiveness_indicators()

        logger.info("=== Model Performance Summary (24h) ===")
        logger.info(f"Questions Processed: {cost_breakdown.question_count}")
        logger.info(f"Total Cost: ${cost_breakdown.total_cost:.4f}")
        logger.info(f"Avg Cost per Question: ${cost_breakdown.avg_cost_per_question:.4f}")
        logger.info(f"Cost Efficiency: {competitiveness.cost_efficiency_score:.1f} questions/$")

        logger.info("--- Quality Metrics ---")
        logger.info(f"Avg Quality Score: {quality_metrics.avg_quality_score:.3f}")
        logger.info(f"Success Rate: {quality_metrics.success_rate:.1%}")
        logger.info(f"Fallback Rate: {quality_metrics.fallback_rate:.1%}")
        logger.info(f"Avg Execution Time: {quality_metrics.avg_execution_time:.2f}s")

        logger.info("--- Tournament Competitiveness ---")
        logger.info(f"Competitiveness Level: {competitiveness.competitiveness_level.upper()}")
        logger.info(f"Budget Utilization: {competitiveness.budget_utilization_rate:.1f}%")
        logger.info(f"Projected Questions Remaining: {competitiveness.projected_questions_remaining}")

        if competitiveness.recommendations:
            logger.info("--- Recommendations ---")
            for i, rec in enumerate(competitiveness.recommendations, 1):
                logger.info(f"{i}. {rec}")


# Global instance
model_performance_tracker = ModelPerformanceTracker()
