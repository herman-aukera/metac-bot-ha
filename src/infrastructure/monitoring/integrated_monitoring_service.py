"""
Integrated monitoring service that combines all monitoring components.
Provides unified interface for performance tracking, analytics, and optimization.
"""

import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from .model_performance_tracker import ModelPerformanceTracker, model_performance_tracker
from .optimization_analytics import OptimizationAnalytics, optimization_analytics
from .performance_tracker import PerformanceTracker, performance_tracker
try:
    from ..config.cost_monitor import CostMonitor, cost_monitor
except ImportError:
    CostMonitor = None
    cost_monitor = None

logger = logging.getLogger(__name__)


@dataclass
class MonitoringAlert:
    """Unified monitoring alert."""
    timestamp: datetime
    alert_type: str  # "performance", "cost", "quality", "tournament"
    severity: str    # "info", "warning", "critical"
    component: str   # Source component
    message: str
    data: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ComprehensiveStatus:
    """Comprehensive system status."""
    timestamp: datetime
    overall_health: str  # "excellent", "good", "concerning", "critical"
    budget_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    optimization_recommendations: List[str]
    tournament_competitiveness: Dict[str, Any]
    active_alerts: List[MonitoringAlert]


class IntegratedMonitoringService:
    """Unified monitoring service for tournament performance optimization."""

    def __init__(
        self,
        model_tracker: ModelPerformanceTracker = None,
        analytics: OptimizationAnalytics = None,
        perf_tracker: PerformanceTracker = None,
        cost_monitor: CostMonitor = None
    ):
        """Initialize integrated monitoring service."""
        self.model_tracker = model_tracker or model_performance_tracker
        self.analytics = analytics or optimization_analytics
        self.perf_tracker = perf_tracker or performance_tracker
        self.cost_monitor = cost_monitor or cost_monitor

        # Monitoring configuration
        self.monitoring_interval = 60  # seconds
        self.alert_history: List[MonitoringAlert] = []
        self.max_alert_history = 1000

        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._is_running = False

    def start_monitoring(self):
        """Start the integrated monitoring service."""
        if self._is_running:
            logger.warning("Monitoring service is already running")
            return

        self._is_running = True
        self._stop_monitoring.clear()

        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="IntegratedMonitoring"
        )
        self._monitoring_thread.start()

        logger.info("Integrated monitoring service started")

    def stop_monitoring(self):
        """Stop the integrated monitoring service."""
        if not self._is_running:
            return

        self._is_running = False
        self._stop_monitoring.set()

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)

        logger.info("Integrated monitoring service stopped")

    def record_model_usage(
        self,
        question_id: str,
        task_type: str,
        selected_model: str,
        selected_tier: str,
        routing_rationale: str,
        estimated_cost: float,
        operation_mode: str = "normal",
        budget_remaining: Optional[float] = None
    ):
        """Record model usage across all monitoring components."""
        # Record in model performance tracker
        self.model_tracker.record_model_selection(
            question_id=question_id,
            task_type=task_type,
            selected_model=selected_model,
            selected_tier=selected_tier,
            routing_rationale=routing_rationale,
            estimated_cost=estimated_cost,
            operation_mode=operation_mode,
            budget_remaining=budget_remaining
        )

        logger.debug(f"Recorded model usage for {question_id}: {selected_model} ({selected_tier})")

    def record_execution_outcome(
        self,
        question_id: str,
        actual_cost: float,
        execution_time: float,
        quality_score: Optional[float] = None,
        success: bool = True,
        fallback_used: bool = False,
        forecast_value: Optional[float] = None,
        confidence: Optional[float] = None
    ):
        """Record execution outcome across all monitoring components."""
        # Update model performance tracker
        self.model_tracker.update_selection_outcome(
            question_id=question_id,
            actual_cost=actual_cost,
            execution_time=execution_time,
            quality_score=quality_score,
            success=success,
            fallback_used=fallback_used
        )

        # Record forecast if provided
        if forecast_value is not None and confidence is not None:
            self.perf_tracker.record_forecast(
                question_id=question_id,
                forecast_value=forecast_value,
                confidence=confidence
            )

        logger.debug(f"Recorded execution outcome for {question_id}: "
                    f"cost=${actual_cost:.4f}, time={execution_time:.2f}s")

    def get_comprehensive_status(self, total_budget: float = 100.0) -> ComprehensiveStatus:
        """Get comprehensive system status."""
        timestamp = datetime.now()

        # Get status from all components
        if self.cost_monitor:
            budget_status = self.cost_monitor.get_comprehensive_status()
        else:
            # Fallback budget status
            cost_breakdown = self.model_tracker.get_cost_breakdown(24)
            budget_status = {
                "budget": {
                    "total": total_budget,
                    "spent": cost_breakdown.total_cost,
                    "remaining": total_budget - cost_breakdown.total_cost,
                    "utilization_percent": (cost_breakdown.total_cost / total_budget) * 100,
                    "questions_processed": cost_breakdown.question_count,
                    "avg_cost_per_question": cost_breakdown.avg_cost_per_question
                }
            }

        cost_breakdown = self.model_tracker.get_cost_breakdown(24)
        quality_metrics = self.model_tracker.get_quality_metrics(24)
        tournament_competitiveness = self.model_tracker.get_tournament_competitiveness_indicators(total_budget)
        cost_effectiveness = self.analytics.analyze_cost_effectiveness(24)
        performance_correlations = self.analytics.analyze_performance_correlations(24)

        # Determine overall health
        overall_health = self._assess_overall_health(
            budget_status, quality_metrics, tournament_competitiveness
        )

        # Compile optimization recommendations
        optimization_recommendations = []
        optimization_recommendations.extend(cost_effectiveness.optimal_routing_suggestions)
        optimization_recommendations.extend(performance_correlations.sweet_spot_recommendations)
        optimization_recommendations.extend(tournament_competitiveness.recommendations)

        # Get active alerts
        active_alerts = self._get_active_alerts()

        return ComprehensiveStatus(
            timestamp=timestamp,
            overall_health=overall_health,
            budget_status=budget_status,
            performance_metrics={
                "cost_breakdown": asdict(cost_breakdown),
                "quality_metrics": asdict(quality_metrics),
                "cost_effectiveness": asdict(cost_effectiveness)
            },
            cost_analysis={
                "performance_correlations": asdict(performance_correlations)
            },
            optimization_recommendations=optimization_recommendations[:10],  # Top 10
            tournament_competitiveness=asdict(tournament_competitiveness),
            active_alerts=active_alerts
        )

    def generate_strategic_recommendations(
        self,
        budget_used_percentage: float,
        total_budget: float = 100.0
    ) -> Dict[str, Any]:
        """Generate strategic recommendations based on current state."""
        # Get tournament phase strategy
        phase_strategy = self.analytics.generate_tournament_phase_strategy(
            budget_used_percentage, total_budget
        )

        # Get budget optimization suggestions
        budget_optimization = self.analytics.generate_budget_optimization_suggestions(total_budget)

        # Get performance trends
        effectiveness_trends = self.model_tracker.get_model_effectiveness_trends(7)

        return {
            "tournament_phase_strategy": asdict(phase_strategy),
            "budget_optimization": asdict(budget_optimization),
            "effectiveness_trends": effectiveness_trends,
            "implementation_priority": self._prioritize_recommendations(
                phase_strategy, budget_optimization
            )
        }

    def check_alerts_and_thresholds(self) -> List[MonitoringAlert]:
        """Check all alert conditions and return new alerts."""
        new_alerts = []

        # Check performance degradation
        perf_alerts = self.perf_tracker.detect_performance_degradation()
        for alert_data in perf_alerts:
            new_alerts.append(MonitoringAlert(
                timestamp=datetime.now(),
                alert_type="performance",
                severity=alert_data.get("severity", "warning"),
                component="performance_tracker",
                message=alert_data.get("message", "Performance issue detected"),
                data=alert_data,
                recommendations=alert_data.get("recommendations", [])
            ))

        # Check tournament competitiveness
        competitiveness = self.model_tracker.get_tournament_competitiveness_indicators()
        if competitiveness.competitiveness_level in ["concerning", "critical"]:
            new_alerts.append(MonitoringAlert(
                timestamp=datetime.now(),
                alert_type="tournament",
                severity="critical" if competitiveness.competitiveness_level == "critical" else "warning",
                component="tournament_competitiveness",
                message=f"Tournament competitiveness: {competitiveness.competitiveness_level}",
                data=asdict(competitiveness),
                recommendations=competitiveness.recommendations
            ))

        # Check cost efficiency
        cost_effectiveness = self.analytics.analyze_cost_effectiveness(24)
        if cost_effectiveness.overall_efficiency < 20:  # Less than 20 questions per dollar
            new_alerts.append(MonitoringAlert(
                timestamp=datetime.now(),
                alert_type="cost",
                severity="warning",
                component="cost_efficiency",
                message=f"Low cost efficiency: {cost_effectiveness.overall_efficiency:.1f} questions/$",
                data={"efficiency": cost_effectiveness.overall_efficiency},
                recommendations=cost_effectiveness.optimal_routing_suggestions
            ))

        # Add to alert history
        self.alert_history.extend(new_alerts)

        # Trim alert history
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[-self.max_alert_history:]

        return new_alerts

    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting integrated monitoring loop")

        while not self._stop_monitoring.is_set():
            try:
                # Check alerts and thresholds
                new_alerts = self.check_alerts_and_thresholds()

                # Log new alerts
                for alert in new_alerts:
                    if alert.severity == "critical":
                        logger.critical(f"CRITICAL ALERT: {alert.message}")
                    elif alert.severity == "warning":
                        logger.warning(f"WARNING: {alert.message}")
                    else:
                        logger.info(f"INFO: {alert.message}")

                # Log periodic status summary
                if datetime.now().minute % 15 == 0:  # Every 15 minutes
                    self._log_periodic_summary()

                # Wait for next monitoring cycle
                self._stop_monitoring.wait(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._stop_monitoring.wait(60)  # Wait longer on error

    def _assess_overall_health(
        self,
        budget_status: Dict[str, Any],
        quality_metrics,
        tournament_competitiveness
    ) -> str:
        """Assess overall system health."""
        health_score = 0

        # Budget health
        budget_util = budget_status.get("budget", {}).get("utilization_percent", 0)
        if budget_util < 70:
            health_score += 3
        elif budget_util < 85:
            health_score += 2
        elif budget_util < 95:
            health_score += 1

        # Quality health
        if quality_metrics.avg_quality_score >= 0.8:
            health_score += 3
        elif quality_metrics.avg_quality_score >= 0.7:
            health_score += 2
        elif quality_metrics.avg_quality_score >= 0.6:
            health_score += 1

        # Success rate health
        if quality_metrics.success_rate >= 0.95:
            health_score += 2
        elif quality_metrics.success_rate >= 0.9:
            health_score += 1

        # Tournament competitiveness health
        if tournament_competitiveness.competitiveness_level == "excellent":
            health_score += 2
        elif tournament_competitiveness.competitiveness_level == "good":
            health_score += 1
        elif tournament_competitiveness.competitiveness_level == "critical":
            health_score -= 2

        # Determine overall health
        if health_score >= 8:
            return "excellent"
        elif health_score >= 6:
            return "good"
        elif health_score >= 4:
            return "concerning"
        else:
            return "critical"

    def _get_active_alerts(self, hours: int = 24) -> List[MonitoringAlert]:
        """Get active alerts from the last specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]

    def _prioritize_recommendations(self, phase_strategy, budget_optimization) -> List[str]:
        """Prioritize implementation recommendations."""
        priority_recommendations = []

        # High priority: budget critical issues
        if budget_optimization.risk_assessment == "high":
            priority_recommendations.extend(budget_optimization.implementation_steps[:2])

        # Medium priority: phase strategy adjustments
        priority_recommendations.extend(phase_strategy.routing_adjustments[:3])

        # Lower priority: optimization suggestions
        if budget_optimization.potential_savings > 1.0:  # $1+ savings
            priority_recommendations.append(
                f"Implement budget optimization for ${budget_optimization.potential_savings:.2f} savings"
            )

        return priority_recommendations[:5]  # Top 5 priorities

    def _log_periodic_summary(self):
        """Log periodic monitoring summary."""
        try:
            status = self.get_comprehensive_status()

            logger.info("=== Integrated Monitoring Summary ===")
            logger.info(f"Overall Health: {status.overall_health.upper()}")

            # Budget summary
            budget = status.budget_status.get("budget", {})
            logger.info(f"Budget: {budget.get('utilization_percent', 0):.1f}% used, "
                       f"${budget.get('remaining', 0):.2f} remaining")

            # Performance summary
            perf = status.performance_metrics.get("quality_metrics", {})
            logger.info(f"Quality: {perf.get('avg_quality_score', 0):.3f}, "
                       f"Success: {perf.get('success_rate', 0):.1%}")

            # Tournament competitiveness
            tournament = status.tournament_competitiveness
            logger.info(f"Competitiveness: {tournament.get('competitiveness_level', 'unknown').upper()}")

            # Active alerts
            if status.active_alerts:
                logger.info(f"Active Alerts: {len(status.active_alerts)}")

            # Top recommendations
            if status.optimization_recommendations:
                logger.info("Top Recommendations:")
                for i, rec in enumerate(status.optimization_recommendations[:3], 1):
                    logger.info(f"  {i}. {rec}")

        except Exception as e:
            logger.error(f"Error in periodic summary: {e}")

    def export_monitoring_data(self, hours: int = 24) -> Dict[str, Any]:
        """Export comprehensive monitoring data for analysis."""
        return {
            "comprehensive_status": asdict(self.get_comprehensive_status()),
            "model_effectiveness_trends": self.model_tracker.get_model_effectiveness_trends(7),
            "cost_breakdown": asdict(self.model_tracker.get_cost_breakdown(hours)),
            "quality_metrics": asdict(self.model_tracker.get_quality_metrics(hours)),
            "optimization_analysis": {
                "cost_effectiveness": asdict(self.analytics.analyze_cost_effectiveness(hours)),
                "performance_correlations": asdict(self.analytics.analyze_performance_correlations(hours))
            },
            "alert_history": [alert.to_dict() for alert in self._get_active_alerts(hours)],
            "export_timestamp": datetime.now().isoformat()
        }


# Global instance
integrated_monitoring_service = IntegratedMonitoringService()

# Backward compatibility alias
integrated_monitoring = integrated_monitoring_service
