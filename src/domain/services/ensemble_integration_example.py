"""
Example integration of DynamicWeightAdjuster with ensemble forecasting.

This demonstrates how the enhanced DynamicWeightAdjuster can be used
for real-time agent selection and automatic rebalancing.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import uuid4
import structlog

from .dynamic_weight_adjuster import DynamicWeightAdjuster, EnsembleComposition
from .ensemble_service import EnsembleService
from ..entities.prediction import Prediction, PredictionMethod
from ..entities.question import Question


logger = structlog.get_logger(__name__)


class EnhancedEnsembleManager:
    """
    Enhanced ensemble manager that integrates DynamicWeightAdjuster
    for performance-based adaptation and automatic rebalancing.
    """

    def __init__(self):
        self.ensemble_service = EnsembleService()
        self.weight_adjuster = DynamicWeightAdjuster(
            lookback_window=30,
            min_predictions_for_weight=5,
            performance_decay_factor=0.95
        )
        self.current_composition: Optional[EnsembleComposition] = None
        self.rebalancing_interval = timedelta(hours=6)  # Check every 6 hours
        self.last_rebalancing_check = datetime.now()

    def record_prediction_outcome(self,
                                prediction: Prediction,
                                actual_outcome: bool) -> None:
        """
        Record the outcome of a prediction for performance tracking.

        Args:
            prediction: The prediction that was made
            actual_outcome: The actual outcome (True/False)
        """
        self.weight_adjuster.record_performance(prediction, actual_outcome)

        logger.info(
            "Prediction outcome recorded",
            agent=prediction.created_by,
            predicted_prob=prediction.result.binary_probability,
            actual_outcome=actual_outcome,
            question_id=str(prediction.question_id)
        )

    def get_optimal_ensemble_composition(self,
                                       available_agents: List[str],
                                       force_rebalancing: bool = False) -> EnsembleComposition:
        """
        Get optimal ensemble composition with automatic rebalancing.

        Args:
            available_agents: List of available agent names
            force_rebalancing: Force rebalancing even if not triggered

        Returns:
            Optimal ensemble composition
        """
        current_time = datetime.now()

        # Check if rebalancing is needed
        should_check_rebalancing = (
            force_rebalancing or
            current_time - self.last_rebalancing_check >= self.rebalancing_interval or
            self.current_composition is None
        )

        if should_check_rebalancing:
            self.last_rebalancing_check = current_time

            current_agents = []
            if self.current_composition:
                current_agents = list(self.current_composition.agent_weights.keys())

            # Try automatic rebalancing
            new_composition = self.weight_adjuster.trigger_automatic_rebalancing(
                current_agents, available_agents
            )

            if new_composition:
                logger.info(
                    "Automatic rebalancing triggered",
                    previous_agents=current_agents,
                    new_agents=list(new_composition.agent_weights.keys()),
                    reason="Performance-based rebalancing"
                )
                self.current_composition = new_composition
            elif self.current_composition is None:
                # First time setup
                self.current_composition = self.weight_adjuster.recommend_ensemble_composition(
                    available_agents, target_size=min(5, len(available_agents))
                )
                logger.info(
                    "Initial ensemble composition created",
                    agents=list(self.current_composition.agent_weights.keys())
                )

        return self.current_composition

    def make_ensemble_prediction(self,
                               question: Question,
                               agent_predictions: Dict[str, Prediction]) -> Prediction:
        """
        Make an ensemble prediction using optimal agent composition.

        Args:
            question: The question to predict
            agent_predictions: Dictionary mapping agent names to their predictions

        Returns:
            Ensemble prediction
        """
        available_agents = list(agent_predictions.keys())

        # Get optimal composition
        composition = self.get_optimal_ensemble_composition(available_agents)

        # Filter predictions to only include agents in the composition
        selected_predictions = []
        weights = []

        for agent_name, weight in composition.agent_weights.items():
            if agent_name in agent_predictions:
                selected_predictions.append(agent_predictions[agent_name])
                weights.append(weight)

        if not selected_predictions:
            raise ValueError("No valid predictions available for ensemble")

        # Create ensemble prediction using weighted average
        ensemble_prediction = self.ensemble_service.aggregate_predictions(
            selected_predictions,
            method="weighted_average",
            weights=weights
        )

        # Update metadata to reflect ensemble composition
        ensemble_prediction.created_by = "ensemble"
        ensemble_prediction.method = PredictionMethod.ENSEMBLE

        # Add composition info to reasoning
        composition_info = f"\nEnsemble composition: {len(composition.agent_weights)} agents\n"
        for agent, weight in composition.agent_weights.items():
            composition_info += f"- {agent}: {weight:.3f}\n"
        composition_info += f"Diversity score: {composition.diversity_score:.3f}\n"
        composition_info += f"Expected performance: {composition.expected_performance:.3f}"

        ensemble_prediction.reasoning += composition_info

        logger.info(
            "Ensemble prediction created",
            question_id=str(question.id),
            agents_used=list(composition.agent_weights.keys()),
            weights=composition.agent_weights,
            predicted_probability=ensemble_prediction.result.binary_probability
        )

        return ensemble_prediction

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive performance dashboard data.

        Returns:
            Dictionary with performance metrics and recommendations
        """
        summary = self.weight_adjuster.get_performance_summary()

        current_agents = []
        if self.current_composition:
            current_agents = list(self.current_composition.agent_weights.keys())

        recommendations = self.weight_adjuster.get_rebalancing_recommendations(current_agents)

        dashboard = {
            "current_composition": {
                "agents": current_agents,
                "weights": self.current_composition.agent_weights if self.current_composition else {},
                "diversity_score": self.current_composition.diversity_score if self.current_composition else 0.0,
                "expected_performance": self.current_composition.expected_performance if self.current_composition else 0.0
            },
            "performance_summary": summary,
            "rebalancing_recommendations": recommendations,
            "system_status": {
                "last_rebalancing_check": self.last_rebalancing_check.isoformat(),
                "next_rebalancing_check": (self.last_rebalancing_check + self.rebalancing_interval).isoformat(),
                "total_agents_tracked": len(summary.get("agent_profiles", {})),
                "total_predictions_recorded": summary.get("total_predictions", 0)
            }
        }

        return dashboard

    def force_rebalancing(self, available_agents: List[str]) -> EnsembleComposition:
        """
        Force immediate rebalancing of the ensemble.

        Args:
            available_agents: List of available agent names

        Returns:
            New ensemble composition
        """
        logger.info("Forcing ensemble rebalancing", available_agents=available_agents)

        new_composition = self.get_optimal_ensemble_composition(
            available_agents, force_rebalancing=True
        )

        return new_composition

    def get_agent_performance_report(self, agent_name: str) -> Dict[str, Any]:
        """
        Get detailed performance report for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Detailed performance report
        """
        profile = self.weight_adjuster.agent_profiles.get(agent_name)
        if not profile:
            return {"error": f"No performance data available for agent: {agent_name}"}

        is_degrading, degradation_explanation = self.weight_adjuster.detect_performance_degradation(agent_name)

        report = {
            "agent_name": agent_name,
            "performance_metrics": {
                "total_predictions": profile.total_predictions,
                "recent_predictions": profile.recent_predictions,
                "overall_brier_score": profile.overall_brier_score,
                "recent_brier_score": profile.recent_brier_score,
                "overall_accuracy": profile.overall_accuracy,
                "recent_accuracy": profile.recent_accuracy,
                "calibration_score": profile.calibration_score,
                "confidence_correlation": profile.confidence_correlation,
                "performance_trend": profile.performance_trend,
                "consistency_score": profile.consistency_score
            },
            "current_status": {
                "current_weight": profile.current_weight,
                "recommended_weight": profile.recommended_weight,
                "is_degrading": is_degrading,
                "degradation_explanation": degradation_explanation,
                "specialization_areas": profile.specialization_areas,
                "last_updated": profile.last_updated.isoformat()
            },
            "recommendations": {
                "should_include_in_ensemble": profile.recommended_weight > 0.1,
                "priority_level": "high" if profile.recommended_weight > 0.5 else "medium" if profile.recommended_weight > 0.2 else "low",
                "improvement_areas": []
            }
        }

        # Add improvement recommendations
        if profile.recent_brier_score > 0.25:
            report["recommendations"]["improvement_areas"].append("Improve prediction accuracy")
        if profile.consistency_score < 0.5:
            report["recommendations"]["improvement_areas"].append("Improve prediction consistency")
        if profile.confidence_correlation < 0.1:
            report["recommendations"]["improvement_areas"].append("Improve confidence calibration")

        return report
