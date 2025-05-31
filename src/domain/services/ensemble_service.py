"""Ensemble service for coordinating multiple forecasting agents."""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import statistics
from datetime import datetime
import structlog

from ..entities.question import Question
from ..entities.prediction import Prediction, PredictionMethod, PredictionConfidence
from ..entities.forecast import Forecast
from ..entities.research_report import ResearchReport
from ..value_objects.probability import Probability
from ..value_objects.confidence import ConfidenceLevel


logger = structlog.get_logger(__name__)


class EnsembleService:
    """
    Domain service for coordinating ensemble forecasting.
    
    Manages multiple forecasting agents, aggregates their predictions,
    and provides ensemble-based forecasting capabilities.
    """
    
    def __init__(self):
        self.aggregation_methods = [
            "simple_average",
            "weighted_average", 
            "median",
            "trimmed_mean",
            "confidence_weighted",
            "performance_weighted"
        ]
        self.supported_agent_types = [
            "chain_of_thought",
            "tree_of_thought", 
            "react",
            "auto_cot",
            "self_consistency"
        ]
    
    def aggregate_predictions(
        self,
        predictions: List[Prediction],
        method: str = "weighted_average",
        weights: Optional[List[float]] = None
    ) -> Prediction:
        """
        Aggregate multiple predictions into a single ensemble prediction.
        
        Args:
            predictions: List of predictions to aggregate
            method: Aggregation method to use
            weights: Optional weights for weighted aggregation
            
        Returns:
            Aggregated ensemble prediction
        """
        if not predictions:
            raise ValueError("Cannot aggregate empty prediction list")
        
        # Ensure all predictions are for the same question
        question_ids = set(p.question_id for p in predictions)
        if len(question_ids) > 1:
            raise ValueError("All predictions must be for the same question")
        
        question_id = predictions[0].question_id
        
        logger.info(
            "Aggregating predictions",
            count=len(predictions),
            method=method,
            question_id=str(question_id)
        )
        
        # Extract probability values
        probabilities = [p.probability.value for p in predictions]
        
        # Calculate aggregated probability based on method
        if method == "simple_average":
            aggregated_prob = statistics.mean(probabilities)
        elif method == "median":
            aggregated_prob = statistics.median(probabilities)
        elif method == "weighted_average":
            if weights is None:
                weights = self._calculate_default_weights(predictions)
            aggregated_prob = self._weighted_average(probabilities, weights)
        elif method == "trimmed_mean":
            aggregated_prob = self._trimmed_mean(probabilities, trim_percent=0.1)
        elif method == "confidence_weighted":
            confidence_weights = [p.confidence.value for p in predictions]
            aggregated_prob = self._weighted_average(probabilities, confidence_weights)
        elif method == "performance_weighted":
            # For now, use equal weights - in real implementation,
            # this would use historical performance data
            performance_weights = [1.0] * len(predictions)
            aggregated_prob = self._weighted_average(probabilities, performance_weights)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        # Ensure probability is within valid range
        aggregated_prob = max(0.0, min(1.0, aggregated_prob))
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(predictions, method)
        
        # Create ensemble reasoning
        ensemble_reasoning = self._create_ensemble_reasoning(
            predictions, method, aggregated_prob
        )
        
        # Create aggregated prediction
        ensemble_prediction = Prediction.create_new(
            question_id=question_id,
            probability=Probability(aggregated_prob),
            confidence=ensemble_confidence,
            reasoning=ensemble_reasoning,
            method=PredictionMethod.ENSEMBLE,
            agent_name=f"EnsembleService-{method}",
            model_version="ensemble-v1.0",
            metadata={
                "aggregation_method": method,
                "input_predictions_count": len(predictions),
                "agent_types": list(set(p.agent_name for p in predictions)),
                "confidence_range": {
                    "min": min(p.confidence.value for p in predictions),
                    "max": max(p.confidence.value for p in predictions),
                    "mean": statistics.mean(p.confidence.value for p in predictions)
                },
                "probability_range": {
                    "min": min(probabilities),
                    "max": max(probabilities),
                    "std": statistics.stdev(probabilities) if len(probabilities) > 1 else 0.0
                }
            }
        )
        
        logger.info(
            "Ensemble prediction created",
            probability=aggregated_prob,
            confidence=ensemble_confidence.value,
            method=method
        )
        
        return ensemble_prediction
    
    def _calculate_default_weights(self, predictions: List[Prediction]) -> List[float]:
        """Calculate default weights based on prediction confidence."""
        confidences = [p.confidence.value for p in predictions]
        total_confidence = sum(confidences)
        
        if total_confidence == 0:
            # Equal weights if no confidence information
            return [1.0 / len(predictions)] * len(predictions)
        
        # Normalize confidences to sum to 1
        return [c / total_confidence for c in confidences]
    
    def _weighted_average(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted average of values."""
        if len(values) != len(weights):
            raise ValueError("Values and weights must have same length")
        
        if sum(weights) == 0:
            return statistics.mean(values)
        
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum
    
    def _trimmed_mean(self, values: List[float], trim_percent: float = 0.1) -> float:
        """Calculate trimmed mean by removing extreme values."""
        if len(values) <= 2:
            return statistics.mean(values)
        
        sorted_values = sorted(values)
        trim_count = int(len(values) * trim_percent)
        
        if trim_count == 0:
            return statistics.mean(values)
        
        trimmed_values = sorted_values[trim_count:-trim_count]
        return statistics.mean(trimmed_values)
    
    def _calculate_ensemble_confidence(
        self, 
        predictions: List[Prediction], 
        method: str
    ) -> ConfidenceLevel:
        """Calculate confidence level for ensemble prediction."""
        
        confidences = [p.confidence.value for p in predictions]
        probabilities = [p.probability.value for p in predictions]
        
        # Base confidence from individual predictions
        mean_confidence = statistics.mean(confidences)
        
        # Adjustment based on agreement between predictions
        prob_variance = statistics.variance(probabilities) if len(probabilities) > 1 else 0.0
        
        # Higher agreement (lower variance) increases ensemble confidence
        agreement_bonus = max(0, 0.2 - prob_variance)
        
        # Diversity bonus - having predictions from different methods increases confidence
        unique_methods = len(set(p.method for p in predictions))
        diversity_bonus = min(0.1, unique_methods * 0.02)
        
        # Sample size bonus - more predictions generally increase confidence
        sample_bonus = min(0.1, len(predictions) * 0.01)
        
        ensemble_confidence = mean_confidence + agreement_bonus + diversity_bonus + sample_bonus
        ensemble_confidence = max(0.0, min(1.0, ensemble_confidence))
        
        return ConfidenceLevel(ensemble_confidence)
    
    def _create_ensemble_reasoning(
        self,
        predictions: List[Prediction],
        method: str,
        final_probability: float
    ) -> str:
        """Create reasoning explanation for ensemble prediction."""
        
        probabilities = [p.probability.value for p in predictions]
        agent_names = [p.agent_name for p in predictions]
        
        reasoning = f"Ensemble prediction using {method} aggregation of {len(predictions)} predictions.\n\n"
        
        reasoning += "Individual predictions:\n"
        for i, pred in enumerate(predictions):
            reasoning += f"- {pred.agent_name}: {pred.probability.value:.3f} (confidence: {pred.confidence.value:.2f})\n"
        
        reasoning += f"\nStatistics:\n"
        reasoning += f"- Mean: {statistics.mean(probabilities):.3f}\n"
        reasoning += f"- Median: {statistics.median(probabilities):.3f}\n"
        
        if len(probabilities) > 1:
            reasoning += f"- Standard deviation: {statistics.stdev(probabilities):.3f}\n"
            reasoning += f"- Range: {min(probabilities):.3f} - {max(probabilities):.3f}\n"
        
        reasoning += f"\nFinal ensemble probability: {final_probability:.3f}\n"
        
        # Add interpretation based on agreement
        prob_variance = statistics.variance(probabilities) if len(probabilities) > 1 else 0.0
        if prob_variance < 0.01:
            reasoning += "\nHigh agreement between predictions increases confidence in ensemble result."
        elif prob_variance > 0.05:
            reasoning += "\nLow agreement between predictions suggests higher uncertainty."
        else:
            reasoning += "\nModerate agreement between predictions."
        
        return reasoning
    
    def evaluate_ensemble_performance(
        self,
        ensemble_predictions: List[Prediction],
        individual_predictions: List[List[Prediction]],
        ground_truth: Optional[List[bool]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble performance against individual predictions.
        
        Args:
            ensemble_predictions: List of ensemble predictions
            individual_predictions: List of lists, each containing individual predictions for a question
            ground_truth: Optional ground truth values for evaluation
            
        Returns:
            Performance metrics dictionary
        """
        
        if len(ensemble_predictions) != len(individual_predictions):
            raise ValueError("Ensemble and individual predictions lists must have same length")
        
        metrics = {
            "ensemble_count": len(ensemble_predictions),
            "average_input_predictions": statistics.mean(len(preds) for preds in individual_predictions),
            "diversity_metrics": self._calculate_diversity_metrics(individual_predictions),
            "confidence_metrics": self._calculate_confidence_metrics(ensemble_predictions),
        }
        
        if ground_truth:
            if len(ground_truth) != len(ensemble_predictions):
                raise ValueError("Ground truth must match ensemble predictions length")
            
            # Calculate accuracy metrics
            metrics["accuracy_metrics"] = self._calculate_accuracy_metrics(
                ensemble_predictions, ground_truth
            )
        
        return metrics
    
    def _calculate_diversity_metrics(
        self, 
        individual_predictions: List[List[Prediction]]
    ) -> Dict[str, float]:
        """Calculate diversity metrics for prediction sets."""
        
        all_variances = []
        all_ranges = []
        
        for pred_set in individual_predictions:
            if len(pred_set) > 1:
                probs = [p.probability.value for p in pred_set]
                all_variances.append(statistics.variance(probs))
                all_ranges.append(max(probs) - min(probs))
        
        return {
            "mean_variance": statistics.mean(all_variances) if all_variances else 0.0,
            "mean_range": statistics.mean(all_ranges) if all_ranges else 0.0,
            "high_diversity_fraction": sum(1 for v in all_variances if v > 0.05) / len(all_variances) if all_variances else 0.0
        }
    
    def _calculate_confidence_metrics(
        self, 
        ensemble_predictions: List[Prediction]
    ) -> Dict[str, float]:
        """Calculate confidence metrics for ensemble predictions."""
        
        confidences = [p.confidence.value for p in ensemble_predictions]
        
        return {
            "mean_confidence": statistics.mean(confidences),
            "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            "high_confidence_fraction": sum(1 for c in confidences if c > 0.8) / len(confidences)
        }
    
    def _calculate_accuracy_metrics(
        self,
        predictions: List[Prediction],
        ground_truth: List[bool]
    ) -> Dict[str, float]:
        """Calculate accuracy metrics against ground truth."""
        
        # Brier score
        brier_scores = []
        for pred, truth in zip(predictions, ground_truth):
            prob = pred.probability.value
            actual = 1.0 if truth else 0.0
            brier_scores.append((prob - actual) ** 2)
        
        # Calibration (simplified)
        # In practice, you'd bin predictions and check if actual frequency matches predicted
        
        return {
            "brier_score": statistics.mean(brier_scores),
            "mean_absolute_error": statistics.mean(
                abs(pred.probability.value - (1.0 if truth else 0.0))
                for pred, truth in zip(predictions, ground_truth)
            )
        }
    
    def get_supported_methods(self) -> List[str]:
        """Get list of supported aggregation methods."""
        return self.aggregation_methods.copy()
    
    def get_supported_agent_types(self) -> List[str]:
        """Get list of supported agent types."""
        return self.supported_agent_types.copy()
    
    def validate_ensemble_config(self, config: Dict[str, Any]) -> bool:
        """Validate ensemble configuration."""
        
        if "aggregation_method" in config:
            if config["aggregation_method"] not in self.aggregation_methods:
                return False
        
        if "weights" in config:
            weights = config["weights"]
            if not isinstance(weights, list) or not all(isinstance(w, (int, float)) for w in weights):
                return False
        
        if "min_predictions" in config:
            min_preds = config["min_predictions"]
            if not isinstance(min_preds, int) or min_preds < 1:
                return False
        
        return True