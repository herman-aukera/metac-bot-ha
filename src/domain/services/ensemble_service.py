"""Ensemble service for coordinating multiple forecasting agents."""

from typing import List, Dict, Any, Optional, Tuple, Callable
from uuid import UUID
import statistics
import math
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
            "performance_weighted",
            "meta_reasoning",
            "bayesian_model_averaging",
            "stacked_generalization",
            "dynamic_selection",
            "outlier_robust_mean",
            "entropy_weighted"
        ]
        self.supported_agent_types = [
            "chain_of_thought",
            "tree_of_thought",
            "react",
            "auto_cot",
            "self_consistency"
        ]

        # Performance tracking for dynamic method selection
        self.method_performance_history: Dict[str, List[float]] = {}
        self.agent_performance_history: Dict[str, List[float]] = {}

        # Method selection strategies
        self.method_selectors = {
            "best_recent": self._select_best_recent_method,
            "ensemble_of_methods": self._select_ensemble_of_methods,
            "adaptive_threshold": self._select_adaptive_threshold_method,
            "diversity_based": self._select_diversity_based_method
        }

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

        # Extract probability values from result attribute
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if not probabilities:
            raise ValueError("No valid binary probabilities found in predictions")

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
            confidence_weights = [p.get_confidence_score() for p in predictions]
            aggregated_prob = self._weighted_average(probabilities, confidence_weights)
        elif method == "performance_weighted":
            # For now, use equal weights - in real implementation,
            # this would use historical performance data
            performance_weights = [1.0] * len(predictions)
            aggregated_prob = self._weighted_average(probabilities, performance_weights)
        elif method == "meta_reasoning":
            aggregated_prob = self._meta_reasoning_aggregation(predictions)
        elif method == "bayesian_model_averaging":
            aggregated_prob = self._bayesian_model_averaging(predictions)
        elif method == "stacked_generalization":
            aggregated_prob = self._stacked_generalization(predictions)
        elif method == "dynamic_selection":
            aggregated_prob = self._dynamic_selection_aggregation(predictions)
        elif method == "outlier_robust_mean":
            aggregated_prob = self._outlier_robust_mean(probabilities)
        elif method == "entropy_weighted":
            aggregated_prob = self._entropy_weighted_aggregation(predictions)
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
        ensemble_prediction = Prediction.create_binary_prediction(
            question_id=question_id,
            probability=aggregated_prob,
            confidence=ensemble_confidence,
            reasoning=ensemble_reasoning,
            method=PredictionMethod.ENSEMBLE,
            created_by=f"EnsembleService-{method}",
            research_report_id=predictions[0].research_report_id,  # Use first prediction's research report
            method_metadata={
                "aggregation_method": method,
                "input_predictions_count": len(predictions),
                "agent_types": list(set(p.created_by for p in predictions)),
                "confidence_range": {
                    "min": min(p.get_confidence_score() for p in predictions),
                    "max": max(p.get_confidence_score() for p in predictions),
                    "mean": statistics.mean(p.get_confidence_score() for p in predictions)
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
        confidences = [p.get_confidence_score() for p in predictions]
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
    ) -> PredictionConfidence:
        """Calculate confidence level for ensemble prediction."""

        confidences = [p.get_confidence_score() for p in predictions]
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        # Base confidence from individual predictions
        mean_confidence = statistics.mean(confidences)

        # Adjustment based on agreement between predictions
        prob_variance = statistics.variance(probabilities) if len(probabilities) > 1 else 0.0

        # Higher agreement (lower variance) increases ensemble confidence
        agreement_bonus = max(0, 0.05 - prob_variance)

        # Diversity bonus - having predictions from different methods increases confidence
        unique_methods = len(set(p.method for p in predictions))
        diversity_bonus = min(0.02, unique_methods * 0.005)

        # Sample size bonus - more predictions generally increase confidence
        sample_bonus = min(0.02, len(predictions) * 0.002)

        ensemble_confidence = mean_confidence + agreement_bonus + diversity_bonus + sample_bonus
        ensemble_confidence = max(0.0, min(1.0, ensemble_confidence))

        # Convert numeric confidence to enum
        if ensemble_confidence >= 0.9:
            return PredictionConfidence.VERY_HIGH
        elif ensemble_confidence >= 0.7:
            return PredictionConfidence.HIGH
        elif ensemble_confidence >= 0.5:
            return PredictionConfidence.MEDIUM
        elif ensemble_confidence >= 0.3:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW

    def _create_ensemble_reasoning(
        self,
        predictions: List[Prediction],
        method: str,
        final_probability: float
    ) -> str:
        """Create reasoning explanation for ensemble prediction."""

        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]
        agent_names = [p.created_by for p in predictions]

        reasoning = f"Ensemble prediction using {method} aggregation of {len(predictions)} predictions.\n\n"

        reasoning += "Individual predictions:\n"
        for i, pred in enumerate(predictions):
            prob_val = pred.result.binary_probability if pred.result.binary_probability is not None else 0.5
            conf_val = pred.get_confidence_score()
            reasoning += f"- {pred.created_by}: {prob_val:.3f} (confidence: {conf_val:.2f})\n"

        reasoning += f"\nStatistics:\n"
        reasoning += f"- Mean: {statistics.mean(probabilities):.3f}\n"
        reasoning += f"- Median: {statistics.median(probabilities):.3f}\n"

        if len(probabilities) > 1:
            reasoning += f"- Standard deviation: {statistics.stdev(probabilities):.3f}\n"
            reasoning += f"- Range: {min(probabilities):.3f} - {max(probabilities):.3f}\n"

        reasoning += f"\nFinal ensemble probability: {final_probability:.3f}\n"

        # Add interpretation based on agreement
        prob_variance = statistics.variance(probabilities) if len(probabilities) > 1 else 0.0
        if prob_variance < 0.005:
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
                probs = [p.result.binary_probability for p in pred_set if p.result.binary_probability is not None]
                if len(probs) > 1:
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

        confidences = [p.get_confidence_score() for p in ensemble_predictions]

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
            prob = pred.result.binary_probability if pred.result.binary_probability is not None else 0.5
            actual = 1.0 if truth else 0.0
            brier_scores.append((prob - actual) ** 2)

        # Calibration (simplified)
        # In practice, you'd bin predictions and check if actual frequency matches predicted

        return {
            "brier_score": statistics.mean(brier_scores),
            "mean_absolute_error": statistics.mean(
                abs((pred.result.binary_probability if pred.result.binary_probability is not None else 0.5) - (1.0 if truth else 0.0))
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

    # ===== SOPHISTICATED AGGREGATION METHODS =====

    def _meta_reasoning_aggregation(self, predictions: List[Prediction]) -> float:
        """
        Meta-reasoning aggregation that considers the reasoning quality and coherence.

        This method analyzes the reasoning chains of predictions and weights them
        based on logical consistency, evidence quality, and reasoning depth.
        """
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if not probabilities:
            return 0.5

        # Calculate reasoning quality scores
        reasoning_scores = []
        for pred in predictions:
            score = self._evaluate_reasoning_quality(pred)
            reasoning_scores.append(score)

        # Normalize reasoning scores to use as weights
        total_score = sum(reasoning_scores)
        if total_score == 0:
            return statistics.mean(probabilities)

        weights = [score / total_score for score in reasoning_scores]

        # Apply meta-reasoning adjustment based on consensus
        consensus_adjustment = self._calculate_consensus_adjustment(predictions)

        weighted_prob = self._weighted_average(probabilities, weights)

        # Apply consensus adjustment
        adjusted_prob = weighted_prob + consensus_adjustment

        return max(0.0, min(1.0, adjusted_prob))

    def _evaluate_reasoning_quality(self, prediction: Prediction) -> float:
        """
        Evaluate the quality of reasoning in a prediction.

        Considers factors like:
        - Length and detail of reasoning
        - Presence of evidence citations
        - Logical structure indicators
        - Uncertainty acknowledgment
        """
        reasoning = prediction.reasoning or ""

        # Base score from reasoning length (diminishing returns)
        length_score = min(1.0, len(reasoning) / 1000.0)

        # Evidence indicators
        evidence_indicators = ["according to", "research shows", "data indicates", "study found", "evidence suggests"]
        evidence_score = sum(0.1 for indicator in evidence_indicators if indicator.lower() in reasoning.lower())
        evidence_score = min(0.5, evidence_score)

        # Logical structure indicators
        structure_indicators = ["therefore", "because", "however", "furthermore", "in contrast", "on the other hand"]
        structure_score = sum(0.05 for indicator in structure_indicators if indicator.lower() in reasoning.lower())
        structure_score = min(0.3, structure_score)

        # Uncertainty acknowledgment (good reasoning acknowledges uncertainty)
        uncertainty_indicators = ["uncertain", "unclear", "might", "could", "possibly", "likely", "probably"]
        uncertainty_score = sum(0.02 for indicator in uncertainty_indicators if indicator.lower() in reasoning.lower())
        uncertainty_score = min(0.2, uncertainty_score)

        total_score = length_score + evidence_score + structure_score + uncertainty_score

        return min(1.0, total_score)

    def _calculate_consensus_adjustment(self, predictions: List[Prediction]) -> float:
        """
        Calculate adjustment based on consensus among high-quality predictions.

        Strong consensus among high-quality predictions increases confidence,
        while disagreement suggests more uncertainty.
        """
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if len(probabilities) < 2:
            return 0.0

        # Calculate variance as measure of disagreement
        variance = statistics.variance(probabilities)

        # High agreement (low variance) provides small positive adjustment
        # High disagreement (high variance) provides small negative adjustment
        if variance < 0.01:  # Very high agreement
            return 0.02
        elif variance < 0.05:  # Moderate agreement
            return 0.01
        elif variance > 0.1:  # High disagreement
            return -0.01
        else:
            return 0.0

    def _bayesian_model_averaging(self, predictions: List[Prediction]) -> float:
        """
        Bayesian Model Averaging that weights predictions based on their likelihood
        and incorporates prior beliefs about agent performance.
        """
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if not probabilities:
            return 0.5

        # Calculate model likelihoods based on confidence and historical performance
        likelihoods = []
        for pred in predictions:
            # Base likelihood from confidence
            confidence_likelihood = pred.get_confidence_score()

            # Historical performance adjustment
            agent_name = pred.created_by
            if agent_name in self.agent_performance_history:
                recent_performance = self.agent_performance_history[agent_name][-10:]  # Last 10 predictions
                if recent_performance:
                    performance_factor = statistics.mean(recent_performance)
                    confidence_likelihood *= (0.5 + performance_factor)  # Scale by performance

            likelihoods.append(confidence_likelihood)

        # Normalize likelihoods to get posterior weights
        total_likelihood = sum(likelihoods)
        if total_likelihood == 0:
            return statistics.mean(probabilities)

        weights = [likelihood / total_likelihood for likelihood in likelihoods]

        return self._weighted_average(probabilities, weights)

    def _stacked_generalization(self, predictions: List[Prediction]) -> float:
        """
        Stacked generalization (stacking) that learns optimal combination weights
        based on prediction patterns and performance.
        """
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if not probabilities:
            return 0.5

        # For now, implement a simplified version that uses confidence and diversity
        # In a full implementation, this would use a trained meta-learner

        # Calculate base weights from confidence
        confidence_weights = [pred.get_confidence_score() for pred in predictions]

        # Adjust weights based on prediction diversity
        diversity_factor = self._calculate_diversity_factor(probabilities)

        # Apply diversity adjustment to weights
        adjusted_weights = []
        for i, weight in enumerate(confidence_weights):
            # Predictions closer to the median get higher diversity bonus
            median_prob = statistics.median(probabilities)
            distance_from_median = abs(probabilities[i] - median_prob)
            diversity_bonus = max(0, 0.1 - distance_from_median)

            adjusted_weights.append(weight + diversity_bonus)

        # Normalize weights
        total_weight = sum(adjusted_weights)
        if total_weight == 0:
            return statistics.mean(probabilities)

        normalized_weights = [w / total_weight for w in adjusted_weights]

        return self._weighted_average(probabilities, normalized_weights)

    def _calculate_diversity_factor(self, probabilities: List[float]) -> float:
        """Calculate diversity factor based on prediction spread."""
        if len(probabilities) < 2:
            return 0.0

        variance = statistics.variance(probabilities)
        # Normalize variance to 0-1 range (assuming max reasonable variance is 0.25)
        return min(1.0, variance / 0.25)

    def _dynamic_selection_aggregation(self, predictions: List[Prediction]) -> float:
        """
        Dynamic selection that chooses the best aggregation method based on
        prediction characteristics and historical performance.
        """
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if not probabilities:
            return 0.5

        # Analyze prediction characteristics
        variance = statistics.variance(probabilities) if len(probabilities) > 1 else 0.0
        confidence_spread = max([p.get_confidence_score() for p in predictions]) - min([p.get_confidence_score() for p in predictions])

        # Select method based on characteristics
        if variance < 0.01:  # High agreement - use confidence weighting
            return self._confidence_weighted_aggregation(predictions)
        elif variance > 0.1:  # High disagreement - use robust methods
            return self._outlier_robust_mean(probabilities)
        elif confidence_spread > 0.4:  # High confidence variation - use confidence weighting
            return self._confidence_weighted_aggregation(predictions)
        else:  # Default to meta-reasoning
            return self._meta_reasoning_aggregation(predictions)

    def _confidence_weighted_aggregation(self, predictions: List[Prediction]) -> float:
        """Helper method for confidence-weighted aggregation."""
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]
        confidence_weights = [p.get_confidence_score() for p in predictions]
        return self._weighted_average(probabilities, confidence_weights)

    def _outlier_robust_mean(self, probabilities: List[float]) -> float:
        """
        Outlier-robust mean using Huber loss-inspired weighting.

        Reduces the influence of extreme predictions that might be outliers.
        """
        if not probabilities:
            return 0.5

        if len(probabilities) <= 2:
            return statistics.mean(probabilities)

        # Calculate median as robust center
        median = statistics.median(probabilities)

        # Calculate robust weights based on distance from median
        weights = []
        threshold = 0.15  # Threshold for outlier detection

        for prob in probabilities:
            distance = abs(prob - median)
            if distance <= threshold:
                weight = 1.0
            else:
                # Reduce weight for outliers using Huber-like function
                weight = threshold / distance
            weights.append(weight)

        return self._weighted_average(probabilities, weights)

    def _entropy_weighted_aggregation(self, predictions: List[Prediction]) -> float:
        """
        Entropy-weighted aggregation that considers the information content
        of each prediction based on its entropy.
        """
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if not probabilities:
            return 0.5

        # Calculate entropy for each prediction
        entropy_weights = []
        for prob in probabilities:
            # Binary entropy: -p*log(p) - (1-p)*log(1-p)
            if prob == 0.0 or prob == 1.0:
                entropy = 0.0  # No uncertainty
            else:
                entropy = -(prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))

            # Higher entropy (more uncertainty) gets lower weight
            # Convert to weight: predictions with moderate uncertainty (high entropy) are valuable
            # but extreme certainty (low entropy) might be overconfident
            if entropy > 0.8:  # High uncertainty
                weight = 0.5
            elif entropy < 0.2:  # Very low uncertainty (might be overconfident)
                weight = 0.7
            else:  # Moderate uncertainty (good calibration)
                weight = 1.0

            entropy_weights.append(weight)

        # Normalize weights
        total_weight = sum(entropy_weights)
        if total_weight == 0:
            return statistics.mean(probabilities)

        normalized_weights = [w / total_weight for w in entropy_weights]

        return self._weighted_average(probabilities, normalized_weights)

    # ===== AGGREGATION METHOD SELECTION =====

    def select_optimal_aggregation_method(
        self,
        predictions: List[Prediction],
        selection_strategy: str = "best_recent"
    ) -> str:
        """
        Select the optimal aggregation method based on prediction characteristics
        and historical performance.

        Args:
            predictions: List of predictions to aggregate
            selection_strategy: Strategy for method selection

        Returns:
            Name of the selected aggregation method
        """
        if selection_strategy not in self.method_selectors:
            logger.warning(f"Unknown selection strategy: {selection_strategy}, using best_recent")
            selection_strategy = "best_recent"

        return self.method_selectors[selection_strategy](predictions)

    def _select_best_recent_method(self, predictions: List[Prediction]) -> str:
        """Select method with best recent performance."""
        if not self.method_performance_history:
            return "meta_reasoning"  # Default to sophisticated method

        # Calculate recent performance for each method
        recent_performance = {}
        for method, history in self.method_performance_history.items():
            if history:
                # Use last 5 predictions for recent performance
                recent_scores = history[-5:]
                recent_performance[method] = statistics.mean(recent_scores)

        if not recent_performance:
            return "meta_reasoning"

        # Return method with best recent performance
        best_method = max(recent_performance.items(), key=lambda x: x[1])[0]
        return best_method

    def _select_ensemble_of_methods(self, predictions: List[Prediction]) -> str:
        """
        Select multiple methods and ensemble their results.
        For now, returns the meta_reasoning method which incorporates multiple approaches.
        """
        return "meta_reasoning"

    def _select_adaptive_threshold_method(self, predictions: List[Prediction]) -> str:
        """Select method based on adaptive thresholds for prediction characteristics."""
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if not probabilities:
            return "simple_average"

        # Calculate characteristics
        variance = statistics.variance(probabilities) if len(probabilities) > 1 else 0.0
        confidence_scores = [p.get_confidence_score() for p in predictions]
        avg_confidence = statistics.mean(confidence_scores)

        # Adaptive thresholds based on characteristics
        if variance > 0.08:  # High disagreement
            return "outlier_robust_mean"
        elif avg_confidence > 0.8:  # High confidence
            return "confidence_weighted"
        elif len(predictions) >= 5:  # Many predictions
            return "bayesian_model_averaging"
        else:
            return "meta_reasoning"

    def _select_diversity_based_method(self, predictions: List[Prediction]) -> str:
        """Select method based on prediction diversity."""
        probabilities = [p.result.binary_probability for p in predictions if p.result.binary_probability is not None]

        if len(probabilities) < 2:
            return "simple_average"

        # Calculate diversity metrics
        variance = statistics.variance(probabilities)
        range_spread = max(probabilities) - min(probabilities)

        # Select based on diversity
        if variance < 0.005 and range_spread < 0.1:  # Very low diversity
            return "simple_average"
        elif variance > 0.05 or range_spread > 0.4:  # High diversity
            return "stacked_generalization"
        else:  # Moderate diversity
            return "meta_reasoning"

    def update_method_performance(self, method: str, performance_score: float) -> None:
        """
        Update performance history for an aggregation method.

        Args:
            method: Name of the aggregation method
            performance_score: Performance score (e.g., 1 - Brier score)
        """
        if method not in self.method_performance_history:
            self.method_performance_history[method] = []

        self.method_performance_history[method].append(performance_score)

        # Keep only recent history (last 50 predictions)
        if len(self.method_performance_history[method]) > 50:
            self.method_performance_history[method] = self.method_performance_history[method][-50:]

    def update_agent_performance(self, agent_name: str, performance_score: float) -> None:
        """
        Update performance history for an agent.

        Args:
            agent_name: Name of the agent
            performance_score: Performance score (e.g., 1 - Brier score)
        """
        if agent_name not in self.agent_performance_history:
            self.agent_performance_history[agent_name] = []

        self.agent_performance_history[agent_name].append(performance_score)

        # Keep only recent history (last 50 predictions)
        if len(self.agent_performance_history[agent_name]) > 50:
            self.agent_performance_history[agent_name] = self.agent_performance_history[agent_name][-50:]

    def get_method_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance summary for all aggregation methods.

        Returns:
            Dictionary with method names and their performance statistics
        """
        summary = {}

        for method, history in self.method_performance_history.items():
            if history:
                summary[method] = {
                    "mean_performance": statistics.mean(history),
                    "recent_performance": statistics.mean(history[-10:]) if len(history) >= 10 else statistics.mean(history),
                    "performance_std": statistics.stdev(history) if len(history) > 1 else 0.0,
                    "prediction_count": len(history)
                }

        return summary
