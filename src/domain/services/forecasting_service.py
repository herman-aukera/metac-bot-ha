"""Forecasting service for domain logic."""

from typing import Any, Dict, List
from uuid import UUID

from ..entities.forecast import Forecast
from ..entities.prediction import Prediction, PredictionConfidence, PredictionMethod


class ForecastingService:
    """
    Domain service for coordinating forecasting activities.

    Contains the core business logic for generating forecasts
    from research reports and predictions.
    """

    def __init__(self):
        self.supported_methods = [
            PredictionMethod.CHAIN_OF_THOUGHT,
            PredictionMethod.TREE_OF_THOUGHT,
            PredictionMethod.REACT,
            PredictionMethod.AUTO_COT,
            PredictionMethod.SELF_CONSISTENCY,
        ]

    def aggregate_predictions(
        self, predictions: List[Prediction], method: str = "weighted_average"
    ) -> Prediction:
        """
        Aggregate multiple predictions into a single final prediction.

        Args:
            predictions: List of predictions to aggregate
            method: Aggregation method to use

        Returns:
            Final aggregated prediction
        """
        if not predictions:
            raise ValueError("Cannot aggregate empty prediction list")

        # Ensure all predictions are for the same question
        question_ids = set(p.question_id for p in predictions)
        if len(question_ids) > 1:
            raise ValueError("All predictions must be for the same question")

        question_id = list(question_ids)[0]

        if method == "weighted_average":
            return self._weighted_average_aggregation(predictions, question_id)
        elif method == "median":
            return self._median_aggregation(predictions, question_id)
        elif method == "confidence_weighted":
            return self._confidence_weighted_aggregation(predictions, question_id)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    def _weighted_average_aggregation(
        self, predictions: List[Prediction], question_id: UUID
    ) -> Prediction:
        """Aggregate predictions using weighted average."""
        # For binary predictions
        binary_predictions = [
            p for p in predictions if p.result.binary_probability is not None
        ]

        if binary_predictions:
            # Weight by confidence score
            weights = [p.get_confidence_score() for p in binary_predictions]
            total_weight = sum(weights)

            if total_weight == 0:
                # Equal weights if all have zero confidence
                weights = [1.0] * len(binary_predictions)
                total_weight = len(binary_predictions)

            weighted_prob = sum(
                p.result.binary_probability * (weight / total_weight)
                for p, weight in zip(binary_predictions, weights)
            )

            # Calculate average confidence
            avg_confidence = self._calculate_average_confidence(predictions)

            # Create aggregated prediction
            return Prediction.create_binary_prediction(
                question_id=question_id,
                research_report_id=binary_predictions[0].research_report_id,
                probability=weighted_prob,
                confidence=avg_confidence,
                method=PredictionMethod.ENSEMBLE,
                reasoning=f"Weighted average of {len(binary_predictions)} predictions",
                created_by="ensemble_service",
                method_metadata={
                    "aggregation_method": "weighted_average",
                    "component_predictions": len(predictions),
                    "weights": weights,
                },
            )

        # Add similar logic for numeric and multiple choice predictions...
        raise NotImplementedError("Only binary predictions supported currently")

    def _median_aggregation(
        self, predictions: List[Prediction], question_id: UUID
    ) -> Prediction:
        """Aggregate predictions using median."""
        binary_predictions = [
            p for p in predictions if p.result.binary_probability is not None
        ]

        if binary_predictions:
            probs = sorted([p.result.binary_probability for p in binary_predictions])
            n = len(probs)

            if n % 2 == 0:
                median_prob = (probs[n // 2 - 1] + probs[n // 2]) / 2
            else:
                median_prob = probs[n // 2]

            avg_confidence = self._calculate_average_confidence(predictions)

            return Prediction.create_binary_prediction(
                question_id=question_id,
                research_report_id=binary_predictions[0].research_report_id,
                probability=median_prob,
                confidence=avg_confidence,
                method=PredictionMethod.ENSEMBLE,
                reasoning=f"Median of {len(binary_predictions)} predictions",
                created_by="ensemble_service",
                method_metadata={
                    "aggregation_method": "median",
                    "component_predictions": len(predictions),
                },
            )

        raise NotImplementedError("Only binary predictions supported currently")

    def _confidence_weighted_aggregation(
        self, predictions: List[Prediction], question_id: UUID
    ) -> Prediction:
        """Aggregate predictions weighted by confidence scores."""
        binary_predictions = [
            p for p in predictions if p.result.binary_probability is not None
        ]

        if binary_predictions:
            confidence_weights = [
                p.get_confidence_score() ** 2 for p in binary_predictions
            ]  # Square for emphasis
            total_weight = sum(confidence_weights)

            if total_weight == 0:
                # Fall back to equal weights
                return self._weighted_average_aggregation(predictions, question_id)

            weighted_prob = sum(
                p.result.binary_probability * (weight / total_weight)
                for p, weight in zip(binary_predictions, confidence_weights)
            )

            # Weighted average of confidences
            weighted_confidence_score = sum(
                p.get_confidence_score() * (weight / total_weight)
                for p, weight in zip(binary_predictions, confidence_weights)
            )

            # Convert back to confidence enum
            confidence = self._score_to_confidence(weighted_confidence_score)

            return Prediction.create_binary_prediction(
                question_id=question_id,
                research_report_id=binary_predictions[0].research_report_id,
                probability=weighted_prob,
                confidence=confidence,
                method=PredictionMethod.ENSEMBLE,
                reasoning=f"Confidence-weighted average of {len(binary_predictions)} predictions",
                created_by="ensemble_service",
                method_metadata={
                    "aggregation_method": "confidence_weighted",
                    "component_predictions": len(predictions),
                    "confidence_weights": confidence_weights,
                },
            )

        raise NotImplementedError("Only binary predictions supported currently")

    def confidence_weighted_average(
        self, predictions: List[Prediction]
    ) -> "Probability":
        """
        Calculate confidence-weighted average of predictions and return as Probability.

        Args:
            predictions: List of predictions to aggregate

        Returns:
            Probability object with confidence-weighted average value
        """
        if not predictions:
            raise ValueError("Cannot calculate average of empty prediction list")

        # Filter to binary predictions only
        binary_predictions = [
            p for p in predictions if p.result.binary_probability is not None
        ]

        if not binary_predictions:
            # Fallback to 0.5 if no binary predictions
            from ..value_objects.probability import Probability

            return Probability(0.5)

        if len(binary_predictions) == 1:
            # Single prediction - return its probability
            from ..value_objects.probability import Probability

            return Probability(binary_predictions[0].result.binary_probability)

        # Calculate confidence-weighted average
        confidence_weights = [
            p.get_confidence_score() ** 2 for p in binary_predictions
        ]  # Square for emphasis
        total_weight = sum(confidence_weights)

        if total_weight == 0:
            # Equal weights if all have zero confidence
            weighted_prob = sum(
                p.result.binary_probability for p in binary_predictions
            ) / len(binary_predictions)
        else:
            weighted_prob = sum(
                p.result.binary_probability * (weight / total_weight)
                for p, weight in zip(binary_predictions, confidence_weights)
            )

        # Ensure probability is within valid range
        weighted_prob = max(0.0, min(1.0, weighted_prob))

        from ..value_objects.probability import Probability

        return Probability(weighted_prob)

    def _calculate_average_confidence(
        self, predictions: List[Prediction]
    ) -> PredictionConfidence:
        """Calculate average confidence level from predictions."""
        confidence_scores = [p.get_confidence_score() for p in predictions]
        avg_score = sum(confidence_scores) / len(confidence_scores)
        return self._score_to_confidence(avg_score)

    def _score_to_confidence(self, score: float) -> PredictionConfidence:
        """Convert numeric confidence score to confidence enum."""
        if score <= 0.2:
            return PredictionConfidence.VERY_LOW
        elif score <= 0.4:
            return PredictionConfidence.LOW
        elif score <= 0.6:
            return PredictionConfidence.MEDIUM
        elif score <= 0.8:
            return PredictionConfidence.HIGH
        else:
            return PredictionConfidence.VERY_HIGH

    def validate_forecast_quality(self, forecast: Forecast) -> Dict[str, Any]:
        """
        Validate the quality of a forecast before submission.

        Returns:
            Dictionary with validation results and quality metrics
        """
        quality_metrics = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "quality_score": 0.0,
        }

        # Check if we have research reports
        if not forecast.research_reports:
            quality_metrics["issues"].append("No research reports provided")
            quality_metrics["is_valid"] = False

        # Check if we have predictions
        if not forecast.predictions:
            quality_metrics["issues"].append("No predictions provided")
            quality_metrics["is_valid"] = False

        # Check prediction variance for consensus
        variance = forecast.calculate_prediction_variance()
        if variance > 0.1:  # High variance threshold
            quality_metrics["warnings"].append(
                f"High prediction variance ({variance:.3f}) - low consensus"
            )

        # Check research quality
        research_quality_scores = [
            report.confidence_level for report in forecast.research_reports
        ]
        avg_research_quality = (
            sum(research_quality_scores) / len(research_quality_scores)
            if research_quality_scores
            else 0
        )

        if avg_research_quality < 0.5:
            quality_metrics["warnings"].append("Low average research quality")

        # Calculate overall quality score
        base_score = 0.5
        if forecast.research_reports:
            base_score += 0.2
        if len(forecast.predictions) >= 3:
            base_score += 0.1
        if variance < 0.05:  # High consensus
            base_score += 0.1
        if avg_research_quality > 0.7:
            base_score += 0.1

        quality_metrics["quality_score"] = min(1.0, base_score)

        return quality_metrics
