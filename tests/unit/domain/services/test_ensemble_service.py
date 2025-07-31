"""
Unit tests for EnsembleService.
Tests aggregation methods, prediction combination, and ensemble configuration.
"""

import pytest
import statistics
from uuid import uuid4
from datetime import datetime, timezone

from src.domain.services.ensemble_service import EnsembleService
from src.domain.entities.prediction import Prediction, PredictionMethod, PredictionConfidence
from src.domain.value_objects.probability import Probability
from src.domain.value_objects.confidence import ConfidenceLevel


class TestEnsembleService:
    """Test suite for EnsembleService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = EnsembleService()
        self.question_id = uuid4()

    def test_simple_initialization(self):
        """Test basic initialization."""
        service = EnsembleService()
        assert service is not None

    def test_get_supported_methods(self):
        """Test getting supported methods."""
        service = EnsembleService()
        methods = service.get_supported_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0

    def test_validate_config_empty(self):
        """Test config validation with empty config."""
        service = EnsembleService()
        result = service.validate_ensemble_config({})
        assert result is True
        
    def test_aggregate_predictions_simple_average(self):
        """Test simple average aggregation."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.8,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second prediction",
                created_by="Agent2"
            )
        ]
        
        result = self.service.aggregate_predictions(predictions, method="simple_average")
        
        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        assert result.result.binary_probability == 0.7  # (0.6 + 0.8) / 2
        assert "simple_average" in result.reasoning
        
    def test_aggregate_predictions_median(self):
        """Test median aggregation."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.3,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High prediction",
                created_by="Agent2"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.REACT,
                reasoning="Medium prediction",
                created_by="Agent3"
            )
        ]
        
        result = self.service.aggregate_predictions(predictions, method="median")
        
        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        assert result.result.binary_probability == 0.5  # median of [0.3, 0.5, 0.7]
        assert "median" in result.reasoning

    def test_aggregate_predictions_weighted_average(self):
        """Test weighted average aggregation with explicit weights."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.2,
                confidence=PredictionConfidence.LOW,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low confidence prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.8,
                confidence=PredictionConfidence.VERY_HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High confidence prediction",
                created_by="Agent2"
            )
        ]
        
        # Give more weight to high confidence prediction
        weights = [0.2, 0.8]
        
        result = self.service.aggregate_predictions(predictions, method="weighted_average", weights=weights)
        
        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        # Weighted average: 0.2 * 0.2 + 0.8 * 0.8 = 0.04 + 0.64 = 0.68
        assert abs(result.result.binary_probability - 0.68) < 0.001
        assert "weighted_average" in result.reasoning

    def test_aggregate_predictions_trimmed_mean(self):
        """Test trimmed mean aggregation."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.1,  # Outlier
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Outlier low",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.4,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Normal prediction",
                created_by="Agent2"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.REACT,
                reasoning="Normal prediction",
                created_by="Agent3"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.AUTO_COT,
                reasoning="Normal prediction",
                created_by="Agent4"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.9,  # Outlier
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.SELF_CONSISTENCY,
                reasoning="Outlier high",
                created_by="Agent5"
            )
        ]
        
        result = self.service.aggregate_predictions(predictions, method="trimmed_mean")
        
        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        # Should exclude outliers (0.1, 0.9) and average [0.4, 0.5, 0.6] = 0.5
        assert abs(result.result.binary_probability - 0.5) < 0.001
        assert "trimmed_mean" in result.reasoning

    def test_aggregate_predictions_confidence_weighted(self):
        """Test confidence-weighted aggregation."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.3,
                confidence=PredictionConfidence.LOW,  # 0.2 confidence
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low confidence prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.VERY_HIGH,  # 0.9 confidence
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High confidence prediction",
                created_by="Agent2"
            )
        ]
        
        result = self.service.aggregate_predictions(predictions, method="confidence_weighted")
        
        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        # Should weight toward high confidence prediction
        assert result.result.binary_probability > 0.5
        assert "confidence_weighted" in result.reasoning

    def test_aggregate_predictions_performance_weighted(self):
        """Test performance-weighted aggregation (fallback to equal weights)."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.3,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second prediction",
                created_by="Agent2"
            )
        ]
        
        result = self.service.aggregate_predictions(predictions, method="performance_weighted")
        
        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        # Should be equal weights (0.3 + 0.7) / 2 = 0.5
        assert abs(result.result.binary_probability - 0.5) < 0.001
        assert "performance_weighted" in result.reasoning

    def test_aggregate_predictions_empty_list(self):
        """Test error handling for empty prediction list."""
        with pytest.raises(ValueError, match="Cannot aggregate empty prediction list"):
            self.service.aggregate_predictions([])

    def test_aggregate_predictions_mismatched_question_ids(self):
        """Test error handling for predictions with different question IDs."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=uuid4(),  # Different question ID
                research_report_id=uuid4(),
                probability=0.3,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=uuid4(),  # Different question ID
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second prediction",
                created_by="Agent2"
            )
        ]
        
        with pytest.raises(ValueError, match="All predictions must be for the same question"):
            self.service.aggregate_predictions(predictions)

    def test_aggregate_predictions_no_valid_probabilities(self):
        """Test error handling when no predictions have valid binary probabilities."""
        # Create predictions with None binary probabilities (shouldn't happen in practice)
        prediction = Prediction.create_binary_prediction(
            question_id=self.question_id,
            research_report_id=uuid4(),
            probability=0.5,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Test prediction",
            created_by="Agent1"
        )
        # Manually set binary_probability to None to test edge case
        prediction.result.binary_probability = None
        
        with pytest.raises(ValueError, match="No valid binary probabilities found in predictions"):
            self.service.aggregate_predictions([prediction])

    def test_aggregate_predictions_invalid_method(self):
        """Test error handling for unsupported aggregation method."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Test prediction",
                created_by="Agent1"
            )
        ]
        
        with pytest.raises(ValueError, match="Unsupported aggregation method: invalid_method"):
            self.service.aggregate_predictions(predictions, method="invalid_method")

    def test_calculate_default_weights(self):
        """Test default weight calculation based on confidence."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.3,
                confidence=PredictionConfidence.LOW,  # 0.2
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low confidence",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,  # 0.8
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High confidence",
                created_by="Agent2"
            )
        ]
        
        weights = self.service._calculate_default_weights(predictions)
        
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 0.001  # Should sum to 1
        assert weights[1] > weights[0]  # High confidence should have higher weight

    def test_calculate_default_weights_zero_confidence(self):
        """Test default weight calculation when all predictions have zero confidence."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.3,
                confidence=PredictionConfidence.VERY_LOW,  # Close to 0
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Very low confidence",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.VERY_LOW,  # Close to 0
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Very low confidence",
                created_by="Agent2"
            )
        ]
        
        weights = self.service._calculate_default_weights(predictions)
        
        assert len(weights) == 2
        assert abs(weights[0] - 0.5) < 0.001  # Should be equal weights
        assert abs(weights[1] - 0.5) < 0.001

    def test_weighted_average_helper(self):
        """Test weighted average calculation helper method."""
        values = [0.2, 0.8]
        weights = [0.3, 0.7]
        
        result = self.service._weighted_average(values, weights)
        
        expected = (0.2 * 0.3 + 0.8 * 0.7) / (0.3 + 0.7)
        assert abs(result - expected) < 0.001

    def test_weighted_average_helper_zero_weights(self):
        """Test weighted average helper with zero weights (fallback to simple mean)."""
        values = [0.2, 0.8]
        weights = [0.0, 0.0]
        
        result = self.service._weighted_average(values, weights)
        
        assert abs(result - 0.5) < 0.001  # Should be simple mean

    def test_weighted_average_helper_mismatched_lengths(self):
        """Test weighted average helper with mismatched value/weight lengths."""
        values = [0.2, 0.8]
        weights = [0.3]  # Wrong length
        
        with pytest.raises(ValueError, match="Values and weights must have same length"):
            self.service._weighted_average(values, weights)

    def test_trimmed_mean_helper(self):
        """Test trimmed mean calculation helper method."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        result = self.service._trimmed_mean(values, trim_percent=0.2)
        
        # Should trim 20% from each end: remove 0.1, 0.2, 0.8, 0.9
        # Mean of [0.3, 0.4, 0.5, 0.6, 0.7] = 0.5
        assert abs(result - 0.5) < 0.001

    def test_trimmed_mean_helper_small_sample(self):
        """Test trimmed mean with small sample (no trimming)."""
        values = [0.3, 0.7]
        
        result = self.service._trimmed_mean(values, trim_percent=0.1)
        
        # Too few values to trim, should return simple mean
        assert abs(result - 0.5) < 0.001

    def test_trimmed_mean_helper_zero_trim(self):
        """Test trimmed mean with zero trim count."""
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        result = self.service._trimmed_mean(values, trim_percent=0.05)  # 5% of 5 = 0.25, rounds to 0
        
        # No trimming, should return simple mean
        assert abs(result - 0.5) < 0.001

    def test_calculate_ensemble_confidence(self):
        """Test ensemble confidence calculation."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.45,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.55,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second prediction",
                created_by="Agent2"
            )
        ]
        
        confidence = self.service._calculate_ensemble_confidence(predictions, "simple_average")
        
        assert isinstance(confidence, PredictionConfidence)
        # Should be high confidence due to high individual confidences and low variance

    def test_create_ensemble_reasoning(self):
        """Test ensemble reasoning generation."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.3,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First reasoning",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second reasoning",
                created_by="Agent2"
            )
        ]
        
        reasoning = self.service._create_ensemble_reasoning(predictions, "weighted_average", 0.5)
        
        assert isinstance(reasoning, str)
        assert "weighted_average" in reasoning
        assert "2 predictions" in reasoning
        assert "0.5" in reasoning

    def test_validate_config_with_valid_settings(self):
        """Test config validation with valid settings."""
        valid_config = {
            "aggregation_method": "weighted_average",
            "weights": [0.3, 0.7],
            "min_predictions": 2
        }
        
        result = self.service.validate_ensemble_config(valid_config)
        assert result is True

    def test_validate_config_invalid_method(self):
        """Test config validation with invalid aggregation method."""
        invalid_config = {
            "aggregation_method": "invalid_method"
        }
        
        result = self.service.validate_ensemble_config(invalid_config)
        assert result is False

    def test_validate_config_invalid_weights(self):
        """Test config validation with invalid weights."""
        invalid_config = {
            "weights": "not_a_list"
        }
        
        result = self.service.validate_ensemble_config(invalid_config)
        assert result is False

    def test_validate_config_invalid_min_predictions(self):
        """Test config validation with invalid min_predictions."""
        invalid_config = {
            "min_predictions": 0
        }
        
        result = self.service.validate_ensemble_config(invalid_config)
        assert result is False

    def test_get_supported_agent_types(self):
        """Test getting supported agent types."""
        agent_types = self.service.get_supported_agent_types()
        
        assert isinstance(agent_types, list)
        assert len(agent_types) > 0
        assert "chain_of_thought" in agent_types
        assert "tree_of_thought" in agent_types

    def test_evaluate_ensemble_performance_basic(self):
        """Test basic ensemble performance evaluation."""
        # Create ensemble predictions
        ensemble_predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.ENSEMBLE,
                reasoning="Ensemble reasoning",
                created_by="EnsembleService"
            )
        ]
        
        # Create individual predictions for the same question
        individual_predictions = [
            [Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Agent 1",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.8,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Agent 2",
                created_by="Agent2"
            )]
        ]
        
        metrics = self.service.evaluate_ensemble_performance(
            ensemble_predictions=ensemble_predictions,
            individual_predictions=individual_predictions
        )
        
        assert "ensemble_count" in metrics
        assert "average_input_predictions" in metrics
        assert "diversity_metrics" in metrics
        assert "confidence_metrics" in metrics
        assert metrics["ensemble_count"] == 1
        assert metrics["average_input_predictions"] == 2.0

    def test_evaluate_ensemble_performance_with_ground_truth(self):
        """Test ensemble performance evaluation with ground truth."""
        ensemble_predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.ENSEMBLE,
                reasoning="Ensemble reasoning",
                created_by="EnsembleService"
            )
        ]
        
        individual_predictions = [
            [Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Agent 1",
                created_by="Agent1"
            )]
        ]
        
        ground_truth = [True]
        
        metrics = self.service.evaluate_ensemble_performance(
            ensemble_predictions=ensemble_predictions,
            individual_predictions=individual_predictions,
            ground_truth=ground_truth
        )
        
        assert "accuracy_metrics" in metrics
        assert "brier_score" in metrics["accuracy_metrics"]
        assert "mean_absolute_error" in metrics["accuracy_metrics"]

    def test_evaluate_ensemble_performance_mismatched_lengths(self):
        """Test ensemble performance evaluation with mismatched lengths."""
        ensemble_predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.ENSEMBLE,
                reasoning="Ensemble reasoning",
                created_by="EnsembleService"
            )
        ]
        
        individual_predictions = [
            [],  # Empty list - mismatched length
            []
        ]
        
        with pytest.raises(ValueError, match="Ensemble and individual predictions lists must have same length"):
            self.service.evaluate_ensemble_performance(
                ensemble_predictions=ensemble_predictions,
                individual_predictions=individual_predictions
            )

    def test_evaluate_ensemble_performance_ground_truth_mismatch(self):
        """Test ensemble performance evaluation with mismatched ground truth."""
        ensemble_predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.ENSEMBLE,
                reasoning="Ensemble reasoning",
                created_by="EnsembleService"
            )
        ]
        
        individual_predictions = [
            [Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Agent 1",
                created_by="Agent1"
            )]
        ]
        
        ground_truth = [True, False]  # Mismatched length
        
        with pytest.raises(ValueError, match="Ground truth must match ensemble predictions length"):
            self.service.evaluate_ensemble_performance(
                ensemble_predictions=ensemble_predictions,
                individual_predictions=individual_predictions,
                ground_truth=ground_truth
            )

    def test_create_ensemble_reasoning_variance_scenarios(self):
        """Test ensemble reasoning with different variance scenarios."""
        # High agreement (low variance)
        high_agreement_predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.71,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.70,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second prediction",
                created_by="Agent2"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.69,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.REACT,
                reasoning="Third prediction",
                created_by="Agent3"
            )
        ]
        
        reasoning = self.service._create_ensemble_reasoning(
            predictions=high_agreement_predictions,
            method="simple_average",
            final_probability=0.7
        )
        
        assert "High agreement between predictions increases confidence" in reasoning
        
        # Low agreement (high variance)
        low_agreement_predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.9,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="High prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.1,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Low prediction",
                created_by="Agent2"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.REACT,
                reasoning="Medium prediction",
                created_by="Agent3"
            )
        ]
        
        reasoning = self.service._create_ensemble_reasoning(
            predictions=low_agreement_predictions,
            method="simple_average",
            final_probability=0.5
        )
        
        assert "Low agreement between predictions suggests higher uncertainty" in reasoning
        
        # Moderate agreement
        moderate_agreement_predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.75,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Above average prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.65,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Average prediction",
                created_by="Agent2"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.55,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.REACT,
                reasoning="Below average prediction",
                created_by="Agent3"
            )
        ]
        
        reasoning = self.service._create_ensemble_reasoning(
            predictions=moderate_agreement_predictions,
            method="simple_average",
            final_probability=0.65
        )
        
        assert "Moderate agreement between predictions" in reasoning

    def test_create_ensemble_reasoning_single_prediction(self):
        """Test ensemble reasoning with single prediction (no variance calculation)."""
        single_prediction = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Single prediction",
                created_by="Agent1"
            )
        ]
        
        reasoning = self.service._create_ensemble_reasoning(
            predictions=single_prediction,
            method="simple_average",
            final_probability=0.7
        )
        
        # Should not include standard deviation or range for single prediction
        assert "Standard deviation" not in reasoning
        assert "Range" not in reasoning
        assert "Final ensemble probability: 0.700" in reasoning

    def test_calculate_ensemble_confidence_very_low(self):
        """Test ensemble confidence calculation for very low confidence."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.VERY_LOW,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Very low confidence prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.VERY_LOW,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Very low confidence prediction",
                created_by="Agent2"
            )
        ]
        
        confidence = self.service._calculate_ensemble_confidence(predictions, "simple_average")
        assert confidence == PredictionConfidence.VERY_LOW

    def test_calculate_ensemble_confidence_low(self):
        """Test ensemble confidence calculation with low confidence inputs gets boosted."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.LOW,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low confidence prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Medium confidence prediction",
                created_by="Agent2"
            )
        ]
        
        confidence = self.service._calculate_ensemble_confidence(predictions, "simple_average")
        # Ensemble should boost confidence when predictions agree
        assert confidence == PredictionConfidence.MEDIUM

    def test_calculate_ensemble_confidence_medium(self):
        """Test ensemble confidence calculation with medium confidence inputs gets boosted."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Medium confidence prediction",
                created_by="Agent1"
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High confidence prediction",
                created_by="Agent2"
            )
        ]
        
        confidence = self.service._calculate_ensemble_confidence(predictions, "simple_average")
        # Ensemble should boost confidence when predictions agree  
        assert confidence == PredictionConfidence.HIGH
