"""
Unit tests for advanced EnsembleService aggregation methods.
Tests sophisticated aggregation methods and method selection.
"""

import statistics
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
)
from src.domain.services.ensemble_service import EnsembleService
from src.domain.value_objects.confidence import ConfidenceLevel
from src.domain.value_objects.probability import Probability


class TestEnsembleServiceAdvanced:
    """Test suite for advanced EnsembleService methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = EnsembleService()
        self.question_id = uuid4()

    def test_meta_reasoning_aggregation(self):
        """Test meta-reasoning aggregation method."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="According to research shows that evidence suggests this outcome is likely because of multiple factors. Therefore, the probability is moderate.",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.8,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Simple reasoning without much detail.",
                created_by="Agent2",
            ),
        ]

        result = self.service.aggregate_predictions(
            predictions, method="meta_reasoning"
        )

        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        assert 0.0 <= result.result.binary_probability <= 1.0
        assert "meta_reasoning" in result.reasoning

    def test_bayesian_model_averaging(self):
        """Test Bayesian model averaging method."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.3,
                confidence=PredictionConfidence.LOW,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low confidence prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.VERY_HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High confidence prediction",
                created_by="Agent2",
            ),
        ]

        result = self.service.aggregate_predictions(
            predictions, method="bayesian_model_averaging"
        )

        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        # Should be weighted toward high confidence prediction
        assert result.result.binary_probability > 0.5
        assert "bayesian_model_averaging" in result.reasoning

    def test_stacked_generalization(self):
        """Test stacked generalization method."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.4,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second prediction",
                created_by="Agent2",
            ),
        ]

        result = self.service.aggregate_predictions(
            predictions, method="stacked_generalization"
        )

        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        assert 0.0 <= result.result.binary_probability <= 1.0
        assert "stacked_generalization" in result.reasoning

    def test_dynamic_selection_aggregation(self):
        """Test dynamic selection aggregation method."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.45,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.55,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second prediction",
                created_by="Agent2",
            ),
        ]

        result = self.service.aggregate_predictions(
            predictions, method="dynamic_selection"
        )

        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        assert 0.0 <= result.result.binary_probability <= 1.0
        assert "dynamic_selection" in result.reasoning

    def test_outlier_robust_mean(self):
        """Test outlier-robust mean method."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.05,  # Outlier
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Outlier prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.45,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Normal prediction",
                created_by="Agent2",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.55,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.REACT,
                reasoning="Normal prediction",
                created_by="Agent3",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.95,  # Outlier
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.AUTO_COT,
                reasoning="Outlier prediction",
                created_by="Agent4",
            ),
        ]

        result = self.service.aggregate_predictions(
            predictions, method="outlier_robust_mean"
        )

        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        # Should be closer to the median, less influenced by outliers
        assert 0.4 <= result.result.binary_probability <= 0.6
        assert "outlier_robust_mean" in result.reasoning

    def test_entropy_weighted_aggregation(self):
        """Test entropy-weighted aggregation method."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.1,  # Low entropy (high certainty)
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Very certain prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,  # High entropy (high uncertainty)
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Uncertain prediction",
                created_by="Agent2",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,  # Moderate entropy
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.REACT,
                reasoning="Moderately certain prediction",
                created_by="Agent3",
            ),
        ]

        result = self.service.aggregate_predictions(
            predictions, method="entropy_weighted"
        )

        assert isinstance(result, Prediction)
        assert result.method == PredictionMethod.ENSEMBLE
        assert 0.0 <= result.result.binary_probability <= 1.0
        assert "entropy_weighted" in result.reasoning

    def test_evaluate_reasoning_quality(self):
        """Test reasoning quality evaluation."""
        # High quality reasoning
        high_quality_pred = Prediction.create_binary_prediction(
            question_id=self.question_id,
            research_report_id=uuid4(),
            probability=0.6,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="According to recent research shows that data indicates multiple studies found evidence suggests this outcome is likely. Therefore, because of these factors, however there might be uncertainty in the analysis.",
            created_by="Agent1",
        )

        # Low quality reasoning
        low_quality_pred = Prediction.create_binary_prediction(
            question_id=self.question_id,
            research_report_id=uuid4(),
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.TREE_OF_THOUGHT,
            reasoning="Yes.",
            created_by="Agent2",
        )

        high_score = self.service._evaluate_reasoning_quality(high_quality_pred)
        low_score = self.service._evaluate_reasoning_quality(low_quality_pred)

        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0

    def test_calculate_consensus_adjustment(self):
        """Test consensus adjustment calculation."""
        # High agreement predictions
        high_agreement = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.60,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.61,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second prediction",
                created_by="Agent2",
            ),
        ]

        # Low agreement predictions
        low_agreement = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.2,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="First prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.8,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Second prediction",
                created_by="Agent2",
            ),
        ]

        high_adjustment = self.service._calculate_consensus_adjustment(high_agreement)
        low_adjustment = self.service._calculate_consensus_adjustment(low_agreement)

        assert high_adjustment >= low_adjustment
        assert -0.1 <= high_adjustment <= 0.1
        assert -0.1 <= low_adjustment <= 0.1

    def test_select_optimal_aggregation_method(self):
        """Test optimal aggregation method selection."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Test prediction",
                created_by="Agent1",
            )
        ]

        # Test different selection strategies
        method1 = self.service.select_optimal_aggregation_method(
            predictions, "best_recent"
        )
        method2 = self.service.select_optimal_aggregation_method(
            predictions, "adaptive_threshold"
        )
        method3 = self.service.select_optimal_aggregation_method(
            predictions, "diversity_based"
        )

        assert method1 in self.service.aggregation_methods
        assert method2 in self.service.aggregation_methods
        assert method3 in self.service.aggregation_methods

    def test_select_best_recent_method(self):
        """Test best recent method selection."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Test prediction",
                created_by="Agent1",
            )
        ]

        # With no history, should return default
        method = self.service._select_best_recent_method(predictions)
        assert method == "meta_reasoning"

        # Add some performance history
        self.service.update_method_performance("confidence_weighted", 0.8)
        self.service.update_method_performance("simple_average", 0.6)

        method = self.service._select_best_recent_method(predictions)
        assert method == "confidence_weighted"

    def test_select_adaptive_threshold_method(self):
        """Test adaptive threshold method selection."""
        # High disagreement predictions
        high_disagreement = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.1,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.9,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High prediction",
                created_by="Agent2",
            ),
        ]

        method = self.service._select_adaptive_threshold_method(high_disagreement)
        assert method == "outlier_robust_mean"

        # High confidence predictions
        high_confidence = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.VERY_HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="High confidence prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.VERY_HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High confidence prediction",
                created_by="Agent2",
            ),
        ]

        method = self.service._select_adaptive_threshold_method(high_confidence)
        assert method == "confidence_weighted"

    def test_select_diversity_based_method(self):
        """Test diversity-based method selection."""
        # Low diversity predictions
        low_diversity = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.50,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Similar prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.51,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Similar prediction",
                created_by="Agent2",
            ),
        ]

        method = self.service._select_diversity_based_method(low_diversity)
        assert method == "simple_average"

        # High diversity predictions
        high_diversity = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.1,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.9,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High prediction",
                created_by="Agent2",
            ),
        ]

        method = self.service._select_diversity_based_method(high_diversity)
        assert method == "stacked_generalization"

    def test_update_method_performance(self):
        """Test method performance tracking."""
        method_name = "meta_reasoning"

        # Add performance scores
        self.service.update_method_performance(method_name, 0.8)
        self.service.update_method_performance(method_name, 0.7)
        self.service.update_method_performance(method_name, 0.9)

        assert method_name in self.service.method_performance_history
        assert len(self.service.method_performance_history[method_name]) == 3
        assert self.service.method_performance_history[method_name] == [0.8, 0.7, 0.9]

    def test_update_agent_performance(self):
        """Test agent performance tracking."""
        agent_name = "Agent1"

        # Add performance scores
        self.service.update_agent_performance(agent_name, 0.85)
        self.service.update_agent_performance(agent_name, 0.75)

        assert agent_name in self.service.agent_performance_history
        assert len(self.service.agent_performance_history[agent_name]) == 2
        assert self.service.agent_performance_history[agent_name] == [0.85, 0.75]

    def test_performance_history_limit(self):
        """Test that performance history is limited to recent entries."""
        method_name = "test_method"

        # Add more than 50 entries
        for i in range(55):
            self.service.update_method_performance(method_name, 0.5 + i * 0.01)

        # Should keep only last 50
        assert len(self.service.method_performance_history[method_name]) == 50
        assert (
            self.service.method_performance_history[method_name][0] == 0.55
        )  # First of last 50

    def test_get_method_performance_summary(self):
        """Test method performance summary generation."""
        # Add performance data
        self.service.update_method_performance("method1", 0.8)
        self.service.update_method_performance("method1", 0.7)
        self.service.update_method_performance("method1", 0.9)

        self.service.update_method_performance("method2", 0.6)
        self.service.update_method_performance("method2", 0.5)

        summary = self.service.get_method_performance_summary()

        assert "method1" in summary
        assert "method2" in summary

        method1_summary = summary["method1"]
        assert "mean_performance" in method1_summary
        assert "recent_performance" in method1_summary
        assert "performance_std" in method1_summary
        assert "prediction_count" in method1_summary

        assert method1_summary["mean_performance"] == pytest.approx(0.8, abs=0.01)
        assert method1_summary["prediction_count"] == 3

    def test_calculate_diversity_factor(self):
        """Test diversity factor calculation."""
        # Low diversity
        low_diversity_probs = [0.5, 0.51, 0.49]
        low_factor = self.service._calculate_diversity_factor(low_diversity_probs)

        # High diversity
        high_diversity_probs = [0.1, 0.5, 0.9]
        high_factor = self.service._calculate_diversity_factor(high_diversity_probs)

        assert high_factor > low_factor
        assert 0.0 <= low_factor <= 1.0
        assert 0.0 <= high_factor <= 1.0

    def test_new_methods_in_supported_list(self):
        """Test that new methods are included in supported methods list."""
        supported_methods = self.service.get_supported_methods()

        new_methods = [
            "meta_reasoning",
            "bayesian_model_averaging",
            "stacked_generalization",
            "dynamic_selection",
            "outlier_robust_mean",
            "entropy_weighted",
        ]

        for method in new_methods:
            assert method in supported_methods

    def test_invalid_selection_strategy(self):
        """Test handling of invalid selection strategy."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Test prediction",
                created_by="Agent1",
            )
        ]

        # Should fall back to best_recent
        method = self.service.select_optimal_aggregation_method(
            predictions, "invalid_strategy"
        )
        assert method in self.service.aggregation_methods

    def test_empty_predictions_handling_in_new_methods(self):
        """Test that new methods handle empty predictions gracefully."""
        # Test with empty probabilities (edge case)
        prediction = Prediction.create_binary_prediction(
            question_id=self.question_id,
            research_report_id=uuid4(),
            probability=0.5,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Test prediction",
            created_by="Agent1",
        )

        # Manually set binary_probability to None to test edge case
        prediction.result.binary_probability = None

        # These should handle the edge case gracefully
        result1 = self.service._meta_reasoning_aggregation([prediction])
        result2 = self.service._bayesian_model_averaging([prediction])
        result3 = self.service._entropy_weighted_aggregation([prediction])

        assert result1 == 0.5  # Default fallback
        assert result2 == 0.5  # Default fallback
        assert result3 == 0.5  # Default fallback
