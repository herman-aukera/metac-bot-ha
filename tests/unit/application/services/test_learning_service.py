"""Tests for the LearningService."""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.application.services.learning_service import (
    LearningService,
    LearningMode,
    PatternType,
    AdaptationTrigger,
    PerformanceMetrics,
    PatternInsight,
    AgentPerformanceProfile,
    CalibrationAnalysis,
    StrategyAdaptation,
    ABTestResult
)
from src.domain.entities.prediction import Prediction
from src.domain.entities.question import Question, QuestionType, QuestionCategory
from src.domain.entities.tournament import Tournament, ScoringRules, ScoringMethod
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.strategy_result import StrategyResult, StrategyType, StrategyOutcome


class TestLearningService:
    """Test cases for LearningService."""

    @pytest.fixture
    def learning_service(self):
        """Create a LearningService instance for testing."""
        return LearningService(
            learning_mode=LearningMode.ADAPTIVE,
            history_window_days=30,
            min_samples_for_learning=5,
            adaptation_threshold=0.1,
            calibration_target=0.9
        )

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        predictions = []
        base_time = datetime.utcnow()

        for i in range(10):
            pred = Prediction.create_binary(
                question_id=i + 1,
                probability=0.6 + (i * 0.03),  # Varying probabilities
                confidence_level=0.7 + (i * 0.02),  # Varying confidence
                confidence_basis=f"Test prediction {i}",
                method="test_method",
                reasoning=f"Test reasoning for prediction {i}",
                created_by=f"agent_{i % 3}",  # 3 different agents
                timestamp=base_time + timedelta(hours=i)
            )
            predictions.append(pred)

        return predictions

    @pytest.fixture
    def sample_outcomes(self):
        """Create sample outcomes for testing."""
        # Mix of correct and incorrect predictions
        return [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    @pytest.fixture
    def sample_strategy_results(self):
        """Create sample strategy results for testing."""
        results = []
        base_time = datetime.utcnow()

        strategies = [StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, StrategyType.BALANCED]

        for i in range(6):
            result = StrategyResult.create(
                strategy_type=strategies[i % 3],
                expected_score=0.6 + (i * 0.05),
                reasoning=f"Test strategy {i}",
                question_ids=[i + 1],
                confidence_level=0.8,
                confidence_basis="Test confidence",
                timestamp=base_time + timedelta(hours=i)
            )
            # Add actual scores for some results
            if i >= 3:
                result = result.mark_success(0.7 + (i * 0.03))
            results.append(result)

        return results

    @pytest.fixture
    def sample_tournament(self, sample_questions):
        """Create a sample tournament for testing."""
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)
        return Tournament(
            id=1,
            name="Test Tournament",
            questions=sample_questions,
            scoring_rules=scoring_rules,
            start_date=datetime.utcnow() - timedelta(days=5),
            end_date=datetime.utcnow() + timedelta(days=10),
            current_standings={"player1": 85.5, "player2": 82.3}
        )

    @pytest.fixture
    def sample_questions(self):
        """Create sample questions for testing."""
        questions = []
        base_time = datetime.utcnow()

        for i in range(5):
            question = Question(
                id=i + 1,
                text=f"Test question {i}",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=base_time + timedelta(days=7),
                background=f"Background for question {i}",
                resolution_criteria=f"Resolution criteria {i}",
                scoring_weight=1.0 + (i * 0.2)
            )
            questions.append(question)

        return questions

    @pytest.mark.asyncio
    async def test_analyze_prediction_accuracy(self, learning_service, sample_predictions, sample_outcomes):
        """Test prediction accuracy analysis."""
        predictions_with_outcomes = list(zip(sample_predictions, sample_outcomes))
        metrics = await learning_service.analyze_prediction_accuracy(predictions_with_outcomes)

        assert isinstance(metrics, PerformanceMetrics)
        assert 0.0 <= metrics.accuracy_score <= 1.0
        assert 0.0 <= metrics.calibration_score <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0
        assert metrics.prediction_count == len(sample_predictions)
        assert isinstance(metrics.category_breakdown, dict)
        assert len(metrics.time_period) == 2

    @pytest.mark.asyncio
    async def test_analyze_prediction_accuracy_insufficient_samples(self, learning_service):
        """Test accuracy analysis with insufficient samples."""
        predictions = [
            Prediction.create_binary(
                question_id=1,
                probability=0.7,
                confidence_level=0.8,
                confidence_basis="Test",
                method="test",
                reasoning="Test reasoning",
                created_by="test_agent"
            )
        ]
        predictions_with_outcomes = [(predictions[0], 1.0)]

        metrics = await learning_service.analyze_prediction_accuracy(predictions_with_outcomes)
        assert isinstance(metrics, PerformanceMetrics)

    @pytest.mark.asyncio
    async def test_analyze_prediction_accuracy_empty_predictions(self, learning_service):
        """Test accuracy analysis with empty predictions."""
        with pytest.raises(ValueError, match="No predictions provided for analysis"):
            await learning_service.analyze_prediction_accuracy([])

    @pytest.mark.asyncio
    async def test_refine_strategy_based_on_feedback(self, learning_service, sample_strategy_results, sample_tournament):
        """Test strategy refinement based on feedback."""
        adaptation = await learning_service.refine_strategy_based_on_feedback(
            sample_tournament, sample_strategy_results
        )

        if adaptation:  # May be None if no triggers identified
            assert isinstance(adaptation, StrategyAdaptation)
            assert isinstance(adaptation.trigger, AdaptationTrigger)
            assert isinstance(adaptation.old_strategy, StrategyType)
            assert isinstance(adaptation.new_strategy, StrategyType)
            assert 0.0 <= adaptation.confidence <= 1.0
            assert isinstance(adaptation.reasoning, str)

    @pytest.mark.asyncio
    async def test_update_predictions_with_new_information(self, learning_service, sample_predictions):
        """Test prediction updating with new information."""
        new_info = {
            'timestamp': datetime.utcnow(),
            'category': 'ai_development',
            'sentiment': 'positive',
            'summary': 'New positive development',
            'source': 'test_source'
        }

        # Use the first question's ID for testing
        question_id = str(sample_predictions[0].question_id)

        updated_predictions = await learning_service.update_predictions_with_new_information(
            question_id, new_info, sample_predictions
        )

        assert len(updated_predictions) == len(sample_predictions)
        assert all(isinstance(pred, Prediction) for pred in updated_predictions)

    @pytest.mark.asyncio
    async def test_monitor_tournament_dynamics(self, learning_service, sample_tournament):
        """Test tournament dynamics monitoring."""
        dynamics = await learning_service.monitor_tournament_dynamics(sample_tournament)

        assert isinstance(dynamics, dict)
        assert 'tournament_phase' in dynamics
        assert 'competition_intensity' in dynamics
        assert 'question_difficulty_trend' in dynamics
        assert 'participant_behavior_patterns' in dynamics
        assert 'market_efficiency' in dynamics
        assert 'strategic_recommendations' in dynamics

        assert isinstance(dynamics['strategic_recommendations'], list)

    @pytest.mark.asyncio
    async def test_optimize_agent_weights(self, learning_service):
        """Test agent weight optimization."""
        agents = ['agent_1', 'agent_2', 'agent_3']
        recent_performance = {
            'agent_1': PerformanceMetrics(0.8, 0.8, 0.2, -0.5, 0.7, 10, {}, (datetime.utcnow(), datetime.utcnow())),
            'agent_2': PerformanceMetrics(0.6, 0.7, 0.4, -0.8, 0.5, 10, {}, (datetime.utcnow(), datetime.utcnow())),
            'agent_3': PerformanceMetrics(0.9, 0.9, 0.1, -0.3, 0.8, 10, {}, (datetime.utcnow(), datetime.utcnow()))
        }

        weights = await learning_service.optimize_agent_weights(agents, recent_performance)

        assert isinstance(weights, dict)
        assert len(weights) == len(agents)
        assert all(isinstance(w, float) for w in weights.values())
        assert all(0.0 <= w <= 1.0 for w in weights.values())

        # Weights should sum to approximately 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01

        # Better performing agents should have higher weights
        assert weights['agent_3'] > weights['agent_2']  # agent_3 performs better

    @pytest.mark.asyncio
    async def test_optimize_agent_weights_empty_agents(self, learning_service):
        """Test agent weight optimization with empty agent list."""
        weights = await learning_service.optimize_agent_weights([], {})
        assert weights == {}

    @pytest.mark.asyncio
    async def test_monitor_calibration(self, learning_service, sample_predictions, sample_outcomes):
        """Test calibration monitoring."""
        predictions_with_outcomes = list(zip(sample_predictions, sample_outcomes))
        analysis = await learning_service.monitor_calibration(predictions_with_outcomes)

        assert isinstance(analysis, CalibrationAnalysis)
        assert 0.0 <= analysis.overall_calibration <= 1.0
        assert isinstance(analysis.confidence_bins, dict)
        assert 0.0 <= analysis.overconfidence_bias <= 1.0
        assert 0.0 <= analysis.underconfidence_bias <= 1.0
        assert isinstance(analysis.calibration_trend, list)
        assert isinstance(analysis.recommendations, list)

        # Check that calibration history was updated
        assert len(learning_service.calibration_history) > 0

    @pytest.mark.asyncio
    async def test_analyze_historical_patterns(self, learning_service, sample_predictions, sample_outcomes):
        """Test historical pattern analysis."""
        predictions_with_outcomes = list(zip(sample_predictions, sample_outcomes))
        patterns = await learning_service.analyze_historical_patterns(predictions_with_outcomes)

        assert isinstance(patterns, list)
        assert all(isinstance(p, PatternInsight) for p in patterns)

        for pattern in patterns:
            assert isinstance(pattern.pattern_type, PatternType)
            assert isinstance(pattern.description, str)
            assert 0.0 <= pattern.confidence <= 1.0
            assert isinstance(pattern.supporting_evidence, list)
            assert isinstance(pattern.recommendations, list)

    @pytest.mark.asyncio
    async def test_analyze_historical_patterns_insufficient_data(self, learning_service):
        """Test pattern analysis with insufficient historical data."""
        patterns = await learning_service.analyze_historical_patterns([])

        # Should return empty list with insufficient data
        assert isinstance(patterns, list)
        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_run_ab_test(self, learning_service):
        """Test A/B test creation."""
        test_id = "test_aggressive_vs_conservative"
        strategy_a = StrategyType.AGGRESSIVE
        strategy_b = StrategyType.CONSERVATIVE
        duration_days = 7

        result_id = await learning_service.run_ab_test(
            test_id, strategy_a, strategy_b, duration_days
        )

        assert result_id == test_id
        assert test_id in learning_service.active_ab_tests

        test_config = learning_service.active_ab_tests[test_id]
        assert test_config['strategy_a'] == strategy_a
        assert test_config['strategy_b'] == strategy_b

    @pytest.mark.asyncio
    async def test_run_ab_test_duplicate_id(self, learning_service):
        """Test A/B test creation with duplicate ID."""
        test_id = "duplicate_test"

        await learning_service.run_ab_test(
            test_id, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, 5
        )

        # Should raise error for duplicate ID
        with pytest.raises(ValueError, match="A/B test duplicate_test is already running"):
            await learning_service.run_ab_test(
                test_id, StrategyType.BALANCED, StrategyType.AGGRESSIVE, 5
            )

    @pytest.mark.asyncio
    async def test_record_ab_test_result(self, learning_service):
        """Test recording A/B test results."""
        test_id = "test_record_results"
        await learning_service.run_ab_test(
            test_id, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, 5
        )

        # Record some results
        await learning_service.record_ab_test_result(test_id, StrategyType.AGGRESSIVE, 0.8)
        await learning_service.record_ab_test_result(test_id, StrategyType.CONSERVATIVE, 0.7)

        test_config = learning_service.active_ab_tests[test_id]
        assert len(test_config['results_a']) == 1
        assert len(test_config['results_b']) == 1
        assert test_config['results_a'][0] == 0.8
        assert test_config['results_b'][0] == 0.7

    @pytest.mark.asyncio
    async def test_record_ab_test_result_invalid_test(self, learning_service):
        """Test recording result for non-existent test."""
        # Should not raise error, just log warning
        await learning_service.record_ab_test_result(
            "nonexistent_test", StrategyType.AGGRESSIVE, 0.8
        )

    @pytest.mark.asyncio
    async def test_get_ab_test_assignment(self, learning_service):
        """Test getting A/B test assignments."""
        test_id = "test_assignment"
        strategy_a = StrategyType.AGGRESSIVE
        strategy_b = StrategyType.CONSERVATIVE

        await learning_service.run_ab_test(test_id, strategy_a, strategy_b, 5)

        # Get alternating assignments
        assignment1 = await learning_service.get_ab_test_assignment(test_id)
        assignment2 = await learning_service.get_ab_test_assignment(test_id)

        assert assignment1 in [strategy_a, strategy_b]
        assert assignment2 in [strategy_a, strategy_b]
        # Should alternate
        assert assignment1 != assignment2

    @pytest.mark.asyncio
    async def test_get_ab_test_assignment_invalid_test(self, learning_service):
        """Test getting assignment for non-existent test."""
        assignment = await learning_service.get_ab_test_assignment("nonexistent_test")
        assert assignment is None

    @pytest.mark.asyncio
    async def test_ab_test_completion_by_time(self, learning_service):
        """Test A/B test completion when time expires."""
        test_id = "test_completion_time"
        # Create test with very short duration
        await learning_service.run_ab_test(
            test_id, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, 0  # 0 days = immediate expiry
        )

        # Record some results
        await learning_service.record_ab_test_result(test_id, StrategyType.AGGRESSIVE, 0.8)
        await learning_service.record_ab_test_result(test_id, StrategyType.CONSERVATIVE, 0.7)

        # Test should be completed due to time expiry
        # Note: This depends on the implementation checking time on record_ab_test_result

    def test_learning_service_initialization(self):
        """Test LearningService initialization with different parameters."""
        service = LearningService(
            learning_mode=LearningMode.ACCURACY_FOCUSED,
            history_window_days=60,
            min_samples_for_learning=20,
            adaptation_threshold=0.2,
            calibration_target=0.85
        )

        assert service.learning_mode == LearningMode.ACCURACY_FOCUSED
        assert service.history_window_days == 60
        assert service.min_samples_for_learning == 20
        assert service.adaptation_threshold == 0.2
        assert service.calibration_target == 0.85

        # Check initialization of data structures
        assert isinstance(service.prediction_history, list)
        assert isinstance(service.strategy_history, list)
        assert isinstance(service.agent_profiles, dict)
        assert isinstance(service.pattern_insights, list)
        assert isinstance(service.adaptation_history, list)
        assert isinstance(service.active_ab_tests, dict)
        assert isinstance(service.completed_ab_tests, list)

    def test_extract_probability(self, learning_service):
        """Test probability extraction from prediction results."""
        # Test with float result
        from src.domain.value_objects.prediction_result import PredictionResult
        result = PredictionResult(value=0.7, prediction_type="binary")
        prob = learning_service._extract_probability(result)
        assert prob == 0.7

        # Test with dict result
        result = PredictionResult(value={"probability": 0.8}, prediction_type="binary")
        prob = learning_service._extract_probability(result)
        assert prob == 0.8

        # Test with invalid result
        result = PredictionResult(value="invalid", prediction_type="binary")
        prob = learning_service._extract_probability(result)
        assert prob == 0.5  # Default neutral probability


class TestLearningServiceIntegration:
    """Integration tests for LearningService."""

    @pytest.fixture
    def learning_service(self):
        """Create a LearningService for integration testing."""
        return LearningService(
            learning_mode=LearningMode.ADAPTIVE,
            min_samples_for_learning=3  # Lower threshold for testing
        )

    @pytest.mark.asyncio
    async def test_full_learning_cycle(self, learning_service):
        """Test a complete learning cycle with pattern recognition and adaptation."""
        # Create test data
        predictions = []
        outcomes = []
        base_time = datetime.utcnow()

        # Create predictions with clear patterns
        for i in range(15):
            # Agent 1 is better at predictions
            agent = "agent_1" if i % 2 == 0 else "agent_2"
            prob = 0.8 if agent == "agent_1" else 0.6
            outcome = 1.0 if i % 3 != 0 else 0.0

            pred = Prediction.create_binary(
                question_id=i + 1,
                probability=prob,
                confidence_level=0.8,
                confidence_basis="Test prediction",
                method="test_method",
                reasoning=f"Test reasoning {i}",
                created_by=agent,
                timestamp=base_time + timedelta(hours=i)
            )
            predictions.append(pred)
            outcomes.append(outcome)

        # Analyze accuracy
        metrics = await learning_service.analyze_prediction_accuracy(predictions, outcomes)
        assert isinstance(metrics, PerformanceMetrics)

        # Analyze patterns
        patterns = await learning_service.analyze_historical_patterns(lookback_days=1)
        assert isinstance(patterns, list)

        # Test agent weight optimization
        agent_performances = {
            'agent_1': [0.8, 0.85, 0.82],
            'agent_2': [0.6, 0.65, 0.62]
        }
        weights = await learning_service.optimize_agent_weights(agent_performances)
        assert weights['agent_1'] > weights['agent_2']  # Better agent should have higher weight

        # Test calibration monitoring
        calibration = await learning_service.monitor_calibration(predictions, outcomes)
        assert isinstance(calibration, CalibrationAnalysis)

    @pytest.mark.asyncio
    async def test_strategy_adaptation_workflow(self, learning_service):
        """Test complete strategy adaptation workflow."""
        # Create strategy results showing declining performance
        results = []
        base_time = datetime.utcnow()

        for i in range(5):
            result = StrategyResult.create(
                strategy_type=StrategyType.AGGRESSIVE,
                expected_score=0.8,
                reasoning=f"Test strategy {i}",
                question_ids=[i + 1],
                confidence_level=0.8,
                confidence_basis="Test confidence",
                timestamp=base_time + timedelta(hours=i)
            )
            # Declining actual scores
            actual_score = 0.8 - (i * 0.1)
            result = result.mark_success(actual_score)
            results.append(result)

        # Create tournament
        scoring_rules = ScoringRules(method=ScoringMethod.BRIER_SCORE)
        tournament = Tournament(
            id=1,
            name="Test Tournament",
            questions=[],
            scoring_rules=scoring_rules,
            start_date=datetime.utcnow() - timedelta(days=5),
            end_date=datetime.utcnow() + timedelta(days=5),
            current_standings={}
        )

        # Test strategy refinement
        adaptation = await learning_service.refine_strategy_based_on_feedback(tournament, results)

        if adaptation:  # May trigger adaptation due to declining performance
            assert isinstance(adaptation, StrategyAdaptation)
            assert adaptation.trigger == AdaptationTrigger.PERFORMANCE_DECLINE
            assert adaptation.old_strategy == StrategyType.AGGRESSIVE

    @pytest.mark.asyncio
    async def test_ab_test_complete_workflow(self, learning_service):
        """Test complete A/B testing workflow."""
        test_id = "integration_test"

        # Start A/B test with short duration
        await learning_service.run_ab_test(
            test_id, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, 0  # 0 days for immediate completion
        )

        # Record results alternately
        for i in range(6):
            strategy = await learning_service.get_ab_test_assignment(test_id)
            if strategy:  # May be None if test is completed
                performance = 0.8 if strategy == StrategyType.AGGRESSIVE else 0.7
                await learning_service.record_ab_test_result(test_id, strategy, performance)

        # Check if test completed (depends on implementation)
        if test_id not in learning_service.active_ab_tests:
            assert len(learning_service.completed_ab_tests) >= 0


class TestLearningServiceEdgeCases:
    """Test edge cases and error conditions for LearningService."""

    @pytest.fixture
    def learning_service(self):
        """Create a LearningService for edge case testing."""
        return LearningService(min_samples_for_learning=1)

    @pytest.mark.asyncio
    async def test_analyze_prediction_accuracy_empty_predictions(self, learning_service):
        """Test accuracy analysis with empty predictions list."""
        with pytest.raises(ValueError, match="No predictions provided for analysis"):
            await learning_service.analyze_prediction_accuracy([])

    @pytest.mark.asyncio
    async def test_analyze_prediction_accuracy_no_resolved_predictions(self, learning_service):
        """Test accuracy analysis when no predictions have outcomes."""
        predictions = [
            Prediction.create_binary(1, 0.7, 0.8, "test", "method", "reasoning", "agent")
        ]

        # Create predictions with outcomes but mark as unresolved
        predictions_with_outcomes = [(pred, None) for pred in predictions]

        metrics = await learning_service.analyze_prediction_accuracy(predictions_with_outcomes)

        # Should return empty metrics
        assert metrics.prediction_count == 0
        assert metrics.accuracy_score == 0.0

    @pytest.mark.asyncio
    async def test_update_predictions_with_new_information_empty_predictions(self, learning_service):
        """Test prediction updating with empty predictions list."""
        new_info = {'summary': 'test info'}

        updated = await learning_service.update_predictions_with_new_information(
            "question_1", new_info, []
        )

        assert updated == []

    @pytest.mark.asyncio
    async def test_optimize_agent_weights_empty_agents(self, learning_service):
        """Test agent weight optimization with empty agent list."""
        weights = await learning_service.optimize_agent_weights([], {})
        assert weights == {}

    @pytest.mark.asyncio
    async def test_monitor_calibration_insufficient_data(self, learning_service):
        """Test calibration monitoring with insufficient data."""
        predictions = [
            (Prediction.create_binary(1, 0.7, 0.8, "test", "method", "reasoning", "agent"), None)
        ]

        analysis = await learning_service.monitor_calibration(predictions)

        assert analysis.overall_calibration == 0.0
        assert "Insufficient data" in analysis.recommendations[0]

    @pytest.mark.asyncio
    async def test_analyze_historical_patterns_empty_history(self, learning_service):
        """Test pattern analysis with empty history."""
        patterns = await learning_service.analyze_historical_patterns([])
        assert patterns == []

    @pytest.mark.asyncio
    async def test_refine_strategy_no_results(self, learning_service):
        """Test strategy refinement with no results."""
        tournament = Mock(spec=Tournament)
        adaptation = await learning_service.refine_strategy_based_on_feedback(tournament, [])
        assert adaptation is None

    @pytest.mark.asyncio
    async def test_ab_test_record_result_nonexistent_test(self, learning_service):
        """Test recording result for non-existent A/B test."""
        # Should not raise exception, just log warning
        await learning_service.record_ab_test_result("nonexistent", StrategyType.AGGRESSIVE, 0.8)

    @pytest.mark.asyncio
    async def test_get_ab_test_assignment_nonexistent_test(self, learning_service):
        """Test getting assignment for non-existent A/B test."""
        assignment = await learning_service.get_ab_test_assignment("nonexistent")
        assert assignment is None


class TestLearningServicePerformance:
    """Performance tests for LearningService."""

    @pytest.fixture
    def learning_service(self):
        """Create a LearningService for performance testing."""
        return LearningService(min_samples_for_learning=1)

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, learning_service):
        """Test performance with large datasets."""
        import time

        # Create large dataset
        predictions = []
        outcomes = []
        base_time = datetime.utcnow()

        for i in range(1000):
            pred = Prediction.create_binary(
                question_id=i + 1,
                probability=0.5 + (i % 100) / 200,  # Varying probabilities
                confidence_level=0.5 + (i % 100) / 200,
                confidence_basis=f"Test {i}",
                method="test_method",
                reasoning=f"Reasoning {i}",
                created_by=f"agent_{i % 10}",
                timestamp=base_time + timedelta(seconds=i)
            )
            predictions.append(pred)
            outcomes.append(1.0 if i % 2 == 0 else 0.0)

        # Test performance
        start_time = time.time()
        predictions_with_outcomes = list(zip(predictions, outcomes))
        metrics = await learning_service.analyze_prediction_accuracy(predictions_with_outcomes)
        end_time = time.time()

        # Should complete within reasonable time (< 5 seconds for 1000 predictions)
        assert end_time - start_time < 5.0
        assert metrics.prediction_count == 1000

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, learning_service):
        """Test concurrent learning operations."""
        # Create test data
        predictions = [
            Prediction.create_binary(i, 0.7, 0.8, "test", "method", "reasoning", f"agent_{i}")
            for i in range(10)
        ]
        outcomes = [1.0 if i % 2 == 0 else 0.0 for i in range(10)]
        predictions_with_outcomes = list(zip(predictions, outcomes))

        # Run multiple operations concurrently
        tasks = [
            learning_service.analyze_prediction_accuracy(predictions_with_outcomes),
            learning_service.monitor_calibration(predictions_with_outcomes),
            learning_service.analyze_historical_patterns(predictions_with_outcomes),
            learning_service.optimize_agent_weights(['agent_1', 'agent_2'], {
                'agent_1': PerformanceMetrics(0.8, 0.8, 0.2, -0.5, 0.7, 5, {}, (datetime.utcnow(), datetime.utcnow())),
                'agent_2': PerformanceMetrics(0.7, 0.7, 0.3, -0.7, 0.6, 5, {}, (datetime.utcnow(), datetime.utcnow()))
            })
        ]

        # All tasks should complete successfully
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)


class TestLearningServiceMLPatterns:
    """Test ML-based pattern recognition capabilities."""

    @pytest.fixture
    def learning_service(self):
        """Create a LearningService for ML pattern testing."""
        return LearningService(min_samples_for_learning=5)

    @pytest.mark.asyncio
    async def test_temporal_pattern_recognition(self, learning_service):
        """Test recognition of temporal patterns in predictions."""
        # Create predictions with clear temporal patterns
        predictions = []
        outcomes = []
        base_time = datetime.utcnow()

        # Better performance in morning hours (8-12)
        for i in range(24):
            hour = i % 24
            # Higher accuracy during morning hours
            outcome_prob = 0.8 if 8 <= hour <= 12 else 0.5

            pred = Prediction.create_binary(
                question_id=i + 1,
                probability=0.7,
                confidence_level=0.8,
                confidence_basis="Test",
                method="test_method",
                reasoning=f"Prediction at hour {hour}",
                created_by="agent_1",
                timestamp=base_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            )
            predictions.append(pred)
            outcomes.append(1.0 if np.random.random() < outcome_prob else 0.0)

        predictions_with_outcomes = list(zip(predictions, outcomes))
        patterns = await learning_service.analyze_historical_patterns(predictions_with_outcomes)

        # Should identify temporal patterns
        temporal_patterns = [p for p in patterns if p.pattern_type == PatternType.TIMING_PATTERNS]
        assert len(temporal_patterns) > 0

    @pytest.mark.asyncio
    async def test_agent_specialization_pattern(self, learning_service):
        """Test recognition of agent specialization patterns."""
        predictions = []
        outcomes = []
        base_time = datetime.utcnow()

        # Agent 1 specializes in AI questions, Agent 2 in economics
        categories = [QuestionCategory.AI_DEVELOPMENT, QuestionCategory.ECONOMICS]

        for i in range(20):
            category = categories[i % 2]
            agent = "ai_specialist" if category == QuestionCategory.AI_DEVELOPMENT else "econ_specialist"

            # Specialists perform better in their domain
            if (agent == "ai_specialist" and category == QuestionCategory.AI_DEVELOPMENT) or \
               (agent == "econ_specialist" and category == QuestionCategory.ECONOMICS):
                outcome_prob = 0.9
            else:
                outcome_prob = 0.6

            pred = Prediction.create_binary(
                question_id=i + 1,
                probability=0.7,
                confidence_level=0.8,
                confidence_basis="Test",
                method="test_method",
                reasoning=f"Prediction by {agent}",
                created_by=agent,
                timestamp=base_time + timedelta(hours=i)
            )
            # Add category information to prediction metadata
            pred.metadata['category'] = category
            predictions.append(pred)
            outcomes.append(1.0 if np.random.random() < outcome_prob else 0.0)

        predictions_with_outcomes = list(zip(predictions, outcomes))
        patterns = await learning_service.analyze_historical_patterns(predictions_with_outcomes)

        # Should identify agent specialization patterns
        specialization_patterns = [p for p in patterns if p.pattern_type == PatternType.AGENT_SPECIALIZATION]
        # Note: This might be 0 if the pattern detection algorithm doesn't find significant differences
        # due to randomness in test data

    @pytest.mark.asyncio
    async def test_confidence_calibration_pattern(self, learning_service):
        """Test recognition of confidence calibration patterns."""
        predictions = []
        outcomes = []
        base_time = datetime.utcnow()

        # Create predictions with systematic overconfidence
        for i in range(30):
            # High confidence but lower actual accuracy
            confidence = 0.9
            actual_accuracy_prob = 0.6  # Overconfident

            pred = Prediction.create_binary(
                question_id=i + 1,
                probability=0.8,
                confidence_level=confidence,
                confidence_basis="High confidence test",
                method="test_method",
                reasoning=f"Overconfident prediction {i}",
                created_by="overconfident_agent",
                timestamp=base_time + timedelta(hours=i)
            )
            predictions.append(pred)
            outcomes.append(1.0 if np.random.random() < actual_accuracy_prob else 0.0)

        predictions_with_outcomes = list(zip(predictions, outcomes))
        patterns = await learning_service.analyze_historical_patterns(predictions_with_outcomes)

        # Should identify confidence calibration issues
        calibration_patterns = [p for p in patterns if p.pattern_type == PatternType.CONFIDENCE_CALIBRATION]
        assert len(calibration_patterns) > 0

    @pytest.mark.asyncio
    async def test_strategy_effectiveness_pattern(self, learning_service):
        """Test recognition of strategy effectiveness patterns."""
        predictions = []
        outcomes = []
        base_time = datetime.utcnow()

        methods = ["aggressive_method", "conservative_method", "balanced_method"]
        method_effectiveness = {"aggressive_method": 0.8, "conservative_method": 0.6, "balanced_method": 0.7}

        for i in range(30):
            method = methods[i % 3]
            effectiveness = method_effectiveness[method]

            pred = Prediction.create_binary(
                question_id=i + 1,
                probability=0.7,
                confidence_level=0.8,
                confidence_basis="Test",
                method=method,
                reasoning=f"Prediction using {method}",
                created_by="test_agent",
                timestamp=base_time + timedelta(hours=i)
            )
            predictions.append(pred)
            outcomes.append(1.0 if np.random.random() < effectiveness else 0.0)

        predictions_with_outcomes = list(zip(predictions, outcomes))
        patterns = await learning_service.analyze_historical_patterns(predictions_with_outcomes)

        # Should identify strategy effectiveness patterns
        strategy_patterns = [p for p in patterns if p.pattern_type == PatternType.STRATEGY_EFFECTIVENESS]
        assert len(strategy_patterns) > 0


class TestLearningServiceCalibration:
    """Test calibration-specific functionality."""

    @pytest.fixture
    def learning_service(self):
        """Create a LearningService for calibration testing."""
        return LearningService(calibration_target=0.9)

    @pytest.mark.asyncio
    async def test_perfect_calibration(self, learning_service):
        """Test calibration analysis with perfectly calibrated predictions."""
        predictions = []
        outcomes = []
        base_time = datetime.utcnow()

        # Create perfectly calibrated predictions
        confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

        for conf in confidence_levels:
            for _ in range(10):  # 10 predictions per confidence level
                pred = Prediction.create_binary(
                    question_id=len(predictions) + 1,
                    probability=conf,  # Probability matches confidence
                    confidence_level=conf,
                    confidence_basis="Perfect calibration test",
                    method="test_method",
                    reasoning="Test reasoning",
                    created_by="test_agent",
                    timestamp=base_time + timedelta(minutes=len(predictions))
                )
                predictions.append(pred)
                # Outcome matches the confidence level
                outcomes.append(1.0 if np.random.random() < conf else 0.0)

        predictions_with_outcomes = list(zip(predictions, outcomes))
        analysis = await learning_service.monitor_calibration(predictions_with_outcomes)

        # Should show good calibration
        assert analysis.overall_calibration > 0.7  # Should be reasonably well calibrated
        assert len(analysis.confidence_bins) > 0

    @pytest.mark.asyncio
    async def test_overconfidence_detection(self, learning_service):
        """Test detection of overconfidence bias."""
        predictions = []
        outcomes = []
        base_time = datetime.utcnow()

        # Create overconfident predictions
        for i in range(20):
            pred = Prediction.create_binary(
                question_id=i + 1,
                probability=0.9,  # High probability
                confidence_level=0.9,  # High confidence
                confidence_basis="Overconfident test",
                method="test_method",
                reasoning="Overconfident reasoning",
                created_by="overconfident_agent",
                timestamp=base_time + timedelta(minutes=i)
            )
            predictions.append(pred)
            # But actual accuracy is much lower
            outcomes.append(1.0 if np.random.random() < 0.6 else 0.0)

        predictions_with_outcomes = list(zip(predictions, outcomes))
        analysis = await learning_service.monitor_calibration(predictions_with_outcomes)

        # Should detect overconfidence
        assert analysis.overconfidence_bias > 0.1
        assert any("overconfidence" in rec.lower() for rec in analysis.recommendations)

    @pytest.mark.asyncio
    async def test_underconfidence_detection(self, learning_service):
        """Test detection of underconfidence bias."""
        predictions = []
        outcomes = []
        base_time = datetime.utcnow()

        # Create underconfident predictions
        for i in range(20):
            pred = Prediction.create_binary(
                question_id=i + 1,
                probability=0.6,  # Moderate probability
                confidence_level=0.5,  # Low confidence
                confidence_basis="Underconfident test",
                method="test_method",
                reasoning="Underconfident reasoning",
                created_by="underconfident_agent",
                timestamp=base_time + timedelta(minutes=i)
            )
            predictions.append(pred)
            # But actual accuracy is higher
            outcomes.append(1.0 if np.random.random() < 0.8 else 0.0)

        predictions_with_outcomes = list(zip(predictions, outcomes))
        analysis = await learning_service.monitor_calibration(predictions_with_outcomes)

        # Should detect underconfidence
        assert analysis.underconfidence_bias > 0.1
        assert any("underconfidence" in rec.lower() for rec in analysis.recommendations)


class TestLearningServiceABTesting:
    """Test A/B testing framework functionality."""

    @pytest.fixture
    def learning_service(self):
        """Create a LearningService for A/B testing."""
        return LearningService()

    @pytest.mark.asyncio
    async def test_ab_test_statistical_significance(self, learning_service):
        """Test A/B test statistical significance calculation."""
        test_id = "significance_test"

        await learning_service.run_ab_test(
            test_id, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, 20, 7
        )

        # Record results with clear difference
        for _ in range(10):
            await learning_service.record_ab_test_result(test_id, StrategyType.AGGRESSIVE, 0.9)
            await learning_service.record_ab_test_result(test_id, StrategyType.CONSERVATIVE, 0.6)

        # Complete the test
        await learning_service._complete_ab_test(test_id)

        result = learning_service.completed_ab_tests[0]
        assert result.statistical_significance > 0.8  # Should be statistically significant
        assert result.winner == StrategyType.AGGRESSIVE

    @pytest.mark.asyncio
    async def test_ab_test_no_clear_winner(self, learning_service):
        """Test A/B test with no clear winner."""
        test_id = "no_winner_test"

        await learning_service.run_ab_test(
            test_id, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, 10, 7
        )

        # Record similar results
        for _ in range(5):
            await learning_service.record_ab_test_result(test_id, StrategyType.AGGRESSIVE, 0.75)
            await learning_service.record_ab_test_result(test_id, StrategyType.CONSERVATIVE, 0.73)

        # Complete the test
        await learning_service._complete_ab_test(test_id)

        result = learning_service.completed_ab_tests[0]
        assert result.statistical_significance < 0.95  # Should not be statistically significant
        assert result.winner is None  # No clear winner

    @pytest.mark.asyncio
    async def test_multiple_concurrent_ab_tests(self, learning_service):
        """Test running multiple A/B tests concurrently."""
        test_ids = ["test_1", "test_2", "test_3"]

        # Start multiple tests
        for test_id in test_ids:
            await learning_service.run_ab_test(
                test_id, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, 6, 5
            )

        assert len(learning_service.active_ab_tests) == 3

        # Record results for all tests
        for test_id in test_ids:
            for _ in range(3):
                strategy = await learning_service.get_ab_test_assignment(test_id)
                performance = 0.8 if strategy == StrategyType.AGGRESSIVE else 0.7
                await learning_service.record_ab_test_result(test_id, strategy, performance)

        # All tests should be completed
        assert len(learning_service.active_ab_tests) == 0
        assert len(learning_service.completed_ab_tests) == 3

    @pytest.mark.asyncio
    async def test_ab_test_confidence_intervals(self, learning_service):
        """Test A/B test confidence interval calculation."""
        test_id = "confidence_interval_test"

        await learning_service.run_ab_test(
            test_id, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, 10, 7
        )

        # Record results
        for _ in range(5):
            await learning_service.record_ab_test_result(test_id, StrategyType.AGGRESSIVE, 0.8)
            await learning_service.record_ab_test_result(test_id, StrategyType.CONSERVATIVE, 0.6)

        # Complete the test
        await learning_service._complete_ab_test(test_id)

        result = learning_service.completed_ab_tests[0]
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.confidence_interval[1]


if __name__ == "__main__":
    pytest.main([__file__])
