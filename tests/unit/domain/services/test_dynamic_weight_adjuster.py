"""Tests for DynamicWeightAdjuster service."""

from datetime import datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
)
from src.domain.entities.question import Question
from src.domain.services.dynamic_weight_adjuster import (
    DynamicWeightAdjuster,
    EnsembleComposition,
    PerformanceRecord,
    WeightAdjustmentStrategy,
)


class TestDynamicWeightAdjuster:
    """Test cases for DynamicWeightAdjuster."""

    @pytest.fixture
    def adjuster(self):
        """Create a DynamicWeightAdjuster instance for testing."""
        return DynamicWeightAdjuster(
            lookback_window=20,
            min_predictions_for_weight=3,
            performance_decay_factor=0.95,
        )

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        from src.domain.entities.question import QuestionType, QuestionStatus

        return Question(
            id=uuid4(),
            metaculus_id=12345,
            title="Will AI achieve AGI by 2030?",
            description="Test question",
            question_type=QuestionType.BINARY,
            status=QuestionStatus.OPEN,
            url="https://example.com/question",
            close_time=datetime.now() + timedelta(days=30),
            resolve_time=datetime.now() + timedelta(days=60),
            categories=["technology"],
            metadata={"test": True},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolution_criteria="Clear criteria",
        )

    @pytest.fixture
    def sample_prediction(self, sample_question):
        """Create a sample prediction for testing."""
        from src.domain.entities.prediction import PredictionResult
        from datetime import datetime

        return Prediction(
            id=uuid4(),
            question_id=sample_question.id,
            research_report_id=uuid4(),
            result=PredictionResult(binary_probability=0.7),
            confidence=PredictionConfidence.MEDIUM,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Test reasoning",
            reasoning_steps=["Step 1", "Step 2"],
            created_at=datetime.now(),
            created_by="test_agent",
        )

    def test_record_performance_basic(self, adjuster, sample_prediction):
        """Test basic performance recording."""
        # Record performance
        adjuster.record_performance(sample_prediction, actual_outcome=True)

        # Check that performance was recorded
        assert len(adjuster.performance_records) == 1
        record = adjuster.performance_records[0]

        assert record.agent_name == "test_agent"
        assert record.predicted_probability == 0.7
        assert record.actual_outcome is True
        assert record.brier_score == (0.7 - 1.0) ** 2  # 0.09

    def test_agent_profile_creation(self, adjuster, sample_prediction):
        """Test that agent profiles are created and updated."""
        # Record multiple performances
        for i, outcome in enumerate([True, False, True, True, False]):
            prediction = Mock()
            prediction.id = uuid4()
            prediction.question_id = sample_prediction.question_id
            prediction.created_by = "test_agent"
            prediction.method = PredictionMethod.CHAIN_OF_THOUGHT
            prediction.result.binary_probability = 0.6 + (i * 0.1)
            prediction.get_confidence_score.return_value = 0.8

            adjuster.record_performance(prediction, actual_outcome=outcome)

        # Check agent profile was created
        assert "test_agent" in adjuster.agent_profiles
        profile = adjuster.agent_profiles["test_agent"]

        assert profile.agent_name == "test_agent"
        assert profile.total_predictions == 5
        assert profile.recent_predictions == 5
        assert profile.overall_brier_score > 0
        assert profile.recent_brier_score > 0

    def test_performance_degradation_detection(self, adjuster):
        """Test performance degradation detection."""
        agent_name = "degrading_agent"

        # Create performance records showing degradation
        base_time = datetime.now() - timedelta(days=20)

        # Good early performance
        for i in range(10):
            record = PerformanceRecord(
                agent_name=agent_name,
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=base_time + timedelta(days=i),
                predicted_probability=0.8,
                actual_outcome=True,
                brier_score=0.04,  # Good score
                accuracy=1.0,
                confidence_score=0.8,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Poor recent performance
        for i in range(10, 20):
            record = PerformanceRecord(
                agent_name=agent_name,
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=base_time + timedelta(days=i),
                predicted_probability=0.3,
                actual_outcome=True,
                brier_score=0.49,  # Poor score
                accuracy=0.0,
                confidence_score=0.3,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Update agent profile
        adjuster._update_agent_profile(agent_name)

        # Test degradation detection
        is_degrading, explanation = adjuster.detect_performance_degradation(agent_name)
        assert is_degrading
        assert "performance" in explanation.lower()

    def test_dynamic_weight_calculation(self, adjuster):
        """Test dynamic weight calculation for multiple agents."""
        agents = ["agent1", "agent2", "agent3"]

        # Create different performance profiles
        for i, agent in enumerate(agents):
            # Create performance records with different quality
            base_score = 0.15 + (i * 0.1)  # agent1: 0.15, agent2: 0.25, agent3: 0.35

            for j in range(10):
                record = PerformanceRecord(
                    agent_name=agent,
                    prediction_id=uuid4(),
                    question_id=uuid4(),
                    timestamp=datetime.now() - timedelta(days=j),
                    predicted_probability=0.7,
                    actual_outcome=True,
                    brier_score=base_score,
                    accuracy=1.0 if base_score < 0.25 else 0.0,
                    confidence_score=0.8,
                    method=PredictionMethod.CHAIN_OF_THOUGHT,
                )
                adjuster.performance_records.append(record)

            adjuster._update_agent_profile(agent)

        # Get dynamic weights
        weights = adjuster.get_dynamic_weights(agents)

        # Check that weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.001

        # Check that better performing agent (agent1) has higher weight
        assert weights["agent1"] > weights["agent2"]
        assert weights["agent2"] > weights["agent3"]

    def test_ensemble_composition_recommendation(self, adjuster):
        """Test ensemble composition recommendation."""
        agents = ["agent1", "agent2", "agent3", "agent4"]

        # Create performance profiles
        for i, agent in enumerate(agents):
            base_score = 0.15 + (i * 0.05)

            for j in range(15):
                record = PerformanceRecord(
                    agent_name=agent,
                    prediction_id=uuid4(),
                    question_id=uuid4(),
                    timestamp=datetime.now() - timedelta(days=j),
                    predicted_probability=0.6 + (i * 0.1),
                    actual_outcome=True,
                    brier_score=base_score,
                    accuracy=1.0 if base_score < 0.25 else 0.0,
                    confidence_score=0.7 + (i * 0.05),
                    method=PredictionMethod.CHAIN_OF_THOUGHT,
                )
                adjuster.performance_records.append(record)

            adjuster._update_agent_profile(agent)

        # Get composition recommendation
        composition = adjuster.recommend_ensemble_composition(agents, target_size=3)

        assert isinstance(composition, EnsembleComposition)
        assert composition.active_agents <= 3
        assert composition.total_agents == 4
        assert len(composition.agent_weights) <= 3
        assert abs(sum(composition.agent_weights.values()) - 1.0) < 0.001
        assert composition.diversity_score >= 0.0
        assert composition.expected_performance >= 0.0

    def test_automatic_rebalancing_trigger(self, adjuster):
        """Test that automatic rebalancing is triggered when performance degrades."""
        agents = ["stable_agent", "degrading_agent"]

        # Create stable agent performance
        for i in range(20):
            record = PerformanceRecord(
                agent_name="stable_agent",
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=datetime.now() - timedelta(days=i),
                predicted_probability=0.7,
                actual_outcome=True,
                brier_score=0.09,
                accuracy=1.0,
                confidence_score=0.8,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Create degrading agent performance (good then bad)
        for i in range(10):
            record = PerformanceRecord(
                agent_name="degrading_agent",
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=datetime.now() - timedelta(days=i + 10),
                predicted_probability=0.8,
                actual_outcome=True,
                brier_score=0.04,
                accuracy=1.0,
                confidence_score=0.9,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        for i in range(10):
            record = PerformanceRecord(
                agent_name="degrading_agent",
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=datetime.now() - timedelta(days=i),
                predicted_probability=0.2,
                actual_outcome=True,
                brier_score=0.64,
                accuracy=0.0,
                confidence_score=0.3,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Update profiles
        for agent in agents:
            adjuster._update_agent_profile(agent)

        # Test that rebalancing is needed
        initial_weights = adjuster.get_dynamic_weights(agents)

        # Simulate rebalancing trigger
        needs_rebalancing = any(
            adjuster.detect_performance_degradation(agent)[0] for agent in agents
        )

        assert needs_rebalancing

        # Get new composition after detecting degradation
        composition = adjuster.recommend_ensemble_composition(agents)

        # The degrading agent should have lower weight than the stable agent
        if (
            "degrading_agent" in composition.agent_weights
            and "stable_agent" in composition.agent_weights
        ):
            assert (
                composition.agent_weights["degrading_agent"]
                < composition.agent_weights["stable_agent"]
            )

    def test_real_time_agent_selection(self, adjuster):
        """Test real-time agent selection based on current performance."""
        agents = ["fast_agent", "slow_agent", "accurate_agent", "inaccurate_agent"]

        # Create different performance characteristics
        performance_data = {
            "fast_agent": (0.20, 0.8, 0.7),  # moderate accuracy, high confidence
            "slow_agent": (0.18, 0.6, 0.9),  # good accuracy, lower confidence
            "accurate_agent": (0.12, 0.9, 0.8),  # high accuracy, high confidence
            "inaccurate_agent": (0.35, 0.4, 0.5),  # poor accuracy, low confidence
        }

        for agent, (brier, accuracy, confidence) in performance_data.items():
            for i in range(12):
                record = PerformanceRecord(
                    agent_name=agent,
                    prediction_id=uuid4(),
                    question_id=uuid4(),
                    timestamp=datetime.now() - timedelta(hours=i),
                    predicted_probability=0.7,
                    actual_outcome=True,
                    brier_score=brier,
                    accuracy=accuracy,
                    confidence_score=confidence,
                    method=PredictionMethod.CHAIN_OF_THOUGHT,
                )
                adjuster.performance_records.append(record)

            adjuster._update_agent_profile(agent)

        # Test real-time selection (should prefer accurate_agent)
        composition = adjuster.recommend_ensemble_composition(agents, target_size=2)

        # Accurate agent should be selected
        assert "accurate_agent" in composition.agent_weights

        # Inaccurate agent should have lowest weight or be excluded
        if "inaccurate_agent" in composition.agent_weights:
            accurate_weight = composition.agent_weights["accurate_agent"]
            inaccurate_weight = composition.agent_weights["inaccurate_agent"]
            assert accurate_weight > inaccurate_weight

    def test_performance_summary(self, adjuster):
        """Test comprehensive performance summary generation."""
        # Add some performance data
        agents = ["agent1", "agent2"]

        for agent in agents:
            for i in range(5):
                record = PerformanceRecord(
                    agent_name=agent,
                    prediction_id=uuid4(),
                    question_id=uuid4(),
                    timestamp=datetime.now() - timedelta(days=i),
                    predicted_probability=0.6,
                    actual_outcome=True,
                    brier_score=0.16,
                    accuracy=1.0,
                    confidence_score=0.7,
                    method=PredictionMethod.CHAIN_OF_THOUGHT,
                )
                adjuster.performance_records.append(record)

            adjuster._update_agent_profile(agent)

        # Get performance summary
        summary = adjuster.get_performance_summary()

        assert "total_agents" in summary
        assert "total_predictions" in summary
        assert "agent_profiles" in summary
        assert "overall_metrics" in summary
        assert "degradation_alerts" in summary

        assert summary["total_agents"] == 2
        assert summary["total_predictions"] == 10
        assert len(summary["agent_profiles"]) == 2

    def test_weight_adjustment_strategies(self, adjuster):
        """Test different weight adjustment strategies."""
        agents = ["agent1", "agent2", "agent3"]

        # Create performance data
        for i, agent in enumerate(agents):
            base_score = 0.15 + (i * 0.1)

            for j in range(10):
                record = PerformanceRecord(
                    agent_name=agent,
                    prediction_id=uuid4(),
                    question_id=uuid4(),
                    timestamp=datetime.now() - timedelta(days=j),
                    predicted_probability=0.7,
                    actual_outcome=True,
                    brier_score=base_score,
                    accuracy=1.0 if base_score < 0.25 else 0.0,
                    confidence_score=0.8,
                    method=PredictionMethod.CHAIN_OF_THOUGHT,
                )
                adjuster.performance_records.append(record)

            adjuster._update_agent_profile(agent)

        # Test different strategies
        strategies = [
            WeightAdjustmentStrategy.EXPONENTIAL_DECAY,
            WeightAdjustmentStrategy.LINEAR_DECAY,
            WeightAdjustmentStrategy.THRESHOLD_BASED,
            WeightAdjustmentStrategy.RELATIVE_RANKING,
            WeightAdjustmentStrategy.ADAPTIVE_LEARNING_RATE,
        ]

        for strategy in strategies:
            weights = adjuster.get_dynamic_weights(agents, strategy)

            # All strategies should return normalized weights
            assert abs(sum(weights.values()) - 1.0) < 0.001

            # Better performing agent should generally have higher weight
            assert weights["agent1"] >= weights["agent3"]

    def test_should_trigger_rebalancing(self, adjuster):
        """Test rebalancing trigger detection."""
        agents = ["good_agent", "bad_agent"]

        # Create good agent performance
        for i in range(10):
            record = PerformanceRecord(
                agent_name="good_agent",
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=datetime.now() - timedelta(days=i),
                predicted_probability=0.8,
                actual_outcome=True,
                brier_score=0.04,
                accuracy=1.0,
                confidence_score=0.9,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Create bad agent performance
        for i in range(10):
            record = PerformanceRecord(
                agent_name="bad_agent",
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=datetime.now() - timedelta(days=i),
                predicted_probability=0.2,
                actual_outcome=True,
                brier_score=0.64,
                accuracy=0.0,
                confidence_score=0.3,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Update profiles
        for agent in agents:
            adjuster._update_agent_profile(agent)

        # Test rebalancing trigger
        should_rebalance, reason = adjuster.should_trigger_rebalancing(agents)
        assert should_rebalance
        assert "degradation" in reason.lower()

    def test_select_optimal_agents_realtime(self, adjuster):
        """Test real-time optimal agent selection."""
        agents = ["excellent_agent", "good_agent", "poor_agent", "new_agent"]

        # Create performance data with different quality levels
        performance_data = {
            "excellent_agent": (
                0.10,
                1.0,
                0.9,
            ),  # Low Brier, high accuracy, high confidence
            "good_agent": (0.20, 0.8, 0.7),  # Moderate performance
            "poor_agent": (0.45, 0.2, 0.4),  # Poor performance
        }

        for agent, (brier, accuracy, confidence) in performance_data.items():
            for i in range(8):
                record = PerformanceRecord(
                    agent_name=agent,
                    prediction_id=uuid4(),
                    question_id=uuid4(),
                    timestamp=datetime.now() - timedelta(days=i),
                    predicted_probability=0.7,
                    actual_outcome=True,
                    brier_score=brier,
                    accuracy=accuracy,
                    confidence_score=confidence,
                    method=PredictionMethod.CHAIN_OF_THOUGHT,
                )
                adjuster.performance_records.append(record)

            adjuster._update_agent_profile(agent)

        # Test optimal selection
        selected = adjuster.select_optimal_agents_realtime(agents, max_agents=2)

        # Should select the best performing agents
        assert "excellent_agent" in selected
        assert "poor_agent" not in selected or len(selected) == len(
            agents
        )  # Only if all agents needed

    def test_trigger_automatic_rebalancing(self, adjuster):
        """Test automatic rebalancing trigger."""
        current_agents = ["degrading_agent"]
        available_agents = ["degrading_agent", "good_agent"]

        # Create degrading agent
        for i in range(10):
            record = PerformanceRecord(
                agent_name="degrading_agent",
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=datetime.now() - timedelta(days=i),
                predicted_probability=0.2,
                actual_outcome=True,
                brier_score=0.64,
                accuracy=0.0,
                confidence_score=0.3,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Create good agent
        for i in range(10):
            record = PerformanceRecord(
                agent_name="good_agent",
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=datetime.now() - timedelta(days=i),
                predicted_probability=0.8,
                actual_outcome=True,
                brier_score=0.04,
                accuracy=1.0,
                confidence_score=0.9,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Update profiles
        for agent in available_agents:
            adjuster._update_agent_profile(agent)

        # Test automatic rebalancing
        new_composition = adjuster.trigger_automatic_rebalancing(
            current_agents, available_agents
        )

        assert new_composition is not None
        assert isinstance(new_composition, EnsembleComposition)
        assert "good_agent" in new_composition.agent_weights

    def test_get_rebalancing_recommendations(self, adjuster):
        """Test rebalancing recommendations."""
        current_agents = ["current_agent"]

        # Create current agent with poor performance
        for i in range(8):
            record = PerformanceRecord(
                agent_name="current_agent",
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=datetime.now() - timedelta(days=i),
                predicted_probability=0.3,
                actual_outcome=True,
                brier_score=0.49,
                accuracy=0.0,
                confidence_score=0.4,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Create potential addition with good performance
        for i in range(8):
            record = PerformanceRecord(
                agent_name="potential_agent",
                prediction_id=uuid4(),
                question_id=uuid4(),
                timestamp=datetime.now() - timedelta(days=i),
                predicted_probability=0.8,
                actual_outcome=True,
                brier_score=0.04,
                accuracy=1.0,
                confidence_score=0.9,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
            )
            adjuster.performance_records.append(record)

        # Update profiles
        adjuster._update_agent_profile("current_agent")
        adjuster._update_agent_profile("potential_agent")

        # Get recommendations
        recommendations = adjuster.get_rebalancing_recommendations(current_agents)

        assert "should_rebalance" in recommendations
        assert "degraded_agents" in recommendations
        assert "recommended_additions" in recommendations
        assert "recommended_removals" in recommendations
        assert "performance_summary" in recommendations

        # Should recommend rebalancing due to poor performance
        assert recommendations["should_rebalance"]
        assert len(recommendations["degraded_agents"]) > 0
        assert len(recommendations["recommended_additions"]) > 0
