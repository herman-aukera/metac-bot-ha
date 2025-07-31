"""Unit tests for Agent entity."""

import pytest
from datetime import datetime, timedelta

from src.domain.entities.agent import (
    Agent, ReasoningStyle, AggregationMethod, 
    PerformanceMetrics, PerformanceHistory
)


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics."""

    def test_performance_metrics_initialization_valid(self):
        """Test valid performance metrics initialization."""
        metrics = PerformanceMetrics(
            total_predictions=100,
            correct_predictions=75,
            average_confidence=0.8,
            calibration_score=0.1,
            brier_score=0.2,
            log_score=-0.5,
            accuracy_by_category={"tech": 0.8, "politics": 0.7},
            confidence_intervals={"90%": 0.85, "95%": 0.9}
        )

        assert metrics.total_predictions == 100
        assert metrics.correct_predictions == 75
        assert metrics.average_confidence == 0.8
        assert metrics.calibration_score == 0.1
        assert metrics.brier_score == 0.2
        assert metrics.log_score == -0.5
        assert metrics.accuracy_by_category == {"tech": 0.8, "politics": 0.7}
        assert metrics.confidence_intervals == {"90%": 0.85, "95%": 0.9}

    def test_performance_metrics_validation_negative_predictions(self):
        """Test performance metrics validation with negative predictions."""
        with pytest.raises(ValueError, match="Total predictions cannot be negative"):
            PerformanceMetrics(total_predictions=-1)

        with pytest.raises(ValueError, match="Correct predictions cannot be negative"):
            PerformanceMetrics(correct_predictions=-1)

    def test_performance_metrics_validation_invalid_confidence(self):
        """Test performance metrics validation with invalid confidence."""
        with pytest.raises(ValueError, match="Average confidence must be between 0.0 and 1.0"):
            PerformanceMetrics(average_confidence=1.5)

    def test_performance_metrics_validation_correct_exceeds_total(self):
        """Test performance metrics validation when correct exceeds total."""
        with pytest.raises(ValueError, match="Correct predictions cannot exceed total predictions"):
            PerformanceMetrics(total_predictions=50, correct_predictions=75)

    def test_performance_metrics_accuracy_calculation(self):
        """Test accuracy calculation."""
        metrics = PerformanceMetrics(total_predictions=100, correct_predictions=80)
        assert metrics.get_accuracy() == 0.8

        # Test zero predictions
        zero_metrics = PerformanceMetrics(total_predictions=0, correct_predictions=0)
        assert zero_metrics.get_accuracy() == 0.0

    def test_performance_metrics_calibration_check(self):
        """Test calibration checking."""
        well_calibrated = PerformanceMetrics(calibration_score=0.05)
        poorly_calibrated = PerformanceMetrics(calibration_score=0.2)

        assert well_calibrated.is_well_calibrated()
        assert not poorly_calibrated.is_well_calibrated()

    def test_performance_metrics_sufficient_data(self):
        """Test sufficient data checking."""
        sufficient = PerformanceMetrics(total_predictions=20)
        insufficient = PerformanceMetrics(total_predictions=5)

        assert sufficient.has_sufficient_data()
        assert not insufficient.has_sufficient_data()

    def test_performance_metrics_defaults(self):
        """Test performance metrics default values."""
        metrics = PerformanceMetrics()
        
        assert metrics.total_predictions == 0
        assert metrics.correct_predictions == 0
        assert metrics.average_confidence == 0.0
        assert metrics.calibration_score == 0.0
        assert metrics.brier_score == 0.0
        assert metrics.log_score == 0.0
        assert metrics.accuracy_by_category == {}
        assert metrics.confidence_intervals == {}
        assert metrics.last_updated is not None


class TestPerformanceHistory:
    """Test cases for PerformanceHistory."""

    def create_sample_metrics(self, accuracy: float = 0.8) -> PerformanceMetrics:
        """Helper to create sample metrics."""
        total = 100
        correct = int(total * accuracy)
        return PerformanceMetrics(
            total_predictions=total,
            correct_predictions=correct,
            average_confidence=0.7
        )

    def test_performance_history_initialization_valid(self):
        """Test valid performance history initialization."""
        current = self.create_sample_metrics(0.8)
        historical = [self.create_sample_metrics(0.7), self.create_sample_metrics(0.75)]
        tournament_perf = {1: self.create_sample_metrics(0.85)}
        question_type_perf = {"binary": self.create_sample_metrics(0.9)}

        history = PerformanceHistory(
            current_metrics=current,
            historical_metrics=historical,
            tournament_performance=tournament_perf,
            question_type_performance=question_type_perf,
            recent_trend="improving"
        )

        assert history.current_metrics == current
        assert len(history.historical_metrics) == 2
        assert history.tournament_performance == tournament_perf
        assert history.question_type_performance == question_type_perf
        assert history.recent_trend == "improving"

    def test_performance_history_validation(self):
        """Test performance history validation."""
        current = self.create_sample_metrics()

        with pytest.raises(ValueError, match="Historical metrics must be a list"):
            PerformanceHistory(
                current_metrics=current,
                historical_metrics="not a list",
                tournament_performance={},
                question_type_performance={}
            )

    def test_performance_history_trend_analysis(self):
        """Test performance trend analysis."""
        current = self.create_sample_metrics(0.8)
        
        # Improving trend
        improving_history = [
            self.create_sample_metrics(0.6),
            self.create_sample_metrics(0.65),
            self.create_sample_metrics(0.7),
            self.create_sample_metrics(0.75),
            self.create_sample_metrics(0.8)
        ]
        
        improving_perf_history = PerformanceHistory(
            current_metrics=current,
            historical_metrics=improving_history,
            tournament_performance={},
            question_type_performance={}
        )
        
        assert improving_perf_history.get_performance_trend() == "improving"

        # Declining trend
        declining_history = [
            self.create_sample_metrics(0.8),
            self.create_sample_metrics(0.75),
            self.create_sample_metrics(0.7),
            self.create_sample_metrics(0.65),
            self.create_sample_metrics(0.6)
        ]
        
        declining_perf_history = PerformanceHistory(
            current_metrics=current,
            historical_metrics=declining_history,
            tournament_performance={},
            question_type_performance={}
        )
        
        assert declining_perf_history.get_performance_trend() == "declining"

        # Insufficient data
        insufficient_history = PerformanceHistory(
            current_metrics=current,
            historical_metrics=[self.create_sample_metrics(0.7)],
            tournament_performance={},
            question_type_performance={}
        )
        
        assert insufficient_history.get_performance_trend() == "insufficient_data"

    def test_performance_history_best_tournament(self):
        """Test best tournament performance identification."""
        current = self.create_sample_metrics()
        tournament_perf = {
            1: self.create_sample_metrics(0.7),
            2: self.create_sample_metrics(0.9),
            3: self.create_sample_metrics(0.8)
        }
        
        history = PerformanceHistory(
            current_metrics=current,
            historical_metrics=[],
            tournament_performance=tournament_perf,
            question_type_performance={}
        )
        
        best_tournament, best_metrics = history.get_best_tournament_performance()
        assert best_tournament == 2
        assert best_metrics.get_accuracy() == 0.9

        # Test empty tournament performance
        empty_history = PerformanceHistory(
            current_metrics=current,
            historical_metrics=[],
            tournament_performance={},
            question_type_performance={}
        )
        
        assert empty_history.get_best_tournament_performance() is None

    def test_performance_history_strongest_question_type(self):
        """Test strongest question type identification."""
        current = self.create_sample_metrics()
        question_type_perf = {
            "binary": self.create_sample_metrics(0.8),
            "numeric": self.create_sample_metrics(0.9),
            "multiple_choice": self.create_sample_metrics(0.7)
        }
        
        history = PerformanceHistory(
            current_metrics=current,
            historical_metrics=[],
            tournament_performance={},
            question_type_performance=question_type_perf
        )
        
        strongest = history.get_strongest_question_type()
        assert strongest == "numeric"

        # Test empty question type performance
        empty_history = PerformanceHistory(
            current_metrics=current,
            historical_metrics=[],
            tournament_performance={},
            question_type_performance={}
        )
        
        assert empty_history.get_strongest_question_type() is None


class TestAgent:
    """Test cases for Agent entity."""

    def create_sample_performance_history(self) -> PerformanceHistory:
        """Helper to create sample performance history."""
        current = PerformanceMetrics(
            total_predictions=100,
            correct_predictions=80,
            average_confidence=0.7
        )
        
        return PerformanceHistory(
            current_metrics=current,
            historical_metrics=[],
            tournament_performance={},
            question_type_performance={}
        )

    def test_agent_initialization_valid(self):
        """Test valid agent initialization."""
        performance_history = self.create_sample_performance_history()
        created_at = datetime.utcnow()
        
        agent = Agent(
            id="agent_001",
            name="Test Agent",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["AI", "Technology"],
            performance_history=performance_history,
            configuration={"temperature": 0.7, "max_tokens": 1000},
            created_at=created_at,
            last_active=created_at,
            is_active=True,
            version="1.0.0"
        )

        assert agent.id == "agent_001"
        assert agent.name == "Test Agent"
        assert agent.reasoning_style == ReasoningStyle.CHAIN_OF_THOUGHT
        assert agent.knowledge_domains == ["AI", "Technology"]
        assert agent.performance_history == performance_history
        assert agent.configuration == {"temperature": 0.7, "max_tokens": 1000}
        assert agent.created_at == created_at
        assert agent.last_active == created_at
        assert agent.is_active is True
        assert agent.version == "1.0.0"

    def test_agent_validation_empty_id(self):
        """Test agent validation with empty ID."""
        performance_history = self.create_sample_performance_history()
        
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            Agent(
                id="",
                name="Test Agent",
                reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
                knowledge_domains=["AI"],
                performance_history=performance_history,
                configuration={},
                created_at=datetime.utcnow()
            )

    def test_agent_validation_empty_name(self):
        """Test agent validation with empty name."""
        performance_history = self.create_sample_performance_history()
        
        with pytest.raises(ValueError, match="Agent name cannot be empty"):
            Agent(
                id="agent_001",
                name="",
                reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
                knowledge_domains=["AI"],
                performance_history=performance_history,
                configuration={},
                created_at=datetime.utcnow()
            )

    def test_agent_domain_expertise(self):
        """Test domain expertise checking."""
        performance_history = self.create_sample_performance_history()
        
        agent = Agent(
            id="agent_001",
            name="AI Expert",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["AI", "Machine Learning", "Technology"],
            performance_history=performance_history,
            configuration={},
            created_at=datetime.utcnow()
        )

        assert agent.has_domain_expertise("AI")
        assert agent.has_domain_expertise("ai")  # Case insensitive
        assert agent.has_domain_expertise("Technology")
        assert not agent.has_domain_expertise("Politics")

    def test_agent_performance_methods(self):
        """Test agent performance-related methods."""
        current_metrics = PerformanceMetrics(
            total_predictions=100,
            correct_predictions=85,
            calibration_score=0.05
        )
        
        performance_history = PerformanceHistory(
            current_metrics=current_metrics,
            historical_metrics=[],
            tournament_performance={},
            question_type_performance={}
        )
        
        agent = Agent(
            id="agent_001",
            name="High Performer",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["AI"],
            performance_history=performance_history,
            configuration={},
            created_at=datetime.utcnow()
        )

        assert agent.get_current_accuracy() == 0.85
        assert agent.is_well_calibrated()
        assert agent.has_sufficient_performance_data()

    def test_agent_specialization_score(self):
        """Test specialization score calculation."""
        performance_history = self.create_sample_performance_history()
        
        # Add category performance
        performance_history.current_metrics.accuracy_by_category = {
            "AI": 0.9,
            "Technology": 0.8
        }
        
        agent = Agent(
            id="agent_001",
            name="AI Specialist",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["AI", "Machine Learning"],
            performance_history=performance_history,
            configuration={},
            created_at=datetime.utcnow()
        )

        # High specialization for AI (domain expertise + good performance)
        ai_score = agent.get_specialization_score("AI")
        assert ai_score > 0.8

        # Lower specialization for unknown domain
        politics_score = agent.get_specialization_score("Politics")
        assert politics_score < 0.7

    def test_agent_suitability_check(self):
        """Test agent suitability for questions."""
        performance_history = self.create_sample_performance_history()
        performance_history.current_metrics.accuracy_by_category = {"AI": 0.9}
        
        agent = Agent(
            id="agent_001",
            name="AI Expert",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["AI"],
            performance_history=performance_history,
            configuration={},
            created_at=datetime.utcnow(),
            is_active=True
        )

        assert agent.is_suitable_for_question("AI", required_confidence=0.6)
        assert not agent.is_suitable_for_question("Politics", required_confidence=0.8)

        # Inactive agent should not be suitable
        inactive_agent = agent.deactivate()
        assert not inactive_agent.is_suitable_for_question("AI", required_confidence=0.6)

    def test_agent_update_methods(self):
        """Test agent update methods."""
        performance_history = self.create_sample_performance_history()
        original_time = datetime.utcnow() - timedelta(hours=1)
        
        agent = Agent(
            id="agent_001",
            name="Test Agent",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["AI"],
            performance_history=performance_history,
            configuration={},
            created_at=original_time,
            last_active=original_time,
            is_active=True
        )

        # Test update last active
        updated_agent = agent.update_last_active()
        assert updated_agent.last_active > agent.last_active
        assert updated_agent.id == agent.id  # Other fields unchanged

        # Test deactivate
        deactivated_agent = agent.deactivate()
        assert not deactivated_agent.is_active
        assert deactivated_agent.id == agent.id

        # Test update performance
        new_performance = self.create_sample_performance_history()
        performance_updated_agent = agent.update_performance(new_performance)
        assert performance_updated_agent.performance_history == new_performance
        assert performance_updated_agent.id == agent.id

    def test_agent_summary_methods(self):
        """Test agent summary generation methods."""
        performance_history = self.create_sample_performance_history()
        performance_history.current_metrics.accuracy_by_category = {"AI": 0.9}
        
        agent = Agent(
            id="agent_001",
            name="AI Expert",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["AI", "Machine Learning", "Technology"],
            performance_history=performance_history,
            configuration={},
            created_at=datetime.utcnow(),
            is_active=True
        )

        # Test get_agent_summary
        summary = agent.get_agent_summary()
        assert summary["id"] == "agent_001"
        assert summary["name"] == "AI Expert"
        assert summary["reasoning_style"] == "chain_of_thought"
        assert summary["knowledge_domains"] == ["AI", "Machine Learning", "Technology"]
        assert summary["is_active"] is True
        assert summary["current_accuracy"] == 0.8

        # Test to_summary
        text_summary = agent.to_summary()
        assert "agent_001" in text_summary
        assert "AI Expert" in text_summary
        assert "chain_of_thought" in text_summary
        assert "0.800" in text_summary
        assert "active" in text_summary

    def test_agent_defaults(self):
        """Test agent default values."""
        performance_history = self.create_sample_performance_history()
        
        agent = Agent(
            id="agent_001",
            name="Test Agent",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["AI"],
            performance_history=performance_history,
            configuration={},
            created_at=datetime.utcnow()
        )

        assert agent.last_active is None
        assert agent.is_active is True
        assert agent.version == "1.0.0"
        assert agent.metadata == {}

    def test_reasoning_style_enum(self):
        """Test reasoning style enum values."""
        assert ReasoningStyle.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningStyle.TREE_OF_THOUGHT.value == "tree_of_thought"
        assert ReasoningStyle.REACT.value == "react"
        assert ReasoningStyle.ENSEMBLE.value == "ensemble"
        assert ReasoningStyle.BAYESIAN.value == "bayesian"
        assert ReasoningStyle.FREQUENTIST.value == "frequentist"
        assert ReasoningStyle.INTUITIVE.value == "intuitive"
        assert ReasoningStyle.ANALYTICAL.value == "analytical"

    def test_aggregation_method_enum(self):
        """Test aggregation method enum values."""
        assert AggregationMethod.SIMPLE_AVERAGE.value == "simple_average"
        assert AggregationMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert AggregationMethod.CONFIDENCE_WEIGHTED.value == "confidence_weighted"
        assert AggregationMethod.MEDIAN.value == "median"
        assert AggregationMethod.TRIMMED_MEAN.value == "trimmed_mean"
        assert AggregationMethod.META_REASONING.value == "meta_reasoning"
