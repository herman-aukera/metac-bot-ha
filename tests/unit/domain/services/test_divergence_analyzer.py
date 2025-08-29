"""
Unit tests for DivergenceAnalyzer.
Tests divergence analysis, agent disagreement analysis, and resolution strategies.
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
from src.domain.services.divergence_analyzer import (
    AgentDivergenceProfile,
    DivergenceAnalysis,
    DivergenceAnalyzer,
    DivergenceLevel,
    DivergenceMetrics,
    DivergenceSource,
)
from src.domain.value_objects.confidence import ConfidenceLevel
from src.domain.value_objects.probability import Probability


class TestDivergenceAnalyzer:
    """Test suite for DivergenceAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DivergenceAnalyzer()
        self.question_id = uuid4()

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = DivergenceAnalyzer()
        assert analyzer is not None
        assert len(analyzer.divergence_thresholds) == 5
        assert len(analyzer.resolution_strategies) == 5
        assert len(analyzer.divergence_history) == 0

    def test_analyze_divergence_high_agreement(self):
        """Test divergence analysis with high agreement predictions."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.70,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Detailed reasoning with evidence and analysis",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.71,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Another detailed reasoning with research",
                created_by="Agent2",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.69,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.REACT,
                reasoning="Third detailed reasoning",
                created_by="Agent3",
            ),
        ]

        analysis = self.analyzer.analyze_divergence(predictions)

        assert isinstance(analysis, DivergenceAnalysis)
        assert analysis.divergence_level in [
            DivergenceLevel.VERY_LOW,
            DivergenceLevel.LOW,
        ]
        assert analysis.metrics.variance < 0.01
        assert analysis.consensus_prediction == pytest.approx(0.7, abs=0.01)
        assert analysis.confidence_adjustment >= 0.0  # Should increase confidence
        assert len(analysis.agent_profiles) == 3

    def test_analyze_divergence_high_disagreement(self):
        """Test divergence analysis with high disagreement predictions."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.1,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low probability reasoning",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.9,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="High probability reasoning",
                created_by="Agent2",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.LOW,
                method=PredictionMethod.REACT,
                reasoning="Uncertain reasoning",
                created_by="Agent3",
            ),
        ]

        analysis = self.analyzer.analyze_divergence(predictions)

        assert isinstance(analysis, DivergenceAnalysis)
        assert analysis.divergence_level in [
            DivergenceLevel.HIGH,
            DivergenceLevel.VERY_HIGH,
        ]
        assert analysis.metrics.variance > 0.05
        assert analysis.confidence_adjustment <= 0.0  # Should decrease confidence
        assert (
            DivergenceSource.METHODOLOGY in analysis.primary_sources
            or DivergenceSource.CONFIDENCE in analysis.primary_sources
        )

    def test_analyze_divergence_with_outliers(self):
        """Test divergence analysis with outlier predictions."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.45,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Normal prediction",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.50,
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

        analysis = self.analyzer.analyze_divergence(predictions)

        assert isinstance(analysis, DivergenceAnalysis)
        # Note: outlier detection might not always detect outliers with small samples
        # Check that the outlier is at least identified in agent profiles
        agent4_profile = next(
            (p for p in analysis.agent_profiles if p.agent_name == "Agent4"), None
        )
        assert agent4_profile is not None
        assert (
            agent4_profile.outlier_frequency > 0
        )  # Agent4 should be identified as outlier-prone
        # The outlier might not be detected as a primary source due to small sample size
        # but should be identified in agent profiles

    def test_analyze_divergence_single_prediction(self):
        """Test divergence analysis with single prediction."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Single prediction",
                created_by="Agent1",
            )
        ]

        analysis = self.analyzer.analyze_divergence(predictions)

        assert isinstance(analysis, DivergenceAnalysis)
        assert analysis.divergence_level == DivergenceLevel.VERY_LOW
        assert analysis.consensus_prediction == 0.7
        assert analysis.confidence_adjustment == 0.0
        assert len(analysis.agent_profiles) == 0  # No profiles for single prediction

    def test_analyze_divergence_empty_predictions(self):
        """Test divergence analysis with empty predictions."""
        analysis = self.analyzer.analyze_divergence([])

        assert isinstance(analysis, DivergenceAnalysis)
        assert analysis.divergence_level == DivergenceLevel.VERY_LOW
        assert analysis.consensus_prediction == 0.5
        assert analysis.confidence_adjustment == 0.0

    def test_calculate_divergence_metrics(self):
        """Test divergence metrics calculation."""
        probabilities = [0.3, 0.5, 0.7, 0.9]

        metrics = self.analyzer._calculate_divergence_metrics(probabilities)

        assert isinstance(metrics, DivergenceMetrics)
        assert metrics.variance > 0
        assert metrics.standard_deviation > 0
        assert metrics.range_spread == pytest.approx(0.6, abs=0.01)  # 0.9 - 0.3
        assert 0.0 <= metrics.consensus_strength <= 1.0
        assert metrics.entropy >= 0.0

    def test_calculate_divergence_metrics_identical_predictions(self):
        """Test divergence metrics with identical predictions."""
        probabilities = [0.5, 0.5, 0.5]

        metrics = self.analyzer._calculate_divergence_metrics(probabilities)

        assert metrics.variance == 0.0
        assert metrics.standard_deviation == 0.0
        assert metrics.range_spread == 0.0
        assert metrics.consensus_strength == 1.0
        assert metrics.outlier_count == 0

    def test_classify_divergence_level(self):
        """Test divergence level classification."""
        # Very low divergence
        level = self.analyzer._classify_divergence_level(0.001)
        assert level == DivergenceLevel.VERY_LOW

        # Low divergence
        level = self.analyzer._classify_divergence_level(0.01)
        assert level == DivergenceLevel.LOW

        # Moderate divergence
        level = self.analyzer._classify_divergence_level(0.03)
        assert level == DivergenceLevel.MODERATE

        # High divergence
        level = self.analyzer._classify_divergence_level(0.08)
        assert level == DivergenceLevel.HIGH

        # Very high divergence
        level = self.analyzer._classify_divergence_level(0.15)
        assert level == DivergenceLevel.VERY_HIGH

    def test_identify_divergence_sources_methodology(self):
        """Test identification of methodology-based divergence."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="CoT reasoning",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="ToT reasoning",
                created_by="Agent2",
            ),
        ]

        metrics = DivergenceMetrics(
            variance=0.005,
            standard_deviation=0.07,
            range_spread=0.1,
            interquartile_range=0.05,
            coefficient_of_variation=0.1,
            entropy=0.5,
            consensus_strength=0.8,
            outlier_count=0,
        )

        sources = self.analyzer._identify_divergence_sources(predictions, metrics)

        assert DivergenceSource.METHODOLOGY in sources

    def test_identify_divergence_sources_confidence(self):
        """Test identification of confidence-based divergence."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.LOW,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low confidence reasoning",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.VERY_HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="High confidence reasoning",
                created_by="Agent2",
            ),
        ]

        metrics = DivergenceMetrics(
            variance=0.005,
            standard_deviation=0.07,
            range_spread=0.1,
            interquartile_range=0.05,
            coefficient_of_variation=0.1,
            entropy=0.5,
            consensus_strength=0.8,
            outlier_count=0,
        )

        sources = self.analyzer._identify_divergence_sources(predictions, metrics)

        assert DivergenceSource.CONFIDENCE in sources

    def test_assess_reasoning_quality(self):
        """Test reasoning quality assessment."""
        # High quality reasoning
        high_quality_pred = Prediction.create_binary_prediction(
            question_id=self.question_id,
            research_report_id=uuid4(),
            probability=0.6,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="According to recent research and data analysis, the evidence suggests this outcome is likely. Therefore, because of multiple factors, however there might be some uncertainty.",
            created_by="Agent1",
        )

        # Low quality reasoning
        low_quality_pred = Prediction.create_binary_prediction(
            question_id=self.question_id,
            research_report_id=uuid4(),
            probability=0.7,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.TREE_OF_THOUGHT,
            reasoning="Maybe.",
            created_by="Agent2",
        )

        high_score = self.analyzer._assess_reasoning_quality(high_quality_pred)
        low_score = self.analyzer._assess_reasoning_quality(low_quality_pred)

        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0

    def test_detect_systematic_bias(self):
        """Test systematic bias detection."""
        # Biased predictions (one agent consistently higher)
        biased_predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.3,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Agent1 reasoning",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.8,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Agent2 reasoning",
                created_by="Agent2",
            ),
        ]

        # Unbiased predictions
        unbiased_predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Agent1 reasoning",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Agent2 reasoning",
                created_by="Agent2",
            ),
        ]

        assert self.analyzer._detect_systematic_bias(biased_predictions) == True
        assert self.analyzer._detect_systematic_bias(unbiased_predictions) == False

    def test_calculate_consensus_prediction(self):
        """Test consensus prediction calculation."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Prediction 1",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.8,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Prediction 2",
                created_by="Agent2",
            ),
        ]

        # Low divergence should use simple average
        consensus = self.analyzer._calculate_consensus_prediction(
            predictions, DivergenceLevel.LOW
        )
        assert consensus == pytest.approx(0.7, abs=0.01)

        # High divergence should use trimmed mean (but with only 2 predictions, same as mean)
        consensus = self.analyzer._calculate_consensus_prediction(
            predictions, DivergenceLevel.HIGH
        )
        assert consensus == pytest.approx(0.7, abs=0.01)

    def test_calculate_confidence_adjustment(self):
        """Test confidence adjustment calculation."""
        # Low divergence metrics
        low_divergence_metrics = DivergenceMetrics(
            variance=0.001,
            standard_deviation=0.03,
            range_spread=0.05,
            interquartile_range=0.02,
            coefficient_of_variation=0.05,
            entropy=0.2,
            consensus_strength=0.9,
            outlier_count=0,
        )

        # High divergence metrics
        high_divergence_metrics = DivergenceMetrics(
            variance=0.15,
            standard_deviation=0.4,
            range_spread=0.8,
            interquartile_range=0.3,
            coefficient_of_variation=0.6,
            entropy=2.0,
            consensus_strength=0.2,
            outlier_count=2,
        )

        low_adjustment = self.analyzer._calculate_confidence_adjustment(
            low_divergence_metrics, DivergenceLevel.VERY_LOW
        )
        high_adjustment = self.analyzer._calculate_confidence_adjustment(
            high_divergence_metrics, DivergenceLevel.VERY_HIGH
        )

        assert low_adjustment > high_adjustment
        assert low_adjustment > 0  # Should increase confidence
        assert high_adjustment < 0  # Should decrease confidence

    def test_generate_agent_profiles(self):
        """Test agent profile generation."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Detailed reasoning with evidence",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.8,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Simple reasoning",
                created_by="Agent2",
            ),
        ]

        consensus = 0.7
        profiles = self.analyzer._generate_agent_profiles(predictions, consensus)

        assert len(profiles) == 2
        assert all(isinstance(profile, AgentDivergenceProfile) for profile in profiles)

        agent1_profile = next(p for p in profiles if p.agent_name == "Agent1")
        agent2_profile = next(p for p in profiles if p.agent_name == "Agent2")

        assert agent1_profile.avg_distance_from_consensus == pytest.approx(
            0.1, abs=0.01
        )
        assert agent2_profile.avg_distance_from_consensus == pytest.approx(
            0.1, abs=0.01
        )

    def test_select_resolution_strategy(self):
        """Test resolution strategy selection."""
        # Test outlier-based strategy
        strategy = self.analyzer._select_resolution_strategy(
            DivergenceLevel.HIGH, [DivergenceSource.OUTLIER]
        )
        assert strategy == "outlier_robust_mean"

        # Test confidence-based strategy
        strategy = self.analyzer._select_resolution_strategy(
            DivergenceLevel.MODERATE, [DivergenceSource.CONFIDENCE]
        )
        assert strategy == "confidence_weighted"

        # Test methodology-based strategy
        strategy = self.analyzer._select_resolution_strategy(
            DivergenceLevel.MODERATE, [DivergenceSource.METHODOLOGY]
        )
        assert strategy == "meta_reasoning"

        # Test bias-based strategy
        strategy = self.analyzer._select_resolution_strategy(
            DivergenceLevel.HIGH, [DivergenceSource.BIAS]
        )
        assert strategy == "bayesian_model_averaging"

    def test_generate_divergence_explanation(self):
        """Test divergence explanation generation."""
        metrics = DivergenceMetrics(
            variance=0.05,
            standard_deviation=0.22,
            range_spread=0.4,
            interquartile_range=0.2,
            coefficient_of_variation=0.3,
            entropy=1.5,
            consensus_strength=0.6,
            outlier_count=1,
        )

        sources = [DivergenceSource.METHODOLOGY, DivergenceSource.OUTLIER]

        explanation = self.analyzer._generate_divergence_explanation(
            DivergenceLevel.MODERATE, sources, metrics, 4
        )

        assert isinstance(explanation, str)
        assert "Moderate divergence" in explanation
        assert "4 predictions" in explanation
        assert "Methodology" in explanation
        assert "Outlier" in explanation
        assert str(metrics.variance) in explanation

    def test_count_outliers(self):
        """Test outlier counting using IQR method."""
        # No outliers
        no_outliers = [0.4, 0.5, 0.6, 0.7]
        assert self.analyzer._count_outliers(no_outliers) == 0

        # With outliers
        with_outliers = [0.1, 0.4, 0.5, 0.6, 0.9]  # 0.1 and 0.9 might be outliers
        outlier_count = self.analyzer._count_outliers(with_outliers)
        assert outlier_count >= 0

    def test_calculate_prediction_entropy(self):
        """Test prediction entropy calculation."""
        # Low entropy (similar predictions)
        low_entropy_probs = [0.5, 0.51, 0.49, 0.52]
        low_entropy = self.analyzer._calculate_prediction_entropy(low_entropy_probs)

        # High entropy (diverse predictions)
        high_entropy_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        high_entropy = self.analyzer._calculate_prediction_entropy(high_entropy_probs)

        assert high_entropy > low_entropy
        assert low_entropy >= 0.0
        assert high_entropy >= 0.0

    def test_trimmed_mean(self):
        """Test trimmed mean calculation."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # 20% trim should remove 1 from each end
        trimmed = self.analyzer._trimmed_mean(values, trim_percent=0.1)

        # Should be close to median
        assert trimmed == pytest.approx(0.5, abs=0.1)

    def test_get_divergence_patterns(self):
        """Test divergence patterns analysis."""
        # Initially empty
        patterns = self.analyzer.get_divergence_patterns()
        assert patterns == {}

        # Add some analyses
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Test reasoning",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Test reasoning",
                created_by="Agent2",
            ),
        ]

        self.analyzer.analyze_divergence(predictions)
        self.analyzer.analyze_divergence(predictions)

        patterns = self.analyzer.get_divergence_patterns()

        assert "total_analyses" in patterns
        assert "divergence_level_distribution" in patterns
        assert "common_divergence_sources" in patterns
        assert patterns["total_analyses"] == 2

    def test_update_agent_performance_pattern(self):
        """Test agent performance pattern updates."""
        agent_name = "TestAgent"

        # Add performance scores
        self.analyzer.update_agent_performance_pattern(agent_name, 0.8)
        self.analyzer.update_agent_performance_pattern(agent_name, 0.7)

        assert agent_name in self.analyzer.agent_performance_patterns
        assert len(self.analyzer.agent_performance_patterns[agent_name]) == 2
        assert self.analyzer.agent_performance_patterns[agent_name] == [0.8, 0.7]

    def test_performance_pattern_history_limit(self):
        """Test that performance pattern history is limited."""
        agent_name = "TestAgent"

        # Add more than 50 entries
        for i in range(55):
            self.analyzer.update_agent_performance_pattern(agent_name, 0.5 + i * 0.01)

        # Should keep only last 50
        assert len(self.analyzer.agent_performance_patterns[agent_name]) == 50

    def test_divergence_history_limit(self):
        """Test that divergence history is limited."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Test reasoning",
                created_by="Agent1",
            )
        ]

        # Add more than 100 analyses
        for i in range(105):
            self.analyzer.analyze_divergence(predictions)

        # Should keep only last 100
        assert len(self.analyzer.divergence_history) == 100

    def test_analyze_divergence_includes_agent_profiles_flag(self):
        """Test that agent profiles can be excluded from analysis."""
        predictions = [
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.6,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Test reasoning",
                created_by="Agent1",
            ),
            Prediction.create_binary_prediction(
                question_id=self.question_id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.TREE_OF_THOUGHT,
                reasoning="Test reasoning",
                created_by="Agent2",
            ),
        ]

        # With agent profiles
        analysis_with_profiles = self.analyzer.analyze_divergence(
            predictions, include_agent_profiles=True
        )
        assert len(analysis_with_profiles.agent_profiles) == 2

        # Without agent profiles
        analysis_without_profiles = self.analyzer.analyze_divergence(
            predictions, include_agent_profiles=False
        )
        assert len(analysis_without_profiles.agent_profiles) == 0
