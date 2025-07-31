"""Comprehensive unit tests for all domain value objects."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.reasoning_step import ReasoningStep
from src.domain.value_objects.strategy_result import StrategyResult, StrategyType, StrategyOutcome
from src.domain.value_objects.source_credibility import SourceCredibility
from src.domain.value_objects.prediction_result import PredictionResult, PredictionType
from src.domain.value_objects.consensus_metrics import ConsensusMetrics


class TestConfidenceComprehensive:
    """Comprehensive tests for Confidence value object."""

    def test_confidence_boundary_values(self):
        """Test confidence with boundary values."""
        # Minimum confidence
        min_conf = Confidence(level=0.0, basis="No confidence at all")
        assert min_conf.level == 0.0
        assert not min_conf.is_high()
        assert not min_conf.is_medium()
        assert min_conf.is_low()

        # Maximum confidence
        max_conf = Confidence(level=1.0, basis="Complete certainty")
        assert max_conf.level == 1.0
        assert max_conf.is_high()
        assert not max_conf.is_medium()
        assert not max_conf.is_low()

        # Boundary between low and medium (0.5)
        boundary_conf = Confidence(level=0.5, basis="Exactly at boundary")
        assert not boundary_conf.is_low()
        assert boundary_conf.is_medium()
        assert not boundary_conf.is_high()

    def test_confidence_combination_edge_cases(self):
        """Test confidence combination edge cases."""
        high_conf = Confidence(level=0.9, basis="High confidence")
        low_conf = Confidence(level=0.1, basis="Low confidence")

        # Combine with extreme weights
        mostly_high = high_conf.combine_with(low_conf, weight=0.99)
        assert mostly_high.level > 0.85

        mostly_low = high_conf.combine_with(low_conf, weight=0.01)
        assert mostly_low.level < 0.15

        # Combine with equal weight
        balanced = high_conf.combine_with(low_conf, weight=0.5)
        assert abs(balanced.level - 0.5) < 0.001

        # Test basis combination
        assert "Combined:" in balanced.basis
        assert "High confidence" in balanced.basis
        assert "Low confidence" in balanced.basis

    def test_confidence_factory_methods_precision(self):
        """Test precision of factory methods."""
        high = Confidence.high("Test")
        assert high.level == 0.9
        assert high.is_high()

        medium = Confidence.medium("Test")
        assert medium.level == 0.6
        assert medium.is_medium()

        low = Confidence.low("Test")
        assert low.level == 0.3
        assert low.is_low()

    def test_confidence_immutability(self):
        """Test that confidence objects are truly immutable."""
        conf = Confidence(level=0.8, basis="Test")

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            conf.level = 0.9

        with pytest.raises(AttributeError):
            conf.basis = "Modified"

    def test_confidence_string_representation(self):
        """Test string representation of confidence."""
        conf = Confidence(level=0.75, basis="Strong evidence from multiple sources")
        str_repr = str(conf)
        assert "0.75" in str_repr
        assert "Strong evidence" in str_repr

    def test_confidence_equality_and_hashing(self):
        """Test confidence equality and hashing."""
        conf1 = Confidence(level=0.8, basis="Same basis")
        conf2 = Confidence(level=0.8, basis="Same basis")
        conf3 = Confidence(level=0.7, basis="Same basis")
        conf4 = Confidence(level=0.8, basis="Different basis")

        # Test equality
        assert conf1 == conf2
        assert conf1 != conf3
        assert conf1 != conf4

        # Test hashing (should be able to use in sets/dicts)
        conf_set = {conf1, conf2, conf3, conf4}
        assert len(conf_set) == 3  # conf1 and conf2 are the same


class TestReasoningStepComprehensive:
    """Comprehensive tests for ReasoningStep value object."""

    def test_reasoning_step_data_validation(self):
        """Test reasoning step data validation."""
        confidence = Confidence(level=0.8, basis="Test")
        timestamp = datetime.utcnow()

        # Test with complex nested data structures
        complex_input = {
            'question': 'Complex question',
            'context': {
                'historical_data': [1, 2, 3, 4, 5],
                'metadata': {'source': 'test', 'quality': 'high'}
            },
            'parameters': {'temperature': 0.7, 'max_tokens': 1000}
        }

        complex_output = {
            'analysis': 'Detailed analysis result',
            'confidence_factors': ['factor1', 'factor2', 'factor3'],
            'intermediate_results': {
                'step1': 0.6,
                'step2': 0.8,
                'final': 0.75
            }
        }

        step = ReasoningStep(
            step_number=1,
            description="Complex reasoning step",
            input_data=complex_input,
            output_data=complex_output,
            confidence=confidence,
            timestamp=timestamp,
            reasoning_type="analytical"
        )

        assert step.input_data['context']['metadata']['quality'] == 'high'
        assert step.output_data['intermediate_results']['final'] == 0.75

    def test_reasoning_step_factory_method_defaults(self):
        """Test reasoning step factory method with defaults."""
        step = ReasoningStep.create(
            step_number=5,
            description="Test step",
            input_data={'input': 'test'},
            output_data={'output': 'result'},
            confidence_level=0.85,
            confidence_basis="Strong reasoning"
        )

        assert step.step_number == 5
        assert step.confidence.level == 0.85
        assert step.reasoning_type is None  # Default
        assert isinstance(step.timestamp, datetime)

    def test_reasoning_step_key_outputs_extraction(self):
        """Test key outputs extraction from complex output data."""
        step = ReasoningStep.create(
            step_number=1,
            description="Test",
            input_data={},
            output_data={
                'result': 'main_result',
                'confidence': 0.9,
                'metadata': {'internal': 'value'},
                'analysis': 'detailed_analysis',
                'score': 0.85
            },
            confidence_level=0.8,
            confidence_basis="Test"
        )

        key_outputs = step.get_key_outputs()
        assert 'result' in key_outputs
        assert 'confidence' in key_outputs
        assert 'analysis' in key_outputs
        assert 'score' in key_outputs
        # Metadata should be filtered out as it's typically internal
        assert key_outputs.get('metadata') is None

    def test_reasoning_step_summary_generation(self):
        """Test reasoning step summary generation."""
        high_conf_step = ReasoningStep.create(
            step_number=3,
            description="High confidence analysis",
            input_data={'question': 'test'},
            output_data={'result': 'success'},
            confidence_level=0.95,
            confidence_basis="Very strong evidence"
        )

        summary = high_conf_step.to_summary()
        assert "Step 3:" in summary
        assert "High confidence analysis" in summary
        assert "very high" in summary.lower()

        low_conf_step = ReasoningStep.create(
            step_number=1,
            description="Uncertain analysis",
            input_data={'question': 'test'},
            output_data={'result': 'uncertain'},
            confidence_level=0.25,
            confidence_basis="Weak evidence"
        )

        summary = low_conf_step.to_summary()
        assert "low" in summary.lower()

    def test_reasoning_step_immutability(self):
        """Test reasoning step immutability."""
        step = ReasoningStep.create(
            step_number=1,
            description="Test",
            input_data={'test': 'data'},
            output_data={'result': 'test'},
            confidence_level=0.8,
            confidence_basis="Test"
        )

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            step.step_number = 2

        with pytest.raises(AttributeError):
            step.description = "Modified"

    def test_reasoning_step_timestamp_handling(self):
        """Test reasoning step timestamp handling."""
        # Test with explicit timestamp
        explicit_time = datetime(2024, 1, 1, 12, 0, 0)
        step_explicit = ReasoningStep(
            step_number=1,
            description="Test",
            input_data={},
            output_data={},
            confidence=Confidence(0.8, "Test"),
            timestamp=explicit_time
        )
        assert step_explicit.timestamp == explicit_time

        # Test with factory method (should use current time)
        with patch('src.domain.value_objects.reasoning_step.datetime') as mock_datetime:
            mock_now = datetime(2024, 6, 15, 10, 30, 0)
            mock_datetime.utcnow.return_value = mock_now

            step_factory = ReasoningStep.create(
                step_number=1,
                description="Test",
                input_data={},
                output_data={},
                confidence_level=0.8,
                confidence_basis="Test"
            )
            assert step_factory.timestamp == mock_now


class TestStrategyResultComprehensive:
    """Comprehensive tests for StrategyResult value object."""

    def test_strategy_result_lifecycle_management(self):
        """Test complete strategy result lifecycle."""
        # Create pending strategy result
        result = StrategyResult.create(
            strategy_type=StrategyType.AGGRESSIVE,
            expected_score=0.8,
            reasoning="High-risk, high-reward approach based on market inefficiencies",
            question_ids=[1, 2, 3, 4, 5],
            confidence_level=0.85,
            confidence_basis="Strong market signals and historical precedent"
        )

        assert result.is_pending()
        assert not result.is_successful()
        assert result.get_score_difference() is None

        # Mark as successful
        success_result = result.mark_success(0.92)
        assert success_result.is_successful()
        assert not success_result.is_pending()
        assert success_result.actual_score == 0.92
        assert abs(success_result.get_score_difference() - 0.12) < 0.001

        # Mark as failure
        failure_result = result.mark_failure(0.45)
        assert not failure_result.is_successful()
        assert failure_result.outcome == StrategyOutcome.FAILURE
        assert failure_result.actual_score == 0.45
        assert abs(failure_result.get_score_difference() - (-0.35)) < 0.001

    def test_strategy_result_metadata_handling(self):
        """Test strategy result metadata handling."""
        complex_metadata = {
            'risk_assessment': {
                'level': 'high',
                'factors': ['market_volatility', 'regulatory_uncertainty'],
                'mitigation_strategies': ['diversification', 'hedging']
            },
            'market_conditions': {
                'volatility': 0.25,
                'trend': 'bullish',
                'sentiment': 'optimistic'
            },
            'execution_details': {
                'start_time': '2024-01-01T10:00:00Z',
                'resources_allocated': 0.3,
                'agents_involved': ['agent1', 'agent2', 'agent3']
            }
        }

        result = StrategyResult.create(
            strategy_type=StrategyType.BALANCED,
            expected_score=0.7,
            reasoning="Balanced approach considering risk factors",
            question_ids=[1, 2],
            confidence_level=0.75,
            confidence_basis="Moderate confidence",
            metadata=complex_metadata
        )

        assert result.metadata['risk_assessment']['level'] == 'high'
        assert result.metadata['market_conditions']['volatility'] == 0.25
        assert len(result.metadata['execution_details']['agents_involved']) == 3

    def test_strategy_result_question_ids_validation(self):
        """Test strategy result question IDs validation."""
        # Valid question IDs
        valid_result = StrategyResult.create(
            strategy_type=StrategyType.CONSERVATIVE,
            expected_score=0.6,
            reasoning="Conservative approach",
            question_ids=[1, 5, 10, 25, 100],
            confidence_level=0.7,
            confidence_basis="Safe approach"
        )
        assert len(valid_result.question_ids) == 5

        # Empty question IDs should be allowed
        empty_result = StrategyResult.create(
            strategy_type=StrategyType.BALANCED,
            expected_score=0.5,
            reasoning="General strategy",
            question_ids=[],
            confidence_level=0.6,
            confidence_basis="General approach"
        )
        assert len(empty_result.question_ids) == 0

        # Invalid question IDs (negative or zero)
        with pytest.raises(ValueError, match="All question IDs must be positive integers"):
            StrategyResult.create(
                strategy_type=StrategyType.AGGRESSIVE,
                expected_score=0.8,
                reasoning="Invalid IDs",
                question_ids=[1, 0, -1],
                confidence_level=0.8,
                confidence_basis="Test"
            )

    def test_strategy_result_summary_variations(self):
        """Test strategy result summary with different scenarios."""
        # Pending aggressive strategy
        pending_aggressive = StrategyResult.create(
            strategy_type=StrategyType.AGGRESSIVE,
            expected_score=0.85,
            reasoning="High-risk approach",
            question_ids=[1, 2, 3],
            confidence_level=0.9,
            confidence_basis="Strong signals"
        )
        summary = pending_aggressive.to_summary()
        assert "aggressive" in summary
        assert "pending" in summary
        assert "0.850" in summary

        # Successful conservative strategy
        successful_conservative = StrategyResult.create(
            strategy_type=StrategyType.CONSERVATIVE,
            expected_score=0.6,
            reasoning="Safe approach",
            question_ids=[4, 5],
            confidence_level=0.7,
            confidence_basis="Historical data"
        ).mark_success(0.75)

        summary = successful_conservative.to_summary()
        assert "conservative" in summary
        assert "success" in summary
        assert "0.750" in summary

    def test_strategy_result_performance_analysis(self):
        """Test strategy result performance analysis."""
        # Over-performing strategy
        over_performer = StrategyResult.create(
            strategy_type=StrategyType.BALANCED,
            expected_score=0.7,
            reasoning="Balanced approach",
            question_ids=[1, 2, 3],
            confidence_level=0.8,
            confidence_basis="Good analysis"
        ).mark_success(0.85)

        assert over_performer.get_score_difference() > 0
        assert over_performer.is_successful()

        # Under-performing strategy
        under_performer = StrategyResult.create(
            strategy_type=StrategyType.AGGRESSIVE,
            expected_score=0.8,
            reasoning="Aggressive approach",
            question_ids=[4, 5, 6],
            confidence_level=0.9,
            confidence_basis="Strong signals"
        ).mark_success(0.65)

        assert under_performer.get_score_difference() < 0
        assert under_performer.is_successful()  # Still successful, just under-performed


class TestSourceCredibilityComprehensive:
    """Comprehensive tests for SourceCredibility value object."""

    def test_source_credibility_score_calculation(self):
        """Test source credibility overall score calculation."""
        # Perfect credibility
        perfect_cred = SourceCredibility(
            authority_score=1.0,
            recency_score=1.0,
            relevance_score=1.0,
            cross_validation_score=1.0
        )
        assert perfect_cred.overall_score == 1.0

        # Poor credibility
        poor_cred = SourceCredibility(
            authority_score=0.2,
            recency_score=0.1,
            relevance_score=0.3,
            cross_validation_score=0.0
        )
        assert poor_cred.overall_score == 0.15

        # Mixed credibility
        mixed_cred = SourceCredibility(
            authority_score=0.9,  # High authority
            recency_score=0.3,   # Old information
            relevance_score=0.8, # Highly relevant
            cross_validation_score=0.6  # Some validation
        )
        expected_score = (0.9 + 0.3 + 0.8 + 0.6) / 4
        assert abs(mixed_cred.overall_score - expected_score) < 0.001

    def test_source_credibility_boundary_values(self):
        """Test source credibility with boundary values."""
        # All minimum values
        min_cred = SourceCredibility(
            authority_score=0.0,
            recency_score=0.0,
            relevance_score=0.0,
            cross_validation_score=0.0
        )
        assert min_cred.overall_score == 0.0

        # All maximum values
        max_cred = SourceCredibility(
            authority_score=1.0,
            recency_score=1.0,
            relevance_score=1.0,
            cross_validation_score=1.0
        )
        assert max_cred.overall_score == 1.0

    def test_source_credibility_validation(self):
        """Test source credibility validation."""
        # Valid credibility scores
        valid_cred = SourceCredibility(
            authority_score=0.8,
            recency_score=0.9,
            relevance_score=0.7,
            cross_validation_score=0.6
        )
        assert valid_cred.authority_score == 0.8

        # Invalid scores (outside 0-1 range)
        with pytest.raises(ValueError):
            SourceCredibility(
                authority_score=-0.1,
                recency_score=0.5,
                relevance_score=0.5,
                cross_validation_score=0.5
            )

        with pytest.raises(ValueError):
            SourceCredibility(
                authority_score=0.5,
                recency_score=1.1,
                relevance_score=0.5,
                cross_validation_score=0.5
            )

    def test_source_credibility_comparison_methods(self):
        """Test source credibility comparison methods."""
        high_cred = SourceCredibility(0.9, 0.8, 0.9, 0.8)
        medium_cred = SourceCredibility(0.6, 0.7, 0.6, 0.5)
        low_cred = SourceCredibility(0.3, 0.2, 0.4, 0.1)

        assert high_cred.is_highly_credible()
        assert not medium_cred.is_highly_credible()
        assert not low_cred.is_highly_credible()

        assert not high_cred.is_low_credibility()
        assert not medium_cred.is_low_credibility()
        assert low_cred.is_low_credibility()

    def test_source_credibility_immutability(self):
        """Test source credibility immutability."""
        cred = SourceCredibility(0.8, 0.7, 0.9, 0.6)

        with pytest.raises(AttributeError):
            cred.authority_score = 0.9

        with pytest.raises(AttributeError):
            cred.overall_score = 0.5


class TestPredictionResultComprehensive:
    """Comprehensive tests for PredictionResult value object."""

    def test_prediction_result_binary_validation(self):
        """Test binary prediction result validation."""
        # Valid binary predictions
        valid_binary_1 = PredictionResult(
            value=0.0,
            prediction_type=PredictionType.BINARY
        )
        assert valid_binary_1.validate()

        valid_binary_2 = PredictionResult(
            value=1.0,
            prediction_type=PredictionType.BINARY
        )
        assert valid_binary_2.validate()

        valid_binary_3 = PredictionResult(
            value=0.75,
            prediction_type=PredictionType.BINARY
        )
        assert valid_binary_3.validate()

        # Invalid binary predictions
        invalid_binary_1 = PredictionResult(
            value=-0.1,
            prediction_type=PredictionType.BINARY
        )
        assert not invalid_binary_1.validate()

        invalid_binary_2 = PredictionResult(
            value=1.1,
            prediction_type=PredictionType.BINARY
        )
        assert not invalid_binary_2.validate()

    def test_prediction_result_numeric_validation(self):
        """Test numeric prediction result validation."""
        # Valid numeric predictions
        valid_numeric_1 = PredictionResult(
            value=42.5,
            prediction_type=PredictionType.NUMERIC
        )
        assert valid_numeric_1.validate()

        valid_numeric_2 = PredictionResult(
            value=-100.0,
            prediction_type=PredictionType.NUMERIC
        )
        assert valid_numeric_2.validate()

        # Numeric with bounds
        valid_bounded = PredictionResult(
            value=50.0,
            prediction_type=PredictionType.NUMERIC,
            bounds=(0.0, 100.0)
        )
        assert valid_bounded.validate()

        # Invalid bounded numeric
        invalid_bounded = PredictionResult(
            value=150.0,
            prediction_type=PredictionType.NUMERIC,
            bounds=(0.0, 100.0)
        )
        assert not invalid_bounded.validate()

    def test_prediction_result_multiple_choice_validation(self):
        """Test multiple choice prediction result validation."""
        # Valid multiple choice
        valid_mc = PredictionResult(
            value={'A': 0.3, 'B': 0.4, 'C': 0.3},
            prediction_type=PredictionType.MULTIPLE_CHOICE
        )
        assert valid_mc.validate()

        # Invalid multiple choice (doesn't sum to 1.0)
        invalid_mc_sum = PredictionResult(
            value={'A': 0.3, 'B': 0.4, 'C': 0.2},  # Sum = 0.9
            prediction_type=PredictionType.MULTIPLE_CHOICE
        )
        assert not invalid_mc_sum.validate()

        # Invalid multiple choice (negative probability)
        invalid_mc_negative = PredictionResult(
            value={'A': 0.5, 'B': -0.1, 'C': 0.6},
            prediction_type=PredictionType.MULTIPLE_CHOICE
        )
        assert not invalid_mc_negative.validate()

        # Invalid multiple choice (wrong type)
        invalid_mc_type = PredictionResult(
            value=0.5,  # Should be dict for multiple choice
            prediction_type=PredictionType.MULTIPLE_CHOICE
        )
        assert not invalid_mc_type.validate()

    def test_prediction_result_bounds_handling(self):
        """Test prediction result bounds handling."""
        # Bounds with numeric prediction
        bounded_result = PredictionResult(
            value=75.0,
            prediction_type=PredictionType.NUMERIC,
            bounds=(0.0, 100.0)
        )
        assert bounded_result.validate()
        assert bounded_result.is_within_bounds()

        # Value outside bounds
        out_of_bounds = PredictionResult(
            value=150.0,
            prediction_type=PredictionType.NUMERIC,
            bounds=(0.0, 100.0)
        )
        assert not out_of_bounds.validate()
        assert not out_of_bounds.is_within_bounds()

        # Bounds with binary prediction (should be ignored)
        binary_with_bounds = PredictionResult(
            value=0.8,
            prediction_type=PredictionType.BINARY,
            bounds=(0.0, 100.0)  # Bounds ignored for binary
        )
        assert binary_with_bounds.validate()

    def test_prediction_result_conversion_methods(self):
        """Test prediction result conversion methods."""
        # Binary to probability
        binary_result = PredictionResult(0.75, PredictionType.BINARY)
        assert binary_result.to_probability() == 0.75

        # Numeric to value
        numeric_result = PredictionResult(42.5, PredictionType.NUMERIC)
        assert numeric_result.to_numeric_value() == 42.5

        # Multiple choice to probabilities
        mc_result = PredictionResult(
            {'A': 0.3, 'B': 0.7},
            PredictionType.MULTIPLE_CHOICE
        )
        probs = mc_result.to_choice_probabilities()
        assert probs['A'] == 0.3
        assert probs['B'] == 0.7

        # Invalid conversions
        with pytest.raises(ValueError):
            numeric_result.to_probability()  # Numeric can't be converted to probability

        with pytest.raises(ValueError):
            binary_result.to_choice_probabilities()  # Binary can't be converted to choices


class TestConsensusMetricsComprehensive:
    """Comprehensive tests for ConsensusMetrics value object."""

    def test_consensus_metrics_calculation(self):
        """Test consensus metrics calculation."""
        # High consensus scenario
        high_consensus = ConsensusMetrics(
            consensus_strength=0.95,
            prediction_variance=0.02,
            agent_diversity_score=0.3,
            confidence_alignment=0.9
        )

        assert high_consensus.consensus_strength == 0.95
        assert high_consensus.prediction_variance == 0.02
        assert high_consensus.has_strong_consensus()
        assert not high_consensus.has_high_disagreement()

        # Low consensus scenario
        low_consensus = ConsensusMetrics(
            consensus_strength=0.3,
            prediction_variance=0.25,
            agent_diversity_score=0.8,
            confidence_alignment=0.4
        )

        assert not low_consensus.has_strong_consensus()
        assert low_consensus.has_high_disagreement()

    def test_consensus_metrics_validation(self):
        """Test consensus metrics validation."""
        # Valid metrics
        valid_metrics = ConsensusMetrics(
            consensus_strength=0.8,
            prediction_variance=0.1,
            agent_diversity_score=0.6,
            confidence_alignment=0.7
        )
        assert valid_metrics.consensus_strength == 0.8

        # Invalid metrics (outside valid ranges)
        with pytest.raises(ValueError):
            ConsensusMetrics(
                consensus_strength=1.5,  # > 1.0
                prediction_variance=0.1,
                agent_diversity_score=0.6,
                confidence_alignment=0.7
            )

        with pytest.raises(ValueError):
            ConsensusMetrics(
                consensus_strength=0.8,
                prediction_variance=-0.1,  # < 0.0
                agent_diversity_score=0.6,
                confidence_alignment=0.7
            )

    def test_consensus_metrics_quality_assessment(self):
        """Test consensus metrics quality assessment."""
        # High quality consensus (high consensus, low variance, good alignment)
        high_quality = ConsensusMetrics(
            consensus_strength=0.9,
            prediction_variance=0.05,
            agent_diversity_score=0.4,
            confidence_alignment=0.85
        )

        quality_score = high_quality.get_quality_score()
        assert quality_score > 0.8

        # Low quality consensus (low consensus, high variance, poor alignment)
        low_quality = ConsensusMetrics(
            consensus_strength=0.4,
            prediction_variance=0.3,
            agent_diversity_score=0.9,
            confidence_alignment=0.3
        )

        quality_score = low_quality.get_quality_score()
        assert quality_score < 0.5

    def test_consensus_metrics_interpretation(self):
        """Test consensus metrics interpretation methods."""
        # Balanced scenario
        balanced = ConsensusMetrics(
            consensus_strength=0.7,
            prediction_variance=0.15,
            agent_diversity_score=0.6,
            confidence_alignment=0.65
        )

        interpretation = balanced.get_interpretation()
        assert 'moderate consensus' in interpretation.lower()
        assert 'balanced' in interpretation.lower()

        # Extreme disagreement scenario
        extreme_disagreement = ConsensusMetrics(
            consensus_strength=0.2,
            prediction_variance=0.4,
            agent_diversity_score=0.95,
            confidence_alignment=0.1
        )

        interpretation = extreme_disagreement.get_interpretation()
        assert 'high disagreement' in interpretation.lower()
        assert 'diverse' in interpretation.lower()

    def test_consensus_metrics_immutability(self):
        """Test consensus metrics immutability."""
        metrics = ConsensusMetrics(0.8, 0.1, 0.6, 0.7)

        with pytest.raises(AttributeError):
            metrics.consensus_strength = 0.9

        with pytest.raises(AttributeError):
            metrics.prediction_variance = 0.2
