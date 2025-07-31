"""Unit tests for domain value objects."""

import pytest
from datetime import datetime, timedelta
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.reasoning_step import ReasoningStep
from src.domain.value_objects.strategy_result import StrategyResult, StrategyType, StrategyOutcome


class TestConfidence:
    """Test cases for Confidence value object."""

    def test_valid_confidence_creation(self):
        """Test creating valid confidence objects."""
        confidence = Confidence(level=0.8, basis="Strong evidence from multiple sources")
        assert confidence.level == 0.8
        assert confidence.basis == "Strong evidence from multiple sources"

    def test_confidence_level_validation(self):
        """Test confidence level validation."""
        # Valid levels
        Confidence(level=0.0, basis="No confidence")
        Confidence(level=1.0, basis="Complete confidence")
        Confidence(level=0.5, basis="Medium confidence")

        # Invalid levels
        with pytest.raises(ValueError, match="Confidence level must be between 0.0 and 1.0"):
            Confidence(level=-0.1, basis="Invalid")

        with pytest.raises(ValueError, match="Confidence level must be between 0.0 and 1.0"):
            Confidence(level=1.1, basis="Invalid")

    def test_confidence_basis_validation(self):
        """Test confidence basis validation."""
        with pytest.raises(ValueError, match="Confidence basis cannot be empty"):
            Confidence(level=0.5, basis="")

        with pytest.raises(ValueError, match="Confidence basis cannot be empty"):
            Confidence(level=0.5, basis="   ")

    def test_confidence_factory_methods(self):
        """Test confidence factory methods."""
        high = Confidence.high("Strong evidence")
        assert high.level == 0.9
        assert high.is_high()

        medium = Confidence.medium("Some evidence")
        assert medium.level == 0.6
        assert medium.is_medium()

        low = Confidence.low("Weak evidence")
        assert low.level == 0.3
        assert low.is_low()

    def test_confidence_level_checks(self):
        """Test confidence level checking methods."""
        high = Confidence(level=0.85, basis="High")
        assert high.is_high()
        assert not high.is_medium()
        assert not high.is_low()

        medium = Confidence(level=0.6, basis="Medium")
        assert not medium.is_high()
        assert medium.is_medium()
        assert not medium.is_low()

        low = Confidence(level=0.3, basis="Low")
        assert not low.is_high()
        assert not low.is_medium()
        assert low.is_low()

    def test_confidence_combination(self):
        """Test combining confidence levels."""
        conf1 = Confidence(level=0.8, basis="Source 1")
        conf2 = Confidence(level=0.6, basis="Source 2")

        combined = conf1.combine_with(conf2, weight=0.7)
        expected_level = 0.8 * 0.7 + 0.6 * 0.3
        assert abs(combined.level - expected_level) < 0.001
        assert "Combined:" in combined.basis

        # Test invalid weight
        with pytest.raises(ValueError, match="Weight must be between 0.0 and 1.0"):
            conf1.combine_with(conf2, weight=1.5)


class TestReasoningStep:
    """Test cases for ReasoningStep value object."""

    def test_valid_reasoning_step_creation(self):
        """Test creating valid reasoning steps."""
        confidence = Confidence(level=0.8, basis="Strong reasoning")
        timestamp = datetime.utcnow()

        step = ReasoningStep(
            step_number=1,
            description="Analyze the problem",
            input_data={"question": "What is 2+2?"},
            output_data={"answer": 4},
            confidence=confidence,
            timestamp=timestamp,
            reasoning_type="deduction"
        )

        assert step.step_number == 1
        assert step.description == "Analyze the problem"
        assert step.input_data == {"question": "What is 2+2?"}
        assert step.output_data == {"answer": 4}
        assert step.confidence == confidence
        assert step.timestamp == timestamp
        assert step.reasoning_type == "deduction"

    def test_reasoning_step_validation(self):
        """Test reasoning step validation."""
        confidence = Confidence(level=0.8, basis="Test")
        timestamp = datetime.utcnow()

        # Invalid step number
        with pytest.raises(ValueError, match="Step number must be positive"):
            ReasoningStep(
                step_number=0,
                description="Test",
                input_data={},
                output_data={},
                confidence=confidence,
                timestamp=timestamp
            )

        # Empty description
        with pytest.raises(ValueError, match="Description cannot be empty"):
            ReasoningStep(
                step_number=1,
                description="",
                input_data={},
                output_data={},
                confidence=confidence,
                timestamp=timestamp
            )

        # Invalid input data type
        with pytest.raises(ValueError, match="Input data must be a dictionary"):
            ReasoningStep(
                step_number=1,
                description="Test",
                input_data="not a dict",
                output_data={},
                confidence=confidence,
                timestamp=timestamp
            )

    def test_reasoning_step_factory_method(self):
        """Test reasoning step factory method."""
        step = ReasoningStep.create(
            step_number=1,
            description="Test step",
            input_data={"input": "test"},
            output_data={"output": "result"},
            confidence_level=0.9,
            confidence_basis="Strong reasoning",
            reasoning_type="analytical"
        )

        assert step.step_number == 1
        assert step.description == "Test step"
        assert step.confidence.level == 0.9
        assert step.reasoning_type == "analytical"
        assert isinstance(step.timestamp, datetime)

    def test_reasoning_step_methods(self):
        """Test reasoning step utility methods."""
        high_confidence_step = ReasoningStep.create(
            step_number=1,
            description="High confidence step",
            input_data={},
            output_data={"result": "success", "confidence": 0.9},
            confidence_level=0.9,
            confidence_basis="Strong evidence"
        )

        assert high_confidence_step.has_high_confidence()

        key_outputs = high_confidence_step.get_key_outputs()
        assert "result" in key_outputs
        assert key_outputs["result"] == "success"

        summary = high_confidence_step.to_summary()
        assert "Step 1" in summary
        assert "high" in summary


class TestStrategyResult:
    """Test cases for StrategyResult value object."""

    def test_valid_strategy_result_creation(self):
        """Test creating valid strategy results."""
        confidence = Confidence(level=0.8, basis="Good strategy")
        timestamp = datetime.utcnow()

        result = StrategyResult(
            strategy_type=StrategyType.AGGRESSIVE,
            outcome=StrategyOutcome.SUCCESS,
            confidence=confidence,
            expected_score=0.75,
            reasoning="Market inefficiency detected",
            metadata={"risk_level": "high"},
            timestamp=timestamp,
            question_ids=[1, 2, 3],
            actual_score=0.82
        )

        assert result.strategy_type == StrategyType.AGGRESSIVE
        assert result.outcome == StrategyOutcome.SUCCESS
        assert result.expected_score == 0.75
        assert result.actual_score == 0.82
        assert result.question_ids == [1, 2, 3]

    def test_strategy_result_validation(self):
        """Test strategy result validation."""
        confidence = Confidence(level=0.8, basis="Test")
        timestamp = datetime.utcnow()

        # Empty reasoning
        with pytest.raises(ValueError, match="Strategy reasoning cannot be empty"):
            StrategyResult(
                strategy_type=StrategyType.BALANCED,
                outcome=StrategyOutcome.PENDING,
                confidence=confidence,
                expected_score=0.5,
                reasoning="",
                metadata={},
                timestamp=timestamp,
                question_ids=[1]
            )

        # Invalid question IDs
        with pytest.raises(ValueError, match="All question IDs must be positive integers"):
            StrategyResult(
                strategy_type=StrategyType.BALANCED,
                outcome=StrategyOutcome.PENDING,
                confidence=confidence,
                expected_score=0.5,
                reasoning="Test",
                metadata={},
                timestamp=timestamp,
                question_ids=[0, -1]
            )

    def test_strategy_result_factory_method(self):
        """Test strategy result factory method."""
        result = StrategyResult.create(
            strategy_type=StrategyType.CONSERVATIVE,
            expected_score=0.6,
            reasoning="Safe approach",
            question_ids=[1, 2],
            confidence_level=0.7,
            confidence_basis="Historical performance"
        )

        assert result.strategy_type == StrategyType.CONSERVATIVE
        assert result.outcome == StrategyOutcome.PENDING
        assert result.expected_score == 0.6
        assert result.confidence.level == 0.7
        assert isinstance(result.timestamp, datetime)

    def test_strategy_result_status_methods(self):
        """Test strategy result status methods."""
        result = StrategyResult.create(
            strategy_type=StrategyType.BALANCED,
            expected_score=0.5,
            reasoning="Test",
            question_ids=[1],
            confidence_level=0.6,
            confidence_basis="Test"
        )

        assert result.is_pending()
        assert not result.is_successful()

        success_result = result.mark_success(0.7)
        assert success_result.is_successful()
        assert not success_result.is_pending()
        assert success_result.actual_score == 0.7

        failure_result = result.mark_failure(0.3)
        assert not failure_result.is_successful()
        assert failure_result.outcome == StrategyOutcome.FAILURE

    def test_strategy_result_score_difference(self):
        """Test score difference calculation."""
        result = StrategyResult.create(
            strategy_type=StrategyType.BALANCED,
            expected_score=0.6,
            reasoning="Test",
            question_ids=[1],
            confidence_level=0.7,
            confidence_basis="Test"
        )

        # No actual score yet
        assert result.get_score_difference() is None

        # With actual score
        completed_result = result.mark_success(0.8)
        assert abs(completed_result.get_score_difference() - 0.2) < 0.001

    def test_strategy_result_summary(self):
        """Test strategy result summary."""
        result = StrategyResult.create(
            strategy_type=StrategyType.AGGRESSIVE,
            expected_score=0.75,
            reasoning="High risk, high reward",
            question_ids=[1, 2],
            confidence_level=0.8,
            confidence_basis="Strong signals"
        )

        summary = result.to_summary()
        assert "aggressive" in summary
        assert "pending" in summary
        assert "0.750" in summary
