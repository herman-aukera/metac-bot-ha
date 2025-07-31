"""Unit tests for Prediction entity."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.domain.entities.prediction import Prediction
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.reasoning_step import ReasoningStep


class TestPrediction:
    """Test cases for Prediction entity."""

    def test_prediction_initialization_valid(self):
        """Test valid prediction initialization."""
        confidence = Confidence(level=0.8, basis="Strong evidence")
        reasoning_step = ReasoningStep.create(
            step_number=1,
            description="Analysis step",
            input_data={"question": "test"},
            output_data={"result": "analysis"},
            confidence_level=0.8,
            confidence_basis="Analysis confidence"
        )

        prediction = Prediction(
            id=uuid4(),
            question_id=1,
            result=0.7,
            confidence=confidence,
            method="chain_of_thought",
            reasoning="Based on historical data",
            created_by="agent_1",
            timestamp=datetime.utcnow(),
            metadata={"test": "data"},
            reasoning_steps=[reasoning_step],
            evidence_sources=["source1", "source2"]
        )

        assert prediction.question_id == 1
        assert prediction.result == 0.7
        assert prediction.confidence == confidence
        assert prediction.method == "chain_of_thought"
        assert prediction.reasoning == "Based on historical data"
        assert prediction.created_by == "agent_1"
        assert len(prediction.reasoning_steps) == 1
        assert len(prediction.evidence_sources) == 2

    def test_prediction_validation_invalid_question_id(self):
        """Test prediction validation with invalid question ID."""
        confidence = Confidence(level=0.8, basis="Strong evidence")

        with pytest.raises(ValueError, match="Question ID must be positive"):
            Prediction(
                id=uuid4(),
                question_id=0,
                result=0.7,
                confidence=confidence,
                method="chain_of_thought",
                reasoning="Based on historical data",
                created_by="agent_1",
                timestamp=datetime.utcnow(),
                metadata={},
                reasoning_steps=[],
                evidence_sources=[]
            )

    def test_prediction_validation_empty_created_by(self):
        """Test prediction validation with empty created_by."""
        confidence = Confidence(level=0.8, basis="Strong evidence")

        with pytest.raises(ValueError, match="Created by cannot be empty"):
            Prediction(
                id=uuid4(),
                question_id=1,
                result=0.7,
                confidence=confidence,
                method="chain_of_thought",
                reasoning="Based on historical data",
                created_by="",
                timestamp=datetime.utcnow(),
                metadata={},
                reasoning_steps=[],
                evidence_sources=[]
            )

    def test_prediction_validation_empty_method(self):
        """Test prediction validation with empty method."""
        confidence = Confidence(level=0.8, basis="Strong evidence")

        with pytest.raises(ValueError, match="Method cannot be empty"):
            Prediction(
                id=uuid4(),
                question_id=1,
                result=0.7,
                confidence=confidence,
                method="",
                reasoning="Based on historical data",
                created_by="agent_1",
                timestamp=datetime.utcnow(),
                metadata={},
                reasoning_steps=[],
                evidence_sources=[]
            )

    def test_prediction_validation_empty_reasoning(self):
        """Test prediction validation with empty reasoning."""
        confidence = Confidence(level=0.8, basis="Strong evidence")

        with pytest.raises(ValueError, match="Reasoning cannot be empty"):
            Prediction(
                id=uuid4(),
                question_id=1,
                result=0.7,
                confidence=confidence,
                method="chain_of_thought",
                reasoning="",
                created_by="agent_1",
                timestamp=datetime.utcnow(),
                metadata={},
                reasoning_steps=[],
                evidence_sources=[]
            )

    def test_prediction_validation_invalid_result_type(self):
        """Test prediction validation with invalid result type."""
        confidence = Confidence(level=0.8, basis="Strong evidence")

        with pytest.raises(ValueError, match="Invalid prediction result type"):
            Prediction(
                id=uuid4(),
                question_id=1,
                result="invalid",
                confidence=confidence,
                method="chain_of_thought",
                reasoning="Based on historical data",
                created_by="agent_1",
                timestamp=datetime.utcnow(),
                metadata={},
                reasoning_steps=[],
                evidence_sources=[]
            )

    def test_create_binary_prediction(self):
        """Test creating binary prediction."""
        prediction = Prediction.create_binary(
            question_id=1,
            probability=0.75,
            confidence_level=0.8,
            confidence_basis="Strong evidence",
            method="chain_of_thought",
            reasoning="Based on analysis",
            created_by="agent_1"
        )

        assert prediction.question_id == 1
        assert prediction.result == 0.75
        assert prediction.confidence.level == 0.8
        assert prediction.is_binary_prediction()
        assert not prediction.is_numeric_prediction()
        assert not prediction.is_multiple_choice_prediction()

    def test_create_binary_prediction_invalid_probability(self):
        """Test creating binary prediction with invalid probability."""
        with pytest.raises(ValueError, match="Binary probability must be between 0.0 and 1.0"):
            Prediction.create_binary(
                question_id=1,
                probability=1.5,
                confidence_level=0.8,
                confidence_basis="Strong evidence",
                method="chain_of_thought",
                reasoning="Based on analysis",
                created_by="agent_1"
            )

    def test_create_numeric_prediction(self):
        """Test creating numeric prediction."""
        prediction = Prediction.create_numeric(
            question_id=1,
            value=42.5,
            confidence_level=0.7,
            confidence_basis="Moderate evidence",
            method="tree_of_thought",
            reasoning="Based on calculation",
            created_by="agent_2"
        )

        assert prediction.question_id == 1
        assert prediction.result == 42.5
        assert prediction.confidence.level == 0.7
        assert not prediction.is_binary_prediction()
        assert prediction.is_numeric_prediction()
        assert not prediction.is_multiple_choice_prediction()

    def test_create_multiple_choice_prediction(self):
        """Test creating multiple choice prediction."""
        choice_probs = {"A": 0.4, "B": 0.6}
        prediction = Prediction.create_multiple_choice(
            question_id=1,
            choice_probabilities=choice_probs,
            confidence_level=0.9,
            confidence_basis="Very strong evidence",
            method="ensemble",
            reasoning="Based on multiple sources",
            created_by="agent_3"
        )

        assert prediction.question_id == 1
        assert prediction.result == choice_probs
        assert prediction.confidence.level == 0.9
        assert not prediction.is_binary_prediction()
        assert not prediction.is_numeric_prediction()
        assert prediction.is_multiple_choice_prediction()

    def test_create_multiple_choice_prediction_invalid_probabilities(self):
        """Test creating multiple choice prediction with invalid probabilities."""
        choice_probs = {"A": 0.3, "B": 0.5}  # Sum = 0.8, not 1.0

        with pytest.raises(ValueError, match="Choice probabilities must sum to 1.0"):
            Prediction.create_multiple_choice(
                question_id=1,
                choice_probabilities=choice_probs,
                confidence_level=0.9,
                confidence_basis="Very strong evidence",
                method="ensemble",
                reasoning="Based on multiple sources",
                created_by="agent_3"
            )

    def test_get_binary_probability(self):
        """Test getting binary probability."""
        prediction = Prediction.create_binary(
            question_id=1,
            probability=0.75,
            confidence_level=0.8,
            confidence_basis="Strong evidence",
            method="chain_of_thought",
            reasoning="Based on analysis",
            created_by="agent_1"
        )

        assert prediction.get_binary_probability() == 0.75

    def test_get_binary_probability_invalid(self):
        """Test getting binary probability from non-binary prediction."""
        prediction = Prediction.create_numeric(
            question_id=1,
            value=42.5,
            confidence_level=0.7,
            confidence_basis="Moderate evidence",
            method="tree_of_thought",
            reasoning="Based on calculation",
            created_by="agent_2"
        )

        with pytest.raises(ValueError, match="Not a binary prediction"):
            prediction.get_binary_probability()

    def test_get_numeric_value(self):
        """Test getting numeric value."""
        prediction = Prediction.create_numeric(
            question_id=1,
            value=42.5,
            confidence_level=0.7,
            confidence_basis="Moderate evidence",
            method="tree_of_thought",
            reasoning="Based on calculation",
            created_by="agent_2"
        )

        assert prediction.get_numeric_value() == 42.5

    def test_get_numeric_value_invalid(self):
        """Test getting numeric value from non-numeric prediction."""
        prediction = Prediction.create_binary(
            question_id=1,
            probability=0.75,
            confidence_level=0.8,
            confidence_basis="Strong evidence",
            method="chain_of_thought",
            reasoning="Based on analysis",
            created_by="agent_1"
        )

        with pytest.raises(ValueError, match="Not a numeric prediction"):
            prediction.get_numeric_value()

    def test_get_choice_probabilities(self):
        """Test getting choice probabilities."""
        choice_probs = {"A": 0.4, "B": 0.6}
        prediction = Prediction.create_multiple_choice(
            question_id=1,
            choice_probabilities=choice_probs,
            confidence_level=0.9,
            confidence_basis="Very strong evidence",
            method="ensemble",
            reasoning="Based on multiple sources",
            created_by="agent_3"
        )

        assert prediction.get_choice_probabilities() == choice_probs

    def test_get_choice_probabilities_invalid(self):
        """Test getting choice probabilities from non-multiple-choice prediction."""
        prediction = Prediction.create_binary(
            question_id=1,
            probability=0.75,
            confidence_level=0.8,
            confidence_basis="Strong evidence",
            method="chain_of_thought",
            reasoning="Based on analysis",
            created_by="agent_1"
        )

        with pytest.raises(ValueError, match="Not a multiple choice prediction"):
            prediction.get_choice_probabilities()

    def test_get_most_likely_choice(self):
        """Test getting most likely choice."""
        choice_probs = {"A": 0.3, "B": 0.7}
        prediction = Prediction.create_multiple_choice(
            question_id=1,
            choice_probabilities=choice_probs,
            confidence_level=0.9,
            confidence_basis="Very strong evidence",
            method="ensemble",
            reasoning="Based on multiple sources",
            created_by="agent_3"
        )

        assert prediction.get_most_likely_choice() == "B"

    def test_has_high_confidence(self):
        """Test checking high confidence."""
        high_conf_prediction = Prediction.create_binary(
            question_id=1,
            probability=0.75,
            confidence_level=0.9,
            confidence_basis="Very strong evidence",
            method="chain_of_thought",
            reasoning="Based on analysis",
            created_by="agent_1"
        )

        low_conf_prediction = Prediction.create_binary(
            question_id=1,
            probability=0.75,
            confidence_level=0.3,
            confidence_basis="Weak evidence",
            method="chain_of_thought",
            reasoning="Based on analysis",
            created_by="agent_1"
        )

        assert high_conf_prediction.has_high_confidence()
        assert not low_conf_prediction.has_high_confidence()

    def test_get_reasoning_summary(self):
        """Test getting reasoning summary."""
        reasoning_step = ReasoningStep.create(
            step_number=1,
            description="Analysis step",
            input_data={"question": "test"},
            output_data={"result": "analysis"},
            confidence_level=0.8,
            confidence_basis="Analysis confidence"
        )

        prediction = Prediction.create_binary(
            question_id=1,
            probability=0.75,
            confidence_level=0.8,
            confidence_basis="Strong evidence",
            method="chain_of_thought",
            reasoning="Based on analysis",
            created_by="agent_1",
            reasoning_steps=[reasoning_step]
        )

        summary = prediction.get_reasoning_summary()
        assert "Step 1: Analysis step" in summary

    def test_to_submission_format_binary(self):
        """Test converting binary prediction to submission format."""
        prediction = Prediction.create_binary(
            question_id=1,
            probability=0.75,
            confidence_level=0.8,
            confidence_basis="Strong evidence",
            method="chain_of_thought",
            reasoning="Based on analysis",
            created_by="agent_1"
        )

        submission = prediction.to_submission_format()
        assert submission["prediction"] == 0.75
        assert submission["confidence"] == 0.8
        assert submission["reasoning"] == "Based on analysis"

    def test_to_submission_format_numeric(self):
        """Test converting numeric prediction to submission format."""
        prediction = Prediction.create_numeric(
            question_id=1,
            value=42.5,
            confidence_level=0.7,
            confidence_basis="Moderate evidence",
            method="tree_of_thought",
            reasoning="Based on calculation",
            created_by="agent_2"
        )

        submission = prediction.to_submission_format()
        assert submission["prediction"] == 42.5
        assert submission["confidence"] == 0.7
        assert submission["reasoning"] == "Based on calculation"

    def test_to_submission_format_multiple_choice(self):
        """Test converting multiple choice prediction to submission format."""
        choice_probs = {"A": 0.4, "B": 0.6}
        prediction = Prediction.create_multiple_choice(
            question_id=1,
            choice_probabilities=choice_probs,
            confidence_level=0.9,
            confidence_basis="Very strong evidence",
            method="ensemble",
            reasoning="Based on multiple sources",
            created_by="agent_3"
        )

        submission = prediction.to_submission_format()
        assert submission["prediction"] == choice_probs
        assert submission["confidence"] == 0.9
        assert submission["reasoning"] == "Based on multiple sources"

    def test_to_summary(self):
        """Test creating prediction summary."""
        prediction = Prediction.create_binary(
            question_id=1,
            probability=0.75,
            confidence_level=0.8,
            confidence_basis="Strong evidence",
            method="chain_of_thought",
            reasoning="Based on analysis",
            created_by="agent_1"
        )

        summary = prediction.to_summary()
        assert "Q1:" in summary
        assert "0.750" in summary
        assert "high" in summary
        assert "agent_1" in summary
