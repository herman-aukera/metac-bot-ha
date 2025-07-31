"""Property-based tests for domain model invariants and edge cases."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
from datetime import datetime, timedelta
from uuid import uuid4

from src.domain.entities.question import Question, QuestionType, QuestionCategory, QuestionStatus
from src.domain.entities.prediction import Prediction
from src.domain.entities.tournament import Tournament, ScoringRules, ScoringMethod
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.reasoning_step import ReasoningStep
from src.domain.value_objects.strategy_result import StrategyResult, StrategyType
from src.domain.value_objects.source_credibility import SourceCredibility
from src.domain.value_objects.prediction_result import PredictionResult, PredictionType


# Hypothesis strategies for generating test data
confidence_level_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
positive_int_strategy = st.integers(min_value=1, max_value=1000000)
non_empty_text_strategy = st.text(min_size=1, max_size=1000).filter(lambda x: x.strip())
probability_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


class TestConfidenceProperties:
    """Property-based tests for Confidence value object."""

    @given(
        level=confidence_level_strategy,
        basis=non_empty_text_strategy
    )
    def test_confidence_level_invariants(self, level, basis):
        """Test that confidence level invariants always hold."""
        confidence = Confidence(level=level, basis=basis)

        # Invariant: level should always be between 0.0 and 1.0
        assert 0.0 <= confidence.level <= 1.0

        # Invariant: basis should never be empty
        assert confidence.basis.strip() != ""

        # Invariant: classification should be consistent
        if confidence.level >= 0.8:
            assert confidence.is_high()
            assert not confidence.is_medium()
            assert not confidence.is_low()
        elif confidence.level >= 0.5:
            assert not confidence.is_high()
            assert confidence.is_medium()
            assert not confidence.is_low()
        else:
            assert not confidence.is_high()
            assert not confidence.is_medium()
            assert confidence.is_low()

    @given(
        level1=confidence_level_strategy,
        level2=confidence_level_strategy,
        weight=confidence_level_strategy,
        basis1=non_empty_text_strategy,
        basis2=non_empty_text_strategy
    )
    def test_confidence_combination_properties(self, level1, level2, weight, basis1, basis2):
        """Test properties of confidence combination."""
        conf1 = Confidence(level=level1, basis=basis1)
        conf2 = Confidence(level=level2, basis=basis2)

        combined = conf1.combine_with(conf2, weight=weight)

        # Invariant: combined level should be weighted average
        expected_level = level1 * weight + level2 * (1 - weight)
        assert abs(combined.level - expected_level) < 0.0001

        # Invariant: combined level should be within valid range
        assert 0.0 <= combined.level <= 1.0

        # Invariant: combined basis should contain both original bases
        assert "Combined:" in combined.basis

        # Invariant: if weights are extreme, result should be close to one input
        if weight > 0.99:
            assert abs(combined.level - level1) < 0.01
        elif weight < 0.01:
            assert abs(combined.level - level2) < 0.01

    @given(confidence_level_strategy)
    def test_confidence_factory_methods_consistency(self, level):
        """Test that factory methods produce consistent results."""
        # Test that factory methods create valid confidence objects
        high_conf = Confidence.high("Test high")
        medium_conf = Confidence.medium("Test medium")
        low_conf = Confidence.low("Test low")

        assert high_conf.is_high()
        assert medium_conf.is_medium()
        assert low_conf.is_low()

        # Test ordering
        assert high_conf.level > medium_conf.level > low_conf.level


class TestQuestionProperties:
    """Property-based tests for Question entity."""

    @given(
        question_id=positive_int_strategy,
        text=non_empty_text_strategy,
        background=non_empty_text_strategy,
        resolution_criteria=non_empty_text_strategy,
        scoring_weight=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False),
        deadline_hours=st.integers(min_value=1, max_value=8760)  # 1 hour to 1 year
    )
    def test_question_basic_invariants(self, question_id, text, background, resolution_criteria, scoring_weight, deadline_hours):
        """Test basic question invariants."""
        deadline = datetime.utcnow() + timedelta(hours=deadline_hours)

        question = Question(
            id=question_id,
            text=text,
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=deadline,
            background=background,
            resolution_criteria=resolution_criteria,
            scoring_weight=scoring_weight,
            status=QuestionStatus.ACTIVE
        )

        # Invariant: ID should be positive
        assert question.id > 0

        # Invariant: text should not be empty
        assert question.text.strip() != ""

        # Invariant: deadline should be in the future
        assert question.deadline > datetime.utcnow()

        # Invariant: scoring weight should be positive
        assert question.scoring_weight > 0

        # Invariant: time until deadline should be positive
        assert question.time_until_deadline() > 0

    @given(
        min_val=st.floats(min_value=-1000.0, max_value=999.0, allow_nan=False),
        max_val=st.floats(min_value=-999.0, max_value=1000.0, allow_nan=False)
    )
    def test_numeric_question_bounds_invariants(self, min_val, max_val):
        """Test numeric question bounds invariants."""
        assume(min_val < max_val)  # Only test valid ranges

        question = Question(
            id=1,
            text="Numeric question",
            question_type=QuestionType.NUMERIC,
            category=QuestionCategory.SCIENCE,
            deadline=datetime.utcnow() + timedelta(days=30),
            background="Background",
            resolution_criteria="Criteria",
            min_value=min_val,
            max_value=max_val
        )

        # Invariant: min_value should be less than max_value
        assert question.min_value < question.max_value

        # Invariant: complexity should increase with range size
        range_size = max_val - min_val
        complexity = question.get_complexity_score()

        # Larger ranges should generally have higher complexity
        if range_size > 1000:
            assert complexity > 1.5

    @given(
        choices_count=st.integers(min_value=2, max_value=10)
    )
    def test_multiple_choice_question_invariants(self, choices_count):
        """Test multiple choice question invariants."""
        choices = [f"Choice_{i}" for i in range(choices_count)]

        question = Question(
            id=1,
            text="Multiple choice question",
            question_type=QuestionType.MULTIPLE_CHOICE,
            category=QuestionCategory.OTHER,
            deadline=datetime.utcnow() + timedelta(days=30),
            background="Background",
            resolution_criteria="Criteria",
            choices=choices
        )

        # Invariant: should have at least 2 choices
        assert len(question.choices) >= 2

        # Invariant: all choices should be non-empty strings
        for choice in question.choices:
            assert isinstance(choice, str)
            assert choice.strip() != ""

        # Invariant: choices should be unique
        assert len(set(question.choices)) == len(question.choices)


class TestPredictionProperties:
    """Property-based tests for Prediction entity."""

    @given(
        question_id=positive_int_strategy,
        probability=probability_strategy,
        confidence_level=confidence_level_strategy,
        method=non_empty_text_strategy,
        reasoning=non_empty_text_strategy,
        created_by=non_empty_text_strategy
    )
    def test_binary_prediction_invariants(self, question_id, probability, confidence_level, method, reasoning, created_by):
        """Test binary prediction invariants."""
        prediction = Prediction.create_binary(
            question_id=question_id,
            probability=probability,
            confidence_level=confidence_level,
            confidence_basis="Test basis",
            method=method,
            reasoning=reasoning,
            created_by=created_by
        )

        # Invariant: binary prediction should be identified correctly
        assert prediction.is_binary_prediction()
        assert not prediction.is_numeric_prediction()
        assert not prediction.is_multiple_choice_prediction()

        # Invariant: probability should be in valid range
        assert 0.0 <= prediction.get_binary_probability() <= 1.0

        # Invariant: confidence should match input
        assert abs(prediction.confidence.level - confidence_level) < 0.0001

        # Invariant: submission format should be valid
        submission = prediction.to_submission_format()
        assert 'prediction' in submission
        assert 'confidence' in submission
        assert 'reasoning' in submission
        assert 0.0 <= submission['prediction'] <= 1.0

    @given(
        question_id=positive_int_strategy,
        value=st.floats(min_value=-1000000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        confidence_level=confidence_level_strategy
    )
    def test_numeric_prediction_invariants(self, question_id, value, confidence_level):
        """Test numeric prediction invariants."""
        prediction = Prediction.create_numeric(
            question_id=question_id,
            value=value,
            confidence_level=confidence_level,
            confidence_basis="Test basis",
            method="test_method",
            reasoning="Test reasoning",
            created_by="test_agent"
        )

        # Invariant: numeric prediction should be identified correctly
        assert not prediction.is_binary_prediction()
        assert prediction.is_numeric_prediction()
        assert not prediction.is_multiple_choice_prediction()

        # Invariant: value should match input
        assert prediction.get_numeric_value() == value

        # Invariant: confidence should be in valid range
        assert 0.0 <= prediction.confidence.level <= 1.0

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=probability_strategy,
            min_size=2,
            max_size=5
        ).filter(lambda d: abs(sum(d.values()) - 1.0) < 0.0001)  # Probabilities sum to 1.0
    )
    def test_multiple_choice_prediction_invariants(self, choice_probabilities):
        """Test multiple choice prediction invariants."""
        prediction = Prediction.create_multiple_choice(
            question_id=1,
            choice_probabilities=choice_probabilities,
            confidence_level=0.8,
            confidence_basis="Test basis",
            method="test_method",
            reasoning="Test reasoning",
            created_by="test_agent"
        )

        # Invariant: multiple choice prediction should be identified correctly
        assert not prediction.is_binary_prediction()
        assert not prediction.is_numeric_prediction()
        assert prediction.is_multiple_choice_prediction()

        # Invariant: probabilities should sum to 1.0
        probs = prediction.get_choice_probabilities()
        assert abs(sum(probs.values()) - 1.0) < 0.0001

        # Invariant: all probabilities should be non-negative
        for prob in probs.values():
            assert prob >= 0.0

        # Invariant: most likely choice should have highest probability
        most_likely = prediction.get_most_likely_choice()
        max_prob = max(probs.values())
        assert probs[most_likely] == max_prob


class TestSourceCredibilityProperties:
    """Property-based tests for SourceCredibility value object."""

    @given(
        authority=score_strategy,
        recency=score_strategy,
        relevance=score_strategy,
        cross_validation=score_strategy
    )
    def test_source_credibility_invariants(self, authority, recency, relevance, cross_validation):
        """Test source credibility invariants."""
        credibility = SourceCredibility(
            authority_score=authority,
            recency_score=recency,
            relevance_score=relevance,
            cross_validation_score=cross_validation
        )

        # Invariant: all scores should be in valid range
        assert 0.0 <= credibility.authority_score <= 1.0
        assert 0.0 <= credibility.recency_score <= 1.0
        assert 0.0 <= credibility.relevance_score <= 1.0
        assert 0.0 <= credibility.cross_validation_score <= 1.0

        # Invariant: overall score should be average of component scores
        expected_overall = (authority + recency + relevance + cross_validation) / 4
        assert abs(credibility.overall_score - expected_overall) < 0.0001

        # Invariant: overall score should be in valid range
        assert 0.0 <= credibility.overall_score <= 1.0

        # Invariant: classification should be consistent
        if credibility.overall_score >= 0.8:
            assert credibility.is_highly_credible()
            assert not credibility.is_low_credibility()
        elif credibility.overall_score <= 0.3:
            assert not credibility.is_highly_credible()
            assert credibility.is_low_credibility()
        else:
            assert not credibility.is_highly_credible()
            assert not credibility.is_low_credibility()


class TestPredictionResultProperties:
    """Property-based tests for PredictionResult value object."""

    @given(probability=probability_strategy)
    def test_binary_prediction_result_invariants(self, probability):
        """Test binary prediction result invariants."""
        result = PredictionResult(
            value=probability,
            prediction_type=PredictionType.BINARY
        )

        # Invariant: binary results should always validate if probability is in range
        assert result.validate()

        # Invariant: conversion should return original value
        assert result.to_probability() == probability

        # Invariant: bounds should not affect binary predictions
        bounded_result = PredictionResult(
            value=probability,
            prediction_type=PredictionType.BINARY,
            bounds=(0.0, 100.0)  # Should be ignored
        )
        assert bounded_result.validate()

    @given(
        value=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_bound=st.floats(min_value=-1000.0, max_value=999.0, allow_nan=False, allow_infinity=False),
        max_bound=st.floats(min_value=-999.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    def test_numeric_prediction_result_invariants(self, value, min_bound, max_bound):
        """Test numeric prediction result invariants."""
        assume(min_bound < max_bound)  # Only test valid bounds

        result = PredictionResult(
            value=value,
            prediction_type=PredictionType.NUMERIC,
            bounds=(min_bound, max_bound)
        )

        # Invariant: validation should depend on bounds
        if min_bound <= value <= max_bound:
            assert result.validate()
            assert result.is_within_bounds()
        else:
            assert not result.validate()
            assert not result.is_within_bounds()

        # Invariant: conversion should return original value
        assert result.to_numeric_value() == value

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=5),
            values=probability_strategy,
            min_size=2,
            max_size=4
        )
    )
    def test_multiple_choice_prediction_result_invariants(self, choice_dict):
        """Test multiple choice prediction result invariants."""
        # Normalize probabilities to sum to 1.0
        total = sum(choice_dict.values())
        assume(total > 0)  # Avoid division by zero

        normalized_dict = {k: v / total for k, v in choice_dict.items()}

        result = PredictionResult(
            value=normalized_dict,
            prediction_type=PredictionType.MULTIPLE_CHOICE
        )

        # Invariant: should validate since we normalized
        assert result.validate()

        # Invariant: probabilities should sum to 1.0
        probs = result.to_choice_probabilities()
        assert abs(sum(probs.values()) - 1.0) < 0.0001

        # Invariant: all probabilities should be non-negative
        for prob in probs.values():
            assert prob >= 0.0


class QuestionStateMachine(RuleBasedStateMachine):
    """Stateful testing for Question entity lifecycle."""

    def __init__(self):
        super().__init__()
        self.questions = {}
        self.next_id = 1

    @rule(
        text=non_empty_text_strategy,
        question_type=st.sampled_from(list(QuestionType)),
        category=st.sampled_from(list(QuestionCategory)),
        scoring_weight=st.floats(min_value=0.1, max_value=10.0),
        deadline_hours=st.integers(min_value=1, max_value=168)  # 1 hour to 1 week
    )
    def create_question(self, text, question_type, category, scoring_weight, deadline_hours):
        """Create a new question."""
        deadline = datetime.utcnow() + timedelta(hours=deadline_hours)

        kwargs = {
            'id': self.next_id,
            'text': text,
            'question_type': question_type,
            'category': category,
            'deadline': deadline,
            'background': f"Background for question {self.next_id}",
            'resolution_criteria': f"Criteria for question {self.next_id}",
            'scoring_weight': scoring_weight,
            'status': QuestionStatus.ACTIVE
        }

        # Add type-specific fields
        if question_type == QuestionType.NUMERIC:
            kwargs.update({'min_value': 0.0, 'max_value': 100.0})
        elif question_type == QuestionType.MULTIPLE_CHOICE:
            kwargs.update({'choices': ['A', 'B', 'C']})

        question = Question(**kwargs)
        self.questions[self.next_id] = question
        self.next_id += 1

    @rule(question_id=st.integers(min_value=1, max_value=100))
    def check_question_properties(self, question_id):
        """Check properties of existing questions."""
        if question_id in self.questions:
            question = self.questions[question_id]

            # Check that question maintains its invariants
            assert question.id > 0
            assert question.text.strip() != ""
            assert question.scoring_weight > 0

            # Check type-specific properties
            if question.question_type == QuestionType.NUMERIC:
                assert question.min_value is not None
                assert question.max_value is not None
                assert question.min_value < question.max_value
            elif question.question_type == QuestionType.MULTIPLE_CHOICE:
                assert question.choices is not None
                assert len(question.choices) >= 2

    @invariant()
    def questions_have_unique_ids(self):
        """Invariant: all questions should have unique IDs."""
        ids = [q.id for q in self.questions.values()]
        assert len(ids) == len(set(ids))

    @invariant()
    def questions_have_future_deadlines(self):
        """Invariant: all active questions should have future deadlines."""
        for question in self.questions.values():
            if question.status == QuestionStatus.ACTIVE:
                # Allow some tolerance for test execution time
                assert question.deadline > datetime.utcnow() - timedelta(seconds=10)


# Stateful test class
TestQuestionStateMachine = QuestionStateMachine.TestCase


class TestPropertyBasedEdgeCases:
    """Property-based tests for edge cases and boundary conditions."""

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=200)
    def test_confidence_combination_commutativity(self, level1, level2):
        """Test that confidence combination is not commutative (by design)."""
        conf1 = Confidence(level=level1, basis="Basis 1")
        conf2 = Confidence(level=level2, basis="Basis 2")

        # Combination should not be commutative due to weighting
        combined_1_2 = conf1.combine_with(conf2, weight=0.7)
        combined_2_1 = conf2.combine_with(conf1, weight=0.7)

        if level1 != level2:
            assert combined_1_2.level != combined_2_1.level

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=2,
            max_size=10
        ).filter(lambda lst: abs(sum(lst) - 1.0) < 0.0001)
    )
    def test_multiple_choice_probability_distribution_properties(self, probabilities):
        """Test properties of multiple choice probability distributions."""
        choices = [f"choice_{i}" for i in range(len(probabilities))]
        choice_dict = dict(zip(choices, probabilities))

        result = PredictionResult(
            value=choice_dict,
            prediction_type=PredictionType.MULTIPLE_CHOICE
        )

        # Should always validate if probabilities sum to 1.0
        assert result.validate()

        # Entropy should be maximized when probabilities are equal
        if len(set(probabilities)) == 1:  # All probabilities equal
            # This represents maximum entropy for this number of choices
            expected_prob = 1.0 / len(probabilities)
            for prob in probabilities:
                assert abs(prob - expected_prob) < 0.0001

    @given(
        st.integers(min_value=1, max_value=1000),
        st.floats(min_value=0.001, max_value=1000.0)
    )
    def test_question_complexity_monotonicity(self, text_length, scoring_weight):
        """Test that question complexity increases with certain factors."""
        # Create questions with different text lengths
        short_text = "Short question?"
        long_text = "Very " * text_length + "complex question with many details?"

        short_question = Question(
            id=1,
            text=short_text,
            question_type=QuestionType.BINARY,
            category=QuestionCategory.OTHER,
            deadline=datetime.utcnow() + timedelta(days=30),
            background="Background",
            resolution_criteria="Criteria",
            scoring_weight=scoring_weight
        )

        long_question = Question(
            id=2,
            text=long_text,
            question_type=QuestionType.CONDITIONAL,  # More complex type
            category=QuestionCategory.AI_DEVELOPMENT,  # Specialized category
            deadline=datetime.utcnow() + timedelta(days=30),
            background="Background",
            resolution_criteria="Criteria",
            scoring_weight=scoring_weight
        )

        # Longer, more complex questions should have higher complexity scores
        if text_length > 10:
            assert long_question.get_complexity_score() > short_question.get_complexity_score()
