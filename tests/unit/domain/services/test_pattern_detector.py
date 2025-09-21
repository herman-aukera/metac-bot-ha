"""Tests for PatternDetector service."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.domain.entities.forecast import Forecast, ForecastStatus
from src.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
)
from src.domain.entities.question import Question, QuestionStatus, QuestionType
from src.domain.services.pattern_detector import (
    AdaptationRecommendation,
    AdaptationStrategy,
    CompetitiveIntelligence,
    DetectedPattern,
    PatternDetector,
    PatternType,
)


class TestPatternDetector:
    """Test cases for PatternDetector."""

    @pytest.fixture
    def detector(self):
        """Create PatternDetector instance."""
        return PatternDetector()

    @pytest.fixture
    def sample_questions(self):
        """Create sample questions for testing."""
        questions = []
        question_types = [
            QuestionType.BINARY,
            QuestionType.NUMERIC,
            QuestionType.MULTIPLE_CHOICE,
        ]

        for i, qtype in enumerate(question_types * 3):  # 9 questions total
            # Set type-specific fields
            min_value = 0.0 if qtype == QuestionType.NUMERIC else None
            max_value = 100.0 if qtype == QuestionType.NUMERIC else None
            choices = (
                ["Option A", "Option B", "Option C"]
                if qtype == QuestionType.MULTIPLE_CHOICE
                else None
            )

            question = Question(
                id=uuid4(),
                metaculus_id=1000 + i,
                title=f"Test Question {i+1}",
                description=f"Description for question {i+1}",
                question_type=qtype,
                status=QuestionStatus.OPEN,
                url=f"https://test.com/question/{i+1}",
                close_time=datetime.utcnow() + timedelta(days=30 + i),
                resolve_time=datetime.utcnow() + timedelta(days=60 + i),
                categories=["test_category"],
                metadata={},
                created_at=datetime.utcnow() - timedelta(days=30 - i),
                updated_at=datetime.utcnow(),
                resolution_criteria="Test criteria",
                min_value=min_value,
                max_value=max_value,
                choices=choices,
            )
            questions.append(question)

        return questions

    @pytest.fixture
    def sample_forecasts_with_patterns(self, sample_questions):
        """Create sample forecasts with detectable patterns."""
        forecasts = []
        research_report_id = uuid4()

        # Create forecasts with question type performance patterns
        for i, question in enumerate(sample_questions):
            # Binary questions perform better (pattern)
            if question.question_type == QuestionType.BINARY:
                prob = 0.8 + (i % 3) * 0.05  # High performance
                conf = PredictionConfidence.HIGH
                agent = "binary_specialist"
            # Numeric questions perform worse (pattern)
            elif question.question_type == QuestionType.NUMERIC:
                prob = 0.3 + (i % 3) * 0.05  # Low performance
                conf = PredictionConfidence.LOW
                agent = "numeric_agent"
            else:
                prob = 0.6 + (i % 3) * 0.05  # Average performance
                conf = PredictionConfidence.MEDIUM
                agent = "general_agent"

            # Add temporal pattern - performance improves over time
            time_factor = i / len(sample_questions)
            prob += time_factor * 0.1

            # Create prediction with method pattern
            methods = [
                PredictionMethod.CHAIN_OF_THOUGHT,
                PredictionMethod.TREE_OF_THOUGHT,
                PredictionMethod.REACT,
            ]
            method = methods[i % 3]

            prediction = Prediction.create_binary_prediction(
                question_id=question.id,
                research_report_id=research_report_id,
                probability=min(0.95, max(0.05, prob)),  # Clamp to valid range
                confidence=conf,
                method=method,
                reasoning=f"Test reasoning for question {i+1}",
                created_by=agent,
            )

            forecast = Forecast.create_new(
                question_id=question.id,
                research_reports=[],
                predictions=[prediction],
                final_prediction=prediction,
            )

            # Set creation time for temporal patterns
            forecast.created_at = datetime.utcnow() - timedelta(days=30 - i * 2)
            forecast.status = ForecastStatus.RESOLVED

            forecasts.append(forecast)

        return forecasts

    @pytest.fixture
    def sample_ground_truth_with_patterns(self, sample_forecasts_with_patterns):
        """Create ground truth that creates detectable patterns."""
        ground_truth = []

        for i, forecast in enumerate(sample_forecasts_with_patterns):
            # Binary questions have higher accuracy (matches the pattern)
            if forecast.final_prediction.created_by == "binary_specialist":
                # High accuracy for binary questions
                truth = (
                    forecast.final_prediction.result.binary_probability > 0.6
                )  # Most should be correct
            elif forecast.final_prediction.created_by == "numeric_agent":
                # Low accuracy for numeric questions
                truth = (
                    forecast.final_prediction.result.binary_probability < 0.4
                )  # Most should be incorrect
            else:
                # Average accuracy for others
                truth = i % 2 == 0  # 50% accuracy

            ground_truth.append(truth)

        return ground_truth

    @pytest.fixture
    def tournament_context(self):
        """Create sample tournament context."""
        return {
            "tournament_id": "test_tournament_2024",
            "deadlines": {
                "submission": datetime.utcnow() + timedelta(days=7),
                "resolution": datetime.utcnow() + timedelta(days=30),
            },
            "competitor_data": {"total_participants": 100, "our_ranking": 15},
        }

    def test_detect_patterns_basic(
        self,
        detector,
        sample_forecasts_with_patterns,
        sample_questions,
        sample_ground_truth_with_patterns,
    ):
        """Test basic pattern detection functionality."""
        results = detector.detect_patterns(
            sample_forecasts_with_patterns,
            sample_questions,
            sample_ground_truth_with_patterns,
        )

        assert "analysis_timestamp" in results
        assert "total_patterns_detected" in results
        assert "significant_patterns" in results
        assert "patterns_by_type" in results
        assert "detected_patterns" in results
        assert "adaptation_recommendations" in results

        assert isinstance(results["detected_patterns"], list)
        assert isinstance(results["adaptation_recommendations"], list)

    def test_detect_question_type_patterns(
        self,
        detector,
        sample_forecasts_with_patterns,
        sample_questions,
        sample_ground_truth_with_patterns,
    ):
        """Test question type pattern detection."""
        patterns = detector._detect_question_type_patterns(
            sample_forecasts_with_patterns,
            sample_questions,
            sample_ground_truth_with_patterns,
            None,
        )

        assert isinstance(patterns, list)

        # Should detect patterns for different question types
        pattern_types = [p.context.get("question_type") for p in patterns]
        assert len(set(pattern_types)) > 0  # At least one question type pattern

        for pattern in patterns:
            assert pattern.pattern_type == PatternType.QUESTION_TYPE_PERFORMANCE
            assert pattern.confidence > 0
            assert pattern.strength >= 0
            assert "question_type" in pattern.context
            assert "performance_difference" in pattern.context

    def test_detect_temporal_patterns(
        self,
        detector,
        sample_forecasts_with_patterns,
        sample_questions,
        sample_ground_truth_with_patterns,
    ):
        """Test temporal pattern detection."""
        patterns = detector._detect_temporal_patterns(
            sample_forecasts_with_patterns,
            sample_questions,
            sample_ground_truth_with_patterns,
            None,
        )

        # Should detect temporal improvement pattern
        if patterns:  # May not always detect depending on data
            temporal_pattern = patterns[0]
            assert temporal_pattern.pattern_type == PatternType.TEMPORAL_PERFORMANCE
            assert temporal_pattern.trend_direction in [
                "improving",
                "declining",
                "stable",
            ]
            assert "correlation" in temporal_pattern.context

    def test_detect_calibration_patterns(
        self,
        detector,
        sample_forecasts_with_patterns,
        sample_questions,
        sample_ground_truth_with_patterns,
    ):
        """Test calibration pattern detection."""
        patterns = detector._detect_calibration_patterns(
            sample_forecasts_with_patterns,
            sample_questions,
            sample_ground_truth_with_patterns,
            None,
        )

        for pattern in patterns:
            assert pattern.pattern_type == PatternType.CONFIDENCE_CALIBRATION
            assert "confidence_level" in pattern.context
            assert "miscalibration_type" in pattern.context
            assert pattern.context["miscalibration_type"] in [
                "overconfident",
                "underconfident",
            ]

    def test_detect_method_patterns(
        self,
        detector,
        sample_forecasts_with_patterns,
        sample_questions,
        sample_ground_truth_with_patterns,
    ):
        """Test method effectiveness pattern detection."""
        patterns = detector._detect_method_patterns(
            sample_forecasts_with_patterns,
            sample_questions,
            sample_ground_truth_with_patterns,
            None,
        )

        if patterns:  # May not always detect significant differences
            method_pattern = patterns[0]
            assert method_pattern.pattern_type == PatternType.METHOD_EFFECTIVENESS
            assert "best_method" in method_pattern.context
            assert "worst_method" in method_pattern.context
            assert "performance_gap" in method_pattern.context

    def test_detect_ensemble_patterns(self, detector, sample_questions):
        """Test ensemble synergy pattern detection."""
        # Create forecasts with ensemble vs individual methods
        forecasts = []
        research_report_id = uuid4()

        # Create ensemble forecasts (better performance)
        for i in range(5):
            prediction = Prediction.create_binary_prediction(
                question_id=sample_questions[i].id,
                research_report_id=research_report_id,
                probability=0.8,  # High accuracy
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.ENSEMBLE,
                reasoning="Ensemble reasoning",
                created_by="ensemble_agent",
            )

            forecast = Forecast.create_new(
                question_id=sample_questions[i].id,
                research_reports=[],
                predictions=[prediction],
                final_prediction=prediction,
            )
            forecast.method = "ensemble"  # Set method for pattern detection
            forecasts.append(forecast)

        # Create individual method forecasts (worse performance)
        for i in range(5, 10):
            if i < len(sample_questions):
                prediction = Prediction.create_binary_prediction(
                    question_id=sample_questions[i].id,
                    research_report_id=research_report_id,
                    probability=0.4,  # Lower accuracy
                    confidence=PredictionConfidence.MEDIUM,
                    method=PredictionMethod.CHAIN_OF_THOUGHT,
                    reasoning="Individual reasoning",
                    created_by="individual_agent",
                )

                forecast = Forecast.create_new(
                    question_id=sample_questions[i].id,
                    research_reports=[],
                    predictions=[prediction],
                    final_prediction=prediction,
                )
                forecast.method = "individual"  # Set method for pattern detection
                forecasts.append(forecast)

        ground_truth = [True] * 5 + [
            False
        ] * 5  # Ensemble correct, individual incorrect

        patterns = detector._detect_ensemble_patterns(
            forecasts, sample_questions, ground_truth, None
        )

        if patterns:
            ensemble_pattern = patterns[0]
            assert ensemble_pattern.pattern_type == PatternType.ENSEMBLE_SYNERGY
            assert "ensemble_advantage" in ensemble_pattern.context

    def test_generate_adaptation_recommendations(self, detector):
        """Test adaptation recommendation generation."""
        # Create test patterns
        test_patterns = [
            DetectedPattern(
                pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
                title="Binary Question Advantage",
                description="Strong performance on binary questions",
                confidence=0.8,
                strength=0.2,
                frequency=0.3,
                context={"question_type": "binary", "performance_difference": 0.2},
                affected_questions=[uuid4()],
                affected_agents=["test_agent"],
                first_observed=datetime.utcnow() - timedelta(days=10),
                last_observed=datetime.utcnow(),
                trend_direction="stable",
                statistical_significance=0.01,
            ),
            DetectedPattern(
                pattern_type=PatternType.CONFIDENCE_CALIBRATION,
                title="Overconfidence Issue",
                description="Overconfident at high confidence",
                confidence=0.9,
                strength=0.3,
                frequency=0.2,
                context={
                    "confidence_level": "high",
                    "miscalibration_type": "overconfident",
                    "calibration_error": 0.25,
                },
                affected_questions=[uuid4()],
                affected_agents=["test_agent"],
                first_observed=datetime.utcnow() - timedelta(days=5),
                last_observed=datetime.utcnow(),
                trend_direction="stable",
                statistical_significance=0.01,
            ),
        ]

        recommendations = detector._generate_adaptation_recommendations(
            test_patterns, None
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        for rec in recommendations:
            assert isinstance(rec, AdaptationRecommendation)
            assert rec.strategy_type in AdaptationStrategy
            assert rec.confidence > 0
            assert rec.expected_impact >= 0
            assert len(rec.specific_actions) > 0

    def test_recommend_question_type_adaptation(self, detector):
        """Test question type adaptation recommendations."""
        # Test strong performance pattern
        strong_pattern = DetectedPattern(
            pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
            title="Binary Question Advantage",
            description="Strong performance on binary questions",
            confidence=0.8,
            strength=0.2,
            frequency=0.3,
            context={
                "question_type": "binary",
                "performance_difference": 0.15,  # Strong positive difference
            },
            affected_questions=[uuid4()],
            affected_agents=["test_agent"],
            first_observed=datetime.utcnow() - timedelta(days=10),
            last_observed=datetime.utcnow(),
            trend_direction="stable",
            statistical_significance=0.01,
        )

        rec = detector._recommend_question_type_adaptation(strong_pattern)

        assert rec is not None
        assert rec.strategy_type == AdaptationStrategy.FOCUS_QUESTION_TYPES
        assert "binary" in rec.affected_contexts

        # Test poor performance pattern
        weak_pattern = DetectedPattern(
            pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
            title="Numeric Question Weakness",
            description="Poor performance on numeric questions",
            confidence=0.8,
            strength=0.2,
            frequency=0.3,
            context={
                "question_type": "numeric",
                "performance_difference": -0.15,  # Strong negative difference
            },
            affected_questions=[uuid4()],
            affected_agents=["test_agent"],
            first_observed=datetime.utcnow() - timedelta(days=10),
            last_observed=datetime.utcnow(),
            trend_direction="stable",
            statistical_significance=0.01,
        )

        rec = detector._recommend_question_type_adaptation(weak_pattern)

        assert rec is not None
        assert rec.strategy_type == AdaptationStrategy.MODIFY_RESEARCH_DEPTH
        assert "numeric" in rec.affected_contexts

    def test_recommend_calibration_adaptation(self, detector):
        """Test calibration adaptation recommendations."""
        # Test overconfidence pattern
        overconfident_pattern = DetectedPattern(
            pattern_type=PatternType.CONFIDENCE_CALIBRATION,
            title="High Confidence Overconfidence",
            description="Overconfident at high confidence level",
            confidence=0.9,
            strength=0.3,
            frequency=0.2,
            context={
                "confidence_level": "high",
                "miscalibration_type": "overconfident",
                "calibration_error": 0.2,
            },
            affected_questions=[uuid4()],
            affected_agents=["test_agent"],
            first_observed=datetime.utcnow() - timedelta(days=5),
            last_observed=datetime.utcnow(),
            trend_direction="stable",
            statistical_significance=0.01,
        )

        rec = detector._recommend_calibration_adaptation(overconfident_pattern)

        assert rec is not None
        assert rec.strategy_type == AdaptationStrategy.DECREASE_CONFIDENCE
        assert "high_confidence" in rec.affected_contexts

        # Test underconfidence pattern
        underconfident_pattern = DetectedPattern(
            pattern_type=PatternType.CONFIDENCE_CALIBRATION,
            title="Low Confidence Underconfidence",
            description="Underconfident at low confidence level",
            confidence=0.8,
            strength=0.2,
            frequency=0.3,
            context={
                "confidence_level": "low",
                "miscalibration_type": "underconfident",
                "calibration_error": 0.15,
            },
            affected_questions=[uuid4()],
            affected_agents=["test_agent"],
            first_observed=datetime.utcnow() - timedelta(days=5),
            last_observed=datetime.utcnow(),
            trend_direction="stable",
            statistical_significance=0.01,
        )

        rec = detector._recommend_calibration_adaptation(underconfident_pattern)

        assert rec is not None
        assert rec.strategy_type == AdaptationStrategy.INCREASE_CONFIDENCE
        assert "low_confidence" in rec.affected_contexts

    def test_recommend_method_adaptation(self, detector):
        """Test method adaptation recommendations."""
        method_pattern = DetectedPattern(
            pattern_type=PatternType.METHOD_EFFECTIVENESS,
            title="Method Performance Gap",
            description="Significant method performance difference",
            confidence=0.9,
            strength=0.25,
            frequency=1.0,
            context={
                "best_method": "ensemble",
                "worst_method": "chain_of_thought",
                "performance_gap": 0.25,
            },
            affected_questions=[uuid4()],
            affected_agents=["test_agent"],
            first_observed=datetime.utcnow() - timedelta(days=10),
            last_observed=datetime.utcnow(),
            trend_direction="stable",
            statistical_significance=0.01,
        )

        rec = detector._recommend_method_adaptation(method_pattern)

        assert rec is not None
        assert rec.strategy_type == AdaptationStrategy.CHANGE_METHOD_PREFERENCE
        assert "ensemble" in str(rec.specific_actions)
        assert "chain_of_thought" in str(rec.specific_actions)

    def test_generate_competitive_intelligence(
        self,
        detector,
        sample_forecasts_with_patterns,
        sample_questions,
        tournament_context,
    ):
        """Test competitive intelligence generation."""
        # Create some patterns first
        patterns = [
            DetectedPattern(
                pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
                title="Binary Question Advantage",
                description="Strong performance on binary questions",
                confidence=0.8,
                strength=0.2,
                frequency=0.3,
                context={"question_type": "binary", "performance_difference": 0.15},
                affected_questions=[uuid4()],
                affected_agents=["test_agent"],
                first_observed=datetime.utcnow() - timedelta(days=10),
                last_observed=datetime.utcnow(),
                trend_direction="stable",
                statistical_significance=0.01,
            )
        ]

        intelligence = detector._generate_competitive_intelligence(
            patterns,
            sample_forecasts_with_patterns,
            sample_questions,
            tournament_context,
        )

        assert intelligence is not None
        assert isinstance(intelligence, CompetitiveIntelligence)
        assert intelligence.tournament_id == tournament_context["tournament_id"]
        assert isinstance(intelligence.market_gaps, list)
        assert isinstance(intelligence.strategic_recommendations, list)

    def test_detect_meta_patterns(self, detector):
        """Test meta-pattern detection."""
        # Create multiple patterns of same type
        patterns = [
            DetectedPattern(
                pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
                title="Pattern 1",
                description="Description 1",
                confidence=0.8,
                strength=0.3,  # High strength
                frequency=0.3,
                context={},
                affected_questions=[uuid4()],
                affected_agents=["agent1"],
                first_observed=datetime.utcnow() - timedelta(days=10),
                last_observed=datetime.utcnow(),
                trend_direction="stable",
                statistical_significance=0.01,
            ),
            DetectedPattern(
                pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
                title="Pattern 2",
                description="Description 2",
                confidence=0.7,
                strength=0.25,  # High strength
                frequency=0.4,
                context={},
                affected_questions=[uuid4()],
                affected_agents=["agent2"],
                first_observed=datetime.utcnow() - timedelta(days=5),
                last_observed=datetime.utcnow(),
                trend_direction="stable",
                statistical_significance=0.01,
            ),
            DetectedPattern(
                pattern_type=PatternType.CONFIDENCE_CALIBRATION,
                title="Pattern 3",
                description="Description 3",
                confidence=0.9,
                strength=0.4,  # High strength
                frequency=0.2,
                context={},
                affected_questions=[uuid4()],
                affected_agents=["agent3"],
                first_observed=datetime.utcnow() - timedelta(days=3),
                last_observed=datetime.utcnow(),
                trend_direction="stable",
                statistical_significance=0.01,
            ),
        ]

        meta_patterns = detector._detect_meta_patterns(patterns)

        assert isinstance(meta_patterns, list)

        # Should detect pattern frequency meta-pattern
        frequency_patterns = [
            mp for mp in meta_patterns if mp["type"] == "pattern_frequency"
        ]
        assert len(frequency_patterns) > 0

        # Should detect high impact patterns meta-pattern
        high_impact_patterns = [
            mp for mp in meta_patterns if mp["type"] == "high_impact_patterns"
        ]
        assert len(high_impact_patterns) > 0

    def test_generate_strategy_evolution_suggestions(self, detector):
        """Test strategy evolution suggestion generation."""
        patterns = [
            DetectedPattern(
                pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
                title="Performance Pattern",
                description="Performance pattern description",
                confidence=0.8,
                strength=0.2,
                frequency=0.3,
                context={},
                affected_questions=[uuid4()],
                affected_agents=["agent1"],
                first_observed=datetime.utcnow() - timedelta(days=10),
                last_observed=datetime.utcnow(),
                trend_direction="stable",
                statistical_significance=0.01,
            ),
            DetectedPattern(
                pattern_type=PatternType.METHOD_EFFECTIVENESS,
                title="Method Pattern",
                description="Method pattern description",
                confidence=0.9,
                strength=0.3,
                frequency=0.4,
                context={},
                affected_questions=[uuid4()],
                affected_agents=["agent2"],
                first_observed=datetime.utcnow() - timedelta(days=5),
                last_observed=datetime.utcnow(),
                trend_direction="stable",
                statistical_significance=0.01,
            ),
            DetectedPattern(
                pattern_type=PatternType.CONFIDENCE_CALIBRATION,
                title="Calibration Pattern",
                description="Calibration pattern description",
                confidence=0.7,
                strength=0.25,
                frequency=0.2,
                context={},
                affected_questions=[uuid4()],
                affected_agents=["agent3"],
                first_observed=datetime.utcnow() - timedelta(days=3),
                last_observed=datetime.utcnow(),
                trend_direction="stable",
                statistical_significance=0.01,
            ),
        ]

        suggestions = detector._generate_strategy_evolution_suggestions(patterns)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        # Should include suggestions for different pattern types
        suggestion_text = " ".join(suggestions)
        assert "calibration" in suggestion_text.lower()
        assert "method" in suggestion_text.lower()

    def test_empty_forecasts_handling(self, detector, sample_questions):
        """Test handling of empty forecast lists."""
        results = detector.detect_patterns([], sample_questions, [])

        assert "message" in results
        assert "No forecasts provided" in results["message"]

    def test_no_ground_truth_handling(
        self, detector, sample_forecasts_with_patterns, sample_questions
    ):
        """Test handling when no ground truth is provided."""
        results = detector.detect_patterns(
            sample_forecasts_with_patterns, sample_questions, None  # No ground truth
        )

        # Should still work but with limited pattern detection
        assert "detected_patterns" in results
        assert isinstance(results["detected_patterns"], list)

    def test_insufficient_samples_handling(self, detector, sample_questions):
        """Test handling of insufficient samples for pattern detection."""
        # Create minimal forecasts
        forecasts = []
        for i in range(3):  # Below minimum threshold
            prediction = Prediction.create_binary_prediction(
                question_id=sample_questions[i].id,
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.ENSEMBLE,
                reasoning="Test reasoning",
                created_by="test_agent",
            )

            forecast = Forecast.create_new(
                question_id=sample_questions[i].id,
                research_reports=[],
                predictions=[prediction],
                final_prediction=prediction,
            )
            forecasts.append(forecast)

        ground_truth = [True, False, True]

        results = detector.detect_patterns(forecasts, sample_questions, ground_truth)

        # Should work but detect fewer patterns
        assert "detected_patterns" in results
        assert results["total_patterns_detected"] >= 0

    def test_serialization_methods(self, detector):
        """Test serialization of pattern objects."""
        # Test pattern serialization
        pattern = DetectedPattern(
            pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
            title="Test Pattern",
            description="Test pattern description",
            confidence=0.8,
            strength=0.2,
            frequency=0.3,
            context={"test_key": "test_value"},
            affected_questions=[uuid4(), uuid4()],
            affected_agents=["agent1", "agent2"],
            first_observed=datetime.utcnow() - timedelta(days=10),
            last_observed=datetime.utcnow(),
            trend_direction="improving",
            statistical_significance=0.01,
            examples=[{"example": "data"}],
        )

        serialized = detector._serialize_pattern(pattern)

        assert serialized["type"] == "question_type_performance"
        assert serialized["title"] == "Test Pattern"
        assert serialized["confidence"] == 0.8
        assert serialized["affected_questions_count"] == 2
        assert len(serialized["affected_agents"]) == 2

        # Test recommendation serialization
        recommendation = AdaptationRecommendation(
            strategy_type=AdaptationStrategy.FOCUS_QUESTION_TYPES,
            title="Test Recommendation",
            description="Test recommendation description",
            rationale="Test rationale",
            expected_impact=0.1,
            confidence=0.8,
            priority=0.7,
            implementation_complexity=0.5,
            affected_contexts=["context1"],
            specific_actions=["action1", "action2"],
            success_metrics=["metric1"],
            timeline="short_term",
        )

        serialized_rec = detector._serialize_recommendation(recommendation)

        assert serialized_rec["strategy_type"] == "focus_question_types"
        assert serialized_rec["title"] == "Test Recommendation"
        assert serialized_rec["expected_impact"] == 0.1
        assert len(serialized_rec["specific_actions"]) == 2

        # Test competitive intelligence serialization
        intelligence = CompetitiveIntelligence(
            tournament_id="test_tournament",
            market_gaps=[{"gap": "test"}],
            competitor_weaknesses=[],
            optimal_positioning={"position": "test"},
            timing_opportunities=[],
            question_type_advantages={},
            confidence_level_opportunities={},
            meta_game_insights=["insight1"],
            strategic_recommendations=["rec1"],
            timestamp=datetime.utcnow(),
            confidence=0.7,
        )

        serialized_intel = detector._serialize_competitive_intelligence(intelligence)

        assert serialized_intel["tournament_id"] == "test_tournament"
        assert serialized_intel["confidence"] == 0.7
        assert len(serialized_intel["meta_game_insights"]) == 1

    def test_get_pattern_history(self, detector):
        """Test pattern history retrieval."""
        # Add some test patterns
        test_patterns = [
            DetectedPattern(
                pattern_type=PatternType.QUESTION_TYPE_PERFORMANCE,
                title="Recent Pattern",
                description="Recent pattern description",
                confidence=0.8,
                strength=0.2,
                frequency=0.3,
                context={},
                affected_questions=[uuid4()],
                affected_agents=["agent1"],
                first_observed=datetime.utcnow() - timedelta(days=5),
                last_observed=datetime.utcnow() - timedelta(days=1),
                trend_direction="stable",
                statistical_significance=0.01,
            ),
            DetectedPattern(
                pattern_type=PatternType.CONFIDENCE_CALIBRATION,
                title="Old Pattern",
                description="Old pattern description",
                confidence=0.7,
                strength=0.15,
                frequency=0.2,
                context={},
                affected_questions=[uuid4()],
                affected_agents=["agent2"],
                first_observed=datetime.utcnow() - timedelta(days=50),
                last_observed=datetime.utcnow() - timedelta(days=40),
                trend_direction="stable",
                statistical_significance=0.05,
            ),
        ]

        detector.detected_patterns.extend(test_patterns)

        history = detector.get_pattern_history(days=30)

        assert "period_days" in history
        assert "total_patterns" in history
        assert "patterns_by_type" in history
        assert history["period_days"] == 30
        assert history["total_patterns"] == 1  # Only recent pattern

    def test_get_adaptation_tracking(self, detector):
        """Test adaptation recommendation tracking."""
        # Add test recommendations
        test_recommendations = [
            AdaptationRecommendation(
                strategy_type=AdaptationStrategy.FOCUS_QUESTION_TYPES,
                title="High Priority Rec",
                description="High priority recommendation",
                rationale="Test rationale",
                expected_impact=0.15,
                confidence=0.9,
                priority=0.9,  # High priority
                implementation_complexity=0.3,
                affected_contexts=["context1"],
                specific_actions=["action1"],
                success_metrics=["metric1"],
                timeline="immediate",
            ),
            AdaptationRecommendation(
                strategy_type=AdaptationStrategy.MODIFY_RESEARCH_DEPTH,
                title="Low Priority Rec",
                description="Low priority recommendation",
                rationale="Test rationale",
                expected_impact=0.05,
                confidence=0.6,
                priority=0.3,  # Low priority
                implementation_complexity=0.7,
                affected_contexts=["context2"],
                specific_actions=["action2"],
                success_metrics=["metric2"],
                timeline="long_term",
            ),
        ]

        detector.adaptation_recommendations.extend(test_recommendations)

        tracking = detector.get_adaptation_tracking()

        assert "total_recommendations" in tracking
        assert "high_priority_recommendations" in tracking
        assert "recommendations_by_strategy" in tracking
        assert "expected_total_impact" in tracking
        assert tracking["total_recommendations"] == 2

    def test_pattern_type_enumeration(self):
        """Test pattern type enumeration completeness."""
        expected_types = [
            "QUESTION_TYPE_PERFORMANCE",
            "TEMPORAL_PERFORMANCE",
            "CONFIDENCE_CALIBRATION",
            "METHOD_EFFECTIVENESS",
            "TOURNAMENT_DYNAMICS",
            "COMPETITIVE_POSITIONING",
            "MARKET_INEFFICIENCY",
            "SEASONAL_TRENDS",
            "COMPLEXITY_CORRELATION",
            "ENSEMBLE_SYNERGY",
        ]

        for expected_type in expected_types:
            assert hasattr(PatternType, expected_type)

    def test_adaptation_strategy_enumeration(self):
        """Test adaptation strategy enumeration completeness."""
        expected_strategies = [
            "INCREASE_CONFIDENCE",
            "DECREASE_CONFIDENCE",
            "CHANGE_METHOD_PREFERENCE",
            "ADJUST_ENSEMBLE_WEIGHTS",
            "MODIFY_RESEARCH_DEPTH",
            "ALTER_SUBMISSION_TIMING",
            "FOCUS_QUESTION_TYPES",
            "EXPLOIT_MARKET_GAP",
            "INCREASE_CONSERVATISM",
            "INCREASE_AGGRESSIVENESS",
        ]

        for expected_strategy in expected_strategies:
            assert hasattr(AdaptationStrategy, expected_strategy)

    def test_comprehensive_pattern_detection_workflow(
        self,
        detector,
        sample_forecasts_with_patterns,
        sample_questions,
        sample_ground_truth_with_patterns,
        tournament_context,
    ):
        """Test the complete pattern detection workflow."""
        results = detector.detect_patterns(
            sample_forecasts_with_patterns,
            sample_questions,
            sample_ground_truth_with_patterns,
            tournament_context,
        )

        # Verify all expected sections are present
        expected_sections = [
            "analysis_timestamp",
            "total_patterns_detected",
            "significant_patterns",
            "patterns_by_type",
            "detected_patterns",
            "adaptation_recommendations",
            "competitive_intelligence",
            "meta_patterns",
            "strategy_evolution_suggestions",
        ]

        for section in expected_sections:
            assert section in results, f"Missing section: {section}"

        # Verify data types and structure
        assert isinstance(results["detected_patterns"], list)
        assert isinstance(results["adaptation_recommendations"], list)
        assert isinstance(results["meta_patterns"], list)
        assert isinstance(results["strategy_evolution_suggestions"], list)

        # Verify patterns have required fields
        for pattern_data in results["detected_patterns"]:
            assert "type" in pattern_data
            assert "confidence" in pattern_data
            assert "strength" in pattern_data
            assert "trend_direction" in pattern_data

        # Verify recommendations have required fields
        for rec_data in results["adaptation_recommendations"]:
            assert "strategy_type" in rec_data
            assert "expected_impact" in rec_data
            assert "priority" in rec_data
            assert "specific_actions" in rec_data
