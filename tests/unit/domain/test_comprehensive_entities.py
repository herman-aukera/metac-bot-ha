"""Comprehensive unit tests for all domain entities."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, patch

from src.domain.entities.question import Question, QuestionType, QuestionCategory, QuestionStatus
from src.domain.entities.prediction import Prediction
from src.domain.entities.tournament import Tournament, ScoringRules, ScoringMethod
from src.domain.entities.agent import Agent, ReasoningStyle
from src.domain.entities.research_report import ResearchReport
from src.domain.entities.forecast import Forecast
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.reasoning_step import ReasoningStep


class TestQuestionComprehensive:
    """Comprehensive tests for Question entity covering edge cases and business logic."""

    def test_question_complexity_calculation_edge_cases(self, test_data_factory):
        """Test complexity calculation with various edge cases."""
        # Very short question
        short_question = test_data_factory.create_question(
            text="Yes?",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.OTHER
        )
        assert short_question.get_complexity_score() == 1.0

        # Maximum complexity question
        very_complex_text = " ".join(["complex"] * 100)  # 100 words
        complex_question = test_data_factory.create_question(
            text=very_complex_text,
            question_type=QuestionType.CONDITIONAL,
            category=QuestionCategory.AI_DEVELOPMENT
        )
        complexity = complex_question.get_complexity_score()
        assert complexity > 3.0

        # Numeric question complexity
        numeric_question = test_data_factory.create_question(
            question_type=QuestionType.NUMERIC,
            min_value=0.0,
            max_value=1000000.0  # Large range
        )
        assert numeric_question.get_complexity_score() > 1.5

    def test_question_deadline_edge_cases(self, test_data_factory):
        """Test deadline-related edge cases."""
        # Question closing in exactly 1 hour
        one_hour_question = test_data_factory.create_question(
            deadline_offset_days=0
        )
        one_hour_question.deadline = datetime.utcnow() + timedelta(hours=1)

        assert one_hour_question.is_deadline_approaching(2.0)  # Within 2 hours
        assert not one_hour_question.is_deadline_approaching(0.5)  # Not within 30 minutes

        # Question with deadline in the past (edge case for resolved questions)
        past_question = test_data_factory.create_question()
        past_question.deadline = datetime.utcnow() - timedelta(hours=1)

        assert past_question.time_until_deadline() == 0.0

    def test_question_specialized_knowledge_categories(self, test_data_factory):
        """Test specialized knowledge requirements for all categories."""
        specialized_categories = [
            QuestionCategory.AI_DEVELOPMENT,
            QuestionCategory.TECHNOLOGY,
            QuestionCategory.SCIENCE,
            QuestionCategory.ECONOMICS,
            QuestionCategory.POLITICS
        ]

        for category in specialized_categories:
            question = test_data_factory.create_question(category=category)
            if category in [QuestionCategory.AI_DEVELOPMENT, QuestionCategory.TECHNOLOGY, QuestionCategory.SCIENCE]:
                assert question.requires_specialized_knowledge()
            else:
                # Some categories might not require specialized knowledge
                pass

    def test_question_validation_boundary_conditions(self, test_data_factory):
        """Test validation at boundary conditions."""
        # Minimum valid scoring weight
        min_weight_question = test_data_factory.create_question(scoring_weight=0.001)
        assert min_weight_question.scoring_weight == 0.001

        # Very large scoring weight
        large_weight_question = test_data_factory.create_question(scoring_weight=1000.0)
        assert large_weight_question.scoring_weight == 1000.0

        # Numeric question with equal min/max (edge case)
        with pytest.raises(ValueError):
            test_data_factory.create_question(
                question_type=QuestionType.NUMERIC,
                min_value=50.0,
                max_value=50.0
            )

    def test_question_metadata_handling(self, test_data_factory):
        """Test metadata handling and edge cases."""
        # Empty metadata
        question = test_data_factory.create_question(metadata={})
        assert question.metadata == {}

        # Complex nested metadata
        complex_metadata = {
            'source': 'test',
            'tags': ['ai', 'forecasting'],
            'difficulty': {'level': 'hard', 'score': 8.5},
            'nested': {'deep': {'value': 'test'}}
        }
        question = test_data_factory.create_question(metadata=complex_metadata)
        assert question.metadata == complex_metadata

    def test_question_immutability_aspects(self, test_data_factory):
        """Test immutability aspects of question entity."""
        question = test_data_factory.create_question()
        original_id = question.id
        original_text = question.text

        # Verify that key properties maintain their values
        assert question.id == original_id
        assert question.text == original_text

        # Test that created_at is set and doesn't change
        assert question.created_at is not None
        assert isinstance(question.created_at, datetime)


class TestPredictionComprehensive:
    """Comprehensive tests for Prediction entity covering all scenarios."""

    def test_prediction_type_detection_edge_cases(self, test_data_factory):
        """Test prediction type detection with edge cases."""
        # Binary prediction with exactly 0.0
        zero_prediction = test_data_factory.create_prediction(result=0.0)
        assert zero_prediction.is_binary_prediction()
        assert zero_prediction.get_binary_probability() == 0.0

        # Binary prediction with exactly 1.0
        one_prediction = test_data_factory.create_prediction(result=1.0)
        assert one_prediction.is_binary_prediction()
        assert one_prediction.get_binary_probability() == 1.0

        # Numeric prediction with negative value
        negative_prediction = test_data_factory.create_prediction(result=-42.5)
        assert negative_prediction.is_numeric_prediction()
        assert negative_prediction.get_numeric_value() == -42.5

        # Multiple choice with single option (edge case)
        single_choice = test_data_factory.create_prediction(result={'A': 1.0})
        assert single_choice.is_multiple_choice_prediction()
        assert single_choice.get_most_likely_choice() == 'A'

    def test_prediction_confidence_levels(self, test_data_factory):
        """Test prediction confidence level handling."""
        # Very low confidence
        low_conf = test_data_factory.create_prediction(confidence_level=0.01)
        assert not low_conf.has_high_confidence()
        assert low_conf.confidence.is_low()

        # Maximum confidence
        max_conf = test_data_factory.create_prediction(confidence_level=1.0)
        assert max_conf.has_high_confidence()
        assert max_conf.confidence.is_high()

        # Boundary confidence (exactly 0.8)
        boundary_conf = test_data_factory.create_prediction(confidence_level=0.8)
        assert boundary_conf.has_high_confidence()

    def test_prediction_reasoning_steps_handling(self, test_data_factory):
        """Test reasoning steps handling in predictions."""
        # Prediction with no reasoning steps
        no_steps = test_data_factory.create_prediction(reasoning_steps=[])
        summary = no_steps.get_reasoning_summary()
        assert "No detailed reasoning steps" in summary

        # Prediction with multiple reasoning steps
        steps = [
            test_data_factory.create_reasoning_step(1, "Step 1"),
            test_data_factory.create_reasoning_step(2, "Step 2"),
            test_data_factory.create_reasoning_step(3, "Step 3")
        ]
        multi_step = test_data_factory.create_prediction(reasoning_steps=steps)
        summary = multi_step.get_reasoning_summary()
        assert "Step 1:" in summary
        assert "Step 2:" in summary
        assert "Step 3:" in summary

    def test_prediction_evidence_sources(self, test_data_factory):
        """Test evidence sources handling."""
        # No evidence sources
        no_sources = test_data_factory.create_prediction(evidence_sources=[])
        assert len(no_sources.evidence_sources) == 0

        # Many evidence sources
        many_sources = [f"source_{i}" for i in range(20)]
        prediction = test_data_factory.create_prediction(evidence_sources=many_sources)
        assert len(prediction.evidence_sources) == 20

    def test_prediction_submission_format_edge_cases(self, test_data_factory):
        """Test submission format conversion edge cases."""
        # Binary prediction with extreme values
        extreme_binary = test_data_factory.create_prediction(result=0.999999)
        submission = extreme_binary.to_submission_format()
        assert submission['prediction'] == 0.999999

        # Multiple choice with many options
        many_choices = {f"option_{i}": 1.0/10 for i in range(10)}
        mc_prediction = test_data_factory.create_prediction(result=many_choices)
        submission = mc_prediction.to_submission_format()
        assert len(submission['prediction']) == 10
        assert abs(sum(submission['prediction'].values()) - 1.0) < 0.0001

    def test_prediction_factory_methods_validation(self, test_data_factory):
        """Test factory method validation edge cases."""
        # Binary prediction with invalid probability
        with pytest.raises(ValueError, match="Binary probability must be between 0.0 and 1.0"):
            Prediction.create_binary(
                question_id=1,
                probability=-0.1,
                confidence_level=0.8,
                confidence_basis="Test",
                method="test",
                reasoning="Test",
                created_by="test"
            )

        # Multiple choice with probabilities that don't sum to 1.0
        with pytest.raises(ValueError, match="Choice probabilities must sum to 1.0"):
            Prediction.create_multiple_choice(
                question_id=1,
                choice_probabilities={'A': 0.3, 'B': 0.3, 'C': 0.3},  # Sum = 0.9
                confidence_level=0.8,
                confidence_basis="Test",
                method="test",
                reasoning="Test",
                created_by="test"
            )

        # Multiple choice with negative probability
        with pytest.raises(ValueError, match="All choice probabilities must be non-negative"):
            Prediction.create_multiple_choice(
                question_id=1,
                choice_probabilities={'A': 0.5, 'B': -0.1, 'C': 0.6},
                confidence_level=0.8,
                confidence_basis="Test",
                method="test",
                reasoning="Test",
                created_by="test"
            )


class TestTournamentComprehensive:
    """Comprehensive tests for Tournament entity covering all scenarios."""

    def test_tournament_question_management_edge_cases(self, test_data_factory):
        """Test tournament question management edge cases."""
        tournament = test_data_factory.create_tournament()

        # Add question with deadline exactly at tournament end
        end_deadline_question = test_data_factory.create_question(
            question_id=999,
            deadline_offset_days=30  # Same as tournament end
        )
        end_deadline_question.deadline = tournament.end_date

        updated_tournament = tournament.add_question(end_deadline_question)
        assert len(updated_tournament.questions) == 4  # 3 original + 1 new

        # Try to add question with deadline 1 second after tournament end
        invalid_question = test_data_factory.create_question(question_id=1000)
        invalid_question.deadline = tournament.end_date + timedelta(seconds=1)

        with pytest.raises(ValueError, match="Question deadline .* is after tournament end"):
            tournament.add_question(invalid_question)

    def test_tournament_standings_edge_cases(self, test_data_factory):
        """Test tournament standings handling edge cases."""
        # Tournament with no participants
        empty_tournament = test_data_factory.create_tournament(current_standings={})
        assert empty_tournament.get_participant_count() == 0
        assert empty_tournament.get_top_participants(5) == []
        assert empty_tournament.get_participant_rank("anyone") is None

        # Tournament with tied scores
        tied_standings = {
            'agent1': 0.85,
            'agent2': 0.85,  # Tied for first
            'agent3': 0.70,
            'agent4': 0.70   # Tied for third
        }
        tied_tournament = test_data_factory.create_tournament(current_standings=tied_standings)

        # Both tied agents should get the same rank
        assert tied_tournament.get_participant_rank('agent1') == 1
        assert tied_tournament.get_participant_rank('agent2') == 1

        # Test top participants with ties
        top_participants = tied_tournament.get_top_participants(3)
        assert len(top_participants) == 3
        # First two should be the tied leaders
        assert top_participants[0][1] == 0.85
        assert top_participants[1][1] == 0.85

    def test_tournament_time_calculations_edge_cases(self, test_data_factory):
        """Test tournament time calculation edge cases."""
        # Tournament ending in exactly 1 hour
        now = datetime.utcnow()
        one_hour_tournament = test_data_factory.create_tournament(
            start_offset_days=-1,
            end_offset_days=0
        )
        one_hour_tournament.end_date = now + timedelta(hours=1)

        time_remaining = one_hour_tournament.time_remaining()
        assert 0.9 < time_remaining < 1.1  # Allow some tolerance

        # Tournament that ended exactly now
        just_ended_tournament = test_data_factory.create_tournament()
        just_ended_tournament.end_date = now
        assert just_ended_tournament.time_remaining() == 0.0

    def test_tournament_statistics_comprehensive(self, test_data_factory):
        """Test comprehensive tournament statistics."""
        # Create tournament with diverse questions
        questions = [
            test_data_factory.create_question(1, question_type=QuestionType.BINARY,
                                            category=QuestionCategory.AI_DEVELOPMENT, scoring_weight=1.0),
            test_data_factory.create_question(2, question_type=QuestionType.NUMERIC,
                                            category=QuestionCategory.TECHNOLOGY, scoring_weight=2.0),
            test_data_factory.create_question(3, question_type=QuestionType.MULTIPLE_CHOICE,
                                            category=QuestionCategory.SCIENCE, scoring_weight=3.0),
            # Resolved question (past deadline)
            test_data_factory.create_question(4, question_type=QuestionType.BINARY,
                                            category=QuestionCategory.ECONOMICS,
                                            deadline_offset_days=-1, scoring_weight=4.0),
            # Urgent question (deadline in 12 hours)
            test_data_factory.create_question(5, question_type=QuestionType.BINARY,
                                            category=QuestionCategory.POLITICS,
                                            deadline_offset_days=0, scoring_weight=1.0)
        ]
        questions[4].deadline = datetime.utcnow() + timedelta(hours=12)  # Urgent

        tournament = test_data_factory.create_tournament(questions=questions)
        stats = tournament.get_tournament_stats()

        assert stats['total_questions'] == 5
        assert stats['active_questions'] == 4  # All except the resolved one
        assert stats['resolved_questions'] == 1
        assert stats['high_value_questions'] == 3  # Weight >= 2.0
        assert stats['urgent_questions'] == 1  # Within 24 hours

        # Check distributions
        assert stats['question_types']['binary'] == 3
        assert stats['question_types']['numeric'] == 1
        assert stats['question_types']['multiple_choice'] == 1

        assert stats['question_categories']['ai_development'] == 1
        assert stats['question_categories']['technology'] == 1
        assert stats['question_categories']['science'] == 1
        assert stats['question_categories']['economics'] == 1
        assert stats['question_categories']['politics'] == 1

    def test_tournament_filtering_comprehensive(self, test_data_factory):
        """Test comprehensive question filtering."""
        # Create questions with various properties
        questions = []

        # High value active question
        questions.append(test_data_factory.create_question(
            1, scoring_weight=5.0, category=QuestionCategory.AI_DEVELOPMENT
        ))

        # Low value resolved question
        resolved_q = test_data_factory.create_question(
            2, scoring_weight=0.5, category=QuestionCategory.TECHNOLOGY
        )
        resolved_q.deadline = datetime.utcnow() - timedelta(days=1)
        questions.append(resolved_q)

        # Urgent medium value question
        urgent_q = test_data_factory.create_question(
            3, scoring_weight=2.0, category=QuestionCategory.SCIENCE
        )
        urgent_q.deadline = datetime.utcnow() + timedelta(hours=6)
        questions.append(urgent_q)

        tournament = test_data_factory.create_tournament(questions=questions)

        # Test various filters
        active = tournament.get_active_questions()
        assert len(active) == 2  # Questions 1 and 3

        resolved = tournament.get_resolved_questions()
        assert len(resolved) == 1  # Question 2

        high_value = tournament.get_high_value_questions(3.0)
        assert len(high_value) == 1  # Question 1

        urgent = tournament.get_urgent_questions(12.0)  # Within 12 hours
        assert len(urgent) == 1  # Question 3

        ai_questions = tournament.get_questions_by_category(QuestionCategory.AI_DEVELOPMENT)
        assert len(ai_questions) == 1  # Question 1


class TestAgentComprehensive:
    """Comprehensive tests for Agent entity."""

    def test_agent_performance_tracking(self, test_data_factory):
        """Test agent performance tracking functionality."""
        # Agent with detailed performance history
        detailed_history = {
            'accuracy': 0.85,
            'calibration': 0.78,
            'total_predictions': 150,
            'correct_predictions': 128,
            'average_confidence': 0.72,
            'brier_score': 0.15,
            'log_score': -0.45,
            'recent_performance': [0.8, 0.9, 0.7, 0.85, 0.88]
        }

        agent = test_data_factory.create_agent(
            agent_id="performance_agent",
            performance_history=detailed_history
        )

        assert agent.performance_history['accuracy'] == 0.85
        assert agent.performance_history['total_predictions'] == 150
        assert len(agent.performance_history['recent_performance']) == 5

    def test_agent_specialization_scoring(self, test_data_factory):
        """Test agent specialization scoring."""
        # Agent specialized in AI development
        ai_agent = test_data_factory.create_agent(
            agent_id="ai_specialist",
            knowledge_domains=['ai_development', 'machine_learning', 'deep_learning']
        )

        # Test specialization for matching domain
        ai_score = ai_agent.get_specialization_score('ai_development')
        assert ai_score > 0.5  # Should have high specialization

        # Test specialization for non-matching domain
        econ_score = ai_agent.get_specialization_score('economics')
        assert econ_score < 0.5  # Should have low specialization

    def test_agent_configuration_handling(self, test_data_factory):
        """Test agent configuration handling."""
        # Agent with complex configuration
        complex_config = {
            'model_parameters': {
                'temperature': 0.7,
                'max_tokens': 2000,
                'top_p': 0.9,
                'frequency_penalty': 0.1
            },
            'reasoning_config': {
                'max_steps': 10,
                'confidence_threshold': 0.8,
                'use_chain_of_thought': True
            },
            'search_config': {
                'max_sources': 5,
                'credibility_threshold': 0.7
            }
        }

        agent = test_data_factory.create_agent(
            agent_id="configured_agent",
            configuration=complex_config
        )

        assert agent.configuration['model_parameters']['temperature'] == 0.7
        assert agent.configuration['reasoning_config']['max_steps'] == 10
        assert agent.configuration['search_config']['max_sources'] == 5

    def test_agent_version_and_lifecycle(self, test_data_factory):
        """Test agent version and lifecycle management."""
        # Test agent creation timestamp
        agent = test_data_factory.create_agent()
        assert agent.created_at is not None
        assert isinstance(agent.created_at, datetime)

        # Test version handling
        versioned_agent = test_data_factory.create_agent(
            agent_id="versioned_agent",
            version="2.1.0"
        )
        assert versioned_agent.version == "2.1.0"

        # Test active/inactive status
        inactive_agent = test_data_factory.create_agent(
            agent_id="inactive_agent",
            is_active=False
        )
        assert not inactive_agent.is_active


class TestResearchReportComprehensive:
    """Comprehensive tests for ResearchReport entity."""

    def test_research_report_source_analysis(self, test_data_factory):
        """Test research report source analysis functionality."""
        # Report with varied source credibility
        sources = [
            'https://arxiv.org/paper1',  # High credibility
            'https://news.example.com/article',  # Medium credibility
            'https://blog.random.com/post',  # Low credibility
            'https://nature.com/article',  # High credibility
            'https://reddit.com/comment'  # Low credibility
        ]

        credibility_scores = {
            sources[0]: 0.95,  # arXiv paper
            sources[1]: 0.65,  # News article
            sources[2]: 0.30,  # Blog post
            sources[3]: 0.90,  # Nature article
            sources[4]: 0.25   # Reddit comment
        }

        report = test_data_factory.create_research_report(
            sources=sources,
            credibility_scores=credibility_scores
        )

        assert len(report.sources) == 5
        assert report.credibility_scores[sources[0]] == 0.95
        assert report.credibility_scores[sources[4]] == 0.25

    def test_research_report_quality_assessment(self, test_data_factory):
        """Test research report quality assessment."""
        # High quality report
        high_quality_report = test_data_factory.create_research_report(
            research_quality_score=0.95,
            knowledge_gaps=['minor_gap_1'],
            sources_count=10
        )
        assert high_quality_report.research_quality_score == 0.95
        assert len(high_quality_report.knowledge_gaps) == 1

        # Low quality report with many gaps
        low_quality_report = test_data_factory.create_research_report(
            research_quality_score=0.45,
            knowledge_gaps=['major_gap_1', 'major_gap_2', 'major_gap_3', 'data_missing'],
            sources_count=2
        )
        assert low_quality_report.research_quality_score == 0.45
        assert len(low_quality_report.knowledge_gaps) == 4

    def test_research_report_base_rates_handling(self, test_data_factory):
        """Test base rates handling in research reports."""
        # Report with comprehensive base rates
        comprehensive_base_rates = {
            'historical_accuracy': 0.72,
            'reference_class_size': 250,
            'similar_questions_resolved': 45,
            'expert_consensus_rate': 0.68,
            'market_prediction_accuracy': 0.75,
            'time_series_trend': 'increasing',
            'seasonal_patterns': {'Q1': 0.65, 'Q2': 0.70, 'Q3': 0.75, 'Q4': 0.68}
        }

        report = test_data_factory.create_research_report(
            base_rates=comprehensive_base_rates
        )

        assert report.base_rates['historical_accuracy'] == 0.72
        assert report.base_rates['reference_class_size'] == 250
        assert 'seasonal_patterns' in report.base_rates

    def test_research_report_evidence_synthesis(self, test_data_factory):
        """Test evidence synthesis in research reports."""
        # Report with detailed evidence synthesis
        detailed_synthesis = """
        Based on analysis of 10 high-quality sources, the evidence suggests:

        1. Strong consensus among experts (8/10 sources agree)
        2. Historical precedent supports the prediction (reference class accuracy: 72%)
        3. Recent developments indicate accelerating trend
        4. Potential confounding factors identified: regulatory changes, market volatility
        5. Confidence level: High (0.85) based on source quality and consensus

        Key uncertainties:
        - Timeline assumptions may be optimistic
        - External factors not fully accounted for
        """

        report = test_data_factory.create_research_report(
            evidence_synthesis=detailed_synthesis
        )

        assert "Strong consensus" in report.evidence_synthesis
        assert "Historical precedent" in report.evidence_synthesis
        assert "Key uncertainties" in report.evidence_synthesis
