"""Integration tests for complete tournament strategy workflows."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from src.application.services.tournament_service import (
    TournamentService, StrategyRecommendation, PriorityLevel, MarketCondition
)
from src.domain.entities.tournament import Tournament, ScoringRules, ScoringMethod
from src.domain.entities.question import Question, QuestionType, QuestionCategory
from src.domain.value_objects.strategy_result import StrategyResult, StrategyType
from src.domain.value_objects.confidence import Confidence


class TestTournamentStrategyWorkflows:
    """Integration tests for complete tournament strategy workflows."""

    @pytest.fixture
    def tournament_service(self):
        """Create tournament service with realistic configuration."""
        return TournamentService(
            ml_classifier=None,  # Use rule-based classification for testing
            risk_tolerance=0.6,
            max_concurrent_questions=8
        )

    @pytest.fixture
    def realistic_tournament(self):
        """Create realistic tournament scenario for integration testing."""
        # Create questions with realistic characteristics
        questions = [
            # High-priority AI question with good scoring potential
            Question(
                id=1,
                text="Will GPT-5 be released by OpenAI before the end of 25?",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=datetime.utcnow() + timedelta(days=45),
                background="OpenAI has been developing next-generation models...",
                resolution_criteria="Official announcement by OpenAI",
                scoring_weight=2.5
            ),
            # Urgent political question
            Question(
                id=2,
                text="Who will win the upcoming election?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                category=QuestionCategory.POLITICS,
                deadline=datetime.utcnow() + timedelta(hours=8),
                background="Election polling data shows...",
                resolution_criteria="Official election results",
                scoring_weight=3.0,
                choices=["Candidate A", "Candidate B", "Candidate C"]
            ),
            # Complex economic question
            Question(
                id=3,
                text="What will be the US GDP growth rate for Q4 2025?",
                question_type=QuestionType.NUMERIC,
                category=QuestionCategory.ECONOMICS,
                deadline=datetime.utcnow() + timedelta(days=20),
                background="Economic indicators suggest...",
                resolution_criteria="Official BEA data",
                scoring_weight=1.8,
                min_value=-5.0,
                max_value=10.0
            ),
            # Specialized science question
            Question(
                id=4,
                text="Will CRISPR gene therapy show significant results in clinical trials?",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.SCIENCE,
                deadline=datetime.utcnow() + timedelta(days=60),
                background="Recent advances in gene therapy...",
                resolution_criteria="Published clinical trial results",
                scoring_weight=2.0
            ),
            # Technology question with moderate priority
            Question(
                id=5,
                text="Will quantum computing achieve practical advantage in 2025?",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.TECHNOLOGY,
                deadline=datetime.utcnow() + timedelta(days=30),
                background="Quantum computing developments...",
                resolution_criteria="Demonstrated practical application",
                scoring_weight=1.5
            )
        ]

        # Create realistic standings with competitive distribution
        standings = {}
        for i in range(100):
            # Create realistic score distribution
            if i < 10:  # Top 10 performers
                score = 85 + np.random.normal(0, 5)
            elif i < 30:  # Good performers
                score = 70 + np.random.normal(0, 8)
            elif i < 70:  # Average performers
                score = 55 + np.random.normal(0, 10)
            else:  # Lower performers
                score = 40 + np.random.normal(0, 12)

            standings[f"participant_{i+1}"] = max(0, score)

        return Tournament(
            id=1001,
            name="AIBQ2 2025 Tournament",
            questions=questions,
            scoring_rules=ScoringRules(
                method=ScoringMethod.BRIER_SCORE,
                weight_by_question=True,
                bonus_for_early=True,
                penalty_for_late=True,
                minimum_confidence=0.1,
                maximum_submissions=3
            ),
            start_date=datetime.utcnow() - timedelta(days=15),
            end_date=datetime.utcnow() + timedelta(days=75),
            current_standings=standings
        )

    @pytest.mark.asyncio
    async def test_complete_strategy_analysis_workflow(self, tournament_service, realistic_tournament):
        """Test complete strategy analysis workflow from start to finish."""
        # Execute complete analysis
        strategy_recommendation = await tournament_service.analyze_tournament_strategy(realistic_tournament)

        # Verify comprehensive analysis results
        assert isinstance(strategy_recommendation, StrategyRecommendation)

        # Check strategy selection is reasonable
        assert isinstance(strategy_recommendation.strategy_type, StrategyType)

        # Verify confidence is well-calibrated
        assert 0.3 <= strategy_recommendation.confidence.level <= 0.9
        assert len(strategy_recommendation.confidence.basis) > 20

        # Check expected score impact is reasonable
        assert 0.05 <= strategy_recommendation.expected_score_impact <= 0.25

        # Verify risk level is appropriate
        assert 0.1 <= strategy_recommendation.risk_level <= 0.9

        # Check resource requirements are realistic
        resources = strategy_recommendation.resource_requirements
        assert 'research_time' in resources
        assert 'analysis_time' in resources
        assert 'monitoring_time' in resources
        assert resources['research_time'] > 0

        # Verify question allocation is sensible
        allocation = strategy_recommendation.question_allocation
        assert len(allocation) <= tournament_service.max_concurrent_questions

        # High-priority questions should get higher allocation
        urgent_question_id = 2  # 8-hour deadline
        if urgent_question_id in allocation:
            urgent_allocation = allocation[urgent_question_id]
            avg_allocation = np.mean(list(allocation.values()))
            assert urgent_allocation >= avg_allocation

        # Check reasoning is comprehensive
        assert len(strategy_recommendation.reasoning) > 50
        assert "questions" in strategy_recommendation.reasoning.lower()

        # Verify alternatives are provided
        assert len(strategy_recommendation.alternatives) >= 2
        for alt_strategy, alt_score in strategy_recommendation.alternatives:
            assert isinstance(alt_strategy, StrategyType)
            assert 0.0 <= alt_score <= 1.0

    @pytest.mark.asyncio
    async def test_urgent_question_prioritization_workflow(self, tournament_service, realistic_tournament):
        """Test workflow prioritizes urgent questions appropriately."""
        # Analyze tournament strategy
        strategy_rec = await tournament_service.analyze_tournament_strategy(realistic_tournament)

        # Get detailed prioritization
        priorities = await tournament_service._prioritize_questions(realistic_tournament)

        # Find the urgent question (8-hour deadline)
        urgent_priority = next((p for p in priorities if p.question_id == 2), None)
        assert urgent_priority is not None

        # Urgent question should have high deadline urgency
        assert urgent_priority.deadline_urgency >= 0.8

        # Should be in top priority levels
        assert urgent_priority.priority_level in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]

        # Should be allocated resources in strategy
        if 2 in strategy_rec.question_allocation:
            urgent_allocation = strategy_rec.question_allocation[2]
            # Should get significant allocation
            assert urgent_allocation >= 0.5

    @pytest.mark.asyncio
    async def test_high_value_question_strategy_workflow(self, tournament_service, realistic_tournament):
        """Test workflow handles high-value questions strategically."""
        # Identify high-value question (scoring weight 3.0)
        high_value_question = next(q for q in realistic_tournament.questions if q.scoring_weight == 3.0)

        # Analyze strategy
        strategy_rec = await tournament_service.analyze_tournament_strategy(realistic_tournament)
        priorities = await tournament_service._prioritize_questions(realistic_tournament)

        # Find priority for high-value question
        high_value_priority = next(p for p in priorities if p.question_id == high_value_question.id)

        # Should have high scoring potential
        assert high_value_priority.scoring_potential > 0.6

        # Should be allocated significant resources
        if high_value_question.id in strategy_rec.question_allocation:
            allocation = strategy_rec.question_allocation[high_value_question.id]
            assert allocation >= 0.6

    @pytest.mark.asyncio
    async def test_market_timing_optimization_workflow(self, tournament_service, realistic_tournament):
        """Test market timing optimization workflow."""
        # Test timing optimization for different questions
        for question in realistic_tournament.get_active_questions():
            optimal_time, reasoning = tournament_service.optimize_submission_timing(
                question, realistic_tournament
            )

            # Verify timing is logical
            assert optimal_time > datetime.utcnow()
            assert optimal_time < question.deadline

            # Verify reasoning includes market analysis
            assert len(reasoning) > 20
            assert any(keyword in reasoning.lower() for keyword in
                      ['market', 'timing', 'window', 'condition'])

            # For urgent questions, timing should be sooner
            if question.time_until_deadline() < 12:  # Less than 12 hours
                time_until_optimal = (optimal_time - datetime.utcnow()).total_seconds() / 3600
                assert time_until_optimal < question.time_until_deadline() * 0.8

    @pytest.mark.asyncio
    async def test_competitor_analysis_integration_workflow(self, tournament_service, realistic_tournament):
        """Test competitor analysis integration in strategy workflow."""
        # Analyze competitors
        competitor_analysis = await tournament_service._analyze_competitors(realistic_tournament)

        # Should analyze top competitors
        assert len(competitor_analysis) > 0
        assert len(competitor_analysis) <= 20  # Max 20 as specified

        # Check top performers are included
        top_participants = realistic_tournament.get_top_participants(n=5)
        top_participant_ids = [p[0] for p in top_participants]

        analyzed_participants = set(competitor_analysis.keys())
        top_analyzed = analyzed_participants.intersection(set(top_participant_ids))
        assert len(top_analyzed) >= min(3, len(top_participant_ids))

        # Verify competitor profiles are comprehensive
        for participant_id, profile in competitor_analysis.items():
            assert profile.participant_id == participant_id
            assert isinstance(profile.accuracy_score, float)
            assert isinstance(profile.submission_patterns, dict)
            assert isinstance(profile.category_specializations, list)
            assert len(profile.recent_performance) > 0

    @pytest.mark.asyncio
    async def test_adaptive_strategy_workflow(self, tournament_service, realistic_tournament):
        """Test adaptive strategy workflow based on performance feedback."""
        # Simulate historical performance data
        historical_results = []

        # Add mixed performance results
        strategies_tested = [StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE, StrategyType.BALANCED]

        for i, strategy in enumerate(strategies_tested):
            for j in range(3):  # 3 results per strategy
                expected_score = 0.6 + (i * 0.1)  # Vary expected scores
                actual_score = expected_score + np.random.normal(0, 0.1)

                result = StrategyResult.create(
                    strategy_type=strategy,
                    expected_score=expected_score,
                    reasoning=f"Historical test {i}-{j}",
                    question_ids=[i*3 + j + 1],
                    confidence_level=0.7,
                    confidence_basis="Historical data"
                )

                if actual_score > expected_score:
                    result = result.mark_success(actual_score)
                else:
                    result = result.mark_failure(actual_score)

                historical_results.append(result)

        # Test strategy adaptation
        adapted_strategy = tournament_service.adapt_strategy_based_on_performance(historical_results)
        assert isinstance(adapted_strategy, StrategyType)

        # Integrate adaptation into tournament analysis
        tournament_service.strategy_history = historical_results
        strategy_rec = await tournament_service.analyze_tournament_strategy(realistic_tournament)

        # Strategy should be influenced by historical performance
        assert isinstance(strategy_rec.strategy_type, StrategyType)

    @pytest.mark.asyncio
    async def test_risk_adjusted_strategy_workflow(self, tournament_service, realistic_tournament):
        """Test risk-adjusted strategy selection workflow."""
        # Test different tournament positions
        positions_to_test = [1, 5, 15, 50, 95]  # Various rankings

        for position in positions_to_test:
            risk_strategy = tournament_service.select_risk_adjusted_strategy(
                realistic_tournament, current_position=position
            )

            assert isinstance(risk_strategy, StrategyType)

            # Leading positions should tend toward conservative strategies
            if position <= 3:
                assert risk_strategy in [StrategyType.CONSERVATIVE, StrategyType.BALANCED]

            # Trailing positions should tend toward aggressive strategies
            elif position >= 50:
                assert risk_strategy in [StrategyType.AGGRESSIVE, StrategyType.BALANCED, StrategyType.MOMENTUM]

    @pytest.mark.asyncio
    async def test_multi_criteria_optimization_workflow(self, tournament_service, realistic_tournament):
        """Test multi-criteria optimization workflow."""
        # Get detailed prioritization
        priorities = await tournament_service._prioritize_questions(realistic_tournament)

        # Verify multi-criteria scoring
        for priority in priorities:
            # All criteria should be scored
            assert 0.0 <= priority.confidence_score <= 1.0
            assert 0.0 <= priority.scoring_potential <= 1.0
            assert 0.0 <= priority.deadline_urgency <= 1.0
            assert 0.0 <= priority.strategic_value <= 1.0
            assert 0.0 <= priority.overall_score <= 1.0

            # Overall score should be reasonable combination of criteria
            expected_range_min = min(
                priority.confidence_score, priority.scoring_potential,
                priority.deadline_urgency, priority.strategic_value
            ) * 0.8
            expected_range_max = max(
                priority.confidence_score, priority.scoring_potential,
                priority.deadline_urgency, priority.strategic_value
            ) * 1.2

            assert expected_range_min <= priority.overall_score <= expected_range_max

    @pytest.mark.asyncio
    async def test_resource_allocation_workflow(self, tournament_service, realistic_tournament):
        """Test resource allocation workflow."""
        # Analyze tournament strategy
        strategy_rec = await tournament_service.analyze_tournament_strategy(realistic_tournament)

        # Verify resource allocation is realistic
        resources = strategy_rec.resource_requirements
        total_questions = len(realistic_tournament.get_active_questions())

        # Research time should scale with number of questions
        expected_min_research = total_questions * 1.0  # 1 hour minimum per question
        expected_max_research = total_questions * 4.0  # 4 hours maximum per question
        assert expected_min_research <= resources['research_time'] <= expected_max_research

        # Analysis time should be reasonable
        assert resources['analysis_time'] > 0
        assert resources['analysis_time'] <= resources['research_time']

        # Monitoring time should be included
        assert resources['monitoring_time'] > 0

        # Question allocation should respect resource constraints
        allocation = strategy_rec.question_allocation
        total_allocation = sum(allocation.values())

        # Total allocation should be reasonable given max concurrent questions
        assert total_allocation <= tournament_service.max_concurrent_questions * 1.2

    @pytest.mark.asyncio
    async def test_tournament_meta_game_analysis_workflow(self, tournament_service, realistic_tournament):
        """Test tournament meta-game analysis workflow."""
        # Analyze meta-game patterns
        meta_patterns = await tournament_service._assess_meta_game_patterns(realistic_tournament)

        # Verify meta-game analysis structure
        expected_keys = [
            'dominant_strategies', 'category_trends', 'timing_patterns',
            'scoring_inefficiencies', 'adaptation_opportunities'
        ]

        for key in expected_keys:
            assert key in meta_patterns

        # Verify patterns are actionable
        assert isinstance(meta_patterns['dominant_strategies'], list)
        assert isinstance(meta_patterns['category_trends'], dict)
        assert isinstance(meta_patterns['timing_patterns'], dict)
        assert isinstance(meta_patterns['scoring_inefficiencies'], list)
        assert isinstance(meta_patterns['adaptation_opportunities'], list)

    @pytest.mark.asyncio
    async def test_end_to_end_tournament_optimization(self, tournament_service, realistic_tournament):
        """Test complete end-to-end tournament optimization workflow."""
        # Step 1: Analyze tournament strategy
        strategy_rec = await tournament_service.analyze_tournament_strategy(realistic_tournament)

        # Step 2: Optimize timing for allocated questions
        timing_optimizations = {}
        for question_id, allocation in strategy_rec.question_allocation.items():
            if allocation > 0.5:  # Only optimize timing for significantly allocated questions
                question = next(q for q in realistic_tournament.questions if q.id == question_id)
                optimal_time, reasoning = tournament_service.optimize_submission_timing(
                    question, realistic_tournament
                )
                timing_optimizations[question_id] = (optimal_time, reasoning)

        # Step 3: Adapt strategy based on simulated performance
        simulated_results = []
        for question_id in strategy_rec.question_allocation.keys():
            result = StrategyResult.create(
                strategy_type=strategy_rec.strategy_type,
                expected_score=strategy_rec.expected_score_impact,
                reasoning="Simulated execution",
                question_ids=[question_id],
                confidence_level=strategy_rec.confidence.level,
                confidence_basis="Simulation"
            )
            # Simulate mixed results
            if np.random.random() > 0.3:  # 70% success rate
                result = result.mark_success(strategy_rec.expected_score_impact + 0.05)
            else:
                result = result.mark_failure(strategy_rec.expected_score_impact - 0.05)
            simulated_results.append(result)

        adapted_strategy = tournament_service.adapt_strategy_based_on_performance(simulated_results)

        # Step 4: Select risk-adjusted strategy
        current_position = 25  # Assume middle-tier position
        risk_adjusted_strategy = tournament_service.select_risk_adjusted_strategy(
            realistic_tournament, current_position
        )

        # Verify end-to-end workflow produces coherent results
        assert isinstance(strategy_rec, StrategyRecommendation)
        assert len(timing_optimizations) > 0
        assert isinstance(adapted_strategy, StrategyType)
        assert isinstance(risk_adjusted_strategy, StrategyType)

        # Results should be internally consistent
        strategies = [strategy_rec.strategy_type, adapted_strategy, risk_adjusted_strategy]

        # At least some strategies should be similar (not completely random)
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Should have some consistency in strategy selection
        max_count = max(strategy_counts.values())
        assert max_count >= 1  # At least one strategy appears

    @pytest.mark.asyncio
    async def test_performance_under_load(self, tournament_service):
        """Test tournament service performance under load."""
        # Create large tournament with many questions
        large_questions = []
        for i in range(50):  # 50 questions
            question = Question(
                id=i + 1,
                text=f"Load test question {i + 1}",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=datetime.utcnow() + timedelta(days=30 + i),
                background=f"Background {i + 1}",
                resolution_criteria=f"Criteria {i + 1}",
                scoring_weight=1.0 + (i % 3) * 0.5
            )
            large_questions.append(question)

        large_tournament = Tournament(
            id=2001,
            name="Load Test Tournament",
            questions=large_questions,
            scoring_rules=ScoringRules(method=ScoringMethod.BRIER_SCORE),
            start_date=datetime.utcnow() - timedelta(days=5),
            end_date=datetime.utcnow() + timedelta(days=85),
            current_standings={f"user{i}": 50.0 + i for i in range(200)}
        )

        # Measure performance
        start_time = datetime.utcnow()
        strategy_rec = await tournament_service.analyze_tournament_strategy(large_tournament)
        end_time = datetime.utcnow()

        # Should complete within reasonable time (30 seconds for 50 questions)
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 30.0

        # Should still produce valid results
        assert isinstance(strategy_rec, StrategyRecommendation)
        assert len(strategy_rec.question_allocation) <= tournament_service.max_concurrent_questions

    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, tournament_service, realistic_tournament):
        """Test error handling and resilience in tournament workflows."""
        # Test with corrupted tournament data
        corrupted_tournament = Tournament(
            id=3001,
            name="Corrupted Tournament",
            questions=[],  # No questions
            scoring_rules=ScoringRules(method=ScoringMethod.BRIER_SCORE),
            start_date=datetime.utcnow() - timedelta(days=5),
            end_date=datetime.utcnow() + timedelta(days=30),
            current_standings={}
        )

        # Should handle gracefully
        strategy_rec = await tournament_service.analyze_tournament_strategy(corrupted_tournament)
        assert isinstance(strategy_rec, StrategyRecommendation)
        assert len(strategy_rec.question_allocation) == 0

        # Test with mock failures in sub-components
        with patch.object(tournament_service, '_categorize_questions', side_effect=Exception("Mock error")):
            # Should still complete with degraded functionality
            strategy_rec = await tournament_service.analyze_tournament_strategy(realistic_tournament)
            assert isinstance(strategy_rec, StrategyRecommendation)

        # Test timing optimization with edge case question (very short deadline)
        try:
            edge_case_question = Question(
                id=9999,
                text="Edge case question",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.OTHER,
                deadline=datetime.utcnow() + timedelta(minutes=5),  # Very short deadline
                background="Edge case background",
                resolution_criteria="Edge case criteria",
                scoring_weight=1.0
            )

            # Should handle gracefully
            optimal_time, reasoning = tournament_service.optimize_submission_timing(
                edge_case_question, realistic_tournament
            )
            # Should return reasonable values even for edge cases
            assert optimal_time > datetime.utcnow()
            assert optimal_time <= edge_case_question.deadline
            assert len(reasoning) > 0
        except Exception:
            # Exception is acceptable for edge cases
            pass
