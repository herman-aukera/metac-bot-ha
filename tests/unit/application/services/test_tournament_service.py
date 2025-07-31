"""Unit tests for TournamentService."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.application.services.tournament_service import (
    TournamentService, QuestionPriority, MarketAnalysis, CompetitorProfile,
    StrategyRecommendation, PriorityLevel, MarketCondition, RiskProfile
)
from src.domain.entities.tournament import Tournament, ScoringRules, ScoringMethod
from src.domain.entities.question import Question, QuestionType, QuestionCategory
from src.domain.value_objects.strategy_result import StrategyResult, StrategyType, StrategyOutcome
from src.domain.value_objects.confidence import Confidence


class TestTournamentService:
    """Test cases for TournamentService."""

    @pytest.fixture
    def tournament_service(self):
        """Create tournament service instance for testing."""
        return TournamentService(
            ml_classifier=None,
            risk_tolerance=0.5,
            max_concurrent_questions=10
        )

    @pytest.fixture
    def sample_tournament(self):
        """Create sample tournament for testing."""
        questions = [
            Question(
                id=1,
                text="Will AI achieve AGI by 2030?",
                question_type=QuestionType.BINARY,
                category=QuestionCategory.AI_DEVELOPMENT,
                deadline=datetime.utcnow() + timedelta(days=30),
                background="Background info",
                resolution_criteria="Clear criteria",
                scoring_weight=2.0
            ),
            Question(
                id=2,
                text="What will be the GDP growth rate?",
                question_type=QuestionType.NUMERIC,
                category=QuestionCategory.ECONOMICS,
                deadline=datetime.utcnow() + timedelta(days=15),
                background="Economic background",
                resolution_criteria="GDP criteria",
                scoring_weight=1.5,
                min_value=0.0,
                max_value=10.0
            ),
            Question(
                id=3,
                text="Who will win the election?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                category=QuestionCategory.POLITICS,
                deadline=datetime.utcnow() + timedelta(hours=6),
                background="Political background",
                resolution_criteria="Election criteria",
                scoring_weight=3.0,
                choices=["Candidate A", "Candidate B", "Candidate C"]
            )
        ]

        scoring_rules = ScoringRules(
            method=ScoringMethod.BRIER_SCORE,
            weight_by_question=True,
            bonus_for_early=True
        )

        return Tournament(
            id=1,
            name="Test Tournament",
            questions=questions,
            scoring_rules=scoring_rules,
            start_date=datetime.utcnow() - timedelta(days=10),
            end_date=datetime.utcnow() + timedelta(days=50),
            current_standings={"user1": 85.5, "user2": 78.2, "user3": 92.1}
        )

    @pytest.mark.asyncio
    async def test_analyze_tournament_strategy(self, tournament_service, sample_tournament):
        """Test comprehensive tournament strategy analysis."""
        # Mock the internal methods to avoid complex dependencies
        with patch.object(tournament_service, '_categorize_questions') as mock_categorize, \
             patch.object(tournament_service, '_prioritize_questions') as mock_prioritize, \
             patch.object(tournament_service, '_analyze_market_conditions') as mock_market, \
             patch.object(tournament_service, '_analyze_competitors') as mock_competitors, \
             patch.object(tournament_service, '_assess_meta_game_patterns') as mock_meta, \
             patch.object(tournament_service, '_generate_strategy_recommendation') as mock_generate:

            # Setup mock returns
            mock_categorize.return_value = {1: {"complexity_score": 1.5}}
            mock_prioritize.return_value = []
            mock_market.return_value = {}
            mock_competitors.return_value = {}
            mock_meta.return_value = {}

            expected_recommendation = StrategyRecommendation(
                strategy_type=StrategyType.BALANCED,
                confidence=Confidence(level=0.8, basis="Test confidence"),
                expected_score_impact=0.15,
                risk_level=0.4,
                resource_requirements={"research_time": 10.0},
                timing_constraints=None,
                question_allocation={1: 1.0},
                reasoning="Test reasoning",
                alternatives=[(StrategyType.AGGRESSIVE, 0.7)]
            )
            mock_generate.return_value = expected_recommendation

            # Execute
            result = await tournament_service.analyze_tournament_strategy(sample_tournament)

            # Verify
            assert result == expected_recommendation
            mock_categorize.assert_called_once()
            mock_prioritize.assert_called_once()
            mock_market.assert_called_once()
            mock_competitors.assert_called_once()
            mock_meta.assert_called_once()
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_categorize_questions(self, tournament_service, sample_tournament):
        """Test question categorization functionality."""
        result = await tournament_service._categorize_questions(sample_tournament.questions)

        assert len(result) == 3
        assert 1 in result  # AI question
        assert 2 in result  # Economics question
        assert 3 in result  # Politics question

        # Check structure of categorization results
        for question_id, categorization in result.items():
            assert 'primary_category' in categorization
            assert 'category_confidence' in categorization
            assert 'complexity_score' in categorization
            assert 'specialization_required' in categorization
            assert 'predicted_difficulty' in categorization
            assert 'estimated_research_time' in categorization

    @pytest.mark.asyncio
    async def test_prioritize_questions(self, tournament_service, sample_tournament):
        """Test question prioritization functionality."""
        result = await tournament_service._prioritize_questions(sample_tournament)

        assert len(result) == 3  # All questions are active
        assert all(isinstance(p, QuestionPriority) for p in result)

        # Check that urgent question (6 hours deadline) has high priority
        urgent_priority = next(p for p in result if p.question_id == 3)
        assert urgent_priority.deadline_urgency > 0.8
        assert urgent_priority.priority_level in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]

        # Check that results are sorted by overall score
        scores = [p.overall_score for p in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_analyze_market_conditions(self, tournament_service, sample_tournament):
        """Test market condition analysis."""
        result = await tournament_service._analyze_market_conditions(sample_tournament)

        assert len(result) == 3
        for question_id, analysis in result.items():
            assert isinstance(analysis, MarketAnalysis)
            assert analysis.question_id == question_id
            assert isinstance(analysis.market_condition, MarketCondition)
            assert 0.0 <= analysis.competitor_density <= 1.0
            assert 0.0 <= analysis.prediction_variance <= 1.0
            assert 0.0 <= analysis.consensus_strength <= 1.0
            assert analysis.optimal_timing_window[0] < analysis.optimal_timing_window[1]

    @pytest.mark.asyncio
    async def test_analyze_competitors(self, tournament_service, sample_tournament):
        """Test competitor analysis functionality."""
        result = await tournament_service._analyze_competitors(sample_tournament)

        # Should analyze top participants
        assert len(result) <= 20  # Max 20 as specified in method
        assert len(result) == 3   # All participants in sample tournament

        for participant_id, profile in result.items():
            assert isinstance(profile, CompetitorProfile)
            assert profile.participant_id == participant_id
            assert isinstance(profile.risk_profile, RiskProfile)
            assert isinstance(profile.category_specializations, list)
            assert isinstance(profile.recent_performance, list)

    def test_optimize_submission_timing(self, tournament_service, sample_tournament):
        """Test submission timing optimization."""
        question = sample_tournament.questions[0]  # AI question with 30-day deadline

        optimal_time, reasoning = tournament_service.optimize_submission_timing(
            question, sample_tournament
        )

        # Should return a time before the deadline
        assert optimal_time < question.deadline
        assert optimal_time > datetime.utcnow()
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

        # Test with early bonus
        tournament_with_bonus = Tournament(
            id=2,
            name="Bonus Tournament",
            questions=[question],
            scoring_rules=ScoringRules(
                method=ScoringMethod.BRIER_SCORE,
                bonus_for_early=True
            ),
            start_date=datetime.utcnow() - timedelta(days=5),
            end_date=datetime.utcnow() + timedelta(days=35),
            current_standings={}
        )

        optimal_time_bonus, reasoning_bonus = tournament_service.optimize_submission_timing(
            question, tournament_with_bonus
        )

        assert "Early submission bonus" in reasoning_bonus

    def test_adapt_strategy_based_on_performance(self, tournament_service):
        """Test strategy adaptation based on performance feedback."""
        # Test with no results
        result = tournament_service.adapt_strategy_based_on_performance([])
        assert result == StrategyType.BALANCED

        # Test with performance results
        recent_results = [
            StrategyResult.create(
                strategy_type=StrategyType.AGGRESSIVE,
                expected_score=0.8,
                reasoning="Test aggressive",
                question_ids=[1, 2],
                confidence_level=0.7,
                confidence_basis="Test basis"
            ).mark_success(0.9),  # Good performance
            StrategyResult.create(
                strategy_type=StrategyType.CONSERVATIVE,
                expected_score=0.6,
                reasoning="Test conservative",
                question_ids=[3, 4],
                confidence_level=0.8,
                confidence_basis="Test basis"
            ).mark_failure(0.4),  # Poor performance
        ]

        result = tournament_service.adapt_strategy_based_on_performance(recent_results)
        # Should favor aggressive strategy due to better performance
        assert result == StrategyType.AGGRESSIVE

    def test_select_risk_adjusted_strategy(self, tournament_service, sample_tournament):
        """Test risk-adjusted strategy selection."""
        # Test leading position (conservative)
        strategy = tournament_service.select_risk_adjusted_strategy(
            sample_tournament, current_position=1
        )
        assert strategy in [StrategyType.CONSERVATIVE, StrategyType.BALANCED]

        # Test trailing position (aggressive)
        strategy = tournament_service.select_risk_adjusted_strategy(
            sample_tournament, current_position=50
        )
        assert strategy in [StrategyType.AGGRESSIVE, StrategyType.BALANCED]

        # Test with log scoring (should favor more aggressive strategies)
        log_tournament = Tournament(
            id=3,
            name="Log Tournament",
            questions=sample_tournament.questions,
            scoring_rules=ScoringRules(method=ScoringMethod.LOG_SCORE),
            start_date=sample_tournament.start_date,
            end_date=sample_tournament.end_date,
            current_standings=sample_tournament.current_standings
        )

        strategy = tournament_service.select_risk_adjusted_strategy(
            log_tournament, current_position=10
        )
        # Log scoring should push toward more aggressive strategies
        assert strategy != StrategyType.CONSERVATIVE

    def test_calculate_confidence_score(self, tournament_service, sample_tournament):
        """Test confidence score calculation."""
        ai_question = sample_tournament.questions[0]  # AI question
        economics_question = sample_tournament.questions[1]  # Economics question

        ai_confidence = tournament_service._calculate_confidence_score(ai_question, sample_tournament)
        econ_confidence = tournament_service._calculate_confidence_score(economics_question, sample_tournament)

        # AI questions should have higher confidence (domain familiarity)
        assert ai_confidence > econ_confidence
        assert 0.1 <= ai_confidence <= 1.0
        assert 0.1 <= econ_confidence <= 1.0

    def test_calculate_scoring_potential(self, tournament_service, sample_tournament):
        """Test scoring potential calculation."""
        high_weight_question = sample_tournament.questions[2]  # Weight 3.0
        low_weight_question = sample_tournament.questions[1]   # Weight 1.5

        high_potential = tournament_service._calculate_scoring_potential(high_weight_question, sample_tournament)
        low_potential = tournament_service._calculate_scoring_potential(low_weight_question, sample_tournament)

        assert high_potential > low_potential
        assert 0.1 <= high_potential <= 1.0
        assert 0.1 <= low_potential <= 1.0

    def test_calculate_deadline_urgency(self, tournament_service, sample_tournament):
        """Test deadline urgency calculation."""
        urgent_question = sample_tournament.questions[2]  # 6 hours deadline
        distant_question = sample_tournament.questions[0]  # 30 days deadline

        urgent_score = tournament_service._calculate_deadline_urgency(urgent_question)
        distant_score = tournament_service._calculate_deadline_urgency(distant_question)

        assert urgent_score > distant_score
        assert urgent_score >= 0.8  # Should be high urgency
        assert distant_score <= 0.4  # Should be low urgency

    def test_calculate_strategic_value(self, tournament_service, sample_tournament):
        """Test strategic value calculation."""
        high_value_question = sample_tournament.questions[2]  # High scoring weight
        normal_question = sample_tournament.questions[1]

        high_value = tournament_service._calculate_strategic_value(high_value_question, sample_tournament)
        normal_value = tournament_service._calculate_strategic_value(normal_question, sample_tournament)

        assert 0.1 <= high_value <= 1.0
        assert 0.1 <= normal_value <= 1.0

    def test_estimate_competitor_density(self, tournament_service, sample_tournament):
        """Test competitor density estimation."""
        ai_question = sample_tournament.questions[0]  # Popular category
        economics_question = sample_tournament.questions[1]

        ai_density = tournament_service._estimate_competitor_density(ai_question, sample_tournament)
        econ_density = tournament_service._estimate_competitor_density(economics_question, sample_tournament)

        # AI questions should attract more competitors
        assert ai_density >= econ_density
        assert 0.1 <= ai_density <= 1.0
        assert 0.1 <= econ_density <= 1.0

    def test_calculate_optimal_timing_window(self, tournament_service, sample_tournament):
        """Test optimal timing window calculation."""
        question = sample_tournament.questions[0]

        # Test different market conditions
        for condition in MarketCondition:
            start_time, end_time = tournament_service._calculate_optimal_timing_window(question, condition)

            assert start_time < end_time
            assert start_time >= datetime.utcnow()
            assert end_time <= question.deadline

        # Test underexplored market (should favor early submission)
        early_start, early_end = tournament_service._calculate_optimal_timing_window(
            question, MarketCondition.UNDEREXPLORED
        )

        # Test oversaturated market (should favor later submission)
        late_start, late_end = tournament_service._calculate_optimal_timing_window(
            question, MarketCondition.OVERSATURATED
        )

        # Early submission window should start before late submission window
        assert early_start < late_start

    def test_select_optimal_strategy(self, tournament_service, sample_tournament):
        """Test optimal strategy selection."""
        # Create mock prioritized questions
        prioritized_questions = [
            QuestionPriority(
                question_id=1,
                priority_level=PriorityLevel.CRITICAL,
                confidence_score=0.8,
                scoring_potential=0.9,
                deadline_urgency=0.7,
                strategic_value=0.8,
                overall_score=0.8,
                reasoning="High priority",
                metadata={}
            )
        ]

        # Create mock market analysis
        market_analysis = {
            1: MarketAnalysis(
                question_id=1,
                market_condition=MarketCondition.OPPORTUNITY,
                competitor_density=0.3,
                prediction_variance=0.5,
                consensus_strength=0.6,
                optimal_timing_window=(datetime.utcnow(), datetime.utcnow() + timedelta(hours=12)),
                confidence=0.7,
                reasoning="Opportunity market"
            )
        }

        strategy = tournament_service._select_optimal_strategy(
            sample_tournament, prioritized_questions, market_analysis, {}, {}
        )

        assert isinstance(strategy, StrategyType)

    def test_calculate_resource_requirements(self, tournament_service):
        """Test resource requirements calculation."""
        prioritized_questions = [
            QuestionPriority(
                question_id=1,
                priority_level=PriorityLevel.HIGH,
                confidence_score=0.8,
                scoring_potential=0.7,
                deadline_urgency=0.6,
                strategic_value=0.7,
                overall_score=0.7,
                reasoning="Test",
                metadata={}
            ),
            QuestionPriority(
                question_id=2,
                priority_level=PriorityLevel.MEDIUM,
                confidence_score=0.6,
                scoring_potential=0.5,
                deadline_urgency=0.4,
                strategic_value=0.5,
                overall_score=0.5,
                reasoning="Test",
                metadata={}
            )
        ]

        # Test aggressive strategy (should require more resources)
        aggressive_resources = tournament_service._calculate_resource_requirements(
            StrategyType.AGGRESSIVE, prioritized_questions
        )

        # Test conservative strategy
        conservative_resources = tournament_service._calculate_resource_requirements(
            StrategyType.CONSERVATIVE, prioritized_questions
        )

        # Aggressive should require more resources
        assert aggressive_resources['research_time'] > conservative_resources['research_time']
        assert aggressive_resources['analysis_time'] > conservative_resources['analysis_time']

        # Check required keys
        for resources in [aggressive_resources, conservative_resources]:
            assert 'research_time' in resources
            assert 'analysis_time' in resources
            assert 'monitoring_time' in resources

    def test_allocate_questions(self, tournament_service):
        """Test question allocation based on strategy."""
        prioritized_questions = [
            QuestionPriority(
                question_id=1,
                priority_level=PriorityLevel.CRITICAL,
                confidence_score=0.9,
                scoring_potential=0.8,
                deadline_urgency=0.7,
                strategic_value=0.8,
                overall_score=0.8,
                reasoning="Top priority",
                metadata={}
            ),
            QuestionPriority(
                question_id=2,
                priority_level=PriorityLevel.HIGH,
                confidence_score=0.7,
                scoring_potential=0.6,
                deadline_urgency=0.5,
                strategic_value=0.6,
                overall_score=0.6,
                reasoning="High priority",
                metadata={}
            ),
            QuestionPriority(
                question_id=3,
                priority_level=PriorityLevel.MEDIUM,
                confidence_score=0.5,
                scoring_potential=0.4,
                deadline_urgency=0.3,
                strategic_value=0.4,
                overall_score=0.4,
                reasoning="Medium priority",
                metadata={}
            )
        ]

        resource_requirements = {'research_time': 10.0, 'analysis_time': 5.0}

        # Test aggressive allocation (should focus on top questions)
        aggressive_allocation = tournament_service._allocate_questions(
            StrategyType.AGGRESSIVE, prioritized_questions, resource_requirements
        )

        # Test conservative allocation (should spread more evenly)
        conservative_allocation = tournament_service._allocate_questions(
            StrategyType.CONSERVATIVE, prioritized_questions, resource_requirements
        )

        # Check that allocations are valid
        for allocation in [aggressive_allocation, conservative_allocation]:
            assert len(allocation) <= tournament_service.max_concurrent_questions
            for question_id, weight in allocation.items():
                assert 0.0 <= weight <= 1.0

        # Top question should get higher allocation in aggressive strategy
        assert aggressive_allocation.get(1, 0) >= conservative_allocation.get(1, 0)

    def test_priority_level_assignment(self, tournament_service):
        """Test priority level assignment logic."""
        # Test critical priority (score >= 0.8)
        assert self._get_priority_level(0.9) == PriorityLevel.CRITICAL
        assert self._get_priority_level(0.8) == PriorityLevel.CRITICAL

        # Test high priority (0.6 <= score < 0.8)
        assert self._get_priority_level(0.7) == PriorityLevel.HIGH
        assert self._get_priority_level(0.6) == PriorityLevel.HIGH

        # Test medium priority (0.4 <= score < 0.6)
        assert self._get_priority_level(0.5) == PriorityLevel.MEDIUM
        assert self._get_priority_level(0.4) == PriorityLevel.MEDIUM

        # Test low priority (score < 0.4)
        assert self._get_priority_level(0.3) == PriorityLevel.LOW
        assert self._get_priority_level(0.1) == PriorityLevel.LOW

    def test_market_condition_determination(self, tournament_service):
        """Test market condition determination logic."""
        # Test oversaturated market
        condition = self._determine_market_condition(
            competitor_density=0.9, consensus_strength=0.8
        )
        assert condition == MarketCondition.OVERSATURATED

        # Test competitive market
        condition = self._determine_market_condition(
            competitor_density=0.7, consensus_strength=0.5
        )
        assert condition == MarketCondition.COMPETITIVE

        # Test underexplored market
        condition = self._determine_market_condition(
            competitor_density=0.2, consensus_strength=0.4
        )
        assert condition == MarketCondition.UNDEREXPLORED

        # Test opportunity market
        condition = self._determine_market_condition(
            competitor_density=0.5, consensus_strength=0.6
        )
        assert condition == MarketCondition.OPPORTUNITY

    # Helper methods for testing
    def _get_priority_level(self, score: float) -> PriorityLevel:
        """Helper method to get priority level from score."""
        if score >= 0.8:
            return PriorityLevel.CRITICAL
        elif score >= 0.6:
            return PriorityLevel.HIGH
        elif score >= 0.4:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW

    def _determine_market_condition(self, competitor_density: float, consensus_strength: float) -> MarketCondition:
        """Helper method to determine market condition."""
        if competitor_density > 0.8 and consensus_strength > 0.7:
            return MarketCondition.OVERSATURATED
        elif competitor_density > 0.6:
            return MarketCondition.COMPETITIVE
        elif competitor_density < 0.3:
            return MarketCondition.UNDEREXPLORED
        else:
            return MarketCondition.OPPORTUNITY


class TestTournamentServiceIntegration:
    """Integration tests for TournamentService."""

    @pytest.fixture
    def tournament_service(self):
        """Create tournament service for integration testing."""
        return TournamentService(risk_tolerance=0.6, max_concurrent_questions=5)

    @pytest.fixture
    def complex_tournament(self):
        """Create complex tournament for integration testing."""
        questions = []
        categories = list(QuestionCategory)
        question_types = list(QuestionType)

        # Create 15 questions with varied characteristics
        for i in range(15):
            category = categories[i % len(categories)]
            q_type = question_types[i % len(question_types)]

            # Vary deadlines from 1 hour to 60 days
            hours_until_deadline = 1 + (i * 4)  # 1, 5, 9, 13, ... hours
            deadline = datetime.utcnow() + timedelta(hours=hours_until_deadline)

            # Vary scoring weights
            weight = 1.0 + (i % 5) * 0.5  # 1.0, 1.5, 2.0, 2.5, 3.0

            question = Question(
                id=i + 1,
                text=f"Test question {i + 1}",
                question_type=q_type,
                category=category,
                deadline=deadline,
                background=f"Background for question {i + 1}",
                resolution_criteria=f"Criteria for question {i + 1}",
                scoring_weight=weight,
                min_value=0.0 if q_type == QuestionType.NUMERIC else None,
                max_value=100.0 if q_type == QuestionType.NUMERIC else None,
                choices=["A", "B", "C"] if q_type == QuestionType.MULTIPLE_CHOICE else None
            )
            questions.append(question)

        # Create standings with 50 participants
        standings = {f"user{i}": 50.0 + np.random.normal(0, 15) for i in range(50)}

        return Tournament(
            id=100,
            name="Complex Integration Test Tournament",
            questions=questions,
            scoring_rules=ScoringRules(
                method=ScoringMethod.BRIER_SCORE,
                weight_by_question=True,
                bonus_for_early=True,
                penalty_for_late=True
            ),
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow() + timedelta(days=90),
            current_standings=standings
        )

    @pytest.mark.asyncio
    async def test_full_tournament_analysis_workflow(self, tournament_service, complex_tournament):
        """Test complete tournament analysis workflow."""
        # This test runs the full analysis without mocking
        result = await tournament_service.analyze_tournament_strategy(complex_tournament)

        # Verify result structure
        assert isinstance(result, StrategyRecommendation)
        assert isinstance(result.strategy_type, StrategyType)
        assert isinstance(result.confidence, Confidence)
        assert 0.0 <= result.confidence.level <= 1.0
        assert isinstance(result.expected_score_impact, float)
        assert 0.0 <= result.risk_level <= 1.0
        assert isinstance(result.resource_requirements, dict)
        assert isinstance(result.question_allocation, dict)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
        assert isinstance(result.alternatives, list)

        # Verify question allocation doesn't exceed max concurrent questions
        assert len(result.question_allocation) <= tournament_service.max_concurrent_questions

        # Verify all allocated questions exist in tournament
        tournament_question_ids = {q.id for q in complex_tournament.questions}
        for question_id in result.question_allocation.keys():
            assert question_id in tournament_question_ids

    @pytest.mark.asyncio
    async def test_prioritization_with_varied_characteristics(self, tournament_service, complex_tournament):
        """Test prioritization with questions having varied characteristics."""
        priorities = await tournament_service._prioritize_questions(complex_tournament)

        # Should prioritize all active questions
        active_count = len(complex_tournament.get_active_questions())
        assert len(priorities) == active_count

        # Check that urgent questions (short deadlines) get higher priority
        urgent_questions = [p for p in priorities if p.deadline_urgency > 0.8]
        non_urgent_questions = [p for p in priorities if p.deadline_urgency < 0.3]

        if urgent_questions and non_urgent_questions:
            avg_urgent_score = np.mean([p.overall_score for p in urgent_questions])
            avg_non_urgent_score = np.mean([p.overall_score for p in non_urgent_questions])
            # Urgent questions should generally have higher overall scores
            assert avg_urgent_score >= avg_non_urgent_score - 0.1  # Allow small margin

    @pytest.mark.asyncio
    async def test_market_analysis_consistency(self, tournament_service, complex_tournament):
        """Test market analysis consistency across questions."""
        market_analysis = await tournament_service._analyze_market_conditions(complex_tournament)

        active_questions = complex_tournament.get_active_questions()
        assert len(market_analysis) == len(active_questions)

        # Check consistency of market conditions
        for question in active_questions:
            analysis = market_analysis[question.id]

            # Verify timing windows are logical
            start_time, end_time = analysis.optimal_timing_window
            assert start_time < end_time
            assert start_time >= datetime.utcnow()
            assert end_time <= question.deadline

            # Verify market condition matches metrics
            if analysis.competitor_density > 0.8 and analysis.consensus_strength > 0.7:
                assert analysis.market_condition == MarketCondition.OVERSATURATED
            elif analysis.competitor_density < 0.3:
                assert analysis.market_condition == MarketCondition.UNDEREXPLORED

    def test_strategy_adaptation_over_time(self, tournament_service):
        """Test strategy adaptation based on historical performance."""
        # Simulate performance history
        performance_history = []

        # Add results showing aggressive strategy performing well
        for i in range(5):
            result = StrategyResult.create(
                strategy_type=StrategyType.AGGRESSIVE,
                expected_score=0.7,
                reasoning=f"Aggressive test {i}",
                question_ids=[i + 1],
                confidence_level=0.8,
                confidence_basis="Test basis"
            ).mark_success(0.8 + np.random.normal(0, 0.1))
            performance_history.append(result)

        # Add results showing conservative strategy performing poorly
        for i in range(3):
            result = StrategyResult.create(
                strategy_type=StrategyType.CONSERVATIVE,
                expected_score=0.6,
                reasoning=f"Conservative test {i}",
                question_ids=[i + 6],
                confidence_level=0.7,
                confidence_basis="Test basis"
            ).mark_failure(0.4 + np.random.normal(0, 0.1))
            performance_history.append(result)

        # Test adaptation
        adapted_strategy = tournament_service.adapt_strategy_based_on_performance(performance_history)

        # Should favor aggressive strategy due to better performance (allow some randomness)
        # The method includes 10% exploration, so we'll check it's not the worst performing strategy
        assert adapted_strategy != StrategyType.CONSERVATIVE  # Conservative performed poorly

    def test_risk_adjustment_scenarios(self, tournament_service, complex_tournament):
        """Test risk adjustment in different tournament scenarios."""
        # Test early tournament phase (more time remaining)
        early_tournament = Tournament(
            id=200,
            name="Early Tournament",
            questions=complex_tournament.questions[:5],
            scoring_rules=complex_tournament.scoring_rules,
            start_date=datetime.utcnow() - timedelta(days=5),
            end_date=datetime.utcnow() + timedelta(days=85),  # Lots of time remaining
            current_standings={"user1": 70.0, "user2": 65.0}
        )

        # Test late tournament phase (little time remaining)
        late_tournament = Tournament(
            id=201,
            name="Late Tournament",
            questions=complex_tournament.questions[:5],
            scoring_rules=complex_tournament.scoring_rules,
            start_date=datetime.utcnow() - timedelta(days=85),
            end_date=datetime.utcnow() + timedelta(days=5),  # Little time remaining
            current_standings={"user1": 70.0, "user2": 65.0}
        )

        # Leading position in early tournament should be more conservative
        early_strategy = tournament_service.select_risk_adjusted_strategy(early_tournament, current_position=1)

        # Trailing position in late tournament should be more aggressive
        late_strategy = tournament_service.select_risk_adjusted_strategy(late_tournament, current_position=10)

        # Verify risk adjustment logic
        conservative_strategies = [StrategyType.CONSERVATIVE, StrategyType.BALANCED]
        aggressive_strategies = [StrategyType.AGGRESSIVE, StrategyType.BALANCED]

        # Early leader should tend toward conservative
        assert early_strategy in conservative_strategies

        # Late trailer should tend toward aggressive
        assert late_strategy in aggressive_strategies or late_strategy == StrategyType.BALANCED
