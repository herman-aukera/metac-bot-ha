"""Tests for the tournament analytics service."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.domain.services.tournament_analytics import (
    CompetitorProfile,
    MarketInefficiency,
    MarketInefficiencyType,
    StrategicOpportunity,
    StrategicOpportunityType,
    TournamentAnalytics,
    TournamentStandings,
)


class TestTournamentAnalytics:
    """Test cases for TournamentAnalytics."""

    @pytest.fixture
    def tournament_analytics(self):
        """Create tournament analytics service for testing."""
        return TournamentAnalytics()

    @pytest.fixture
    def sample_standings_data(self):
        """Create sample tournament standings data."""
        return {
            "participants": [
                {
                    "user_id": "user_1",
                    "username": "top_performer",
                    "score": 95.5,
                    "questions_answered": 50,
                    "questions_resolved": 45,
                    "average_brier_score": 0.12,
                    "calibration_score": 0.88,
                    "average_confidence": 0.75,
                    "prediction_frequency": 0.9,
                    "category_performance": {
                        "politics": {"brier_score": 0.10},
                        "economics": {"brier_score": 0.14},
                    },
                },
                {
                    "user_id": "user_2",
                    "username": "second_place",
                    "score": 89.2,
                    "questions_answered": 48,
                    "questions_resolved": 42,
                    "average_brier_score": 0.15,
                    "calibration_score": 0.82,
                },
                {
                    "user_id": "our_user",
                    "username": "our_bot",
                    "score": 75.8,
                    "questions_answered": 35,
                    "questions_resolved": 30,
                    "average_brier_score": 0.18,
                    "calibration_score": 0.78,
                },
                {
                    "user_id": "user_4",
                    "username": "fourth_place",
                    "score": 70.1,
                    "questions_answered": 40,
                    "questions_resolved": 35,
                    "average_brier_score": 0.22,
                    "calibration_score": 0.72,
                },
            ]
        }

    @pytest.fixture
    def sample_community_predictions(self):
        """Create sample community predictions data."""
        base_time = datetime.utcnow()
        return [
            {
                "question_id": str(uuid4()),
                "prediction": 0.8,
                "timestamp": (base_time - timedelta(hours=24)).isoformat() + "Z",
                "user_id": "user_1",
            },
            {
                "question_id": str(uuid4()),
                "prediction": 0.85,
                "timestamp": (base_time - timedelta(hours=20)).isoformat() + "Z",
                "user_id": "user_2",
            },
            {
                "question_id": str(uuid4()),
                "prediction": 0.9,
                "timestamp": (base_time - timedelta(hours=16)).isoformat() + "Z",
                "user_id": "user_3",
            },
            {
                "question_id": str(uuid4()),
                "prediction": 0.95,
                "timestamp": (base_time - timedelta(hours=12)).isoformat() + "Z",
                "user_id": "user_4",
            },
            {
                "question_id": str(uuid4()),
                "prediction": 0.92,
                "timestamp": (base_time - timedelta(hours=8)).isoformat() + "Z",
                "user_id": "user_5",
            },
        ]

    def test_initialization(self, tournament_analytics):
        """Test tournament analytics initialization."""
        assert len(tournament_analytics.competitor_profiles) == 0
        assert len(tournament_analytics.tournament_standings) == 0
        assert len(tournament_analytics.market_inefficiencies) == 0
        assert len(tournament_analytics.strategic_opportunities) == 0
        assert tournament_analytics.inefficiency_detection_threshold == 0.7
        assert tournament_analytics.opportunity_confidence_threshold == 0.6

    def test_analyze_tournament_standings(
        self, tournament_analytics, sample_standings_data
    ):
        """Test tournament standings analysis."""
        tournament_id = 12345
        our_user_id = "our_user"

        standings = tournament_analytics.analyze_tournament_standings(
            tournament_id=tournament_id,
            standings_data=sample_standings_data,
            our_user_id=our_user_id,
        )

        # Check basic standings information
        assert standings.tournament_id == tournament_id
        assert standings.our_ranking == 3  # Third place
        assert standings.our_score == 75.8
        assert standings.total_participants == 4
        assert standings.our_percentile == 0.5  # (4-3+1)/4 = 0.5

        # Check top performers analysis
        assert len(standings.top_performers) <= 4  # All participants in this case
        top_performer = standings.top_performers[0]
        assert top_performer.competitor_id == "user_1"
        assert top_performer.current_ranking == 1
        assert "excellent_accuracy" in top_performer.strengths  # Brier score < 0.2

        # Check score distribution
        assert "mean" in standings.score_distribution
        assert "median" in standings.score_distribution
        assert "std" in standings.score_distribution

        # Check competitive gaps
        assert "next_rank" in standings.competitive_gaps
        assert (
            standings.competitive_gaps["next_rank"] == 89.2 - 75.8
        )  # Gap to second place

        # Check improvement opportunities
        assert len(standings.improvement_opportunities) > 0
        assert isinstance(standings.improvement_opportunities[0], str)

    def test_competitor_profile_analysis(
        self, tournament_analytics, sample_standings_data
    ):
        """Test competitor profile analysis."""
        tournament_id = 12345
        our_user_id = "our_user"

        standings = tournament_analytics.analyze_tournament_standings(
            tournament_id, sample_standings_data, our_user_id
        )

        # Check top performer profile
        top_performer = standings.top_performers[0]
        assert top_performer.competitor_id == "user_1"
        assert top_performer.username == "top_performer"
        assert top_performer.average_brier_score == 0.12
        assert top_performer.calibration_score == 0.88

        # Check strengths identification
        assert "excellent_accuracy" in top_performer.strengths
        assert "well_calibrated" in top_performer.strengths
        assert "high_volume" in top_performer.strengths
        assert "consistent_participation" in top_performer.strengths

        # Check prediction patterns
        patterns = top_performer.prediction_patterns
        assert patterns["average_confidence"] == 0.75
        assert patterns["prediction_frequency"] == 0.9
        assert patterns["risk_profile"] in ["conservative", "moderate", "aggressive"]

    def test_detect_overconfidence_bias(self, tournament_analytics):
        """Test overconfidence bias detection."""
        question_data = {"id": str(uuid4()), "title": "Test question"}

        # Create predictions with high proportion of extreme values
        extreme_predictions = [
            0.05,
            0.95,
            0.02,
            0.98,
            0.03,
            0.97,
            0.5,
            0.6,
        ]  # 6/8 = 75% extreme
        community_predictions = [
            {"prediction": pred, "user_id": f"user_{i}"}
            for i, pred in enumerate(extreme_predictions)
        ]

        inefficiencies = tournament_analytics.detect_market_inefficiencies(
            question_data=question_data, community_predictions=community_predictions
        )

        # Should detect overconfidence bias
        overconfidence_inefficiencies = [
            ineff
            for ineff in inefficiencies
            if ineff.inefficiency_type == MarketInefficiencyType.OVERCONFIDENCE_BIAS
        ]
        assert len(overconfidence_inefficiencies) > 0

        overconfidence = overconfidence_inefficiencies[0]
        assert overconfidence.confidence_level > 0.5
        assert overconfidence.potential_advantage > 0.1
        assert "extreme predictions" in overconfidence.description.lower()

    def test_detect_herding_behavior(self, tournament_analytics):
        """Test herding behavior detection."""
        question_data = {"id": str(uuid4()), "title": "Test question"}

        # Create tightly clustered predictions (herding)
        base_time = datetime.utcnow()
        clustered_predictions = []
        for i in range(12):
            clustered_predictions.append(
                {
                    "question_id": str(uuid4()),
                    "prediction": 0.65
                    + (i % 3) * 0.01,  # Very tight clustering around 0.65
                    "timestamp": (base_time - timedelta(hours=i)).isoformat() + "Z",
                    "user_id": f"user_{i}",
                }
            )

        inefficiencies = tournament_analytics.detect_market_inefficiencies(
            question_data=question_data, community_predictions=clustered_predictions
        )

        # Should detect herding behavior
        herding_inefficiencies = [
            ineff
            for ineff in inefficiencies
            if ineff.inefficiency_type == MarketInefficiencyType.HERDING_BEHAVIOR
        ]
        assert len(herding_inefficiencies) > 0

        herding = herding_inefficiencies[0]
        assert herding.confidence_level > 0.5
        assert "clustering" in herding.description.lower()

    def test_detect_anchoring_bias(self, tournament_analytics):
        """Test anchoring bias detection."""
        question_data = {"id": str(uuid4()), "title": "Test question"}
        historical_patterns = {"similar_questions": []}

        # Create predictions clustered around round number (0.5)
        anchored_predictions = []
        for i in range(15):
            if i < 8:  # 8/15 > 40% clustered around 0.5
                pred = 0.5 + (i % 3 - 1) * 0.01  # 0.49, 0.5, 0.51
            else:
                pred = 0.3 + i * 0.05

            anchored_predictions.append({"prediction": pred, "user_id": f"user_{i}"})

        inefficiencies = tournament_analytics.detect_market_inefficiencies(
            question_data=question_data,
            community_predictions=anchored_predictions,
            historical_patterns=historical_patterns,
        )

        # Should detect anchoring bias
        anchoring_inefficiencies = [
            ineff
            for ineff in inefficiencies
            if ineff.inefficiency_type == MarketInefficiencyType.ANCHORING_BIAS
        ]
        assert len(anchoring_inefficiencies) > 0

        anchoring = anchoring_inefficiencies[0]
        assert "0.5" in anchoring.description
        assert "clustering" in anchoring.description.lower()

    def test_detect_recency_bias(self, tournament_analytics):
        """Test recency bias detection."""
        question_data = {"id": str(uuid4()), "title": "Test question"}

        # Create predictions with significant recent shift
        base_time = datetime.utcnow()
        predictions_with_shift = []

        # Older predictions around 0.3
        for i in range(8):
            predictions_with_shift.append(
                {
                    "prediction": 0.3 + i * 0.02,
                    "timestamp": (base_time - timedelta(hours=48 + i)).isoformat()
                    + "Z",
                    "user_id": f"old_user_{i}",
                }
            )

        # Recent predictions around 0.7 (significant shift)
        for i in range(5):
            predictions_with_shift.append(
                {
                    "prediction": 0.7 + i * 0.02,
                    "timestamp": (base_time - timedelta(hours=i)).isoformat() + "Z",
                    "user_id": f"new_user_{i}",
                }
            )

        inefficiencies = tournament_analytics.detect_market_inefficiencies(
            question_data=question_data, community_predictions=predictions_with_shift
        )

        # Should detect recency bias
        recency_inefficiencies = [
            ineff
            for ineff in inefficiencies
            if ineff.inefficiency_type == MarketInefficiencyType.RECENCY_BIAS
        ]
        assert len(recency_inefficiencies) > 0

        recency = recency_inefficiencies[0]
        assert recency.potential_advantage > 0.1
        assert "shifted" in recency.description.lower()

    def test_detect_momentum_patterns(self, tournament_analytics):
        """Test momentum pattern detection."""
        question_data = {"id": str(uuid4()), "title": "Test question"}

        # Create predictions with strong upward momentum
        base_time = datetime.utcnow()
        momentum_predictions = []

        for i in range(10):
            # Strong upward trend from 0.3 to 0.8
            prediction = 0.3 + (i / 9) * 0.5
            momentum_predictions.append(
                {
                    "question_id": str(uuid4()),
                    "prediction": prediction,
                    "timestamp": (base_time - timedelta(hours=24 - i * 2)).isoformat()
                    + "Z",
                    "user_id": f"user_{i}",
                }
            )

        inefficiencies = tournament_analytics.detect_market_inefficiencies(
            question_data=question_data, community_predictions=momentum_predictions
        )

        # Should detect momentum effects
        momentum_inefficiencies = [
            ineff
            for ineff in inefficiencies
            if ineff.inefficiency_type
            in [
                MarketInefficiencyType.MOMENTUM_EFFECT,
                MarketInefficiencyType.CONTRARIAN_OPPORTUNITY,
            ]
        ]
        assert len(momentum_inefficiencies) > 0

        # Check for upward momentum detection
        upward_momentum = [
            ineff
            for ineff in momentum_inefficiencies
            if "upward" in ineff.description.lower()
        ]
        assert len(upward_momentum) > 0

    def test_identify_timing_opportunities(self, tournament_analytics):
        """Test timing opportunity identification."""
        tournament_context = {"tournament_id": 12345}
        our_performance = {"category_performance": {}}

        # Create question pipeline with timing opportunities
        base_time = datetime.utcnow()
        question_pipeline = [
            {
                "id": str(uuid4()),
                "title": "Early opportunity question",
                "deadline": (base_time + timedelta(hours=72)).isoformat() + "Z",
                "prediction_count": 5,  # Few predictions
                "category": "politics",
            },
            {
                "id": str(uuid4()),
                "title": "Late opportunity question",
                "deadline": (base_time + timedelta(hours=8)).isoformat() + "Z",
                "prediction_count": 25,  # Many predictions
                "category": "economics",
            },
        ]

        opportunities = tournament_analytics.identify_strategic_opportunities(
            tournament_context=tournament_context,
            our_performance=our_performance,
            question_pipeline=question_pipeline,
        )

        # Should identify timing opportunities
        timing_opportunities = [
            opp
            for opp in opportunities
            if opp.opportunity_type
            in [
                StrategicOpportunityType.EARLY_MOVER_ADVANTAGE,
                StrategicOpportunityType.LATE_MOVER_ADVANTAGE,
            ]
        ]
        assert len(timing_opportunities) > 0

        # Check early mover advantage
        early_mover = [
            opp
            for opp in timing_opportunities
            if opp.opportunity_type == StrategicOpportunityType.EARLY_MOVER_ADVANTAGE
        ]
        assert len(early_mover) > 0
        assert early_mover[0].time_sensitivity > 0.5

    def test_identify_information_edge_opportunities(self, tournament_analytics):
        """Test information edge opportunity identification."""
        tournament_context = {"tournament_id": 12345}
        our_performance = {
            "category_performance": {
                "politics": {
                    "brier_score": 0.15,  # Strong performance
                    "question_count": 10,
                },
                "economics": {
                    "brier_score": 0.35,  # Weak performance
                    "question_count": 8,
                },
            }
        }

        question_pipeline = [
            {
                "id": str(uuid4()),
                "title": "Politics question",
                "category": "politics",
                "deadline": (datetime.utcnow() + timedelta(hours=48)).isoformat() + "Z",
            },
            {
                "id": str(uuid4()),
                "title": "Economics question",
                "category": "economics",
                "deadline": (datetime.utcnow() + timedelta(hours=48)).isoformat() + "Z",
            },
        ]

        opportunities = tournament_analytics.identify_strategic_opportunities(
            tournament_context=tournament_context,
            our_performance=our_performance,
            question_pipeline=question_pipeline,
        )

        # Should identify niche expertise opportunity for politics
        expertise_opportunities = [
            opp
            for opp in opportunities
            if opp.opportunity_type == StrategicOpportunityType.NICHE_EXPERTISE
        ]
        assert len(expertise_opportunities) > 0

        politics_opportunity = [
            opp
            for opp in expertise_opportunities
            if "politics" in opp.description.lower()
        ]
        assert len(politics_opportunity) > 0
        assert politics_opportunity[0].confidence > 0.7

    def test_identify_positioning_opportunities(self, tournament_analytics):
        """Test competitive positioning opportunity identification."""
        # Test low ranking scenario (aggressive strategy)
        low_ranking_context = {"our_ranking": 85, "total_participants": 100}
        our_performance = {}
        question_pipeline = []

        opportunities = tournament_analytics.identify_strategic_opportunities(
            tournament_context=low_ranking_context,
            our_performance=our_performance,
            question_pipeline=question_pipeline,
        )

        # Should suggest aggressive strategy for low ranking
        aggressive_opportunities = [
            opp
            for opp in opportunities
            if opp.opportunity_type == StrategicOpportunityType.CONSENSUS_EXPLOITATION
        ]
        assert len(aggressive_opportunities) > 0

        # Test middle ranking scenario (selective strategy)
        middle_ranking_context = {"our_ranking": 25, "total_participants": 100}

        opportunities = tournament_analytics.identify_strategic_opportunities(
            tournament_context=middle_ranking_context,
            our_performance=our_performance,
            question_pipeline=question_pipeline,
        )

        # Should suggest selective strategy for middle ranking
        selective_opportunities = [
            opp
            for opp in opportunities
            if opp.opportunity_type == StrategicOpportunityType.VOLATILITY_ARBITRAGE
        ]
        assert len(selective_opportunities) > 0

    def test_generate_competitive_intelligence_report(
        self, tournament_analytics, sample_standings_data
    ):
        """Test competitive intelligence report generation."""
        tournament_id = 12345
        our_user_id = "our_user"

        # First analyze standings
        tournament_analytics.analyze_tournament_standings(
            tournament_id, sample_standings_data, our_user_id
        )

        # Add some market inefficiencies and opportunities
        question_data = {"id": str(uuid4()), "title": "Test question"}
        community_predictions = [
            {"prediction": 0.95, "user_id": "user_1"}
        ] * 10  # Extreme predictions

        tournament_analytics.detect_market_inefficiencies(
            question_data, community_predictions
        )

        # Generate report
        report = tournament_analytics.generate_competitive_intelligence_report(
            tournament_id=tournament_id, include_recommendations=True
        )

        # Check report structure
        assert "tournament_id" in report
        assert "generated_at" in report
        assert "competitive_position" in report
        assert "competitive_landscape" in report
        assert "market_analysis" in report
        assert "strategic_opportunities" in report
        assert "improvement_opportunities" in report
        assert "strategic_recommendations" in report

        # Check competitive position
        position = report["competitive_position"]
        assert position["current_ranking"] == 3
        assert position["total_participants"] == 4
        assert position["score"] == 75.8

        # Check strategic recommendations (new structure)
        recommendations = report["strategic_recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_competitor_strength_identification(self, tournament_analytics):
        """Test competitor strength identification."""
        participant_data = {
            "user_id": "test_user",
            "average_brier_score": 0.15,  # Excellent accuracy
            "calibration_score": 0.85,  # Well calibrated
            "questions_answered": 60,  # High volume
            "prediction_frequency": 0.9,  # Consistent participation
            "category_performance": {"politics": {"brier_score": 0.12}},  # Expert level
        }

        strengths = tournament_analytics._identify_competitor_strengths(
            participant_data
        )

        assert "excellent_accuracy" in strengths
        assert "well_calibrated" in strengths
        assert "high_volume" in strengths
        assert "consistent_participation" in strengths
        assert "expert_in_politics" in strengths

    def test_competitor_weakness_identification(self, tournament_analytics):
        """Test competitor weakness identification."""
        participant_data = {
            "user_id": "test_user",
            "average_brier_score": 0.4,  # Poor accuracy
            "calibration_score": 0.3,  # Poorly calibrated
            "prediction_variance": 0.25,  # Inconsistent
            "questions_answered": 5,  # Low participation
            "timing_patterns": {"late_submissions": 0.7},  # Poor timing
        }

        weaknesses = tournament_analytics._identify_competitor_weaknesses(
            participant_data
        )

        assert "poor_accuracy" in weaknesses
        assert "poorly_calibrated" in weaknesses
        assert "inconsistent_predictions" in weaknesses
        assert "low_participation" in weaknesses
        assert "poor_timing" in weaknesses

    def test_risk_profile_assessment(self, tournament_analytics):
        """Test risk profile assessment."""
        # Conservative profile
        conservative_data = {"average_confidence": 0.85, "prediction_variance": 0.03}
        risk_profile = tournament_analytics._assess_risk_profile(conservative_data)
        assert risk_profile == "conservative"

        # Aggressive profile
        aggressive_data = {"average_confidence": 0.55, "prediction_variance": 0.18}
        risk_profile = tournament_analytics._assess_risk_profile(aggressive_data)
        assert risk_profile == "aggressive"

        # Moderate profile
        moderate_data = {"average_confidence": 0.7, "prediction_variance": 0.1}
        risk_profile = tournament_analytics._assess_risk_profile(moderate_data)
        assert risk_profile == "moderate"

    def test_score_distribution_calculation(self, tournament_analytics):
        """Test score distribution calculation."""
        scores = [95.5, 89.2, 75.8, 70.1, 65.3, 60.7, 55.2]

        distribution = tournament_analytics._calculate_score_distribution(scores)

        assert "mean" in distribution
        assert "median" in distribution
        assert "std" in distribution
        assert "min" in distribution
        assert "max" in distribution
        assert "q25" in distribution
        assert "q75" in distribution

        assert distribution["min"] == 55.2
        assert distribution["max"] == 95.5
        assert distribution["mean"] == sum(scores) / len(scores)

    def test_competitive_gaps_identification(self, tournament_analytics):
        """Test competitive gaps identification."""
        our_score = 75.8
        all_scores = [95.5, 89.2, 75.8, 70.1, 65.3]
        our_ranking = 3

        gaps = tournament_analytics._identify_competitive_gaps(
            our_score, all_scores, our_ranking
        )

        assert "next_rank" in gaps
        assert "top_10_percent" in gaps
        assert "top_5_percent" in gaps
        assert "leader" in gaps

        assert gaps["next_rank"] == 89.2 - 75.8  # Gap to second place
        assert gaps["leader"] == 95.5 - 75.8  # Gap to first place

    def test_improvement_opportunities_generation(self, tournament_analytics):
        """Test improvement opportunities generation."""
        our_ranking = 3
        our_score = 75.8

        # Create top performers with common strengths
        top_performers = [
            CompetitorProfile(
                competitor_id="user_1",
                username="top_1",
                current_ranking=1,
                total_score=95.5,
                questions_answered=50,
                questions_resolved=45,
                average_brier_score=0.12,
                calibration_score=0.88,
                prediction_patterns={},
                strengths=["excellent_accuracy", "well_calibrated"],
                weaknesses=[],
                last_updated=datetime.utcnow(),
            ),
            CompetitorProfile(
                competitor_id="user_2",
                username="top_2",
                current_ranking=2,
                total_score=89.2,
                questions_answered=48,
                questions_resolved=42,
                average_brier_score=0.15,
                calibration_score=0.82,
                prediction_patterns={},
                strengths=["excellent_accuracy", "high_volume"],
                weaknesses=[],
                last_updated=datetime.utcnow(),
            ),
        ]

        competitive_gaps = {"next_rank": 13.4, "top_10_percent": 15.0}

        opportunities = tournament_analytics._generate_improvement_opportunities(
            our_ranking, our_score, top_performers, competitive_gaps
        )

        assert len(opportunities) > 0
        assert isinstance(opportunities[0], str)

        # Should suggest accuracy improvement (common strength)
        accuracy_suggestions = [
            opp for opp in opportunities if "accuracy" in opp.lower()
        ]
        assert len(accuracy_suggestions) > 0

    def test_cleanup_expired_opportunities(self, tournament_analytics):
        """Test cleanup of expired opportunities."""
        # Add some opportunities with different ages
        current_time = datetime.utcnow()

        # Recent opportunity (should be kept)
        recent_opportunity = StrategicOpportunity(
            opportunity_type=StrategicOpportunityType.TIMING_ADVANTAGE,
            question_id=uuid4(),
            title="Recent opportunity",
            description="Recent opportunity",
            potential_impact=0.1,
            confidence=0.7,
            time_sensitivity=0.5,
            resource_requirements={},
            recommended_actions=[],
            identified_at=current_time - timedelta(days=3),
        )

        # Old opportunity (should be removed)
        old_opportunity = StrategicOpportunity(
            opportunity_type=StrategicOpportunityType.TIMING_ADVANTAGE,
            question_id=uuid4(),
            title="Old opportunity",
            description="Old opportunity",
            potential_impact=0.1,
            confidence=0.7,
            time_sensitivity=0.5,
            resource_requirements={},
            recommended_actions=[],
            identified_at=current_time - timedelta(days=10),
        )

        tournament_analytics.strategic_opportunities = [
            recent_opportunity,
            old_opportunity,
        ]

        # Cleanup expired opportunities
        tournament_analytics._cleanup_expired_opportunities()

        # Should only keep recent opportunity
        assert len(tournament_analytics.strategic_opportunities) == 1
        assert (
            tournament_analytics.strategic_opportunities[0].title
            == "Recent opportunity"
        )

    def test_analyze_performance_attribution(self, tournament_analytics):
        """Test performance attribution analysis."""
        our_performance_data = {
            "average_brier_score": 0.25,
            "calibration_score": 0.75,
            "questions_answered": 45,
            "category_performance": {
                "politics": {"brier_score": 0.20, "question_count": 15},
                "economics": {"brier_score": 0.30, "question_count": 10},
            },
        }

        competitor_performance_data = [
            {
                "average_brier_score": 0.15,
                "calibration_score": 0.85,
                "questions_answered": 50,
                "category_performance": {
                    "politics": {"brier_score": 0.12},
                    "economics": {"brier_score": 0.18},
                },
            },
            {
                "average_brier_score": 0.35,
                "calibration_score": 0.65,
                "questions_answered": 30,
                "category_performance": {
                    "politics": {"brier_score": 0.32},
                    "economics": {"brier_score": 0.38},
                },
            },
        ]

        question_history = [
            {
                "research_depth": 0.8,
                "brier_score": 0.15,
                "confidence": 0.7,
                "was_correct": True,
                "category": "politics",
                "hours_before_deadline": 48,
            },
            {
                "research_depth": 0.6,
                "brier_score": 0.25,
                "confidence": 0.6,
                "was_correct": False,
                "category": "economics",
                "hours_before_deadline": 12,
            },
            {
                "research_depth": 0.9,
                "brier_score": 0.10,
                "confidence": 0.8,
                "was_correct": True,
                "category": "politics",
                "hours_before_deadline": 72,
            },
        ]

        attribution = tournament_analytics.analyze_performance_attribution(
            our_performance_data=our_performance_data,
            competitor_performance_data=competitor_performance_data,
            question_history=question_history,
        )

        # Check attribution structure
        assert "accuracy_drivers" in attribution
        assert "calibration_analysis" in attribution
        assert "category_performance" in attribution
        assert "timing_impact" in attribution
        assert "confidence_optimization" in attribution
        assert "competitive_advantages" in attribution
        assert "performance_gaps" in attribution

        # Check accuracy drivers
        accuracy_drivers = attribution["accuracy_drivers"]
        assert "research_depth_correlation" in accuracy_drivers
        assert "confidence_accuracy_relationship" in accuracy_drivers

        # Check category performance attribution
        category_performance = attribution["category_performance"]
        assert "politics" in category_performance
        assert "economics" in category_performance

        politics_perf = category_performance["politics"]
        assert politics_perf["our_brier_score"] == 0.20
        assert "competitor_average_brier" in politics_perf
        assert "relative_performance" in politics_perf

    def test_generate_optimization_recommendations(self, tournament_analytics):
        """Test optimization recommendations generation."""
        tournament_id = 12345

        performance_attribution = {
            "accuracy_drivers": {
                "research_depth_correlation": 0.4,  # Strong correlation
                "confidence_accuracy_relationship": {
                    "high": {"average_accuracy": 0.8, "sample_size": 10}
                },
            },
            "calibration_analysis": {"overall_calibration": 0.6},  # Below optimal
            "category_performance": {
                "politics": {"competitive_advantage": False, "sample_size": 10}
            },
            "timing_impact": {
                "early_vs_late_performance": {
                    "timing_advantage": 0.08  # Early predictions better
                }
            },
            "performance_gaps": [
                {
                    "type": "accuracy_gap",
                    "description": "Accuracy gap to top performers",
                    "severity": "high",
                    "improvement_potential": 15,
                    "recommendations": ["Improve research methodology"],
                }
            ],
        }

        current_standings = TournamentStandings(
            tournament_id=tournament_id,
            our_ranking=25,
            our_score=75.0,
            total_participants=100,
            top_performers=[],
            our_percentile=0.75,
            score_distribution={},
            competitive_gaps={"next_rank": 3.5},
            improvement_opportunities=[],
            last_updated=datetime.utcnow(),
        )

        recommendations = tournament_analytics.generate_optimization_recommendations(
            tournament_id=tournament_id,
            performance_attribution=performance_attribution,
            current_standings=current_standings,
        )

        # Should generate multiple recommendations
        assert len(recommendations) > 0

        # Check for research optimization recommendation
        research_recs = [
            r for r in recommendations if r["category"] == "research_optimization"
        ]
        assert len(research_recs) > 0
        assert research_recs[0]["priority"] == "high"

        # Check for calibration improvement recommendation
        calibration_recs = [
            r for r in recommendations if r["category"] == "calibration_improvement"
        ]
        assert len(calibration_recs) > 0

        # Check for timing optimization recommendation
        timing_recs = [
            r for r in recommendations if r["category"] == "timing_optimization"
        ]
        assert len(timing_recs) > 0

        # Verify recommendation structure
        for rec in recommendations:
            assert "category" in rec
            assert "priority" in rec
            assert "title" in rec
            assert "description" in rec
            assert "specific_actions" in rec
            assert "expected_impact" in rec
            assert "implementation_effort" in rec
            assert "timeline" in rec

    def test_enhanced_competitive_intelligence_report(
        self, tournament_analytics, sample_standings_data
    ):
        """Test enhanced competitive intelligence report with new features."""
        tournament_id = 12345
        our_user_id = "our_user"

        # First analyze standings
        tournament_analytics.analyze_tournament_standings(
            tournament_id, sample_standings_data, our_user_id
        )

        # Add some market inefficiencies
        question_data = {"id": str(uuid4()), "title": "Test question"}
        community_predictions = [{"prediction": 0.95, "user_id": "user_1"}] * 10

        tournament_analytics.detect_market_inefficiencies(
            question_data, community_predictions
        )

        # Generate enhanced report
        report = tournament_analytics.generate_competitive_intelligence_report(
            tournament_id=tournament_id,
            include_recommendations=True,
            include_performance_attribution=True,
        )

        # Check enhanced report structure
        assert "competitive_position" in report
        assert "competitive_landscape" in report
        assert "market_analysis" in report
        assert "strategic_opportunities" in report
        assert "strategic_recommendations" in report

        # Check enhanced competitive position
        position = report["competitive_position"]
        assert "ranking_trend" in position
        assert "competitive_momentum" in position

        # Check enhanced competitive landscape
        landscape = report["competitive_landscape"]
        assert "market_concentration" in landscape
        assert "competitive_threats" in landscape

        # Check enhanced market analysis
        market = report["market_analysis"]
        assert "exploitable_opportunities" in market
        assert "market_efficiency_score" in market

        # Check enhanced strategic opportunities
        opportunities = report["strategic_opportunities"]
        assert "high_impact_opportunities" in opportunities
        assert "time_sensitive_opportunities" in opportunities

        # Check strategic recommendations structure
        recommendations = report["strategic_recommendations"]
        assert isinstance(recommendations, list)
        if recommendations:
            rec = recommendations[0]
            assert "category" in rec
            assert "priority" in rec
            assert "title" in rec
            assert "description" in rec
            assert "rationale" in rec
            assert "specific_actions" in rec
            assert "expected_impact" in rec
            assert "risk_level" in rec

    def test_competitive_advantages_identification(self, tournament_analytics):
        """Test identification of competitive advantages."""
        our_performance = {
            "average_brier_score": 0.15,  # Better than average
            "calibration_score": 0.85,  # Well calibrated
            "questions_answered": 60,  # High volume
        }

        competitor_performance = [
            {
                "average_brier_score": 0.25,
                "calibration_score": 0.70,
                "questions_answered": 40,
            },
            {
                "average_brier_score": 0.30,
                "calibration_score": 0.65,
                "questions_answered": 35,
            },
            {
                "average_brier_score": 0.20,
                "calibration_score": 0.75,
                "questions_answered": 45,
            },
        ]

        advantages = tournament_analytics._identify_competitive_advantages(
            our_performance, competitor_performance
        )

        # Should identify multiple advantages
        assert len(advantages) > 0

        # Check for accuracy advantage
        accuracy_advantages = [
            a for a in advantages if a["type"] == "accuracy_advantage"
        ]
        assert len(accuracy_advantages) > 0
        assert accuracy_advantages[0]["strength"] > 0.5

        # Check for calibration advantage
        calibration_advantages = [
            a for a in advantages if a["type"] == "calibration_advantage"
        ]
        assert len(calibration_advantages) > 0

        # Check for participation advantage
        participation_advantages = [
            a for a in advantages if a["type"] == "participation_advantage"
        ]
        assert len(participation_advantages) > 0

    def test_performance_gaps_analysis(self, tournament_analytics):
        """Test performance gaps analysis."""
        our_performance = {
            "average_brier_score": 0.30,  # Below top performers
            "calibration_score": 0.60,  # Below top performers
        }

        competitor_performance = [
            {"average_brier_score": 0.15, "calibration_score": 0.85},  # Top performer
            {"average_brier_score": 0.18, "calibration_score": 0.82},  # Top performer
            {"average_brier_score": 0.20, "calibration_score": 0.80},  # Top performer
            {"average_brier_score": 0.35, "calibration_score": 0.55},  # Below us
            {"average_brier_score": 0.40, "calibration_score": 0.50},  # Below us
        ]

        gaps = tournament_analytics._analyze_performance_gaps(
            our_performance, competitor_performance
        )

        # Should identify gaps
        assert len(gaps) > 0

        # Check for accuracy gap
        accuracy_gaps = [g for g in gaps if g["type"] == "accuracy_gap"]
        assert len(accuracy_gaps) > 0
        assert accuracy_gaps[0]["severity"] in ["high", "medium", "low"]
        assert "recommendations" in accuracy_gaps[0]

        # Check for calibration gap
        calibration_gaps = [g for g in gaps if g["type"] == "calibration_gap"]
        assert len(calibration_gaps) > 0

    def test_market_efficiency_calculation(self, tournament_analytics):
        """Test market efficiency score calculation."""
        # High inefficiency scenario
        high_inefficiencies = [
            MarketInefficiency(
                inefficiency_type=MarketInefficiencyType.OVERCONFIDENCE_BIAS,
                question_id=uuid4(),
                description="High overconfidence",
                confidence_level=0.8,
                potential_advantage=0.2,
                detected_at=datetime.utcnow(),
                expiration_estimate=None,
                exploitation_strategy="Test",
            ),
            MarketInefficiency(
                inefficiency_type=MarketInefficiencyType.HERDING_BEHAVIOR,
                question_id=uuid4(),
                description="Strong herding",
                confidence_level=0.7,
                potential_advantage=0.15,
                detected_at=datetime.utcnow(),
                expiration_estimate=None,
                exploitation_strategy="Test",
            ),
        ]

        efficiency_score = tournament_analytics._calculate_market_efficiency_score(
            high_inefficiencies
        )
        assert 0.0 <= efficiency_score <= 1.0
        assert efficiency_score < 0.9  # Should indicate inefficient market

        # No inefficiencies scenario
        no_inefficiencies = []
        efficiency_score = tournament_analytics._calculate_market_efficiency_score(
            no_inefficiencies
        )
        assert efficiency_score == 1.0  # Perfectly efficient
