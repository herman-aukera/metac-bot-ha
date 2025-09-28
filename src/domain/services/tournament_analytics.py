"""
Tournament analytics service for competitive intelligence and strategic optimization.

This service provides tournament standings analysis, competitive positioning,
market inefficiency detection, and strategic opportunity identification.
"""

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID


class MarketInefficiencyType(Enum):
    """Types of market inefficiencies that can be detected."""

    OVERCONFIDENCE_BIAS = "overconfidence_bias"
    ANCHORING_BIAS = "anchoring_bias"
    HERDING_BEHAVIOR = "herding_behavior"
    RECENCY_BIAS = "recency_bias"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    UNDERREACTION = "underreaction"
    OVERREACTION = "overreaction"
    MOMENTUM_EFFECT = "momentum_effect"
    CONTRARIAN_OPPORTUNITY = "contrarian_opportunity"


class StrategicOpportunityType(Enum):
    """Types of strategic opportunities."""

    TIMING_ADVANTAGE = "timing_advantage"
    INFORMATION_EDGE = "information_edge"
    CONTRARIAN_POSITION = "contrarian_position"
    CONSENSUS_EXPLOITATION = "consensus_exploitation"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"
    LATE_MOVER_ADVANTAGE = "late_mover_advantage"
    EARLY_MOVER_ADVANTAGE = "early_mover_advantage"
    NICHE_EXPERTISE = "niche_expertise"


@dataclass
class CompetitorProfile:
    """Profile of a tournament competitor."""

    competitor_id: str
    username: Optional[str]
    current_ranking: Optional[int]
    total_score: Optional[float]
    questions_answered: int
    questions_resolved: int
    average_brier_score: Optional[float]
    calibration_score: Optional[float]
    prediction_patterns: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]
    last_updated: datetime


@dataclass
class MarketInefficiency:
    """Detected market inefficiency."""

    inefficiency_type: MarketInefficiencyType
    question_id: UUID
    description: str
    confidence_level: float
    potential_advantage: float
    detected_at: datetime
    expiration_estimate: Optional[datetime]
    exploitation_strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategicOpportunity:
    """Strategic opportunity for competitive advantage."""

    opportunity_type: StrategicOpportunityType
    question_id: Optional[UUID]
    title: str
    description: str
    potential_impact: float
    confidence: float
    time_sensitivity: float
    resource_requirements: Dict[str, float]
    recommended_actions: List[str]
    identified_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TournamentStandings:
    """Tournament standings and competitive analysis."""

    tournament_id: int
    our_ranking: Optional[int]
    our_score: Optional[float]
    total_participants: int
    top_performers: List[CompetitorProfile]
    our_percentile: Optional[float]
    score_distribution: Dict[str, float]
    competitive_gaps: Dict[str, float]
    improvement_opportunities: List[str]
    last_updated: datetime


class TournamentAnalytics:
    """
    Tournament analytics service for competitive intelligence.

    Provides tournament standings analysis, competitive positioning,
    market inefficiency detection, and strategic opportunity identification.
    """

    def __init__(self):
        """Initialize the tournament analytics service."""
        self.logger = logging.getLogger(__name__)

        # Competitive intelligence data
        self.competitor_profiles: Dict[str, CompetitorProfile] = {}
        self.tournament_standings: Dict[int, TournamentStandings] = {}
        self.market_inefficiencies: List[MarketInefficiency] = []
        self.strategic_opportunities: List[StrategicOpportunity] = []

        # Analysis parameters
        self.inefficiency_detection_threshold = 0.7
        self.opportunity_confidence_threshold = 0.6
        self.competitor_analysis_window_days = 30

        self.logger.info("Tournament analytics service initialized")

    def analyze_tournament_standings(
        self, tournament_id: int, standings_data: Dict[str, Any], our_user_id: str
    ) -> TournamentStandings:
        """
        Analyze tournament standings and competitive positioning.

        Args:
            tournament_id: Tournament identifier
            standings_data: Raw standings data from tournament API
            our_user_id: Our user identifier in the tournament

        Returns:
            Analyzed tournament standings with competitive intelligence
        """
        try:
            # Extract our performance
            our_ranking = None
            our_score = None
            our_percentile = None

            participants = standings_data.get("participants", [])
            total_participants = len(participants)

            # Find our position
            for i, participant in enumerate(participants):
                if participant.get("user_id") == our_user_id:
                    our_ranking = i + 1
                    our_score = participant.get("score", 0.0)
                    our_percentile = (
                        total_participants - our_ranking + 1
                    ) / total_participants
                    break

            # Analyze top performers
            top_performers = self._analyze_top_performers(participants[:10])

            # Calculate score distribution
            scores = [
                p.get("score", 0.0) for p in participants if p.get("score") is not None
            ]
            score_distribution = self._calculate_score_distribution(scores)

            # Identify competitive gaps
            competitive_gaps = self._identify_competitive_gaps(
                our_score, scores, our_ranking
            )

            # Generate improvement opportunities
            improvement_opportunities = self._generate_improvement_opportunities(
                our_ranking, our_score, top_performers, competitive_gaps
            )

            standings = TournamentStandings(
                tournament_id=tournament_id,
                our_ranking=our_ranking,
                our_score=our_score,
                total_participants=total_participants,
                top_performers=top_performers,
                our_percentile=our_percentile,
                score_distribution=score_distribution,
                competitive_gaps=competitive_gaps,
                improvement_opportunities=improvement_opportunities,
                last_updated=datetime.utcnow(),
            )

            self.tournament_standings[tournament_id] = standings

            self.logger.info(
                f"Analyzed tournament standings for tournament {tournament_id}: "
                f"Ranking {our_ranking}/{total_participants} ({our_percentile:.1%} percentile)"
            )

            return standings

        except Exception as e:
            self.logger.error(f"Error analyzing tournament standings: {e}")
            raise

    def _analyze_top_performers(
        self, top_participants: List[Dict[str, Any]]
    ) -> List[CompetitorProfile]:
        """Analyze top performers to understand competitive landscape."""
        top_performers = []

        for i, participant in enumerate(top_participants):
            try:
                profile = CompetitorProfile(
                    competitor_id=participant.get("user_id", f"unknown_{i}"),
                    username=participant.get("username"),
                    current_ranking=i + 1,
                    total_score=participant.get("score"),
                    questions_answered=participant.get("questions_answered", 0),
                    questions_resolved=participant.get("questions_resolved", 0),
                    average_brier_score=participant.get("average_brier_score"),
                    calibration_score=participant.get("calibration_score"),
                    prediction_patterns=self._analyze_prediction_patterns(participant),
                    strengths=self._identify_competitor_strengths(participant),
                    weaknesses=self._identify_competitor_weaknesses(participant),
                    last_updated=datetime.utcnow(),
                )
                top_performers.append(profile)

                # Update competitor profiles cache
                self.competitor_profiles[profile.competitor_id] = profile

            except Exception as e:
                self.logger.warning(f"Error analyzing competitor {i}: {e}")
                continue

        return top_performers

    def _analyze_prediction_patterns(
        self, participant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze prediction patterns for a competitor."""
        patterns = {
            "average_confidence": participant.get("average_confidence", 0.5),
            "prediction_frequency": participant.get("prediction_frequency", 0.0),
            "category_preferences": participant.get("category_preferences", {}),
            "timing_patterns": participant.get("timing_patterns", {}),
            "risk_profile": self._assess_risk_profile(participant),
        }

        return patterns

    def _assess_risk_profile(self, participant: Dict[str, Any]) -> str:
        """Assess competitor's risk profile."""
        avg_confidence = participant.get("average_confidence", 0.5)
        prediction_variance = participant.get("prediction_variance", 0.1)

        if avg_confidence > 0.8 and prediction_variance < 0.05:
            return "conservative"
        elif avg_confidence < 0.6 and prediction_variance > 0.15:
            return "aggressive"
        else:
            return "moderate"

    def _identify_competitor_strengths(self, participant: Dict[str, Any]) -> List[str]:
        """Identify competitor's strengths."""
        strengths = []

        if participant.get("average_brier_score", 1.0) < 0.2:
            strengths.append("excellent_accuracy")

        if participant.get("calibration_score", 0.0) > 0.8:
            strengths.append("well_calibrated")

        if participant.get("questions_answered", 0) >= 50:
            strengths.append("high_volume")

        if participant.get("prediction_frequency", 0.0) > 0.8:
            strengths.append("consistent_participation")

        category_performance = participant.get("category_performance", {})
        for category, performance in category_performance.items():
            if performance.get("brier_score", 1.0) < 0.15:
                strengths.append(f"expert_in_{category}")

        return strengths

    def _identify_competitor_weaknesses(self, participant: Dict[str, Any]) -> List[str]:
        """Identify competitor's weaknesses."""
        weaknesses = []

        if participant.get("average_brier_score", 0.0) > 0.35:
            weaknesses.append("poor_accuracy")

        if participant.get("calibration_score", 1.0) < 0.4:
            weaknesses.append("poorly_calibrated")

        if participant.get("prediction_variance", 0.0) > 0.2:
            weaknesses.append("inconsistent_predictions")

        if participant.get("questions_answered", 100) < 10:
            weaknesses.append("low_participation")

        timing_data = participant.get("timing_patterns", {})
        if timing_data.get("late_submissions", 0) > 0.5:
            weaknesses.append("poor_timing")

        return weaknesses

    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, float]:
        """Calculate score distribution statistics."""
        if not scores:
            return {}

        return {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "q25": (
                statistics.quantiles(scores, n=4)[0]
                if len(scores) >= 4
                else min(scores)
            ),
            "q75": (
                statistics.quantiles(scores, n=4)[2]
                if len(scores) >= 4
                else max(scores)
            ),
        }

    def _identify_competitive_gaps(
        self,
        our_score: Optional[float],
        all_scores: List[float],
        our_ranking: Optional[int],
    ) -> Dict[str, float]:
        """Identify competitive gaps and improvement targets."""
        gaps = {}

        if our_score is None or not all_scores:
            return gaps

        sorted_scores = sorted(all_scores, reverse=True)

        # Gap to next rank
        if our_ranking and our_ranking > 1:
            next_rank_score = sorted_scores[our_ranking - 2]
            gaps["next_rank"] = next_rank_score - our_score

        # Gap to top 10%
        top_10_threshold = sorted_scores[max(0, len(sorted_scores) // 10 - 1)]
        gaps["top_10_percent"] = max(0, top_10_threshold - our_score)

        # Gap to top 5%
        top_5_threshold = sorted_scores[max(0, len(sorted_scores) // 20 - 1)]
        gaps["top_5_percent"] = max(0, top_5_threshold - our_score)

        # Gap to leader
        if sorted_scores:
            gaps["leader"] = sorted_scores[0] - our_score

        return gaps

    def _generate_improvement_opportunities(
        self,
        our_ranking: Optional[int],
        our_score: Optional[float],
        top_performers: List[CompetitorProfile],
        competitive_gaps: Dict[str, float],
    ) -> List[str]:
        """Generate improvement opportunities based on competitive analysis."""
        opportunities = []

        if not our_ranking or not our_score:
            return ["Insufficient data for improvement analysis"]

        # Analyze top performers for patterns
        if top_performers:
            # Common strengths among top performers
            all_strengths = []
            for performer in top_performers:
                all_strengths.extend(performer.strengths)

            strength_counts = {}
            for strength in all_strengths:
                strength_counts[strength] = strength_counts.get(strength, 0) + 1

            # Most common strengths
            common_strengths = [
                strength
                for strength, count in strength_counts.items()
                if count
                >= len(top_performers) * 0.6  # 60% of top performers have this strength
            ]

            for strength in common_strengths:
                if strength == "excellent_accuracy":
                    opportunities.append(
                        "Focus on improving prediction accuracy through better research and reasoning"
                    )
                elif strength == "well_calibrated":
                    opportunities.append(
                        "Improve confidence calibration through systematic feedback analysis"
                    )
                elif strength == "high_volume":
                    opportunities.append(
                        "Increase participation rate to answer more questions"
                    )
                elif strength == "consistent_participation":
                    opportunities.append("Maintain more consistent prediction schedule")
                elif strength.startswith("expert_in_"):
                    category = strength.replace("expert_in_", "")
                    opportunities.append(f"Develop expertise in {category} questions")

        # Gap-based opportunities
        if competitive_gaps.get("next_rank", 0) < 5:  # Small gap to next rank
            opportunities.append(
                "Small gap to next rank - focus on consistency to move up"
            )

        if competitive_gaps.get("top_10_percent", 0) > 20:  # Large gap to top 10%
            opportunities.append(
                "Significant improvement needed - consider strategy overhaul"
            )

        return opportunities or ["Continue current strong performance"]

    def detect_market_inefficiencies(
        self,
        question_data: Dict[str, Any],
        community_predictions: List[Dict[str, Any]],
        historical_patterns: Optional[Dict[str, Any]] = None,
    ) -> List[MarketInefficiency]:
        """
        Detect market inefficiencies that can be exploited for competitive advantage.

        Args:
            question_data: Question metadata and context
            community_predictions: Community prediction data
            historical_patterns: Historical patterns for similar questions

        Returns:
            List of detected market inefficiencies
        """
        try:
            inefficiencies = []
            question_id = UUID(
                question_data.get("id", "00000000-0000-0000-0000-000000000000")
            )

            # Analyze prediction distribution
            predictions = [
                p.get("prediction", 0.5)
                for p in community_predictions
                if p.get("prediction") is not None
            ]

            if len(predictions) < 5:  # Not enough data
                return inefficiencies

            # Detect overconfidence bias
            overconfidence = self._detect_overconfidence_bias(
                predictions, question_data
            )
            if overconfidence:
                inefficiencies.append(overconfidence)

            # Detect herding behavior
            herding = self._detect_herding_behavior(predictions, community_predictions)
            if herding:
                inefficiencies.append(herding)

            # Detect anchoring bias
            anchoring = self._detect_anchoring_bias(
                predictions, question_data, historical_patterns
            )
            if anchoring:
                inefficiencies.append(anchoring)

            # Detect recency bias
            recency = self._detect_recency_bias(community_predictions, question_data)
            if recency:
                inefficiencies.append(recency)

            # Detect momentum/contrarian opportunities
            momentum = self._detect_momentum_patterns(
                predictions, community_predictions
            )
            if momentum:
                inefficiencies.extend(momentum)

            self.market_inefficiencies.extend(inefficiencies)

            self.logger.info(
                f"Detected {len(inefficiencies)} market inefficiencies for question {question_id}"
            )

            return inefficiencies

        except Exception as e:
            self.logger.error(f"Error detecting market inefficiencies: {e}")
            return []

    def _detect_overconfidence_bias(
        self, predictions: List[float], question_data: Dict[str, Any]
    ) -> Optional[MarketInefficiency]:
        """Detect overconfidence bias in community predictions."""
        if not predictions:
            return None

        # Check for extreme predictions (very close to 0 or 1)
        extreme_predictions = [p for p in predictions if p < 0.1 or p > 0.9]
        extreme_ratio = len(extreme_predictions) / len(predictions)

        if extreme_ratio > 0.3:  # More than 30% extreme predictions
            confidence = min(0.9, extreme_ratio)

            return MarketInefficiency(
                inefficiency_type=MarketInefficiencyType.OVERCONFIDENCE_BIAS,
                question_id=UUID(
                    question_data.get("id", "00000000-0000-0000-0000-000000000000")
                ),
                description=f"High proportion ({extreme_ratio:.1%}) of extreme predictions suggests overconfidence",
                confidence_level=confidence,
                potential_advantage=0.1 + (extreme_ratio - 0.3) * 0.2,
                detected_at=datetime.utcnow(),
                expiration_estimate=datetime.utcnow() + timedelta(days=3),
                exploitation_strategy="Consider more moderate predictions with better calibration",
                metadata={
                    "extreme_ratio": extreme_ratio,
                    "extreme_predictions_count": len(extreme_predictions),
                    "total_predictions": len(predictions),
                },
            )

        return None

    def _detect_herding_behavior(
        self, predictions: List[float], community_predictions: List[Dict[str, Any]]
    ) -> Optional[MarketInefficiency]:
        """Detect herding behavior in community predictions."""
        if len(predictions) < 10:
            return None

        # Calculate prediction clustering
        prediction_std = statistics.stdev(predictions)
        prediction_mean = statistics.mean(predictions)

        # Check for unusual clustering (low variance)
        if (
            prediction_std < 0.05 and 0.2 < prediction_mean < 0.8
        ):  # Tight clustering away from extremes
            # Check if this clustering happened recently (herding)
            current_time = datetime.utcnow()
            recent_predictions = []
            for p in community_predictions:
                if p.get("timestamp"):
                    try:
                        timestamp_str = p["timestamp"].replace("Z", "+00:00")
                        timestamp = datetime.fromisoformat(timestamp_str)
                        # Convert to UTC if timezone-aware
                        if timestamp.tzinfo is not None:
                            timestamp = timestamp.replace(tzinfo=None)
                        if (
                            current_time - timestamp
                        ).total_seconds() < 86400:  # Last 24 hours
                            recent_predictions.append(p)
                    except (ValueError, AttributeError):
                        continue

            if (
                len(recent_predictions) > len(community_predictions) * 0.5
            ):  # Most predictions are recent
                return MarketInefficiency(
                    inefficiency_type=MarketInefficiencyType.HERDING_BEHAVIOR,
                    question_id=UUID(
                        community_predictions[0].get(
                            "question_id", "00000000-0000-0000-0000-000000000000"
                        )
                    ),
                    description=f"Tight clustering of predictions (std={prediction_std:.3f}) suggests herding",
                    confidence_level=0.7,
                    potential_advantage=0.15,
                    detected_at=datetime.utcnow(),
                    expiration_estimate=datetime.utcnow() + timedelta(days=2),
                    exploitation_strategy="Consider contrarian position if you have independent information",
                    metadata={
                        "prediction_std": prediction_std,
                        "prediction_mean": prediction_mean,
                        "recent_predictions_ratio": len(recent_predictions)
                        / len(community_predictions),
                    },
                )

        return None

    def _detect_anchoring_bias(
        self,
        predictions: List[float],
        question_data: Dict[str, Any],
        historical_patterns: Optional[Dict[str, Any]],
    ) -> Optional[MarketInefficiency]:
        """Detect anchoring bias based on question framing or initial predictions."""
        if not predictions or not historical_patterns:
            return None

        # Look for anchoring to round numbers or question framing
        question_data.get("title", "").lower()

        # Check for clustering around round numbers
        round_numbers = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

        for round_num in round_numbers:
            close_predictions = [p for p in predictions if abs(p - round_num) < 0.02]
            if (
                len(close_predictions) > len(predictions) * 0.4
            ):  # 40% cluster around round number
                return MarketInefficiency(
                    inefficiency_type=MarketInefficiencyType.ANCHORING_BIAS,
                    question_id=UUID(
                        question_data.get("id", "00000000-0000-0000-0000-000000000000")
                    ),
                    description=f"Clustering around {round_num} suggests anchoring bias",
                    confidence_level=0.6,
                    potential_advantage=0.1,
                    detected_at=datetime.utcnow(),
                    expiration_estimate=datetime.utcnow() + timedelta(days=1),
                    exploitation_strategy="Analyze question independently without anchoring to round numbers",
                    metadata={
                        "anchor_value": round_num,
                        "clustered_predictions": len(close_predictions),
                        "cluster_ratio": len(close_predictions) / len(predictions),
                    },
                )

        return None

    def _detect_recency_bias(
        self, community_predictions: List[Dict[str, Any]], question_data: Dict[str, Any]
    ) -> Optional[MarketInefficiency]:
        """Detect recency bias in prediction updates."""
        if len(community_predictions) < 10:
            return None

        # Sort predictions by timestamp
        timestamped_predictions = [
            p
            for p in community_predictions
            if p.get("timestamp") and p.get("prediction") is not None
        ]

        if len(timestamped_predictions) < 5:
            return None

        try:
            # Sort predictions by timestamp, handling timezone issues
            def parse_timestamp(pred):
                timestamp_str = pred["timestamp"].replace("Z", "+00:00")
                timestamp = datetime.fromisoformat(timestamp_str)
                # Convert to UTC if timezone-aware
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
                return timestamp

            timestamped_predictions.sort(key=parse_timestamp)

            # Check for significant shifts in recent predictions
            recent_predictions = timestamped_predictions[-5:]  # Last 5 predictions
            older_predictions = timestamped_predictions[:-5]

            if older_predictions:
                recent_mean = statistics.mean(
                    [p["prediction"] for p in recent_predictions]
                )
                older_mean = statistics.mean(
                    [p["prediction"] for p in older_predictions]
                )

                shift_magnitude = abs(recent_mean - older_mean)

                if shift_magnitude > 0.15:  # Significant shift
                    return MarketInefficiency(
                        inefficiency_type=MarketInefficiencyType.RECENCY_BIAS,
                        question_id=UUID(
                            question_data.get(
                                "id", "00000000-0000-0000-0000-000000000000"
                            )
                        ),
                        description=f"Recent predictions shifted by {shift_magnitude:.2f} from historical average",
                        confidence_level=0.6,
                        potential_advantage=0.12,
                        detected_at=datetime.utcnow(),
                        expiration_estimate=datetime.utcnow() + timedelta(days=2),
                        exploitation_strategy="Consider whether recent events are truly informative or just recency bias",
                        metadata={
                            "recent_mean": recent_mean,
                            "older_mean": older_mean,
                            "shift_magnitude": shift_magnitude,
                        },
                    )

        except (ValueError, KeyError):
            pass

        return None

    def _detect_momentum_patterns(
        self, predictions: List[float], community_predictions: List[Dict[str, Any]]
    ) -> List[MarketInefficiency]:
        """Detect momentum and contrarian opportunities."""
        inefficiencies = []

        if len(community_predictions) < 10:
            return inefficiencies

        # Analyze prediction trends
        timestamped_predictions = [
            p
            for p in community_predictions
            if p.get("timestamp") and p.get("prediction") is not None
        ]

        if len(timestamped_predictions) < 8:
            return inefficiencies

        try:
            # Sort predictions by timestamp, handling timezone issues
            def parse_timestamp_momentum(pred):
                timestamp_str = pred["timestamp"].replace("Z", "+00:00")
                timestamp = datetime.fromisoformat(timestamp_str)
                # Convert to UTC if timezone-aware
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
                return timestamp

            timestamped_predictions.sort(key=parse_timestamp_momentum)

            # Calculate trend
            recent_window = timestamped_predictions[-6:]  # Last 6 predictions
            trend_values = [p["prediction"] for p in recent_window]

            # Simple trend calculation
            if len(trend_values) >= 4:
                early_mean = statistics.mean(trend_values[:3])
                late_mean = statistics.mean(trend_values[-3:])
                trend_strength = late_mean - early_mean

                # Strong upward momentum
                if trend_strength > 0.1:
                    inefficiencies.append(
                        MarketInefficiency(
                            inefficiency_type=MarketInefficiencyType.MOMENTUM_EFFECT,
                            question_id=UUID(
                                community_predictions[0].get(
                                    "question_id",
                                    "00000000-0000-0000-0000-000000000000",
                                )
                            ),
                            description=f"Strong upward momentum detected (trend: +{trend_strength:.2f})",
                            confidence_level=0.6,
                            potential_advantage=0.08,
                            detected_at=datetime.utcnow(),
                            expiration_estimate=datetime.utcnow() + timedelta(days=1),
                            exploitation_strategy="Consider whether momentum will continue or reverse",
                            metadata={
                                "trend_strength": trend_strength,
                                "direction": "upward",
                            },
                        )
                    )

                # Strong downward momentum
                elif trend_strength < -0.1:
                    inefficiencies.append(
                        MarketInefficiency(
                            inefficiency_type=MarketInefficiencyType.MOMENTUM_EFFECT,
                            question_id=UUID(
                                community_predictions[0].get(
                                    "question_id",
                                    "00000000-0000-0000-0000-000000000000",
                                )
                            ),
                            description=f"Strong downward momentum detected (trend: {trend_strength:.2f})",
                            confidence_level=0.6,
                            potential_advantage=0.08,
                            detected_at=datetime.utcnow(),
                            expiration_estimate=datetime.utcnow() + timedelta(days=1),
                            exploitation_strategy="Consider whether momentum will continue or reverse",
                            metadata={
                                "trend_strength": trend_strength,
                                "direction": "downward",
                            },
                        )
                    )

                # Potential contrarian opportunity
                if abs(trend_strength) > 0.15:
                    inefficiencies.append(
                        MarketInefficiency(
                            inefficiency_type=MarketInefficiencyType.CONTRARIAN_OPPORTUNITY,
                            question_id=UUID(
                                community_predictions[0].get(
                                    "question_id",
                                    "00000000-0000-0000-0000-000000000000",
                                )
                            ),
                            description=f"Strong trend ({trend_strength:.2f}) may present contrarian opportunity",
                            confidence_level=0.5,
                            potential_advantage=0.12,
                            detected_at=datetime.utcnow(),
                            expiration_estimate=datetime.utcnow() + timedelta(days=1),
                            exploitation_strategy="Analyze if trend is overdone and consider contrarian position",
                            metadata={"trend_strength": trend_strength},
                        )
                    )

        except (ValueError, KeyError):
            pass

        return inefficiencies

    def identify_strategic_opportunities(
        self,
        tournament_context: Dict[str, Any],
        our_performance: Dict[str, Any],
        question_pipeline: List[Dict[str, Any]],
    ) -> List[StrategicOpportunity]:
        """
        Identify strategic opportunities for competitive advantage.

        Args:
            tournament_context: Current tournament context and standings
            our_performance: Our current performance metrics
            question_pipeline: Upcoming questions and opportunities

        Returns:
            List of strategic opportunities
        """
        try:
            opportunities = []

            # Timing-based opportunities
            timing_opportunities = self._identify_timing_opportunities(
                question_pipeline, tournament_context
            )
            opportunities.extend(timing_opportunities)

            # Information edge opportunities
            info_opportunities = self._identify_information_edge_opportunities(
                question_pipeline, our_performance
            )
            opportunities.extend(info_opportunities)

            # Competitive positioning opportunities
            positioning_opportunities = self._identify_positioning_opportunities(
                tournament_context, our_performance
            )
            opportunities.extend(positioning_opportunities)

            # Resource allocation opportunities
            resource_opportunities = self._identify_resource_allocation_opportunities(
                question_pipeline, our_performance
            )
            opportunities.extend(resource_opportunities)

            # Store opportunities
            self.strategic_opportunities.extend(opportunities)

            # Clean up old opportunities
            self._cleanup_expired_opportunities()

            self.logger.info(f"Identified {len(opportunities)} strategic opportunities")

            return opportunities

        except Exception as e:
            self.logger.error(f"Error identifying strategic opportunities: {e}")
            return []

    def _identify_timing_opportunities(
        self,
        question_pipeline: List[Dict[str, Any]],
        tournament_context: Dict[str, Any],
    ) -> List[StrategicOpportunity]:
        """Identify timing-based strategic opportunities."""
        opportunities = []

        for question in question_pipeline:
            try:
                question_id = UUID(
                    question.get("id", "00000000-0000-0000-0000-000000000000")
                )
                deadline = question.get("deadline")

                if not deadline:
                    continue

                deadline_str = deadline.replace("Z", "+00:00")
                deadline_dt = datetime.fromisoformat(deadline_str)
                # Convert to UTC if timezone-aware
                if deadline_dt.tzinfo is not None:
                    deadline_dt = deadline_dt.replace(tzinfo=None)
                time_to_deadline = (
                    deadline_dt - datetime.utcnow()
                ).total_seconds() / 3600  # hours

                # Early mover advantage
                if time_to_deadline > 48 and question.get("prediction_count", 0) < 10:
                    opportunities.append(
                        StrategicOpportunity(
                            opportunity_type=StrategicOpportunityType.EARLY_MOVER_ADVANTAGE,
                            question_id=question_id,
                            title="Early Mover Advantage",
                            description=f"Question has few predictions ({question.get('prediction_count', 0)}) with long deadline",
                            potential_impact=0.15,
                            confidence=0.7,
                            time_sensitivity=0.8,
                            resource_requirements={
                                "research_time": 2.0,
                                "analysis_depth": 0.8,
                            },
                            recommended_actions=[
                                "Conduct thorough research before others",
                                "Submit high-quality prediction early",
                                "Monitor for information changes",
                            ],
                            identified_at=datetime.utcnow(),
                            metadata={"time_to_deadline_hours": time_to_deadline},
                        )
                    )

                # Late mover advantage
                elif time_to_deadline < 12 and question.get("prediction_count", 0) > 20:
                    opportunities.append(
                        StrategicOpportunity(
                            opportunity_type=StrategicOpportunityType.LATE_MOVER_ADVANTAGE,
                            question_id=question_id,
                            title="Late Mover Information Advantage",
                            description="Can benefit from observing community predictions and recent information",
                            potential_impact=0.12,
                            confidence=0.6,
                            time_sensitivity=0.9,
                            resource_requirements={
                                "research_time": 1.5,
                                "analysis_depth": 0.6,
                            },
                            recommended_actions=[
                                "Analyze community prediction patterns",
                                "Look for recent information updates",
                                "Submit refined prediction",
                            ],
                            identified_at=datetime.utcnow(),
                            metadata={"time_to_deadline_hours": time_to_deadline},
                        )
                    )

            except (ValueError, KeyError):
                continue

        return opportunities

    def _identify_information_edge_opportunities(
        self, question_pipeline: List[Dict[str, Any]], our_performance: Dict[str, Any]
    ) -> List[StrategicOpportunity]:
        """Identify opportunities where we might have an information edge."""
        opportunities = []

        # Analyze our historical performance by category
        category_performance = our_performance.get("category_performance", {})

        for question in question_pipeline:
            try:
                question_id = UUID(
                    question.get("id", "00000000-0000-0000-0000-000000000000")
                )
                category = question.get("category", "general")

                # Check if we have strong performance in this category
                if category in category_performance:
                    cat_perf = category_performance[category]
                    if (
                        cat_perf.get("brier_score", 1.0) < 0.2
                        and cat_perf.get("question_count", 0) >= 5
                    ):
                        opportunities.append(
                            StrategicOpportunity(
                                opportunity_type=StrategicOpportunityType.NICHE_EXPERTISE,
                                question_id=question_id,
                                title=f"Expertise in {category}",
                                description=f"Strong historical performance in {category} category",
                                potential_impact=0.2,
                                confidence=0.8,
                                time_sensitivity=0.5,
                                resource_requirements={
                                    "research_time": 1.0,
                                    "analysis_depth": 0.9,
                                },
                                recommended_actions=[
                                    f"Leverage expertise in {category}",
                                    "Apply specialized knowledge",
                                    "Consider higher confidence prediction",
                                ],
                                identified_at=datetime.utcnow(),
                                metadata={
                                    "category": category,
                                    "historical_brier": cat_perf.get("brier_score"),
                                    "question_count": cat_perf.get("question_count"),
                                },
                            )
                        )

            except (ValueError, KeyError):
                continue

        return opportunities

    def _identify_positioning_opportunities(
        self, tournament_context: Dict[str, Any], our_performance: Dict[str, Any]
    ) -> List[StrategicOpportunity]:
        """Identify competitive positioning opportunities."""
        opportunities = []

        our_ranking = tournament_context.get("our_ranking")
        total_participants = tournament_context.get("total_participants", 100)

        if not our_ranking:
            return opportunities

        # Identify positioning strategies based on current rank
        percentile = our_ranking / total_participants

        if percentile > 0.8:  # Bottom 20%
            opportunities.append(
                StrategicOpportunity(
                    opportunity_type=StrategicOpportunityType.CONSENSUS_EXPLOITATION,
                    question_id=None,
                    title="Aggressive Improvement Strategy",
                    description="Low ranking allows for higher-risk, higher-reward strategies",
                    potential_impact=0.3,
                    confidence=0.6,
                    time_sensitivity=0.7,
                    resource_requirements={"research_time": 2.5, "risk_tolerance": 0.8},
                    recommended_actions=[
                        "Take contrarian positions when confident",
                        "Focus on high-impact questions",
                        "Increase prediction volume",
                    ],
                    identified_at=datetime.utcnow(),
                    metadata={"current_percentile": percentile},
                )
            )

        elif 0.2 < percentile < 0.4:  # Middle-upper range
            opportunities.append(
                StrategicOpportunity(
                    opportunity_type=StrategicOpportunityType.VOLATILITY_ARBITRAGE,
                    question_id=None,
                    title="Selective Optimization Strategy",
                    description="Good position allows for selective high-confidence plays",
                    potential_impact=0.15,
                    confidence=0.7,
                    time_sensitivity=0.5,
                    resource_requirements={"research_time": 2.0, "selectivity": 0.8},
                    recommended_actions=[
                        "Focus on highest-confidence predictions",
                        "Avoid unnecessary risks",
                        "Maintain consistent quality",
                    ],
                    identified_at=datetime.utcnow(),
                    metadata={"current_percentile": percentile},
                )
            )

        return opportunities

    def _identify_resource_allocation_opportunities(
        self, question_pipeline: List[Dict[str, Any]], our_performance: Dict[str, Any]
    ) -> List[StrategicOpportunity]:
        """Identify optimal resource allocation opportunities."""
        opportunities = []

        # Analyze question scoring potential
        high_value_questions = []
        for question in question_pipeline:
            scoring_potential = question.get("scoring_potential", 1.0)
            difficulty = question.get("difficulty", 0.5)

            # High value = high scoring potential, moderate difficulty
            value_score = scoring_potential * (1 - abs(difficulty - 0.5))

            if value_score > 0.7:
                high_value_questions.append((question, value_score))

        if high_value_questions:
            # Sort by value score
            high_value_questions.sort(key=lambda x: x[1], reverse=True)
            top_questions = high_value_questions[:3]  # Top 3 opportunities

            for question, value_score in top_questions:
                question_id = UUID(
                    question.get("id", "00000000-0000-0000-0000-000000000000")
                )

                opportunities.append(
                    StrategicOpportunity(
                        opportunity_type=StrategicOpportunityType.TIMING_ADVANTAGE,
                        question_id=question_id,
                        title="High-Value Question Opportunity",
                        description=f"Question with high scoring potential ({value_score:.2f})",
                        potential_impact=value_score * 0.2,
                        confidence=0.7,
                        time_sensitivity=0.6,
                        resource_requirements={
                            "research_time": 2.0 + value_score,
                            "analysis_depth": 0.8 + (value_score - 0.7) * 0.5,
                        },
                        recommended_actions=[
                            "Allocate extra research time",
                            "Use best available agents",
                            "Validate prediction thoroughly",
                        ],
                        identified_at=datetime.utcnow(),
                        metadata={
                            "value_score": value_score,
                            "scoring_potential": question.get("scoring_potential"),
                            "difficulty": question.get("difficulty"),
                        },
                    )
                )

        return opportunities

    def _cleanup_expired_opportunities(self) -> None:
        """Clean up expired strategic opportunities."""
        current_time = datetime.utcnow()

        # Remove opportunities older than 7 days
        self.strategic_opportunities = [
            opp
            for opp in self.strategic_opportunities
            if (current_time - opp.identified_at).days < 7
        ]

    def analyze_performance_attribution(
        self,
        our_performance_data: Dict[str, Any],
        competitor_performance_data: List[Dict[str, Any]],
        question_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze performance attribution to identify what drives performance differences.

        Args:
            our_performance_data: Our detailed performance metrics
            competitor_performance_data: Competitor performance data
            question_history: Historical question and prediction data

        Returns:
            Detailed performance attribution analysis
        """
        try:
            attribution = {
                "accuracy_drivers": self._analyze_accuracy_drivers(
                    our_performance_data, question_history
                ),
                "calibration_analysis": self._analyze_calibration_factors(
                    our_performance_data, question_history
                ),
                "category_performance": self._analyze_category_performance_attribution(
                    our_performance_data, competitor_performance_data
                ),
                "timing_impact": self._analyze_timing_impact(
                    our_performance_data, question_history
                ),
                "confidence_optimization": self._analyze_confidence_optimization(
                    our_performance_data, question_history
                ),
                "competitive_advantages": self._identify_competitive_advantages(
                    our_performance_data, competitor_performance_data
                ),
                "performance_gaps": self._analyze_performance_gaps(
                    our_performance_data, competitor_performance_data
                ),
            }

            self.logger.info("Completed performance attribution analysis")
            return attribution

        except Exception as e:
            self.logger.error(f"Error in performance attribution analysis: {e}")
            return {"error": str(e)}

    def _analyze_accuracy_drivers(
        self, our_performance: Dict[str, Any], question_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze what drives prediction accuracy."""
        accuracy_drivers = {
            "research_depth_correlation": 0.0,
            "confidence_accuracy_relationship": {},
            "question_complexity_impact": {},
            "information_quality_impact": 0.0,
            "reasoning_method_effectiveness": {},
        }

        if not question_history:
            return accuracy_drivers

        # Analyze research depth vs accuracy
        research_scores = []
        accuracy_scores = []
        for question in question_history:
            if question.get("research_depth") and question.get("brier_score"):
                research_scores.append(question["research_depth"])
                accuracy_scores.append(
                    1 - question["brier_score"]
                )  # Convert to accuracy

        if len(research_scores) >= 3:
            # Simple correlation calculation
            if len(set(research_scores)) > 1:  # Avoid division by zero
                correlation = self._calculate_correlation(
                    research_scores, accuracy_scores
                )
                accuracy_drivers["research_depth_correlation"] = correlation

        # Analyze confidence vs accuracy by confidence bands
        confidence_bands = {"low": [], "medium": [], "high": []}
        for question in question_history:
            confidence = question.get("confidence", 0.5)
            accuracy = 1 - question.get("brier_score", 0.5)

            if confidence < 0.6:
                confidence_bands["low"].append(accuracy)
            elif confidence < 0.8:
                confidence_bands["medium"].append(accuracy)
            else:
                confidence_bands["high"].append(accuracy)

        for band, accuracies in confidence_bands.items():
            if accuracies:
                accuracy_drivers["confidence_accuracy_relationship"][band] = {
                    "average_accuracy": statistics.mean(accuracies),
                    "sample_size": len(accuracies),
                }

        return accuracy_drivers

    def _analyze_calibration_factors(
        self, our_performance: Dict[str, Any], question_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze factors affecting calibration."""
        calibration_analysis = {
            "overall_calibration": our_performance.get("calibration_score", 0.0),
            "calibration_by_category": {},
            "calibration_drift_over_time": [],
            "overconfidence_patterns": {},
            "underconfidence_patterns": {},
        }

        # Analyze calibration by category
        category_predictions = {}
        for question in question_history:
            category = question.get("category", "general")
            if category not in category_predictions:
                category_predictions[category] = []

            if question.get("confidence") and question.get("was_correct") is not None:
                category_predictions[category].append(
                    {
                        "confidence": question["confidence"],
                        "correct": question["was_correct"],
                    }
                )

        for category, predictions in category_predictions.items():
            if len(predictions) >= 5:  # Minimum sample size
                calibration_score = self._calculate_calibration_score(predictions)
                calibration_analysis["calibration_by_category"][category] = {
                    "calibration_score": calibration_score,
                    "sample_size": len(predictions),
                }

        return calibration_analysis

    def _analyze_category_performance_attribution(
        self,
        our_performance: Dict[str, Any],
        competitor_performance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze performance attribution by question category."""
        category_attribution = {}

        our_categories = our_performance.get("category_performance", {})

        for category, our_perf in our_categories.items():
            # Find competitor performance in same category
            competitor_scores = []
            for competitor in competitor_performance:
                comp_categories = competitor.get("category_performance", {})
                if category in comp_categories:
                    competitor_scores.append(
                        comp_categories[category].get("brier_score", 0.5)
                    )

            if competitor_scores:
                our_brier = our_perf.get("brier_score", 0.5)
                avg_competitor_brier = statistics.mean(competitor_scores)

                category_attribution[category] = {
                    "our_brier_score": our_brier,
                    "competitor_average_brier": avg_competitor_brier,
                    "relative_performance": avg_competitor_brier
                    - our_brier,  # Positive = we're better
                    "percentile_rank": sum(
                        1 for score in competitor_scores if score > our_brier
                    )
                    / len(competitor_scores),
                    "sample_size": our_perf.get("question_count", 0),
                    "competitive_advantage": our_brier < avg_competitor_brier,
                }

        return category_attribution

    def _analyze_timing_impact(
        self, our_performance: Dict[str, Any], question_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze impact of prediction timing on performance."""
        timing_analysis = {
            "early_vs_late_performance": {},
            "optimal_timing_windows": [],
            "deadline_pressure_impact": 0.0,
        }

        early_predictions = []
        late_predictions = []

        for question in question_history:
            time_to_deadline = question.get("hours_before_deadline", 24)
            accuracy = 1 - question.get("brier_score", 0.5)

            if time_to_deadline > 48:  # Early prediction
                early_predictions.append(accuracy)
            elif time_to_deadline < 12:  # Late prediction
                late_predictions.append(accuracy)

        if early_predictions and late_predictions:
            timing_analysis["early_vs_late_performance"] = {
                "early_average_accuracy": statistics.mean(early_predictions),
                "late_average_accuracy": statistics.mean(late_predictions),
                "early_sample_size": len(early_predictions),
                "late_sample_size": len(late_predictions),
                "timing_advantage": statistics.mean(early_predictions)
                - statistics.mean(late_predictions),
            }

        return timing_analysis

    def _analyze_confidence_optimization(
        self, our_performance: Dict[str, Any], question_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze confidence level optimization opportunities."""
        confidence_analysis = {
            "optimal_confidence_ranges": {},
            "overconfidence_cost": 0.0,
            "underconfidence_opportunity": 0.0,
            "confidence_calibration_recommendations": [],
        }

        # Group predictions by confidence level
        confidence_buckets = {
            "very_low": [],  # 0.0-0.4
            "low": [],  # 0.4-0.6
            "medium": [],  # 0.6-0.8
            "high": [],  # 0.8-0.95
            "very_high": [],  # 0.95-1.0
        }

        for question in question_history:
            confidence = question.get("confidence", 0.5)
            brier_score = question.get("brier_score", 0.5)

            if confidence < 0.4:
                confidence_buckets["very_low"].append(brier_score)
            elif confidence < 0.6:
                confidence_buckets["low"].append(brier_score)
            elif confidence < 0.8:
                confidence_buckets["medium"].append(brier_score)
            elif confidence < 0.95:
                confidence_buckets["high"].append(brier_score)
            else:
                confidence_buckets["very_high"].append(brier_score)

        # Analyze performance in each bucket
        for bucket, scores in confidence_buckets.items():
            if scores:
                confidence_analysis["optimal_confidence_ranges"][bucket] = {
                    "average_brier_score": statistics.mean(scores),
                    "sample_size": len(scores),
                    "performance_quality": (
                        "excellent"
                        if statistics.mean(scores) < 0.2
                        else (
                            "good"
                            if statistics.mean(scores) < 0.3
                            else "needs_improvement"
                        )
                    ),
                }

        return confidence_analysis

    def _identify_competitive_advantages(
        self,
        our_performance: Dict[str, Any],
        competitor_performance: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify our competitive advantages over other participants."""
        advantages = []

        our_brier = our_performance.get("average_brier_score", 0.5)
        our_calibration = our_performance.get("calibration_score", 0.5)
        our_volume = our_performance.get("questions_answered", 0)

        if competitor_performance:
            competitor_briers = [
                c.get("average_brier_score", 0.5) for c in competitor_performance
            ]
            competitor_calibrations = [
                c.get("calibration_score", 0.5) for c in competitor_performance
            ]
            competitor_volumes = [
                c.get("questions_answered", 0) for c in competitor_performance
            ]

            # Accuracy advantage
            if our_brier < statistics.mean(competitor_briers):
                percentile = sum(1 for b in competitor_briers if b > our_brier) / len(
                    competitor_briers
                )
                advantages.append(
                    {
                        "type": "accuracy_advantage",
                        "description": f"Superior accuracy (top {percentile:.0%} of competitors)",
                        "strength": percentile,
                        "recommendation": "Leverage accuracy advantage in high-stakes questions",
                    }
                )

            # Calibration advantage
            if our_calibration > statistics.mean(competitor_calibrations):
                percentile = sum(
                    1 for c in competitor_calibrations if c < our_calibration
                ) / len(competitor_calibrations)
                advantages.append(
                    {
                        "type": "calibration_advantage",
                        "description": f"Better calibrated confidence (top {percentile:.0%} of competitors)",
                        "strength": percentile,
                        "recommendation": "Use confidence levels strategically for tournament scoring",
                    }
                )

            # Volume advantage
            if our_volume > statistics.mean(competitor_volumes):
                percentile = sum(1 for v in competitor_volumes if v < our_volume) / len(
                    competitor_volumes
                )
                advantages.append(
                    {
                        "type": "participation_advantage",
                        "description": f"Higher participation rate (top {percentile:.0%} of competitors)",
                        "strength": percentile,
                        "recommendation": "Maintain high participation to maximize scoring opportunities",
                    }
                )

        return advantages

    def _analyze_performance_gaps(
        self,
        our_performance: Dict[str, Any],
        competitor_performance: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Analyze performance gaps that need addressing."""
        gaps = []

        our_brier = our_performance.get("average_brier_score", 0.5)
        our_calibration = our_performance.get("calibration_score", 0.5)

        if competitor_performance:
            # Find top performers
            top_performers = sorted(
                competitor_performance, key=lambda x: x.get("average_brier_score", 1.0)
            )[:5]

            if top_performers:
                top_avg_brier = statistics.mean(
                    [p.get("average_brier_score", 0.5) for p in top_performers]
                )

                if our_brier > top_avg_brier:
                    gap_size = our_brier - top_avg_brier
                    gaps.append(
                        {
                            "type": "accuracy_gap",
                            "description": f"Accuracy gap to top performers: {gap_size:.3f} Brier score points",
                            "severity": (
                                "high"
                                if gap_size > 0.1
                                else "medium"
                                if gap_size > 0.05
                                else "low"
                            ),
                            "improvement_potential": gap_size
                            * 100,  # Rough scoring improvement estimate
                            "recommendations": [
                                "Improve research methodology",
                                "Enhance reasoning processes",
                                "Focus on question categories with largest gaps",
                            ],
                        }
                    )

                top_avg_calibration = statistics.mean(
                    [p.get("calibration_score", 0.5) for p in top_performers]
                )

                if our_calibration < top_avg_calibration:
                    gap_size = top_avg_calibration - our_calibration
                    gaps.append(
                        {
                            "type": "calibration_gap",
                            "description": f"Calibration gap to top performers: {gap_size:.3f} points",
                            "severity": (
                                "high"
                                if gap_size > 0.2
                                else "medium"
                                if gap_size > 0.1
                                else "low"
                            ),
                            "improvement_potential": gap_size
                            * 50,  # Rough scoring improvement estimate
                            "recommendations": [
                                "Implement systematic confidence calibration",
                                "Track prediction outcomes more carefully",
                                "Adjust confidence based on historical performance",
                            ],
                        }
                    )

        return gaps

    def generate_optimization_recommendations(
        self,
        tournament_id: int,
        performance_attribution: Dict[str, Any],
        current_standings: Optional[TournamentStandings] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate specific optimization recommendations based on performance analysis.

        Args:
            tournament_id: Tournament identifier
            performance_attribution: Results from performance attribution analysis
            current_standings: Current tournament standings

        Returns:
            List of actionable optimization recommendations
        """
        try:
            recommendations = []

            # Accuracy optimization recommendations
            accuracy_drivers = performance_attribution.get("accuracy_drivers", {})
            if accuracy_drivers.get("research_depth_correlation", 0) > 0.3:
                recommendations.append(
                    {
                        "category": "research_optimization",
                        "priority": "high",
                        "title": "Increase Research Depth",
                        "description": "Strong correlation between research depth and accuracy detected",
                        "specific_actions": [
                            "Allocate more time for question research",
                            "Use additional information sources",
                            "Implement deeper fact-checking processes",
                        ],
                        "expected_impact": "5-15% improvement in accuracy",
                        "implementation_effort": "medium",
                        "timeline": "immediate",
                    }
                )

            # Calibration optimization recommendations
            calibration_analysis = performance_attribution.get(
                "calibration_analysis", {}
            )
            overall_calibration = calibration_analysis.get("overall_calibration", 0.5)
            if overall_calibration < 0.7:
                recommendations.append(
                    {
                        "category": "calibration_improvement",
                        "priority": "high",
                        "title": "Improve Confidence Calibration",
                        "description": f"Current calibration score ({overall_calibration:.2f}) below optimal range",
                        "specific_actions": [
                            "Implement systematic confidence adjustment based on historical performance",
                            "Track calibration metrics by question category",
                            "Use external calibration benchmarks",
                        ],
                        "expected_impact": "10-20% improvement in tournament scoring",
                        "implementation_effort": "medium",
                        "timeline": "1-2 weeks",
                    }
                )

            # Category-specific recommendations
            category_performance = performance_attribution.get(
                "category_performance", {}
            )
            for category, perf in category_performance.items():
                if (
                    not perf.get("competitive_advantage", False)
                    and perf.get("sample_size", 0) >= 5
                ):
                    recommendations.append(
                        {
                            "category": "category_specialization",
                            "priority": "medium",
                            "title": f"Improve {category.title()} Performance",
                            "description": f"Below-average performance in {category} category",
                            "specific_actions": [
                                f"Develop specialized knowledge base for {category}",
                                f"Analyze top performer strategies in {category}",
                                f"Allocate additional research time for {category} questions",
                            ],
                            "expected_impact": f"Potential to move from bottom 50% to top 30% in {category}",
                            "implementation_effort": "high",
                            "timeline": "2-4 weeks",
                        }
                    )

            # Timing optimization recommendations
            timing_impact = performance_attribution.get("timing_impact", {})
            early_vs_late = timing_impact.get("early_vs_late_performance", {})
            if early_vs_late:
                timing_advantage = early_vs_late.get("timing_advantage", 0)
                if timing_advantage > 0.05:  # Early predictions significantly better
                    recommendations.append(
                        {
                            "category": "timing_optimization",
                            "priority": "medium",
                            "title": "Prioritize Early Predictions",
                            "description": "Early predictions show significantly better performance",
                            "specific_actions": [
                                "Implement early question detection and prioritization",
                                "Allocate resources to answer questions within 48 hours of release",
                                "Develop rapid research and analysis capabilities",
                            ],
                            "expected_impact": f"{timing_advantage:.1%} improvement in accuracy",
                            "implementation_effort": "medium",
                            "timeline": "1 week",
                        }
                    )

            # Competitive positioning recommendations
            if current_standings:
                percentile = current_standings.our_percentile or 0.5
                if percentile < 0.3:  # Bottom 30%
                    recommendations.append(
                        {
                            "category": "competitive_strategy",
                            "priority": "high",
                            "title": "Aggressive Improvement Strategy",
                            "description": "Current ranking requires significant strategy changes",
                            "specific_actions": [
                                "Focus on high-impact, high-confidence questions",
                                "Take calculated risks on contrarian positions",
                                "Increase prediction volume to maximize opportunities",
                            ],
                            "expected_impact": "Potential to move up 20-40 percentile points",
                            "implementation_effort": "high",
                            "timeline": "immediate",
                        }
                    )
                elif percentile > 0.8:  # Top 20%
                    recommendations.append(
                        {
                            "category": "competitive_strategy",
                            "priority": "medium",
                            "title": "Maintain Leading Position",
                            "description": "Focus on consistency to maintain top ranking",
                            "specific_actions": [
                                "Avoid unnecessary risks",
                                "Focus on highest-confidence predictions",
                                "Monitor competitor strategies for defensive positioning",
                            ],
                            "expected_impact": "Maintain current ranking with reduced risk",
                            "implementation_effort": "low",
                            "timeline": "ongoing",
                        }
                    )

            # Performance gap recommendations
            performance_gaps = performance_attribution.get("performance_gaps", [])
            for gap in performance_gaps:
                if gap.get("severity") == "high":
                    recommendations.append(
                        {
                            "category": "performance_gap",
                            "priority": "high",
                            "title": f"Address {gap['type'].replace('_', ' ').title()}",
                            "description": gap["description"],
                            "specific_actions": gap.get("recommendations", []),
                            "expected_impact": f"Up to {gap.get('improvement_potential', 0):.0f} point improvement",
                            "implementation_effort": "high",
                            "timeline": "2-3 weeks",
                        }
                    )

            # Sort recommendations by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(
                key=lambda x: priority_order.get(x.get("priority", "low"), 2)
            )

            self.logger.info(
                f"Generated {len(recommendations)} optimization recommendations"
            )
            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {e}")
            return []

    def generate_competitive_intelligence_report(
        self,
        tournament_id: int,
        include_recommendations: bool = True,
        include_performance_attribution: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive competitive intelligence report.

        Args:
            tournament_id: Tournament to analyze
            include_recommendations: Whether to include strategic recommendations
            include_performance_attribution: Whether to include detailed performance attribution

        Returns:
            Comprehensive competitive intelligence report
        """
        try:
            standings = self.tournament_standings.get(tournament_id)
            if not standings:
                return {"error": f"No standings data for tournament {tournament_id}"}

            # Recent market inefficiencies
            recent_inefficiencies = [
                ineff
                for ineff in self.market_inefficiencies
                if (datetime.utcnow() - ineff.detected_at).days < 7
            ]

            # Recent strategic opportunities
            recent_opportunities = [
                opp
                for opp in self.strategic_opportunities
                if (datetime.utcnow() - opp.identified_at).days < 7
            ]

            report = {
                "tournament_id": tournament_id,
                "generated_at": datetime.utcnow().isoformat(),
                "competitive_position": {
                    "current_ranking": standings.our_ranking,
                    "percentile": standings.our_percentile,
                    "total_participants": standings.total_participants,
                    "score": standings.our_score,
                    "ranking_trend": self._calculate_ranking_trend(tournament_id),
                    "competitive_momentum": self._assess_competitive_momentum(
                        standings
                    ),
                },
                "competitive_landscape": {
                    "top_performers_count": len(standings.top_performers),
                    "score_distribution": standings.score_distribution,
                    "competitive_gaps": standings.competitive_gaps,
                    "market_concentration": self._calculate_market_concentration(
                        standings
                    ),
                    "competitive_threats": self._identify_competitive_threats(
                        standings
                    ),
                },
                "market_analysis": {
                    "inefficiencies_detected": len(recent_inefficiencies),
                    "inefficiency_types": list(
                        set(
                            ineff.inefficiency_type.value
                            for ineff in recent_inefficiencies
                        )
                    ),
                    "average_potential_advantage": (
                        statistics.mean(
                            [
                                ineff.potential_advantage
                                for ineff in recent_inefficiencies
                            ]
                        )
                        if recent_inefficiencies
                        else 0.0
                    ),
                    "exploitable_opportunities": len(
                        [
                            ineff
                            for ineff in recent_inefficiencies
                            if ineff.potential_advantage > 0.1
                        ]
                    ),
                    "market_efficiency_score": self._calculate_market_efficiency_score(
                        recent_inefficiencies
                    ),
                },
                "strategic_opportunities": {
                    "opportunities_count": len(recent_opportunities),
                    "opportunity_types": list(
                        set(opp.opportunity_type.value for opp in recent_opportunities)
                    ),
                    "total_potential_impact": sum(
                        opp.potential_impact for opp in recent_opportunities
                    ),
                    "high_impact_opportunities": len(
                        [
                            opp
                            for opp in recent_opportunities
                            if opp.potential_impact > 0.15
                        ]
                    ),
                    "time_sensitive_opportunities": len(
                        [
                            opp
                            for opp in recent_opportunities
                            if opp.time_sensitivity > 0.7
                        ]
                    ),
                },
                "improvement_opportunities": standings.improvement_opportunities,
            }

            if include_recommendations:
                report["strategic_recommendations"] = (
                    self._generate_strategic_recommendations(
                        standings, recent_inefficiencies, recent_opportunities
                    )
                )

            return report

        except Exception as e:
            self.logger.error(f"Error generating competitive intelligence report: {e}")
            return {"error": str(e)}

    def _calculate_correlation(
        self, x_values: List[float], y_values: List[float]
    ) -> float:
        """Calculate simple correlation coefficient between two lists."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)

        denominator = math.sqrt(
            (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
        )
        if denominator == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _calculate_calibration_score(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate calibration score for a set of predictions."""
        if not predictions:
            return 0.0

        # Simple calibration calculation
        confidence_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        calibration_error = 0.0
        total_weight = 0

        for i in range(len(confidence_bins) - 1):
            bin_predictions = [
                p
                for p in predictions
                if confidence_bins[i] <= p["confidence"] < confidence_bins[i + 1]
            ]

            if bin_predictions:
                avg_confidence = sum(p["confidence"] for p in bin_predictions) / len(
                    bin_predictions
                )
                accuracy = sum(1 for p in bin_predictions if p["correct"]) / len(
                    bin_predictions
                )
                weight = len(bin_predictions)

                calibration_error += weight * abs(avg_confidence - accuracy)
                total_weight += weight

        if total_weight == 0:
            return 0.0

        # Convert to calibration score (1 - normalized error)
        normalized_error = calibration_error / total_weight
        return max(0.0, 1.0 - normalized_error)

    def _calculate_ranking_trend(self, tournament_id: int) -> str:
        """Calculate ranking trend over time."""
        # This would typically use historical ranking data
        # For now, return a placeholder
        return "stable"

    def _assess_competitive_momentum(self, standings: TournamentStandings) -> str:
        """Assess competitive momentum based on recent performance."""
        if not standings.our_percentile:
            return "unknown"

        if standings.our_percentile > 0.8:
            return "strong_positive"
        elif standings.our_percentile > 0.6:
            return "positive"
        elif standings.our_percentile > 0.4:
            return "neutral"
        else:
            return "needs_improvement"

    def _calculate_market_concentration(
        self, standings: TournamentStandings
    ) -> Dict[str, float]:
        """Calculate market concentration metrics."""
        if not standings.score_distribution:
            return {}

        scores = [
            standings.score_distribution.get(key, 0)
            for key in ["max", "q75", "median", "q25", "min"]
        ]
        if not any(scores):
            return {}

        # Calculate concentration metrics
        score_range = standings.score_distribution.get(
            "max", 0
        ) - standings.score_distribution.get("min", 0)
        iqr = standings.score_distribution.get(
            "q75", 0
        ) - standings.score_distribution.get("q25", 0)

        return {
            "score_range": score_range,
            "interquartile_range": iqr,
            "concentration_ratio": iqr / score_range if score_range > 0 else 0,
            "market_tightness": (
                "high"
                if iqr / score_range < 0.3
                else "medium"
                if iqr / score_range < 0.6
                else "low"
            ),
        }

    def _identify_competitive_threats(
        self, standings: TournamentStandings
    ) -> List[Dict[str, Any]]:
        """Identify competitive threats from other participants."""
        threats = []

        if not standings.our_ranking or not standings.our_score:
            return threats

        # Identify close competitors
        for performer in standings.top_performers:
            if performer.current_ranking and performer.total_score:
                rank_diff = abs(performer.current_ranking - standings.our_ranking)
                score_diff = abs(performer.total_score - standings.our_score)

                # Close competitor (within 5 ranks and 10 points)
                if rank_diff <= 5 and score_diff <= 10:
                    threat_level = "high" if rank_diff <= 2 else "medium"

                    threats.append(
                        {
                            "competitor_id": performer.competitor_id,
                            "username": performer.username,
                            "threat_level": threat_level,
                            "rank_difference": performer.current_ranking
                            - standings.our_ranking,
                            "score_difference": performer.total_score
                            - standings.our_score,
                            "strengths": performer.strengths,
                            "competitive_advantages": [
                                strength
                                for strength in performer.strengths
                                if strength
                                in [
                                    "excellent_accuracy",
                                    "well_calibrated",
                                    "high_volume",
                                ]
                            ],
                        }
                    )

        return sorted(threats, key=lambda x: x["rank_difference"])

    def _calculate_market_efficiency_score(
        self, inefficiencies: List[MarketInefficiency]
    ) -> float:
        """Calculate overall market efficiency score."""
        if not inefficiencies:
            return 1.0  # Perfectly efficient market

        # Calculate weighted inefficiency score
        total_weight = 0
        weighted_inefficiency = 0

        for ineff in inefficiencies:
            weight = ineff.confidence_level * ineff.potential_advantage
            weighted_inefficiency += weight
            total_weight += ineff.confidence_level

        if total_weight == 0:
            return 1.0

        avg_inefficiency = weighted_inefficiency / total_weight
        return max(0.0, 1.0 - avg_inefficiency)

    def _generate_strategic_recommendations(
        self,
        standings: TournamentStandings,
        inefficiencies: List[MarketInefficiency],
        opportunities: List[StrategicOpportunity],
    ) -> List[Dict[str, Any]]:
        """Generate strategic recommendations based on analysis."""
        recommendations = []

        # Position-based recommendations
        if standings.our_percentile and standings.our_percentile < 0.3:
            recommendations.append(
                {
                    "category": "competitive_positioning",
                    "priority": "high",
                    "title": "Aggressive Improvement Strategy",
                    "description": "Low ranking requires significant strategy changes",
                    "rationale": f"Currently in bottom {(1 - standings.our_percentile) * 100:.0f}% of participants",
                    "specific_actions": [
                        "Focus on high-impact, high-confidence questions",
                        "Take calculated risks on contrarian positions",
                        "Increase prediction volume to maximize opportunities",
                    ],
                    "expected_impact": "20-40 percentile point improvement",
                    "risk_level": "medium-high",
                }
            )
        elif standings.our_percentile and standings.our_percentile > 0.8:
            recommendations.append(
                {
                    "category": "competitive_positioning",
                    "priority": "medium",
                    "title": "Defensive Excellence Strategy",
                    "description": "Maintain top position through consistent performance",
                    "rationale": f"Currently in top {standings.our_percentile * 100:.0f}% of participants",
                    "specific_actions": [
                        "Focus on maintaining prediction quality",
                        "Avoid unnecessary risks",
                        "Monitor competitor strategies for defensive positioning",
                    ],
                    "expected_impact": "Maintain current ranking with reduced risk",
                    "risk_level": "low",
                }
            )

        # Inefficiency-based recommendations
        if inefficiencies:
            common_inefficiencies = {}
            total_advantage = 0
            for ineff in inefficiencies:
                ineff_type = ineff.inefficiency_type.value
                common_inefficiencies[ineff_type] = (
                    common_inefficiencies.get(ineff_type, 0) + 1
                )
                total_advantage += ineff.potential_advantage

            if common_inefficiencies:
                most_common = max(common_inefficiencies.items(), key=lambda x: x[1])
                recommendations.append(
                    {
                        "category": "market_exploitation",
                        "priority": "high",
                        "title": f"Exploit {most_common[0].replace('_', ' ').title()}",
                        "description": f"Detected {most_common[1]} instances of {most_common[0]} recently",
                        "rationale": f"Total potential advantage: {total_advantage:.2f} points",
                        "specific_actions": [
                            f"Monitor for {most_common[0]} patterns in new questions",
                            "Develop systematic approach to exploit this bias",
                            "Track exploitation success rate",
                        ],
                        "expected_impact": f"Up to {total_advantage:.1f} point improvement",
                        "risk_level": "medium",
                    }
                )

        # Opportunity-based recommendations
        if opportunities:
            high_impact_opportunities = [
                opp for opp in opportunities if opp.potential_impact > 0.15
            ]
            time_sensitive_opportunities = [
                opp for opp in opportunities if opp.time_sensitivity > 0.7
            ]

            if high_impact_opportunities:
                recommendations.append(
                    {
                        "category": "opportunity_capture",
                        "priority": "high",
                        "title": "Prioritize High-Impact Opportunities",
                        "description": f"Identified {len(high_impact_opportunities)} high-impact opportunities",
                        "rationale": f"Total potential impact: {sum(opp.potential_impact for opp in high_impact_opportunities):.2f}",
                        "specific_actions": [
                            "Allocate additional resources to high-impact questions",
                            "Fast-track analysis for these opportunities",
                            "Monitor opportunity windows closely",
                        ],
                        "expected_impact": f"Up to {sum(opp.potential_impact for opp in high_impact_opportunities):.1f} point improvement",
                        "risk_level": "medium",
                    }
                )

            if time_sensitive_opportunities:
                recommendations.append(
                    {
                        "category": "timing_optimization",
                        "priority": "urgent",
                        "title": "Act on Time-Sensitive Opportunities",
                        "description": f"Identified {len(time_sensitive_opportunities)} time-sensitive opportunities",
                        "rationale": "These opportunities may expire soon",
                        "specific_actions": [
                            "Immediately prioritize time-sensitive questions",
                            "Expedite research and analysis processes",
                            "Submit predictions before opportunity windows close",
                        ],
                        "expected_impact": "Prevent missed opportunities",
                        "risk_level": "low",
                    }
                )

        # Gap-based recommendations
        next_rank_gap = standings.competitive_gaps.get("next_rank", float("inf"))
        if next_rank_gap < 5:
            recommendations.append(
                {
                    "category": "competitive_advancement",
                    "priority": "medium",
                    "title": "Close Gap to Next Rank",
                    "description": f"Only {next_rank_gap:.1f} points behind next rank",
                    "rationale": "Small gap presents immediate advancement opportunity",
                    "specific_actions": [
                        "Focus on consistency over innovation",
                        "Avoid risky predictions that could backfire",
                        "Target questions where you have highest confidence",
                    ],
                    "expected_impact": "Move up one ranking position",
                    "risk_level": "low",
                }
            )

        # Sort recommendations by priority
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(
            key=lambda x: priority_order.get(x.get("priority", "low"), 3)
        )

        return recommendations or [
            {
                "category": "status_quo",
                "priority": "low",
                "title": "Continue Current Approach",
                "description": "No specific optimization opportunities identified",
                "rationale": "Current performance appears optimal given available data",
                "specific_actions": [
                    "Maintain current strategy",
                    "Monitor for new opportunities",
                ],
                "expected_impact": "Stable performance",
                "risk_level": "low",
            }
        ]
