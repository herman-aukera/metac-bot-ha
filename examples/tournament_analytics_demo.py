#!/usr/bin/env python3
"""
Demo script showing enhanced TournamentAnalytics capabilities for competitive intelligence.

This script demonstrates the new features added in task 8.2:
- Tournament standings analysis and competitive positioning
- Market inefficiency detection and strategic opportunity identification
- Performance attribution analysis and optimization recommendations
"""

from datetime import datetime, timedelta
from uuid import uuid4

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.services.tournament_analytics import TournamentAnalytics


def main():
    """Demonstrate enhanced tournament analytics capabilities."""
    print("🏆 Tournament Analytics Demo - Enhanced Competitive Intelligence")
    print("=" * 70)

    # Initialize tournament analytics service
    analytics = TournamentAnalytics()

    # Sample tournament standings data
    standings_data = {
        "participants": [
            {
                "user_id": "top_performer",
                "username": "AlphaForecaster",
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
                    "technology": {"brier_score": 0.13}
                }
            },
            {
                "user_id": "second_place",
                "username": "BetaPredictor",
                "score": 89.2,
                "questions_answered": 48,
                "questions_resolved": 42,
                "average_brier_score": 0.15,
                "calibration_score": 0.82,
                "category_performance": {
                    "politics": {"brier_score": 0.16},
                    "economics": {"brier_score": 0.14}
                }
            },
            {
                "user_id": "our_bot",
                "username": "MetaculusBot",
                "score": 75.8,
                "questions_answered": 35,
                "questions_resolved": 30,
                "average_brier_score": 0.18,
                "calibration_score": 0.78,
                "category_performance": {
                    "politics": {"brier_score": 0.20},
                    "economics": {"brier_score": 0.16},
                    "technology": {"brier_score": 0.18}
                }
            },
            {
                "user_id": "fourth_place",
                "username": "GammaGuesser",
                "score": 70.1,
                "questions_answered": 40,
                "questions_resolved": 35,
                "average_brier_score": 0.22,
                "calibration_score": 0.72
            }
        ]
    }

    # 1. Analyze tournament standings
    print("\n1. 📊 Tournament Standings Analysis")
    print("-" * 40)

    tournament_id = 12345
    our_user_id = "our_bot"

    standings = analytics.analyze_tournament_standings(
        tournament_id=tournament_id,
        standings_data=standings_data,
        our_user_id=our_user_id
    )

    print(f"Our Position: Rank {standings.our_ranking}/{standings.total_participants}")
    print(f"Percentile: {standings.our_percentile:.1%}")
    print(f"Score: {standings.our_score}")
    print(f"Gap to next rank: {standings.competitive_gaps.get('next_rank', 0):.1f} points")
    print(f"Gap to leader: {standings.competitive_gaps.get('leader', 0):.1f} points")

    # 2. Detect market inefficiencies
    print("\n2. 🎯 Market Inefficiency Detection")
    print("-" * 40)

    # Sample question with community predictions showing overconfidence bias
    question_data = {"id": str(uuid4()), "title": "Will AI achieve AGI by 2030?"}

    # Create predictions showing overconfidence (many extreme values)
    base_time = datetime.utcnow()
    community_predictions = []
    for i in range(15):
        # 60% extreme predictions (overconfidence bias)
        if i < 9:
            prediction = 0.05 if i % 2 == 0 else 0.95
        else:
            prediction = 0.3 + (i - 9) * 0.1

        community_predictions.append({
            "question_id": str(uuid4()),
            "prediction": prediction,
            "timestamp": (base_time - timedelta(hours=i)).isoformat() + "Z",
            "user_id": f"user_{i}"
        })

    inefficiencies = analytics.detect_market_inefficiencies(
        question_data=question_data,
        community_predictions=community_predictions
    )

    print(f"Detected {len(inefficiencies)} market inefficiencies:")
    for ineff in inefficiencies:
        print(f"  • {ineff.inefficiency_type.value}: {ineff.description}")
        print(f"    Potential advantage: {ineff.potential_advantage:.2f}")
        print(f"    Confidence: {ineff.confidence_level:.2f}")
        print(f"    Strategy: {ineff.exploitation_strategy}")

    # 3. Identify strategic opportunities
    print("\n3. 🚀 Strategic Opportunity Identification")
    print("-" * 40)

    tournament_context = {
        "tournament_id": tournament_id,
        "our_ranking": standings.our_ranking,
        "total_participants": standings.total_participants
    }

    our_performance = {
        "category_performance": {
            "politics": {"brier_score": 0.20, "question_count": 10},
            "economics": {"brier_score": 0.16, "question_count": 8},
            "technology": {"brier_score": 0.15, "question_count": 12}  # Strong performance
        }
    }

    # Sample question pipeline
    question_pipeline = [
        {
            "id": str(uuid4()),
            "title": "Early opportunity question",
            "deadline": (base_time + timedelta(hours=72)).isoformat() + "Z",
            "prediction_count": 5,
            "category": "technology",  # Our strong category
            "scoring_potential": 0.8,
            "difficulty": 0.4
        },
        {
            "id": str(uuid4()),
            "title": "High-value economics question",
            "deadline": (base_time + timedelta(hours=48)).isoformat() + "Z",
            "prediction_count": 15,
            "category": "economics",
            "scoring_potential": 0.9,
            "difficulty": 0.6
        }
    ]

    opportunities = analytics.identify_strategic_opportunities(
        tournament_context=tournament_context,
        our_performance=our_performance,
        question_pipeline=question_pipeline
    )

    print(f"Identified {len(opportunities)} strategic opportunities:")
    for opp in opportunities:
        print(f"  • {opp.opportunity_type.value}: {opp.title}")
        print(f"    Impact: {opp.potential_impact:.2f}, Confidence: {opp.confidence:.2f}")
        print(f"    Time sensitivity: {opp.time_sensitivity:.2f}")
        print(f"    Actions: {', '.join(opp.recommended_actions[:2])}")

    # 4. Performance attribution analysis
    print("\n4. 📈 Performance Attribution Analysis")
    print("-" * 40)

    our_performance_data = {
        "average_brier_score": 0.18,
        "calibration_score": 0.78,
        "questions_answered": 35,
        "category_performance": {
            "politics": {"brier_score": 0.20, "question_count": 10},
            "economics": {"brier_score": 0.16, "question_count": 8},
            "technology": {"brier_score": 0.15, "question_count": 12}
        }
    }

    competitor_performance_data = [
        {"average_brier_score": 0.12, "calibration_score": 0.88, "questions_answered": 50},
        {"average_brier_score": 0.15, "calibration_score": 0.82, "questions_answered": 48},
        {"average_brier_score": 0.22, "calibration_score": 0.72, "questions_answered": 40}
    ]

    question_history = [
        {
            "research_depth": 0.8, "brier_score": 0.15, "confidence": 0.7,
            "was_correct": True, "category": "technology", "hours_before_deadline": 48
        },
        {
            "research_depth": 0.6, "brier_score": 0.25, "confidence": 0.6,
            "was_correct": False, "category": "politics", "hours_before_deadline": 12
        },
        {
            "research_depth": 0.9, "brier_score": 0.10, "confidence": 0.8,
            "was_correct": True, "category": "technology", "hours_before_deadline": 72
        }
    ]

    attribution = analytics.analyze_performance_attribution(
        our_performance_data=our_performance_data,
        competitor_performance_data=competitor_performance_data,
        question_history=question_history
    )

    print("Performance Attribution Results:")

    # Research depth correlation
    research_corr = attribution["accuracy_drivers"]["research_depth_correlation"]
    print(f"  • Research depth correlation: {research_corr:.3f}")

    # Competitive advantages
    advantages = attribution["competitive_advantages"]
    print(f"  • Competitive advantages: {len(advantages)}")
    for adv in advantages:
        print(f"    - {adv['type']}: {adv['description']}")

    # Performance gaps
    gaps = attribution["performance_gaps"]
    print(f"  • Performance gaps: {len(gaps)}")
    for gap in gaps:
        print(f"    - {gap['type']}: {gap['description']} (severity: {gap['severity']})")

    # 5. Generate optimization recommendations
    print("\n5. 🎯 Optimization Recommendations")
    print("-" * 40)

    recommendations = analytics.generate_optimization_recommendations(
        tournament_id=tournament_id,
        performance_attribution=attribution,
        current_standings=standings
    )

    print(f"Generated {len(recommendations)} optimization recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
        print(f"\n  {i}. {rec['title']} (Priority: {rec['priority']})")
        print(f"     Category: {rec['category']}")
        print(f"     Description: {rec['description']}")
        print(f"     Expected Impact: {rec['expected_impact']}")
        print("     Actions:")
        for action in rec['specific_actions'][:2]:  # Show first 2 actions
            print(f"       - {action}")

    # 6. Generate comprehensive competitive intelligence report
    print("\n6. 📋 Comprehensive Competitive Intelligence Report")
    print("-" * 40)

    report = analytics.generate_competitive_intelligence_report(
        tournament_id=tournament_id,
        include_recommendations=True,
        include_performance_attribution=True
    )

    print("Report Summary:")
    print(f"  • Tournament ID: {report['tournament_id']}")
    print(f"  • Current Ranking: {report['competitive_position']['current_ranking']}")
    print(f"  • Percentile: {report['competitive_position']['percentile']:.1%}")
    print(f"  • Competitive Momentum: {report['competitive_position']['competitive_momentum']}")
    print(f"  • Market Inefficiencies: {report['market_analysis']['inefficiencies_detected']}")
    print(f"  • Strategic Opportunities: {report['strategic_opportunities']['opportunities_count']}")
    print(f"  • Market Efficiency Score: {report['market_analysis']['market_efficiency_score']:.3f}")

    # Strategic recommendations summary
    strategic_recs = report['strategic_recommendations']
    print(f"\n  Strategic Recommendations ({len(strategic_recs)}):")
    for rec in strategic_recs[:2]:  # Show top 2
        print(f"    • {rec['title']} ({rec['priority']} priority)")
        print(f"      {rec['description']}")
        print(f"      Expected Impact: {rec['expected_impact']}")

    print("\n" + "=" * 70)
    print("✅ Tournament Analytics Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Tournament standings analysis and competitive positioning")
    print("  • Market inefficiency detection (overconfidence, herding, etc.)")
    print("  • Strategic opportunity identification (timing, expertise, etc.)")
    print("  • Performance attribution analysis")
    print("  • Optimization recommendations generation")
    print("  • Comprehensive competitive intelligence reporting")


if __name__ == "__main__":
    main()
