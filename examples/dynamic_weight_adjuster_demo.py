#!/usr/bin/env python3
"""
Demonstration of enhanced DynamicWeightAdjuster functionality.

This script shows how the DynamicWeightAdjuster can:
1. Track historical performance and adjust weights
2. Perform real-time agent selection
3. Detect performance degradation
4. Trigger automatic rebalancing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from uuid import uuid4

from src.domain.services.dynamic_weight_adjuster import (
    DynamicWeightAdjuster,
    PerformanceRecord,
    WeightAdjustmentStrategy
)
from src.domain.entities.prediction import PredictionMethod


def create_performance_record(agent_name: str,
                            brier_score: float,
                            accuracy: float,
                            confidence: float,
                            days_ago: int = 0) -> PerformanceRecord:
    """Create a performance record for testing."""
    return PerformanceRecord(
        agent_name=agent_name,
        prediction_id=uuid4(),
        question_id=uuid4(),
        timestamp=datetime.now() - timedelta(days=days_ago),
        predicted_probability=0.7,
        actual_outcome=True,
        brier_score=brier_score,
        accuracy=accuracy,
        confidence_score=confidence,
        method=PredictionMethod.CHAIN_OF_THOUGHT
    )


def main():
    print("=== Dynamic Weight Adjuster Enhanced Functionality Demo ===\n")

    # Initialize the adjuster
    adjuster = DynamicWeightAdjuster(
        lookback_window=20,
        min_predictions_for_weight=3,
        performance_decay_factor=0.95
    )

    # Simulate different agent performances
    agents = {
        "excellent_agent": (0.10, 0.95, 0.9),    # Low Brier, high accuracy, high confidence
        "good_agent": (0.18, 0.85, 0.8),         # Good performance
        "average_agent": (0.25, 0.70, 0.7),      # Average performance
        "poor_agent": (0.40, 0.45, 0.5),         # Poor performance
        "degrading_agent": None                    # Will simulate degradation
    }

    print("1. Simulating historical performance data...")

    # Add historical performance for stable agents
    for agent, performance_data in agents.items():
        if agent == "degrading_agent" or performance_data is None:
            continue

        brier, accuracy, confidence = performance_data

        print(f"   Adding performance data for {agent}")
        for i in range(15):
            record = create_performance_record(
                agent, brier, accuracy, confidence, days_ago=i
            )
            adjuster.performance_records.append(record)

        adjuster._update_agent_profile(agent)

    # Simulate degrading agent (good performance initially, then poor)
    print("   Adding performance data for degrading_agent (showing degradation)")

    # Good early performance
    for i in range(10, 15):
        record = create_performance_record(
            "degrading_agent", 0.12, 0.9, 0.85, days_ago=i
        )
        adjuster.performance_records.append(record)

    # Poor recent performance
    for i in range(5):
        record = create_performance_record(
            "degrading_agent", 0.45, 0.3, 0.4, days_ago=i
        )
        adjuster.performance_records.append(record)

    adjuster._update_agent_profile("degrading_agent")

    print("\n2. Performance Summary:")
    summary = adjuster.get_performance_summary()
    for agent, profile in summary["agent_profiles"].items():
        print(f"   {agent}:")
        print(f"     Recent Brier Score: {profile['recent_brier_score']:.3f}")
        print(f"     Recent Accuracy: {profile['recent_accuracy']:.3f}")
        print(f"     Performance Trend: {profile['performance_trend']:+.3f}")
        print(f"     Recommended Weight: {profile['recommended_weight']:.3f}")

    print("\n3. Performance Degradation Detection:")
    all_agents = list(agents.keys())
    for agent in all_agents:
        is_degrading, explanation = adjuster.detect_performance_degradation(agent)
        status = "⚠️  DEGRADING" if is_degrading else "✅ STABLE"
        print(f"   {agent}: {status}")
        if is_degrading:
            print(f"     Reason: {explanation}")

    print("\n4. Dynamic Weight Calculation (Different Strategies):")
    strategies = [
        WeightAdjustmentStrategy.ADAPTIVE_LEARNING_RATE,
        WeightAdjustmentStrategy.EXPONENTIAL_DECAY,
        WeightAdjustmentStrategy.THRESHOLD_BASED
    ]

    for strategy in strategies:
        weights = adjuster.get_dynamic_weights(all_agents, strategy)
        print(f"   {strategy.value}:")
        for agent, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"     {agent}: {weight:.3f}")

    print("\n5. Real-time Agent Selection:")
    selected_agents = adjuster.select_optimal_agents_realtime(
        all_agents, max_agents=3
    )
    print(f"   Selected agents: {selected_agents}")

    print("\n6. Rebalancing Trigger Detection:")
    current_ensemble = ["average_agent", "degrading_agent"]
    should_rebalance, reason = adjuster.should_trigger_rebalancing(current_ensemble)
    print(f"   Should rebalance: {'YES' if should_rebalance else 'NO'}")
    print(f"   Reason: {reason}")

    print("\n7. Automatic Rebalancing:")
    if should_rebalance:
        new_composition = adjuster.trigger_automatic_rebalancing(
            current_ensemble, all_agents
        )
        if new_composition:
            print("   New ensemble composition:")
            for agent, weight in new_composition.agent_weights.items():
                print(f"     {agent}: {weight:.3f}")
            print(f"   Diversity Score: {new_composition.diversity_score:.3f}")
            print(f"   Expected Performance: {new_composition.expected_performance:.3f}")

    print("\n8. Rebalancing Recommendations:")
    recommendations = adjuster.get_rebalancing_recommendations(current_ensemble)
    print(f"   Should rebalance: {recommendations['should_rebalance']}")
    print(f"   Degraded agents: {len(recommendations['degraded_agents'])}")
    print(f"   Recommended additions: {len(recommendations['recommended_additions'])}")
    print(f"   Recommended removals: {len(recommendations['recommended_removals'])}")

    if recommendations['recommended_additions']:
        print("   Top additions:")
        for addition in recommendations['recommended_additions'][:2]:
            print(f"     {addition['agent']} (Brier: {addition['recent_brier_score']:.3f})")

    print("\n9. Ensemble Composition Recommendation:")
    optimal_composition = adjuster.recommend_ensemble_composition(
        all_agents, target_size=3, diversity_weight=0.3
    )
    print("   Optimal ensemble:")
    for agent, weight in optimal_composition.agent_weights.items():
        print(f"     {agent}: {weight:.3f}")
    print("   Composition rationale:")
    print(f"     {optimal_composition.composition_rationale}")

    print("\n=== Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("✅ Historical performance tracking and weight adjustment")
    print("✅ Real-time agent selection and ensemble composition optimization")
    print("✅ Performance degradation detection and automatic rebalancing")
    print("✅ Multiple weight adjustment strategies")
    print("✅ Comprehensive rebalancing recommendations")


if __name__ == "__main__":
    main()
