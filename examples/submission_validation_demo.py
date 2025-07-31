#!/usr/bin/env python3
"""
Demo script showcasing the enhanced submission validation and audit trail system.

This script demonstrates the key features implemented for task 10.2:
1. Comprehensive prediction validation and formatting
2. Submission confirmation and audit trail maintenance
3. Dry-run mode with tournament condition simulation
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.infrastructure.external_apis.submission_validator import (
    SubmissionValidator,
    AuditTrailManager,
    DryRunManager,
    ValidationResult,
    SubmissionStatus
)
from src.domain.entities.question import Question, QuestionType
from src.domain.entities.prediction import Prediction, PredictionResult
from src.domain.value_objects.confidence import ConfidenceLevel


def create_sample_questions():
    """Create sample questions for demonstration."""
    now = datetime.now(timezone.utc)

    questions = [
        Question.create(
            title="Will AI achieve AGI by 2030?",
            description="This question asks about artificial general intelligence development.",
            question_type=QuestionType.BINARY,
            resolution_criteria="AGI is defined as AI that can perform any intellectual task that a human can.",
            close_time=now + timedelta(hours=12),
            resolve_time=now + timedelta(days=365),
            created_at=now - timedelta(days=30),
            metadata={
                "category": "technology",
                "tournament_priority": "high",
                "community_prediction": 0.35,
                "prediction_count": 150,
                "urgency_score": 0.8
            }
        ),
        Question.create(
            title="What will be the global temperature anomaly in 2030?",
            description="Temperature anomaly relative to 1951-1980 baseline.",
            question_type=QuestionType.CONTINUOUS,
            resolution_criteria="Based on NASA GISS data.",
            close_time=now + timedelta(days=7),
            resolve_time=now + timedelta(days=365),
            created_at=now - timedelta(days=60),
            min_value=-2.0,
            max_value=5.0,
            metadata={
                "category": "science",
                "tournament_priority": "medium",
                "community_prediction": 1.2,
                "prediction_count": 75
            }
        )
    ]

    return questions


def create_sample_predictions():
    """Create sample predictions for demonstration."""
    predictions = [
        {
            "question_id": "agi_2030",
            "prediction_value": 0.65,
            "reasoning": "Based on recent advances in large language models, multimodal AI, and increasing investment in AI research, there's a significant probability of achieving AGI by 2030. However, significant technical challenges remain in areas like reasoning, planning, and general problem-solving.",
            "confidence": 0.75,
            "agent_type": "ensemble",
            "reasoning_method": "chain_of_thought"
        },
        {
            "question_id": "temperature_2030",
            "prediction_value": 1.4,
            "reasoning": "Climate models consistently show continued warming trends. Current trajectory suggests approximately 1.4°C anomaly by 2030, considering current emission patterns and climate policies.",
            "confidence": 0.8,
            "agent_type": "research_focused",
            "reasoning_method": "data_analysis"
        }
    ]

    return predictions


def demonstrate_validation_features():
    """Demonstrate comprehensive validation features."""
    print("=== Submission Validation Demo ===\n")

    # Create validator in tournament mode
    validator = SubmissionValidator(tournament_mode=True)
    questions = create_sample_questions()
    predictions = create_sample_predictions()

    print("1. Basic Validation")
    print("-" * 50)

    for i, (question, prediction) in enumerate(zip(questions, predictions)):
        print(f"\nValidating prediction {i+1}: {question.title[:50]}...")

        result, errors = validator.validate_prediction(question, prediction)
        print(f"Validation result: {result.value}")

        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error.field}: {error.message} ({error.code})")
        else:
            print("No validation errors found.")

        # Demonstrate formatting
        formatted = validator.format_prediction_for_submission(question, prediction)
        print(f"Formatted prediction keys: {list(formatted.keys())}")

        if validator.tournament_mode and "tournament_metadata" in formatted:
            tm = formatted["tournament_metadata"]
            print(f"Tournament metadata: category={tm['question_category']}, priority={tm['tournament_priority']}")

    print("\n2. Tournament Condition Simulation")
    print("-" * 50)

    tournament_context = {
        "tournament_id": "demo_tournament",
        "current_ranking": 25,
        "participant_count": 100,
        "completion_rate": 0.6
    }

    question = questions[0]
    prediction = predictions[0]

    print(f"\nSimulating tournament conditions for: {question.title[:50]}...")
    simulation_results = validator.simulate_tournament_conditions(
        question, prediction, tournament_context
    )

    print(f"Question analysis:")
    qa = simulation_results["question_analysis"]
    print(f"  - Category: {qa['category']}")
    print(f"  - Priority: {qa['tournament_priority']}")
    print(f"  - Is urgent: {qa['is_urgent']}")
    print(f"  - Community prediction: {qa['community_prediction']}")

    print(f"\nTournament simulation:")
    ts = simulation_results["tournament_simulation"]
    print(f"  - Market efficiency: {ts['market_efficiency']}")
    print(f"  - Competitive pressure: {ts['competitive_pressure']}")
    print(f"  - Scoring impact: {ts['scoring_impact']['impact']}")

    print(f"\nRisk assessment:")
    ra = simulation_results["risk_assessment"]
    print(f"  - Risk level: {ra['risk_level']}")
    print(f"  - Identified risks: {', '.join(ra['identified_risks'])}")

    print(f"\nRecommendations: {len(simulation_results['recommendations'])}")
    for rec in simulation_results["recommendations"]:
        print(f"  - {rec['type']}: {rec['message']}")


def demonstrate_audit_trail():
    """Demonstrate audit trail management."""
    print("\n\n=== Audit Trail Demo ===\n")

    audit_manager = AuditTrailManager(storage_path="demo_audit.jsonl")
    questions = create_sample_questions()
    predictions = create_sample_predictions()

    print("1. Creating Submission Records")
    print("-" * 50)

    records = []
    for i, (question, prediction) in enumerate(zip(questions, predictions)):
        record = audit_manager.create_submission_record(
            question_id=question.id,
            prediction_value=prediction["prediction_value"],
            reasoning=prediction["reasoning"],
            confidence=prediction["confidence"],
            dry_run=i == 1,  # Make second one a dry run
            metadata={
                "agent_type": prediction["agent_type"],
                "reasoning_method": prediction["reasoning_method"]
            }
        )
        records.append(record)
        print(f"Created record {record.submission_id} for {question.title[:30]}...")

    print("\n2. Submission Confirmation")
    print("-" * 50)

    # Simulate successful submission
    api_response_success = {
        "status_code": 200,
        "message": "Prediction submitted successfully",
        "prediction_id": "metaculus_12345"
    }

    audit_manager.confirm_submission(
        records[0].submission_id,
        api_response_success,
        success=True
    )
    print(f"Confirmed successful submission for {records[0].submission_id}")

    # Simulate failed submission
    api_response_failed = {
        "status_code": 400,
        "message": "Validation error: Invalid prediction format"
    }

    # Create another record for failure demo
    failed_record = audit_manager.create_submission_record(
        question_id="demo_question",
        prediction_value=1.5,  # Invalid for binary
        reasoning="This will fail validation",
        confidence=0.5
    )

    audit_manager.confirm_submission(
        failed_record.submission_id,
        api_response_failed,
        success=False
    )
    print(f"Confirmed failed submission for {failed_record.submission_id}")

    print("\n3. Performance Metrics")
    print("-" * 50)

    metrics = audit_manager.get_performance_metrics()
    print(f"Total submissions: {metrics['total_submissions']}")
    print(f"Real submissions: {metrics['real_submissions']}")
    print(f"Success rate: {metrics['success_rate']:.1%}")

    print("\n4. Audit Report")
    print("-" * 50)

    report = audit_manager.generate_audit_report()
    print(f"Status distribution: {report['status_distribution']}")
    print(f"Dry run submissions: {report['dry_run_submissions']}")

    # Clean up demo file
    import os
    if os.path.exists("demo_audit.jsonl"):
        os.remove("demo_audit.jsonl")


def demonstrate_dry_run_mode():
    """Demonstrate dry-run mode with tournament simulation."""
    print("\n\n=== Dry-Run Mode Demo ===\n")

    validator = SubmissionValidator(tournament_mode=True)
    audit_manager = AuditTrailManager()
    dry_run_manager = DryRunManager(validator, audit_manager)

    questions = create_sample_questions()
    predictions = create_sample_predictions()

    print("1. Starting Dry-Run Session")
    print("-" * 50)

    tournament_context = {
        "tournament_id": "demo_tournament",
        "current_ranking": 35,
        "participant_count": 150,
        "completion_rate": 0.45
    }

    session_id = dry_run_manager.start_dry_run_session(
        "Demo Tournament Session",
        tournament_context
    )
    print(f"Started dry-run session: {session_id}")

    print("\n2. Simulating Submissions")
    print("-" * 50)

    for i, (question, prediction) in enumerate(zip(questions, predictions)):
        print(f"\nSimulating submission {i+1}: {question.title[:40]}...")

        agent_metadata = {
            "agent_type": prediction["agent_type"],
            "reasoning_method": prediction["reasoning_method"]
        }

        simulation_results = dry_run_manager.simulate_submission(
            session_id,
            question,
            prediction,
            agent_metadata
        )

        print(f"  Validation result: {simulation_results['validation_results']['result']}")
        print(f"  Would succeed: {simulation_results['api_simulation']['would_succeed']}")
        print(f"  Risk level: {simulation_results['risk_assessment']['risk_level']}")
        print(f"  Learning opportunities: {len(simulation_results['learning_opportunities'])}")

        # Show competitive analysis if available
        comp_analysis = simulation_results["competitive_analysis"]
        if "potential_change" in comp_analysis:
            print(f"  Potential ranking change: {comp_analysis['potential_change']}")

    print("\n3. Session Report")
    print("-" * 50)

    report = dry_run_manager.end_dry_run_session(session_id)

    session_summary = report["session_summary"]
    print(f"Session completed:")
    print(f"  - Total submissions: {session_summary['total_submissions']}")
    print(f"  - Validation success rate: {session_summary['validation_success_rate']:.1%}")
    print(f"  - Duration: {session_summary['duration_seconds']:.1f} seconds")

    risk_analysis = report["risk_analysis"]
    print(f"\nRisk analysis:")
    print(f"  - High risk submissions: {risk_analysis['high_risk_submissions']}")
    print(f"  - Risk rate: {risk_analysis['risk_rate']:.1%}")

    learning_analysis = report["learning_analysis"]
    print(f"\nLearning analysis:")
    print(f"  - Total opportunities: {learning_analysis['total_opportunities']}")
    print(f"  - Top learning areas: {', '.join(learning_analysis['top_learning_areas'])}")

    competitive_analysis = report["competitive_analysis"]
    print(f"\nCompetitive analysis:")
    print(f"  - Average ranking change: {competitive_analysis['average_ranking_change']:.1f}")
    print(f"  - Competitive readiness: {competitive_analysis['competitive_readiness']}")

    print(f"\nRecommendations: {len(report['recommendations'])}")
    for rec in report["recommendations"]:
        print(f"  - {rec['type']}: {rec['message']}")


def main():
    """Run the complete demonstration."""
    print("Enhanced Submission Validation and Audit Trail System Demo")
    print("=" * 60)
    print("This demo showcases the implementation of task 10.2:")
    print("- Comprehensive prediction validation and formatting")
    print("- Submission confirmation and audit trail maintenance")
    print("- Dry-run mode with tournament condition simulation")
    print("=" * 60)

    try:
        demonstrate_validation_features()
        demonstrate_audit_trail()
        demonstrate_dry_run_mode()

        print("\n\n=== Demo Complete ===")
        print("All features demonstrated successfully!")
        print("\nKey capabilities implemented:")
        print("✓ Tournament-specific validation and formatting")
        print("✓ Comprehensive audit trail with confirmation tracking")
        print("✓ Advanced dry-run mode with tournament simulation")
        print("✓ Risk assessment and learning opportunity identification")
        print("✓ Performance metrics and competitive analysis")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
