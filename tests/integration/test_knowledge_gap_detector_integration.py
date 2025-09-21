"""
Integration test for KnowledgeGapDetector to verify the implementation works correctly.
"""

from datetime import datetime, timedelta
from uuid import uuid4

from src.domain.entities.question import Question, QuestionStatus, QuestionType
from src.domain.entities.research_report import ResearchQuality
from src.domain.services.authoritative_source_manager import (
    AuthoritativeSource,
    ExpertiseArea,
    ExpertProfile,
    KnowledgeBase,
    SourceType,
)
from src.domain.services.knowledge_gap_detector import (
    GapSeverity,
    KnowledgeGapDetector,
    ResearchStrategy,
)


def test_knowledge_gap_detector_integration():
    """Integration test for KnowledgeGapDetector functionality."""

    # Create detector
    detector = KnowledgeGapDetector()

    # Create test question
    question = Question(
        id=uuid4(),
        metaculus_id=12345,
        title="Will AI achieve AGI by 2030?",
        description="Question about artificial general intelligence timeline",
        question_type=QuestionType.BINARY,
        status=QuestionStatus.OPEN,
        url="https://metaculus.com/questions/12345",
        close_time=datetime.utcnow() + timedelta(days=365),
        resolve_time=None,
        categories=["AI", "Technology"],
        metadata={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        resolution_criteria="Clear definition of AGI achievement",
    )

    # Test with limited sources (should detect many gaps)
    limited_sources = [
        AuthoritativeSource(
            url="https://old-blog.com/ai-predictions",
            title="Old AI Predictions",
            summary="Outdated predictions about AI development",
            source_type=SourceType.NEWS_ANALYSIS,
            credibility_score=0.4,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=800),
            authors=["Blogger"],
        )
    ]

    # Test gap detection
    gaps = detector.detect_knowledge_gaps(limited_sources, question)
    assert len(gaps) >= 5, f"Expected at least 5 gaps, got {len(gaps)}"

    # Should have critical gaps
    critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
    assert len(critical_gaps) >= 1, "Should have at least one critical gap"

    # Test research quality assessment
    assessment = detector.assess_research_quality(limited_sources, question)
    assert assessment.overall_quality == ResearchQuality.LOW
    assert assessment.confidence_level < 0.6
    assert assessment.completeness_score < 0.6
    assert len(assessment.critical_gaps) >= 1

    # Test adaptive research plan creation
    constraints = {"total_time": timedelta(hours=6), "max_gaps": 3}

    plan = detector.create_adaptive_research_plan(assessment, question, constraints)
    assert plan.plan_id
    assert plan.strategy in ResearchStrategy
    assert len(plan.priority_gaps) <= constraints["max_gaps"]
    assert plan.confidence_improvement_estimate > 0

    # Test with diverse sources (should detect fewer gaps)
    expert_profile = ExpertProfile(
        name="Prof. AGI Expert",
        institution="MIT AI Lab",
        expertise_areas=[ExpertiseArea.ARTIFICIAL_INTELLIGENCE],
        h_index=50,
        reputation_score=0.95,
    )

    diverse_sources = [
        AuthoritativeSource(
            url="https://arxiv.org/abs/2024.1001",
            title="Recent Advances in AGI Research",
            summary="Comprehensive review of AGI progress with statistical analysis",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.9,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=30),
            authors=["Dr. AI Researcher"],
            knowledge_base=KnowledgeBase.ARXIV,
            peer_review_status="peer_reviewed",
        ),
        AuthoritativeSource(
            url="https://expert-network.com/agi-timeline",
            title="Expert Opinion on AGI Timeline",
            summary="Expert analysis of AGI development trends and challenges",
            source_type=SourceType.EXPERT_OPINION,
            credibility_score=0.85,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=15),
            expert_profile=expert_profile,
            knowledge_base=KnowledgeBase.EXPERT_NETWORKS,
        ),
        AuthoritativeSource(
            url="https://nsf.gov/ai-research-report",
            title="National AI Research Investment Report",
            summary="Government data on AI research funding and progress metrics",
            source_type=SourceType.GOVERNMENT_DATA,
            credibility_score=0.88,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=60),
            institution="National Science Foundation",
            knowledge_base=KnowledgeBase.GOVERNMENT_DATABASES,
        ),
    ]

    # Test with diverse sources
    diverse_gaps = detector.detect_knowledge_gaps(diverse_sources, question)
    assert len(diverse_gaps) < len(gaps), "Diverse sources should have fewer gaps"

    diverse_assessment = detector.assess_research_quality(diverse_sources, question)
    assert diverse_assessment.overall_quality in [
        ResearchQuality.HIGH,
        ResearchQuality.MEDIUM,
    ]
    assert diverse_assessment.confidence_level > assessment.confidence_level
    assert diverse_assessment.completeness_score > assessment.completeness_score

    print("✅ KnowledgeGapDetector integration test passed!")
    print(f"   - Limited sources: {len(gaps)} gaps detected")
    print(f"   - Diverse sources: {len(diverse_gaps)} gaps detected")
    print(
        f"   - Quality improvement: {assessment.overall_quality.value} → {diverse_assessment.overall_quality.value}"
    )
    print(
        f"   - Confidence improvement: {assessment.confidence_level:.2f} → {diverse_assessment.confidence_level:.2f}"
    )


if __name__ == "__main__":
    test_knowledge_gap_detector_integration()
