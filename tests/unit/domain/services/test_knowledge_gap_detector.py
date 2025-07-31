"""
Tests for KnowledgeGapDetector service.

Tests the knowledge gap detection, research quality assessment, and adaptive
research planning functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

from src.domain.services.knowledge_gap_detector import (
    KnowledgeGapDetector,
    KnowledgeGap,
    GapType,
    GapSeverity,
    ResearchStrategy,
    ResearchQualityAssessment,
    AdaptiveResearchPlan,
    SourceDiversityDetector,
    TemporalCoverageDetector,
    CredibilityGapDetector,
    QuantitativeDataDetector
)
from src.domain.services.authoritative_source_manager import (
    AuthoritativeSource,
    SourceType,
    ExpertiseArea,
    KnowledgeBase,
    ExpertProfile
)
from src.domain.entities.question import Question, QuestionType, QuestionStatus
from src.domain.entities.research_report import ResearchQuality


def create_test_question(title="Test question", metaculus_id=12346):
    """Helper function to create test questions."""
    return Question(
        id=uuid4(),
        metaculus_id=metaculus_id,
        title=title,
        description="Test description",
        question_type=QuestionType.BINARY,
        status=QuestionStatus.OPEN,
        url=f"https://metaculus.com/questions/{metaculus_id}",
        close_time=datetime.utcnow() + timedelta(days=30),
        resolve_time=None,
        categories=["Test"],
        metadata={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        resolution_criteria="Test criteria"
    )


class TestKnowledgeGapDetector:
    """Test suite for KnowledgeGapDetector."""

    @pytest.fixture
    def detector(self):
        """Create a KnowledgeGapDetector instance."""
        return KnowledgeGapDetector()

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return create_test_question("Will AI achieve AGI by 2030?", 12345)

    @pytest.fixture
    def diverse_sources(self):
        """Create a diverse set of high-quality sources."""
        sources = []

        # Academic source
        sources.append(AuthoritativeSource(
            url="https://arxiv.org/abs/2024.1001",
            title="Recent Advances in AGI Research",
            summary="Comprehensive review of AGI progress with statistical analysis",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.9,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=30),
            authors=["Dr. AI Researcher"],
            knowledge_base=KnowledgeBase.ARXIV,
            peer_review_status="peer_reviewed"
        ))

        # Expert opinion
        expert_profile = ExpertProfile(
            name="Prof. AGI Expert",
            institution="MIT AI Lab",
            expertise_areas=[ExpertiseArea.ARTIFICIAL_INTELLIGENCE],
            h_index=50,
            reputation_score=0.95
        )

        sources.append(AuthoritativeSource(
            url="https://expert-network.com/agi-timeline",
            title="Expert Opinion on AGI Timeline",
            summary="Expert analysis of AGI development trends and challenges",
            source_type=SourceType.EXPERT_OPINION,
            credibility_score=0.85,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=15),
            expert_profile=expert_profile,
            knowledge_base=KnowledgeBase.EXPERT_NETWORKS
        ))

        # Government data
        sources.append(AuthoritativeSource(
            url="https://nsf.gov/ai-research-report",
            title="National AI Research Investment Report",
            summary="Government data on AI research funding and progress metrics",
            source_type=SourceType.GOVERNMENT_DATA,
            credibility_score=0.88,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=60),
            institution="National Science Foundation",
            knowledge_base=KnowledgeBase.GOVERNMENT_DATABASES
        ))

        return sources

    @pytest.fixture
    def limited_sources(self):
        """Create a limited set of sources with gaps."""
        sources = []

        # Only one source type, outdated
        sources.append(AuthoritativeSource(
            url="https://old-blog.com/ai-predictions",
            title="Old AI Predictions",
            summary="Outdated predictions about AI development",
            source_type=SourceType.NEWS_ANALYSIS,
            credibility_score=0.4,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=800),  # Very old
            authors=["Blogger"]
        ))

        return sources

    def test_detect_knowledge_gaps_with_diverse_sources(self, detector, sample_question, diverse_sources):
        """Test gap detection with diverse, high-quality sources."""
        gaps = detector.detect_knowledge_gaps(diverse_sources, sample_question)

        # Should have minimal gaps with diverse sources
        assert len(gaps) <= 3

        # No critical gaps should be present
        critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
        assert len(critical_gaps) == 0

        # All gaps should have proper structure
        for gap in gaps:
            assert gap.gap_id
            assert gap.gap_type
            assert gap.severity
            assert gap.description
            assert 0.0 <= gap.impact_on_forecast <= 1.0
            assert 0.0 <= gap.confidence_reduction <= 1.0
            assert gap.question_context == sample_question.title

    def test_detect_knowledge_gaps_with_limited_sources(self, detector, sample_question, limited_sources):
        """Test gap detection with limited, low-quality sources."""
        gaps = detector.detect_knowledge_gaps(limited_sources, sample_question)

        # Should detect multiple gaps
        assert len(gaps) >= 3

        # Should have critical gaps
        critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
        assert len(critical_gaps) >= 1

        # Should detect specific gap types
        gap_types = {gap.gap_type for gap in gaps}
        expected_gaps = {
            GapType.SOURCE_DIVERSITY_GAP,
            GapType.RECENT_DEVELOPMENTS_GAP,
            GapType.CREDIBILITY_GAP
        }
        # At least some of these gaps should be present
        assert len(expected_gaps.intersection(gap_types)) >= 2

    def test_detect_knowledge_gaps_empty_sources(self, detector, sample_question):
        """Test gap detection with no sources."""
        gaps = detector.detect_knowledge_gaps([], sample_question)

        # Should detect critical gaps for missing sources
        assert len(gaps) >= 1

        # Should have at least one critical gap
        critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
        assert len(critical_gaps) >= 1

    def test_assess_research_quality_high_quality(self, detector, sample_question, diverse_sources):
        """Test research quality assessment with high-quality sources."""
        assessment = detector.assess_research_quality(diverse_sources, sample_question)

        assert assessment.overall_quality in [ResearchQuality.HIGH, ResearchQuality.MEDIUM]
        assert assessment.confidence_level >= 0.6
        assert assessment.completeness_score >= 0.6
        assert assessment.source_count == len(diverse_sources)
        assert assessment.source_diversity_score > 0.5

        # Should have fewer critical gaps
        assert len(assessment.critical_gaps) <= 1

        # Should have reasonable recommendations
        assert assessment.recommended_strategy
        assert len(assessment.priority_actions) > 0
        assert len(assessment.resource_allocation) > 0

    def test_assess_research_quality_low_quality(self, detector, sample_question, limited_sources):
        """Test research quality assessment with low-quality sources."""
        assessment = detector.assess_research_quality(limited_sources, sample_question)

        assert assessment.overall_quality == ResearchQuality.LOW
        assert assessment.confidence_level < 0.6
        assert assessment.completeness_score < 0.6
        assert assessment.source_diversity_score < 0.5

        # Should have critical gaps
        assert len(assessment.critical_gaps) >= 1

        # Should recommend intensive strategies
        assert assessment.recommended_strategy in [
            ResearchStrategy.INTENSIVE_SEARCH,
            ResearchStrategy.DIVERSIFICATION_FOCUS
        ]

    def test_create_adaptive_research_plan(self, detector, sample_question, limited_sources):
        """Test adaptive research plan creation."""
        assessment = detector.assess_research_quality(limited_sources, sample_question)

        constraints = {
            "total_time": timedelta(hours=6),
            "max_gaps": 3
        }

        plan = detector.create_adaptive_research_plan(assessment, sample_question, constraints)

        assert plan.plan_id
        assert plan.strategy
        assert len(plan.priority_gaps) <= constraints["max_gaps"]
        assert len(plan.search_expansions) >= 0
        assert len(plan.expert_consultations) >= 0
        assert len(plan.source_diversification) >= 0

        # Time allocation should sum to total time
        total_allocated = sum(plan.time_allocation.values(), timedelta())
        assert abs((total_allocated - constraints["total_time"]).total_seconds()) < 3600  # Within 1 hour

        # Effort distribution should sum to 1.0
        total_effort = sum(plan.effort_distribution.values())
        assert abs(total_effort - 1.0) < 0.1

        # Should have realistic improvement estimates
        assert 0.0 <= plan.confidence_improvement_estimate <= 1.0
        assert plan.cost_benefit_ratio > 0

    def test_gap_prioritization(self, detector, sample_question):
        """Test gap prioritization logic."""
        # Create gaps with different severities
        gaps = [
            KnowledgeGap(
                gap_id="low_gap",
                gap_type=GapType.QUANTITATIVE_DATA_GAP,
                severity=GapSeverity.LOW,
                description="Minor quantitative gap",
                impact_on_forecast=0.2,
                confidence_reduction=0.1,
                time_sensitivity=0.3
            ),
            KnowledgeGap(
                gap_id="critical_gap",
                gap_type=GapType.INSUFFICIENT_SOURCES,
                severity=GapSeverity.CRITICAL,
                description="Critical source shortage",
                impact_on_forecast=0.9,
                confidence_reduction=0.6,
                time_sensitivity=0.9
            ),
            KnowledgeGap(
                gap_id="medium_gap",
                gap_type=GapType.SOURCE_DIVERSITY_GAP,
                severity=GapSeverity.MEDIUM,
                description="Medium diversity gap",
                impact_on_forecast=0.5,
                confidence_reduction=0.3,
                time_sensitivity=0.6
            )
        ]

        prioritized = detector._prioritize_gaps(gaps)

        # Critical gap should be first
        assert prioritized[0].gap_id == "critical_gap"

        # Low severity gap should be last
        assert prioritized[-1].gap_id == "low_gap"

    def test_strategy_selection(self, detector):
        """Test research strategy selection based on gaps."""
        # Test with insufficient sources
        gaps = [
            KnowledgeGap(
                gap_id="insufficient",
                gap_type=GapType.INSUFFICIENT_SOURCES,
                severity=GapSeverity.HIGH,
                description="Not enough sources",
                impact_on_forecast=0.7,
                confidence_reduction=0.4,
                time_sensitivity=0.8
            )
        ]

        strategy = detector._select_research_strategy(gaps, 0.3)
        assert strategy == ResearchStrategy.INTENSIVE_SEARCH

        # Test with diversity gaps
        gaps = [
            KnowledgeGap(
                gap_id="diversity",
                gap_type=GapType.SOURCE_DIVERSITY_GAP,
                severity=GapSeverity.MEDIUM,
                description="Limited diversity",
                impact_on_forecast=0.5,
                confidence_reduction=0.3,
                time_sensitivity=0.6
            )
        ]

        strategy = detector._select_research_strategy(gaps, 0.7)
        assert strategy == ResearchStrategy.DIVERSIFICATION_FOCUS

    def test_resource_allocation(self, detector):
        """Test resource allocation calculation."""
        gaps = [
            KnowledgeGap(
                gap_id="sources",
                gap_type=GapType.INSUFFICIENT_SOURCES,
                severity=GapSeverity.HIGH,
                description="Need more sources",
                impact_on_forecast=0.7,
                confidence_reduction=0.4,
                time_sensitivity=0.8
            ),
            KnowledgeGap(
                gap_id="credibility",
                gap_type=GapType.CREDIBILITY_GAP,
                severity=GapSeverity.MEDIUM,
                description="Low credibility",
                impact_on_forecast=0.5,
                confidence_reduction=0.3,
                time_sensitivity=0.6
            )
        ]

        allocation = detector._calculate_resource_allocation(gaps)

        # Should sum to 1.0
        assert abs(sum(allocation.values()) - 1.0) < 0.01

        # Should have all expected categories
        expected_categories = {
            "search_expansion",
            "source_verification",
            "expert_consultation",
            "analysis_synthesis",
            "quality_assurance"
        }
        assert set(allocation.keys()) == expected_categories

        # Should allocate more to search expansion due to insufficient sources
        assert allocation["search_expansion"] > 0.3


class TestSourceDiversityDetector:
    """Test suite for SourceDiversityDetector."""

    @pytest.fixture
    def detector(self):
        return SourceDiversityDetector()

    @pytest.fixture
    def sample_question(self):
        return create_test_question()

    def test_detect_source_type_diversity_gap(self, detector, sample_question):
        """Test detection of source type diversity gaps."""
        # Create sources with limited diversity (only one type)
        sources = [
            AuthoritativeSource(
                url="https://example1.com",
                title="Source 1",
                summary="Summary 1",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.8,
                credibility_factors={}
            ),
            AuthoritativeSource(
                url="https://example2.com",
                title="Source 2",
                summary="Summary 2",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.8,
                credibility_factors={}
            )
        ]

        gaps = detector.detect_gaps(sources, sample_question, {})

        # Should detect source diversity gap
        diversity_gaps = [g for g in gaps if g.gap_type == GapType.SOURCE_DIVERSITY_GAP]
        assert len(diversity_gaps) >= 1

        gap = diversity_gaps[0]
        assert gap.severity in [GapSeverity.HIGH, GapSeverity.MEDIUM]
        assert "diversity" in gap.description.lower()

    def test_detect_expertise_gap(self, detector, sample_question):
        """Test detection of expertise area gaps."""
        # Create sources with limited expertise coverage
        expert_profile = ExpertProfile(
            name="Expert 1",
            institution="University",
            expertise_areas=[ExpertiseArea.ARTIFICIAL_INTELLIGENCE],  # Only one area
            reputation_score=0.9
        )

        sources = [
            AuthoritativeSource(
                url="https://expert1.com",
                title="Expert Opinion 1",
                summary="AI expert opinion",
                source_type=SourceType.EXPERT_OPINION,
                credibility_score=0.9,
                credibility_factors={},
                expert_profile=expert_profile
            )
        ]

        gaps = detector.detect_gaps(sources, sample_question, {})

        # Should detect expertise gap
        expertise_gaps = [g for g in gaps if g.gap_type == GapType.EXPERTISE_GAP]
        assert len(expertise_gaps) >= 1

        gap = expertise_gaps[0]
        assert gap.severity == GapSeverity.HIGH
        assert "expertise" in gap.description.lower()


class TestTemporalCoverageDetector:
    """Test suite for TemporalCoverageDetector."""

    @pytest.fixture
    def detector(self):
        return TemporalCoverageDetector()

    @pytest.fixture
    def sample_question(self):
        return create_test_question()

    def test_detect_no_publication_dates(self, detector, sample_question):
        """Test detection when sources have no publication dates."""
        sources = [
            AuthoritativeSource(
                url="https://example.com",
                title="Undated Source",
                summary="No publication date",
                source_type=SourceType.NEWS_ANALYSIS,
                credibility_score=0.7,
                credibility_factors={},
                publish_date=None  # No date
            )
        ]

        gaps = detector.detect_gaps(sources, sample_question, {})

        # Should detect critical temporal gap
        temporal_gaps = [g for g in gaps if g.gap_type == GapType.TEMPORAL_GAP]
        assert len(temporal_gaps) >= 1

        gap = temporal_gaps[0]
        assert gap.severity == GapSeverity.CRITICAL
        assert "publication dates" in gap.description.lower()

    def test_detect_insufficient_recent_sources(self, detector, sample_question):
        """Test detection of insufficient recent sources."""
        now = datetime.utcnow()

        sources = [
            AuthoritativeSource(
                url="https://old1.com",
                title="Old Source 1",
                summary="Very old information",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.8,
                credibility_factors={},
                publish_date=now - timedelta(days=500)  # Very old
            ),
            AuthoritativeSource(
                url="https://old2.com",
                title="Old Source 2",
                summary="Also old information",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.8,
                credibility_factors={},
                publish_date=now - timedelta(days=600)  # Very old
            )
        ]

        gaps = detector.detect_gaps(sources, sample_question, {})

        # Should detect recent developments gap
        recent_gaps = [g for g in gaps if g.gap_type == GapType.RECENT_DEVELOPMENTS_GAP]
        assert len(recent_gaps) >= 1

        gap = recent_gaps[0]
        assert gap.severity in [GapSeverity.CRITICAL, GapSeverity.HIGH]
        assert "recent" in gap.description.lower()


class TestCredibilityGapDetector:
    """Test suite for CredibilityGapDetector."""

    @pytest.fixture
    def detector(self):
        return CredibilityGapDetector()

    @pytest.fixture
    def sample_question(self):
        return create_test_question()

    def test_detect_insufficient_high_credibility(self, detector, sample_question):
        """Test detection of insufficient high-credibility sources."""
        sources = [
            AuthoritativeSource(
                url="https://low1.com",
                title="Low Credibility 1",
                summary="Unreliable source",
                source_type=SourceType.NEWS_ANALYSIS,
                credibility_score=0.5,  # Low credibility
                credibility_factors={}
            ),
            AuthoritativeSource(
                url="https://low2.com",
                title="Low Credibility 2",
                summary="Another unreliable source",
                source_type=SourceType.NEWS_ANALYSIS,
                credibility_score=0.6,  # Medium credibility
                credibility_factors={}
            )
        ]

        gaps = detector.detect_gaps(sources, sample_question, {})

        # Should detect credibility gap
        credibility_gaps = [g for g in gaps if g.gap_type == GapType.CREDIBILITY_GAP]
        assert len(credibility_gaps) >= 1

        gap = credibility_gaps[0]
        assert gap.severity == GapSeverity.HIGH
        assert "high-credibility" in gap.description.lower()


class TestQuantitativeDataDetector:
    """Test suite for QuantitativeDataDetector."""

    @pytest.fixture
    def detector(self):
        return QuantitativeDataDetector()

    @pytest.fixture
    def sample_question(self):
        return create_test_question()

    def test_detect_insufficient_quantitative_data(self, detector, sample_question):
        """Test detection of insufficient quantitative data."""
        sources = [
            AuthoritativeSource(
                url="https://qualitative1.com",
                title="Opinion Piece",
                summary="Subjective analysis without data or statistics",
                source_type=SourceType.EXPERT_OPINION,
                credibility_score=0.7,
                credibility_factors={}
            ),
            AuthoritativeSource(
                url="https://qualitative2.com",
                title="Commentary",
                summary="General commentary without numbers or measurements",
                source_type=SourceType.NEWS_ANALYSIS,
                credibility_score=0.6,
                credibility_factors={}
            )
        ]

        gaps = detector.detect_gaps(sources, sample_question, {})

        # Should detect quantitative data gap
        quant_gaps = [g for g in gaps if g.gap_type == GapType.QUANTITATIVE_DATA_GAP]
        assert len(quant_gaps) >= 1

        gap = quant_gaps[0]
        assert gap.severity == GapSeverity.MEDIUM
        assert "quantitative" in gap.description.lower()

    def test_no_gap_with_sufficient_quantitative_data(self, detector, sample_question):
        """Test that no gap is detected with sufficient quantitative data."""
        sources = [
            AuthoritativeSource(
                url="https://data1.com",
                title="Statistical Analysis",
                summary="Comprehensive data analysis with statistics and trends",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.9,
                credibility_factors={}
            ),
            AuthoritativeSource(
                url="https://data2.com",
                title="Research Study",
                summary="Empirical study with survey data and measurements",
                source_type=SourceType.PEER_REVIEWED,
                credibility_score=0.8,
                credibility_factors={}
            ),
            AuthoritativeSource(
                url="https://data3.com",
                title="Market Research",
                summary="Market analysis with percentage growth and rate trends",
                source_type=SourceType.INSTITUTIONAL_REPORT,
                credibility_score=0.8,
                credibility_factors={}
            )
        ]

        gaps = detector.detect_gaps(sources, sample_question, {})

        # Should not detect quantitative data gap
        quant_gaps = [g for g in gaps if g.gap_type == GapType.QUANTITATIVE_DATA_GAP]
        assert len(quant_gaps) == 0


if __name__ == "__main__":
    pytest.main([__file__])
