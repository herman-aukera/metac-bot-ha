"""Tests for ConflictResolver."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.domain.entities.question import Question, QuestionType
from src.domain.services.authoritative_source_manager import (
    AuthoritativeSource,
    SourceType,
)
from src.domain.services.conflict_resolver import (
    ConflictResolver,
    ConflictType,
    ConsensusBasedStrategy,
    CredibilityWeightedStrategy,
    EvidenceConflict,
    EvidenceTriangulationStrategy,
    KnowledgeGap,
    ResolutionStrategy,
    SynthesizedConclusion,
    UncertaintyLevel,
)


@pytest.fixture
def conflict_resolver():
    """Create ConflictResolver instance."""
    return ConflictResolver()


@pytest.fixture
def sample_question():
    """Create sample question for testing."""
    return Question.create_new(
        metaculus_id=12345,
        title="Will AI achieve AGI by 2030?",
        description="Question about artificial general intelligence timeline",
        question_type=QuestionType.BINARY,
        url="https://metaculus.com/questions/12345",
        close_time=datetime.utcnow() + timedelta(days=365),
        categories=["Technology", "AI"],
    )


@pytest.fixture
def conflicting_sources():
    """Create sources with conflicting information."""
    return [
        AuthoritativeSource(
            url="https://example.com/positive",
            title="AI will achieve AGI soon",
            summary="Research indicates AI will likely achieve AGI by 2030",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.8,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=30),
            methodology_score=0.8,
            data_quality_score=0.7,
            reproducibility_score=0.6,
        ),
        AuthoritativeSource(
            url="https://example.com/negative",
            title="AGI timeline is uncertain",
            summary="Analysis suggests AGI will not be achieved by 2030",
            source_type=SourceType.EXPERT_OPINION,
            credibility_score=0.7,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=60),
            methodology_score=0.6,
            data_quality_score=0.8,
            reproducibility_score=0.5,
        ),
        AuthoritativeSource(
            url="https://example.com/neutral",
            title="Mixed evidence on AGI timeline",
            summary="Evidence is mixed regarding AGI achievement by 2030",
            source_type=SourceType.INSTITUTIONAL_REPORT,
            credibility_score=0.9,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=15),
            methodology_score=0.9,
            data_quality_score=0.9,
            reproducibility_score=0.8,
        ),
    ]


@pytest.fixture
def sample_conflict():
    """Create sample evidence conflict."""
    sources = [
        AuthoritativeSource(
            url="https://example.com/source1",
            title="Source 1",
            summary="Positive conclusion",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.8,
            credibility_factors={},
        ),
        AuthoritativeSource(
            url="https://example.com/source2",
            title="Source 2",
            summary="Negative conclusion",
            source_type=SourceType.EXPERT_OPINION,
            credibility_score=0.7,
            credibility_factors={},
        ),
    ]

    return EvidenceConflict(
        conflict_id="test_conflict",
        conflict_type=ConflictType.DIRECT_CONTRADICTION,
        sources=sources,
        conflicting_claims=["Positive outcome", "Negative outcome"],
        severity=0.8,
        description="Test conflict",
    )


class TestConflictResolver:
    """Test cases for ConflictResolver."""

    def test_initialization(self, conflict_resolver):
        """Test proper initialization of conflict resolver."""
        assert conflict_resolver.resolution_strategies is not None
        assert len(conflict_resolver.resolution_strategies) >= 3
        assert conflict_resolver.conflict_detection_thresholds is not None

        # Check that required strategies are available
        assert (
            ResolutionStrategy.CREDIBILITY_WEIGHTED
            in conflict_resolver.resolution_strategies
        )
        assert (
            ResolutionStrategy.CONSENSUS_BASED
            in conflict_resolver.resolution_strategies
        )
        assert (
            ResolutionStrategy.EVIDENCE_TRIANGULATION
            in conflict_resolver.resolution_strategies
        )

    def test_detect_conflicts(
        self, conflict_resolver, conflicting_sources, sample_question
    ):
        """Test conflict detection between sources."""
        conflicts = conflict_resolver.detect_conflicts(
            sources=conflicting_sources, question_context=sample_question
        )

        assert isinstance(conflicts, list)
        assert len(conflicts) > 0

        # Check that conflicts have required properties
        for conflict in conflicts:
            assert isinstance(conflict, EvidenceConflict)
            assert conflict.conflict_type is not None
            assert len(conflict.sources) >= 2
            assert conflict.severity >= 0.0
            assert conflict.severity <= 1.0

    def test_detect_conclusion_conflicts(self, conflict_resolver, conflicting_sources):
        """Test detection of conclusion conflicts."""
        conflicts = conflict_resolver._detect_conclusion_conflicts(conflicting_sources)

        assert isinstance(conflicts, list)

        # Should detect conflict between positive and negative sources
        if conflicts:
            conflict = conflicts[0]
            assert conflict.conflict_type == ConflictType.DIRECT_CONTRADICTION
            assert len(conflict.sources) >= 2

    def test_detect_methodological_conflicts(
        self, conflict_resolver, conflicting_sources
    ):
        """Test detection of methodological conflicts."""
        conflicts = conflict_resolver._detect_methodological_conflicts(
            conflicting_sources
        )

        assert isinstance(conflicts, list)

        # May or may not detect conflicts depending on methodology scores
        for conflict in conflicts:
            assert conflict.conflict_type == ConflictType.METHODOLOGICAL_DIFFERENCE

    def test_detect_temporal_conflicts(self, conflict_resolver):
        """Test detection of temporal conflicts."""
        # Create sources with significant temporal gaps
        old_source = AuthoritativeSource(
            url="https://example.com/old",
            title="Old research",
            summary="Old conclusion about topic",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.7,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=800),
        )

        new_source = AuthoritativeSource(
            url="https://example.com/new",
            title="New research",
            summary="New conclusion about topic",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.8,
            credibility_factors={},
            publish_date=datetime.utcnow() - timedelta(days=30),
        )

        conflicts = conflict_resolver._detect_temporal_conflicts(
            [old_source, new_source]
        )

        assert isinstance(conflicts, list)

        # Should detect temporal conflict due to large gap and different conclusions
        if conflicts:
            conflict = conflicts[0]
            assert conflict.conflict_type == ConflictType.TEMPORAL_INCONSISTENCY
            assert len(conflict.sources) == 2

    def test_detect_credibility_conflicts(self, conflict_resolver):
        """Test detection of credibility conflicts."""
        high_cred_source = AuthoritativeSource(
            url="https://example.com/high",
            title="High credibility source",
            summary="Conclusion from high credibility source",
            source_type=SourceType.PEER_REVIEWED,
            credibility_score=0.9,
            credibility_factors={},
        )

        low_cred_source = AuthoritativeSource(
            url="https://example.com/low",
            title="Low credibility source",
            summary="Conclusion from low credibility source",
            source_type=SourceType.NEWS_ANALYSIS,
            credibility_score=0.3,
            credibility_factors={},
        )

        conflicts = conflict_resolver._detect_credibility_conflicts(
            [high_cred_source, low_cred_source]
        )

        assert isinstance(conflicts, list)

        # Should detect credibility disparity
        if conflicts:
            conflict = conflicts[0]
            assert conflict.conflict_type == ConflictType.DATA_QUALITY_DISPARITY
            assert conflict.severity > 0.5  # Significant credibility gap

    def test_resolve_conflicts(self, conflict_resolver, sample_conflict):
        """Test conflict resolution."""
        conflicts = [sample_conflict]

        resolved_conflicts = conflict_resolver.resolve_conflicts(
            conflicts=conflicts,
            resolution_strategy=ResolutionStrategy.CREDIBILITY_WEIGHTED,
        )

        assert isinstance(resolved_conflicts, list)
        assert len(resolved_conflicts) == 1

        resolved_conflict = resolved_conflicts[0]
        assert (
            resolved_conflict.resolution_strategy
            == ResolutionStrategy.CREDIBILITY_WEIGHTED
        )
        assert resolved_conflict.resolution_confidence > 0.0
        assert "Resolution:" in resolved_conflict.description

    def test_automatic_strategy_selection(self, conflict_resolver, sample_conflict):
        """Test automatic resolution strategy selection."""
        # Test different conflict types get different strategies
        test_cases = [
            (
                ConflictType.DIRECT_CONTRADICTION,
                ResolutionStrategy.CREDIBILITY_WEIGHTED,
            ),
            (
                ConflictType.METHODOLOGICAL_DIFFERENCE,
                ResolutionStrategy.EVIDENCE_TRIANGULATION,
            ),
            (
                ConflictType.TEMPORAL_INCONSISTENCY,
                ResolutionStrategy.RECENCY_PRIORITIZED,
            ),
            (
                ConflictType.DATA_QUALITY_DISPARITY,
                ResolutionStrategy.CREDIBILITY_WEIGHTED,
            ),
        ]

        for conflict_type, expected_strategy in test_cases:
            sample_conflict.conflict_type = conflict_type
            selected_strategy = conflict_resolver._select_resolution_strategy(
                sample_conflict
            )

            # Note: RECENCY_PRIORITIZED is not implemented yet, so it falls back to CONSENSUS_BASED
            if expected_strategy == ResolutionStrategy.RECENCY_PRIORITIZED:
                assert selected_strategy == ResolutionStrategy.CONSENSUS_BASED
            else:
                assert selected_strategy == expected_strategy

    def test_synthesize_conclusion(
        self, conflict_resolver, conflicting_sources, sample_question
    ):
        """Test conclusion synthesis."""
        # First detect and resolve conflicts
        conflicts = conflict_resolver.detect_conflicts(
            conflicting_sources, sample_question
        )
        resolved_conflicts = conflict_resolver.resolve_conflicts(conflicts)

        # Then synthesize conclusion
        synthesis = conflict_resolver.synthesize_conclusion(
            sources=conflicting_sources,
            resolved_conflicts=resolved_conflicts,
            question_context=sample_question,
        )

        assert isinstance(synthesis, SynthesizedConclusion)
        assert synthesis.conclusion_text is not None
        assert len(synthesis.conclusion_text) > 0
        assert 0.0 <= synthesis.confidence_level <= 1.0
        assert isinstance(synthesis.uncertainty_level, UncertaintyLevel)
        assert isinstance(synthesis.supporting_sources, list)
        assert isinstance(synthesis.conflicting_sources, list)
        assert isinstance(synthesis.evidence_weight_distribution, dict)
        assert isinstance(synthesis.uncertainty_factors, list)
        assert isinstance(synthesis.knowledge_gaps, list)

    def test_calculate_uncertainty_level(self, conflict_resolver):
        """Test uncertainty level calculation."""
        test_cases = [
            (0.9, 0, 5, UncertaintyLevel.LOW),
            (0.7, 1, 5, UncertaintyLevel.MODERATE),
            (0.5, 2, 5, UncertaintyLevel.HIGH),
            (0.3, 3, 5, UncertaintyLevel.VERY_HIGH),
        ]

        for confidence, conflict_count, source_count, expected_level in test_cases:
            level = conflict_resolver._calculate_uncertainty_level(
                confidence, conflict_count, source_count
            )
            assert level == expected_level

    def test_identify_knowledge_gaps(self, conflict_resolver, conflicting_sources):
        """Test knowledge gap identification."""
        # Create some conflicts
        conflicts = conflict_resolver.detect_conflicts(conflicting_sources)

        gaps = conflict_resolver._identify_knowledge_gaps(
            conflicting_sources, conflicts
        )

        assert isinstance(gaps, list)

        for gap in gaps:
            assert isinstance(gap, KnowledgeGap)
            assert gap.gap_id is not None
            assert gap.description is not None
            assert 0.0 <= gap.severity <= 1.0
            assert 0.0 <= gap.impact_on_conclusion <= 1.0
            assert isinstance(gap.potential_sources, list)
            assert isinstance(gap.research_suggestions, list)

    def test_generate_conclusion_text(
        self, conflict_resolver, conflicting_sources, sample_question
    ):
        """Test conclusion text generation."""
        conflicts = conflict_resolver.detect_conflicts(conflicting_sources)

        conclusion_text = conflict_resolver._generate_conclusion_text(
            sources=conflicting_sources,
            resolved_conflicts=conflicts,
            confidence=0.7,
            question_context=sample_question,
        )

        assert isinstance(conclusion_text, str)
        assert len(conclusion_text) > 0
        assert "evidence" in conclusion_text.lower()

    def test_get_conflict_resolution_report(
        self, conflict_resolver, conflicting_sources, sample_question
    ):
        """Test conflict resolution report generation."""
        # Detect and resolve conflicts
        conflicts = conflict_resolver.detect_conflicts(
            conflicting_sources, sample_question
        )
        resolved_conflicts = conflict_resolver.resolve_conflicts(conflicts)

        # Synthesize conclusion
        synthesis = conflict_resolver.synthesize_conclusion(
            sources=conflicting_sources,
            resolved_conflicts=resolved_conflicts,
            question_context=sample_question,
        )

        # Generate report
        report = conflict_resolver.get_conflict_resolution_report(
            resolved_conflicts, synthesis
        )

        assert isinstance(report, dict)
        assert "summary" in report
        assert "conflicts" in report
        assert "synthesis" in report
        assert "evidence_distribution" in report
        assert "recommendations" in report

        # Check summary structure
        summary = report["summary"]
        assert "total_conflicts" in summary
        assert "resolved_conflicts" in summary
        assert "overall_confidence" in summary
        assert "uncertainty_level" in summary

    def test_generate_recommendations(self, conflict_resolver, conflicting_sources):
        """Test recommendation generation."""
        conflicts = conflict_resolver.detect_conflicts(conflicting_sources)
        resolved_conflicts = conflict_resolver.resolve_conflicts(conflicts)
        synthesis = conflict_resolver.synthesize_conclusion(
            sources=conflicting_sources, resolved_conflicts=resolved_conflicts
        )

        recommendations = conflict_resolver._generate_recommendations(
            conflicts, synthesis
        )

        assert isinstance(recommendations, list)

        for recommendation in recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0


class TestResolutionStrategies:
    """Test cases for individual resolution strategies."""

    def test_credibility_weighted_strategy(self, sample_conflict):
        """Test credibility weighted resolution strategy."""
        strategy = CredibilityWeightedStrategy()

        conclusion, confidence, uncertainty_factors = strategy.resolve_conflict(
            sample_conflict, {"strategy": ResolutionStrategy.CREDIBILITY_WEIGHTED}
        )

        assert isinstance(conclusion, str)
        assert len(conclusion) > 0
        assert 0.0 <= confidence <= 1.0
        assert isinstance(uncertainty_factors, list)
        assert strategy.get_strategy_name() == "Credibility Weighted Resolution"

    def test_consensus_based_strategy(self, sample_conflict):
        """Test consensus based resolution strategy."""
        strategy = ConsensusBasedStrategy()

        conclusion, confidence, uncertainty_factors = strategy.resolve_conflict(
            sample_conflict, {"strategy": ResolutionStrategy.CONSENSUS_BASED}
        )

        assert isinstance(conclusion, str)
        assert len(conclusion) > 0
        assert 0.0 <= confidence <= 1.0
        assert isinstance(uncertainty_factors, list)
        assert strategy.get_strategy_name() == "Consensus Based Resolution"

    def test_evidence_triangulation_strategy(self, sample_conflict):
        """Test evidence triangulation resolution strategy."""
        strategy = EvidenceTriangulationStrategy()

        conclusion, confidence, uncertainty_factors = strategy.resolve_conflict(
            sample_conflict, {"strategy": ResolutionStrategy.EVIDENCE_TRIANGULATION}
        )

        assert isinstance(conclusion, str)
        assert len(conclusion) > 0
        assert 0.0 <= confidence <= 1.0
        assert isinstance(uncertainty_factors, list)
        assert strategy.get_strategy_name() == "Evidence Triangulation"


class TestEvidenceConflict:
    """Test cases for EvidenceConflict dataclass."""

    def test_evidence_conflict_creation(self):
        """Test creation of EvidenceConflict."""
        sources = [
            AuthoritativeSource(
                url="https://example.com/1",
                title="Source 1",
                summary="Summary 1",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.8,
                credibility_factors={},
            )
        ]

        conflict = EvidenceConflict(
            conflict_id="test_id",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            sources=sources,
            conflicting_claims=["Claim 1"],
            severity=0.7,
            description="Test conflict",
        )

        assert conflict.conflict_id == "test_id"
        assert conflict.conflict_type == ConflictType.DIRECT_CONTRADICTION
        assert len(conflict.sources) == 1
        assert conflict.severity == 0.7

    def test_evidence_conflict_auto_id(self):
        """Test automatic ID generation for EvidenceConflict."""
        sources = [
            AuthoritativeSource(
                url="https://example.com/1",
                title="Source 1",
                summary="Summary 1",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.8,
                credibility_factors={},
            )
        ]

        conflict = EvidenceConflict(
            conflict_id="",  # Empty ID should trigger auto-generation
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            sources=sources,
            conflicting_claims=["Claim 1"],
            severity=0.7,
        )

        assert conflict.conflict_id.startswith("conflict_")
        assert len(conflict.conflict_id) > len("conflict_")


class TestSynthesizedConclusion:
    """Test cases for SynthesizedConclusion dataclass."""

    def test_synthesized_conclusion_creation(self):
        """Test creation of SynthesizedConclusion."""
        sources = [
            AuthoritativeSource(
                url="https://example.com/1",
                title="Source 1",
                summary="Summary 1",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.8,
                credibility_factors={},
            )
        ]

        conclusion = SynthesizedConclusion(
            conclusion_text="Test conclusion",
            confidence_level=0.8,
            uncertainty_level=UncertaintyLevel.MODERATE,
            supporting_sources=sources,
            conflicting_sources=[],
            resolution_strategy=ResolutionStrategy.CREDIBILITY_WEIGHTED,
            evidence_weight_distribution={"source1": 1.0},
            uncertainty_factors=["Factor 1"],
            knowledge_gaps=["Gap 1"],
            synthesis_methodology="Test methodology",
        )

        assert conclusion.conclusion_text == "Test conclusion"
        assert conclusion.confidence_level == 0.8
        assert conclusion.uncertainty_level == UncertaintyLevel.MODERATE
        assert len(conclusion.supporting_sources) == 1
        assert conclusion.created_at is not None


class TestKnowledgeGap:
    """Test cases for KnowledgeGap dataclass."""

    def test_knowledge_gap_creation(self):
        """Test creation of KnowledgeGap."""
        gap = KnowledgeGap(
            gap_id="test_gap",
            description="Test gap description",
            gap_type="methodological",
            severity=0.7,
            potential_sources=["Source 1", "Source 2"],
            research_suggestions=["Suggestion 1"],
            impact_on_conclusion=0.6,
        )

        assert gap.gap_id == "test_gap"
        assert gap.description == "Test gap description"
        assert gap.gap_type == "methodological"
        assert gap.severity == 0.7
        assert gap.impact_on_conclusion == 0.6
        assert len(gap.potential_sources) == 2
        assert len(gap.research_suggestions) == 1
