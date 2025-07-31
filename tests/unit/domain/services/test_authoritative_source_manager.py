"""Tests for AuthoritativeSourceManager."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.domain.services.authoritative_source_manager import (
    AuthoritativeSourceManager,
    AuthoritativeSource,
    SourceType,
    CredibilityFactor
)
from src.domain.entities.question import Question


@pytest.fixture
def source_manager():
    """Create AuthoritativeSourceManager instance."""
    return AuthoritativeSourceManager()


@pytest.fixture
def sample_question():
    """Create sample question for testing."""
    from src.domain.entities.question import QuestionType
    return Question.create_new(
        metaculus_id=12345,
        title="Will AI achieve AGI by 2030?",
        description="Question about artificial general intelligence timeline",
        question_type=QuestionType.BINARY,
        url="https://metaculus.com/questions/12345",
        close_time=datetime.utcnow() + timedelta(days=365),
        categories=["Technology", "AI"]
    )


@pytest.fixture
def sample_academic_source():
    """Create sample academic source."""
    return AuthoritativeSource(
        url="https://arxiv.org/abs/2024.1001",
        title="Advances in Artificial General Intelligence",
        summary="Comprehensive review of AGI research progress",
        source_type=SourceType.ACADEMIC_PAPER,
        credibility_score=0.0,  # Will be calculated
        credibility_factors={},
        publish_date=datetime.utcnow() - timedelta(days=30),
        authors=["Dr. AI Researcher", "Prof. ML Expert"],
        institution="MIT",
        journal_or_venue="Nature Communications",
        citation_count=75
    )


class TestAuthoritativeSourceManager:
    """Test cases for AuthoritativeSourceManager."""

    def test_initialization(self, source_manager):
        """Test proper initialization of source manager."""
        assert source_manager.domain_authority_scores is not None
        assert source_manager.journal_impact_scores is not None
        assert source_manager.institution_rankings is not None
        assert source_manager.expert_databases is not None

        # Check some known high-authority domains
        assert source_manager.domain_authority_scores["nature.com"] > 0.9
        assert source_manager.domain_authority_scores["arxiv.org"] > 0.8
        assert source_manager.domain_authority_scores["census.gov"] > 0.9

    @pytest.mark.asyncio
    async def test_find_authoritative_sources(self, source_manager, sample_question):
        """Test finding authoritative sources for a question."""
        sources = await source_manager.find_authoritative_sources(
            question=sample_question,
            max_sources=10,
            min_credibility=0.5
        )

        assert isinstance(sources, list)
        assert len(sources) <= 10

        # All sources should meet minimum credibility
        for source in sources:
            assert source.credibility_score >= 0.5
            assert isinstance(source, AuthoritativeSource)

        # Sources should be sorted by credibility (descending)
        if len(sources) > 1:
            for i in range(len(sources) - 1):
                assert sources[i].credibility_score >= sources[i + 1].credibility_score

    @pytest.mark.asyncio
    async def test_find_sources_by_type(self, source_manager, sample_question):
        """Test finding sources filtered by type."""
        academic_sources = await source_manager.find_authoritative_sources(
            question=sample_question,
            source_types=[SourceType.ACADEMIC_PAPER],
            max_sources=5
        )

        assert isinstance(academic_sources, list)
        for source in academic_sources:
            assert source.source_type == SourceType.ACADEMIC_PAPER

    def test_calculate_credibility_score(self, source_manager, sample_academic_source):
        """Test credibility score calculation."""
        score = source_manager.calculate_credibility_score(sample_academic_source)

        assert 0.0 <= score <= 1.0
        assert sample_academic_source.credibility_score == score
        assert len(sample_academic_source.credibility_factors) > 0

        # Check that all credibility factors are present
        expected_factors = list(CredibilityFactor)
        for factor in expected_factors:
            assert factor in sample_academic_source.credibility_factors

    def test_domain_extraction(self, source_manager):
        """Test domain extraction from URLs."""
        test_cases = [
            ("https://www.nature.com/articles/123", "nature.com"),
            ("http://arxiv.org/abs/2024.1001", "arxiv.org"),
            ("https://census.gov/data/report", "census.gov"),
            ("https://www.example.com/path", "example.com")
        ]

        for url, expected_domain in test_cases:
            domain = source_manager._extract_domain(url)
            assert domain == expected_domain

    def test_peer_review_assessment(self, source_manager):
        """Test peer review status assessment."""
        # Peer-reviewed source
        peer_reviewed_source = AuthoritativeSource(
            url="https://nature.com/article",
            title="Test Article",
            summary="Test summary",
            source_type=SourceType.PEER_REVIEWED,
            credibility_score=0.0,
            credibility_factors={}
        )

        score = source_manager._assess_peer_review_status(peer_reviewed_source)
        assert score == 0.9

        # Preprint source
        preprint_source = AuthoritativeSource(
            url="https://arxiv.org/abs/2024.1001",
            title="Test Preprint",
            summary="Test summary",
            source_type=SourceType.PREPRINT,
            credibility_score=0.0,
            credibility_factors={}
        )

        score = source_manager._assess_peer_review_status(preprint_source)
        assert score == 0.4

    def test_citation_impact_assessment(self, source_manager):
        """Test citation impact assessment."""
        test_cases = [
            (150, 0.9),  # High citations
            (75, 0.8),   # Medium-high citations
            (25, 0.7),   # Medium citations
            (15, 0.6),   # Low-medium citations
            (5, 0.5),    # Low citations
            (None, 0.5)  # No citation data
        ]

        for citation_count, expected_score in test_cases:
            source = AuthoritativeSource(
                url="https://example.com",
                title="Test",
                summary="Test",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.0,
                credibility_factors={},
                citation_count=citation_count
            )

            score = source_manager._assess_citation_impact(source)
            assert score == expected_score

    def test_recency_assessment(self, source_manager):
        """Test recency assessment."""
        now = datetime.utcnow()

        test_cases = [
            (now - timedelta(days=15), 0.9),    # Very recent
            (now - timedelta(days=60), 0.8),    # Recent
            (now - timedelta(days=120), 0.7),   # Moderately recent
            (now - timedelta(days=300), 0.6),   # Somewhat old
            (now - timedelta(days=600), 0.5),   # Old
            (now - timedelta(days=1000), 0.4),  # Very old
            (None, 0.5)                         # No date
        ]

        for publish_date, expected_score in test_cases:
            source = AuthoritativeSource(
                url="https://example.com",
                title="Test",
                summary="Test",
                source_type=SourceType.ACADEMIC_PAPER,
                credibility_score=0.0,
                credibility_factors={},
                publish_date=publish_date
            )

            score = source_manager._assess_recency(source)
            assert score == expected_score

    def test_source_validation(self, source_manager, sample_academic_source):
        """Test source authenticity validation."""
        # Valid source
        is_valid, issues = source_manager.validate_source_authenticity(sample_academic_source)
        assert is_valid
        assert len(issues) == 0

        # Invalid source - missing URL
        invalid_source = AuthoritativeSource(
            url="",
            title="Test",
            summary="Test",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.5,
            credibility_factors={}
        )

        is_valid, issues = source_manager.validate_source_authenticity(invalid_source)
        assert not is_valid
        assert "Invalid or missing URL" in issues

        # Invalid source - missing title
        invalid_source.url = "https://example.com"
        invalid_source.title = ""

        is_valid, issues = source_manager.validate_source_authenticity(invalid_source)
        assert not is_valid
        assert "Missing title" in issues

    def test_credibility_breakdown(self, source_manager, sample_academic_source):
        """Test credibility breakdown functionality."""
        # Calculate credibility first
        source_manager.calculate_credibility_score(sample_academic_source)

        breakdown = source_manager.get_source_credibility_breakdown(sample_academic_source)

        assert "overall_score" in breakdown
        assert "factors" in breakdown
        assert "source_type" in breakdown
        assert "domain" in breakdown

        assert breakdown["overall_score"] == sample_academic_source.credibility_score
        assert breakdown["source_type"] == SourceType.ACADEMIC_PAPER.value
        assert len(breakdown["factors"]) > 0

    def test_to_research_source_conversion(self, sample_academic_source):
        """Test conversion to ResearchSource."""
        research_source = sample_academic_source.to_research_source()

        assert research_source.url == sample_academic_source.url
        assert research_source.title == sample_academic_source.title
        assert research_source.summary == sample_academic_source.summary
        assert research_source.credibility_score == sample_academic_source.credibility_score
        assert research_source.publish_date == sample_academic_source.publish_date
        assert research_source.source_type == sample_academic_source.source_type.value

    def test_suspicious_pattern_detection(self, source_manager):
        """Test detection of suspicious patterns."""
        # Source with suspicious language
        suspicious_source = AuthoritativeSource(
            url="https://example.com",
            title="Amazing Breakthrough in AI - Guaranteed Results!",
            summary="This revolutionary secret will change everything",
            source_type=SourceType.NEWS_ANALYSIS,
            credibility_score=0.5,
            credibility_factors={}
        )

        has_suspicious = source_manager._has_suspicious_patterns(suspicious_source)
        assert has_suspicious

        # Normal source
        normal_source = AuthoritativeSource(
            url="https://nature.com",
            title="Research on AI Development",
            summary="Systematic analysis of artificial intelligence progress",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.8,
            credibility_factors={}
        )

        has_suspicious = source_manager._has_suspicious_patterns(normal_source)
        assert not has_suspicious

    def test_supported_types_and_factors(self, source_manager):
        """Test getting supported types and factors."""
        source_types = source_manager.get_supported_source_types()
        assert isinstance(source_types, list)
        assert len(source_types) > 0
        assert all(isinstance(st, SourceType) for st in source_types)

        credibility_factors = source_manager.get_credibility_factors()
        assert isinstance(credibility_factors, list)
        assert len(credibility_factors) > 0
        assert all(isinstance(cf, CredibilityFactor) for cf in credibility_factors)


class TestAuthoritativeSource:
    """Test cases for AuthoritativeSource dataclass."""

    def test_authoritative_source_creation(self):
        """Test creation of AuthoritativeSource."""
        source = AuthoritativeSource(
            url="https://example.com",
            title="Test Source",
            summary="Test summary",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.8,
            credibility_factors={}
        )

        assert source.url == "https://example.com"
        assert source.title == "Test Source"
        assert source.source_type == SourceType.ACADEMIC_PAPER
        assert source.credibility_score == 0.8
        assert source.authors == []  # Default empty list
        assert source.access_date is not None  # Auto-set

    def test_authoritative_source_with_authors(self):
        """Test AuthoritativeSource with authors."""
        authors = ["Dr. Smith", "Prof. Jones"]
        source = AuthoritativeSource(
            url="https://example.com",
            title="Test Source",
            summary="Test summary",
            source_type=SourceType.ACADEMIC_PAPER,
            credibility_score=0.8,
            credibility_factors={},
            authors=authors
        )

        assert source.authors == authors


class TestEnhancedAuthoritativeSourceManager:
    """Test cases for enhanced AuthoritativeSourceManager functionality."""

    @pytest.mark.asyncio
    async def test_find_authoritative_sources_enhanced(self, source_manager, sample_question):
        """Test enhanced source finding with knowledge base integration."""
        from src.domain.services.authoritative_source_manager import KnowledgeBaseQuery, KnowledgeBase, ExpertiseArea

        query_config = KnowledgeBaseQuery(
            query_text=sample_question.title,
            knowledge_bases=[KnowledgeBase.ARXIV, KnowledgeBase.EXPERT_NETWORKS],
            max_results=5,
            min_credibility=0.6,
            expertise_areas=[ExpertiseArea.ARTIFICIAL_INTELLIGENCE]
        )

        sources = await source_manager.find_authoritative_sources_enhanced(
            question=sample_question,
            query_config=query_config,
            max_sources=10
        )

        assert isinstance(sources, list)
        assert len(sources) <= 10

        # All sources should meet minimum credibility
        for source in sources:
            assert source.credibility_score >= 0.6
            assert hasattr(source, 'knowledge_base')
            assert hasattr(source, 'methodology_score')
            assert hasattr(source, 'data_quality_score')

    @pytest.mark.asyncio
    async def test_search_specialized_knowledge_bases(self, source_manager, sample_question):
        """Test specialized knowledge base search."""
        from src.domain.services.authoritative_source_manager import (
            KnowledgeBaseQuery, KnowledgeBase, ExpertiseArea
        )

        query = KnowledgeBaseQuery(
            query_text=sample_question.title,
            knowledge_bases=[KnowledgeBase.ARXIV, KnowledgeBase.PUBMED],
            max_results=5
        )

        sources = await source_manager.search_specialized_knowledge_bases(
            query=query,
            expertise_areas=[ExpertiseArea.ARTIFICIAL_INTELLIGENCE, ExpertiseArea.HEALTHCARE]
        )

        assert isinstance(sources, list)
        for source in sources:
            assert source.knowledge_base is not None
            assert source.credibility_score > 0

    def test_calculate_enhanced_credibility_score(self, source_manager):
        """Test enhanced credibility score calculation."""
        from src.domain.services.authoritative_source_manager import (
            AuthoritativeSource, SourceType, ExpertProfile, ExpertiseArea, KnowledgeBase
        )

        # Create source with expert profile
        expert_profile = ExpertProfile(
            name="Dr. Test Expert",
            institution="Test University",
            expertise_areas=[ExpertiseArea.ARTIFICIAL_INTELLIGENCE],
            h_index=30,
            years_experience=10,
            reputation_score=0.85
        )

        source = AuthoritativeSource(
            url="https://arxiv.org/abs/2024.1001",
            title="Test Paper",
            summary="Test summary",
            source_type=SourceType.EXPERT_OPINION,
            credibility_score=0.0,
            credibility_factors={},
            expert_profile=expert_profile,
            knowledge_base=KnowledgeBase.EXPERT_NETWORKS,
            doi="10.1000/test.2024.001",
            abstract="Detailed abstract for testing methodology assessment"
        )

        score = source_manager.calculate_enhanced_credibility_score(source)

        assert 0.0 <= score <= 1.0
        assert source.credibility_score == score
        assert source.methodology_score > 0
        assert source.data_quality_score > 0
        assert source.reproducibility_score > 0
        assert source.expert_consensus_score > 0

    @pytest.mark.asyncio
    async def test_validate_source_authenticity_enhanced(self, source_manager):
        """Test enhanced source validation."""
        from src.domain.services.authoritative_source_manager import (
            AuthoritativeSource, SourceType, ExpertProfile, ExpertiseArea, KnowledgeBase
        )

        expert_profile = ExpertProfile(
            name="Dr. Valid Expert",
            institution="Valid University",
            expertise_areas=[ExpertiseArea.ARTIFICIAL_INTELLIGENCE],
            reputation_score=0.8
        )

        valid_source = AuthoritativeSource(
            url="https://arxiv.org/abs/2024.1001",
            title="Valid Paper",
            summary="Valid summary",
            source_type=SourceType.PREPRINT,
            credibility_score=0.8,
            credibility_factors={},
            expert_profile=expert_profile,
            knowledge_base=KnowledgeBase.ARXIV,
            doi="10.1000/valid.2024.001",
            publish_date=datetime.utcnow() - timedelta(days=30)
        )

        is_valid, issues, details = await source_manager.validate_source_authenticity_enhanced(valid_source)

        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
        assert isinstance(details, dict)
        assert "overall_valid" in details
        assert "knowledge_base_validation" in details
        assert "expert_validation" in details
        assert "metadata_validation" in details

    def test_get_knowledge_base_capabilities(self, source_manager):
        """Test knowledge base capabilities retrieval."""
        capabilities = source_manager.get_knowledge_base_capabilities()

        assert isinstance(capabilities, dict)
        assert len(capabilities) > 0

        for kb_type, capability in capabilities.items():
            assert "supported_expertise_areas" in capability
            assert "description" in capability
            assert isinstance(capability["supported_expertise_areas"], list)
            assert isinstance(capability["description"], str)

    @pytest.mark.asyncio
    async def test_get_expert_recommendations(self, source_manager):
        """Test expert recommendations."""
        from src.domain.services.authoritative_source_manager import ExpertiseArea

        experts = await source_manager.get_expert_recommendations(
            expertise_areas=[ExpertiseArea.ARTIFICIAL_INTELLIGENCE],
            min_reputation=0.7
        )

        assert isinstance(experts, list)
        for expert in experts:
            assert expert.reputation_score >= 0.7
            assert ExpertiseArea.ARTIFICIAL_INTELLIGENCE in expert.expertise_areas

    def test_get_source_quality_metrics(self, source_manager, sample_academic_source):
        """Test source quality metrics."""
        # Calculate credibility first
        source_manager.calculate_enhanced_credibility_score(sample_academic_source)

        metrics = source_manager.get_source_quality_metrics(sample_academic_source)

        assert isinstance(metrics, dict)
        expected_metrics = [
            "overall_credibility", "methodology_quality", "data_quality",
            "reproducibility", "expert_consensus", "domain_authority",
            "peer_review_status", "recency"
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0.0 <= metrics[metric] <= 1.0

    def test_enhanced_metadata_functionality(self, sample_academic_source):
        """Test enhanced metadata functionality."""
        from src.domain.services.authoritative_source_manager import ExpertProfile, ExpertiseArea, KnowledgeBase

        # Add enhanced metadata
        sample_academic_source.doi = "10.1000/test.2024.001"
        sample_academic_source.abstract = "This is a detailed abstract for testing purposes"
        sample_academic_source.keywords = ["AI", "machine learning", "research"]
        sample_academic_source.knowledge_base = KnowledgeBase.ARXIV
        sample_academic_source.expert_profile = ExpertProfile(
            name="Test Expert",
            institution="Test Institution",
            expertise_areas=[ExpertiseArea.ARTIFICIAL_INTELLIGENCE],
            reputation_score=0.9
        )

        # Test enhanced metadata retrieval
        metadata = sample_academic_source.get_enhanced_metadata()

        assert isinstance(metadata, dict)
        assert "basic_info" in metadata
        assert "credibility" in metadata
        assert "academic_metrics" in metadata
        assert "source_details" in metadata
        assert "expert_info" in metadata

        # Check specific fields
        assert metadata["basic_info"]["title"] == sample_academic_source.title
        assert metadata["academic_metrics"]["doi"] == sample_academic_source.doi
        assert metadata["source_details"]["knowledge_base"] == KnowledgeBase.ARXIV.value
        assert metadata["expert_info"]["expert_name"] == "Test Expert"
