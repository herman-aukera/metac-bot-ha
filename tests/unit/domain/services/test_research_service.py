"""Unit tests for ResearchService domain service."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.domain.entities.question import Question, QuestionType
from src.domain.entities.research_report import (
    ResearchQuality,
    ResearchReport,
    ResearchSource,
)
from src.domain.services.research_service import ResearchService


class TestResearchService:
    """Test suite for ResearchService class."""

    @pytest.fixture
    def research_service(self):
        """Create a ResearchService instance."""
        return ResearchService()

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return Question.create_new(
            metaculus_id=12345,
            title="Will AI achieve AGI by 2030?",
            description="This question asks about artificial general intelligence timeline.",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=datetime.now(timezone.utc) + timedelta(days=365),
            categories=["AI", "Technology"],
        )

    @pytest.fixture
    def sample_sources(self):
        """Create sample research sources."""
        return [
            ResearchSource(
                url="https://example.com/ai-report",
                title="AI Progress Report 2025",
                summary="Recent advances in AI research show significant progress...",
                credibility_score=0.8,
                publish_date=datetime.now(timezone.utc) - timedelta(days=30),
                source_type="academic",
            ),
            ResearchSource(
                url="https://example.com/expert-survey",
                title="Expert Survey on AGI Timeline",
                summary="Survey of AI researchers shows mixed opinions...",
                credibility_score=0.7,
                publish_date=datetime.now(timezone.utc) - timedelta(days=15),
                source_type="survey",
            ),
            ResearchSource(
                url="https://example.com/low-quality",
                title="Random Blog Post",
                summary="Someone's opinion on AI...",
                credibility_score=0.3,
                publish_date=datetime.now(timezone.utc) - timedelta(days=5),
                source_type="blog",
            ),
        ]

    def test_init(self, research_service):
        """Test ResearchService initialization."""
        assert isinstance(research_service.supported_providers, list)
        assert "duckduckgo" in research_service.supported_providers
        assert "exa" in research_service.supported_providers
        assert isinstance(research_service.quality_thresholds, dict)
        assert "high" in research_service.quality_thresholds
        assert "medium" in research_service.quality_thresholds
        assert "low" in research_service.quality_thresholds

    def test_identify_research_areas_basic(self, research_service, sample_question):
        """Test identification of basic research areas."""
        areas = research_service._identify_research_areas(sample_question)

        assert isinstance(areas, list)
        assert len(areas) > 0
        assert "historical trends" in areas
        assert "current status" in areas
        assert "expert opinions" in areas
        assert "market indicators" in areas
        assert "policy implications" in areas

    def test_identify_research_areas_technology_specific(self, research_service):
        """Test identification of technology-specific research areas."""
        tech_question = Question.create_new(
            metaculus_id=123,
            title="Will AI technology be widely adopted by 2025?",
            description="Question about AI technology adoption",
            question_type=QuestionType.BINARY,
            url="https://example.com/tech",
            close_time=datetime.now(timezone.utc) + timedelta(days=300),
            categories=["Technology"],
        )

        areas = research_service._identify_research_areas(tech_question)

        assert "technology adoption" in areas
        assert "innovation metrics" in areas

    def test_identify_research_areas_economic_specific(self, research_service):
        """Test identification of economic-specific research areas."""
        economic_question = Question.create_new(
            metaculus_id=124,
            title="Will the economy grow by 3% next year?",
            description="Question about economic growth",
            question_type=QuestionType.BINARY,
            url="https://example.com/economy",
            close_time=datetime.now(timezone.utc) + timedelta(days=300),
            categories=["Economy"],
        )

        areas = research_service._identify_research_areas(economic_question)

        assert "economic indicators" in areas
        assert "financial metrics" in areas

    def test_identify_research_areas_political_specific(self, research_service):
        """Test identification of political-specific research areas."""
        political_question = Question.create_new(
            metaculus_id=125,
            title="Will the election result in a policy change?",
            description="Question about political outcomes",
            question_type=QuestionType.BINARY,
            url="https://example.com/politics",
            close_time=datetime.now(timezone.utc) + timedelta(days=300),
            categories=["Politics"],
        )

        areas = research_service._identify_research_areas(political_question)

        assert "polling data" in areas
        assert "political analysis" in areas

    def test_identify_research_areas_health_specific(self, research_service):
        """Test identification of health-specific research areas."""
        health_question = Question.create_new(
            metaculus_id=126,
            title="Will a new pandemic emerge by 2026?",
            description="Question about health risks",
            question_type=QuestionType.BINARY,
            url="https://example.com/health",
            close_time=datetime.now(timezone.utc) + timedelta(days=300),
            categories=["Health"],
        )

        areas = research_service._identify_research_areas(health_question)

        assert "medical research" in areas
        assert "health statistics" in areas

    def test_identify_research_areas_climate_specific(self, research_service):
        """Test identification of climate-specific research areas."""
        climate_question = Question.create_new(
            metaculus_id=127,
            title="Will climate change accelerate next decade?",
            description="Question about environmental trends",
            question_type=QuestionType.BINARY,
            url="https://example.com/climate",
            close_time=datetime.now(timezone.utc) + timedelta(days=300),
            categories=["Environment"],
        )

        areas = research_service._identify_research_areas(climate_question)

        assert "climate data" in areas
        assert "environmental metrics" in areas

    @pytest.mark.asyncio
    async def test_gather_sources_for_area(self, research_service, sample_question):
        """Test gathering sources for a specific research area."""
        sources = await research_service._gather_sources_for_area(
            sample_question, "technology adoption", {}
        )

        assert isinstance(sources, list)
        assert len(sources) > 0

        source = sources[0]
        assert isinstance(source, ResearchSource)
        assert source.url is not None
        assert source.title is not None
        assert "technology adoption" in source.title.lower()
        assert source.credibility_score == 0.7  # Default score for mock implementation

    def test_validate_and_score_sources(self, research_service, sample_sources):
        """Test source validation and scoring."""
        validated_sources = research_service._validate_and_score_sources(sample_sources)

        # All sources should be included since _calculate_credibility_score updates the scores
        assert len(validated_sources) == 3

        for source in validated_sources:
            assert (
                source.credibility_score >= research_service.quality_thresholds["low"]
            )
            assert source.url is not None
            assert source.title is not None

    def test_validate_and_score_sources_empty_input(self, research_service):
        """Test source validation with empty input."""
        validated_sources = research_service._validate_and_score_sources([])

        assert validated_sources == []

    def test_validate_and_score_sources_invalid_sources(self, research_service):
        """Test source validation with invalid sources."""
        invalid_sources = [
            ResearchSource(
                url="", title="", summary="No URL or title", credibility_score=0.8
            ),
            ResearchSource(
                url="http://example.com",
                title="",
                summary="No title",
                credibility_score=0.8,
            ),
            ResearchSource(
                url="", title="Has title", summary="No URL", credibility_score=0.8
            ),
        ]

        validated_sources = research_service._validate_and_score_sources(
            invalid_sources
        )

        # All should be filtered out due to missing URL or title
        assert len(validated_sources) == 0

    def test_calculate_credibility_score(self, research_service):
        """Test credibility score calculation."""
        source = ResearchSource(
            url="https://example.com/academic",
            title="Academic Paper on AI",
            summary="Peer-reviewed research",
            credibility_score=0.0,  # Will be calculated
            source_type="academic",
        )

        score = research_service._calculate_credibility_score(source)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_extract_key_factors(
        self, research_service, sample_question, sample_sources
    ):
        """Test key factor extraction."""
        key_factors = research_service._extract_key_factors(
            sample_question, sample_sources
        )

        assert isinstance(key_factors, list)
        assert len(key_factors) > 0
        assert "Historical precedent" in key_factors
        assert "Current trends" in key_factors
        assert "Expert consensus" in key_factors

    def test_extract_base_rates(
        self, research_service, sample_question, sample_sources
    ):
        """Test base rate extraction."""
        base_rates = research_service._extract_base_rates(
            sample_question, sample_sources
        )

        assert isinstance(base_rates, dict)
        assert len(base_rates) > 0
        assert "historical_frequency" in base_rates
        assert "similar_events" in base_rates
        assert "expert_estimates" in base_rates

        for rate in base_rates.values():
            assert isinstance(rate, float)
            assert 0.0 <= rate <= 1.0

    def test_synthesize_research_findings(
        self, research_service, sample_question, sample_sources
    ):
        """Test research findings synthesis."""
        key_factors = ["Factor 1", "Factor 2"]
        base_rates = {"rate1": 0.3, "rate2": 0.7}

        synthesis = research_service._synthesize_research_findings(
            sample_question, sample_sources, key_factors, base_rates
        )

        assert isinstance(synthesis, dict)
        assert "executive_summary" in synthesis
        assert "detailed_analysis" in synthesis
        assert "confidence_level" in synthesis
        assert "methods_used" in synthesis
        assert "limitations" in synthesis

        assert isinstance(synthesis["executive_summary"], str)
        assert len(synthesis["executive_summary"]) > 0
        assert sample_question.title in synthesis["executive_summary"]

        assert isinstance(synthesis["confidence_level"], float)
        assert 0.0 <= synthesis["confidence_level"] <= 1.0

    def test_synthesize_research_findings_empty_base_rates(
        self, research_service, sample_question, sample_sources
    ):
        """Test research synthesis with empty base rates."""
        key_factors = ["Factor 1"]
        base_rates = {}

        # Should handle empty base_rates gracefully
        synthesis = research_service._synthesize_research_findings(
            sample_question, sample_sources, key_factors, base_rates
        )

        assert isinstance(synthesis, dict)
        assert "executive_summary" in synthesis

    def test_assess_research_quality_no_sources(self, research_service):
        """Test research quality assessment with no sources."""
        quality = research_service._assess_research_quality([], {})

        assert quality == ResearchQuality.LOW

    def test_assess_research_quality_high(self, research_service):
        """Test research quality assessment for high quality."""
        high_quality_sources = []
        for i in range(6):  # Need >= 5 sources for high quality
            source = ResearchSource(
                url=f"https://example.com/{i}",
                title=f"High Quality Source {i}",
                summary="High quality content",
                credibility_score=0.85,  # Above high threshold (0.8)
                source_type="academic",
            )
            high_quality_sources.append(source)

        quality = research_service._assess_research_quality(high_quality_sources, {})

        assert quality == ResearchQuality.HIGH

    def test_assess_research_quality_medium(self, research_service):
        """Test research quality assessment for medium quality."""
        medium_quality_sources = []
        for i in range(4):  # Need >= 3 sources with medium credibility
            source = ResearchSource(
                url=f"https://example.com/{i}",
                title=f"Medium Quality Source {i}",
                summary="Medium quality content",
                credibility_score=0.7,  # Above medium threshold (0.6)
                source_type="news",
            )
            medium_quality_sources.append(source)

        quality = research_service._assess_research_quality(medium_quality_sources, {})

        assert quality == ResearchQuality.MEDIUM

    def test_assess_research_quality_low(self, research_service):
        """Test research quality assessment for low quality."""
        low_quality_sources = [
            ResearchSource(
                url="https://example.com/low",
                title="Low Quality Source",
                summary="Low quality content",
                credibility_score=0.3,  # Below medium threshold
                source_type="blog",
            )
        ]

        quality = research_service._assess_research_quality(low_quality_sources, {})

        assert quality == ResearchQuality.LOW

    def test_determine_time_horizon(self, research_service, sample_question):
        """Test time horizon determination."""
        time_horizon = research_service._determine_time_horizon(sample_question)

        # Current implementation returns None - this is expected
        assert time_horizon is None

    def test_create_fallback_research_report(self, research_service, sample_question):
        """Test fallback research report creation."""
        error_message = "Search API unavailable"

        report = research_service._create_fallback_research_report(
            sample_question, error_message
        )

        assert isinstance(report, ResearchReport)
        assert report.question_id == sample_question.id
        assert error_message in report.executive_summary
        assert "Limited Research" in report.title
        assert report.quality == ResearchQuality.LOW
        assert report.confidence_level == 0.2
        assert len(report.sources) == 0
        assert "Limited information" in report.key_factors
        assert "fallback" in report.research_methodology

    @pytest.mark.asyncio
    async def test_conduct_comprehensive_research_success(
        self, research_service, sample_question
    ):
        """Test successful comprehensive research."""
        # Mock the _gather_sources_for_area method to return mock sources
        with patch.object(research_service, "_gather_sources_for_area") as mock_gather:
            mock_source = ResearchSource(
                url="https://example.com/research",
                title="Test Research Source",
                summary="Test summary",
                credibility_score=0.8,
                source_type="academic",
            )
            mock_gather.return_value = [mock_source]

            report = await research_service.conduct_comprehensive_research(
                sample_question
            )

            assert isinstance(report, ResearchReport)
            assert report.question_id == sample_question.id
            assert "Comprehensive Research" in report.title
            assert sample_question.title in report.title
            assert len(report.sources) > 0
            assert len(report.key_factors) > 0
            assert len(report.base_rates) > 0
            assert report.quality in ResearchQuality
            assert 0.0 <= report.confidence_level <= 1.0
            assert len(report.research_methodology) > 0

    @pytest.mark.asyncio
    async def test_conduct_comprehensive_research_with_config(
        self, research_service, sample_question
    ):
        """Test comprehensive research with custom configuration."""
        config = {"max_sources_per_area": 5, "min_credibility_threshold": 0.6}

        with patch.object(research_service, "_gather_sources_for_area") as mock_gather:
            mock_source = ResearchSource(
                url="https://example.com/research",
                title="Test Research Source",
                summary="Test summary",
                credibility_score=0.8,
                source_type="academic",
            )
            mock_gather.return_value = [mock_source]

            report = await research_service.conduct_comprehensive_research(
                sample_question, config
            )

            assert isinstance(report, ResearchReport)
            assert report.question_id == sample_question.id

    @pytest.mark.asyncio
    async def test_conduct_comprehensive_research_exception(
        self, research_service, sample_question
    ):
        """Test comprehensive research with exception handling."""
        # Mock _gather_sources_for_area to raise an exception
        with patch.object(research_service, "_gather_sources_for_area") as mock_gather:
            mock_gather.side_effect = Exception("API error")

            report = await research_service.conduct_comprehensive_research(
                sample_question
            )

            # Should return fallback report on exception
            assert isinstance(report, ResearchReport)
            assert report.question_id == sample_question.id
            assert "Limited Research" in report.title
            assert report.quality == ResearchQuality.LOW
            assert "API error" in report.executive_summary

    def test_validate_research_config_valid(self, research_service):
        """Test research configuration validation with valid config."""
        valid_config = {
            "max_sources_per_area": 10,
            "min_credibility_threshold": 0.5,
            "search_timeout": 30,
            "preferred_providers": ["duckduckgo", "exa"],
        }

        is_valid = research_service.validate_research_config(valid_config)

        assert is_valid is True

    def test_validate_research_config_empty(self, research_service):
        """Test research configuration validation with empty config."""
        is_valid = research_service.validate_research_config({})

        assert is_valid is True

    def test_validate_research_config_invalid_threshold(self, research_service):
        """Test research configuration validation with invalid threshold."""
        invalid_config = {"min_credibility_threshold": 1.5}  # Invalid: > 1.0

        is_valid = research_service.validate_research_config(invalid_config)

        assert is_valid is False

    def test_validate_research_config_invalid_max_sources(self, research_service):
        """Test research configuration validation with invalid max sources."""
        invalid_config = {"max_sources_per_area": 0}  # Invalid: < 1

        is_valid = research_service.validate_research_config(invalid_config)

        assert is_valid is False

    def test_validate_research_config_unknown_fields(self, research_service):
        """Test research configuration validation with unknown fields."""
        invalid_config = {"unknown_field": "value"}

        is_valid = research_service.validate_research_config(invalid_config)

        assert is_valid is False

    def test_get_supported_providers(self, research_service):
        """Test getting list of supported providers."""
        providers = research_service.get_supported_providers()

        assert isinstance(providers, list)
        assert "duckduckgo" in providers
        assert "exa" in providers
        assert "asknews" in providers
        assert "perplexity" in providers
        assert "manual" in providers

        # Should return a copy, not the original list
        providers.append("new_provider")
        original_providers = research_service.get_supported_providers()
        assert "new_provider" not in original_providers

    def test_get_quality_metrics(self, research_service, sample_question):
        """Test getting quality metrics for a research report."""
        sources = [
            ResearchSource(
                url="https://example.com/1",
                title="Source 1",
                summary="Summary 1",
                credibility_score=0.8,
                source_type="academic",
            ),
            ResearchSource(
                url="https://example.com/2",
                title="Source 2",
                summary="Summary 2",
                credibility_score=0.6,
                source_type="news",
            ),
        ]

        report = ResearchReport.create_new(
            question_id=sample_question.id,
            title="Test Report",
            executive_summary="Test summary",
            detailed_analysis="Test analysis",
            sources=sources,
            created_by="test",
            key_factors=["Factor 1", "Factor 2"],
            base_rates={"rate1": 0.3, "rate2": 0.7},
            quality=ResearchQuality.MEDIUM,
            confidence_level=0.75,
        )

        metrics = research_service.get_quality_metrics(report)

        assert isinstance(metrics, dict)
        assert metrics["source_count"] == 2
        assert metrics["average_credibility"] == 0.7  # (0.8 + 0.6) / 2
        assert metrics["quality_level"] == "medium"
        assert metrics["confidence_level"] == 0.75
        assert metrics["key_factors_count"] == 2
        assert metrics["base_rates_count"] == 2
        assert "has_reasoning_steps" in metrics
        assert "has_evidence" in metrics

    def test_get_quality_metrics_no_sources(self, research_service, sample_question):
        """Test getting quality metrics for report with no sources."""
        report = ResearchReport.create_new(
            question_id=sample_question.id,
            title="Test Report",
            executive_summary="Test summary",
            detailed_analysis="Test analysis",
            sources=[],
            created_by="test",
            key_factors=[],
            base_rates={},
            quality=ResearchQuality.LOW,
            confidence_level=0.3,
        )

        metrics = research_service.get_quality_metrics(report)

        assert metrics["source_count"] == 0
        assert metrics["average_credibility"] == 0.0
        assert metrics["quality_level"] == "low"
