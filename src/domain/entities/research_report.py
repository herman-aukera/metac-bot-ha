"""Research report entity for representing comprehensive research findings."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4


@dataclass
class Source:
    """Represents a research source with metadata."""
    url: str
    title: str
    content: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    domain: Optional[str] = None
    credibility_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate source data."""
        if not self.url or not self.url.strip():
            raise ValueError("Source URL cannot be empty")

        if not self.title or not self.title.strip():
            raise ValueError("Source title cannot be empty")

        if not self.content or not self.content.strip():
            raise ValueError("Source content cannot be empty")

        if not 0.0 <= self.credibility_score <= 1.0:
            raise ValueError(f"Credibility score must be between 0.0 and 1.0, got {self.credibility_score}")


@dataclass
class BaseRateData:
    """Historical patterns and reference class data."""
    reference_class: str
    historical_frequency: float
    sample_size: int
    time_period: str
    confidence_interval: tuple[float, float]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate base rate data."""
        if not self.reference_class or not self.reference_class.strip():
            raise ValueError("Reference class cannot be empty")

        if not 0.0 <= self.historical_frequency <= 1.0:
            raise ValueError(f"Historical frequency must be between 0.0 and 1.0, got {self.historical_frequency}")

        if self.sample_size <= 0:
            raise ValueError(f"Sample size must be positive, got {self.sample_size}")

        if not self.time_period or not self.time_period.strip():
            raise ValueError("Time period cannot be empty")

        if not isinstance(self.metadata, dict):
            object.__setattr__(self, 'metadata', {})


@dataclass
class ResearchReport:
    """Comprehensive research findings for a question.

    Attributes:
        id: Unique identifier for the research report
        question_id: ID of the question this research addresses
        sources: List of research sources with metadata
        credibility_scores: Credibility scores for each source
        evidence_synthesis: Synthesized evidence and conclusions
        base_rates: Historical patterns and reference class data
        knowledge_gaps: Identified gaps in available information
        research_quality_score: Overall quality score of research
        timestamp: When research was conducted
        research_method: Method used for research
        search_queries: Queries used during research
        metadata: Additional research-specific data
    """
    id: UUID
    question_id: int
    sources: List[Source]
    credibility_scores: Dict[str, float]
    evidence_synthesis: str
    base_rates: List[BaseRateData]
    knowledge_gaps: List[str]
    research_quality_score: float
    timestamp: datetime
    research_method: str
    search_queries: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate research report data and set defaults."""
        if self.id is None:
            object.__setattr__(self, 'id', uuid4())

        if self.question_id <= 0:
            raise ValueError(f"Question ID must be positive, got {self.question_id}")

        if not isinstance(self.sources, list):
            raise ValueError("Sources must be a list")

        if not isinstance(self.credibility_scores, dict):
            object.__setattr__(self, 'credibility_scores', {})

        if not self.evidence_synthesis or not self.evidence_synthesis.strip():
            raise ValueError("Evidence synthesis cannot be empty")

        if not isinstance(self.base_rates, list):
            object.__setattr__(self, 'base_rates', [])

        if not isinstance(self.knowledge_gaps, list):
            object.__setattr__(self, 'knowledge_gaps', [])

        if not 0.0 <= self.research_quality_score <= 1.0:
            raise ValueError(f"Research quality score must be between 0.0 and 1.0, got {self.research_quality_score}")

        if not self.research_method or not self.research_method.strip():
            raise ValueError("Research method cannot be empty")

        if not isinstance(self.search_queries, list):
            object.__setattr__(self, 'search_queries', [])

        if not isinstance(self.metadata, dict):
            object.__setattr__(self, 'metadata', {})

    @classmethod
    def create(
        cls,
        question_id: int,
        sources: List[Source],
        evidence_synthesis: str,
        research_method: str = "multi_provider_search",
        search_queries: Optional[List[str]] = None,
        base_rates: Optional[List[BaseRateData]] = None,
        knowledge_gaps: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> "ResearchReport":
        """Create a research report with automatic quality scoring.

        Args:
            question_id: ID of the question
            sources: List of research sources
            evidence_synthesis: Synthesized evidence
            research_method: Method used for research
            search_queries: Queries used during research
            base_rates: Historical patterns found
            knowledge_gaps: Identified knowledge gaps
            metadata: Additional metadata
            timestamp: Optional timestamp, defaults to now

        Returns:
            New ResearchReport instance
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Calculate credibility scores for sources
        credibility_scores = {}
        for source in sources:
            credibility_scores[source.url] = source.credibility_score

        # Calculate research quality score
        quality_score = cls._calculate_quality_score(sources, evidence_synthesis, base_rates or [])

        return cls(
            id=uuid4(),
            question_id=question_id,
            sources=sources,
            credibility_scores=credibility_scores,
            evidence_synthesis=evidence_synthesis,
            base_rates=base_rates or [],
            knowledge_gaps=knowledge_gaps or [],
            research_quality_score=quality_score,
            timestamp=timestamp,
            research_method=research_method,
            search_queries=search_queries or [],
            metadata=metadata or {}
        )

    @staticmethod
    def _calculate_quality_score(sources: List[Source], evidence_synthesis: str, base_rates: List[BaseRateData]) -> float:
        """Calculate overall research quality score."""
        if not sources:
            return 0.0

        # Source quality component (40%)
        avg_source_credibility = sum(source.credibility_score for source in sources) / len(sources)
        source_diversity = min(len(set(source.domain for source in sources if source.domain)) / 3.0, 1.0)
        source_score = (avg_source_credibility + source_diversity) / 2.0

        # Evidence synthesis quality component (40%)
        synthesis_length_score = min(len(evidence_synthesis) / 1000.0, 1.0)  # Normalize by expected length
        synthesis_score = synthesis_length_score  # Simplified - could add more sophisticated analysis

        # Base rate component (20%)
        base_rate_score = min(len(base_rates) / 2.0, 1.0)  # Normalize by expected number

        # Weighted combination
        quality_score = (source_score * 0.4 + synthesis_score * 0.4 + base_rate_score * 0.2)
        return min(quality_score, 1.0)

    def get_high_credibility_sources(self, threshold: float = 0.7) -> List[Source]:
        """Get sources with credibility above threshold."""
        return [source for source in self.sources if source.credibility_score >= threshold]

    def get_average_credibility(self) -> float:
        """Get average credibility score across all sources."""
        if not self.sources:
            return 0.0
        return sum(source.credibility_score for source in self.sources) / len(self.sources)

    def get_source_domains(self) -> List[str]:
        """Get unique domains from sources."""
        domains = set()
        for source in self.sources:
            if source.domain:
                domains.add(source.domain)
        return list(domains)

    def has_sufficient_sources(self, min_sources: int = 3) -> bool:
        """Check if research has sufficient number of sources."""
        return len(self.sources) >= min_sources

    def has_diverse_sources(self, min_domains: int = 2) -> bool:
        """Check if research has diverse source domains."""
        return len(self.get_source_domains()) >= min_domains

    def get_base_rate_summary(self) -> Dict[str, Any]:
        """Get summary of base rate data."""
        if not self.base_rates:
            return {"count": 0, "average_frequency": 0.0}

        avg_frequency = sum(br.historical_frequency for br in self.base_rates) / len(self.base_rates)
        total_sample_size = sum(br.sample_size for br in self.base_rates)

        return {
            "count": len(self.base_rates),
            "average_frequency": avg_frequency,
            "total_sample_size": total_sample_size,
            "reference_classes": [br.reference_class for br in self.base_rates]
        }

    def identify_critical_gaps(self) -> List[str]:
        """Identify critical knowledge gaps that could affect prediction quality."""
        critical_gaps = []

        # Check for insufficient sources
        if not self.has_sufficient_sources():
            critical_gaps.append("Insufficient number of sources")

        # Check for low credibility
        if self.get_average_credibility() < 0.5:
            critical_gaps.append("Low average source credibility")

        # Check for lack of diversity
        if not self.has_diverse_sources():
            critical_gaps.append("Lack of source diversity")

        # Check for missing base rates
        if not self.base_rates:
            critical_gaps.append("No historical base rate data")

        # Add explicit knowledge gaps
        critical_gaps.extend(self.knowledge_gaps)

        return critical_gaps

    def is_research_sufficient(self) -> bool:
        """Check if research is sufficient for making predictions."""
        return (
            self.has_sufficient_sources() and
            self.get_average_credibility() >= 0.5 and
            self.research_quality_score >= 0.6 and
            len(self.identify_critical_gaps()) <= 2
        )

    def to_summary(self) -> str:
        """Create a brief summary of the research report."""
        source_count = len(self.sources)
        avg_credibility = self.get_average_credibility()
        domain_count = len(self.get_source_domains())

        return (f"Research Q{self.question_id}: {source_count} sources, "
                f"{avg_credibility:.2f} avg credibility, {domain_count} domains, "
                f"quality: {self.research_quality_score:.2f}")

    def get_evidence_summary(self, max_length: int = 500) -> str:
        """Get truncated evidence synthesis for display."""
        if len(self.evidence_synthesis) <= max_length:
            return self.evidence_synthesis

        return self.evidence_synthesis[:max_length-3] + "..."
