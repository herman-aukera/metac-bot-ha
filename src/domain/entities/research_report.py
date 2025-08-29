"""Research report domain entity."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class ResearchQuality(Enum):
    """Quality levels for research reports."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ResearchSource:
    """A source used in research."""

    url: str
    title: str
    summary: str
    credibility_score: float
    publish_date: Optional[datetime] = None
    source_type: str = "web"


@dataclass
class ResearchReport:
    """
    Domain entity representing a research report for a question.

    Contains all the research findings, sources, and analysis
    that will be used to make a forecast.
    """

    id: UUID
    question_id: UUID
    title: str
    executive_summary: str
    detailed_analysis: str
    sources: List[ResearchSource]
    key_factors: List[str]
    base_rates: Dict[str, float]
    quality: ResearchQuality
    confidence_level: float
    research_methodology: str
    created_at: datetime
    created_by: str  # Agent or researcher identifier

    # Reasoning traces for transparency
    reasoning_steps: List[str]
    evidence_for: List[str]
    evidence_against: List[str]
    uncertainties: List[str]

    @classmethod
    def create_new(
        cls,
        question_id: UUID,
        title: str,
        executive_summary: str,
        detailed_analysis: str,
        sources: List[ResearchSource],
        created_by: str,
        **kwargs,
    ) -> "ResearchReport":
        """Factory method to create a new research report."""
        return cls(
            id=uuid4(),
            question_id=question_id,
            title=title,
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            sources=sources,
            key_factors=kwargs.get("key_factors", []),
            base_rates=kwargs.get("base_rates", {}),
            quality=kwargs.get("quality", ResearchQuality.MEDIUM),
            confidence_level=kwargs.get("confidence_level", 0.5),
            research_methodology=kwargs.get("research_methodology", ""),
            created_at=datetime.utcnow(),
            created_by=created_by,
            reasoning_steps=kwargs.get("reasoning_steps", []),
            evidence_for=kwargs.get("evidence_for", []),
            evidence_against=kwargs.get("evidence_against", []),
            uncertainties=kwargs.get("uncertainties", []),
        )

    def add_source(self, source: ResearchSource) -> None:
        """Add a new source to the research report."""
        self.sources.append(source)

    def calculate_overall_credibility(self) -> float:
        """Calculate the overall credibility score from all sources."""
        if not self.sources:
            return 0.0
        return sum(source.credibility_score for source in self.sources) / len(
            self.sources
        )

    def get_recent_sources(self, days: int = 30) -> List[ResearchSource]:
        """Get sources published within the last N days."""
        cutoff_date = datetime.utcnow() - datetime.timedelta(days=days)
        return [
            source
            for source in self.sources
            if source.publish_date and source.publish_date > cutoff_date
        ]
