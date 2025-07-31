"""
Knowledge Gap Detector for adaptive research and information quality assessment.

This service implements insufficient information detection, gap analysis, and adaptive
research strategies based on information quality. It provides research depth optimization
and source diversification to ensure comprehensive evidence gathering.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
from abc import ABC, abstractmethod
import statistics
from collections import defaultdict, Counter

from .authoritative_source_manager import AuthoritativeSource, SourceType, ExpertiseArea, KnowledgeBase
from .conflict_resolver import SynthesizedConclusion, UncertaintyLevel
from ..entities.research_report import ResearchSource, ResearchQuality
from ..entities.question import Question

logger = structlog.get_logger(__name__)


class GapType(Enum):
    """Types of knowledge gaps that can be detected."""
    INSUFFICIENT_SOURCES = "insufficient_sources"
    SOURCE_DIVERSITY_GAP = "source_diversity_gap"
    TEMPORAL_GAP = "temporal_gap"
    EXPERTISE_GAP = "expertise_gap"
    METHODOLOGICAL_GAP = "methodological_gap"
    CREDIBILITY_GAP = "credibility_gap"
    GEOGRAPHIC_GAP = "geographic_gap"
    PERSPECTIVE_GAP = "perspective_gap"
    QUANTITATIVE_DATA_GAP = "quantitative_data_gap"
    RECENT_DEVELOPMENTS_GAP = "recent_developments_gap"


class GapSeverity(Enum):
    """Severity levels for knowledge gaps."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchStrategy(Enum):
    """Adaptive research strategies based on gap analysis."""
    INTENSIVE_SEARCH = "intensive_search"
    DIVERSIFICATION_FOCUS = "diversification_focus"
    EXPERT_CONSULTATION = "expert_consultation"
    TEMPORAL_EXPANSION = "temporal_expansion"
    METHODOLOGICAL_TRIANGULATION = "methodological_triangulation"
    CONSERVATIVE_APPROACH = "conservative_approach"
    RAPID_ASSESSMENT = "rapid_assessment"


@dataclass
class KnowledgeGap:
    """Represents an identified knowledge gap with analysis and recommendations."""
    gap_id: str
    gap_type: GapType
    severity: GapSeverity
    description: str
    impact_on_forecast: float  # 0.0 to 1.0
    confidence_reduction: float  # How much this gap reduces confidence

    # Gap analysis details
    missing_elements: List[str] = field(default_factory=list)
    available_alternatives: List[str] = field(default_factory=list)
    research_suggestions: List[str] = field(default_factory=list)

    # Quantitative measures
    current_coverage: float = 0.0  # 0.0 to 1.0
    desired_coverage: float = 1.0
    gap_size: float = 0.0  # desired - current

    # Temporal aspects
    time_sensitivity: float = 0.5  # How time-sensitive is addressing this gap
    research_time_estimate: Optional[timedelta] = None

    # Context
    question_context: Optional[str] = None
    related_gaps: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.gap_id:
            self.gap_id = f"gap_{hash((self.gap_type.value, self.description))}"
        self.gap_size = self.desired_coverage - self.current_coverage


@dataclass
class ResearchQualityAssessment:
    """Assessment of current research quality and gaps."""
    overall_quality: ResearchQuality
    confidence_level: float
    completeness_score: float  # 0.0 to 1.0

    # Source analysis
    source_count: int
    source_diversity_score: float
    credibility_distribution: Dict[str, int]  # credibility ranges -> count
    temporal_coverage_score: float

    # Gap analysis
    identified_gaps: List[KnowledgeGap]
    critical_gaps: List[KnowledgeGap]
    addressable_gaps: List[KnowledgeGap]

    # Recommendations
    recommended_strategy: ResearchStrategy
    priority_actions: List[str]
    resource_allocation: Dict[str, float]

    # Uncertainty factors
    uncertainty_sources: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]


@dataclass
class AdaptiveResearchPlan:
    """Adaptive research plan based on gap analysis."""
    plan_id: str
    strategy: ResearchStrategy
    priority_gaps: List[KnowledgeGap]

    # Research actions
    search_expansions: List[Dict[str, Any]]
    expert_consultations: List[Dict[str, Any]]
    source_diversification: List[Dict[str, Any]]

    # Resource allocation
    time_allocation: Dict[str, timedelta]
    effort_distribution: Dict[str, float]

    # Success criteria
    target_improvements: Dict[str, float]
    minimum_thresholds: Dict[str, float]

    # Execution details
    estimated_duration: timedelta
    confidence_improvement_estimate: float
    cost_benefit_ratio: float


class GapDetectionStrategy(ABC):
    """Abstract base class for gap detection strategies."""

    @abstractmethod
    def detect_gaps(
        self,
        sources: List[AuthoritativeSource],
        question: Question,
        context: Dict[str, Any]
    ) -> List[KnowledgeGap]:
        """Detect knowledge gaps in the provided sources."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this detection strategy."""
        pass


class SourceDiversityDetector(GapDetectionStrategy):
    """Detects gaps in source diversity across multiple dimensions."""

    def __init__(self):
        self.minimum_thresholds = {
            "source_types": 3,
            "knowledge_bases": 2,
            "expertise_areas": 2,
            "publication_venues": 3,
            "geographic_regions": 2
        }

    def detect_gaps(
        self,
        sources: List[AuthoritativeSource],
        question: Question,
        context: Dict[str, Any]
    ) -> List[KnowledgeGap]:
        """Detect source diversity gaps."""
        gaps = []

        # Analyze source type diversity
        source_types = set()
        for source in sources:
            if hasattr(source, 'source_type'):
                source_types.add(source.source_type)

        if len(source_types) < self.minimum_thresholds["source_types"]:
            gap = KnowledgeGap(
                gap_id="",
                gap_type=GapType.SOURCE_DIVERSITY_GAP,
                severity=GapSeverity.HIGH if len(source_types) <= 1 else GapSeverity.MEDIUM,
                description=f"Limited source type diversity: only {len(source_types)} types found",
                impact_on_forecast=0.6,
                confidence_reduction=0.3,
                missing_elements=[f"Need {self.minimum_thresholds['source_types'] - len(source_types)} more source types"],
                research_suggestions=[
                    "Search academic databases",
                    "Consult expert opinions",
                    "Review government data",
                    "Check institutional reports"
                ],
                current_coverage=len(source_types) / self.minimum_thresholds["source_types"],
                desired_coverage=1.0,
                time_sensitivity=0.7
            )
            gaps.append(gap)

        # Analyze knowledge base diversity
        knowledge_bases = set()
        for source in sources:
            if hasattr(source, 'knowledge_base') and source.knowledge_base:
                knowledge_bases.add(source.knowledge_base)

        if len(knowledge_bases) < self.minimum_thresholds["knowledge_bases"]:
            gap = KnowledgeGap(
                gap_id="",
                gap_type=GapType.SOURCE_DIVERSITY_GAP,
                severity=GapSeverity.MEDIUM,
                description=f"Limited knowledge base diversity: only {len(knowledge_bases)} bases used",
                impact_on_forecast=0.4,
                confidence_reduction=0.2,
                missing_elements=[f"Need {self.minimum_thresholds['knowledge_bases'] - len(knowledge_bases)} more knowledge bases"],
                research_suggestions=[
                    "Search additional academic databases",
                    "Consult specialized repositories",
                    "Review expert networks"
                ],
                current_coverage=len(knowledge_bases) / self.minimum_thresholds["knowledge_bases"],
                desired_coverage=1.0,
                time_sensitivity=0.5
            )
            gaps.append(gap)

        # Analyze expertise area coverage
        expertise_areas = set()
        for source in sources:
            if hasattr(source, 'expert_profile') and source.expert_profile:
                expertise_areas.update(source.expert_profile.expertise_areas)

        if len(expertise_areas) < self.minimum_thresholds["expertise_areas"]:
            gap = KnowledgeGap(
                gap_id="",
                gap_type=GapType.EXPERTISE_GAP,
                severity=GapSeverity.HIGH,
                description=f"Limited expertise area coverage: only {len(expertise_areas)} areas represented",
                impact_on_forecast=0.7,
                confidence_reduction=0.4,
                missing_elements=[f"Need {self.minimum_thresholds['expertise_areas'] - len(expertise_areas)} more expertise areas"],
                research_suggestions=[
                    "Consult experts from different domains",
                    "Search interdisciplinary sources",
                    "Review cross-domain research"
                ],
                current_coverage=len(expertise_areas) / self.minimum_thresholds["expertise_areas"],
                desired_coverage=1.0,
                time_sensitivity=0.8
            )
            gaps.append(gap)

        return gaps

    def get_strategy_name(self) -> str:
        return "Source Diversity Detection"


class TemporalCoverageDetector(GapDetectionStrategy):
    """Detects gaps in temporal coverage of sources."""

    def __init__(self):
        self.recency_thresholds = {
            "very_recent": timedelta(days=30),
            "recent": timedelta(days=90),
            "current": timedelta(days=365),
            "historical": timedelta(days=1825)  # 5 years
        }
        self.minimum_recent_sources = 2

    def detect_gaps(
        self,
        sources: List[AuthoritativeSource],
        question: Question,
        context: Dict[str, Any]
    ) -> List[KnowledgeGap]:
        """Detect temporal coverage gaps."""
        gaps = []

        # Filter sources with publication dates
        dated_sources = [s for s in sources if s.publish_date]

        if not dated_sources:
            gap = KnowledgeGap(
                gap_id="",
                gap_type=GapType.TEMPORAL_GAP,
                severity=GapSeverity.CRITICAL,
                description="No sources have publication dates",
                impact_on_forecast=0.8,
                confidence_reduction=0.5,
                missing_elements=["Publication dates for all sources"],
                research_suggestions=[
                    "Find sources with clear publication dates",
                    "Verify source recency",
                    "Search for recent developments"
                ],
                current_coverage=0.0,
                desired_coverage=1.0,
                time_sensitivity=0.9
            )
            gaps.append(gap)
            return gaps

        now = datetime.utcnow()

        # Analyze recency distribution
        recency_counts = {
            "very_recent": 0,
            "recent": 0,
            "current": 0,
            "historical": 0,
            "outdated": 0
        }

        for source in dated_sources:
            age = now - source.publish_date

            if age <= self.recency_thresholds["very_recent"]:
                recency_counts["very_recent"] += 1
            elif age <= self.recency_thresholds["recent"]:
                recency_counts["recent"] += 1
            elif age <= self.recency_thresholds["current"]:
                recency_counts["current"] += 1
            elif age <= self.recency_thresholds["historical"]:
                recency_counts["historical"] += 1
            else:
                recency_counts["outdated"] += 1

        # Check for insufficient recent sources
        recent_sources = recency_counts["very_recent"] + recency_counts["recent"]
        if recent_sources < self.minimum_recent_sources:
            severity = GapSeverity.CRITICAL if recent_sources == 0 else GapSeverity.HIGH
            gap = KnowledgeGap(
                gap_id="",
                gap_type=GapType.RECENT_DEVELOPMENTS_GAP,
                severity=severity,
                description=f"Insufficient recent sources: only {recent_sources} found",
                impact_on_forecast=0.7,
                confidence_reduction=0.4,
                missing_elements=[f"Need {self.minimum_recent_sources - recent_sources} more recent sources"],
                research_suggestions=[
                    "Search for recent news and developments",
                    "Check latest academic publications",
                    "Consult current expert opinions",
                    "Review recent data releases"
                ],
                current_coverage=recent_sources / self.minimum_recent_sources,
                desired_coverage=1.0,
                time_sensitivity=0.9
            )
            gaps.append(gap)

        # Check for excessive reliance on outdated sources
        outdated_ratio = recency_counts["outdated"] / len(dated_sources)
        if outdated_ratio > 0.5:
            gap = KnowledgeGap(
                gap_id="",
                gap_type=GapType.TEMPORAL_GAP,
                severity=GapSeverity.MEDIUM,
                description=f"High proportion of outdated sources: {outdated_ratio:.1%}",
                impact_on_forecast=0.5,
                confidence_reduction=0.3,
                missing_elements=["More current sources"],
                research_suggestions=[
                    "Replace outdated sources with recent ones",
                    "Verify if outdated information is still relevant",
                    "Search for updated research"
                ],
                current_coverage=1.0 - outdated_ratio,
                desired_coverage=0.8,  # Allow some historical context
                time_sensitivity=0.6
            )
            gaps.append(gap)

        return gaps

    def get_strategy_name(self) -> str:
        return "Temporal Coverage Detection"


class CredibilityGapDetector(GapDetectionStrategy):
    """Detects gaps in source credibility and quality."""

    def __init__(self):
        self.credibility_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        self.minimum_high_credibility_sources = 2
        self.minimum_average_credibility = 0.6

    def detect_gaps(
        self,
        sources: List[AuthoritativeSource],
        question: Question,
        context: Dict[str, Any]
    ) -> List[KnowledgeGap]:
        """Detect credibility gaps in sources."""
        gaps = []

        if not sources:
            return gaps

        # Analyze credibility distribution
        credibility_scores = [s.credibility_score for s in sources]
        avg_credibility = statistics.mean(credibility_scores)

        high_credibility_count = sum(1 for score in credibility_scores
                                   if score >= self.credibility_thresholds["high"])
        low_credibility_count = sum(1 for score in credibility_scores
                                  if score < self.credibility_thresholds["low"])

        # Check for insufficient high-credibility sources
        if high_credibility_count < self.minimum_high_credibility_sources:
            gap = KnowledgeGap(
                gap_id="",
                gap_type=GapType.CREDIBILITY_GAP,
                severity=GapSeverity.HIGH,
                description=f"Insufficient high-credibility sources: only {high_credibility_count} found",
                impact_on_forecast=0.6,
                confidence_reduction=0.4,
                missing_elements=[f"Need {self.minimum_high_credibility_sources - high_credibility_count} more high-credibility sources"],
                research_suggestions=[
                    "Search academic databases",
                    "Consult peer-reviewed sources",
                    "Review government publications",
                    "Check institutional reports"
                ],
                current_coverage=high_credibility_count / self.minimum_high_credibility_sources,
                desired_coverage=1.0,
                time_sensitivity=0.7
            )
            gaps.append(gap)

        # Check for low average credibility
        if avg_credibility < self.minimum_average_credibility:
            gap = KnowledgeGap(
                gap_id="",
                gap_type=GapType.CREDIBILITY_GAP,
                severity=GapSeverity.MEDIUM,
                description=f"Low average credibility: {avg_credibility:.2f}",
                impact_on_forecast=0.5,
                confidence_reduction=0.3,
                missing_elements=["Higher quality sources"],
                research_suggestions=[
                    "Replace low-credibility sources",
                    "Verify source authenticity",
                    "Search authoritative databases"
                ],
                current_coverage=avg_credibility / self.minimum_average_credibility,
                desired_coverage=1.0,
                time_sensitivity=0.6
            )
            gaps.append(gap)

        # Check for excessive low-credibility sources
        low_credibility_ratio = low_credibility_count / len(sources)
        if low_credibility_ratio > 0.3:
            gap = KnowledgeGap(
                gap_id="",
                gap_type=GapType.CREDIBILITY_GAP,
                severity=GapSeverity.MEDIUM,
                description=f"High proportion of low-credibility sources: {low_credibility_ratio:.1%}",
                impact_on_forecast=0.4,
                confidence_reduction=0.2,
                missing_elements=["Replacement of low-credibility sources"],
                research_suggestions=[
                    "Filter out unreliable sources",
                    "Verify source credibility",
                    "Search for authoritative alternatives"
                ],
                current_coverage=1.0 - low_credibility_ratio,
                desired_coverage=0.8,
                time_sensitivity=0.5
            )
            gaps.append(gap)

        return gaps

    def get_strategy_name(self) -> str:
        return "Credibility Gap Detection"


class QuantitativeDataDetector(GapDetectionStrategy):
    """Detects gaps in quantitative data and empirical evidence."""

    def detect_gaps(
        self,
        sources: List[AuthoritativeSource],
        question: Question,
        context: Dict[str, Any]
    ) -> List[KnowledgeGap]:
        """Detect quantitative data gaps."""
        gaps = []

        # Check for quantitative indicators in source content
        quantitative_sources = 0
        qualitative_sources = 0

        quantitative_indicators = [
            "data", "statistics", "numbers", "percentage", "rate", "trend",
            "measurement", "survey", "study", "analysis", "research", "findings"
        ]

        for source in sources:
            content = (source.summary + " " + source.title).lower()
            has_quantitative = any(indicator in content for indicator in quantitative_indicators)

            if has_quantitative:
                quantitative_sources += 1
            else:
                qualitative_sources += 1

        # Check if we need more quantitative data
        total_sources = len(sources)
        if total_sources > 0:
            quantitative_ratio = quantitative_sources / total_sources

            if quantitative_ratio < 0.4:  # Less than 40% quantitative
                gap = KnowledgeGap(
                    gap_id="",
                    gap_type=GapType.QUANTITATIVE_DATA_GAP,
                    severity=GapSeverity.MEDIUM,
                    description=f"Insufficient quantitative data: only {quantitative_ratio:.1%} of sources",
                    impact_on_forecast=0.5,
                    confidence_reduction=0.3,
                    missing_elements=["Statistical data", "Empirical evidence", "Quantitative analysis"],
                    research_suggestions=[
                        "Search for statistical databases",
                        "Look for empirical studies",
                        "Find survey data",
                        "Check government statistics"
                    ],
                    current_coverage=quantitative_ratio,
                    desired_coverage=0.6,
                    time_sensitivity=0.6
                )
                gaps.append(gap)

        return gaps

    def get_strategy_name(self) -> str:
        return "Quantitative Data Detection"


class KnowledgeGapDetector:
    """
    Service for detecting knowledge gaps and implementing adaptive research strategies.

    Analyzes information quality, identifies gaps in coverage, and provides
    recommendations for improving research depth and source diversification.
    """

    def __init__(self):
        self.gap_detectors = {
            "source_diversity": SourceDiversityDetector(),
            "temporal_coverage": TemporalCoverageDetector(),
            "credibility_gaps": CredibilityGapDetector(),
            "quantitative_data": QuantitativeDataDetector()
        }

        # Configuration for gap analysis
        self.analysis_config = {
            "minimum_sources": 5,
            "minimum_credibility": 0.6,
            "maximum_gap_tolerance": 0.3,
            "critical_gap_threshold": 0.7
        }

        # Strategy selection weights
        self.strategy_weights = {
            GapType.INSUFFICIENT_SOURCES: {
                ResearchStrategy.INTENSIVE_SEARCH: 0.8,
                ResearchStrategy.DIVERSIFICATION_FOCUS: 0.6
            },
            GapType.SOURCE_DIVERSITY_GAP: {
                ResearchStrategy.DIVERSIFICATION_FOCUS: 0.9,
                ResearchStrategy.METHODOLOGICAL_TRIANGULATION: 0.7
            },
            GapType.EXPERTISE_GAP: {
                ResearchStrategy.EXPERT_CONSULTATION: 0.9,
                ResearchStrategy.DIVERSIFICATION_FOCUS: 0.6
            },
            GapType.TEMPORAL_GAP: {
                ResearchStrategy.TEMPORAL_EXPANSION: 0.8,
                ResearchStrategy.INTENSIVE_SEARCH: 0.6
            },
            GapType.CREDIBILITY_GAP: {
                ResearchStrategy.INTENSIVE_SEARCH: 0.7,
                ResearchStrategy.EXPERT_CONSULTATION: 0.8
            }
        }

    def detect_knowledge_gaps(
        self,
        sources: List[AuthoritativeSource],
        question: Question,
        context: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeGap]:
        """
        Detect knowledge gaps in the provided sources.

        Args:
            sources: List of sources to analyze
            question: The question being researched
            context: Optional context for gap detection

        Returns:
            List of identified knowledge gaps
        """
        logger.info(
            "Detecting knowledge gaps",
            source_count=len(sources),
            question_id=str(question.id)
        )

        context = context or {}
        all_gaps = []

        # Run all gap detection strategies
        for detector_name, detector in self.gap_detectors.items():
            try:
                gaps = detector.detect_gaps(sources, question, context)

                # Add detector context to gaps
                for gap in gaps:
                    gap.question_context = question.title
                    if not gap.gap_id:
                        gap.gap_id = f"{detector_name}_{hash((gap.gap_type.value, gap.description))}"

                all_gaps.extend(gaps)

                logger.info(
                    "Gap detection completed",
                    detector=detector_name,
                    gaps_found=len(gaps)
                )

            except Exception as e:
                logger.error(
                    "Gap detection failed",
                    detector=detector_name,
                    error=str(e)
                )

        # Remove duplicate gaps and prioritize
        unique_gaps = self._deduplicate_gaps(all_gaps)
        prioritized_gaps = self._prioritize_gaps(unique_gaps)

        logger.info(
            "Knowledge gap detection completed",
            total_gaps=len(all_gaps),
            unique_gaps=len(unique_gaps),
            prioritized_gaps=len(prioritized_gaps)
        )

        return prioritized_gaps

    def assess_research_quality(
        self,
        sources: List[AuthoritativeSource],
        question: Question,
        gaps: Optional[List[KnowledgeGap]] = None
    ) -> ResearchQualityAssessment:
        """
        Assess the overall quality of research and identify improvement areas.

        Args:
            sources: Sources to assess
            question: The question being researched
            gaps: Optional pre-identified gaps

        Returns:
            Comprehensive research quality assessment
        """
        logger.info(
            "Assessing research quality",
            source_count=len(sources),
            question_id=str(question.id)
        )

        # Detect gaps if not provided
        if gaps is None:
            gaps = self.detect_knowledge_gaps(sources, question)

        # Calculate basic metrics
        source_count = len(sources)
        avg_credibility = statistics.mean([s.credibility_score for s in sources]) if sources else 0.0

        # Assess source diversity
        source_types = set()
        knowledge_bases = set()
        for source in sources:
            if hasattr(source, 'source_type'):
                source_types.add(source.source_type)
            if hasattr(source, 'knowledge_base') and source.knowledge_base:
                knowledge_bases.add(source.knowledge_base)

        source_diversity_score = min(1.0, (len(source_types) + len(knowledge_bases)) / 8.0)

        # Assess temporal coverage
        dated_sources = [s for s in sources if s.publish_date]
        if dated_sources:
            now = datetime.utcnow()
            recent_sources = sum(1 for s in dated_sources
                               if (now - s.publish_date).days <= 365)
            temporal_coverage_score = min(1.0, recent_sources / max(1, len(dated_sources)))
        else:
            temporal_coverage_score = 0.0

        # Calculate completeness score
        completeness_factors = [
            min(1.0, source_count / self.analysis_config["minimum_sources"]),
            avg_credibility,
            source_diversity_score,
            temporal_coverage_score
        ]
        completeness_score = statistics.mean(completeness_factors)

        # Categorize gaps
        critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
        addressable_gaps = [g for g in gaps if g.time_sensitivity > 0.5]

        # Determine overall quality
        if completeness_score >= 0.8 and not critical_gaps:
            overall_quality = ResearchQuality.HIGH
        elif completeness_score >= 0.6 and len(critical_gaps) <= 1:
            overall_quality = ResearchQuality.MEDIUM
        else:
            overall_quality = ResearchQuality.LOW

        # Calculate confidence level
        gap_penalty = sum(g.confidence_reduction for g in gaps)
        confidence_level = max(0.1, avg_credibility - gap_penalty)

        # Credibility distribution
        credibility_distribution = {
            "high (0.8+)": sum(1 for s in sources if s.credibility_score >= 0.8),
            "medium (0.6-0.8)": sum(1 for s in sources if 0.6 <= s.credibility_score < 0.8),
            "low (<0.6)": sum(1 for s in sources if s.credibility_score < 0.6)
        }

        # Select recommended strategy
        recommended_strategy = self._select_research_strategy(gaps, completeness_score)

        # Generate priority actions
        priority_actions = self._generate_priority_actions(gaps, sources)

        # Resource allocation recommendations
        resource_allocation = self._calculate_resource_allocation(gaps)

        # Uncertainty sources
        uncertainty_sources = [
            f"Knowledge gaps: {len(gaps)}",
            f"Average credibility: {avg_credibility:.2f}",
            f"Source diversity: {source_diversity_score:.2f}"
        ]

        assessment = ResearchQualityAssessment(
            overall_quality=overall_quality,
            confidence_level=confidence_level,
            completeness_score=completeness_score,
            source_count=source_count,
            source_diversity_score=source_diversity_score,
            credibility_distribution=credibility_distribution,
            temporal_coverage_score=temporal_coverage_score,
            identified_gaps=gaps,
            critical_gaps=critical_gaps,
            addressable_gaps=addressable_gaps,
            recommended_strategy=recommended_strategy,
            priority_actions=priority_actions,
            resource_allocation=resource_allocation,
            uncertainty_sources=uncertainty_sources,
            confidence_intervals={
                "credibility": (max(0.0, avg_credibility - 0.1), min(1.0, avg_credibility + 0.1)),
                "completeness": (max(0.0, completeness_score - 0.1), min(1.0, completeness_score + 0.1))
            }
        )

        logger.info(
            "Research quality assessment completed",
            overall_quality=overall_quality.value,
            confidence_level=confidence_level,
            gaps_count=len(gaps),
            critical_gaps=len(critical_gaps)
        )

        return assessment

    def create_adaptive_research_plan(
        self,
        assessment: ResearchQualityAssessment,
        question: Question,
        constraints: Optional[Dict[str, Any]] = None
    ) -> AdaptiveResearchPlan:
        """
        Create an adaptive research plan based on quality assessment.

        Args:
            assessment: Research quality assessment
            question: The question being researched
            constraints: Optional constraints (time, resources, etc.)

        Returns:
            Adaptive research plan with specific actions
        """
        logger.info(
            "Creating adaptive research plan",
            strategy=assessment.recommended_strategy.value,
            gaps_count=len(assessment.identified_gaps)
        )

        constraints = constraints or {}

        # Select priority gaps to address
        priority_gaps = self._select_priority_gaps(
            assessment.identified_gaps,
            constraints.get("max_gaps", 5)
        )

        # Generate specific research actions
        search_expansions = self._generate_search_expansions(priority_gaps, question)
        expert_consultations = self._generate_expert_consultations(priority_gaps, question)
        source_diversification = self._generate_diversification_actions(priority_gaps, question)

        # Calculate time allocation
        total_time = constraints.get("total_time", timedelta(hours=8))
        time_allocation = self._allocate_time(priority_gaps, total_time)

        # Calculate effort distribution
        effort_distribution = self._calculate_effort_distribution(priority_gaps)

        # Set target improvements
        target_improvements = {
            "completeness_score": min(1.0, assessment.completeness_score + 0.2),
            "confidence_level": min(0.95, assessment.confidence_level + 0.15),
            "source_diversity": min(1.0, assessment.source_diversity_score + 0.3)
        }

        # Set minimum thresholds
        minimum_thresholds = {
            "completeness_score": 0.7,
            "confidence_level": 0.6,
            "critical_gaps_resolved": 0.8
        }

        # Estimate improvements
        confidence_improvement = sum(
            gap.confidence_reduction * 0.7  # Assume 70% gap resolution
            for gap in priority_gaps
        )

        # Calculate total allocated time in hours for cost-benefit ratio
        total_allocated_seconds = sum(td.total_seconds() for td in time_allocation.values())
        total_allocated_hours = total_allocated_seconds / 3600

        plan = AdaptiveResearchPlan(
            plan_id=f"plan_{question.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            strategy=assessment.recommended_strategy,
            priority_gaps=priority_gaps,
            search_expansions=search_expansions,
            expert_consultations=expert_consultations,
            source_diversification=source_diversification,
            time_allocation=time_allocation,
            effort_distribution=effort_distribution,
            target_improvements=target_improvements,
            minimum_thresholds=minimum_thresholds,
            estimated_duration=total_time,
            confidence_improvement_estimate=confidence_improvement,
            cost_benefit_ratio=confidence_improvement / max(0.1, total_allocated_hours)
        )

        logger.info(
            "Adaptive research plan created",
            plan_id=plan.plan_id,
            priority_gaps=len(priority_gaps),
            estimated_improvement=confidence_improvement
        )

        return plan

    def _deduplicate_gaps(self, gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """Remove duplicate gaps based on type and description similarity."""
        unique_gaps = []
        seen_combinations = set()

        for gap in gaps:
            # Create a key based on gap type and key words from description
            key_words = set(gap.description.lower().split()[:5])  # First 5 words
            combination_key = (gap.gap_type, frozenset(key_words))

            if combination_key not in seen_combinations:
                seen_combinations.add(combination_key)
                unique_gaps.append(gap)

        return unique_gaps

    def _prioritize_gaps(self, gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """Prioritize gaps based on severity, impact, and time sensitivity."""

        def gap_priority_score(gap: KnowledgeGap) -> float:
            severity_weights = {
                GapSeverity.CRITICAL: 1.0,
                GapSeverity.HIGH: 0.8,
                GapSeverity.MEDIUM: 0.6,
                GapSeverity.LOW: 0.4
            }

            severity_score = severity_weights.get(gap.severity, 0.5)
            impact_score = gap.impact_on_forecast
            time_score = gap.time_sensitivity

            return (severity_score * 0.4 + impact_score * 0.4 + time_score * 0.2)

        return sorted(gaps, key=gap_priority_score, reverse=True)

    def _select_research_strategy(
        self,
        gaps: List[KnowledgeGap],
        completeness_score: float
    ) -> ResearchStrategy:
        """Select the most appropriate research strategy based on gaps."""

        if not gaps:
            return ResearchStrategy.CONSERVATIVE_APPROACH

        # Count gap types
        gap_type_counts = Counter(gap.gap_type for gap in gaps)

        # Find the most common gap type
        most_common_gap_type = gap_type_counts.most_common(1)[0][0]

        # Select strategy based on most common gap type and completeness
        if completeness_score < 0.3:
            return ResearchStrategy.INTENSIVE_SEARCH
        elif most_common_gap_type in self.strategy_weights:
            strategies = self.strategy_weights[most_common_gap_type]
            return max(strategies.items(), key=lambda x: x[1])[0]
        else:
            return ResearchStrategy.DIVERSIFICATION_FOCUS

    def _generate_priority_actions(
        self,
        gaps: List[KnowledgeGap],
        sources: List[AuthoritativeSource]
    ) -> List[str]:
        """Generate priority actions based on identified gaps."""
        actions = []

        # Group gaps by type
        gap_types = Counter(gap.gap_type for gap in gaps)

        for gap_type, count in gap_types.most_common():
            if gap_type == GapType.INSUFFICIENT_SOURCES:
                actions.append(f"Expand search to find {count * 2} additional sources")
            elif gap_type == GapType.SOURCE_DIVERSITY_GAP:
                actions.append("Diversify source types (academic, expert, government)")
            elif gap_type == GapType.TEMPORAL_GAP:
                actions.append("Search for more recent sources and developments")
            elif gap_type == GapType.CREDIBILITY_GAP:
                actions.append("Replace low-credibility sources with authoritative ones")
            elif gap_type == GapType.EXPERTISE_GAP:
                actions.append("Consult experts from relevant domains")

        return actions[:5]  # Return top 5 actions

    def _calculate_resource_allocation(self, gaps: List[KnowledgeGap]) -> Dict[str, float]:
        """Calculate recommended resource allocation based on gaps."""
        allocation = {
            "search_expansion": 0.3,
            "source_verification": 0.2,
            "expert_consultation": 0.2,
            "analysis_synthesis": 0.2,
            "quality_assurance": 0.1
        }

        # Adjust based on gap types
        gap_types = Counter(gap.gap_type for gap in gaps)

        if GapType.INSUFFICIENT_SOURCES in gap_types:
            allocation["search_expansion"] += 0.2
            allocation["analysis_synthesis"] -= 0.1

        if GapType.CREDIBILITY_GAP in gap_types:
            allocation["source_verification"] += 0.15
            allocation["search_expansion"] -= 0.1

        if GapType.EXPERTISE_GAP in gap_types:
            allocation["expert_consultation"] += 0.15
            allocation["analysis_synthesis"] -= 0.1

        # Normalize to sum to 1.0
        total = sum(allocation.values())
        return {k: v / total for k, v in allocation.items()}

    def _select_priority_gaps(self, gaps: List[KnowledgeGap], max_gaps: int) -> List[KnowledgeGap]:
        """Select priority gaps to address based on impact and feasibility."""

        # Score gaps based on impact, severity, and addressability
        def gap_score(gap: KnowledgeGap) -> float:
            impact_score = gap.impact_on_forecast
            severity_weights = {
                GapSeverity.CRITICAL: 1.0,
                GapSeverity.HIGH: 0.8,
                GapSeverity.MEDIUM: 0.6,
                GapSeverity.LOW: 0.4
            }
            severity_score = severity_weights.get(gap.severity, 0.5)

            # Prefer gaps that are more addressable (higher time sensitivity)
            addressability_score = gap.time_sensitivity

            return impact_score * 0.4 + severity_score * 0.4 + addressability_score * 0.2

        scored_gaps = sorted(gaps, key=gap_score, reverse=True)
        return scored_gaps[:max_gaps]

    def _generate_search_expansions(
        self,
        gaps: List[KnowledgeGap],
        question: Question
    ) -> List[Dict[str, Any]]:
        """Generate specific search expansion actions."""
        expansions = []

        for gap in gaps:
            if gap.gap_type in [GapType.INSUFFICIENT_SOURCES, GapType.SOURCE_DIVERSITY_GAP]:
                expansion = {
                    "gap_id": gap.gap_id,
                    "action": "expand_search",
                    "target": gap.missing_elements,
                    "suggestions": gap.research_suggestions,
                    "priority": gap.severity.value
                }
                expansions.append(expansion)

        return expansions

    def _generate_expert_consultations(
        self,
        gaps: List[KnowledgeGap],
        question: Question
    ) -> List[Dict[str, Any]]:
        """Generate expert consultation recommendations."""
        consultations = []

        for gap in gaps:
            if gap.gap_type == GapType.EXPERTISE_GAP:
                consultation = {
                    "gap_id": gap.gap_id,
                    "action": "consult_expert",
                    "expertise_needed": gap.missing_elements,
                    "consultation_type": "domain_expert",
                    "priority": gap.severity.value
                }
                consultations.append(consultation)

        return consultations

    def _generate_diversification_actions(
        self,
        gaps: List[KnowledgeGap],
        question: Question
    ) -> List[Dict[str, Any]]:
        """Generate source diversification actions."""
        actions = []

        for gap in gaps:
            if gap.gap_type == GapType.SOURCE_DIVERSITY_GAP:
                action = {
                    "gap_id": gap.gap_id,
                    "action": "diversify_sources",
                    "target_types": gap.missing_elements,
                    "current_coverage": gap.current_coverage,
                    "target_coverage": gap.desired_coverage
                }
                actions.append(action)

        return actions

    def _allocate_time(
        self,
        gaps: List[KnowledgeGap],
        total_time: timedelta
    ) -> Dict[str, timedelta]:
        """Allocate time across different research activities."""

        # Base allocation
        allocation = {
            "gap_analysis": total_time * 0.1,
            "search_expansion": total_time * 0.4,
            "source_verification": total_time * 0.2,
            "expert_consultation": total_time * 0.2,
            "synthesis": total_time * 0.1
        }

        # Adjust based on gap severity
        critical_gaps = sum(1 for gap in gaps if gap.severity == GapSeverity.CRITICAL)
        if critical_gaps > 0:
            # Allocate more time to search and verification
            allocation["search_expansion"] += total_time * 0.1
            allocation["source_verification"] += total_time * 0.1
            allocation["synthesis"] -= total_time * 0.2

        return allocation

    def _calculate_effort_distribution(self, gaps: List[KnowledgeGap]) -> Dict[str, float]:
        """Calculate effort distribution across gap types."""

        if not gaps:
            return {"general_research": 1.0}

        gap_type_counts = Counter(gap.gap_type for gap in gaps)
        total_gaps = len(gaps)

        distribution = {}
        for gap_type, count in gap_type_counts.items():
            distribution[gap_type.value] = count / total_gaps

        return distribution
