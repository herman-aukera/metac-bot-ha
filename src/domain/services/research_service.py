"""Research service for coordinating information gathering activities."""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timezone
import structlog

from ..entities.question import Question
from ..entities.research_report import ResearchReport, ResearchSource, ResearchQuality
from ..value_objects.confidence import ConfidenceLevel
from ..value_objects.time_range import TimeRange


logger = structlog.get_logger(__name__)


class ResearchService:
    """
    Domain service for coordinating research activities.
    
    Handles the orchestration of information gathering from multiple sources,
    quality assessment of research materials, and synthesis of findings
    into structured research reports.
    """
    
    def __init__(self):
        self.supported_providers = [
            "asknews",
            "exa", 
            "perplexity",
            "duckduckgo",
            "manual"
        ]
        self.quality_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
    
    async def conduct_comprehensive_research(
        self,
        question: Question,
        research_config: Optional[Dict[str, Any]] = None
    ) -> ResearchReport:
        """
        Conduct comprehensive research for a question using multiple approaches.
        
        Args:
            question: The question to research
            research_config: Configuration for research behavior
            
        Returns:
            Comprehensive research report with findings from multiple sources
        """
        logger.info(
            "Starting comprehensive research",
            question_id=str(question.id),
            title=question.title
        )
        
        config = research_config or {}
        
        # Initialize research components
        all_sources = []
        key_factors = []
        base_rates = {}
        
        try:
            # Step 1: Break down the research question into key areas
            research_areas = self._identify_research_areas(question)
            logger.info("Identified research areas", areas=research_areas)
            
            # Step 2: Gather information from multiple sources
            for area in research_areas:
                sources = await self._gather_sources_for_area(
                    question, area, config
                )
                all_sources.extend(sources)
            
            # Step 3: Analyze and validate sources
            validated_sources = self._validate_and_score_sources(all_sources)
            
            # Step 4: Extract key factors and base rates
            key_factors = self._extract_key_factors(question, validated_sources)
            base_rates = self._extract_base_rates(question, validated_sources)
            
            # Step 5: Synthesize findings
            synthesis = self._synthesize_research_findings(
                question, validated_sources, key_factors, base_rates
            )
            
            # Step 6: Determine research quality
            quality = self._assess_research_quality(validated_sources, synthesis)
            
            # Create research report
            research_report = ResearchReport.create_new(
                question_id=question.id,
                title=f"Comprehensive Research: {question.title}",
                executive_summary=synthesis.get("executive_summary", ""),
                detailed_analysis=synthesis.get("detailed_analysis", ""),
                sources=validated_sources,
                created_by="ResearchService",
                key_factors=key_factors,
                base_rates=base_rates,
                quality=quality,
                confidence_level=synthesis.get("confidence_level", 0.7),
                research_methodology=", ".join(synthesis.get("methods_used", ["web_search", "source_validation", "synthesis"])),
                reasoning_steps=synthesis.get("reasoning_steps", []),
                evidence_for=synthesis.get("evidence_for", []),
                evidence_against=synthesis.get("evidence_against", []),
                uncertainties=synthesis.get("uncertainties", [])
            )
            
            logger.info(
                "Research completed successfully",
                sources_count=len(validated_sources),
                quality=quality.value,
                confidence=research_report.confidence_level
            )
            
            return research_report
            
        except Exception as e:
            logger.error("Research failed", error=str(e))
            # Return a minimal research report to avoid breaking the pipeline
            return self._create_fallback_research_report(question, str(e))
    
    def _identify_research_areas(self, question: Question) -> List[str]:
        """
        Identify key research areas based on the question.
        
        Args:
            question: The question to analyze
            
        Returns:
            List of research areas to investigate
        """
        # Basic research areas that apply to most forecasting questions
        base_areas = [
            "historical trends",
            "current status", 
            "expert opinions",
            "market indicators",
            "policy implications"
        ]
        
        # Add question-specific areas based on content
        question_text = question.title.lower()
        specific_areas = []
        
        # Technology questions
        if any(term in question_text for term in ["ai", "technology", "software", "tech"]):
            specific_areas.extend(["technology adoption", "innovation metrics"])
        
        # Economic questions  
        if any(term in question_text for term in ["economy", "gdp", "market", "finance"]):
            specific_areas.extend(["economic indicators", "financial metrics"])
            
        # Political questions
        if any(term in question_text for term in ["election", "policy", "government", "political"]):
            specific_areas.extend(["polling data", "political analysis"])
            
        # Health/medical questions
        if any(term in question_text for term in ["health", "medical", "disease", "pandemic"]):
            specific_areas.extend(["medical research", "health statistics"])
            
        # Climate/environmental questions
        if any(term in question_text for term in ["climate", "environment", "energy", "carbon"]):
            specific_areas.extend(["climate data", "environmental metrics"])
        
        return base_areas + specific_areas
    
    async def _gather_sources_for_area(
        self, 
        question: Question, 
        research_area: str,
        config: Dict[str, Any]
    ) -> List[ResearchSource]:
        """
        Gather sources for a specific research area.
        
        This is a placeholder implementation that would integrate with
        external search clients in a real implementation.
        """
        sources = []
        
        # Create mock sources for now - in real implementation,
        # this would call actual search APIs
        mock_source = ResearchSource(
            url=f"https://example.com/research/{research_area.replace(' ', '-')}",
            title=f"Research on {research_area} for {question.title}",
            summary=f"Information about {research_area} relevant to the question.",
            credibility_score=0.7,
            publish_date=datetime.now(),
            source_type="web"
        )
        sources.append(mock_source)
        
        return sources
    
    def _validate_and_score_sources(self, sources: List[ResearchSource]) -> List[ResearchSource]:
        """
        Validate and score the credibility of research sources.
        
        Args:
            sources: Raw sources to validate
            
        Returns:
            Validated and scored sources
        """
        validated_sources = []
        
        for source in sources:
            # Basic validation
            if not source.url or not source.title:
                continue
                
            # Score credibility based on various factors
            credibility_score = self._calculate_credibility_score(source)
            
            # Update source with calculated score
            validated_source = ResearchSource(
                url=source.url,
                title=source.title,
                summary=source.summary,
                credibility_score=credibility_score,
                publish_date=source.publish_date,
                source_type=source.source_type
            )
            
            # Only include sources above minimum threshold
            if credibility_score >= self.quality_thresholds["low"]:
                validated_sources.append(validated_source)
        
        return validated_sources
        
    def _calculate_credibility_score(self, source: ResearchSource) -> float:
        """Calculate credibility score for a source."""
        score = 0.5  # Base score

        # Domain credibility
        domain = source.url.split("//")[-1].split("/")[0] if source.url else ""

        high_credibility_domains = [
            "arxiv.org", "nature.com", "science.org", "pubmed.gov",
            "census.gov", "worldbank.org", "imf.org", "oecd.org",
            "who.int", "cdc.gov", "fda.gov"
        ]

        medium_credibility_domains = [
            "reuters.com", "bbc.com", "economist.com", "ft.com",
            "wsj.com", "nytimes.com", "bloomberg.com"
        ]

        if any(d in domain for d in high_credibility_domains):
            score += 0.3
        elif any(d in domain for d in medium_credibility_domains):
            score += 0.2

        # Recency bonus
        if source.publish_date:
            days_old = (datetime.now(timezone.utc) - source.publish_date).days
            if days_old < 30:
                score += 0.1
            elif days_old < 90:
                score += 0.05
        
        return min(1.0, score)
    
    def _extract_key_factors(
        self, 
        question: Question, 
        sources: List[ResearchSource]
    ) -> List[str]:
        """Extract key factors that could influence the forecast."""
        # In a real implementation, this would use NLP to extract
        # key factors from the source content
        
        base_factors = [
            "Historical precedent",
            "Current trends", 
            "Expert consensus",
            "Market dynamics",
            "Regulatory environment"
        ]
        
        return base_factors
    
    def _extract_base_rates(
        self, 
        question: Question, 
        sources: List[ResearchSource]
    ) -> Dict[str, float]:
        """Extract relevant base rates from research sources."""
        # In a real implementation, this would analyze sources
        # to find relevant historical frequencies
        
        return {
            "historical_frequency": 0.3,
            "similar_events": 0.25,
            "expert_estimates": 0.4
        }
    
    def _synthesize_research_findings(
        self,
        question: Question,
        sources: List[ResearchSource], 
        key_factors: List[str],
        base_rates: Dict[str, float]
    ) -> Dict[str, Any]:
        """Synthesize research findings into coherent analysis."""
        
        # Calculate overall confidence based on source quality
        avg_credibility = sum(s.credibility_score for s in sources) / len(sources) if sources else 0.5
        
        executive_summary = (
            f"Research conducted on '{question.title}' identified {len(key_factors)} "
            f"key factors from {len(sources)} sources. Average source credibility: "
            f"{avg_credibility:.2f}. Key considerations include historical precedent, "
            f"current trends, and expert opinions."
        )
        
        detailed_analysis = (
            f"Comprehensive analysis of available information reveals several important "
            f"considerations for forecasting '{question.title}'. "
            f"The research identified {len(key_factors)} primary factors that could "
            f"influence the outcome. "
        )
        
        # Add base rates information only if we have base rates
        if base_rates:
            detailed_analysis += (
                f"Historical base rates suggest relevant frequencies "
                f"ranging from {min(base_rates.values()):.2f} to {max(base_rates.values()):.2f}. "
            )
        else:
            detailed_analysis += "No historical base rates were identified. "
            
        detailed_analysis += (
            f"Source quality is generally {'high' if avg_credibility > 0.7 else 'medium' if avg_credibility > 0.5 else 'low'} "
            f"with an average credibility score of {avg_credibility:.2f}."
        )
        
        return {
            "executive_summary": executive_summary,
            "detailed_analysis": detailed_analysis,
            "confidence_level": min(avg_credibility + 0.1, 0.9),
            "methods_used": ["web_search", "source_validation", "synthesis"],
            "limitations": [
                "Limited to available online sources",
                "Automated synthesis may miss nuanced insights",
                "Source credibility assessment is algorithmic"
            ]
        }
    
    def _assess_research_quality(
        self, 
        sources: List[ResearchSource], 
        synthesis: Dict[str, Any]
    ) -> ResearchQuality:
        """Assess the overall quality of the research."""
        
        if not sources:
            return ResearchQuality.LOW
        
        avg_credibility = sum(s.credibility_score for s in sources) / len(sources)
        source_count = len(sources)
        
        # High quality: many high-credibility sources
        if avg_credibility >= self.quality_thresholds["high"] and source_count >= 5:
            return ResearchQuality.HIGH
        
        # Medium quality: decent sources or good sources but fewer
        elif (avg_credibility >= self.quality_thresholds["medium"] and source_count >= 3) or \
             (avg_credibility >= self.quality_thresholds["high"] and source_count >= 2):
            return ResearchQuality.MEDIUM
        
        # Low quality: few sources or low credibility
        else:
            return ResearchQuality.LOW
            
    def _determine_time_horizon(self, question: Question) -> Optional[TimeRange]:
        """Determine the time horizon for the research based on question."""
        # In a real implementation, this would parse the question
        # to extract timeline information

        # For now, return None since ResearchReport doesn't store time_horizon
        return None
    
    def _create_fallback_research_report(
        self, 
        question: Question, 
        error_message: str
    ) -> ResearchReport:
        """Create a minimal research report when research fails."""
        
        return ResearchReport.create_new(
            question_id=question.id,
            title=f"Limited Research: {question.title}",
            executive_summary=f"Research failed due to: {error_message}. Limited information available.",
            detailed_analysis="Unable to conduct comprehensive research. Forecast will be based on minimal information.",
            sources=[],
            created_by="ResearchService",
            key_factors=["Limited information"],
            base_rates={},
            quality=ResearchQuality.LOW,
            confidence_level=0.2,
            research_methodology="fallback",
            reasoning_steps=["Research failure", "No external sources", "Minimal analysis"],
            evidence_for=[],
            evidence_against=[],
            uncertainties=["Research failure", "No external sources", "Minimal analysis"]
        )
    
    def validate_research_config(self, config: Dict[str, Any]) -> bool:
        """Validate research configuration parameters."""
        
        required_fields = []  # No required fields for basic operation
        optional_fields = [
            "max_sources_per_area",
            "min_credibility_threshold", 
            "search_timeout",
            "preferred_providers",
            "date_range"
        ]
        
        # Check for unknown fields
        unknown_fields = set(config.keys()) - set(required_fields + optional_fields)
        if unknown_fields:
            logger.warning("Unknown config fields", fields=list(unknown_fields))
            return False
        
        # Validate specific field values
        if "min_credibility_threshold" in config:
            threshold = config["min_credibility_threshold"]
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                return False
        
        if "max_sources_per_area" in config:
            max_sources = config["max_sources_per_area"]
            if not isinstance(max_sources, int) or max_sources < 1:
                return False
        
        return True
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported research providers."""
        return self.supported_providers.copy()
        
    def get_quality_metrics(self, research_report: ResearchReport) -> Dict[str, Any]:
        """Get quality metrics for a research report."""

        source_count = len(research_report.sources)
        avg_credibility = (
            sum(s.credibility_score for s in research_report.sources) / source_count
            if source_count > 0 else 0.0
        )

        return {
            "source_count": source_count,
            "average_credibility": avg_credibility,
            "quality_level": research_report.quality.value,
            "confidence_level": research_report.confidence_level,
            "key_factors_count": len(research_report.key_factors),
            "base_rates_count": len(research_report.base_rates),
            "has_reasoning_steps": len(research_report.reasoning_steps) > 0,
            "has_evidence": len(research_report.evidence_for) > 0 or len(research_report.evidence_against) > 0
        }