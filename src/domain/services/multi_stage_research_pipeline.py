"""
Multi-Stage Research Pipeline with AskNews and GPT-5-Mini Synthesis.
Implements task 4.1 requirements with cost-optimized research strategy.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchStageResult:
    """Result from a research stage."""

    content: str
    sources_used: List[str]
    model_used: str
    cost_estimate: float
    quality_score: float
    stage_name: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ResearchQualityMetrics:
    """Quality metrics for research validation."""

    citation_count: int
    source_credibility_score: float
    recency_score: float
    coverage_completeness: float
    factual_accuracy_score: float
    overall_quality: float
    gaps_identified: List[str]


class MultiStageResearchPipeline:
    """
    Multi-stage research pipeline implementing task 4.1 requirements.

    Features:
    - AskNews API prioritization (free via METACULUSQ4)
    - GPT-5-mini synthesis with mandatory citations
    - 48-hour news focus
    - Free model fallbacks (gpt-oss-20b:free, kimi-k2:free)
    - Research quality validation and gap detection
    """

    def __init__(self, tri_model_router=None, tournament_asknews=None):
        """Initialize the multi-stage research pipeline."""
        self.tri_model_router = tri_model_router
        self.tournament_asknews = tournament_asknews
        self.logger = logging.getLogger(__name__)

        # Research configuration
        self.news_focus_hours = 48
        self.max_sources_per_stage = 10
        self.quality_threshold = 0.6

        # Stage configuration
        self.stages = [
            "asknews_research",
            "synthesis_analysis",
            "quality_validation",
            "gap_detection",
        ]

    async def execute_research_pipeline(
        self, question: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete multi-stage research pipeline.

        Args:
            question: Research question
            context: Additional context for research

        Returns:
            Complete research results with quality metrics
        """
        context = context or {}
        pipeline_start = datetime.now()

        self.logger.info(
            f"Starting multi-stage research pipeline for: {question[:100]}..."
        )

        results = {
            "question": question,
            "pipeline_start": pipeline_start,
            "stages": {},
            "final_research": "",
            "quality_metrics": None,
            "total_cost": 0.0,
            "success": False,
        }

        try:
            # Stage 1: AskNews Research (prioritized)
            asknews_result = await self._execute_asknews_research_stage(
                question, context
            )
            results["stages"]["asknews_research"] = asknews_result
            results["total_cost"] += asknews_result.cost_estimate

            # Stage 2: GPT-5-Mini Synthesis
            synthesis_result = await self._execute_synthesis_stage(
                question, asknews_result.content, context
            )
            results["stages"]["synthesis_analysis"] = synthesis_result
            results["total_cost"] += synthesis_result.cost_estimate

            # Stage 3: Quality Validation
            validation_result = await self._execute_quality_validation_stage(
                synthesis_result.content, context
            )
            results["stages"]["quality_validation"] = validation_result
            results["total_cost"] += validation_result.cost_estimate

            # Stage 4: Gap Detection
            gap_result = await self._execute_gap_detection_stage(
                question, synthesis_result.content, context
            )
            results["stages"]["gap_detection"] = gap_result
            results["total_cost"] += gap_result.cost_estimate

            # Compile final research
            results["final_research"] = synthesis_result.content
            results["quality_metrics"] = self._calculate_quality_metrics(
                results["stages"]
            )
            results["success"] = True

            execution_time = (datetime.now() - pipeline_start).total_seconds()
            self.logger.info(
                f"Research pipeline completed in {execution_time:.2f}s, cost: ${results['total_cost']:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Research pipeline failed: {e}")
            results["error"] = str(e)
            results["success"] = False

        return results

    async def _execute_asknews_research_stage(
        self, question: str, context: Dict[str, Any]
    ) -> ResearchStageResult:
        """
        Execute AskNews research stage with 48-hour focus.
        Prioritizes AskNews API (free via METACULUSQ4) with fallbacks.
        """
        stage_start = datetime.now()

        try:
            # AskNews only – all other external/news model fallbacks removed to stop 404 & noisy retries.
            if self.tournament_asknews:
                try:
                    research_content = await self.tournament_asknews.get_news_research(
                        question
                    )
                    if research_content and len(research_content.strip()) > 0:
                        execution_time = (datetime.now() - stage_start).total_seconds()
                        return ResearchStageResult(
                            content=research_content,
                            sources_used=["AskNews API"],
                            model_used="asknews",
                            cost_estimate=0.0,
                            quality_score=0.8,
                            stage_name="asknews_research",
                            execution_time=execution_time,
                            success=True,
                        )
                except Exception as e:
                    self.logger.warning(f"Tournament AskNews failed: {e}")

            # AskNews failed or unavailable: attempt multi-source search fallback with incremental backoff
            sources_used: List[str] = []
            aggregated_snippets: List[str] = []
            try:
                # NOTE: Avoid relative import past top-level (previous '...' caused runtime error).
                # Domain layer exceptional import of infrastructure search factory kept TEMPORARILY until refactor
                # (will be moved behind an injected adapter). This unblocks research fallback reliability.
                from src.infrastructure.config.settings import get_settings  # type: ignore  # noqa: E402
                from src.infrastructure.external_apis.search_client import (
                    create_search_client,
                )  # type: ignore  # noqa: E402

                settings = get_settings()
                search_client = create_search_client(settings)
                # Derive a concise subject-style query (avoid sending full question sentences to simple APIs)

                def _extract_subject(q: str) -> str:
                    base = q.strip().rstrip("?")
                    # If starts with 'Will ' or similar, drop auxiliary and keep entity/phrase up to first punctuation
                    lowers = base.lower()
                    for prefix in (
                        "will ",
                        "is ",
                        "are ",
                        "does ",
                        "do ",
                        "who ",
                        "what ",
                    ):
                        if lowers.startswith(prefix):
                            base = base[len(prefix) :]
                            break
                    # Keep first 8 words to stay concise
                    parts = base.split()
                    return " ".join(parts[:8])

                concise_query = _extract_subject(question)
                # Up to 2 attempts with exponential backoff (1s, 2s)
                for attempt in range(2):
                    try:
                        # Attempt with concise query first; if empty on first attempt, try original question second
                        query_to_use = concise_query if attempt == 0 else question
                        results = await search_client.search(
                            query_to_use, max_results=8
                        )
                        if results:
                            sources_used = list(
                                {r.get("source", "unknown") for r in results}
                            )
                            for r in results:
                                aggregated_snippets.append(
                                    f"- {r.get('title', '')} | {r.get('url', '')}\n  {r.get('snippet', '')[:240]}"
                                )
                            break
                    except Exception as se:  # pragma: no cover - defensive
                        self.logger.warning(
                            f"Search fallback attempt {attempt + 1} failed: {se}"
                        )
                    await asyncio.sleep(1 * (2**attempt))
            except Exception as se:  # pragma: no cover
                self.logger.warning(f"Multi-source search fallback init failed: {se}")

            if aggregated_snippets:
                snippet_block = "\n".join(aggregated_snippets)
                fallback_content = (
                    "Research synthesized from public multi-source search (AskNews unavailable).\n\n"
                    + snippet_block
                )
                execution_time = (datetime.now() - stage_start).total_seconds()
                return ResearchStageResult(
                    content=fallback_content,
                    sources_used=sources_used,
                    model_used="multi_source_search",
                    cost_estimate=0.0,
                    quality_score=0.55,  # moderate provisional quality
                    stage_name="asknews_research",
                    execution_time=execution_time,
                    success=True,
                    error_message="AskNews failed; used multi-source search",
                )

            # Still no research
            execution_time = (datetime.now() - stage_start).total_seconds()
            return ResearchStageResult(
                content="Research unavailable – AskNews failed and search fallback returned no results.",
                sources_used=[],
                model_used="none",
                cost_estimate=0.0,
                quality_score=0.1,
                stage_name="asknews_research",
                execution_time=execution_time,
                success=False,
                error_message="AskNews unavailable and no search results",
            )

        except Exception as e:
            execution_time = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"AskNews research stage failed: {e}")

            return ResearchStageResult(
                content="",
                sources_used=[],
                model_used="none",
                cost_estimate=0.0,
                quality_score=0.0,
                stage_name="asknews_research",
                execution_time=execution_time,
                success=False,
                error_message=str(e),
            )

    # Removed: free model & Perplexity fallbacks (deprecated). Keeping method slot intentionally empty
    # to avoid accidental external calls if referenced elsewhere.
    async def _execute_free_model_research_fallback(self, *args, **kwargs):  # type: ignore
        raise RuntimeError("Free model research fallback removed – use AskNews only.")

    async def _execute_synthesis_stage(
        self, question: str, research_content: str, context: Dict[str, Any]
    ) -> ResearchStageResult:
        """
        Execute synthesis stage using GPT-5-mini with mandatory citations.
        Implements structured output formatting as per requirements.
        """
        stage_start = datetime.now()

        if not self.tri_model_router:
            raise Exception("Tri-model router not available for synthesis")

        try:
            # Get GPT-5-mini model for synthesis
            mini_model = self.tri_model_router.models.get("mini")
            if not mini_model:
                raise Exception("GPT-5-mini model not available")

            # Import anti-slop prompts for synthesis
            from src.prompts.anti_slop_prompts import (
                anti_slop_prompts,
            )  # absolute import (domain -> permitted prompts)

            # Create synthesis prompt with mandatory citations
            synthesis_prompt = anti_slop_prompts.get_research_prompt(
                question_text=question, model_tier="mini"
            )

            # Add research content and specific synthesis instructions
            full_prompt = f"""{synthesis_prompt}

## RAW RESEARCH DATA TO SYNTHESIZE:
{research_content}

## SYNTHESIS REQUIREMENTS:
- MANDATORY: Every factual claim must include [Source: URL/Publication, Date] citation
- Focus on 48-hour news window for recent developments
- Use structured output with bullet points for clarity
- Acknowledge information gaps explicitly
- Provide source reliability assessment
- Maximum 300 words for efficiency

## OUTPUT FORMAT:
### Key Findings
• [Finding 1 with citation]
• [Finding 2 with citation]

### Recent Developments (48-hour focus)
• [Recent development with citation]

### Information Gaps
• [Gap 1]
• [Gap 2]

### Source Assessment
• [Source reliability notes]
"""

            # Execute synthesis with GPT-5-mini
            synthesis_result = await mini_model.invoke(full_prompt)

            execution_time = (datetime.now() - stage_start).total_seconds()

            # Estimate cost for GPT-5-mini (0.25 per million tokens)
            estimated_tokens = len(full_prompt.split()) + len(synthesis_result.split())
            cost_estimate = (estimated_tokens / 1_000_000) * 0.25

            return ResearchStageResult(
                content=synthesis_result,
                sources_used=["GPT-5-mini synthesis"],
                model_used="openai/gpt-5-mini",
                cost_estimate=cost_estimate,
                quality_score=0.8,  # High quality for GPT-5-mini synthesis
                stage_name="synthesis_analysis",
                execution_time=execution_time,
                success=True,
            )

        except Exception as e:
            execution_time = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"Synthesis stage failed: {e}")

            return ResearchStageResult(
                content=research_content,  # Fallback to original research
                sources_used=["fallback"],
                model_used="none",
                cost_estimate=0.0,
                quality_score=0.3,
                stage_name="synthesis_analysis",
                execution_time=execution_time,
                success=False,
                error_message=str(e),
            )

    async def _execute_quality_validation_stage(
        self, content: str, context: Dict[str, Any]
    ) -> ResearchStageResult:
        """
        Execute enhanced quality validation stage using ValidationStageService with GPT-5-nano.
        Implements task 4.2 requirements with comprehensive quality assurance.
        """
        stage_start = datetime.now()

        try:
            # Import and initialize the enhanced validation service
            from .validation_stage_service import ValidationStageService

            validation_service = ValidationStageService(self.tri_model_router)

            # Execute comprehensive validation
            validation_result = await validation_service.validate_content(
                content=content, task_type="research_synthesis", context=context
            )

            # Generate quality report
            quality_report = await validation_service.generate_quality_report(
                validation_result, content
            )

            execution_time = (datetime.now() - stage_start).total_seconds()

            return ResearchStageResult(
                content=quality_report,
                sources_used=["ValidationStageService with GPT-5-nano"],
                model_used="openai/gpt-5-nano",
                cost_estimate=validation_result.cost_estimate,
                quality_score=validation_result.quality_score,
                stage_name="quality_validation",
                execution_time=execution_time,
                success=validation_result.is_valid,
            )

        except Exception as e:
            execution_time = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"Enhanced quality validation stage failed: {e}")

            return ResearchStageResult(
                content=f"Enhanced validation unavailable: {str(e)}",
                sources_used=[],
                model_used="none",
                cost_estimate=0.0,
                quality_score=0.5,  # Neutral score when validation fails
                stage_name="quality_validation",
                execution_time=execution_time,
                success=False,
                error_message=str(e),
            )

    async def _execute_gap_detection_stage(
        self, question: str, content: str, context: Dict[str, Any]
    ) -> ResearchStageResult:
        """
        Execute gap detection stage to identify research gaps and limitations.
        Creates research quality validation and gap detection as per requirements.
        """
        stage_start = datetime.now()

        if not self.tri_model_router:
            raise Exception("Tri-model router not available for gap detection")

        try:
            # Get GPT-5-nano model for fast gap detection
            nano_model = self.tri_model_router.models.get("nano")
            if not nano_model:
                raise Exception("GPT-5-nano model not available")

            # Create gap detection prompt
            gap_prompt = f"""
Analyze the following research synthesis for gaps and limitations:

ORIGINAL QUESTION: {question}

RESEARCH SYNTHESIS:
{content}

## GAP DETECTION ANALYSIS:
Identify specific gaps in the research:

1. **Missing Information**: What key information is absent?
2. **Source Limitations**: Are there credibility or coverage issues?
3. **Temporal Gaps**: Is recent information (48-hour window) missing?
4. **Perspective Gaps**: Are important viewpoints missing?
5. **Data Gaps**: What quantitative data is missing?

## OUTPUT FORMAT:
### Critical Gaps Identified:
• [Gap 1 with impact assessment]
• [Gap 2 with impact assessment]

### Recommendations:
• [Recommendation 1]
• [Recommendation 2]

### Confidence Assessment:
Overall research confidence: LOW/MEDIUM/HIGH
Key uncertainty factors: [List factors]

Keep response concise and focused on actionable gaps.
"""

            # Execute gap detection with GPT-5-nano
            gap_result = await nano_model.invoke(gap_prompt)

            execution_time = (datetime.now() - stage_start).total_seconds()

            # Estimate cost for GPT-5-nano (0.05 per million tokens)
            estimated_tokens = len(gap_prompt.split()) + len(gap_result.split())
            cost_estimate = (estimated_tokens / 1_000_000) * 0.05

            return ResearchStageResult(
                content=gap_result,
                sources_used=["GPT-5-nano gap detection"],
                model_used="openai/gpt-5-nano",
                cost_estimate=cost_estimate,
                quality_score=0.7,
                stage_name="gap_detection",
                execution_time=execution_time,
                success=True,
            )

        except Exception as e:
            execution_time = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"Gap detection stage failed: {e}")

            return ResearchStageResult(
                content="Gap detection unavailable",
                sources_used=[],
                model_used="none",
                cost_estimate=0.0,
                quality_score=0.5,
                stage_name="gap_detection",
                execution_time=execution_time,
                success=False,
                error_message=str(e),
            )

    def _calculate_quality_metrics(
        self, stages: Dict[str, ResearchStageResult]
    ) -> ResearchQualityMetrics:
        """Calculate comprehensive quality metrics from all stages."""

        # Extract content for analysis
        synthesis_content = stages.get(
            "synthesis_analysis", ResearchStageResult("", [], "", 0, 0, "", 0, False)
        ).content
        validation_content = stages.get(
            "quality_validation", ResearchStageResult("", [], "", 0, 0, "", 0, False)
        ).content
        gap_content = stages.get(
            "gap_detection", ResearchStageResult("", [], "", 0, 0, "", 0, False)
        ).content

        # Calculate citation count
        citation_count = synthesis_content.count("[Source:") if synthesis_content else 0

        # Calculate source credibility score (based on successful stages)
        successful_stages = sum(1 for stage in stages.values() if stage.success)
        total_stages = len(stages)
        source_credibility_score = (
            successful_stages / total_stages if total_stages > 0 else 0.0
        )

        # Calculate recency score (based on 48-hour focus)
        recency_indicators = [
            "recent",
            "today",
            "yesterday",
            "48 hour",
            "latest",
            "current",
        ]
        recency_mentions = sum(
            synthesis_content.lower().count(indicator)
            for indicator in recency_indicators
        )
        recency_score = min(1.0, recency_mentions / 3.0)  # Normalize to 0-1

        # Calculate coverage completeness (based on validation results)
        coverage_score = 0.8 if "VALID" in validation_content.upper() else 0.5

        # Calculate factual accuracy score (based on validation and gap detection)
        accuracy_indicators = ["consistent", "accurate", "verified", "confirmed"]
        accuracy_mentions = sum(
            validation_content.lower().count(indicator)
            for indicator in accuracy_indicators
        )
        factual_accuracy_score = min(1.0, accuracy_mentions / 2.0)

        # Extract gaps from gap detection
        gaps_identified = []
        if gap_content and "Critical Gaps Identified:" in gap_content:
            gap_section = gap_content.split("Critical Gaps Identified:")[1].split(
                "### Recommendations:"
            )[0]
            gaps_identified = [
                line.strip("• ").strip()
                for line in gap_section.split("\n")
                if line.strip().startswith("•")
            ]

        # Calculate overall quality score
        overall_quality = (
            (citation_count > 0) * 0.2  # Citations present
            + source_credibility_score * 0.2  # Source reliability
            + recency_score * 0.2  # Recency focus
            + coverage_score * 0.2  # Coverage completeness
            + factual_accuracy_score * 0.2  # Factual accuracy
        )

        return ResearchQualityMetrics(
            citation_count=citation_count,
            source_credibility_score=source_credibility_score,
            recency_score=recency_score,
            coverage_completeness=coverage_score,
            factual_accuracy_score=factual_accuracy_score,
            overall_quality=overall_quality,
            gaps_identified=gaps_identified,
        )

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline configuration and status."""
        return {
            "stages": self.stages,
            "news_focus_hours": self.news_focus_hours,
            "max_sources_per_stage": self.max_sources_per_stage,
            "quality_threshold": self.quality_threshold,
            "asknews_available": bool(self.tournament_asknews),
            "tri_model_router_available": bool(self.tri_model_router),
            "free_models_configured": [],  # Purged to prevent 404 spam
        }
