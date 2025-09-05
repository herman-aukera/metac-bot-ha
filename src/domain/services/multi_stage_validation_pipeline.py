"""
Multi-Stage Validation Pipeline Implementation.
Integrates research, validation, and forecasting stages for complete question processing.
Implements task 4 requirements with comprehensive quality assurance.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .forecasting_stage_service import ForecastingStageService, ForecastResult
from .multi_stage_research_pipeline import (
    MultiStageResearchPipeline,
    ResearchStageResult,
)
from .validation_stage_service import ValidationResult, ValidationStageService

logger = logging.getLogger(__name__)


@dataclass
class MultiStageResult:
    """Complete result from multi-stage validation pipeline."""

    question: str
    question_type: str
    research_result: ResearchStageResult
    validation_result: ValidationResult
    forecast_result: ForecastResult
    pipeline_success: bool
    total_execution_time: float
    total_cost: float
    quality_score: float
    tournament_compliant: bool
    final_forecast: Any
    reasoning: str


class MultiStageValidationPipeline:
    """
    Complete multi-stage validation pipeline integrating all three stages.

    Implements the full task 4 requirements:
    - Stage 1: Research with AskNews and GPT-5-mini synthesis (task 4.1)
    - Stage 2: Validation with GPT-5-nano quality assurance (task 4.2)
    - Stage 3: Forecasting with GPT-5 and calibration (task 4.3)

    Features:
    - Cost-optimized research strategy with AskNews (free via METACULUSQ4)
    - Evidence traceability verification and hallucination detection
    - GPT-5 maximum reasoning capability with calibration
    - Tournament compliance validation
    - Comprehensive quality assurance
    """

    def __init__(self, tri_model_router=None, tournament_asknews=None):
        """Initialize the multi-stage validation pipeline."""
        self.tri_model_router = tri_model_router
        self.tournament_asknews = tournament_asknews
        self.logger = logging.getLogger(__name__)

        # Initialize stage services
        self.research_pipeline = MultiStageResearchPipeline(
            tri_model_router=tri_model_router, tournament_asknews=tournament_asknews
        )

        self.validation_service = ValidationStageService(
            tri_model_router=tri_model_router
        )

        self.forecasting_service = ForecastingStageService(
            tri_model_router=tri_model_router
        )

        # Pipeline configuration
        self.quality_threshold = 0.6
        self.cost_budget_per_question = 0.02  # $0.02 per question target

    async def process_question(
        self, question: str, question_type: str, context: Dict[str, Any] = None
    ) -> MultiStageResult:
        """
        Process a complete question through all three validation stages.

        Args:
            question: The forecasting question
            question_type: Type of forecast ("binary", "multiple_choice", "numeric")
            context: Additional context including options, bounds, etc.

        Returns:
            MultiStageResult with complete processing results
        """
        context = context or {}
        pipeline_start = datetime.now()

        self.logger.info(
            f"Starting multi-stage validation pipeline for {question_type} question..."
        )

        try:
            # Stage 1: Research with AskNews and GPT-5-mini synthesis
            self.logger.info("Executing Stage 1: Research and synthesis...")
            research_results = await self.research_pipeline.execute_research_pipeline(
                question=question, context=context
            )

            if not research_results["success"]:
                self.logger.warning(
                    "Research stage failed, proceeding with limited data"
                )

            research_content = research_results.get("final_research", "")

            # Stage 2: Validation with GPT-5-nano quality assurance
            self.logger.info("Executing Stage 2: Quality validation...")
            validation_result = await self.validation_service.validate_content(
                content=research_content,
                task_type="research_synthesis",
                context=context,
            )

            # Stage 3: Forecasting with GPT-5 and calibration
            # If this pipeline is invoked for research-only processing, skip forecasting gracefully.
            if question_type == "research":
                self.logger.info("Skipping forecasting stage (research-only mode)")
                forecast_result = ForecastResult(
                    forecast_type="binary",
                    prediction=0.5,
                    confidence_score=0.0,
                    uncertainty_bounds=None,
                    calibration_score=0.0,
                    overconfidence_detected=False,
                    quality_validation_passed=True,
                    tournament_compliant=True,
                    reasoning="Forecasting skipped in research-only mode.",
                    execution_time=0.0,
                    cost_estimate=0.0,
                    model_used="none",
                )
            else:
                self.logger.info("Executing Stage 3: GPT-5 forecasting with calibration...")
                forecast_result = await self.forecasting_service.generate_forecast(
                    question=question,
                    question_type=question_type,
                    research_data=research_content,
                    context=context,
                )

            # Calculate overall metrics
            total_execution_time = (datetime.now() - pipeline_start).total_seconds()
            total_cost = (
                research_results.get("total_cost", 0.0)
                + validation_result.cost_estimate
                + forecast_result.cost_estimate
            )

            # Calculate overall quality score
            quality_score = self._calculate_overall_quality_score(
                research_results, validation_result, forecast_result
            )

            # Check tournament compliance
            tournament_compliant = self._check_overall_tournament_compliance(
                validation_result, forecast_result
            )

            # Determine pipeline success
            pipeline_success = (
                research_results.get("success", False)
                and validation_result.is_valid
                and forecast_result.quality_validation_passed
                and quality_score >= self.quality_threshold
            )

            # Create research stage result for compatibility
            research_quality_score = 0.5  # Default
            if research_results.get("quality_metrics"):
                quality_metrics = research_results["quality_metrics"]
                if hasattr(quality_metrics, "overall_quality"):
                    research_quality_score = quality_metrics.overall_quality
                elif isinstance(quality_metrics, dict):
                    research_quality_score = quality_metrics.get("overall_quality", 0.5)

            research_stage_result = ResearchStageResult(
                content=research_content,
                sources_used=["Multi-stage research pipeline"],
                model_used="multi-stage",
                cost_estimate=research_results.get("total_cost", 0.0),
                quality_score=research_quality_score,
                stage_name="multi_stage_research",
                execution_time=(
                    research_results.get("pipeline_start", datetime.now()).timestamp()
                    if research_results.get("pipeline_start")
                    else 0.0
                ),
                success=research_results.get("success", False),
            )

            # Compile final reasoning
            final_reasoning = self._compile_final_reasoning(
                research_content, validation_result, forecast_result
            )

            result = MultiStageResult(
                question=question,
                question_type=question_type,
                research_result=research_stage_result,
                validation_result=validation_result,
                forecast_result=forecast_result,
                pipeline_success=pipeline_success,
                total_execution_time=total_execution_time,
                total_cost=total_cost,
                quality_score=quality_score,
                tournament_compliant=tournament_compliant,
                final_forecast=forecast_result.prediction,
                reasoning=final_reasoning,
            )

            self.logger.info(
                f"Multi-stage pipeline completed: "
                f"Success={pipeline_success}, Quality={quality_score:.2f}, "
                f"Cost=${total_cost:.4f}, Time={total_execution_time:.2f}s"
            )

            return result

        except Exception as e:
            total_execution_time = (datetime.now() - pipeline_start).total_seconds()
            self.logger.error(f"Multi-stage pipeline failed: {e}")

            # Return failed result
            return MultiStageResult(
                question=question,
                question_type=question_type,
                research_result=ResearchStageResult(
                    content="",
                    sources_used=[],
                    model_used="none",
                    cost_estimate=0.0,
                    quality_score=0.0,
                    stage_name="failed",
                    execution_time=0.0,
                    success=False,
                    error_message=str(e),
                ),
                validation_result=ValidationResult(
                    is_valid=False,
                    quality_score=0.0,
                    evidence_traceability_score=0.0,
                    hallucination_detected=True,
                    logical_consistency_score=0.0,
                    issues_identified=[f"Pipeline error: {str(e)}"],
                    recommendations=["Retry with different approach"],
                    confidence_level="low",
                    execution_time=0.0,
                    cost_estimate=0.0,
                ),
                forecast_result=ForecastResult(
                    forecast_type=question_type,
                    prediction=0.5 if question_type == "binary" else {},
                    confidence_score=0.0,
                    uncertainty_bounds=None,
                    calibration_score=0.0,
                    overconfidence_detected=True,
                    quality_validation_passed=False,
                    tournament_compliant=False,
                    reasoning=f"Pipeline error: {str(e)}",
                    execution_time=0.0,
                    cost_estimate=0.0,
                    model_used="none",
                ),
                pipeline_success=False,
                total_execution_time=total_execution_time,
                total_cost=0.0,
                quality_score=0.0,
                tournament_compliant=False,
                final_forecast=0.5 if question_type == "binary" else {},
                reasoning=f"Multi-stage pipeline failed: {str(e)}",
            )

    def _calculate_overall_quality_score(
        self,
        research_results: Dict[str, Any],
        validation_result: ValidationResult,
        forecast_result: ForecastResult,
    ) -> float:
        """Calculate overall quality score from all stages."""

        # Research quality (30% weight)
        research_quality = 0.0
        if research_results.get("quality_metrics"):
            quality_metrics = research_results["quality_metrics"]
            if hasattr(quality_metrics, "overall_quality"):
                research_quality = quality_metrics.overall_quality
            elif isinstance(quality_metrics, dict):
                research_quality = quality_metrics.get("overall_quality", 0.0)
        elif research_results.get("success"):
            research_quality = 0.6  # Default for successful research

        # Validation quality (30% weight)
        validation_quality = validation_result.quality_score

        # Forecasting quality (40% weight)
        forecast_quality = forecast_result.calibration_score

        # Weighted average
        overall_quality = (
            research_quality * 0.3 + validation_quality * 0.3 + forecast_quality * 0.4
        )

        return overall_quality

    def _check_overall_tournament_compliance(
        self, validation_result: ValidationResult, forecast_result: ForecastResult
    ) -> bool:
        """Check overall tournament compliance across all stages."""

        compliance_checks = []

        # Validation stage compliance
        compliance_checks.append(validation_result.is_valid)
        compliance_checks.append(not validation_result.hallucination_detected)
        compliance_checks.append(validation_result.evidence_traceability_score >= 0.5)

        # Forecasting stage compliance
        compliance_checks.append(forecast_result.tournament_compliant)
        compliance_checks.append(forecast_result.quality_validation_passed)
        compliance_checks.append(not forecast_result.overconfidence_detected)

        # At least 4 out of 6 compliance checks must pass
        return sum(compliance_checks) >= 4

    def _compile_final_reasoning(
        self,
        research_content: str,
        validation_result: ValidationResult,
        forecast_result: ForecastResult,
    ) -> str:
        """Compile final reasoning from all stages."""

        reasoning_sections = []

        # Research summary
        reasoning_sections.append("## Research Summary")
        if research_content:
            # Extract key findings from research
            research_lines = research_content.split("\n")[:10]  # First 10 lines
            reasoning_sections.append("\n".join(research_lines))
        else:
            reasoning_sections.append("Research data unavailable.")

        # Validation summary
        reasoning_sections.append("\n## Quality Validation")
        reasoning_sections.append(
            f"Quality Score: {validation_result.quality_score:.2f}"
        )
        reasoning_sections.append(
            f"Evidence Traceability: {validation_result.evidence_traceability_score:.2f}"
        )
        reasoning_sections.append(
            f"Hallucination Check: {'✅ Clean' if not validation_result.hallucination_detected else '⚠️ Issues detected'}"
        )

        if validation_result.issues_identified:
            reasoning_sections.append("Quality Issues:")
            for issue in validation_result.issues_identified[:3]:  # Top 3 issues
                reasoning_sections.append(f"- {issue}")

        # Forecasting reasoning
        reasoning_sections.append("\n## Forecasting Analysis")
        reasoning_sections.append(forecast_result.reasoning)

        # Final assessment
        reasoning_sections.append("\n## Final Assessment")
        reasoning_sections.append(
            f"Calibration Score: {forecast_result.calibration_score:.2f}"
        )
        reasoning_sections.append(
            f"Confidence Level: {forecast_result.confidence_score:.2f}"
        )
        reasoning_sections.append(
            f"Tournament Compliant: {'✅ Yes' if forecast_result.tournament_compliant else '❌ No'}"
        )

        return "\n".join(reasoning_sections)

    async def get_pipeline_health_check(self) -> Dict[str, Any]:
        """Get health check status for all pipeline components."""

        health_status = {
            "pipeline": "MultiStageValidationPipeline",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        # Check research pipeline
        try:
            research_status = self.research_pipeline.get_pipeline_status()
            health_status["components"]["research"] = {
                "status": "healthy",
                "details": research_status,
            }
        except Exception as e:
            health_status["components"]["research"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check validation service
        try:
            validation_status = self.validation_service.get_validation_status()
            health_status["components"]["validation"] = {
                "status": "healthy",
                "details": validation_status,
            }
        except Exception as e:
            health_status["components"]["validation"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check forecasting service
        try:
            forecasting_status = self.forecasting_service.get_service_status()
            health_status["components"]["forecasting"] = {
                "status": "healthy",
                "details": forecasting_status,
            }
        except Exception as e:
            health_status["components"]["forecasting"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Overall health
        component_statuses = [
            comp["status"] for comp in health_status["components"].values()
        ]
        health_status["overall_health"] = (
            "healthy" if all(s == "healthy" for s in component_statuses) else "degraded"
        )

        return health_status

    def get_pipeline_configuration(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return {
            "pipeline": "MultiStageValidationPipeline",
            "stages": [
                "research_with_asknews_and_gpt5_mini",
                "validation_with_gpt5_nano",
                "forecasting_with_gpt5_full",
            ],
            "models_used": {
                "research": [
                    "AskNews API (free)",
                    "openai/gpt-5-mini",
                    "openai/gpt-oss-20b:free",
                    "moonshotai/kimi-k2:free",
                ],
                "validation": ["openai/gpt-5-nano"],
                "forecasting": ["openai/gpt-5"],
            },
            "cost_optimization": {
                "asknews_free_via_metaculusq4": True,
                "free_model_fallbacks": True,
                "target_cost_per_question": self.cost_budget_per_question,
            },
            "quality_thresholds": {
                "overall_quality_threshold": self.quality_threshold,
                "validation_threshold": 0.6,
                "calibration_threshold": 0.5,
            },
            "tournament_compliance": {
                "evidence_traceability_required": True,
                "hallucination_detection_enabled": True,
                "calibration_checks_enabled": True,
                "uncertainty_quantification_required": True,
            },
        }
