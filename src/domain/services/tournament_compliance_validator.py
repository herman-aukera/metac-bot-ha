"""Tournament compliance validation service for reasoning comments and transparency requirements."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from ..entities.forecast import Forecast
from ..entities.prediction import Prediction


@dataclass
class ComplianceIssue:
    """Represents a compliance issue found during validation."""

    severity: str  # "error", "warning", "info"
    category: str  # "reasoning", "transparency", "format", "content"
    message: str
    suggestion: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        # Use message as description if description not provided
        if self.description is None:
            self.description = self.message


@dataclass
class ComplianceReport:
    """Report of compliance validation results."""

    is_compliant: bool
    issues: List[ComplianceIssue]
    score: float  # 0.0 to 1.0
    validation_timestamp: datetime
    compliance_areas_checked: List[str] = None

    def __post_init__(self):
        if self.compliance_areas_checked is None:
            self.compliance_areas_checked = []


class TournamentComplianceValidator:
    """Validates tournament compliance for reasoning comments and transparency."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Tournament transparency requirements
        self.min_reasoning_length = 100
        self.required_reasoning_elements = [
            "analysis",
            "evidence",
            "conclusion",
            "uncertainty",
        ]

        # Formatting requirements
        self.max_reasoning_length = 2000  # Reasonable limit for readability
        self.prohibited_patterns = [
            r"I am an AI",
            r"As an AI",
            r"I cannot",
            r"I don't have access",
            r"I'm not able to",
        ]

    def validate_reasoning_comment(self, prediction: Prediction) -> ComplianceReport:
        """Validate a single prediction's reasoning comment for tournament compliance."""
        issues = []

        # Check reasoning exists and has minimum length
        if (
            not prediction.reasoning
            or len(prediction.reasoning.strip()) < self.min_reasoning_length
        ):
            issues.append(
                ComplianceIssue(
                    severity="error",
                    category="reasoning",
                    message=f"Reasoning comment too short ({len(prediction.reasoning if prediction.reasoning else 0)} chars, minimum {self.min_reasoning_length})",
                    suggestion="Provide more detailed analysis including evidence, uncertainty assessment, and clear reasoning steps",
                )
            )

        # Check for maximum length (readability)
        if (
            prediction.reasoning
            and len(prediction.reasoning) > self.max_reasoning_length
        ):
            issues.append(
                ComplianceIssue(
                    severity="warning",
                    category="format",
                    message=f"Reasoning comment very long ({len(prediction.reasoning)} chars, recommended max {self.max_reasoning_length})",
                    suggestion="Consider condensing the reasoning while maintaining key points",
                )
            )

        # Check for prohibited AI disclosure patterns
        if prediction.reasoning:
            for pattern in self.prohibited_patterns:
                if re.search(pattern, prediction.reasoning, re.IGNORECASE):
                    issues.append(
                        ComplianceIssue(
                            severity="warning",
                            category="transparency",
                            message=f"Reasoning contains AI self-reference pattern: {pattern}",
                            suggestion="Remove AI self-references to maintain tournament compliance",
                        )
                    )

        # Check for structured reasoning elements
        reasoning_quality_issues = self._validate_reasoning_quality(
            prediction.reasoning
        )
        issues.extend(reasoning_quality_issues)

        # Check for transparency requirements
        transparency_issues = self._validate_transparency_requirements(prediction)
        issues.extend(transparency_issues)

        # Calculate compliance score
        score = self._calculate_compliance_score(issues)
        is_compliant = score >= 0.8 and not any(
            issue.severity == "error" for issue in issues
        )

        return ComplianceReport(
            is_compliant=is_compliant,
            issues=issues,
            score=score,
            validation_timestamp=datetime.utcnow(),
        )

    def validate_forecast_compliance(self, forecast: Forecast) -> ComplianceReport:
        """Validate entire forecast for tournament compliance."""
        all_issues = []
        prediction_scores = []

        # Validate each prediction
        for prediction in forecast.predictions:
            pred_report = self.validate_reasoning_comment(prediction)
            all_issues.extend(pred_report.issues)
            prediction_scores.append(pred_report.score)

        # Check forecast-level requirements
        forecast_issues = self._validate_forecast_level_requirements(forecast)
        all_issues.extend(forecast_issues)

        # Calculate overall score
        if prediction_scores:
            avg_prediction_score = sum(prediction_scores) / len(prediction_scores)
        else:
            avg_prediction_score = 0.0

        forecast_score = self._calculate_compliance_score(forecast_issues)
        overall_score = (avg_prediction_score + forecast_score) / 2

        is_compliant = overall_score >= 0.8 and not any(
            issue.severity == "error" for issue in all_issues
        )

        return ComplianceReport(
            is_compliant=is_compliant,
            issues=all_issues,
            score=overall_score,
            validation_timestamp=datetime.utcnow(),
        )

    def format_reasoning_for_tournament(self, prediction: Prediction) -> str:
        """Format reasoning comment to meet tournament transparency requirements."""
        if not prediction.reasoning:
            return self._generate_minimal_reasoning(prediction)

        # Clean up AI self-references
        formatted_reasoning = prediction.reasoning
        for pattern in self.prohibited_patterns:
            formatted_reasoning = re.sub(
                pattern, "", formatted_reasoning, flags=re.IGNORECASE
            )

        # Ensure structured format
        if not self._has_structured_format(formatted_reasoning):
            formatted_reasoning = self._add_structure_to_reasoning(
                formatted_reasoning, prediction
            )

        # Add transparency footer if needed
        if not self._has_transparency_elements(formatted_reasoning):
            formatted_reasoning = self._add_transparency_elements(
                formatted_reasoning, prediction
            )

        return formatted_reasoning.strip()

    def _validate_reasoning_quality(self, reasoning: str) -> List[ComplianceIssue]:
        """Validate the quality and structure of reasoning."""
        issues = []

        if not reasoning:
            return issues

        # Check for evidence-based reasoning
        evidence_indicators = [
            "evidence",
            "data",
            "source",
            "study",
            "report",
            "analysis",
        ]
        if not any(indicator in reasoning.lower() for indicator in evidence_indicators):
            issues.append(
                ComplianceIssue(
                    severity="warning",
                    category="content",
                    message="Reasoning lacks clear evidence references",
                    suggestion="Include specific evidence, data sources, or analytical basis for the prediction",
                )
            )

        # Check for uncertainty acknowledgment
        uncertainty_indicators = [
            "uncertain",
            "confidence",
            "likely",
            "probability",
            "risk",
            "doubt",
        ]
        if not any(
            indicator in reasoning.lower() for indicator in uncertainty_indicators
        ):
            issues.append(
                ComplianceIssue(
                    severity="warning",
                    category="content",
                    message="Reasoning lacks uncertainty assessment",
                    suggestion="Include discussion of confidence level and potential uncertainties",
                )
            )

        # Check for logical structure
        if not self._has_logical_structure(reasoning):
            issues.append(
                ComplianceIssue(
                    severity="info",
                    category="format",
                    message="Reasoning could benefit from clearer logical structure",
                    suggestion="Consider organizing reasoning with clear analysis → evidence → conclusion flow",
                )
            )

        return issues

    def _validate_transparency_requirements(
        self, prediction: Prediction
    ) -> List[ComplianceIssue]:
        """Validate tournament transparency requirements."""
        issues = []

        # Check for reasoning steps documentation
        if not prediction.reasoning_steps or len(prediction.reasoning_steps) < 2:
            issues.append(
                ComplianceIssue(
                    severity="info",
                    category="transparency",
                    message="Limited reasoning steps documentation",
                    suggestion="Document key reasoning steps for better transparency",
                )
            )

        # Check for method transparency
        if not prediction.method_metadata:
            issues.append(
                ComplianceIssue(
                    severity="info",
                    category="transparency",
                    message="No method metadata provided",
                    suggestion="Include metadata about the forecasting method used",
                )
            )

        return issues

    def _validate_forecast_level_requirements(
        self, forecast: Forecast
    ) -> List[ComplianceIssue]:
        """Validate forecast-level tournament requirements."""
        issues = []

        # Check that reasoning is published (this should be enforced at submission level)
        if not forecast.predictions:
            issues.append(
                ComplianceIssue(
                    severity="error",
                    category="reasoning",
                    message="Forecast has no predictions with reasoning",
                    suggestion="Ensure at least one prediction with reasoning is included",
                )
            )

        # Check for ensemble reasoning if multiple predictions
        if len(forecast.predictions) > 1 and not forecast.reasoning_summary:
            issues.append(
                ComplianceIssue(
                    severity="warning",
                    category="transparency",
                    message="Multiple predictions without ensemble reasoning summary",
                    suggestion="Provide summary reasoning for how multiple predictions were combined",
                )
            )

        return issues

    def _calculate_compliance_score(self, issues: List[ComplianceIssue]) -> float:
        """Calculate compliance score based on issues found."""
        if not issues:
            return 1.0

        # Weight issues by severity
        penalty_weights = {"error": 0.3, "warning": 0.1, "info": 0.05}

        total_penalty = sum(
            penalty_weights.get(issue.severity, 0.1) for issue in issues
        )
        score = max(0.0, 1.0 - total_penalty)

        return score

    def _has_structured_format(self, reasoning: str) -> bool:
        """Check if reasoning has a structured format."""
        structure_indicators = [
            "analysis:",
            "evidence:",
            "conclusion:",
            "1.",
            "2.",
            "3.",
            "•",
            "-",
            "first",
            "second",
            "finally",
        ]
        return any(indicator in reasoning.lower() for indicator in structure_indicators)

    def _has_logical_structure(self, reasoning: str) -> bool:
        """Check if reasoning follows logical structure."""
        # Simple heuristic: check for transition words and logical flow
        transition_words = [
            "therefore",
            "however",
            "because",
            "since",
            "given",
            "consequently",
            "furthermore",
            "moreover",
            "in conclusion",
        ]
        return any(word in reasoning.lower() for word in transition_words)

    def _has_transparency_elements(self, reasoning: str) -> bool:
        """Check if reasoning includes transparency elements."""
        transparency_elements = [
            "confidence",
            "uncertainty",
            "evidence",
            "method",
            "approach",
        ]
        return any(element in reasoning.lower() for element in transparency_elements)

    def _generate_minimal_reasoning(self, prediction: Prediction) -> str:
        """Generate minimal compliant reasoning when none exists."""
        return f"""Analysis: Based on available information and forecasting method {prediction.method.value}.

Evidence: Prediction generated using {prediction.method.value} approach with {prediction.confidence.value} confidence level.

Conclusion: Probability assessment reflects current understanding with appropriate uncertainty acknowledgment.

Confidence: {prediction.confidence.value} confidence based on available evidence and analytical approach."""

    def _add_structure_to_reasoning(
        self, reasoning: str, prediction: Prediction
    ) -> str:
        """Add structure to unstructured reasoning."""
        # Simple approach: add headers if reasoning is long enough
        if len(reasoning) > 200:
            # Try to identify natural breaks and add structure
            sentences = reasoning.split(". ")
            if len(sentences) >= 3:
                mid_point = len(sentences) // 2
                structured = f"Analysis: {'. '.join(sentences[:mid_point])}.\n\nConclusion: {'. '.join(sentences[mid_point:])}."
                return structured

        return f"Analysis: {reasoning}\n\nConfidence: {prediction.confidence.value} confidence level."

    def _add_transparency_elements(self, reasoning: str, prediction: Prediction) -> str:
        """Add transparency elements to reasoning."""
        transparency_footer = f"\n\nMethod: {prediction.method.value} | Confidence: {prediction.confidence.value}"

        if prediction.method_metadata:
            metadata_str = ", ".join(
                f"{k}: {v}" for k, v in prediction.method_metadata.items()
            )
            transparency_footer += f" | Details: {metadata_str}"

        return reasoning + transparency_footer

    def validate_reasoning_transparency(
        self, prediction: Prediction, question
    ) -> ComplianceReport:
        """Validate reasoning transparency requirements for tournament compliance."""
        issues = []

        # Check if reasoning exists and has sufficient detail
        if (
            not prediction.reasoning
            or len(prediction.reasoning.strip()) < self.min_reasoning_length
        ):
            issues.append(
                ComplianceIssue(
                    severity="error",
                    category="transparency",
                    message="Reasoning has insufficient detail for transparency requirements",
                    suggestion="Provide detailed analysis with evidence and reasoning steps",
                )
            )

        # Check for required reasoning elements
        if prediction.reasoning:
            reasoning_lower = prediction.reasoning.lower()
            missing_elements = []

            if "analysis" not in reasoning_lower and "based on" not in reasoning_lower:
                missing_elements.append("analysis")
            if "evidence" not in reasoning_lower and "data" not in reasoning_lower:
                missing_elements.append("evidence")
            if (
                "therefore" not in reasoning_lower
                and "conclusion" not in reasoning_lower
                and "estimate" not in reasoning_lower
                and "probability" not in reasoning_lower
            ):
                missing_elements.append("conclusion")

            if missing_elements:
                issues.append(
                    ComplianceIssue(
                        severity="warning",
                        category="transparency",
                        message=f"Reasoning missing elements: {', '.join(missing_elements)}",
                        suggestion="Include analysis, evidence, and clear conclusions",
                    )
                )

        # Calculate compliance score
        score = 1.0 - (len(issues) * 0.3)  # Reduce score for each issue
        is_compliant = len([i for i in issues if i.severity == "error"]) == 0

        return ComplianceReport(
            is_compliant=is_compliant,
            issues=issues,
            score=max(0.0, score),
            validation_timestamp=datetime.utcnow(),
            compliance_areas_checked=["transparency", "reasoning_quality"],
        )

    def validate_automated_decision_making(
        self, process_metadata: Dict
    ) -> ComplianceReport:
        """Validate that decision-making process is fully automated."""
        issues = []

        # Check for human involvement
        human_involvement = process_metadata.get("human_involvement", "none")
        if human_involvement != "none":
            issues.append(
                ComplianceIssue(
                    severity="error",
                    category="automation",
                    message=f"Human involvement detected: {human_involvement}",
                    suggestion="Ensure all decision points are fully automated",
                )
            )

        # Check decision points
        decision_points = process_metadata.get("decision_points", [])
        for point in decision_points:
            if not point.get("automated", True):
                issues.append(
                    ComplianceIssue(
                        severity="error",
                        category="automation",
                        message=f"Non-automated decision point: {point.get('step', 'unknown')}",
                        suggestion="All decision points must be automated for tournament compliance",
                    )
                )

        score = 1.0 - (len(issues) * 0.5)  # Strict scoring for automation
        is_compliant = len([i for i in issues if i.severity == "error"]) == 0

        return ComplianceReport(
            is_compliant=is_compliant,
            issues=issues,
            score=max(0.0, score),
            validation_timestamp=datetime.utcnow(),
            compliance_areas_checked=["automation", "decision_making"],
        )

    def validate_data_source_compliance(self, data_sources: Dict) -> ComplianceReport:
        """Validate compliance with data source restrictions."""
        issues = []

        # Check for private information usage
        if data_sources.get("private_information", False):
            issues.append(
                ComplianceIssue(
                    severity="error",
                    category="data_sources",
                    message="Private information usage detected",
                    suggestion="Only use publicly available data sources",
                )
            )

        # Check for restricted sources
        restricted_sources = data_sources.get("restricted_sources", [])
        if restricted_sources:
            issues.append(
                ComplianceIssue(
                    severity="error",
                    category="data_sources",
                    message=f"Restricted sources used: {', '.join(restricted_sources)}",
                    suggestion="Remove restricted data sources from analysis",
                )
            )

        # Check source types for compliance
        sources_used = data_sources.get("sources_used", [])
        for source in sources_used:
            source_type = source.get("type", "")
            if source_type in [
                "private_database",
                "insider_information",
                "confidential",
            ]:
                issues.append(
                    ComplianceIssue(
                        severity="error",
                        category="data_sources",
                        message=f"Non-compliant source type: {source_type}",
                        suggestion="Use only public data sources",
                    )
                )

        score = 1.0 - (len(issues) * 0.4)
        is_compliant = len([i for i in issues if i.severity == "error"]) == 0

        return ComplianceReport(
            is_compliant=is_compliant,
            issues=issues,
            score=max(0.0, score),
            validation_timestamp=datetime.utcnow(),
            compliance_areas_checked=["data_sources", "privacy"],
        )

    def validate_prediction_format(
        self, prediction: Prediction, question
    ) -> ComplianceReport:
        """Validate prediction format compliance."""
        issues = []

        # Check for required fields based on question type
        if hasattr(prediction, "result") and prediction.result:
            if question.question_type.value == "binary":
                if prediction.result.binary_probability is None:
                    issues.append(
                        ComplianceIssue(
                            severity="error",
                            category="format",
                            message="Missing required binary probability",
                            suggestion="Provide binary probability for binary questions",
                        )
                    )
            elif question.question_type.value == "numeric":
                if prediction.result.numeric_value is None:
                    issues.append(
                        ComplianceIssue(
                            severity="error",
                            category="format",
                            message="Missing required numeric value",
                            suggestion="Provide numeric value for numeric questions",
                        )
                    )
        else:
            # Fallback for older prediction format
            if not hasattr(prediction, "probability") or prediction.probability is None:
                issues.append(
                    ComplianceIssue(
                        severity="error",
                        category="format",
                        message="Missing required probability field",
                        suggestion="Provide probability value for prediction",
                    )
                )

        # Check reasoning field
        if not prediction.reasoning or len(prediction.reasoning.strip()) == 0:
            issues.append(
                ComplianceIssue(
                    severity="error",
                    category="format",
                    message="Missing required reasoning field",
                    suggestion="Provide detailed reasoning for prediction",
                )
            )

        # Check format version if available
        if hasattr(prediction, "metadata") and prediction.metadata:
            format_version = prediction.metadata.get("format_version")
            if format_version and format_version < "1.0":
                issues.append(
                    ComplianceIssue(
                        severity="warning",
                        category="format",
                        message=f"Outdated format version: {format_version}",
                        suggestion="Update to latest format version",
                    )
                )

        score = 1.0 - (len(issues) * 0.3)
        is_compliant = len([i for i in issues if i.severity == "error"]) == 0

        return ComplianceReport(
            is_compliant=is_compliant,
            issues=issues,
            score=max(0.0, score),
            validation_timestamp=datetime.utcnow(),
            compliance_areas_checked=["format", "required_fields"],
        )

    def run_comprehensive_compliance_check(
        self, prediction: Prediction, question, metadata: Dict
    ) -> ComplianceReport:
        """Run comprehensive compliance validation across all areas."""
        all_issues = []
        all_areas_checked = []
        scores = []

        # Run transparency validation
        transparency_report = self.validate_reasoning_transparency(prediction, question)
        all_issues.extend(transparency_report.issues)
        all_areas_checked.extend(transparency_report.compliance_areas_checked)
        scores.append(transparency_report.score)

        # Run automation validation
        automation_report = self.validate_automated_decision_making(metadata)
        all_issues.extend(automation_report.issues)
        all_areas_checked.extend(automation_report.compliance_areas_checked)
        scores.append(automation_report.score)

        # Run data source validation
        data_sources = metadata.get("data_sources", {})
        if isinstance(data_sources, list):
            # Convert list format to dict format
            data_sources = {
                "sources_used": data_sources,
                "private_information": False,
                "restricted_sources": [],
            }

        data_report = self.validate_data_source_compliance(data_sources)
        all_issues.extend(data_report.issues)
        all_areas_checked.extend(data_report.compliance_areas_checked)
        scores.append(data_report.score)

        # Run format validation
        format_report = self.validate_prediction_format(prediction, question)
        all_issues.extend(format_report.issues)
        all_areas_checked.extend(format_report.compliance_areas_checked)
        scores.append(format_report.score)

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0.0
        is_compliant = len([i for i in all_issues if i.severity == "error"]) == 0

        return ComplianceReport(
            is_compliant=is_compliant,
            issues=all_issues,
            score=overall_score,
            validation_timestamp=datetime.utcnow(),
            compliance_areas_checked=list(set(all_areas_checked)),
        )
