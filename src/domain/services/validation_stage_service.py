"""
Validation Stage Service with GPT-5-Nano for Quality Assurance.
Implements task 4.2 requirements with evidence traceability and hallucination detection.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from validation stage analysis."""
    is_valid: bool
    quality_score: float
    evidence_traceability_score: float
    hallucination_detected: bool
    logical_consistency_score: float
    issues_identified: List[str]
    recommendations: List[str]
    confidence_level: str
    execution_time: float
    cost_estimate: float


@dataclass
class QualityIssue:
    """Represents a quality issue found during validation."""
    issue_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    location: str
    recommendation: str


class ValidationStageService:
    """
    Enhanced validation stage service using GPT-5-nano for quality assurance.

    Features:
    - Evidence traceability verification
    - Hallucination detection
    - Logical consistency checking
    - Quality scoring and issue identification
    - Automated quality reporting
    """

    def __init__(self, tri_model_router=None):
        """Initialize the validation stage service."""
        self.tri_model_router = tri_model_router
        self.logger = logging.getLogger(__name__)

        # Validation thresholds
        self.quality_threshold = 0.7
        self.evidence_threshold = 0.6
        self.consistency_threshold = 0.8

    async def validate_content(self, content: str, task_type: str = "research_synthesis",
                             context: Dict[str, Any] = None) -> ValidationResult:
        """
        Execute comprehensive validation using GPT-5-nano for quality assurance.

        Args:
            content: Content to validate
            task_type: Type of task being validated
            context: Additional context for validation

        Returns:
            ValidationResult with comprehensive quality assessment
        """
        context = context or {}
        validation_start = datetime.now()

        self.logger.info(f"Starting validation for {task_type} content...")

        try:
            # Step 1: Create validation prompts optimized for gpt-5-nano
            validation_prompts = await self._create_validation_prompts(content, task_type, context)

            # Step 2: Execute evidence traceability verification
            evidence_result = await self._verify_evidence_traceability(content, validation_prompts["evidence"])

            # Step 3: Execute hallucination detection
            hallucination_result = await self._detect_hallucinations(content, validation_prompts["hallucination"])

            # Step 4: Execute logical consistency checking
            consistency_result = await self._check_logical_consistency(content, validation_prompts["consistency"])

            # Step 5: Generate quality scoring
            quality_score = await self._calculate_quality_score(content, validation_prompts["quality"])

            # Step 6: Identify and compile issues
            issues = await self._identify_quality_issues(
                evidence_result, hallucination_result, consistency_result, quality_score
            )

            execution_time = (datetime.now() - validation_start).total_seconds()

            return ValidationResult(
                is_valid=quality_score.overall_score >= self.quality_threshold,
                quality_score=quality_score.overall_score,
                evidence_traceability_score=evidence_result.score,
                hallucination_detected=hallucination_result.detected,
                logical_consistency_score=consistency_result.score,
                issues_identified=[issue.description for issue in issues],
                recommendations=[issue.recommendation for issue in issues],
                confidence_level=self._determine_confidence_level(quality_score.overall_score),
                execution_time=execution_time,
                cost_estimate=evidence_result.cost + hallucination_result.cost +
                             consistency_result.cost + quality_score.cost
            )

        except Exception as e:
            execution_time = (datetime.now() - validation_start).total_seconds()
            self.logger.error(f"Validation failed: {e}")

            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                evidence_traceability_score=0.0,
                hallucination_detected=True,
                logical_consistency_score=0.0,
                issues_identified=[f"Validation error: {str(e)}"],
                recommendations=["Retry validation with different approach"],
                confidence_level="low",
                execution_time=execution_time,
                cost_estimate=0.0
            )
    async def _create_validation_prompts(self, content: str, task_type: str,
                                       context: Dict[str, Any]) -> Dict[str, str]:
        """Create validation prompts optimized for gpt-5-nano capabilities."""

        # Import anti-slop prompts for base validation structure
        from ...prompts.anti_slop_prompts import anti_slop_prompts

        # Evidence traceability prompt
        evidence_prompt = f"""
{anti_slop_prompts.get_base_anti_slop_directives()}

## GPT-5-NANO EVIDENCE TRACEABILITY VERIFICATION:

### TASK: Verify evidence traceability in the following content
### FOCUS: Check for proper source citations and evidence backing

CONTENT TO ANALYZE:
{content}

### VERIFICATION CHECKLIST:
1. Citation Format: Are sources cited as [Source: URL/Publication, Date]?
2. Citation Coverage: Does every factual claim have a citation?
3. Citation Quality: Are citations specific and verifiable?
4. Evidence Gaps: Are unsupported claims flagged appropriately?

### OUTPUT FORMAT:
- Citations Found: X/Y claims cited
- Citation Quality: GOOD/FAIR/POOR
- Evidence Gaps: [List any gaps]
- Overall Evidence Score: X/10
- Status: PASS/FAIL

Keep response concise and focused on evidence verification.
"""

        # Hallucination detection prompt
        hallucination_prompt = f"""
{anti_slop_prompts.get_base_anti_slop_directives()}

## GPT-5-NANO HALLUCINATION DETECTION:

### TASK: Detect potential hallucinations and unsupported claims
### FOCUS: Identify statements that cannot be verified or seem fabricated

CONTENT TO ANALYZE:
{content}

### DETECTION CRITERIA:
1. Fabricated Facts: Claims that seem made up or too specific without sources
2. Impossible Claims: Statements that contradict known facts
3. Overly Precise Data: Exact numbers/dates without proper attribution
4. Speculation Presented as Fact: Uncertain information stated definitively

### OUTPUT FORMAT:
- Potential Hallucinations: [List specific examples]
- Severity: LOW/MEDIUM/HIGH
- Confidence in Detection: LOW/MEDIUM/HIGH
- Hallucination Risk Score: X/10
- Status: CLEAN/SUSPICIOUS/PROBLEMATIC

Focus on clear, verifiable issues only.
"""
        # Logical consistency prompt
        consistency_prompt = f"""
{anti_slop_prompts.get_base_anti_slop_directives()}

## GPT-5-NANO LOGICAL CONSISTENCY CHECK:

### TASK: Check logical consistency and coherence
### FOCUS: Identify contradictions and logical errors

CONTENT TO ANALYZE:
{content}

### CONSISTENCY CHECKS:
1. Internal Contradictions: Do statements contradict each other?
2. Logical Flow: Does reasoning follow logically?
3. Temporal Consistency: Are dates and timelines coherent?
4. Causal Relationships: Are cause-effect claims logical?

### OUTPUT FORMAT:
- Contradictions Found: [List specific contradictions]
- Logic Issues: [List logical problems]
- Consistency Score: X/10
- Status: CONSISTENT/MINOR_ISSUES/MAJOR_ISSUES

Keep analysis focused and specific.
"""

        # Quality scoring prompt
        quality_prompt = f"""
{anti_slop_prompts.get_base_anti_slop_directives()}

## GPT-5-NANO QUALITY SCORING:

### TASK: Provide overall quality assessment
### FOCUS: Comprehensive quality evaluation

CONTENT TO ANALYZE:
{content}

TASK TYPE: {task_type}

### QUALITY DIMENSIONS:
1. Accuracy: Are facts correct and verifiable?
2. Completeness: Is coverage comprehensive?
3. Clarity: Is information clearly presented?
4. Relevance: Is content relevant to the task?
5. Reliability: Are sources credible?

### OUTPUT FORMAT:
- Accuracy Score: X/10
- Completeness Score: X/10
- Clarity Score: X/10
- Relevance Score: X/10
- Reliability Score: X/10
- Overall Quality Score: X/10
- Status: EXCELLENT/GOOD/FAIR/POOR

Provide brief justification for scores.
"""

        return {
            "evidence": evidence_prompt,
            "hallucination": hallucination_prompt,
            "consistency": consistency_prompt,
            "quality": quality_prompt
        }
    async def _verify_evidence_traceability(self, content: str, prompt: str) -> Any:
        """Execute evidence traceability verification using GPT-5-nano."""

        @dataclass
        class EvidenceResult:
            score: float
            citations_found: int
            citations_expected: int
            gaps_identified: List[str]
            cost: float

        if not self.tri_model_router:
            return EvidenceResult(0.0, 0, 0, ["Router unavailable"], 0.0)

        try:
            nano_model = self.tri_model_router.models.get("nano")
            if not nano_model:
                return EvidenceResult(0.0, 0, 0, ["GPT-5-nano unavailable"], 0.0)

            result = await nano_model.invoke(prompt)

            # Parse evidence verification result
            citations_found = self._extract_number_from_text(result, "Citations Found:")
            evidence_score = self._extract_score_from_text(result, "Overall Evidence Score:")
            gaps = self._extract_list_from_text(result, "Evidence Gaps:")

            # Estimate cost for GPT-5-nano
            estimated_tokens = len(prompt.split()) + len(result.split())
            cost = (estimated_tokens / 1_000_000) * 0.05

            return EvidenceResult(
                score=evidence_score / 10.0 if evidence_score else 0.5,
                citations_found=citations_found or 0,
                citations_expected=content.count("[Source:") if content else 0,
                gaps_identified=gaps,
                cost=cost
            )

        except Exception as e:
            self.logger.error(f"Evidence verification failed: {e}")
            return EvidenceResult(0.0, 0, 0, [f"Error: {str(e)}"], 0.0)
    async def _detect_hallucinations(self, content: str, prompt: str) -> Any:
        """Execute hallucination detection using GPT-5-nano."""

        @dataclass
        class HallucinationResult:
            detected: bool
            severity: str
            examples: List[str]
            confidence: str
            risk_score: float
            cost: float

        if not self.tri_model_router:
            return HallucinationResult(True, "high", ["Router unavailable"], "low", 1.0, 0.0)

        try:
            nano_model = self.tri_model_router.models.get("nano")
            if not nano_model:
                return HallucinationResult(True, "high", ["GPT-5-nano unavailable"], "low", 1.0, 0.0)

            result = await nano_model.invoke(prompt)

            # Parse hallucination detection result
            hallucinations = self._extract_list_from_text(result, "Potential Hallucinations:")
            severity = self._extract_value_from_text(result, "Severity:", ["LOW", "MEDIUM", "HIGH"])
            confidence = self._extract_value_from_text(result, "Confidence in Detection:", ["LOW", "MEDIUM", "HIGH"])
            risk_score = self._extract_score_from_text(result, "Hallucination Risk Score:")
            status = self._extract_value_from_text(result, "Status:", ["CLEAN", "SUSPICIOUS", "PROBLEMATIC"])

            # Estimate cost for GPT-5-nano
            estimated_tokens = len(prompt.split()) + len(result.split())
            cost = (estimated_tokens / 1_000_000) * 0.05

            # Filter out empty or invalid hallucination examples
            valid_hallucinations = [h for h in hallucinations if h and h.strip() and
                                   not h.lower().startswith(('none', 'severity:', 'confidence:'))]

            return HallucinationResult(
                detected=status in ["SUSPICIOUS", "PROBLEMATIC"] or len(valid_hallucinations) > 0,
                severity=severity.lower() if severity else "medium",
                examples=valid_hallucinations,
                confidence=confidence.lower() if confidence else "medium",
                risk_score=(risk_score / 10.0) if risk_score else 0.5,
                cost=cost
            )

        except Exception as e:
            self.logger.error(f"Hallucination detection failed: {e}")
            return HallucinationResult(True, "high", [f"Error: {str(e)}"], "low", 1.0, 0.0)
    async def _check_logical_consistency(self, content: str, prompt: str) -> Any:
        """Execute logical consistency checking using GPT-5-nano."""

        @dataclass
        class ConsistencyResult:
            score: float
            contradictions: List[str]
            logic_issues: List[str]
            status: str
            cost: float

        if not self.tri_model_router:
            return ConsistencyResult(0.0, ["Router unavailable"], [], "MAJOR_ISSUES", 0.0)

        try:
            nano_model = self.tri_model_router.models.get("nano")
            if not nano_model:
                return ConsistencyResult(0.0, ["GPT-5-nano unavailable"], [], "MAJOR_ISSUES", 0.0)

            result = await nano_model.invoke(prompt)

            # Parse consistency check result
            contradictions = self._extract_list_from_text(result, "Contradictions Found:")
            logic_issues = self._extract_list_from_text(result, "Logic Issues:")
            consistency_score = self._extract_score_from_text(result, "Consistency Score:")
            status = self._extract_value_from_text(result, "Status:", ["CONSISTENT", "MINOR_ISSUES", "MAJOR_ISSUES"])

            # Estimate cost for GPT-5-nano
            estimated_tokens = len(prompt.split()) + len(result.split())
            cost = (estimated_tokens / 1_000_000) * 0.05

            return ConsistencyResult(
                score=(consistency_score / 10.0) if consistency_score else 0.5,
                contradictions=contradictions,
                logic_issues=logic_issues,
                status=status if status else "MINOR_ISSUES",
                cost=cost
            )

        except Exception as e:
            self.logger.error(f"Consistency check failed: {e}")
            return ConsistencyResult(0.0, [f"Error: {str(e)}"], [], "MAJOR_ISSUES", 0.0)
    async def _calculate_quality_score(self, content: str, prompt: str) -> Any:
        """Calculate comprehensive quality score using GPT-5-nano."""

        @dataclass
        class QualityScore:
            overall_score: float
            accuracy_score: float
            completeness_score: float
            clarity_score: float
            relevance_score: float
            reliability_score: float
            status: str
            cost: float

        if not self.tri_model_router:
            return QualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "POOR", 0.0)

        try:
            nano_model = self.tri_model_router.models.get("nano")
            if not nano_model:
                return QualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "POOR", 0.0)

            result = await nano_model.invoke(prompt)

            # Parse quality scores
            accuracy = self._extract_score_from_text(result, "Accuracy Score:")
            completeness = self._extract_score_from_text(result, "Completeness Score:")
            clarity = self._extract_score_from_text(result, "Clarity Score:")
            relevance = self._extract_score_from_text(result, "Relevance Score:")
            reliability = self._extract_score_from_text(result, "Reliability Score:")
            overall = self._extract_score_from_text(result, "Overall Quality Score:")
            status = self._extract_value_from_text(result, "Status:", ["EXCELLENT", "GOOD", "FAIR", "POOR"])

            # Estimate cost for GPT-5-nano
            estimated_tokens = len(prompt.split()) + len(result.split())
            cost = (estimated_tokens / 1_000_000) * 0.05

            return QualityScore(
                overall_score=(overall / 10.0) if overall else 0.5,
                accuracy_score=(accuracy / 10.0) if accuracy else 0.5,
                completeness_score=(completeness / 10.0) if completeness else 0.5,
                clarity_score=(clarity / 10.0) if clarity else 0.5,
                relevance_score=(relevance / 10.0) if relevance else 0.5,
                reliability_score=(reliability / 10.0) if reliability else 0.5,
                status=status if status else "FAIR",
                cost=cost
            )

        except Exception as e:
            self.logger.error(f"Quality scoring failed: {e}")
            return QualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "POOR", 0.0)
    async def _identify_quality_issues(self, evidence_result, hallucination_result,
                                      consistency_result, quality_score) -> List[QualityIssue]:
        """Identify and compile quality issues from validation results."""
        issues = []

        # Evidence traceability issues
        if evidence_result.score < self.evidence_threshold:
            issues.append(QualityIssue(
                issue_type="evidence_traceability",
                severity="high" if evidence_result.score < 0.3 else "medium",
                description=f"Poor evidence traceability (score: {evidence_result.score:.2f})",
                location="citations",
                recommendation="Add proper source citations with [Source: URL/Publication, Date] format"
            ))

        for gap in evidence_result.gaps_identified:
            if gap and gap.strip() and not gap.lower().startswith('none'):
                issues.append(QualityIssue(
                    issue_type="evidence_gap",
                    severity="medium",
                    description=f"Evidence gap identified: {gap}",
                    location="content",
                    recommendation="Provide supporting evidence or acknowledge uncertainty"
                ))

        # Hallucination issues
        if hallucination_result.detected:
            severity = hallucination_result.severity
            issues.append(QualityIssue(
                issue_type="hallucination",
                severity=severity,
                description=f"Potential hallucinations detected (risk: {hallucination_result.risk_score:.2f})",
                location="content",
                recommendation="Verify claims against reliable sources and remove unsupported statements"
            ))

        for example in hallucination_result.examples:
            if example and example.strip() and not example.lower().startswith(('none', 'severity:', 'confidence:')):
                issues.append(QualityIssue(
                    issue_type="specific_hallucination",
                    severity="medium",
                    description=f"Potential hallucination: {example}",
                    location="content",
                    recommendation="Verify this specific claim or remove if unverifiable"
                ))

        # Logical consistency issues
        if consistency_result.score < self.consistency_threshold:
            issues.append(QualityIssue(
                issue_type="logical_consistency",
                severity="high" if consistency_result.score < 0.5 else "medium",
                description=f"Poor logical consistency (score: {consistency_result.score:.2f})",
                location="reasoning",
                recommendation="Review logical flow and resolve contradictions"
            ))

        for contradiction in consistency_result.contradictions:
            if contradiction and contradiction.strip() and not contradiction.lower().startswith('none'):
                issues.append(QualityIssue(
                    issue_type="contradiction",
                    severity="high",
                    description=f"Contradiction found: {contradiction}",
                    location="content",
                    recommendation="Resolve contradiction or acknowledge conflicting information"
                ))

        for logic_issue in consistency_result.logic_issues:
            if logic_issue and logic_issue.strip() and not logic_issue.lower().startswith('none'):
                issues.append(QualityIssue(
                    issue_type="logic_error",
                    severity="medium",
                    description=f"Logic issue: {logic_issue}",
                    location="reasoning",
                    recommendation="Review and correct logical reasoning"
                ))

        # Overall quality issues
        if quality_score.overall_score < self.quality_threshold:
            issues.append(QualityIssue(
                issue_type="overall_quality",
                severity="high" if quality_score.overall_score < 0.4 else "medium",
                description=f"Overall quality below threshold (score: {quality_score.overall_score:.2f})",
                location="content",
                recommendation="Improve content quality across all dimensions"
            ))

        return issues
    def _determine_confidence_level(self, quality_score: float) -> str:
        """Determine confidence level based on quality score."""
        if quality_score >= 0.8:
            return "high"
        elif quality_score >= 0.6:
            return "medium"
        else:
            return "low"

    def _extract_number_from_text(self, text: str, prefix: str) -> Optional[int]:
        """Extract number from text after a specific prefix."""
        try:
            lines = text.split('\n')
            for line in lines:
                if prefix in line:
                    # Extract number from line like "Citations Found: 5/10 claims cited"
                    parts = line.split(prefix)[1].strip()
                    # Look for first number
                    import re
                    numbers = re.findall(r'\d+', parts)
                    if numbers:
                        return int(numbers[0])
            return None
        except Exception:
            return None

    def _extract_score_from_text(self, text: str, prefix: str) -> Optional[float]:
        """Extract score from text after a specific prefix."""
        try:
            lines = text.split('\n')
            for line in lines:
                if prefix in line:
                    # Extract score from line like "Overall Evidence Score: 7/10"
                    parts = line.split(prefix)[1].strip()
                    import re
                    # Look for pattern like "7/10" or "7.5/10" or just "7.5"
                    score_match = re.search(r'(\d+(?:\.\d+)?)', parts)
                    if score_match:
                        return float(score_match.group(1))
            return None
        except Exception:
            return None

    def _extract_value_from_text(self, text: str, prefix: str, valid_values: List[str]) -> Optional[str]:
        """Extract value from text after a specific prefix, checking against valid values."""
        try:
            lines = text.split('\n')
            for line in lines:
                if prefix in line:
                    parts = line.split(prefix)[1].strip()
                    for value in valid_values:
                        if value in parts.upper():
                            return value
            return None
        except Exception:
            return None

    def _extract_list_from_text(self, text: str, prefix: str) -> List[str]:
        """Extract list items from text after a specific prefix."""
        try:
            items = []
            lines = text.split('\n')
            found_prefix = False

            for line in lines:
                if prefix in line:
                    found_prefix = True
                    # Check if there's content on the same line after the prefix
                    after_prefix = line.split(prefix)[1].strip()
                    # Only add if it's not a placeholder or empty
                    if (after_prefix and
                        not after_prefix.startswith('[') and
                        after_prefix != '[List any gaps]' and
                        not after_prefix.lower().startswith(('none', 'severity:', 'confidence:', 'logic issues:'))):
                        items.append(after_prefix)
                    continue

                if found_prefix:
                    line = line.strip()
                    # Stop if we hit another section, score line, or status line
                    if (line.startswith(('###', '##')) or
                        'Score:' in line or
                        'Status:' in line or
                        'Severity:' in line or
                        'Confidence:' in line or
                        (not line and len(items) > 0)):
                        break
                    # Add list items (lines starting with -, •, or numbers)
                    if line.startswith(('-', '•', '1.', '2.', '3.', '4.', '5.')):
                        item = line.lstrip('-•123456789. ')
                        if not item.lower().startswith(('none', 'severity:', 'confidence:')):
                            items.append(item)
                    elif (line and
                          not line.startswith('[') and
                          not line.lower().startswith(('none', 'severity:', 'confidence:', 'logic issues:'))):
                        items.append(line)

            # Filter out invalid items
            valid_items = []
            for item in items:
                if (item and
                    item.strip() and
                    item != '[List any gaps]' and
                    not item.lower().startswith(('none', 'severity:', 'confidence:', 'logic issues:'))):
                    valid_items.append(item)

            return valid_items
        except Exception:
            return []
    async def generate_quality_report(self, validation_result: ValidationResult,
                                     content: str) -> str:
        """Generate automated quality issue identification and reporting."""

        report_sections = []

        # Header
        report_sections.append("# VALIDATION QUALITY REPORT")
        report_sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append(f"Execution Time: {validation_result.execution_time:.2f}s")
        report_sections.append(f"Cost Estimate: ${validation_result.cost_estimate:.4f}")
        report_sections.append("")

        # Overall Assessment
        status = "✅ VALID" if validation_result.is_valid else "❌ INVALID"
        report_sections.append(f"## Overall Status: {status}")
        report_sections.append(f"**Quality Score:** {validation_result.quality_score:.2f}/1.0")
        report_sections.append(f"**Confidence Level:** {validation_result.confidence_level.upper()}")
        report_sections.append("")

        # Detailed Scores
        report_sections.append("## Detailed Assessment")
        report_sections.append(f"- **Evidence Traceability:** {validation_result.evidence_traceability_score:.2f}/1.0")
        report_sections.append(f"- **Hallucination Detection:** {'⚠️ DETECTED' if validation_result.hallucination_detected else '✅ CLEAN'}")
        report_sections.append(f"- **Logical Consistency:** {validation_result.logical_consistency_score:.2f}/1.0")
        report_sections.append("")

        # Issues Identified
        if validation_result.issues_identified:
            report_sections.append("## Issues Identified")
            for i, issue in enumerate(validation_result.issues_identified, 1):
                report_sections.append(f"{i}. {issue}")
            report_sections.append("")

        # Recommendations
        if validation_result.recommendations:
            report_sections.append("## Recommendations")
            for i, recommendation in enumerate(validation_result.recommendations, 1):
                report_sections.append(f"{i}. {recommendation}")
            report_sections.append("")

        # Content Analysis Summary
        word_count = len(content.split()) if content else 0
        citation_count = content.count("[Source:") if content else 0

        report_sections.append("## Content Analysis Summary")
        report_sections.append(f"- **Word Count:** {word_count}")
        report_sections.append(f"- **Citations Found:** {citation_count}")
        report_sections.append(f"- **Citation Density:** {(citation_count/max(word_count/100, 1)):.1f} per 100 words")

        return "\n".join(report_sections)

    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation service configuration and status."""
        return {
            "service": "ValidationStageService",
            "model_used": "openai/gpt-5-nano",
            "quality_threshold": self.quality_threshold,
            "evidence_threshold": self.evidence_threshold,
            "consistency_threshold": self.consistency_threshold,
            "tri_model_router_available": bool(self.tri_model_router),
            "capabilities": [
                "evidence_traceability_verification",
                "hallucination_detection",
                "logical_consistency_checking",
                "quality_scoring",
                "automated_issue_identification",
                "quality_reporting"
            ]
        }
