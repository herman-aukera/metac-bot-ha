"""
Tests for ValidationStageService implementing task 4.2 requirements.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.domain.services.validation_stage_service import ValidationStageService, ValidationResult


class TestValidationStageService:
    """Test suite for ValidationStageService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_router = Mock()
        self.mock_nano_model = AsyncMock()
        self.mock_router.models = {"nano": self.mock_nano_model}

        self.validation_service = ValidationStageService(self.mock_router)

    @pytest.mark.asyncio
    async def test_validate_content_with_good_content(self):
        """Test validation with high-quality content."""

        # Mock GPT-5-nano responses for each validation check
        self.mock_nano_model.invoke.side_effect = [
            # Evidence verification response
            """Citations Found: 5/5 claims cited
Citation Quality: GOOD
Evidence Gaps: None identified
Overall Evidence Score: 9/10
Status: PASS""",

            # Hallucination detection response
            """Potential Hallucinations: None found
Severity: LOW
Confidence in Detection: HIGH
Hallucination Risk Score: 1/10
Status: CLEAN""",

            # Consistency check response
            """Contradictions Found: None
Logic Issues: None
Consistency Score: 9/10
Status: CONSISTENT""",

            # Quality scoring response
            """Accuracy Score: 9/10
Completeness Score: 8/10
Clarity Score: 9/10
Relevance Score: 9/10
Reliability Score: 8/10
Overall Quality Score: 8.6/10
Status: EXCELLENT"""
        ]

        test_content = """
        Recent developments show significant progress [Source: TechNews, 2024-01-15].
        The implementation follows established patterns [Source: Research Paper, 2024-01-14].
        Key findings indicate positive outcomes with 85% success rate [Source: Study Results, 2024-01-15].
        """

        result = await self.validation_service.validate_content(test_content, "research_synthesis")

        assert isinstance(result, ValidationResult)
        assert result.is_valid == True
        assert result.quality_score > 0.7
        assert result.hallucination_detected == False
        assert result.confidence_level == "high"
        assert len(result.issues_identified) == 0

    @pytest.mark.asyncio
    async def test_validate_content_with_poor_content(self):
        """Test validation with low-quality content."""

        # Mock GPT-5-nano responses indicating quality issues
        self.mock_nano_model.invoke.side_effect = [
            # Evidence verification response
            """Citations Found: 1/5 claims cited
Citation Quality: POOR
Evidence Gaps: Missing sources for key claims, No recent data citations
Overall Evidence Score: 3/10
Status: FAIL""",

            # Hallucination detection response
            """Potential Hallucinations: Exact statistics without source, Specific dates that seem fabricated
Severity: HIGH
Confidence in Detection: MEDIUM
Hallucination Risk Score: 8/10
Status: PROBLEMATIC""",

            # Consistency check response
            """Contradictions Found: Timeline inconsistency between sections
Logic Issues: Causal relationship not supported
Consistency Score: 4/10
Status: MAJOR_ISSUES""",

            # Quality scoring response
            """Accuracy Score: 4/10
Completeness Score: 5/10
Clarity Score: 6/10
Relevance Score: 7/10
Reliability Score: 3/10
Overall Quality Score: 5/10
Status: FAIR"""
        ]

        test_content = """
        The system achieved 97.3% accuracy yesterday.
        Implementation was completed by the team.
        Results show improvement but data is missing.
        """

        result = await self.validation_service.validate_content(test_content, "research_synthesis")

        assert isinstance(result, ValidationResult)
        assert result.is_valid == False
        assert result.quality_score < 0.7
        assert result.hallucination_detected == True
        assert result.confidence_level == "low"
        assert len(result.issues_identified) > 0

    @pytest.mark.asyncio
    async def test_generate_quality_report(self):
        """Test quality report generation."""

        validation_result = ValidationResult(
            is_valid=False,
            quality_score=0.5,
            evidence_traceability_score=0.3,
            hallucination_detected=True,
            logical_consistency_score=0.6,
            issues_identified=["Poor evidence traceability", "Potential hallucinations detected"],
            recommendations=["Add proper citations", "Verify claims against sources"],
            confidence_level="low",
            execution_time=2.5,
            cost_estimate=0.0025
        )

        test_content = "Test content for report generation."

        report = await self.validation_service.generate_quality_report(validation_result, test_content)

        assert "VALIDATION QUALITY REPORT" in report
        assert "❌ INVALID" in report
        assert "**Quality Score:** 0.50/1.0" in report
        assert "Poor evidence traceability" in report
        assert "Add proper citations" in report
        assert "Cost Estimate: $0.0025" in report

    def test_extract_score_from_text(self):
        """Test score extraction utility method."""

        text = """
        Overall Evidence Score: 7/10
        Other content here
        """

        score = self.validation_service._extract_score_from_text(text, "Overall Evidence Score:")
        assert score == 7.0

    def test_extract_list_from_text(self):
        """Test list extraction utility method."""

        text = """
        Evidence Gaps:
        - Missing source for claim A
        - No citation for statistic B

        Next Section:
        """

        gaps = self.validation_service._extract_list_from_text(text, "Evidence Gaps:")
        assert len(gaps) == 2
        assert "Missing source for claim A" in gaps
        assert "No citation for statistic B" in gaps

    def test_get_validation_status(self):
        """Test validation service status reporting."""

        status = self.validation_service.get_validation_status()

        assert status["service"] == "ValidationStageService"
        assert status["model_used"] == "openai/gpt-5-nano"
        assert "evidence_traceability_verification" in status["capabilities"]
        assert "hallucination_detection" in status["capabilities"]
        assert "logical_consistency_checking" in status["capabilities"]


if __name__ == "__main__":
    pytest.main([__file__])
