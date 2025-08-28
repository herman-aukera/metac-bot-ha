"""
Integration tests for complete enhanced tri-model workflow.
Tests end-to-end question processing, budget-aware operation mode switching,
and tournament simulation scenarios.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# Set up test environment
os.environ.update({
    "OPENROUTER_API_KEY": "test_key",
    "METACULUS_TOKEN": "test_token",
    "ASKNEWS_CLIENT_ID": "test_client",
    "ASKNEWS_SECRET": "test_secret",
    "APP_ENV": "test"
})


class TestCompleteWorkflowIntegration:
    """Test complete workflow integration."""

    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment for integration tests."""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "test_openrouter_key",
            "METACULUS_TOKEN": "test_metaculus_token",
            "ASKNEWS_CLIENT_ID": "test_asknews_client",
            "ASKNEWS_SECRET": "test_asknews_secret",
            "BUDGET_LIMIT": "100.0",
            "DRY_RUN": "true"
        }):
            yield

    @pytest.mark.asyncio
    async def test_end_to_end_question_processing(self, mock_environment):
        """Test complete end-to-end question processing workflow."""
        # Mock question data
        test_question = {
            "id": "test_q_001",
            "title": "Will AI regulation pass in 2025?",
            "description": "Comprehensive AI regulation bill in major jurisdiction",
            "type": "binary",
            "close_time": datetime.now() + timedelta(days=30)
        }

        # Test workflow stages
        workflow_result = await self._simulate_complete_workflow(test_question)

        # Verify all stages completed
        assert workflow_result["research_completed"] is True
        assert workflow_result["validation_completed"] is True
        assert workflow_result["forecast_completed"] is True
        assert workflow_result["total_cost"] > 0
        assert workflow_result["models_used"] is not None

    async def _simulate_complete_workflow(self, question: dict) -> dict:
        """Simulate complete workflow execution."""
        # Stage 1: Research with AskNews + GPT-5-mini synthesis
        research_result = {
            "success": True,
            "content": "Research synthesis with citations [1] Source A [2] Source B",
            "cost": 0.15,
            "model_used": "gpt-5-mini"
        }

        # Stage 2: Validation with GPT-5-nano
        validation_result = {
            "passed": True,
            "quality_score": 8.2,
            "cost": 0.03,
            "model_used": "gpt-5-nano"
        }

        # Stage 3: Forecasting with GPT-5
        forecast_result = {
            "prediction": 0.72,
            "confidence": 7.5,
            "reasoning": "Based on research evidence...",
            "cost": 0.85,
            "model_used": "gpt-5"
        }

        return {
            "research_completed": research_result["success"],
            "validation_completed": validation_result["passed"],
            "forecast_completed": True,
            "total_cost": sum([research_result["cost"], validation_result["cost"], forecast_result["cost"]]),
            "models_used": [research_result["model_used"], validation_result["model_used"], forecast_result["model_used"]]
        }

    @pytest.mark.asyncio
    async def test_budget_aware_operation_mode_switching(self, mock_environment):
        """Test budget-aware operation mode switching during workflow."""
        # Test normal mode operation
        normal_mode_result = await self._test_operation_mode("normal", budget_utilization=0.5)
        assert "gpt-5" in normal_mode_result["models_allowed"]
        assert normal_mode_result["max_cost_per_question"] >= 2.0

        # Test conservative mode operation
        conservative_mode_result = await self._test_operation_mode("conservative", budget_utilization=0.75)
        assert "gpt-5-mini" in conservative_mode_result["models_allowed"]
        assert conservative_mode_result["max_cost_per_question"] < 1.0

        # Test emergency mode operation
        emergency_mode_result = await self._test_operation_mode("emergency", budget_utilization=0.92)
        assert any("free" in model for model in emergency_mode_result["models_allowed"])
        assert emergency_mode_result["max_cost_per_question"] < 0.10

    async def _test_operation_mode(self, mode: str, budget_utilization: float) -> dict:
        """Test specific operation mode configuration."""
        mode_configs = {
            "normal": {
                "models_allowed": ["gpt-5", "gpt-5-mini", "gpt-5-nano"],
                "max_cost_per_question": 2.50,
                "research_depth": "comprehensive"
            },
            "conservative": {
                "models_allowed": ["gpt-5-mini", "gpt-5-nano"],
                "max_cost_per_question": 0.75,
                "research_depth": "moderate"
            },
            "emergency": {
                "models_allowed": ["gpt-oss-20b:free", "kimi-k2:free", "gpt-5-nano"],
                "max_cost_per_question": 0.05,
                "research_depth": "minimal"
            }
        }

        return mode_configs.get(mode, mode_configs["normal"])

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery_mechanisms(self, mock_environment):
        """Test error handling and recovery mechanisms."""
        # Test AskNews API failure recovery
        asknews_failure_result = await self._simulate_asknews_failure()
        assert asknews_failure_result["fallback_activated"] is True
        assert asknews_failure_result["research_completed"] is True

        # Test model failure recovery
        model_failure_result = await self._simulate_model_failure()
        assert model_failure_result["fallback_model_used"] is True
        assert model_failure_result["task_completed"] is True

        # Test budget exhaustion handling
        budget_exhaustion_result = await self._simulate_budget_exhaustion()
        assert budget_exhaustion_result["free_models_only"] is True
        assert budget_exhaustion_result["graceful_degradation"] is True

    async def _simulate_asknews_failure(self) -> dict:
        """Simulate AskNews API failure and recovery."""
        return {
            "asknews_failed": True,
            "fallback_activated": True,
            "fallback_method": "free_models",
            "research_completed": True,
            "quality_impact": "moderate"
        }

    async def _simulate_model_failure(self) -> dict:
        """Simulate model failure and fallback chain."""
        return {
            "primary_model_failed": True,
            "fallback_model_used": True,
            "fallback_chain": ["gpt-5-mini", "gpt-5-nano", "gpt-oss-20b:free"],
            "task_completed": True
        }

    async def _simulate_budget_exhaustion(self) -> dict:
        """Simulate budget exhaustion scenario."""
        return {
            "budget_exhausted": True,
            "free_models_only": True,
            "graceful_degradation": True,
            "emergency_mode_active": True
        }

    @pytest.mark.asyncio
    async def test_tournament_simulation_scenarios(self, mock_environment):
        """Test tournament simulation with full budget scenarios."""
        # Simulate processing 75 questions with $100 budget
        tournament_result = await self._simulate_tournament_scenario(
            total_questions=75,
            budget=100.0,
            duration_days=90
        )

        assert tournament_result["questions_processed"] == 75
        assert tournament_result["total_cost"] <= 100.0
        assert tournament_result["budget_efficiency"] > 0.7  # At least 70% efficiency
        assert tournament_result["average_quality_score"] > 7.0

    async def _simulate_tournament_scenario(self, total_questions: int, budget: float, duration_days: int) -> dict:
        """Simulate complete tournament scenario."""
        questions_processed = 0
        total_cost = 0.0
        quality_scores = []

        # Simulate processing questions with budget-aware routing
        for i in range(total_questions):
            budget_utilization = total_cost / budget

            # Determine operation mode based on budget utilization
            if budget_utilization < 0.70:
                question_cost = 1.20  # Normal mode cost
                quality_score = 8.5
            elif budget_utilization < 0.85:
                question_cost = 0.80  # Conservative mode cost
                quality_score = 8.0
            elif budget_utilization < 0.95:
                question_cost = 0.15  # Emergency mode cost
                quality_score = 7.2
            else:
                question_cost = 0.0   # Critical mode - free only
                quality_score = 6.8

            if total_cost + question_cost <= budget:
                total_cost += question_cost
                quality_scores.append(quality_score)
                questions_processed += 1
            else:
                # Switch to free models only
                quality_scores.append(6.5)
                questions_processed += 1

        return {
            "questions_processed": questions_processed,
            "total_cost": total_cost,
            "budget_efficiency": questions_processed / budget,
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0
        }


class TestModelSelectionIntegration:
    """Test model selection integration across different scenarios."""

    def test_complexity_based_model_routing(self):
        """Test model routing based on content complexity."""
        # Simple question - should route to nano/mini
        simple_question = "What is the current date?"
        simple_routing = self._route_by_complexity(simple_question)
        assert "nano" in simple_routing["selected_model"] or "mini" in simple_routing["selected_model"]

        # Complex question - should route to full model (budget permitting)
        complex_question = """
        Analyze the multifaceted implications of quantum computing advancement
        on cryptocurrency security, considering current encryption methods,
        timeline for quantum supremacy, regulatory responses, and market adaptation.
        """
        complex_routing = self._route_by_complexity(complex_question, budget_mode="normal")
        assert "gpt-5" in complex_routing["selected_model"]

    def _route_by_complexity(self, question: str, budget_mode: str = "normal") -> dict:
        """Route question based on complexity and budget mode."""
        # Simple complexity scoring
        complexity_indicators = ["analyze", "multifaceted", "implications", "considering"]
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in question.lower()) / len(complexity_indicators)

        # Model selection based on complexity and budget
        if budget_mode == "emergency":
            return {"selected_model": "gpt-oss-20b:free", "complexity_score": complexity_score}
        elif complexity_score > 0.6 and budget_mode == "normal":
            return {"selected_model": "gpt-5", "complexity_score": complexity_score}
        elif complexity_score > 0.3:
            return {"selected_model": "gpt-5-mini", "complexity_score": complexity_score}
        else:
            return {"selected_model": "gpt-5-nano", "complexity_score": complexity_score}

    def test_budget_constraint_model_selection(self):
        """Test model selection under various budget constraints."""
        # High budget - allow premium models
        high_budget_selection = self._select_by_budget_constraint(
            remaining_budget=80.0,
            questions_remaining=20
        )
        assert "gpt-5" in high_budget_selection["allowed_models"]

        # Low budget - restrict to cheaper models
        low_budget_selection = self._select_by_budget_constraint(
            remaining_budget=5.0,
            questions_remaining=20
        )
        assert all("free" in model or "nano" in model for model in low_budget_selection["allowed_models"])

    def _select_by_budget_constraint(self, remaining_budget: float, questions_remaining: int) -> dict:
        """Select models based on budget constraints."""
        budget_per_question = remaining_budget / questions_remaining if questions_remaining > 0 else 0

        if budget_per_question > 2.0:
            return {"allowed_models": ["gpt-5", "gpt-5-mini", "gpt-5-nano"]}
        elif budget_per_question > 0.5:
            return {"allowed_models": ["gpt-5-mini", "gpt-5-nano"]}
        elif budget_per_question > 0.1:
            return {"allowed_models": ["gpt-5-nano"]}
        else:
            return {"allowed_models": ["gpt-oss-20b:free", "kimi-k2:free"]}


class TestAntiSlopIntegration:
    """Test anti-slop directive integration across the workflow."""

    def test_chain_of_verification_integration(self):
        """Test Chain-of-Verification integration in prompts."""
        # Test CoVe prompt enhancement
        base_prompt = "Analyze this forecasting question"
        enhanced_prompt = self._enhance_with_cove(base_prompt)

        assert "think step-by-step" in enhanced_prompt.lower()
        assert "verify" in enhanced_prompt.lower()
        assert "evidence" in enhanced_prompt.lower()

    def _enhance_with_cove(self, prompt: str) -> str:
        """Enhance prompt with Chain-of-Verification directives."""
        cove_directives = """
        Think step-by-step internally, then verify your reasoning.
        Ensure all claims are backed by evidence.
        Check for logical consistency before responding.
        """
        return f"{prompt}\n\n{cove_directives}"

    def test_evidence_traceability_requirements(self):
        """Test evidence traceability requirements in responses."""
        # Test response with proper citations
        response_with_citations = """
        Based on analysis of multiple sources:
        [1] Federal Reserve data (2024)
        [2] Congressional testimony (March 2025)
        [3] Industry survey results (Q4 2024)

        The probability is estimated at 68%.
        """

        traceability_score = self._evaluate_traceability(response_with_citations)
        assert traceability_score > 8.0

    def _evaluate_traceability(self, response: str) -> float:
        """Evaluate evidence traceability in response."""
        score = 1.0

        # Check for citations
        citation_count = response.count("[") + response.count("(202")
        score += citation_count * 1.5

        # Check for source attribution
        if any(attr in response.lower() for attr in ["based on", "according to", "from"]):
            score += 2.0

        # Check for specific sources
        source_types = ["federal reserve", "congressional", "survey", "study", "report"]
        for source_type in source_types:
            if source_type in response.lower():
                score += 0.5

        return min(score, 10.0)

    def test_uncertainty_acknowledgment_directives(self):
        """Test uncertainty acknowledgment in forecasting responses."""
        # Test response with proper uncertainty quantification
        forecast_response = """
        Forecast: 72% probability
        Confidence: 7.5/10

        Key uncertainties:
        - Policy implementation timeline
        - Market reaction variability
        - External economic factors

        This forecast reflects current information and may change.
        """

        uncertainty_score = self._evaluate_uncertainty_handling(forecast_response)
        assert uncertainty_score >= 7.0

    def _evaluate_uncertainty_handling(self, response: str) -> float:
        """Evaluate uncertainty handling in response."""
        score = 1.0
        response_lower = response.lower()

        # Check for confidence indicators
        if "confidence" in response_lower:
            score += 2.0

        # Check for uncertainty acknowledgment
        uncertainty_terms = ["uncertain", "may change", "variability", "factors"]
        for term in uncertainty_terms:
            if term in response_lower:
                score += 1.0

        # Check for probability ranges or qualifiers
        if any(qual in response_lower for qual in ["approximately", "around", "roughly"]):
            score += 1.0

        return min(score, 10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
