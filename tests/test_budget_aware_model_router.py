"""
Tests for Budget-Aware Model Router

Ensures proper model selection based on budget constraints and testing environment.
"""

import os
import pytest
from unittest.mock import patch

from src.infrastructure.model_router import (
    BudgetAwareModelRouter,
    get_model_router,
    select_model_for_task,
)


class TestBudgetAwareModelRouter:
    """Test budget-aware model routing functionality."""

    def test_router_initialization(self):
        """Test router initializes with correct defaults."""
        router = BudgetAwareModelRouter()
        assert router.budget_limit == 100.0
        assert router.current_spend == 0.0

    def test_budget_utilization_calculation(self):
        """Test budget utilization calculation."""
        router = BudgetAwareModelRouter(budget_limit=100.0)
        router.current_spend = 50.0

        utilization = router.get_budget_utilization()
        assert utilization == 50.0

    def test_test_environment_detection(self):
        """Test detection of test/CI environments."""
        router = BudgetAwareModelRouter()

        # Test CI environment detection
        with patch.dict(os.environ, {"CI": "true"}):
            assert router.is_test_environment() is True

        # Test pytest environment detection
        with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_something"}):
            assert router.is_test_environment() is True

        # Test GitHub Actions detection
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            assert router.is_test_environment() is True

        # Test normal environment
        with patch.dict(os.environ, {}, clear=True):
            assert router.is_test_environment() is False

    def test_model_selection_in_test_environment(self):
        """Test that free models are used in test environments."""
        router = BudgetAwareModelRouter()

        with patch.dict(os.environ, {"CI": "true"}):
            model = router.select_model("forecasting", requires_tools=True)
            assert model == "openai/gpt-oss-120b:free"

            model = router.select_model("research", requires_tools=False)
            assert model == "openai/gpt-oss-120b:free"

    def test_model_selection_critical_budget(self):
        """Test model selection when budget is critical (>95%)."""
        router = BudgetAwareModelRouter(budget_limit=100.0)
        router.current_spend = 96.0  # 96% utilization

        # Should use free models only
        model = router.select_model("forecasting", requires_tools=True, is_test=False)
        assert model == "openai/gpt-oss-120b:free"

        model = router.select_model("research", requires_tools=False, is_test=False)
        assert model == "moonshotai/kimi-k2:free"

    def test_model_selection_high_budget(self):
        """Test model selection when budget is high (>80%)."""
        router = BudgetAwareModelRouter(budget_limit=100.0)
        router.current_spend = 85.0  # 85% utilization

        # Should use budget-conscious models
        model = router.select_model("forecasting", requires_tools=True, is_test=False)
        assert model == "gpt-4o-mini"

    def test_model_selection_medium_budget(self):
        """Test model selection when budget is medium (>50%)."""
        router = BudgetAwareModelRouter(budget_limit=100.0)
        router.current_spend = 60.0  # 60% utilization

        # Should use balanced approach
        research_model = router.select_model(
            "research", requires_tools=False, is_test=False
        )
        assert research_model == "gpt-4o-mini"

        forecast_model = router.select_model(
            "forecasting", requires_tools=False, is_test=False
        )
        assert forecast_model == "gpt-4o"

    def test_model_selection_low_budget(self):
        """Test model selection when budget is low (<50%)."""
        router = BudgetAwareModelRouter(budget_limit=100.0)
        router.current_spend = 30.0  # 30% utilization

        # Should allow premium models for forecasting
        forecast_model = router.select_model(
            "forecasting", requires_tools=False, is_test=False
        )
        assert forecast_model == "gpt-4o"

        research_model = router.select_model(
            "research", requires_tools=False, is_test=False
        )
        assert research_model == "gpt-4o-mini"

    def test_model_config_retrieval(self):
        """Test model configuration retrieval."""
        router = BudgetAwareModelRouter()

        config = router.get_model_config("gpt-4o")
        assert config["provider"] == "openrouter"
        assert config["model"] == "openai/gpt-4o"
        assert config["supports_tools"] is True
        assert config["cost_per_1k_tokens"] > 0

        free_config = router.get_model_config("openai/gpt-oss-120b:free")
        assert free_config["cost_per_1k_tokens"] == 0.0

    def test_cost_estimation(self):
        """Test cost estimation for models."""
        router = BudgetAwareModelRouter()

        # Test premium model cost
        cost = router.estimate_cost("gpt-4o", 1000)
        assert cost > 0

        # Test free model cost
        free_cost = router.estimate_cost("openai/gpt-oss-120b:free", 1000)
        assert free_cost == 0.0

    def test_affordability_check(self):
        """Test affordability checking."""
        router = BudgetAwareModelRouter(budget_limit=100.0)
        router.current_spend = 95.0  # Only $5 remaining

        # Should not be able to afford expensive model
        can_afford_expensive = router.can_afford_model(
            "gpt-4o", 500000
        )  # ~$7.50, more than $5 remaining
        assert can_afford_expensive is False

        # Should be able to afford free model
        can_afford_free = router.can_afford_model("openai/gpt-oss-120b:free", 10000)
        assert can_afford_free is True

    def test_fallback_model_selection(self):
        """Test fallback model selection."""
        router = BudgetAwareModelRouter()

        fallback_with_tools = router.get_fallback_model("gpt-4o", requires_tools=True)
        assert fallback_with_tools == "openai/gpt-oss-120b:free"

        fallback_without_tools = router.get_fallback_model(
            "gpt-4o", requires_tools=False
        )
        assert fallback_without_tools == "moonshotai/kimi-k2:free"

    def test_model_selection_with_fallback(self):
        """Test model selection with automatic fallback."""
        router = BudgetAwareModelRouter(budget_limit=100.0)
        router.current_spend = 99.0  # Only $1 remaining

        # Should fall back to free model even if primary selection would be premium
        model = router.select_model_with_fallback(
            "forecasting",
            requires_tools=True,
            estimated_tokens=10000,  # Would cost ~$0.15 for gpt-4o
            is_test=False,
        )
        assert model == "openai/gpt-oss-120b:free"

    def test_model_usage_logging(self):
        """Test model usage logging."""
        router = BudgetAwareModelRouter(budget_limit=100.0)
        initial_spend = router.current_spend

        router.log_model_usage("gpt-4o-mini", 1000, actual_cost=0.0015)

        assert router.current_spend == initial_spend + 0.0015

    def test_budget_status_reporting(self):
        """Test comprehensive budget status reporting."""
        router = BudgetAwareModelRouter(budget_limit=100.0)
        router.current_spend = 85.0

        status = router.get_budget_status()

        assert status["budget_limit"] == 100.0
        assert status["current_spend"] == 85.0
        assert status["remaining_budget"] == 15.0
        assert status["utilization_percent"] == 85.0
        assert status["operation_mode"] == "emergency"
        assert "recommended_models" in status

    def test_environment_variable_loading(self):
        """Test loading budget status from environment variables."""
        with patch.dict(os.environ, {"BUDGET_LIMIT": "200.0", "CURRENT_SPEND": "50.0"}):
            router = BudgetAwareModelRouter()
            router.load_budget_status()

            assert router.budget_limit == 200.0
            assert router.current_spend == 50.0

    def test_global_router_instance(self):
        """Test global router instance functionality."""
        router1 = get_model_router()
        router2 = get_model_router()

        # Should return the same instance
        assert router1 is router2

    def test_convenience_function(self):
        """Test convenience function for model selection."""
        with patch.dict(os.environ, {"CI": "true"}):
            model = select_model_for_task("forecasting", requires_tools=True)
            assert model == "openai/gpt-oss-120b:free"

    def test_operation_mode_determination(self):
        """Test operation mode determination based on utilization."""
        router = BudgetAwareModelRouter()

        assert router._get_operation_mode(30.0) == "normal"
        assert router._get_operation_mode(60.0) == "conservative"
        assert router._get_operation_mode(85.0) == "emergency"
        assert router._get_operation_mode(96.0) == "critical"

    def test_recommended_models_by_utilization(self):
        """Test recommended models change based on utilization."""
        router = BudgetAwareModelRouter()

        # Normal utilization - mixed models
        normal_recs = router._get_recommended_models(30.0)
        assert normal_recs["forecasting"] == "gpt-4o"

        # High utilization - budget models only
        high_recs = router._get_recommended_models(85.0)
        assert high_recs["forecasting"] == "gpt-4o-mini"

        # Critical utilization - free models only
        critical_recs = router._get_recommended_models(96.0)
        assert critical_recs["forecasting"] == "openai/gpt-oss-120b:free"

    @pytest.mark.parametrize(
        "utilization,expected_mode",
        [
            (0.0, "normal"),
            (30.0, "normal"),
            (60.0, "conservative"),
            (85.0, "emergency"),
            (96.0, "critical"),
            (99.0, "critical"),
        ],
    )
    def test_operation_modes(self, utilization, expected_mode):
        """Test operation mode determination for various utilization levels."""
        router = BudgetAwareModelRouter()
        mode = router._get_operation_mode(utilization)
        assert mode == expected_mode
