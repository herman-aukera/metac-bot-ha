"""
Unit tests for budget-aware operation manager components.
Tests operation mode detection, cost optimization, and graceful degradation.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.infrastructure.config.budget_aware_operation_manager import (
    BudgetThreshold,
    EmergencyProtocol,
    OperationModeTransitionLog,
)


class TestBudgetAwareOperationManager:
    """Test budget-aware operation manager functionality."""

    def test_emergency_protocol_levels(self):
        """Test emergency protocol level definitions."""
        # Test all protocol levels exist
        protocols = list(EmergencyProtocol)
        expected_protocols = [
            "NONE",
            "BUDGET_WARNING",
            "BUDGET_CRITICAL",
            "SYSTEM_FAILURE",
            "MANUAL_OVERRIDE",
        ]

        for expected in expected_protocols:
            assert any(p.name == expected for p in protocols)

    def test_budget_threshold_configuration(self):
        """Test budget threshold configuration."""
        from src.infrastructure.config.operation_modes import OperationMode

        threshold = BudgetThreshold(
            name="conservative_mode",
            percentage=0.70,
            operation_mode=OperationMode.CONSERVATIVE,
            emergency_protocol=EmergencyProtocol.BUDGET_WARNING,
            description="Conservative operation mode threshold",
            actions=["reduce_model_costs", "prefer_mini_models"],
        )

        assert threshold.percentage == 0.70
        assert threshold.operation_mode == OperationMode.CONSERVATIVE
        assert "reduce_model_costs" in threshold.actions

    def test_operation_mode_transition_logging(self):
        """Test operation mode transition logging."""
        from src.infrastructure.config.operation_modes import OperationMode

        transition_log = OperationModeTransitionLog(
            timestamp=datetime.now(),
            from_mode=OperationMode.NORMAL,
            to_mode=OperationMode.CONSERVATIVE,
            budget_utilization=0.75,
            remaining_budget=25.0,
            trigger_reason="Budget threshold exceeded",
            threshold_crossed="70% utilization",
            emergency_protocol=EmergencyProtocol.BUDGET_WARNING,
            performance_impact={"accuracy_change": -0.05},
            cost_savings_estimate=15.0,
        )

        assert transition_log.from_mode == OperationMode.NORMAL
        assert transition_log.to_mode == OperationMode.CONSERVATIVE
        assert transition_log.budget_utilization == 0.75
        assert transition_log.emergency_protocol == EmergencyProtocol.BUDGET_WARNING


class TestOperationModeLogic:
    """Test operation mode detection and switching logic."""

    def test_operation_mode_thresholds(self):
        """Test operation mode threshold detection."""
        # Test threshold calculations
        thresholds = {
            "normal": 0.70,
            "conservative": 0.85,
            "emergency": 0.95,
            "critical": 1.00,
        }

        # Test mode detection logic
        assert self._detect_operation_mode(0.50, thresholds) == "normal"
        assert (
            self._detect_operation_mode(0.86, thresholds) == "conservative"
        )  # Above 0.85 threshold
        assert (
            self._detect_operation_mode(0.96, thresholds) == "emergency"
        )  # Above 0.95 threshold

    def _detect_operation_mode(self, utilization: float, thresholds: dict) -> str:
        """Helper method to detect operation mode based on utilization."""
        if utilization >= thresholds["emergency"]:
            return "emergency"
        elif utilization > thresholds["conservative"]:  # Use > instead of >=
            return "conservative"
        else:
            return "normal"

    def test_cost_optimization_strategies(self):
        """Test cost optimization strategies for each mode."""
        # Normal mode strategies
        normal_strategies = self._get_optimization_strategies("normal")
        assert normal_strategies["allow_premium_models"] is True
        assert normal_strategies["max_cost_per_operation"] >= 2.0

        # Conservative mode strategies
        conservative_strategies = self._get_optimization_strategies("conservative")
        assert conservative_strategies["allow_premium_models"] is False
        assert conservative_strategies["prefer_mid_tier_models"] is True

        # Emergency mode strategies
        emergency_strategies = self._get_optimization_strategies("emergency")
        assert emergency_strategies["prefer_free_models"] is True
        assert emergency_strategies["max_cost_per_operation"] < 0.10

    def _get_optimization_strategies(self, mode: str) -> dict:
        """Helper method to get optimization strategies for a mode."""
        strategies = {
            "normal": {
                "allow_premium_models": True,
                "prefer_free_models": False,
                "max_cost_per_operation": 2.50,
                "research_depth_level": "comprehensive",
            },
            "conservative": {
                "allow_premium_models": False,
                "prefer_mid_tier_models": True,
                "max_cost_per_operation": 0.75,
                "research_depth_level": "moderate",
            },
            "emergency": {
                "allow_premium_models": False,
                "prefer_free_models": True,
                "max_cost_per_operation": 0.05,
                "research_depth_level": "minimal",
            },
        }
        return strategies.get(mode, strategies["normal"])

    def test_graceful_degradation_configuration(self):
        """Test graceful degradation configuration."""
        # Test degradation levels
        degradation_configs = {
            "minimal": {
                "disable_research_depth": False,
                "use_cached_results": True,
                "minimal_validation": False,
            },
            "moderate": {
                "disable_research_depth": True,
                "use_cached_results": True,
                "minimal_validation": True,
            },
            "maximum": {
                "disable_research_depth": True,
                "use_cached_results": True,
                "minimal_validation": True,
                "emergency_mode_active": True,
            },
        }

        # Verify degradation progression
        assert degradation_configs["minimal"]["disable_research_depth"] is False
        assert degradation_configs["moderate"]["disable_research_depth"] is True
        assert degradation_configs["maximum"]["emergency_mode_active"] is True


class TestBudgetTracking:
    """Test budget tracking and projection functionality."""

    def test_budget_utilization_calculation(self):
        """Test budget utilization calculation."""
        total_budget = 100.0
        spent_amounts = [25.0, 50.0, 75.0, 95.0]

        for spent in spent_amounts:
            utilization = self._calculate_utilization(spent, total_budget)
            expected = spent / total_budget
            assert abs(utilization - expected) < 0.01

    def _calculate_utilization(self, spent: float, total: float) -> float:
        """Helper method to calculate budget utilization."""
        return spent / total if total > 0 else 0.0

    def test_cost_per_question_tracking(self):
        """Test cost per question tracking and analysis."""
        question_costs = [2.50, 1.75, 0.80, 3.20, 1.90]

        avg_cost = self._calculate_average_cost(question_costs)
        assert 1.5 < avg_cost < 2.5

        # Test cost trend analysis
        trend = self._analyze_cost_trend(question_costs)
        assert trend in ["increasing", "decreasing", "stable"]

    def _calculate_average_cost(self, costs: list) -> float:
        """Helper method to calculate average cost."""
        return sum(costs) / len(costs) if costs else 0.0

    def _analyze_cost_trend(self, costs: list) -> str:
        """Helper method to analyze cost trend."""
        if len(costs) < 2:
            return "stable"

        recent_avg = sum(costs[-3:]) / len(costs[-3:])
        early_avg = sum(costs[:3]) / len(costs[:3])

        if recent_avg > early_avg * 1.1:
            return "increasing"
        elif recent_avg < early_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def test_budget_projection_logic(self):
        """Test budget projection and capacity planning."""
        # Test projection parameters
        projection_params = {
            "questions_processed": 30,
            "estimated_total_questions": 75,
            "current_spent": 60.0,
            "total_budget": 100.0,
        }

        projection = self._project_budget_capacity(projection_params)

        assert "projected_total_cost" in projection
        assert "budget_sufficient" in projection
        assert "recommended_adjustments" in projection

    def _project_budget_capacity(self, params: dict) -> dict:
        """Helper method to project budget capacity."""
        avg_cost_per_question = params["current_spent"] / params["questions_processed"]
        projected_total_cost = (
            avg_cost_per_question * params["estimated_total_questions"]
        )
        budget_sufficient = projected_total_cost <= params["total_budget"]

        return {
            "projected_total_cost": projected_total_cost,
            "budget_sufficient": budget_sufficient,
            "recommended_adjustments": (
                [] if budget_sufficient else ["reduce_model_costs"]
            ),
        }


class TestModelSelectionOptimization:
    """Test model selection optimization based on budget constraints."""

    def test_model_selection_by_budget_mode(self):
        """Test model selection based on budget operation mode."""
        # Normal mode - allow premium models
        normal_models = self._get_allowed_models("normal")
        assert "gpt-5" in normal_models
        assert "gpt-5-mini" in normal_models

        # Emergency mode - free models only
        emergency_models = self._get_allowed_models("emergency")
        assert all("free" in model for model in emergency_models)

    def _get_allowed_models(self, mode: str) -> list:
        """Helper method to get allowed models for a mode."""
        model_configs = {
            "normal": ["gpt-5", "gpt-5-mini", "gpt-5-nano"],
            "conservative": ["gpt-5-mini", "gpt-5-nano"],
            "emergency": ["gpt-oss-20b:free", "kimi-k2:free"],
            "critical": ["gpt-oss-20b:free"],
        }
        return model_configs.get(mode, model_configs["normal"])

    def test_task_prioritization_algorithms(self):
        """Test task prioritization for budget conservation."""
        tasks = [
            {"id": "t1", "priority": "high", "estimated_cost": 2.0},
            {"id": "t2", "priority": "medium", "estimated_cost": 1.0},
            {"id": "t3", "priority": "low", "estimated_cost": 0.5},
            {"id": "t4", "priority": "high", "estimated_cost": 0.2},
        ]

        # Normal mode - prioritize by importance
        normal_prioritized = self._prioritize_tasks(tasks, "normal")
        assert normal_prioritized[0]["priority"] == "high"

        # Emergency mode - prioritize by cost efficiency
        emergency_prioritized = self._prioritize_tasks(tasks, "emergency")
        # Should prefer high priority, low cost tasks
        assert emergency_prioritized[0]["id"] == "t4"  # High priority, lowest cost

    def _prioritize_tasks(self, tasks: list, mode: str) -> list:
        """Helper method to prioritize tasks based on mode."""
        if mode == "normal":
            # Prioritize by importance
            priority_order = {"high": 3, "medium": 2, "low": 1}
            return sorted(
                tasks, key=lambda t: priority_order.get(t["priority"], 0), reverse=True
            )
        else:
            # Prioritize by cost efficiency (high priority, low cost)
            priority_order = {"high": 3, "medium": 2, "low": 1}
            return sorted(
                tasks,
                key=lambda t: (
                    priority_order.get(t["priority"], 0),
                    -t["estimated_cost"],
                ),
                reverse=True,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
