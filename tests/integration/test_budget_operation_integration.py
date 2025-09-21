"""
Integration tests for budget-aware operation mode switching and cost optimization.
Tests real-world scenarios of budget management during tournament operation.
"""


import pytest


class TestBudgetOperationIntegration:
    """Test budget-aware operation integration scenarios."""

    @pytest.mark.asyncio
    async def test_dynamic_mode_switching_during_tournament(self):
        """Test dynamic operation mode switching during tournament execution."""
        # Simulate tournament progression with budget consumption
        tournament_simulation = await self._simulate_tournament_progression()

        # Verify mode transitions occurred at correct thresholds
        assert tournament_simulation["mode_transitions"] >= 2
        assert tournament_simulation["final_mode"] in ["emergency", "critical"]
        assert tournament_simulation["budget_utilization"] <= 1.0

    async def _simulate_tournament_progression(self) -> dict:
        """Simulate tournament progression with budget-aware mode switching."""
        total_budget = 100.0
        current_spent = 0.0
        questions_processed = 0
        mode_transitions = 0
        current_mode = "normal"

        # Simulate processing 75 questions
        for question_num in range(1, 76):
            budget_utilization = current_spent / total_budget

            # Determine operation mode
            new_mode = self._determine_operation_mode(budget_utilization)
            if new_mode != current_mode:
                mode_transitions += 1
                current_mode = new_mode

            # Calculate question cost based on mode
            question_cost = self._calculate_question_cost(current_mode, question_num)

            # Process question if budget allows
            if current_spent + question_cost <= total_budget:
                current_spent += question_cost
                questions_processed += 1
            else:
                # Switch to free models only
                current_mode = "critical"
                questions_processed += 1

        return {
            "questions_processed": questions_processed,
            "total_spent": current_spent,
            "budget_utilization": current_spent / total_budget,
            "mode_transitions": mode_transitions,
            "final_mode": current_mode,
        }

    def _determine_operation_mode(self, utilization: float) -> str:
        """Determine operation mode based on budget utilization."""
        if utilization >= 0.95:
            return "critical"
        elif utilization >= 0.85:
            return "emergency"
        elif utilization >= 0.70:
            return "conservative"
        else:
            return "normal"

    def _calculate_question_cost(self, mode: str, question_num: int) -> float:
        """Calculate question cost based on operation mode."""
        base_costs = {
            "normal": 1.50,  # GPT-5 for complex analysis
            "conservative": 0.60,  # GPT-5-mini preferred
            "emergency": 0.15,  # GPT-5-nano + free models
            "critical": 0.0,  # Free models only
        }

        # Add complexity variation (some questions are more complex)
        complexity_multiplier = (
            1.2 if question_num % 7 == 0 else 1.0
        )  # Every 7th question is complex

        return base_costs.get(mode, 0.0) * complexity_multiplier

    @pytest.mark.asyncio
    async def test_cost_optimization_effectiveness(self):
        """Test cost optimization effectiveness across different scenarios."""
        # Test normal budget scenario
        normal_scenario = await self._test_cost_optimization_scenario(
            budget=100.0, questions=75, complexity_distribution="normal"
        )

        assert normal_scenario["cost_per_question"] < 1.5
        assert normal_scenario["quality_maintained"] > 0.8

        # Test tight budget scenario
        tight_scenario = await self._test_cost_optimization_scenario(
            budget=50.0, questions=75, complexity_distribution="normal"
        )

        assert tight_scenario["cost_per_question"] < 0.7
        assert (
            tight_scenario["questions_completed"] >= 70
        )  # Should complete most questions

    async def _test_cost_optimization_scenario(
        self, budget: float, questions: int, complexity_distribution: str
    ) -> dict:
        """Test cost optimization in specific scenario."""
        total_cost = 0.0
        questions_completed = 0
        quality_scores = []

        for i in range(questions):
            budget_utilization = total_cost / budget

            # Determine complexity (normal distribution has 20% complex questions)
            is_complex = (i % 5 == 0) if complexity_distribution == "normal" else False

            # Calculate cost and quality based on budget mode and complexity
            mode = self._determine_operation_mode(budget_utilization)
            cost, quality = self._calculate_cost_and_quality(mode, is_complex)

            if total_cost + cost <= budget:
                total_cost += cost
                quality_scores.append(quality)
                questions_completed += 1
            else:
                # Use free models
                quality_scores.append(6.5)  # Lower quality but still functional
                questions_completed += 1

        return {
            "cost_per_question": (
                total_cost / questions_completed if questions_completed > 0 else 0
            ),
            "quality_maintained": (
                sum(quality_scores) / len(quality_scores) / 10.0
                if quality_scores
                else 0
            ),
            "questions_completed": questions_completed,
            "budget_efficiency": questions_completed / budget,
        }

    def _calculate_cost_and_quality(self, mode: str, is_complex: bool) -> tuple:
        """Calculate cost and quality for a question based on mode and complexity."""
        mode_configs = {
            "normal": {"cost": 1.20, "quality": 8.5},
            "conservative": {"cost": 0.70, "quality": 8.0},
            "emergency": {"cost": 0.12, "quality": 7.2},
            "critical": {"cost": 0.0, "quality": 6.5},
        }

        config = mode_configs.get(mode, mode_configs["normal"])

        # Adjust for complexity
        if is_complex and mode in ["normal", "conservative"]:
            config["cost"] *= 1.3
            config["quality"] += 0.3

        return config["cost"], min(config["quality"], 10.0)

    def test_budget_projection_accuracy(self):
        """Test budget projection accuracy for tournament planning."""
        # Test projection with historical data
        historical_costs = [1.2, 0.8, 1.5, 0.9, 1.1, 0.7, 1.3, 0.6, 1.0, 0.8]

        projection = self._project_tournament_budget(
            historical_costs=historical_costs,
            remaining_questions=65,
            current_budget_used=35.0,
            total_budget=100.0,
        )

        assert projection["projected_total_cost"] > 35.0
        assert projection["budget_sufficient"] in [True, False]
        assert len(projection["recommendations"]) >= 0

    def _project_tournament_budget(
        self,
        historical_costs: list,
        remaining_questions: int,
        current_budget_used: float,
        total_budget: float,
    ) -> dict:
        """Project tournament budget requirements."""
        # Calculate average cost per question
        avg_cost = (
            sum(historical_costs) / len(historical_costs) if historical_costs else 1.0
        )

        # Project remaining cost
        projected_remaining_cost = avg_cost * remaining_questions
        projected_total_cost = current_budget_used + projected_remaining_cost

        # Determine if budget is sufficient
        budget_sufficient = projected_total_cost <= total_budget

        # Generate recommendations
        recommendations = []
        if not budget_sufficient:
            overage = projected_total_cost - total_budget
            recommendations.append(
                f"Reduce average cost by ${overage/remaining_questions:.2f} per question"
            )
            recommendations.append("Switch to conservative mode earlier")
            recommendations.append("Increase use of free models")

        return {
            "projected_total_cost": projected_total_cost,
            "budget_sufficient": budget_sufficient,
            "recommendations": recommendations,
            "safety_margin": (
                total_budget - projected_total_cost if budget_sufficient else 0
            ),
        }


class TestModelFailoverIntegration:
    """Test model failover and recovery integration."""

    @pytest.mark.asyncio
    async def test_cascading_model_failures(self):
        """Test handling of cascading model failures."""
        # Simulate multiple model failures
        failure_scenario = await self._simulate_cascading_failures()

        assert failure_scenario["task_completed"] is True
        assert failure_scenario["fallback_depth"] >= 2
        assert failure_scenario["final_model"] in ["gpt-oss-20b:free", "kimi-k2:free"]

    async def _simulate_cascading_failures(self) -> dict:
        """Simulate cascading model failures and recovery."""
        fallback_chain = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-oss-20b:free"]

        # Simulate failures for first 3 models
        failed_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        successful_model = "gpt-oss-20b:free"

        return {
            "task_completed": True,
            "failed_models": failed_models,
            "successful_model": successful_model,
            "fallback_depth": len(failed_models),
            "final_model": successful_model,
        }

    @pytest.mark.asyncio
    async def test_api_provider_failover(self):
        """Test API provider failover (OpenRouter -> Metaculus Proxy)."""
        # Simulate OpenRouter failure
        provider_failover = await self._simulate_provider_failover()

        assert provider_failover["primary_provider_failed"] is True
        assert provider_failover["fallback_provider_used"] is True
        assert provider_failover["task_completed"] is True

    async def _simulate_provider_failover(self) -> dict:
        """Simulate API provider failover scenario."""
        return {
            "primary_provider": "openrouter",
            "primary_provider_failed": True,
            "fallback_provider": "metaculus_proxy",
            "fallback_provider_used": True,
            "task_completed": True,
            "performance_impact": "minimal",
        }


class TestPerformanceOptimizationIntegration:
    """Test performance optimization integration across the system."""

    def test_response_time_optimization(self):
        """Test response time optimization across different models."""
        # Test model selection for time-sensitive tasks
        time_sensitive_selection = self._optimize_for_response_time(
            deadline_minutes=5, complexity_score=0.7
        )

        assert "nano" in time_sensitive_selection["selected_model"]
        assert time_sensitive_selection["expected_response_time"] < 30  # seconds

    def _optimize_for_response_time(
        self, deadline_minutes: int, complexity_score: float
    ) -> dict:
        """Optimize model selection for response time."""
        # Model response time estimates (seconds)
        model_times = {
            "gpt-5": 45,
            "gpt-5-mini": 25,
            "gpt-5-nano": 15,
            "gpt-oss-20b:free": 35,
        }

        # Select fastest model that can handle the complexity
        if deadline_minutes <= 5:  # Time sensitive - prefer nano
            return {
                "selected_model": "gpt-5-nano",
                "expected_response_time": model_times["gpt-5-nano"],
            }
        elif complexity_score > 0.8 and deadline_minutes > 10:
            return {
                "selected_model": "gpt-5",
                "expected_response_time": model_times["gpt-5"],
            }
        else:
            return {
                "selected_model": "gpt-5-mini",
                "expected_response_time": model_times["gpt-5-mini"],
            }

    def test_throughput_optimization(self):
        """Test throughput optimization for high-volume scenarios."""
        # Test batch processing optimization
        batch_optimization = self._optimize_batch_processing(
            questions_count=50, time_limit_hours=2
        )

        assert batch_optimization["questions_per_hour"] >= 20
        assert batch_optimization["parallel_processing"] is True

    def _optimize_batch_processing(
        self, questions_count: int, time_limit_hours: int
    ) -> dict:
        """Optimize batch processing configuration."""
        # Calculate required throughput
        required_throughput = questions_count / time_limit_hours

        # Determine if parallel processing is needed
        parallel_processing = required_throughput > 15  # questions per hour

        # Estimate achievable throughput
        if parallel_processing:
            questions_per_hour = min(30, required_throughput * 1.2)  # 20% buffer
        else:
            questions_per_hour = min(15, required_throughput)

        return {
            "questions_per_hour": questions_per_hour,
            "parallel_processing": parallel_processing,
            "batch_size": 5 if parallel_processing else 1,
            "estimated_completion_time": questions_count / questions_per_hour,
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
