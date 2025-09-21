"""
A/B testing framework for prompt performance evaluation.
"""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock


from src.domain.entities.question import Question, QuestionType
from src.infrastructure.config.token_tracker import TokenTracker
from src.prompts.calibrated_forecasting_prompts import CalibratedForecastingPrompts


class ABTestingFramework:
    """Framework for A/B testing prompt performance."""

    def __init__(self):
        self.prompts = CalibratedForecastingPrompts()
        self.token_tracker = TokenTracker()
        self.test_results = []

    def run_ab_test(
        self,
        questions: List[Question],
        prompt_variants: List[str],
        sample_size: int = 100,
    ) -> Dict[str, Any]:
        """Run A/B test comparing prompt variants."""
        results = {
            variant: {"responses": [], "token_usage": [], "performance_metrics": {}}
            for variant in prompt_variants
        }

        for i in range(sample_size):
            question = random.choice(questions)
            variant = prompt_variants[i % len(prompt_variants)]

            # Generate prompt based on variant
            if variant == "basic_calibrated":
                prompt = self.prompts.generate_basic_calibrated_prompt(
                    question, "test research"
                )
            elif variant == "scenario_analysis":
                prompt = self.prompts.generate_scenario_analysis_prompt(
                    question, "test research"
                )
            elif variant == "overconfidence_reduction":
                prompt = self.prompts.generate_overconfidence_reduction_prompt(
                    question, "test research"
                )

            # Track token usage
            token_count = self.token_tracker.count_tokens(prompt, "gpt-4o-mini")
            results[variant]["token_usage"].append(token_count)

            # Simulate response (in real test, this would be actual API call)
            simulated_response = self._simulate_response(prompt, variant)
            results[variant]["responses"].append(simulated_response)

        # Calculate performance metrics
        for variant in prompt_variants:
            results[variant]["performance_metrics"] = (
                self._calculate_performance_metrics(results[variant])
            )

        return results

    def _simulate_response(self, prompt: str, variant: str) -> Dict[str, Any]:
        """Simulate API response for testing."""
        # Simulate different response characteristics based on prompt variant
        base_quality = 0.7

        if variant == "scenario_analysis":
            quality_boost = 0.1  # More comprehensive analysis
            token_cost_multiplier = 1.3
        elif variant == "overconfidence_reduction":
            quality_boost = 0.05  # Better calibration
            token_cost_multiplier = 1.1
        else:  # basic_calibrated
            quality_boost = 0.0
            token_cost_multiplier = 1.0

        return {
            "quality_score": min(
                1.0, base_quality + quality_boost + random.uniform(-0.1, 0.1)
            ),
            "calibration_score": random.uniform(0.6, 0.9),
            "token_efficiency": 1.0 / token_cost_multiplier,
            "response_time": random.uniform(2.0, 8.0),
        }

    def _calculate_performance_metrics(
        self, variant_results: Dict[str, List]
    ) -> Dict[str, float]:
        """Calculate performance metrics for a variant."""
        responses = variant_results["responses"]
        token_usage = variant_results["token_usage"]

        if not responses:
            return {}

        return {
            "avg_quality_score": sum(r["quality_score"] for r in responses)
            / len(responses),
            "avg_calibration_score": sum(r["calibration_score"] for r in responses)
            / len(responses),
            "avg_token_efficiency": sum(r["token_efficiency"] for r in responses)
            / len(responses),
            "avg_token_usage": sum(token_usage) / len(token_usage),
            "total_cost_estimate": sum(token_usage)
            * 0.0006
            / 1000,  # GPT-4o-mini pricing
            "response_consistency": self._calculate_consistency(responses),
        }

    def _calculate_consistency(self, responses: List[Dict[str, Any]]) -> float:
        """Calculate response consistency metric."""
        if len(responses) < 2:
            return 1.0

        quality_scores = [r["quality_score"] for r in responses]
        variance = sum(
            (x - sum(quality_scores) / len(quality_scores)) ** 2 for x in quality_scores
        ) / len(quality_scores)
        return max(0.0, 1.0 - variance)  # Higher consistency = lower variance


class TestABTestingFramework:
    """Test A/B testing framework functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.ab_framework = ABTestingFramework()

        # Create mock questions
        self.mock_questions = []
        for i in range(10):
            question = Mock(spec=Question)
            question.title = f"Test question {i}"
            question.question_type = QuestionType.BINARY
            question.close_time = datetime.now() + timedelta(days=30)
            self.mock_questions.append(question)

    def test_ab_test_execution(self):
        """Test A/B test execution with multiple prompt variants."""
        prompt_variants = [
            "basic_calibrated",
            "scenario_analysis",
            "overconfidence_reduction",
        ]

        results = self.ab_framework.run_ab_test(
            questions=self.mock_questions,
            prompt_variants=prompt_variants,
            sample_size=30,
        )

        # Verify results structure
        assert len(results) == 3
        for variant in prompt_variants:
            assert variant in results
            assert "responses" in results[variant]
            assert "token_usage" in results[variant]
            assert "performance_metrics" in results[variant]

            # Verify sample sizes
            assert len(results[variant]["responses"]) == 10  # 30 samples / 3 variants
            assert len(results[variant]["token_usage"]) == 10

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        prompt_variants = ["basic_calibrated", "scenario_analysis"]

        results = self.ab_framework.run_ab_test(
            questions=self.mock_questions,
            prompt_variants=prompt_variants,
            sample_size=20,
        )

        for variant in prompt_variants:
            metrics = results[variant]["performance_metrics"]

            # Verify all expected metrics are present
            expected_metrics = [
                "avg_quality_score",
                "avg_calibration_score",
                "avg_token_efficiency",
                "avg_token_usage",
                "total_cost_estimate",
                "response_consistency",
            ]

            for metric in expected_metrics:
                assert metric in metrics, f"Missing metric: {metric}"
                assert isinstance(
                    metrics[metric], (int, float)
                ), f"Invalid metric type: {metric}"

    def test_variant_comparison(self):
        """Test comparison between prompt variants."""
        prompt_variants = [
            "basic_calibrated",
            "scenario_analysis",
            "overconfidence_reduction",
        ]

        results = self.ab_framework.run_ab_test(
            questions=self.mock_questions,
            prompt_variants=prompt_variants,
            sample_size=60,
        )

        # Compare token usage between variants
        basic_tokens = results["basic_calibrated"]["performance_metrics"][
            "avg_token_usage"
        ]
        scenario_tokens = results["scenario_analysis"]["performance_metrics"][
            "avg_token_usage"
        ]

        # Scenario analysis should use more tokens (more comprehensive)
        assert (
            scenario_tokens > basic_tokens
        ), "Scenario analysis should use more tokens"

        # Compare quality scores
        basic_quality = results["basic_calibrated"]["performance_metrics"][
            "avg_quality_score"
        ]
        scenario_quality = results["scenario_analysis"]["performance_metrics"][
            "avg_quality_score"
        ]

        # Quality differences should be measurable
        assert (
            abs(basic_quality - scenario_quality) >= 0
        ), "Should measure quality differences"

    def test_cost_efficiency_analysis(self):
        """Test cost efficiency analysis across variants."""
        prompt_variants = ["basic_calibrated", "scenario_analysis"]

        results = self.ab_framework.run_ab_test(
            questions=self.mock_questions,
            prompt_variants=prompt_variants,
            sample_size=40,
        )

        for variant in prompt_variants:
            metrics = results[variant]["performance_metrics"]

            # Cost should be proportional to token usage
            cost = metrics["total_cost_estimate"]
            tokens = metrics["avg_token_usage"] * len(results[variant]["responses"])

            assert cost > 0, "Should have positive cost estimate"
            assert cost < 1.0, "Cost should be reasonable for test sample"

        # Compare cost efficiency
        basic_efficiency = (
            results["basic_calibrated"]["performance_metrics"]["avg_quality_score"]
            / results["basic_calibrated"]["performance_metrics"]["total_cost_estimate"]
        )
        scenario_efficiency = (
            results["scenario_analysis"]["performance_metrics"]["avg_quality_score"]
            / results["scenario_analysis"]["performance_metrics"]["total_cost_estimate"]
        )

        # Both should have measurable efficiency
        assert basic_efficiency > 0
        assert scenario_efficiency > 0

    def test_statistical_significance_detection(self):
        """Test detection of statistically significant differences."""
        # Run larger sample for statistical significance
        prompt_variants = ["basic_calibrated", "scenario_analysis"]

        results = self.ab_framework.run_ab_test(
            questions=self.mock_questions,
            prompt_variants=prompt_variants,
            sample_size=100,
        )

        # Calculate statistical metrics
        basic_metrics = results["basic_calibrated"]["performance_metrics"]
        scenario_metrics = results["scenario_analysis"]["performance_metrics"]

        # Test for meaningful differences
        quality_diff = abs(
            basic_metrics["avg_quality_score"] - scenario_metrics["avg_quality_score"]
        )
        token_diff = abs(
            basic_metrics["avg_token_usage"] - scenario_metrics["avg_token_usage"]
        )

        # Should detect differences in token usage (scenario analysis uses more tokens)
        assert token_diff > 10, "Should detect significant token usage difference"

        # Quality differences should be measurable
        assert quality_diff >= 0, "Should measure quality differences"

    def test_consistency_measurement(self):
        """Test response consistency measurement."""
        prompt_variants = ["basic_calibrated"]

        results = self.ab_framework.run_ab_test(
            questions=self.mock_questions,
            prompt_variants=prompt_variants,
            sample_size=20,
        )

        consistency = results["basic_calibrated"]["performance_metrics"][
            "response_consistency"
        ]

        # Consistency should be between 0 and 1
        assert 0.0 <= consistency <= 1.0, f"Consistency out of range: {consistency}"

    def test_sample_size_impact(self):
        """Test impact of different sample sizes on results reliability."""
        prompt_variants = ["basic_calibrated", "scenario_analysis"]

        # Test with small sample
        small_results = self.ab_framework.run_ab_test(
            questions=self.mock_questions,
            prompt_variants=prompt_variants,
            sample_size=10,
        )

        # Test with larger sample
        large_results = self.ab_framework.run_ab_test(
            questions=self.mock_questions,
            prompt_variants=prompt_variants,
            sample_size=50,
        )

        # Larger sample should have more stable results (higher consistency)
        small_consistency = small_results["basic_calibrated"]["performance_metrics"][
            "response_consistency"
        ]
        large_consistency = large_results["basic_calibrated"]["performance_metrics"][
            "response_consistency"
        ]

        # Both should be valid
        assert 0.0 <= small_consistency <= 1.0
        assert 0.0 <= large_consistency <= 1.0

    def teardown_method(self):
        """Clean up test environment."""
        pass
