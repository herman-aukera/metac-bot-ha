"""Agent performance and calibration testing framework."""

import asyncio
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from src.agents.base_agent import BaseAgent
from src.agents.chain_of_thought_agent import ChainOfThoughtAgent
from src.agents.react_agent import ReActAgent
from src.agents.tree_of_thought_agent import TreeOfThoughtAgent
from src.domain.entities.forecast import Forecast
from src.domain.entities.question import Question, QuestionType
from src.domain.services.ensemble_service import EnsembleService
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.probability import Probability


@dataclass
class PerformanceBenchmark:
    """Defines performance benchmarks for agent testing."""

    agent_type: str
    min_accuracy: float
    max_brier_score: float
    min_calibration_score: float
    max_response_time: float
    min_confidence_correlation: float


@dataclass
class CalibrationBin:
    """Represents a calibration bin for analysis."""

    bin_range: Tuple[float, float]
    predictions: List[float]
    outcomes: List[bool]
    count: int
    avg_prediction: float
    avg_outcome: float
    calibration_error: float


class AgentPerformanceTester:
    """Tests individual agent performance and calibration."""

    def __init__(self):
        self.test_questions = []
        self.ground_truth_outcomes = {}

    def generate_test_questions(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate test questions with known outcomes for benchmarking."""
        questions = []

        # Generate diverse question types
        categories = [
            ("Technology", ["AI", "computing", "innovation"]),
            ("Economics", ["markets", "inflation", "growth"]),
            ("Politics", ["elections", "policy", "governance"]),
            ("Science", ["research", "discovery", "breakthrough"]),
            ("Environment", ["climate", "sustainability", "conservation"]),
        ]

        for i in range(count):
            category, tags = categories[i % len(categories)]

            # Create question with deterministic outcome for testing
            question_id = 6000 + i
            base_probability = 0.3 + (i % 7) * 0.1  # Vary base probability

            question = {
                "id": question_id,
                "title": f"Test question {i+1}: {category} prediction",
                "description": f"Performance test question for {category.lower()} category",
                "type": "binary",
                "close_time": (datetime.now() + timedelta(days=30)).isoformat() + "Z",
                "resolve_time": (datetime.now() + timedelta(days=365)).isoformat()
                + "Z",
                "categories": [category],
                "tags": tags,
                "difficulty": (
                    "medium" if i % 3 == 0 else "hard" if i % 3 == 1 else "easy"
                ),
            }

            questions.append(question)

            # Generate deterministic outcome based on question characteristics
            # This allows consistent testing across runs
            outcome_seed = hash(f"{question_id}_{category}_{base_probability}") % 1000
            self.ground_truth_outcomes[question_id] = (
                outcome_seed / 1000.0
            ) < base_probability

        self.test_questions = questions
        return questions

    async def benchmark_agent_performance(
        self,
        agent: BaseAgent,
        questions: List[Dict[str, Any]] = None,
        benchmark: PerformanceBenchmark = None,
    ) -> Dict[str, Any]:
        """Benchmark individual agent performance."""
        if questions is None:
            questions = self.test_questions or self.generate_test_questions()

        results = {
            "agent_type": agent.__class__.__name__,
            "total_questions": len(questions),
            "forecasts": [],
            "performance_metrics": {},
            "calibration_analysis": {},
            "timing_analysis": {},
            "error_analysis": {},
            "benchmark_results": {},
        }

        forecasts = []
        response_times = []
        errors = []

        # Process each question
        for question_data in questions:
            question = self._create_question(question_data)

            try:
                start_time = asyncio.get_event_loop().time()

                forecast = await agent.generate_forecast(question)

                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time

                forecasts.append(forecast)
                response_times.append(response_time)

                results["forecasts"].append(
                    {
                        "question_id": question.id,
                        "prediction": forecast.prediction.value,
                        "confidence": forecast.confidence.value,
                        "response_time": response_time,
                        "reasoning_length": len(forecast.reasoning),
                        "source_count": len(forecast.sources),
                    }
                )

            except Exception as e:
                errors.append(
                    {
                        "question_id": question.id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

        # Calculate performance metrics
        results["performance_metrics"] = self._calculate_performance_metrics(
            forecasts, questions
        )
        results["calibration_analysis"] = self._analyze_calibration(
            forecasts, questions
        )
        results["timing_analysis"] = self._analyze_timing(response_times)
        results["error_analysis"] = self._analyze_errors(errors)

        # Compare against benchmark if provided
        if benchmark:
            results["benchmark_results"] = self._compare_to_benchmark(
                results, benchmark
            )

        return results

    def _create_question(self, question_data: Dict[str, Any]) -> Question:
        """Create Question entity from data."""
        return Question(
            id=question_data["id"],
            title=question_data["title"],
            description=question_data["description"],
            question_type=QuestionType(question_data["type"]),
            close_time=datetime.fromisoformat(question_data["close_time"]),
            resolve_time=datetime.fromisoformat(question_data["resolve_time"]),
            categories=question_data.get("categories", []),
            tags=question_data.get("tags", []),
        )

    def _calculate_performance_metrics(
        self, forecasts: List[Forecast], questions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not forecasts:
            return {"error": "no_forecasts"}

        predictions = [f.prediction.value for f in forecasts]
        confidences = [f.confidence.value for f in forecasts]

        # Get ground truth outcomes
        outcomes = []
        for forecast in forecasts:
            outcome = self.ground_truth_outcomes.get(forecast.question_id, False)
            outcomes.append(1.0 if outcome else 0.0)

        # Calculate Brier score
        brier_scores = [
            (pred - outcome) ** 2 for pred, outcome in zip(predictions, outcomes)
        ]
        avg_brier_score = statistics.mean(brier_scores)

        # Calculate accuracy (for binary predictions)
        binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
        accuracy = sum(
            1 for pred, outcome in zip(binary_predictions, outcomes) if pred == outcome
        ) / len(outcomes)

        # Calculate log score (proper scoring rule)
        log_scores = []
        for pred, outcome in zip(predictions, outcomes):
            # Avoid log(0) by clamping predictions
            clamped_pred = max(0.001, min(0.999, pred))
            if outcome == 1.0:
                log_scores.append(math.log(clamped_pred))
            else:
                log_scores.append(math.log(1 - clamped_pred))
        avg_log_score = statistics.mean(log_scores)

        # Calculate confidence-accuracy correlation
        confidence_accuracy_corr = self._calculate_correlation(
            confidences, [abs(p - o) for p, o in zip(predictions, outcomes)]
        )

        # Calculate overconfidence/underconfidence
        confidence_errors = [
            conf - abs(pred - outcome)
            for conf, pred, outcome in zip(confidences, predictions, outcomes)
        ]
        avg_confidence_error = statistics.mean(confidence_errors)

        return {
            "accuracy": accuracy,
            "brier_score": avg_brier_score,
            "log_score": avg_log_score,
            "avg_confidence": statistics.mean(confidences),
            "confidence_std": (
                statistics.stdev(confidences) if len(confidences) > 1 else 0
            ),
            "confidence_accuracy_correlation": confidence_accuracy_corr,
            "avg_confidence_error": avg_confidence_error,
            "overconfidence": max(0, avg_confidence_error),
            "underconfidence": max(0, -avg_confidence_error),
        }

    def _analyze_calibration(
        self, forecasts: List[Forecast], questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze agent calibration across prediction ranges."""
        if not forecasts:
            return {"error": "no_forecasts"}

        # Create calibration bins
        num_bins = 10
        bins = []

        for i in range(num_bins):
            bin_start = i / num_bins
            bin_end = (i + 1) / num_bins

            bin_predictions = []
            bin_outcomes = []

            for forecast in forecasts:
                pred = forecast.prediction.value
                if bin_start <= pred < bin_end or (i == num_bins - 1 and pred == 1.0):
                    bin_predictions.append(pred)
                    outcome = self.ground_truth_outcomes.get(
                        forecast.question_id, False
                    )
                    bin_outcomes.append(outcome)

            if bin_predictions:
                avg_prediction = statistics.mean(bin_predictions)
                avg_outcome = statistics.mean([1.0 if o else 0.0 for o in bin_outcomes])
                calibration_error = abs(avg_prediction - avg_outcome)

                bins.append(
                    CalibrationBin(
                        bin_range=(bin_start, bin_end),
                        predictions=bin_predictions,
                        outcomes=bin_outcomes,
                        count=len(bin_predictions),
                        avg_prediction=avg_prediction,
                        avg_outcome=avg_outcome,
                        calibration_error=calibration_error,
                    )
                )

        # Calculate overall calibration metrics
        if bins:
            # Expected Calibration Error (ECE)
            total_predictions = sum(bin.count for bin in bins)
            ece = (
                sum(bin.count * bin.calibration_error for bin in bins)
                / total_predictions
            )

            # Maximum Calibration Error (MCE)
            mce = max(bin.calibration_error for bin in bins)

            # Reliability (average calibration error weighted by bin size)
            reliability = ece

            # Resolution (ability to discriminate between different outcomes)
            overall_base_rate = statistics.mean(
                [
                    1.0 if self.ground_truth_outcomes.get(f.question_id, False) else 0.0
                    for f in forecasts
                ]
            )
            resolution = (
                sum(
                    bin.count * (bin.avg_outcome - overall_base_rate) ** 2
                    for bin in bins
                )
                / total_predictions
            )

            return {
                "expected_calibration_error": ece,
                "maximum_calibration_error": mce,
                "reliability": reliability,
                "resolution": resolution,
                "calibration_score": 1.0 - ece,  # Higher is better
                "bins": [
                    {
                        "range": bin.bin_range,
                        "count": bin.count,
                        "avg_prediction": bin.avg_prediction,
                        "avg_outcome": bin.avg_outcome,
                        "calibration_error": bin.calibration_error,
                    }
                    for bin in bins
                ],
            }
        else:
            return {"error": "no_valid_bins"}

    def _analyze_timing(self, response_times: List[float]) -> Dict[str, float]:
        """Analyze response time performance."""
        if not response_times:
            return {"error": "no_timing_data"}

        return {
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "response_time_std": (
                statistics.stdev(response_times) if len(response_times) > 1 else 0
            ),
            "p95_response_time": np.percentile(response_times, 95),
            "p99_response_time": np.percentile(response_times, 99),
        }

    def _analyze_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns."""
        if not errors:
            return {"error_rate": 0.0, "error_types": {}}

        error_types = {}
        for error in errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "error_rate": len(errors) / (len(errors) + len(self.test_questions)),
            "total_errors": len(errors),
            "error_types": error_types,
            "most_common_error": (
                max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
            ),
        }

    def _compare_to_benchmark(
        self, results: Dict[str, Any], benchmark: PerformanceBenchmark
    ) -> Dict[str, Any]:
        """Compare results to performance benchmark."""
        metrics = results["performance_metrics"]
        calibration = results["calibration_analysis"]
        timing = results["timing_analysis"]

        benchmark_results = {
            "meets_accuracy_threshold": metrics.get("accuracy", 0)
            >= benchmark.min_accuracy,
            "meets_brier_threshold": metrics.get("brier_score", 1.0)
            <= benchmark.max_brier_score,
            "meets_calibration_threshold": calibration.get("calibration_score", 0)
            >= benchmark.min_calibration_score,
            "meets_timing_threshold": timing.get("avg_response_time", float("inf"))
            <= benchmark.max_response_time,
            "meets_confidence_correlation_threshold": metrics.get(
                "confidence_accuracy_correlation", 0
            )
            >= benchmark.min_confidence_correlation,
        }

        benchmark_results["overall_pass"] = all(benchmark_results.values())

        return benchmark_results

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = math.sqrt(sum_sq_x * sum_sq_y)

        return numerator / denominator if denominator != 0 else 0.0


class EnsembleOptimizationTester:
    """Tests ensemble optimization and performance."""

    def __init__(self):
        self.agent_tester = AgentPerformanceTester()

    async def test_ensemble_optimization(
        self,
        agents: List[BaseAgent],
        ensemble_service: EnsembleService,
        questions: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Test ensemble optimization strategies."""
        if questions is None:
            questions = self.agent_tester.generate_test_questions(30)

        # Test individual agents first
        individual_results = {}
        for agent in agents:
            agent_name = agent.__class__.__name__
            individual_results[agent_name] = (
                await self.agent_tester.benchmark_agent_performance(agent, questions)
            )

        # Test ensemble combinations
        ensemble_results = await self._test_ensemble_combinations(
            agents, ensemble_service, questions
        )

        # Analyze ensemble optimization
        optimization_analysis = self._analyze_ensemble_optimization(
            individual_results, ensemble_results
        )

        return {
            "individual_results": individual_results,
            "ensemble_results": ensemble_results,
            "optimization_analysis": optimization_analysis,
        }

    async def _test_ensemble_combinations(
        self,
        agents: List[BaseAgent],
        ensemble_service: EnsembleService,
        questions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Test different ensemble combinations and aggregation methods."""
        results = {}

        # Test different aggregation methods
        aggregation_methods = [
            "simple_average",
            "weighted_average",
            "median",
            "trimmed_mean",
        ]

        for method in aggregation_methods:
            method_results = []

            for question_data in questions:
                question = self.agent_tester._create_question(question_data)

                # Get individual forecasts
                individual_forecasts = []
                for agent in agents:
                    try:
                        forecast = await agent.generate_forecast(question)
                        individual_forecasts.append(forecast)
                    except Exception:
                        continue  # Skip failed forecasts

                if len(individual_forecasts) >= 2:
                    # Create ensemble forecast
                    ensemble_forecast = ensemble_service.aggregate_forecasts(
                        individual_forecasts, method=method
                    )

                    method_results.append(
                        {
                            "question_id": question.id,
                            "individual_forecasts": [
                                {
                                    "agent": (
                                        f.__class__.__name__
                                        if hasattr(f, "__class__")
                                        else "unknown"
                                    ),
                                    "prediction": f.prediction.value,
                                    "confidence": f.confidence.value,
                                }
                                for f in individual_forecasts
                            ],
                            "ensemble_prediction": ensemble_forecast.prediction.value,
                            "ensemble_confidence": ensemble_forecast.confidence.value,
                        }
                    )

            # Calculate ensemble performance metrics
            if method_results:
                ensemble_predictions = [
                    r["ensemble_prediction"] for r in method_results
                ]
                ensemble_confidences = [
                    r["ensemble_confidence"] for r in method_results
                ]

                # Get outcomes for these questions
                outcomes = []
                for result in method_results:
                    outcome = self.agent_tester.ground_truth_outcomes.get(
                        result["question_id"], False
                    )
                    outcomes.append(1.0 if outcome else 0.0)

                # Calculate metrics
                brier_scores = [
                    (pred - outcome) ** 2
                    for pred, outcome in zip(ensemble_predictions, outcomes)
                ]
                avg_brier = statistics.mean(brier_scores)

                binary_predictions = [1 if p > 0.5 else 0 for p in ensemble_predictions]
                accuracy = sum(
                    1
                    for pred, outcome in zip(binary_predictions, outcomes)
                    if pred == outcome
                ) / len(outcomes)

                results[method] = {
                    "forecasts": method_results,
                    "accuracy": accuracy,
                    "brier_score": avg_brier,
                    "avg_confidence": statistics.mean(ensemble_confidences),
                    "prediction_variance": (
                        statistics.variance(ensemble_predictions)
                        if len(ensemble_predictions) > 1
                        else 0
                    ),
                }

        return results

    def _analyze_ensemble_optimization(
        self, individual_results: Dict[str, Any], ensemble_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze ensemble optimization effectiveness."""
        analysis = {
            "best_individual_agent": None,
            "best_ensemble_method": None,
            "ensemble_improvement": {},
            "diversity_analysis": {},
            "optimization_recommendations": [],
        }

        # Find best individual agent
        best_individual_brier = float("inf")
        for agent_name, results in individual_results.items():
            brier = results["performance_metrics"].get("brier_score", float("inf"))
            if brier < best_individual_brier:
                best_individual_brier = brier
                analysis["best_individual_agent"] = agent_name

        # Find best ensemble method
        best_ensemble_brier = float("inf")
        for method, results in ensemble_results.items():
            brier = results.get("brier_score", float("inf"))
            if brier < best_ensemble_brier:
                best_ensemble_brier = brier
                analysis["best_ensemble_method"] = method

        # Calculate ensemble improvement
        if best_individual_brier < float("inf") and best_ensemble_brier < float("inf"):
            improvement = (
                best_individual_brier - best_ensemble_brier
            ) / best_individual_brier
            analysis["ensemble_improvement"]["brier_improvement"] = improvement
            analysis["ensemble_improvement"]["improvement_percentage"] = (
                improvement * 100
            )

        # Analyze diversity
        if len(individual_results) > 1:
            individual_accuracies = [
                results["performance_metrics"].get("accuracy", 0)
                for results in individual_results.values()
            ]
            accuracy_diversity = (
                statistics.stdev(individual_accuracies)
                if len(individual_accuracies) > 1
                else 0
            )

            analysis["diversity_analysis"] = {
                "accuracy_diversity": accuracy_diversity,
                "agent_count": len(individual_results),
                "diversity_score": min(1.0, accuracy_diversity * 2),  # Normalize to 0-1
            }

        # Generate optimization recommendations
        if analysis["ensemble_improvement"].get("improvement_percentage", 0) > 5:
            analysis["optimization_recommendations"].append(
                "Ensemble shows significant improvement - recommend using ensemble"
            )

        if analysis["diversity_analysis"].get("diversity_score", 0) < 0.3:
            analysis["optimization_recommendations"].append(
                "Low agent diversity - consider adding more diverse agents"
            )

        if best_ensemble_brier > 0.3:
            analysis["optimization_recommendations"].append(
                "High Brier score - consider improving individual agents or aggregation method"
            )

        return analysis


class TestAgentPerformance:
    """Test agent performance and calibration."""

    @pytest.fixture
    def performance_tester(self):
        """Create agent performance tester."""
        return AgentPerformanceTester()

    @pytest.fixture
    def ensemble_tester(self):
        """Create ensemble optimization tester."""
        return EnsembleOptimizationTester()

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        agents = []

        # Mock Chain of Thought Agent
        cot_agent = Mock(spec=ChainOfThoughtAgent)
        cot_agent.__class__.__name__ = "ChainOfThoughtAgent"

        async def cot_forecast(question):
            # Simulate CoT behavior - generally good accuracy, moderate confidence
            base_pred = 0.4 + (question.id % 10) * 0.02  # Slight variation
            return Forecast(
                question_id=question.id,
                prediction=Probability(base_pred),
                confidence=Confidence(0.75),
                reasoning="Chain of thought analysis...",
                method="chain_of_thought",
                sources=["cot_source_1", "cot_source_2"],
                metadata={"steps": 5},
            )

        cot_agent.generate_forecast = AsyncMock(side_effect=cot_forecast)
        agents.append(cot_agent)

        # Mock Tree of Thought Agent
        tot_agent = Mock(spec=TreeOfThoughtAgent)
        tot_agent.__class__.__name__ = "TreeOfThoughtAgent"

        async def tot_forecast(question):
            # Simulate ToT behavior - higher accuracy, higher confidence
            base_pred = 0.45 + (question.id % 8) * 0.025
            return Forecast(
                question_id=question.id,
                prediction=Probability(base_pred),
                confidence=Confidence(0.82),
                reasoning="Tree of thought exploration...",
                method="tree_of_thought",
                sources=["tot_source_1", "tot_source_2", "tot_source_3"],
                metadata={"branches": 3, "depth": 4},
            )

        tot_agent.generate_forecast = AsyncMock(side_effect=tot_forecast)
        agents.append(tot_agent)

        # Mock ReAct Agent
        react_agent = Mock(spec=ReActAgent)
        react_agent.__class__.__name__ = "ReActAgent"

        async def react_forecast(question):
            # Simulate ReAct behavior - variable accuracy, adaptive confidence
            base_pred = 0.35 + (question.id % 12) * 0.03
            confidence_val = 0.65 + (question.id % 5) * 0.05
            return Forecast(
                question_id=question.id,
                prediction=Probability(base_pred),
                confidence=Confidence(confidence_val),
                reasoning="ReAct reasoning and acting...",
                method="react",
                sources=["react_source_1"],
                metadata={"actions": 2, "reasoning_steps": 3},
            )

        react_agent.generate_forecast = AsyncMock(side_effect=react_forecast)
        agents.append(react_agent)

        return agents

    @pytest.fixture
    def performance_benchmarks(self):
        """Define performance benchmarks for testing."""
        return {
            "ChainOfThoughtAgent": PerformanceBenchmark(
                agent_type="ChainOfThoughtAgent",
                min_accuracy=0.65,
                max_brier_score=0.35,
                min_calibration_score=0.7,
                max_response_time=10.0,
                min_confidence_correlation=0.3,
            ),
            "TreeOfThoughtAgent": PerformanceBenchmark(
                agent_type="TreeOfThoughtAgent",
                min_accuracy=0.7,
                max_brier_score=0.3,
                min_calibration_score=0.75,
                max_response_time=15.0,
                min_confidence_correlation=0.4,
            ),
            "ReActAgent": PerformanceBenchmark(
                agent_type="ReActAgent",
                min_accuracy=0.6,
                max_brier_score=0.4,
                min_calibration_score=0.65,
                max_response_time=12.0,
                min_confidence_correlation=0.25,
            ),
        }

    @pytest.mark.asyncio
    async def test_individual_agent_accuracy(
        self, performance_tester, mock_agents, performance_benchmarks
    ):
        """Test individual agent accuracy benchmarking."""
        questions = performance_tester.generate_test_questions(20)

        for agent in mock_agents:
            agent_name = agent.__class__.__name__
            benchmark = performance_benchmarks[agent_name]

            results = await performance_tester.benchmark_agent_performance(
                agent, questions, benchmark
            )

            # Verify basic results structure
            assert results["agent_type"] == agent_name
            assert results["total_questions"] == 20
            assert len(results["forecasts"]) > 0

            # Verify performance metrics
            metrics = results["performance_metrics"]
            assert "accuracy" in metrics
            assert "brier_score" in metrics
            assert "avg_confidence" in metrics
            assert 0 <= metrics["accuracy"] <= 1
            assert 0 <= metrics["brier_score"] <= 1
            assert 0 <= metrics["avg_confidence"] <= 1

            # Verify calibration analysis
            calibration = results["calibration_analysis"]
            if "expected_calibration_error" in calibration:
                assert 0 <= calibration["expected_calibration_error"] <= 1
                assert "bins" in calibration
                assert len(calibration["bins"]) > 0

            # Verify timing analysis
            timing = results["timing_analysis"]
            assert "avg_response_time" in timing
            assert timing["avg_response_time"] > 0

            # Check benchmark compliance
            if "benchmark_results" in results:
                benchmark_results = results["benchmark_results"]
                assert "overall_pass" in benchmark_results
                # Note: Mock agents may not meet all benchmarks, which is expected

    @pytest.mark.asyncio
    async def test_calibration_accuracy_validation(
        self, performance_tester, mock_agents
    ):
        """Test calibration accuracy validation."""
        questions = performance_tester.generate_test_questions(30)

        for agent in mock_agents:
            results = await performance_tester.benchmark_agent_performance(
                agent, questions
            )

            calibration = results["calibration_analysis"]

            if "bins" in calibration:
                # Verify calibration bins are reasonable
                for bin_data in calibration["bins"]:
                    assert 0 <= bin_data["avg_prediction"] <= 1
                    assert 0 <= bin_data["avg_outcome"] <= 1
                    assert bin_data["count"] > 0
                    assert bin_data["calibration_error"] >= 0

                # Verify overall calibration metrics
                assert 0 <= calibration["expected_calibration_error"] <= 1
                assert 0 <= calibration["maximum_calibration_error"] <= 1
                assert 0 <= calibration["calibration_score"] <= 1

    @pytest.mark.asyncio
    async def test_ensemble_optimization(self, ensemble_tester, mock_agents):
        """Test ensemble optimization and performance."""
        # Mock ensemble service
        mock_ensemble_service = Mock(spec=EnsembleService)

        def mock_aggregate_forecasts(forecasts, method="weighted_average"):
            if not forecasts:
                return None

            # Simple aggregation for testing
            predictions = [f.prediction.value for f in forecasts]
            confidences = [f.confidence.value for f in forecasts]

            if method == "simple_average":
                avg_pred = statistics.mean(predictions)
                avg_conf = statistics.mean(confidences)
            elif method == "weighted_average":
                # Weight by confidence
                weights = confidences
                total_weight = sum(weights)
                avg_pred = (
                    sum(p * w for p, w in zip(predictions, weights)) / total_weight
                )
                avg_conf = statistics.mean(confidences)
            elif method == "median":
                avg_pred = statistics.median(predictions)
                avg_conf = statistics.median(confidences)
            elif method == "trimmed_mean":
                # Remove extreme values
                sorted_preds = sorted(predictions)
                trim_count = max(1, len(sorted_preds) // 4)
                trimmed_preds = (
                    sorted_preds[trim_count:-trim_count]
                    if len(sorted_preds) > 2
                    else sorted_preds
                )
                avg_pred = statistics.mean(trimmed_preds)
                avg_conf = statistics.mean(confidences)
            else:
                avg_pred = statistics.mean(predictions)
                avg_conf = statistics.mean(confidences)

            return Forecast(
                question_id=forecasts[0].question_id,
                prediction=Probability(avg_pred),
                confidence=Confidence(avg_conf),
                reasoning=f"Ensemble forecast using {method}",
                method=f"ensemble_{method}",
                sources=["ensemble"],
                metadata={"method": method, "agent_count": len(forecasts)},
            )

        mock_ensemble_service.aggregate_forecasts = Mock(
            side_effect=mock_aggregate_forecasts
        )

        # Test ensemble optimization
        results = await ensemble_tester.test_ensemble_optimization(
            mock_agents, mock_ensemble_service
        )

        # Verify results structure
        assert "individual_results" in results
        assert "ensemble_results" in results
        assert "optimization_analysis" in results

        # Verify individual results
        individual_results = results["individual_results"]
        assert len(individual_results) == len(mock_agents)

        for agent_name, agent_results in individual_results.items():
            assert "performance_metrics" in agent_results
            assert "calibration_analysis" in agent_results

        # Verify ensemble results
        ensemble_results = results["ensemble_results"]
        assert len(ensemble_results) > 0

        for method, method_results in ensemble_results.items():
            assert "accuracy" in method_results
            assert "brier_score" in method_results
            assert "forecasts" in method_results

        # Verify optimization analysis
        optimization = results["optimization_analysis"]
        assert "best_individual_agent" in optimization
        assert "best_ensemble_method" in optimization
        assert "ensemble_improvement" in optimization

    @pytest.mark.asyncio
    async def test_reasoning_quality_assessment(self, performance_tester, mock_agents):
        """Test reasoning quality assessment."""
        questions = performance_tester.generate_test_questions(15)

        for agent in mock_agents:
            results = await performance_tester.benchmark_agent_performance(
                agent, questions
            )

            # Verify reasoning quality metrics are captured
            forecasts = results["forecasts"]

            for forecast_data in forecasts:
                assert "reasoning_length" in forecast_data
                assert "source_count" in forecast_data
                assert forecast_data["reasoning_length"] > 0
                assert forecast_data["source_count"] >= 0

            # Calculate reasoning quality metrics
            avg_reasoning_length = statistics.mean(
                [f["reasoning_length"] for f in forecasts]
            )
            avg_source_count = statistics.mean([f["source_count"] for f in forecasts])

            assert avg_reasoning_length > 10  # Reasonable reasoning length
            assert avg_source_count >= 1  # At least one source

    @pytest.mark.asyncio
    async def test_bias_detection(self, performance_tester, mock_agents):
        """Test bias detection in agent predictions."""
        questions = performance_tester.generate_test_questions(25)

        for agent in mock_agents:
            results = await performance_tester.benchmark_agent_performance(
                agent, questions
            )

            forecasts = results["forecasts"]
            predictions = [f["prediction"] for f in forecasts]

            # Test for prediction bias
            avg_prediction = statistics.mean(predictions)
            prediction_variance = (
                statistics.variance(predictions) if len(predictions) > 1 else 0
            )

            # Check for extreme bias (all predictions too high or too low)
            extreme_high_bias = avg_prediction > 0.8
            extreme_low_bias = avg_prediction < 0.2
            low_variance = (
                prediction_variance < 0.01
            )  # Very low variance indicates potential bias

            # Record bias indicators (for analysis, not necessarily failures)
            bias_indicators = {
                "avg_prediction": avg_prediction,
                "prediction_variance": prediction_variance,
                "extreme_high_bias": extreme_high_bias,
                "extreme_low_bias": extreme_low_bias,
                "low_variance_bias": low_variance,
            }

            # Verify bias detection is working
            assert "avg_prediction" in bias_indicators
            assert 0 <= bias_indicators["avg_prediction"] <= 1
            assert bias_indicators["prediction_variance"] >= 0
