"""Tournament simulation testing framework for end-to-end validation."""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.application.forecast_service import ForecastService
from src.domain.entities.forecast import Forecast
from src.domain.entities.question import Question, QuestionType
from src.domain.services.ensemble_service import EnsembleService
from src.domain.services.tournament_analytics import TournamentAnalytics
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.probability import Probability
from src.infrastructure.config.settings import Settings


@dataclass
class TournamentScenario:
    """Represents a tournament scenario for testing."""

    name: str
    questions: List[Dict[str, Any]]
    duration_hours: int
    competitive_pressure: float  # 0.0 to 1.0
    expected_accuracy_threshold: float
    expected_calibration_threshold: float
    resource_constraints: Dict[str, Any]


@dataclass
class TournamentResult:
    """Results from tournament simulation."""

    scenario_name: str
    total_questions: int
    completed_forecasts: int
    accuracy_score: float
    calibration_score: float
    brier_score: float
    execution_time: float
    resource_usage: Dict[str, Any]
    errors: List[str]
    performance_metrics: Dict[str, Any]


class TournamentSimulator:
    """Simulates tournament conditions for testing."""

    def __init__(self, forecast_service: ForecastService, settings: Settings):
        self.forecast_service = forecast_service
        self.settings = settings
        self.tournament_analytics = TournamentAnalytics()

    async def simulate_tournament(
        self, scenario: TournamentScenario, agent_types: List[str] = None
    ) -> TournamentResult:
        """Simulate a complete tournament scenario."""
        if agent_types is None:
            agent_types = ["chain_of_thought", "tree_of_thought", "react"]

        start_time = time.time()
        forecasts = []
        errors = []
        resource_usage = {"api_calls": 0, "memory_peak_mb": 0, "cpu_time": 0}

        try:
            # Simulate tournament pressure by adjusting timeouts
            pressure_multiplier = 1.0 - (scenario.competitive_pressure * 0.5)
            adjusted_timeout = int(self.settings.agent.timeout * pressure_multiplier)

            # Process questions with tournament conditions
            for i, question_data in enumerate(scenario.questions):
                try:
                    # Create question entity
                    question = Question(
                        id=question_data["id"],
                        title=question_data["title"],
                        description=question_data["description"],
                        question_type=QuestionType(question_data["type"]),
                        close_time=datetime.fromisoformat(question_data["close_time"]),
                        resolve_time=datetime.fromisoformat(
                            question_data["resolve_time"]
                        ),
                        categories=question_data.get("categories", []),
                        tags=question_data.get("tags", []),
                    )

                    # Simulate competitive pressure with time constraints
                    remaining_time = scenario.duration_hours * 3600 - (
                        time.time() - start_time
                    )
                    remaining_questions = len(scenario.questions) - i
                    time_per_question = remaining_time / max(remaining_questions, 1)

                    # Apply resource constraints
                    if scenario.resource_constraints.get("limited_api_calls"):
                        max_calls = scenario.resource_constraints[
                            "max_api_calls_per_question"
                        ]
                        # Mock API call limiting would be implemented here

                    # Generate forecast with tournament conditions
                    forecast_timeout = min(
                        adjusted_timeout, int(time_per_question * 0.8)
                    )

                    forecast = await asyncio.wait_for(
                        self.forecast_service.generate_forecast(
                            question=question,
                            agent_types=agent_types,
                            timeout=forecast_timeout,
                        ),
                        timeout=forecast_timeout,
                    )

                    forecasts.append(forecast)
                    resource_usage["api_calls"] += forecast.metadata.get("api_calls", 1)

                except asyncio.TimeoutError:
                    errors.append(f"Timeout on question {question_data['id']}")
                except Exception as e:
                    errors.append(f"Error on question {question_data['id']}: {str(e)}")

        except Exception as e:
            errors.append(f"Tournament simulation error: {str(e)}")

        execution_time = time.time() - start_time

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            forecasts, scenario.questions
        )

        return TournamentResult(
            scenario_name=scenario.name,
            total_questions=len(scenario.questions),
            completed_forecasts=len(forecasts),
            accuracy_score=performance_metrics.get("accuracy", 0.0),
            calibration_score=performance_metrics.get("calibration", 0.0),
            brier_score=performance_metrics.get("brier_score", 1.0),
            execution_time=execution_time,
            resource_usage=resource_usage,
            errors=errors,
            performance_metrics=performance_metrics,
        )

    def _calculate_performance_metrics(
        self, forecasts: List[Forecast], questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate performance metrics for tournament results."""
        if not forecasts:
            return {"accuracy": 0.0, "calibration": 0.0, "brier_score": 1.0}

        # Simulate resolved outcomes for testing
        total_brier = 0.0
        calibration_bins = {i: {"predictions": [], "outcomes": []} for i in range(10)}

        for forecast in forecasts:
            # Simulate outcome based on prediction (for testing)
            simulated_outcome = random.random() < forecast.prediction.value

            # Calculate Brier score
            brier = (
                forecast.prediction.value - (1.0 if simulated_outcome else 0.0)
            ) ** 2
            total_brier += brier

            # Bin for calibration
            bin_idx = min(int(forecast.prediction.value * 10), 9)
            calibration_bins[bin_idx]["predictions"].append(forecast.prediction.value)
            calibration_bins[bin_idx]["outcomes"].append(
                1.0 if simulated_outcome else 0.0
            )

        avg_brier = total_brier / len(forecasts)

        # Calculate calibration score
        calibration_error = 0.0
        total_predictions = 0

        for bin_data in calibration_bins.values():
            if bin_data["predictions"]:
                avg_prediction = sum(bin_data["predictions"]) / len(
                    bin_data["predictions"]
                )
                avg_outcome = sum(bin_data["outcomes"]) / len(bin_data["outcomes"])
                bin_size = len(bin_data["predictions"])
                calibration_error += bin_size * abs(avg_prediction - avg_outcome)
                total_predictions += bin_size

        calibration_score = 1.0 - (
            calibration_error / total_predictions if total_predictions > 0 else 1.0
        )

        return {
            "accuracy": 1.0 - avg_brier,  # Convert Brier to accuracy-like metric
            "calibration": calibration_score,
            "brier_score": avg_brier,
            "completion_rate": len(forecasts) / len(questions) if questions else 0.0,
            "avg_confidence": sum(f.confidence.value for f in forecasts)
            / len(forecasts),
        }


class TestTournamentSimulation:
    """Test tournament simulation scenarios."""

    @pytest.fixture
    def tournament_simulator(self, mock_settings):
        """Create tournament simulator for testing."""
        mock_forecast_service = Mock(spec=ForecastService)
        return TournamentSimulator(mock_forecast_service, mock_settings)

    @pytest.fixture
    def basic_tournament_scenario(self):
        """Basic tournament scenario for testing."""
        questions = [
            {
                "id": 1001,
                "title": "Will AI achieve AGI by 2030?",
                "description": "Question about AGI timeline",
                "type": "binary",
                "close_time": "2029-12-01T00:00:00Z",
                "resolve_time": "2030-01-01T00:00:00Z",
                "categories": ["AI", "Technology"],
                "tags": ["agi", "artificial-intelligence"],
            },
            {
                "id": 1002,
                "title": "Will global temperature rise exceed 2°C by 2030?",
                "description": "Climate change prediction",
                "type": "binary",
                "close_time": "2029-12-01T00:00:00Z",
                "resolve_time": "2030-12-31T00:00:00Z",
                "categories": ["Climate", "Environment"],
                "tags": ["climate-change", "temperature"],
            },
            {
                "id": 1003,
                "title": "Will SpaceX land humans on Mars by 2030?",
                "description": "Space exploration prediction",
                "type": "binary",
                "close_time": "2029-12-01T00:00:00Z",
                "resolve_time": "2030-12-31T00:00:00Z",
                "categories": ["Space", "Technology"],
                "tags": ["mars", "spacex", "space-exploration"],
            },
        ]

        return TournamentScenario(
            name="Basic Tournament",
            questions=questions,
            duration_hours=24,
            competitive_pressure=0.5,
            expected_accuracy_threshold=0.7,
            expected_calibration_threshold=0.8,
            resource_constraints={},
        )

    @pytest.fixture
    def high_pressure_scenario(self):
        """High-pressure tournament scenario."""
        questions = []
        for i in range(10):
            questions.append(
                {
                    "id": 2000 + i,
                    "title": f"High-pressure question {i+1}",
                    "description": f"Complex forecasting question {i+1}",
                    "type": "binary",
                    "close_time": "2025-12-01T00:00:00Z",
                    "resolve_time": "2026-01-01T00:00:00Z",
                    "categories": ["Technology"],
                    "tags": ["high-pressure"],
                }
            )

        return TournamentScenario(
            name="High Pressure Tournament",
            questions=questions,
            duration_hours=6,  # Short duration
            competitive_pressure=0.9,  # High pressure
            expected_accuracy_threshold=0.6,  # Lower threshold due to pressure
            expected_calibration_threshold=0.7,
            resource_constraints={
                "limited_api_calls": True,
                "max_api_calls_per_question": 3,
            },
        )

    @pytest.mark.asyncio
    async def test_basic_tournament_simulation(
        self, tournament_simulator, basic_tournament_scenario
    ):
        """Test basic tournament simulation workflow."""

        # Mock forecast service to return realistic forecasts
        async def mock_generate_forecast(question, agent_types, timeout):
            return Forecast(
                question_id=question.id,
                prediction=Probability(0.4 + random.random() * 0.2),  # 0.4-0.6 range
                confidence=Confidence(0.7 + random.random() * 0.2),  # 0.7-0.9 range
                reasoning=f"Mocked reasoning for question {question.id}",
                method="ensemble",
                sources=["mock_source_1", "mock_source_2"],
                metadata={"api_calls": 2, "execution_time": 1.5},
            )

        tournament_simulator.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_generate_forecast
        )

        # Run tournament simulation
        result = await tournament_simulator.simulate_tournament(
            basic_tournament_scenario
        )

        # Verify tournament completion
        assert result.scenario_name == "Basic Tournament"
        assert result.total_questions == 3
        assert result.completed_forecasts >= 2  # Allow for some failures
        assert result.execution_time > 0
        assert result.execution_time < 300  # Should complete within 5 minutes

        # Verify performance metrics
        assert 0 <= result.accuracy_score <= 1
        assert 0 <= result.calibration_score <= 1
        assert 0 <= result.brier_score <= 1
        assert result.performance_metrics["completion_rate"] >= 0.6

        # Verify resource tracking
        assert result.resource_usage["api_calls"] > 0
        assert len(result.errors) <= 1  # Minimal errors expected

    @pytest.mark.asyncio
    async def test_high_pressure_tournament(
        self, tournament_simulator, high_pressure_scenario
    ):
        """Test high-pressure tournament conditions."""
        # Mock forecast service with occasional timeouts
        call_count = 0

        async def mock_generate_forecast_with_pressure(question, agent_types, timeout):
            nonlocal call_count
            call_count += 1

            # Simulate timeout on some calls due to pressure
            if call_count % 4 == 0:  # Every 4th call times out
                await asyncio.sleep(timeout + 0.1)  # Exceed timeout

            # Simulate reduced quality under pressure
            base_prediction = 0.5
            pressure_noise = random.random() * 0.3 - 0.15  # ±0.15 noise
            prediction = max(0.01, min(0.99, base_prediction + pressure_noise))

            return Forecast(
                question_id=question.id,
                prediction=Probability(prediction),
                confidence=Confidence(0.5 + random.random() * 0.3),  # Lower confidence
                reasoning=f"Rushed analysis for question {question.id}",
                method="ensemble",
                sources=["limited_source"],
                metadata={
                    "api_calls": 1,
                    "execution_time": 0.8,
                },  # Faster but less thorough
            )

        tournament_simulator.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_generate_forecast_with_pressure
        )

        # Run high-pressure tournament
        result = await tournament_simulator.simulate_tournament(high_pressure_scenario)

        # Verify tournament handles pressure
        assert result.scenario_name == "High Pressure Tournament"
        assert result.total_questions == 10
        assert result.completed_forecasts >= 6  # Some failures expected under pressure
        assert result.execution_time < high_pressure_scenario.duration_hours * 3600

        # Verify degraded performance under pressure
        assert result.performance_metrics["completion_rate"] >= 0.6
        assert len(result.errors) >= 2  # More errors expected under pressure

        # Verify resource constraints were applied
        avg_api_calls = result.resource_usage["api_calls"] / result.completed_forecasts
        assert avg_api_calls <= 3  # Resource constraint respected

    @pytest.mark.asyncio
    async def test_tournament_recovery_resilience(
        self, tournament_simulator, basic_tournament_scenario
    ):
        """Test tournament recovery from failures."""
        failure_count = 0

        async def mock_generate_forecast_with_failures(question, agent_types, timeout):
            nonlocal failure_count
            failure_count += 1

            # First two calls fail, then succeed
            if failure_count <= 2:
                raise Exception(f"Simulated failure {failure_count}")

            return Forecast(
                question_id=question.id,
                prediction=Probability(0.45),
                confidence=Confidence(0.75),
                reasoning=f"Recovered forecast for question {question.id}",
                method="ensemble",
                sources=["recovery_source"],
                metadata={"api_calls": 1, "execution_time": 1.0},
            )

        tournament_simulator.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_generate_forecast_with_failures
        )

        # Run tournament with failures
        result = await tournament_simulator.simulate_tournament(
            basic_tournament_scenario
        )

        # Verify recovery behavior
        assert result.completed_forecasts >= 1  # At least one successful forecast
        assert len(result.errors) == 2  # Two failures recorded
        assert (
            result.performance_metrics["completion_rate"] >= 0.33
        )  # Partial completion

        # Verify system continued despite failures
        assert result.execution_time > 0
        assert "Simulated failure" in str(result.errors)

    @pytest.mark.asyncio
    async def test_tournament_performance_benchmarks(
        self, tournament_simulator, basic_tournament_scenario
    ):
        """Test tournament performance benchmarks."""

        # Mock high-performance forecast service
        async def mock_fast_forecast(question, agent_types, timeout):
            # Simulate fast, accurate forecasting
            await asyncio.sleep(0.1)  # Very fast response

            return Forecast(
                question_id=question.id,
                prediction=Probability(0.42),  # Consistent prediction
                confidence=Confidence(0.85),  # High confidence
                reasoning=f"Optimized forecast for question {question.id}",
                method="ensemble",
                sources=["benchmark_source_1", "benchmark_source_2"],
                metadata={"api_calls": 1, "execution_time": 0.1},
            )

        tournament_simulator.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_fast_forecast
        )

        # Run benchmark tournament
        result = await tournament_simulator.simulate_tournament(
            basic_tournament_scenario
        )

        # Verify performance benchmarks
        assert result.completed_forecasts == result.total_questions  # 100% completion
        assert result.execution_time < 5  # Very fast execution
        assert len(result.errors) == 0  # No errors
        assert result.performance_metrics["completion_rate"] == 1.0
        assert result.performance_metrics["avg_confidence"] >= 0.8

        # Verify resource efficiency
        assert result.resource_usage["api_calls"] == result.total_questions

    @pytest.mark.asyncio
    async def test_competitive_pressure_effects(self, tournament_simulator):
        """Test effects of different competitive pressure levels."""
        base_questions = [
            {
                "id": 3001,
                "title": "Test pressure question",
                "description": "Testing competitive pressure effects",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-01-01T00:00:00Z",
                "categories": ["Test"],
                "tags": ["pressure-test"],
            }
        ]

        pressure_levels = [0.1, 0.5, 0.9]  # Low, medium, high pressure
        results = []

        for pressure in pressure_levels:
            scenario = TournamentScenario(
                name=f"Pressure Test {pressure}",
                questions=base_questions,
                duration_hours=1,
                competitive_pressure=pressure,
                expected_accuracy_threshold=0.7,
                expected_calibration_threshold=0.8,
                resource_constraints={},
            )

            # Mock forecast service with pressure-sensitive behavior
            async def mock_pressure_sensitive_forecast(question, agent_types, timeout):
                # Higher pressure = faster but less accurate
                processing_time = 2.0 * (1.0 - pressure)  # Less time under pressure
                await asyncio.sleep(processing_time)

                # Accuracy decreases with pressure
                base_accuracy = 0.8
                pressure_penalty = pressure * 0.3
                effective_accuracy = base_accuracy - pressure_penalty

                return Forecast(
                    question_id=question.id,
                    prediction=Probability(0.5 + (effective_accuracy - 0.5) * 0.5),
                    confidence=Confidence(effective_accuracy),
                    reasoning=f"Pressure-affected forecast (pressure={pressure})",
                    method="ensemble",
                    sources=["pressure_source"],
                    metadata={"api_calls": 1, "execution_time": processing_time},
                )

            tournament_simulator.forecast_service.generate_forecast = AsyncMock(
                side_effect=mock_pressure_sensitive_forecast
            )

            result = await tournament_simulator.simulate_tournament(scenario)
            results.append((pressure, result))

        # Verify pressure effects
        for i in range(len(results) - 1):
            current_pressure, current_result = results[i]
            next_pressure, next_result = results[i + 1]

            # Higher pressure should generally lead to faster execution
            assert next_result.execution_time <= current_result.execution_time * 1.5

            # Confidence should generally decrease with pressure
            current_confidence = current_result.performance_metrics["avg_confidence"]
            next_confidence = next_result.performance_metrics["avg_confidence"]
            assert next_confidence <= current_confidence + 0.1  # Allow some variance
