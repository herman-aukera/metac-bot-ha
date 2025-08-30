"""Competitive pressure testing for tournament conditions."""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.application.forecast_service import ForecastService
from src.domain.entities.forecast import Forecast
from src.domain.entities.question import Question, QuestionType
from src.domain.services.ensemble_service import EnsembleService
from src.domain.value_objects.confidence import ConfidenceLevel
from src.domain.value_objects.probability import Probability


@dataclass
class CompetitivePressureTest:
    """Defines a competitive pressure test scenario."""

    name: str
    time_pressure_factor: float  # 0.1 = very tight, 1.0 = normal
    resource_pressure_factor: float  # 0.1 = very limited, 1.0 = normal
    accuracy_pressure_factor: float  # 0.1 = low stakes, 1.0 = high stakes
    concurrent_questions: int
    expected_degradation_threshold: float  # Max acceptable performance drop


class CompetitivePressureTester:
    """Tests system behavior under competitive pressure."""

    def __init__(self, forecast_service: ForecastService):
        self.forecast_service = forecast_service
        self.baseline_performance = None

    async def establish_baseline(
        self, questions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Establish baseline performance under normal conditions."""
        start_time = time.time()
        forecasts = []

        for question_data in questions:
            question = self._create_question(question_data)

            try:
                forecast = await self.forecast_service.generate_forecast(
                    question=question,
                    agent_types=["chain_of_thought"],
                    timeout=300,  # Normal timeout
                )
                forecasts.append(forecast)
            except Exception as e:
                print(f"Baseline error on question {question.id}: {e}")

        execution_time = time.time() - start_time

        self.baseline_performance = {
            "completion_rate": len(forecasts) / len(questions),
            "avg_confidence": (
                sum(f.confidence.value for f in forecasts) / len(forecasts)
                if forecasts
                else 0
            ),
            "avg_execution_time": execution_time / len(questions),
            "error_rate": (len(questions) - len(forecasts)) / len(questions),
        }

        return self.baseline_performance

    async def test_competitive_pressure(
        self, pressure_test: CompetitivePressureTest, questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test system under competitive pressure conditions."""
        if not self.baseline_performance:
            await self.establish_baseline(questions)

        start_time = time.time()
        forecasts = []
        errors = []

        # Apply time pressure
        base_timeout = 300
        pressure_timeout = int(base_timeout * pressure_test.time_pressure_factor)

        # Apply resource pressure by limiting concurrent operations
        semaphore = asyncio.Semaphore(
            max(1, int(5 * pressure_test.resource_pressure_factor))
        )

        async def process_question_under_pressure(question_data):
            async with semaphore:
                question = self._create_question(question_data)

                try:
                    # Apply accuracy pressure by reducing research time
                    research_time = int(60 * pressure_test.accuracy_pressure_factor)

                    forecast = await asyncio.wait_for(
                        self.forecast_service.generate_forecast(
                            question=question,
                            agent_types=["chain_of_thought"],
                            timeout=pressure_timeout,
                            research_time_limit=research_time,
                        ),
                        timeout=pressure_timeout,
                    )
                    return forecast

                except asyncio.TimeoutError:
                    errors.append(f"Timeout on question {question.id}")
                    return None
                except Exception as e:
                    errors.append(f"Error on question {question.id}: {str(e)}")
                    return None

        # Process questions with pressure conditions
        if pressure_test.concurrent_questions > 1:
            # Concurrent processing adds pressure
            tasks = [
                process_question_under_pressure(q)
                for q in questions[: pressure_test.concurrent_questions]
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            forecasts = [r for r in results if isinstance(r, Forecast)]
        else:
            # Sequential processing under pressure
            for question_data in questions:
                result = await process_question_under_pressure(question_data)
                if result:
                    forecasts.append(result)

        execution_time = time.time() - start_time

        # Calculate performance under pressure
        pressure_performance = {
            "completion_rate": len(forecasts) / len(questions),
            "avg_confidence": (
                sum(f.confidence.value for f in forecasts) / len(forecasts)
                if forecasts
                else 0
            ),
            "avg_execution_time": execution_time / len(questions),
            "error_rate": len(errors) / len(questions),
            "total_errors": len(errors),
        }

        # Calculate performance degradation
        degradation = self._calculate_degradation(pressure_performance)

        return {
            "test_name": pressure_test.name,
            "pressure_factors": {
                "time": pressure_test.time_pressure_factor,
                "resource": pressure_test.resource_pressure_factor,
                "accuracy": pressure_test.accuracy_pressure_factor,
            },
            "baseline_performance": self.baseline_performance,
            "pressure_performance": pressure_performance,
            "degradation": degradation,
            "passed": degradation["overall"]
            <= pressure_test.expected_degradation_threshold,
            "errors": errors,
        }

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

    def _calculate_degradation(
        self, pressure_performance: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate performance degradation compared to baseline."""
        if not self.baseline_performance:
            return {"overall": 1.0}

        degradation = {}

        # Completion rate degradation (lower is worse)
        baseline_completion = self.baseline_performance["completion_rate"]
        pressure_completion = pressure_performance["completion_rate"]
        degradation["completion_rate"] = (
            max(0, (baseline_completion - pressure_completion) / baseline_completion)
            if baseline_completion > 0
            else 0
        )

        # Confidence degradation (lower is worse)
        baseline_confidence = self.baseline_performance["avg_confidence"]
        pressure_confidence = pressure_performance["avg_confidence"]
        degradation["confidence"] = (
            max(0, (baseline_confidence - pressure_confidence) / baseline_confidence)
            if baseline_confidence > 0
            else 0
        )

        # Error rate degradation (higher is worse)
        baseline_errors = self.baseline_performance["error_rate"]
        pressure_errors = pressure_performance["error_rate"]
        degradation["error_rate"] = (
            max(0, (pressure_errors - baseline_errors) / (1 - baseline_errors))
            if baseline_errors < 1
            else 1
        )

        # Overall degradation (weighted average)
        degradation["overall"] = (
            degradation["completion_rate"] * 0.4
            + degradation["confidence"] * 0.3
            + degradation["error_rate"] * 0.3
        )

        return degradation


class TestCompetitivePressure:
    """Test competitive pressure scenarios."""

    @pytest.fixture
    def pressure_tester(self, mock_settings):
        """Create competitive pressure tester."""
        mock_forecast_service = Mock(spec=ForecastService)
        return CompetitivePressureTester(mock_forecast_service)

    @pytest.fixture
    def test_questions(self):
        """Sample questions for pressure testing."""
        return [
            {
                "id": 4001,
                "title": "Will quantum computing breakthrough occur by 2026?",
                "description": "Quantum computing advancement prediction",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-12-31T00:00:00Z",
                "categories": ["Technology", "Computing"],
                "tags": ["quantum-computing", "breakthrough"],
            },
            {
                "id": 4002,
                "title": "Will renewable energy exceed 50% globally by 2027?",
                "description": "Renewable energy adoption prediction",
                "type": "binary",
                "close_time": "2026-12-01T00:00:00Z",
                "resolve_time": "2027-12-31T00:00:00Z",
                "categories": ["Energy", "Environment"],
                "tags": ["renewable-energy", "global"],
            },
            {
                "id": 4003,
                "title": "Will autonomous vehicles be mainstream by 2028?",
                "description": "Autonomous vehicle adoption prediction",
                "type": "binary",
                "close_time": "2027-12-01T00:00:00Z",
                "resolve_time": "2028-12-31T00:00:00Z",
                "categories": ["Transportation", "Technology"],
                "tags": ["autonomous-vehicles", "mainstream"],
            },
        ]

    @pytest.mark.asyncio
    async def test_time_pressure_effects(self, pressure_tester, test_questions):
        """Test effects of time pressure on forecasting performance."""

        # Mock forecast service with time-sensitive behavior
        async def mock_time_sensitive_forecast(
            question, agent_types, timeout, **kwargs
        ):
            # Simulate processing time based on available time
            base_processing_time = 5.0
            actual_processing_time = min(base_processing_time, timeout * 0.8)
            await asyncio.sleep(actual_processing_time)

            # Quality decreases with time pressure
            time_quality_factor = min(
                1.0, actual_processing_time / base_processing_time
            )

            return Forecast(
                question_id=question.id,
                prediction=Probability(0.5 + (time_quality_factor - 0.5) * 0.3),
                confidence=ConfidenceLevel(0.6 + time_quality_factor * 0.3),
                reasoning=f"Time-constrained analysis (quality={time_quality_factor:.2f})",
                method="chain_of_thought",
                sources=["time_limited_source"],
                metadata={"processing_time": actual_processing_time},
            )

        pressure_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_time_sensitive_forecast
        )

        # Test different time pressure levels
        time_pressures = [1.0, 0.5, 0.2]  # Normal, moderate, high pressure
        results = []

        for time_factor in time_pressures:
            pressure_test = CompetitivePressureTest(
                name=f"Time Pressure {time_factor}",
                time_pressure_factor=time_factor,
                resource_pressure_factor=1.0,
                accuracy_pressure_factor=1.0,
                concurrent_questions=1,
                expected_degradation_threshold=0.3,
            )

            result = await pressure_tester.test_competitive_pressure(
                pressure_test, test_questions
            )
            results.append((time_factor, result))

        # Verify time pressure effects
        for i in range(len(results) - 1):
            current_factor, current_result = results[i]
            next_factor, next_result = results[i + 1]

            # Higher time pressure should increase degradation
            current_degradation = current_result["degradation"]["overall"]
            next_degradation = next_result["degradation"]["overall"]
            assert next_degradation >= current_degradation - 0.1  # Allow some variance

            # Completion rate should decrease under higher pressure
            current_completion = current_result["pressure_performance"][
                "completion_rate"
            ]
            next_completion = next_result["pressure_performance"]["completion_rate"]
            assert next_completion <= current_completion + 0.1

    @pytest.mark.asyncio
    async def test_resource_pressure_effects(self, pressure_tester, test_questions):
        """Test effects of resource pressure on forecasting performance."""
        # Mock forecast service with resource-sensitive behavior
        call_count = 0

        async def mock_resource_sensitive_forecast(
            question, agent_types, timeout, **kwargs
        ):
            nonlocal call_count
            call_count += 1

            # Simulate resource contention
            await asyncio.sleep(0.5 + call_count * 0.1)  # Increasing delay

            # Quality decreases with resource pressure
            resource_quality_factor = max(0.3, 1.0 - call_count * 0.1)

            return Forecast(
                question_id=question.id,
                prediction=Probability(0.5 + (resource_quality_factor - 0.5) * 0.4),
                confidence=ConfidenceLevel(0.5 + resource_quality_factor * 0.4),
                reasoning=f"Resource-constrained analysis (quality={resource_quality_factor:.2f})",
                method="chain_of_thought",
                sources=["resource_limited_source"],
                metadata={"resource_usage": call_count},
            )

        pressure_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_resource_sensitive_forecast
        )

        # Test resource pressure
        pressure_test = CompetitivePressureTest(
            name="Resource Pressure Test",
            time_pressure_factor=1.0,
            resource_pressure_factor=0.3,  # High resource pressure
            accuracy_pressure_factor=1.0,
            concurrent_questions=3,  # Concurrent processing
            expected_degradation_threshold=0.4,
        )

        result = await pressure_tester.test_competitive_pressure(
            pressure_test, test_questions
        )

        # Verify resource pressure handling
        assert (
            result["passed"] or result["degradation"]["overall"] <= 0.5
        )  # Reasonable degradation
        assert (
            result["pressure_performance"]["completion_rate"] >= 0.6
        )  # Minimum completion
        assert len(result["errors"]) <= 2  # Limited errors

    @pytest.mark.asyncio
    async def test_accuracy_pressure_effects(self, pressure_tester, test_questions):
        """Test effects of accuracy pressure on forecasting performance."""

        # Mock forecast service with accuracy-sensitive behavior
        async def mock_accuracy_sensitive_forecast(
            question, agent_types, timeout, research_time_limit=60, **kwargs
        ):
            # Simulate research quality based on time limit
            research_quality = min(1.0, research_time_limit / 60.0)
            await asyncio.sleep(
                research_time_limit * 0.05
            )  # Proportional processing time

            # Accuracy pressure affects confidence more than prediction
            base_confidence = 0.8
            pressure_adjusted_confidence = base_confidence * research_quality

            return Forecast(
                question_id=question.id,
                prediction=Probability(0.45 + random.random() * 0.1),  # Slight variance
                confidence=ConfidenceLevel(pressure_adjusted_confidence),
                reasoning=f"Research-limited analysis (research_time={research_time_limit}s)",
                method="chain_of_thought",
                sources=["accuracy_pressure_source"],
                metadata={"research_time": research_time_limit},
            )

        pressure_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_accuracy_sensitive_forecast
        )

        # Test different accuracy pressure levels
        accuracy_pressures = [1.0, 0.5, 0.2]  # Normal, moderate, high pressure

        for accuracy_factor in accuracy_pressures:
            pressure_test = CompetitivePressureTest(
                name=f"Accuracy Pressure {accuracy_factor}",
                time_pressure_factor=1.0,
                resource_pressure_factor=1.0,
                accuracy_pressure_factor=accuracy_factor,
                concurrent_questions=1,
                expected_degradation_threshold=0.3,
            )

            result = await pressure_tester.test_competitive_pressure(
                pressure_test, test_questions
            )

            # Verify accuracy pressure effects
            if accuracy_factor < 0.5:  # High pressure
                assert (
                    result["degradation"]["confidence"] >= 0.1
                )  # Confidence should drop

            # System should still complete forecasts
            assert result["pressure_performance"]["completion_rate"] >= 0.8

    @pytest.mark.asyncio
    async def test_combined_pressure_resilience(self, pressure_tester, test_questions):
        """Test system resilience under combined pressure factors."""

        # Mock forecast service with combined pressure effects
        async def mock_combined_pressure_forecast(
            question, agent_types, timeout, research_time_limit=60, **kwargs
        ):
            # All pressure factors combined
            time_factor = min(1.0, timeout / 300.0)
            research_factor = min(1.0, research_time_limit / 60.0)

            combined_quality = (time_factor + research_factor) / 2
            processing_time = min(timeout * 0.8, 2.0 + (1.0 - combined_quality) * 3.0)

            await asyncio.sleep(processing_time)

            # Simulate potential failure under extreme pressure
            if combined_quality < 0.3 and random.random() < 0.3:
                raise Exception("System overload under combined pressure")

            return Forecast(
                question_id=question.id,
                prediction=Probability(0.4 + combined_quality * 0.2),
                confidence=ConfidenceLevel(0.4 + combined_quality * 0.4),
                reasoning=f"Combined pressure analysis (quality={combined_quality:.2f})",
                method="chain_of_thought",
                sources=["combined_pressure_source"],
                metadata={"combined_quality": combined_quality},
            )

        pressure_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_combined_pressure_forecast
        )

        # Test extreme combined pressure
        extreme_pressure_test = CompetitivePressureTest(
            name="Extreme Combined Pressure",
            time_pressure_factor=0.2,  # Very tight time
            resource_pressure_factor=0.3,  # Limited resources
            accuracy_pressure_factor=0.2,  # High stakes, low research time
            concurrent_questions=2,
            expected_degradation_threshold=0.6,  # Higher threshold for extreme conditions
        )

        result = await pressure_tester.test_competitive_pressure(
            extreme_pressure_test, test_questions
        )

        # Verify system survives extreme pressure
        assert (
            result["pressure_performance"]["completion_rate"] >= 0.5
        )  # At least half complete
        assert (
            result["degradation"]["overall"] <= 0.8
        )  # Significant but not total degradation
        assert len(result["errors"]) <= len(test_questions)  # Errors are contained

        # System should maintain some level of functionality
        if result["pressure_performance"]["completion_rate"] > 0:
            assert (
                result["pressure_performance"]["avg_confidence"] >= 0.3
            )  # Minimum confidence

    @pytest.mark.asyncio
    async def test_pressure_recovery_patterns(self, pressure_tester, test_questions):
        """Test system recovery patterns after pressure periods."""
        recovery_data = []

        # Mock forecast service that improves over time
        call_count = 0

        async def mock_recovery_forecast(question, agent_types, timeout, **kwargs):
            nonlocal call_count
            call_count += 1

            # Simulate recovery: performance improves with each call
            recovery_factor = min(1.0, call_count * 0.3)
            base_quality = 0.4 + recovery_factor * 0.4

            await asyncio.sleep(1.0 - recovery_factor * 0.5)  # Faster over time

            return Forecast(
                question_id=question.id,
                prediction=Probability(0.4 + base_quality * 0.2),
                confidence=ConfidenceLevel(base_quality),
                reasoning=f"Recovery analysis (call={call_count}, recovery={recovery_factor:.2f})",
                method="chain_of_thought",
                sources=["recovery_source"],
                metadata={"recovery_factor": recovery_factor},
            )

        pressure_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_recovery_forecast
        )

        # Test recovery pattern
        recovery_test = CompetitivePressureTest(
            name="Recovery Pattern Test",
            time_pressure_factor=0.5,
            resource_pressure_factor=0.5,
            accuracy_pressure_factor=0.5,
            concurrent_questions=1,
            expected_degradation_threshold=0.4,
        )

        result = await pressure_tester.test_competitive_pressure(
            recovery_test, test_questions
        )

        # Verify recovery behavior
        assert result["pressure_performance"]["completion_rate"] >= 0.8  # Good recovery
        assert result["degradation"]["overall"] <= 0.4  # Acceptable degradation

        # Later forecasts should show improvement (if metadata available)
        # This would be verified through detailed logging in a real implementation
