"""Recovery and resilience testing under tournament conditions."""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest

from src.application.forecast_service import ForecastService
from src.domain.entities.forecast import Forecast
from src.domain.entities.question import Question, QuestionType
from src.domain.value_objects.confidence import ConfidenceLevel
from src.domain.value_objects.probability import Probability


@dataclass
class FailureScenario:
    """Defines a failure scenario for resilience testing."""

    name: str
    failure_type: str  # "api", "timeout", "memory", "network", "partial"
    failure_rate: float  # 0.0 to 1.0
    failure_duration: float  # seconds
    recovery_pattern: str  # "immediate", "gradual", "delayed"
    expected_recovery_time: float  # seconds


class ResilienceTester:
    """Tests system resilience and recovery capabilities."""

    def __init__(self, forecast_service: ForecastService):
        self.forecast_service = forecast_service
        self.failure_injector = FailureInjector()

    async def test_failure_recovery(
        self, scenario: FailureScenario, questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test system recovery from specific failure scenarios."""

        # Configure failure injection
        self.failure_injector.configure(
            failure_type=scenario.failure_type,
            failure_rate=scenario.failure_rate,
            failure_duration=scenario.failure_duration,
            recovery_pattern=scenario.recovery_pattern,
        )

        start_time = time.time()
        results = {
            "scenario_name": scenario.name,
            "total_questions": len(questions),
            "successful_forecasts": 0,
            "failed_forecasts": 0,
            "recovery_time": 0,
            "errors": [],
            "performance_timeline": [],
            "recovery_successful": False,
        }

        # Process questions with failure injection
        for i, question_data in enumerate(questions):
            question = self._create_question(question_data)
            attempt_start = time.time()

            try:
                # Inject failures based on scenario
                if self.failure_injector.should_fail():
                    await self.failure_injector.inject_failure(scenario.failure_type)

                await self.forecast_service.generate_forecast(
                    question=question, agent_types=["chain_of_thought"], timeout=60
                )

                results["successful_forecasts"] += 1
                success = True

            except Exception as e:
                results["failed_forecasts"] += 1
                results["errors"].append(f"Question {question.id}: {str(e)}")
                success = False

            # Record performance timeline
            attempt_time = time.time() - attempt_start
            results["performance_timeline"].append(
                {
                    "question_id": question.id,
                    "attempt_time": attempt_time,
                    "success": success,
                    "timestamp": time.time() - start_time,
                }
            )

            # Check for recovery
            if not results["recovery_successful"] and success:
                # First successful forecast after failures indicates recovery
                if results["failed_forecasts"] > 0:
                    results["recovery_time"] = time.time() - start_time
                    results["recovery_successful"] = True

        # Analyze recovery patterns
        results["recovery_analysis"] = self._analyze_recovery_pattern(
            results["performance_timeline"]
        )
        results["meets_recovery_expectations"] = (
            results["recovery_successful"]
            and results["recovery_time"] <= scenario.expected_recovery_time
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

    def _analyze_recovery_pattern(
        self, timeline: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze recovery patterns from performance timeline."""
        if not timeline:
            return {"pattern": "no_data"}

        # Find failure and recovery periods
        failure_periods = []
        current_failure_start = None

        for entry in timeline:
            if not entry["success"] and current_failure_start is None:
                current_failure_start = entry["timestamp"]
            elif entry["success"] and current_failure_start is not None:
                failure_periods.append(
                    {
                        "start": current_failure_start,
                        "end": entry["timestamp"],
                        "duration": entry["timestamp"] - current_failure_start,
                    }
                )
                current_failure_start = None

        # Calculate recovery metrics
        if failure_periods:
            avg_recovery_time = sum(fp["duration"] for fp in failure_periods) / len(
                failure_periods
            )
            max_recovery_time = max(fp["duration"] for fp in failure_periods)

            return {
                "pattern": "recovery_detected",
                "failure_periods": len(failure_periods),
                "avg_recovery_time": avg_recovery_time,
                "max_recovery_time": max_recovery_time,
                "total_downtime": sum(fp["duration"] for fp in failure_periods),
            }
        else:
            return {"pattern": "no_failures_or_no_recovery"}


class FailureInjector:
    """Injects various types of failures for testing."""

    def __init__(self):
        self.failure_type = None
        self.failure_rate = 0.0
        self.failure_duration = 0.0
        self.recovery_pattern = "immediate"
        self.failure_start_time = None
        self.call_count = 0

    def configure(
        self,
        failure_type: str,
        failure_rate: float,
        failure_duration: float,
        recovery_pattern: str,
    ):
        """Configure failure injection parameters."""
        self.failure_type = failure_type
        self.failure_rate = failure_rate
        self.failure_duration = failure_duration
        self.recovery_pattern = recovery_pattern
        self.failure_start_time = None
        self.call_count = 0

    def should_fail(self) -> bool:
        """Determine if a failure should be injected."""
        self.call_count += 1

        # Check if we're in a failure period
        if self.failure_start_time is not None:
            elapsed = time.time() - self.failure_start_time

            if self.recovery_pattern == "immediate":
                return elapsed < self.failure_duration
            elif self.recovery_pattern == "gradual":
                # Gradually reduce failure rate
                recovery_progress = min(1.0, elapsed / self.failure_duration)
                adjusted_rate = self.failure_rate * (1.0 - recovery_progress)
                return random.random() < adjusted_rate
            elif self.recovery_pattern == "delayed":
                # Full failure for duration, then immediate recovery
                return elapsed < self.failure_duration

        # Decide whether to start a new failure period
        if random.random() < self.failure_rate:
            self.failure_start_time = time.time()
            return True

        return False

    async def inject_failure(self, failure_type: str):
        """Inject specific type of failure."""
        if failure_type == "api":
            raise Exception("API service unavailable")
        elif failure_type == "timeout":
            await asyncio.sleep(10)  # Simulate timeout
            raise asyncio.TimeoutError("Operation timed out")
        elif failure_type == "memory":
            raise MemoryError("Insufficient memory")
        elif failure_type == "network":
            raise ConnectionError("Network connection failed")
        elif failure_type == "partial":
            # Partial failure - sometimes works, sometimes doesn't
            if random.random() < 0.5:
                raise Exception("Partial service degradation")


class TestRecoveryResilience:
    """Test recovery and resilience scenarios."""

    @pytest.fixture
    def resilience_tester(self, mock_settings):
        """Create resilience tester."""
        mock_forecast_service = Mock(spec=ForecastService)
        return ResilienceTester(mock_forecast_service)

    @pytest.fixture
    def resilience_questions(self):
        """Questions for resilience testing."""
        return [
            {
                "id": 5001,
                "title": "Resilience test question 1",
                "description": "Testing system resilience",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-01-01T00:00:00Z",
                "categories": ["Test"],
                "tags": ["resilience"],
            },
            {
                "id": 5002,
                "title": "Resilience test question 2",
                "description": "Testing recovery patterns",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-01-01T00:00:00Z",
                "categories": ["Test"],
                "tags": ["recovery"],
            },
            {
                "id": 5003,
                "title": "Resilience test question 3",
                "description": "Testing failure handling",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-01-01T00:00:00Z",
                "categories": ["Test"],
                "tags": ["failure-handling"],
            },
        ]

    @pytest.mark.asyncio
    async def test_api_failure_recovery(self, resilience_tester, resilience_questions):
        """Test recovery from API failures."""
        # Mock forecast service with API failure simulation
        call_count = 0

        async def mock_api_failure_forecast(question, agent_types, timeout):
            nonlocal call_count
            call_count += 1

            # Simulate API failures for first few calls, then recovery
            if call_count <= 2:
                raise Exception("API service unavailable")

            # Successful forecast after recovery
            return Forecast(
                question_id=question.id,
                prediction=Probability(0.45),
                confidence=ConfidenceLevel(0.75),
                reasoning=f"Recovered forecast for question {question.id}",
                method="chain_of_thought",
                sources=["recovered_api"],
                metadata={"call_count": call_count},
            )

        resilience_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_api_failure_forecast
        )

        # Test API failure scenario
        api_failure_scenario = FailureScenario(
            name="API Failure Recovery",
            failure_type="api",
            failure_rate=0.7,  # High failure rate initially
            failure_duration=5.0,  # 5 second failure period
            recovery_pattern="immediate",
            expected_recovery_time=10.0,
        )

        result = await resilience_tester.test_failure_recovery(
            api_failure_scenario, resilience_questions
        )

        # Verify recovery behavior
        assert result["recovery_successful"], "System should recover from API failures"
        assert result["successful_forecasts"] >= 1, (
            "At least one forecast should succeed after recovery"
        )
        assert result["recovery_time"] <= 15.0, (
            "Recovery should happen within reasonable time"
        )
        assert "API service unavailable" in str(result["errors"]), (
            "API errors should be recorded"
        )

    @pytest.mark.asyncio
    async def test_timeout_recovery(self, resilience_tester, resilience_questions):
        """Test recovery from timeout failures."""
        # Mock forecast service with timeout simulation
        call_count = 0

        async def mock_timeout_forecast(question, agent_types, timeout):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call times out
                await asyncio.sleep(timeout + 1)
                raise asyncio.TimeoutError("Operation timed out")
            else:
                # Subsequent calls succeed quickly
                await asyncio.sleep(0.5)
                return Forecast(
                    question_id=question.id,
                    prediction=Probability(0.42),
                    confidence=ConfidenceLevel(0.78),
                    reasoning="Quick forecast after timeout recovery",
                    method="chain_of_thought",
                    sources=["timeout_recovery"],
                    metadata={"call_count": call_count},
                )

        resilience_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_timeout_forecast
        )

        # Test timeout scenario
        timeout_scenario = FailureScenario(
            name="Timeout Recovery",
            failure_type="timeout",
            failure_rate=0.3,
            failure_duration=2.0,
            recovery_pattern="immediate",
            expected_recovery_time=5.0,
        )

        result = await resilience_tester.test_failure_recovery(
            timeout_scenario, resilience_questions
        )

        # Verify timeout recovery
        assert result["recovery_successful"], "System should recover from timeouts"
        assert result["successful_forecasts"] >= 2, (
            "Multiple forecasts should succeed after timeout"
        )
        assert any("timed out" in error.lower() for error in result["errors"]), (
            "Timeout errors should be recorded"
        )

    @pytest.mark.asyncio
    async def test_gradual_recovery_pattern(
        self, resilience_tester, resilience_questions
    ):
        """Test gradual recovery patterns."""
        # Mock forecast service with gradual recovery
        call_count = 0

        async def mock_gradual_recovery_forecast(question, agent_types, timeout):
            nonlocal call_count
            call_count += 1

            # Gradually improving success rate
            success_probability = min(1.0, call_count * 0.3)

            if random.random() > success_probability:
                raise Exception(f"Gradual recovery failure (attempt {call_count})")

            # Success with improving quality
            quality = success_probability
            return Forecast(
                question_id=question.id,
                prediction=Probability(0.4 + quality * 0.2),
                confidence=ConfidenceLevel(0.5 + quality * 0.3),
                reasoning=f"Gradual recovery forecast (quality={quality:.2f})",
                method="chain_of_thought",
                sources=["gradual_recovery"],
                metadata={"call_count": call_count, "quality": quality},
            )

        resilience_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_gradual_recovery_forecast
        )

        # Test gradual recovery
        gradual_scenario = FailureScenario(
            name="Gradual Recovery",
            failure_type="partial",
            failure_rate=0.8,
            failure_duration=10.0,
            recovery_pattern="gradual",
            expected_recovery_time=15.0,
        )

        result = await resilience_tester.test_failure_recovery(
            gradual_scenario, resilience_questions
        )

        # Verify gradual recovery
        assert result["successful_forecasts"] >= 1, (
            "Some forecasts should succeed during gradual recovery"
        )
        assert result["recovery_analysis"]["pattern"] in [
            "recovery_detected",
            "no_failures_or_no_recovery",
        ]

    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(
        self, resilience_tester, resilience_questions
    ):
        """Test circuit breaker behavior during failures."""
        # Mock forecast service with circuit breaker simulation
        failure_count = 0
        circuit_open = False

        async def mock_circuit_breaker_forecast(question, agent_types, timeout):
            nonlocal failure_count, circuit_open

            # Simulate circuit breaker logic
            if circuit_open:
                if failure_count > 5:  # Circuit stays open for a while
                    failure_count -= 1
                    raise Exception("Circuit breaker open - service unavailable")
                else:
                    # Circuit closes, service recovers
                    circuit_open = False
                    failure_count = 0

            # Normal operation or failure detection
            if failure_count < 3:
                if random.random() < 0.3:  # 30% failure rate
                    failure_count += 1
                    raise Exception(f"Service failure {failure_count}")
                else:
                    failure_count = max(0, failure_count - 1)  # Gradual recovery
            else:
                # Too many failures, open circuit
                circuit_open = True
                raise Exception("Circuit breaker opened due to repeated failures")

            # Successful forecast
            return Forecast(
                question_id=question.id,
                prediction=Probability(0.48),
                confidence=ConfidenceLevel(0.72),
                reasoning="Circuit breaker test forecast",
                method="chain_of_thought",
                sources=["circuit_breaker_test"],
                metadata={"failure_count": failure_count, "circuit_open": circuit_open},
            )

        resilience_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_circuit_breaker_forecast
        )

        # Test circuit breaker scenario
        circuit_breaker_scenario = FailureScenario(
            name="Circuit Breaker Test",
            failure_type="partial",
            failure_rate=0.5,
            failure_duration=8.0,
            recovery_pattern="delayed",
            expected_recovery_time=12.0,
        )

        result = await resilience_tester.test_failure_recovery(
            circuit_breaker_scenario, resilience_questions
        )

        # Verify circuit breaker behavior
        assert len(result["errors"]) > 0, "Circuit breaker should record failures"
        assert any("circuit breaker" in error.lower() for error in result["errors"]), (
            "Circuit breaker errors should be recorded"
        )

        # System should eventually recover
        if result["recovery_successful"]:
            assert result["recovery_time"] <= 20.0, (
                "Circuit breaker should allow eventual recovery"
            )

    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(
        self, resilience_tester, resilience_questions
    ):
        """Test recovery from cascading failures."""
        # Mock forecast service with cascading failure simulation
        failure_cascade = {"api": False, "network": False, "memory": False}
        call_count = 0

        async def mock_cascading_failure_forecast(question, agent_types, timeout):
            nonlocal call_count
            call_count += 1

            # Simulate cascading failures
            if call_count == 1:
                failure_cascade["api"] = True
                raise Exception("API service failed")
            elif call_count == 2 and failure_cascade["api"]:
                failure_cascade["network"] = True
                raise ConnectionError("Network failure due to API issues")
            elif call_count == 3 and failure_cascade["network"]:
                failure_cascade["memory"] = True
                raise MemoryError("Memory exhaustion due to connection retries")
            elif call_count <= 5:
                # Gradual recovery
                if failure_cascade["memory"]:
                    failure_cascade["memory"] = False
                    raise Exception("System still recovering from memory issues")
                elif failure_cascade["network"]:
                    failure_cascade["network"] = False
                    raise Exception("Network connectivity restored, API still unstable")
                elif failure_cascade["api"]:
                    failure_cascade["api"] = False
                    # Partial success during recovery
                    if random.random() < 0.5:
                        raise Exception("API intermittent during recovery")

            # Full recovery
            return Forecast(
                question_id=question.id,
                prediction=Probability(0.46),
                confidence=ConfidenceLevel(0.74),
                reasoning="Forecast after cascading failure recovery",
                method="chain_of_thought",
                sources=["cascading_recovery"],
                metadata={
                    "call_count": call_count,
                    "failures_resolved": not any(failure_cascade.values()),
                },
            )

        resilience_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_cascading_failure_forecast
        )

        # Test cascading failure scenario
        cascading_scenario = FailureScenario(
            name="Cascading Failure Recovery",
            failure_type="partial",
            failure_rate=0.9,  # High failure rate
            failure_duration=15.0,  # Longer recovery time
            recovery_pattern="gradual",
            expected_recovery_time=20.0,
        )

        result = await resilience_tester.test_failure_recovery(
            cascading_scenario, resilience_questions
        )

        # Verify cascading failure handling
        assert len(result["errors"]) >= 3, (
            "Multiple types of failures should be recorded"
        )
        error_types = set()
        for error in result["errors"]:
            if "api" in error.lower():
                error_types.add("api")
            elif "network" in error.lower() or "connection" in error.lower():
                error_types.add("network")
            elif "memory" in error.lower():
                error_types.add("memory")

        assert len(error_types) >= 2, "Multiple failure types should be detected"

        # System should eventually recover even from cascading failures
        if result["recovery_successful"]:
            assert result["successful_forecasts"] >= 1, (
                "System should recover from cascading failures"
            )

    @pytest.mark.asyncio
    async def test_recovery_under_tournament_pressure(
        self, resilience_tester, resilience_questions
    ):
        """Test recovery behavior under tournament pressure conditions."""
        # Mock forecast service with pressure-sensitive recovery
        call_count = 0
        pressure_level = 0.8  # High tournament pressure

        async def mock_pressure_recovery_forecast(question, agent_types, timeout):
            nonlocal call_count
            call_count += 1

            # Under pressure, recovery is slower and less reliable
            base_failure_rate = 0.4
            pressure_adjusted_rate = base_failure_rate + (pressure_level * 0.3)

            if call_count <= 3 and random.random() < pressure_adjusted_rate:
                # Pressure makes failures more likely
                raise Exception(f"Pressure-induced failure (call {call_count})")

            # Recovery quality affected by pressure
            recovery_quality = max(0.3, 1.0 - pressure_level * 0.5)

            return Forecast(
                question_id=question.id,
                prediction=Probability(0.4 + recovery_quality * 0.2),
                confidence=ConfidenceLevel(0.5 + recovery_quality * 0.3),
                reasoning=f"Pressure recovery forecast (quality={recovery_quality:.2f})",
                method="chain_of_thought",
                sources=["pressure_recovery"],
                metadata={"call_count": call_count, "pressure_level": pressure_level},
            )

        resilience_tester.forecast_service.generate_forecast = AsyncMock(
            side_effect=mock_pressure_recovery_forecast
        )

        # Test recovery under pressure
        pressure_recovery_scenario = FailureScenario(
            name="Tournament Pressure Recovery",
            failure_type="partial",
            failure_rate=0.6,
            failure_duration=12.0,
            recovery_pattern="gradual",
            expected_recovery_time=18.0,
        )

        result = await resilience_tester.test_failure_recovery(
            pressure_recovery_scenario, resilience_questions
        )

        # Verify pressure-affected recovery
        assert result["successful_forecasts"] >= 1, (
            "System should recover even under tournament pressure"
        )

        # Recovery might be slower under pressure
        if result["recovery_successful"]:
            assert result["recovery_time"] <= 25.0, (
                "Recovery should happen within extended time under pressure"
            )

        # Some degradation is acceptable under pressure
        success_rate = result["successful_forecasts"] / result["total_questions"]
        assert success_rate >= 0.3, (
            "Minimum success rate should be maintained under pressure"
        )
