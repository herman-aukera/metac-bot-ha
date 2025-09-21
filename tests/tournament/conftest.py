"""Tournament testing configuration and fixtures."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from src.application.forecast_service import ForecastService
from src.domain.entities.forecast import Forecast
from src.domain.services.ensemble_service import EnsembleService
from src.domain.value_objects.confidence import ConfidenceLevel
from src.domain.value_objects.probability import Probability
from src.infrastructure.config.settings import Settings


@pytest.fixture
def tournament_settings():
    """Tournament-specific settings for testing."""
    settings = Mock(spec=Settings)
    settings.agent.timeout = 60
    settings.agent.max_iterations = 5
    settings.agent.confidence_threshold = 0.7
    settings.ensemble.aggregation_method = "weighted_average"
    settings.ensemble.min_agents = 2
    settings.ensemble.max_agents = 5
    settings.tournament.competitive_pressure = 0.5
    settings.tournament.time_pressure_factor = 1.0
    settings.tournament.resource_constraints = {}
    return settings


@pytest.fixture
def mock_tournament_forecast_service():
    """Mock forecast service for tournament testing."""
    service = Mock(spec=ForecastService)

    async def mock_generate_forecast(question, agent_types=None, timeout=60, **kwargs):
        # Simulate realistic tournament forecasting
        base_prediction = 0.4 + (question.id % 10) * 0.02
        base_confidence = 0.7 + (question.id % 5) * 0.05

        # Add some variance based on agent types
        if agent_types and "tree_of_thought" in agent_types:
            base_confidence += 0.1
        if agent_types and "react" in agent_types:
            base_prediction += 0.05

        # Simulate processing time
        await asyncio.sleep(0.1)

        return Forecast(
            question_id=question.id,
            prediction=Probability(min(0.99, max(0.01, base_prediction))),
            confidence=ConfidenceLevel(min(0.99, max(0.01, base_confidence))),
            reasoning=f"Tournament forecast for question {question.id}",
            method=(
                "ensemble" if agent_types and len(agent_types) > 1 else "single_agent"
            ),
            sources=["tournament_source_1", "tournament_source_2"],
            metadata={
                "agent_types": agent_types or ["default"],
                "timeout": timeout,
                "processing_time": 0.1,
            },
        )

    service.generate_forecast = AsyncMock(side_effect=mock_generate_forecast)
    return service


@pytest.fixture
def mock_tournament_ensemble_service():
    """Mock ensemble service for tournament testing."""
    service = Mock(spec=EnsembleService)

    def mock_aggregate_forecasts(forecasts, method="weighted_average"):
        if not forecasts:
            return None

        predictions = [f.prediction.value for f in forecasts]
        confidences = [f.confidence.value for f in forecasts]

        if method == "weighted_average":
            weights = confidences
            total_weight = sum(weights)
            if total_weight > 0:
                avg_prediction = (
                    sum(p * w for p, w in zip(predictions, weights)) / total_weight
                )
            else:
                avg_prediction = sum(predictions) / len(predictions)
            avg_confidence = sum(confidences) / len(confidences)
        elif method == "simple_average":
            avg_prediction = sum(predictions) / len(predictions)
            avg_confidence = sum(confidences) / len(confidences)
        elif method == "median":
            avg_prediction = sorted(predictions)[len(predictions) // 2]
            avg_confidence = sorted(confidences)[len(confidences) // 2]
        else:
            avg_prediction = sum(predictions) / len(predictions)
            avg_confidence = sum(confidences) / len(confidences)

        return Forecast(
            question_id=forecasts[0].question_id,
            prediction=Probability(avg_prediction),
            confidence=ConfidenceLevel(avg_confidence),
            reasoning=f"Ensemble forecast using {method}",
            method=f"ensemble_{method}",
            sources=["ensemble"],
            metadata={
                "aggregation_method": method,
                "individual_forecasts": len(forecasts),
            },
        )

    service.aggregate_forecasts = Mock(side_effect=mock_aggregate_forecasts)
    return service


@pytest.fixture
def sample_tournament_questions():
    """Sample questions for tournament testing."""
    questions = []

    categories = [
        ("Technology", ["AI", "computing", "innovation"]),
        ("Economics", ["markets", "inflation", "growth"]),
        ("Politics", ["elections", "policy", "governance"]),
        ("Science", ["research", "discovery", "breakthrough"]),
        ("Environment", ["climate", "sustainability", "conservation"]),
    ]

    for i in range(20):
        category, tags = categories[i % len(categories)]

        questions.append(
            {
                "id": 9000 + i,
                "title": f"Tournament question {i+1}: {category} prediction",
                "description": f"Tournament test question for {category.lower()} category",
                "type": "binary",
                "close_time": (datetime.now() + timedelta(days=30)).isoformat() + "Z",
                "resolve_time": (datetime.now() + timedelta(days=365)).isoformat()
                + "Z",
                "categories": [category],
                "tags": tags,
                "difficulty": (
                    "medium" if i % 3 == 0 else "hard" if i % 3 == 1 else "easy"
                ),
                "tournament_weight": 1.0 + (i % 3) * 0.5,  # Varying importance
            }
        )

    return questions


@pytest.fixture
def tournament_ground_truth():
    """Ground truth outcomes for tournament questions."""
    ground_truth = {}

    # Generate deterministic outcomes for consistent testing
    for i in range(100):
        question_id = 9000 + i
        # Use question characteristics to determine outcome
        outcome_probability = 0.3 + (question_id % 7) * 0.1
        outcome_seed = hash(f"tournament_{question_id}") % 1000
        ground_truth[question_id] = (outcome_seed / 1000.0) < outcome_probability

    return ground_truth


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for tournament testing."""
    return {
        "accuracy_threshold": 0.65,
        "brier_score_threshold": 0.35,
        "calibration_threshold": 0.7,
        "response_time_threshold": 10.0,
        "completion_rate_threshold": 0.9,
        "error_rate_threshold": 0.1,
        "confidence_correlation_threshold": 0.3,
    }


@pytest.fixture
def tournament_failure_scenarios():
    """Failure scenarios for resilience testing."""
    return [
        {
            "name": "API Failure",
            "failure_type": "api",
            "failure_rate": 0.3,
            "failure_duration": 5.0,
            "recovery_pattern": "immediate",
            "expected_recovery_time": 10.0,
        },
        {
            "name": "Network Timeout",
            "failure_type": "timeout",
            "failure_rate": 0.2,
            "failure_duration": 3.0,
            "recovery_pattern": "gradual",
            "expected_recovery_time": 8.0,
        },
        {
            "name": "Resource Exhaustion",
            "failure_type": "memory",
            "failure_rate": 0.1,
            "failure_duration": 10.0,
            "recovery_pattern": "delayed",
            "expected_recovery_time": 15.0,
        },
    ]


@pytest.fixture
def competitive_pressure_scenarios():
    """Competitive pressure scenarios for testing."""
    return [
        {
            "name": "Low Pressure",
            "time_pressure_factor": 1.0,
            "resource_pressure_factor": 1.0,
            "accuracy_pressure_factor": 1.0,
            "concurrent_questions": 1,
            "expected_degradation_threshold": 0.1,
        },
        {
            "name": "Medium Pressure",
            "time_pressure_factor": 0.6,
            "resource_pressure_factor": 0.7,
            "accuracy_pressure_factor": 0.8,
            "concurrent_questions": 2,
            "expected_degradation_threshold": 0.25,
        },
        {
            "name": "High Pressure",
            "time_pressure_factor": 0.3,
            "resource_pressure_factor": 0.4,
            "accuracy_pressure_factor": 0.5,
            "concurrent_questions": 3,
            "expected_degradation_threshold": 0.4,
        },
    ]


# Tournament-specific pytest markers


def pytest_configure(config):
    """Configure tournament-specific pytest markers."""
    config.addinivalue_line("markers", "tournament: Tournament simulation tests")
    config.addinivalue_line("markers", "pressure: Competitive pressure tests")
    config.addinivalue_line("markers", "resilience: Recovery and resilience tests")
    config.addinivalue_line("markers", "performance: Agent performance tests")
    config.addinivalue_line("markers", "calibration: Calibration validation tests")
    config.addinivalue_line("markers", "bias: Bias detection tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for tournament tests."""
    for item in items:
        # Add tournament marker to all tests in tournament directory
        if "tournament" in str(item.fspath):
            item.add_marker(pytest.mark.tournament)

        # Add specific markers based on test names
        if "pressure" in item.name:
            item.add_marker(pytest.mark.pressure)
        if "resilience" in item.name or "recovery" in item.name:
            item.add_marker(pytest.mark.resilience)
        if "performance" in item.name:
            item.add_marker(pytest.mark.performance)
        if "calibration" in item.name:
            item.add_marker(pytest.mark.calibration)
        if "bias" in item.name:
            item.add_marker(pytest.mark.bias)
