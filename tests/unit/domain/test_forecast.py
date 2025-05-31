# filepath: tests/unit/domain/test_forecast.py
"""Unit tests for forecast scoring."""

import pytest
from src.domain.entities.forecast import calculate_brier_score

def test_calculate_brier_score_perfect_forecast_correct_outcome():
    """Test Brier score for a perfect forecast (1.0) with outcome 1."""
    assert calculate_brier_score(forecast=1.0, outcome=1) == 0.0

def test_calculate_brier_score_perfect_forecast_incorrect_outcome():
    """Test Brier score for a perfect forecast (1.0) with outcome 0."""
    assert calculate_brier_score(forecast=1.0, outcome=0) == 1.0

def test_calculate_brier_score_no_confidence_forecast_correct_outcome():
    """Test Brier score for a no-confidence forecast (0.0) with outcome 1."""
    assert calculate_brier_score(forecast=0.0, outcome=1) == 1.0

def test_calculate_brier_score_no_confidence_forecast_incorrect_outcome():
    """Test Brier score for a no-confidence forecast (0.0) with outcome 0."""
    assert calculate_brier_score(forecast=0.0, outcome=0) == 0.0

def test_calculate_brier_score_mid_forecast_correct_outcome():
    """Test Brier score for a mid-range forecast (0.5) with outcome 1."""
    assert calculate_brier_score(forecast=0.5, outcome=1) == 0.25

def test_calculate_brier_score_mid_forecast_incorrect_outcome():
    """Test Brier score for a mid-range forecast (0.5) with outcome 0."""
    assert calculate_brier_score(forecast=0.5, outcome=0) == 0.25

def test_calculate_brier_score_various_cases():
    """Test Brier score with various valid inputs."""
    assert calculate_brier_score(forecast=0.75, outcome=1) == pytest.approx(0.0625)
    assert calculate_brier_score(forecast=0.25, outcome=1) == pytest.approx(0.5625)
    assert calculate_brier_score(forecast=0.75, outcome=0) == pytest.approx(0.5625)
    assert calculate_brier_score(forecast=0.25, outcome=0) == pytest.approx(0.0625)
    assert calculate_brier_score(forecast=0.1, outcome=0) == pytest.approx(0.01)
    assert calculate_brier_score(forecast=0.9, outcome=1) == pytest.approx(0.01)
    assert calculate_brier_score(forecast=0.1, outcome=1) == pytest.approx(0.81)
    assert calculate_brier_score(forecast=0.9, outcome=0) == pytest.approx(0.81)

def test_calculate_brier_score_invalid_forecast_input():
    """Test Brier score with invalid forecast inputs."""
    with pytest.raises(ValueError, match="Forecast probability must be between 0.0 and 1.0"):
        calculate_brier_score(forecast=-0.1, outcome=1)
    with pytest.raises(ValueError, match="Forecast probability must be between 0.0 and 1.0"):
        calculate_brier_score(forecast=1.1, outcome=0)

def test_calculate_brier_score_invalid_outcome_input():
    """Test Brier score with invalid outcome inputs."""
    with pytest.raises(ValueError, match="Outcome must be 0 or 1"):
        calculate_brier_score(forecast=0.5, outcome=2)
    with pytest.raises(ValueError, match="Outcome must be 0 or 1"):
        calculate_brier_score(forecast=0.5, outcome=-1)
    with pytest.raises(ValueError, match="Outcome must be 0 or 1"):
        calculate_brier_score(forecast=0.5, outcome=0.5)

# TODO: Add tests for multiclass Brier score or other scoring rules
