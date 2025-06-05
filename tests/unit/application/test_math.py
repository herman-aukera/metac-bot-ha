# test_math.py
# Unit tests for Brier and log score utilities

import math
from src.agents.math import brier_score, log_score

def test_brier_score():
    assert brier_score(1.0, 1) == 0.0
    assert brier_score(0.0, 1) == 1.0
    assert brier_score(0.5, 1) == 0.25
    assert brier_score(0.5, 0) == 0.25

def test_log_score():
    # log(1) = 0, log(0) is clipped
    assert math.isclose(log_score(1.0, 1), 0, abs_tol=1e-6)
    assert log_score(0.0, 1) < -10  # clipped
    assert log_score(1.0, 0) < -10  # clipped
    assert log_score(0.5, 1) < 0
    assert log_score(0.5, 0) < 0
