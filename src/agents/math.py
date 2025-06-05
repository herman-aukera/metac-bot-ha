# math.py
# Brier/log score utilities for forecast evaluation

def brier_score(prob: float, outcome: int) -> float:
    """Compute Brier score for binary forecast."""
    return (prob - outcome) ** 2

def log_score(prob: float, outcome: int) -> float:
    """Compute log score for binary forecast (clipped for stability)."""
    import math
    prob = min(max(prob, 1e-8), 1 - 1e-8)
    if outcome == 1:
        return math.log(prob)
    else:
        return math.log(1 - prob)
