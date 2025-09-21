
from typing import cast, Any

from src.infrastructure.external_apis.reasoning_comment_formatter import ReasoningCommentFormatter


class DummyForecast:
    def __init__(self, n_preds=2):
        self.predictions = [object()] * n_preds
        self.ensemble_method = "confidence_weighted"
        self.weight_distribution = {"agent_a": 0.6, "agent_b": 0.4}
        self.reasoning_summary = "Consensus leans toward outcome A based on available evidence."


def test_format_ensemble_information_basic():
    formatter = ReasoningCommentFormatter()
    fc = DummyForecast(n_preds=3)

    text = formatter._format_ensemble_information(cast(Any, fc))  # type: ignore[arg-type]

    assert "Ensemble Analysis:" in text
    assert "Combined 3 predictions" in text
    assert "Method: confidence_weighted" in text
    assert "Weights: agent_a: 0.60, agent_b: 0.40" in text
    assert "Summary: Consensus leans toward outcome A" in text
