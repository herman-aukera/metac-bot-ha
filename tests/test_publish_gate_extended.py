from src.application.publish_gate import evaluate_publish


def test_soft_low_info_mc_allows_publish() -> None:
    probs = [0.251, 0.249, 0.25, 0.25]
    d = evaluate_publish(
        rationale="adequate rationale with more than sixty characters to pass threshold",
        probabilities=probs,
    )
    # Should classify as near uniform but allow publish (soft)
    assert d.publish is True
    assert any(r.startswith("NEAR_UNIFORM") or r == "LOW_INFO_MC" for r in d.reasons)


def test_uniform_block_still_blocks() -> None:
    probs = [0.25, 0.25, 0.25, 0.25]
    d = evaluate_publish(rationale="short", probabilities=probs)
    assert d.publish is False
    assert d.blocked is True


def test_fallback_distribution_flag() -> None:
    probs = [0.1, 0.5, 0.4]
    d = evaluate_publish(
        rationale="fallback injection", probabilities=probs, fallback_flag=True
    )
    assert d.publish is True
    assert "FALLBACK_DISTRIBUTION" in d.reasons
