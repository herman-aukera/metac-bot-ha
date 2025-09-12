from src.application.publish_gate import evaluate_publish


def test_publish_gate_blocks_neutral_phrase() -> None:
    d = evaluate_publish(
        rationale="Assigning neutral probability due to lack of data",
        probabilities=[0.5, 0.5],
        is_binary=True,
        confidence=0.9,
        min_confidence=0.3,
    )
    assert not d.publish
    assert any("BANNED_PHRASE" in r for r in d.reasons)
