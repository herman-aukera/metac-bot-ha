from src.application.subject_extractor import extract_subject


def test_subject_extractor_samples() -> None:
    result = extract_subject(
        "Will Paul Biya be reelected President of Cameroon in 2025?"
    )
    assert "paul" in result["simplified"]
    assert "reelection" in result["simplified"]
