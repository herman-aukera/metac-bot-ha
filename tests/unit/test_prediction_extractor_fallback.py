import math


def test_extract_last_percentage_value_default():
    import main as m

    # If forecasting_tools' extractor is available, it raises on no-percent text.
    if getattr(m, "_FTPredictionExtractor", None) is not None:
        import pytest

        with pytest.raises(Exception):
            m.PredictionExtractor.extract_last_percentage_value("no percents here")
    else:
        val = m.PredictionExtractor.extract_last_percentage_value("no percents here")
        assert 0.49 < val < 0.51


def test_extract_last_percentage_value_parses_number():
    import main as m

    val = m.PredictionExtractor.extract_last_percentage_value("Foo 75% bar")
    assert math.isclose(val, 0.75, rel_tol=0, abs_tol=1e-9)


def test_extract_option_list_with_percentage_afterwards_simple_lines():
    import main as m

    text = "A: 10%\nB: 20%\nC: 70%\n"
    opts = ["A", "B", "C"]
    res = m.PredictionExtractor.extract_option_list_with_percentage_afterwards(
        text, opts
    )
    assert res is not None
    probs = [o.probability for o in res.predicted_options]
    assert len(probs) == 3
    # Should be normalized to 1.0 exactly or very close
    assert math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-9)
