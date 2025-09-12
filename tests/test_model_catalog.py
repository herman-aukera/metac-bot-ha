from src.infrastructure.openrouter_model_catalog import ModelInfo, filter_available


def test_model_catalog_filters_unavailable() -> None:
    catalog = [ModelInfo(name="openai/gpt-5"), ModelInfo(name="moonshotai/kimi-k2:free"), ModelInfo(name="dead/model", available=False)]
    requested = ["openai/gpt-5", "dead/model", "moonshotai/kimi-k2:free"]
    filtered = filter_available(requested, catalog)
    assert filtered == ["openai/gpt-5", "moonshotai/kimi-k2:free"]
