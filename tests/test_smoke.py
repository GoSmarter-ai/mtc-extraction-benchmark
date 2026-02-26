"""Smoke tests — verify imports and basic structures."""


def test_imports():
    """Verify core modules can be imported."""
    from src.extraction import llm_models_extraction

    assert hasattr(llm_models_extraction, "RANKED_MODELS")
    assert hasattr(llm_models_extraction, "LLMModelBenchmark")


def test_ranked_models_structure():
    """Verify model list has expected fields."""
    from src.extraction.llm_models_extraction import RANKED_MODELS

    assert len(RANKED_MODELS) > 0
    for model in RANKED_MODELS:
        assert "id" in model
        assert "label" in model
        assert "provider" in model
        assert "tier" in model


def test_ranked_models_order():
    """Top model should be gpt-4o."""
    from src.extraction.llm_models_extraction import RANKED_MODELS

    assert RANKED_MODELS[0]["id"] == "gpt-4o"
