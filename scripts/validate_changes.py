"""Quick validation that Phase 1 & 2 changes work correctly."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.llm_models_extraction import (
    ALL_MODELS,
    ALL_MODELS_REGISTRY,
    RANKED_MODELS,
    HF_MODELS,
    LLMModelBenchmark,
    MTCEvaluator,
)

assert len(HF_MODELS) == 2, f"Expected 2 HF models, got {len(HF_MODELS)}"
for m in ALL_MODELS:
    assert "base_url" in m, f"Missing base_url in {m['id']}"
    assert "api_key_env" in m, f"Missing api_key_env in {m['id']}"
assert "Qwen/Qwen2.5-72B-Instruct" in ALL_MODELS_REGISTRY
assert "gpt-4o" in ALL_MODELS_REGISTRY
assert MTCEvaluator is not None

print("✅  All assertions passed.")
print("Models in registry:")
for m in ALL_MODELS:
    print(f"  [{m['api_key_env']:14}]  {m['id']}")
