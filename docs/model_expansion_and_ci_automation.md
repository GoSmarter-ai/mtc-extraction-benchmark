# Model Expansion, CI Automation & Developer Tooling

> Covers the changes introduced in commits `d0cbeaa` → `4cca2fb` on `main`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [What Changed](#2-what-changed)
   - 2.1 [Multi-Provider Model Registry](#21-multi-provider-model-registry)
   - 2.2 [Per-Provider Client Factory (`_make_client`)](#22-per-provider-client-factory-_make_client)
   - 2.3 [HuggingFace / Qwen2.5 Models](#23-huggingface--qwen25-models)
   - 2.4 [Auto-Discovery: Env-Var Gating](#24-auto-discovery-env-var-gating)
   - 2.5 [Skip-Existing Cache (`--skip-existing`)](#25-skip-existing-cache---skip-existing)
   - 2.6 [New CLI Flags](#26-new-cli-flags)
   - 2.7 [CI Pipeline Overhaul (`.github/workflows/ci.yml`)](#27-ci-pipeline-overhaul-githubworkflowsciyml)
   - 2.8 [Quarto Benchmark Report (`benchmark_report.qmd`)](#28-quarto-benchmark-report-benchmark_reportqmd)
   - 2.9 [Pre-Commit Hook (ruff format + check)](#29-pre-commit-hook-ruff-format--check)
3. [Why Each Change Was Made](#3-why-each-change-was-made)
4. [How to Add a New Model](#4-how-to-add-a-new-model)
5. [How to Run Locally](#5-how-to-run-locally)
6. [Architecture Diagram](#6-architecture-diagram)

---

## 1. Overview

The benchmark runner originally supported only four models, all served through **GitHub Models** (`https://models.inference.ai.azure.com`). Every model was hardcoded in the CI job, and adding a new provider required changes in at least five places.

This update transforms the runner into a **self-maintaining, multi-provider benchmark**:

- A single registry (`ALL_MODELS`) drives everything — the CLI, CI, and skip-logic all read from it.
- New providers (HuggingFace, and any future endpoint) are added by appending one dict to a list and setting one repo secret.
- The CI pipeline has three layers of caching so re-runs that don't change inputs are near-instant.
- A Quarto report (`benchmark_report.qmd`) renders interactive charts from the benchmark output.
- A git pre-commit hook ensures `ruff format` and `ruff check --fix` are always applied before code reaches CI, eliminating formatting-related lint failures.

---

## 2. What Changed

### 2.1 Multi-Provider Model Registry

**File:** `src/extraction/llm_models_extraction.py` (lines 75–138)

Each model definition was extended with two new keys:

| Key | Purpose |
|---|---|
| `base_url` | OpenAI-compatible API endpoint for this provider |
| `api_key_env` | Name of the environment variable that holds the API key |

```python
_GH_BASE = "https://models.inference.ai.azure.com"

RANKED_MODELS: List[Dict[str, str]] = [
    {
        "id": "gpt-4o",
        "label": "GPT-4o",
        "provider": "OpenAI",
        "tier": "top",
        "base_url": _GH_BASE,
        "api_key_env": "GITHUB_TOKEN",   # ← new
    },
    # … three more GitHub Models entries
]
```

A fast lookup dict is derived automatically:

```python
ALL_MODELS_REGISTRY: Dict[str, dict] = {m["id"]: m for m in ALL_MODELS}
```

**Why:** Previously `base_url` and the token name were scattered as hardcoded strings throughout the class. Centralising them in the model dict makes every downstream consumer (client factory, env-var gating, CI) pull from a single source of truth.

---

### 2.2 Per-Provider Client Factory (`_make_client`)

**File:** `src/extraction/llm_models_extraction.py` (lines 168–186)

Replaced the single shared `OpenAI` client stored at `self.client` with a factory method:

```python
def _make_client(self, model_id: str) -> OpenAI:
    cfg = self._model_registry[model_id]
    base_url = cfg.get("base_url", _GH_BASE)
    api_key_env = cfg.get("api_key_env", "GITHUB_TOKEN")

    cache_key = f"{base_url}::{api_key_env}"
    if cache_key not in self._client_cache:
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"{api_key_env} is not set. Required for model '{model_id}' ({base_url})."
            )
        self._client_cache[cache_key] = OpenAI(base_url=base_url, api_key=api_key)
    return self._client_cache[cache_key]
```

Clients are **cached by `(base_url, api_key_env)`**, meaning all models on the same provider (e.g., both Qwen models) share one `OpenAI` instance — no redundant connection setup.

**Why:** A single hardcoded `OpenAI(base_url=..., api_key=os.environ["GITHUB_TOKEN"])` in `__init__` cannot support multiple providers. The factory pattern also defers client creation until the model is actually called, so unused providers never raise `EnvironmentError`.

---

### 2.3 HuggingFace / Qwen2.5 Models

**File:** `src/extraction/llm_models_extraction.py` (lines 115–136)

```python
_HF_BASE = "https://api-inference.huggingface.co/v1"

HF_MODELS: List[Dict[str, str]] = [
    {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "label": "Qwen2.5 72B",
        "provider": "Qwen/Alibaba",
        "tier": "top",
        "base_url": _HF_BASE,
        "api_key_env": "HF_TOKEN",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen2.5 7B",
        "provider": "Qwen/Alibaba",
        "tier": "small",
        "base_url": _HF_BASE,
        "api_key_env": "HF_TOKEN",
    },
]

ALL_MODELS: List[Dict[str, str]] = RANKED_MODELS + HF_MODELS
```

HuggingFace's Inference API exposes an **OpenAI-compatible `/v1/chat/completions` endpoint**, so the same `openai` Python package works with no additional dependencies — only `base_url` and the token differ.

**Why:** Open-source models (Qwen2.5, Llama, etc.) hosted on HuggingFace give a cost-efficient, non-proprietary comparison point. Qwen2.5-72B in particular is competitive with GPT-4o on many structured-extraction tasks.

---

### 2.4 Auto-Discovery: Env-Var Gating

**File:** `src/extraction/llm_models_extraction.py` (lines ~1484–1498, inside `main()`)

After the CLI resolves the candidate model list, every model is checked against its required env var. Models whose secret is absent are silently skipped:

```python
available_ids: List[str] = []
for mid in model_ids:
    key_env = ALL_MODELS_REGISTRY.get(mid, {}).get("api_key_env", "GITHUB_TOKEN")
    if os.environ.get(key_env):
        available_ids.append(mid)
    else:
        print(f"   ⏭️  Auto-skipping {mid!r} — {key_env} not set")
model_ids = available_ids

if not model_ids:
    print("❌ No models available — check that at least GITHUB_TOKEN is set.")
    return 1
```

**Why:** Without this, adding `Qwen/Qwen2.5-72B-Instruct` to `HF_MODELS` would cause CI to crash with `EnvironmentError` whenever `HF_TOKEN` was not set. With env-var gating, the model simply doesn't run — zero friction, zero CI breakage. Adding `HF_TOKEN` as a repo secret is the only step needed to activate HF models in CI.

---

### 2.5 Skip-Existing Cache (`--skip-existing`)

**File:** `src/extraction/llm_models_extraction.py` (lines ~1109–1185, `benchmark()` method)

When `--skip-existing` is passed and a per-model output file already exists in `--output`, the benchmark loads the JSON from disk instead of making an API call:

```python
if skip_existing and output_dir:
    safe_name = model_id.replace("/", "_").replace(".", "_")
    cached_file = output_dir / f"{safe_name}_extracted.json"
    if cached_file.exists():
        cached_result = json.loads(cached_file.read_text())
        run = {
            "result": cached_result,
            "elapsed": 0.0,
            "status": "cached",
            "error": None,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        run["metrics"] = self.compute_metrics(cached_result, ground_truth)
        if ground_truth:
            run["field_metrics"] = MTCEvaluator.evaluate(cached_result, ground_truth)
        results[model_id] = run
        continue            # ← skips the API call entirely
```

`MTCEvaluator.evaluate()` is still run against the cached result so scores in `benchmark_summary.json` remain up to date whenever ground truth changes.

**Why:** Without this, every CI run calls every model's API even when the document and prompt haven't changed — wasting tokens and introducing rate-limit failures. With `--skip-existing`, only genuinely new models (or models whose cached file was invalidated by the output cache key) make API calls.

---

### 2.6 New CLI Flags

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--providers github huggingface` | `nargs="+"` | `None` (all) | Restrict which providers to include in a run |
| `--rate-limit-sleep N` | `float` | `0.0` | Sleep N seconds between API calls — avoids 429s on HuggingFace free tier |
| `--skip-existing` | `store_true` | `False` | Load cached JSON instead of calling the API for already-extracted models |

---

### 2.7 CI Pipeline Overhaul (`.github/workflows/ci.yml`)

The `benchmark` job was rewritten from scratch.

#### Before (hardcoded, no cache)

```yaml
- name: Run benchmark (confirmed working models)
  run: |
    python src/extraction/llm_models_extraction.py \
      --use-cached-ocr \
      --models gpt-4o Meta-Llama-3.1-405B-Instruct Meta-Llama-3.1-8B-Instruct gpt-4o-mini \
      --output data/processed/benchmark_output
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  timeout-minutes: 20
```

Problems:
- `--models` hardcoded — adding a new model required editing the CI file
- No `HF_TOKEN`, so HuggingFace models could never run
- No benchmark output cache — every push pays full API cost for all models
- No Python bytecode cache
- No Quarto report generation

#### After (auto-discovery, 3-layer cache, report rendering)

```yaml
# Cache 1: pip packages
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: benchmark-${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

# Cache 2: Python __pycache__ (import time savings)
- uses: actions/cache@v4
  with:
    path: |
      src/**/__pycache__
      .pytest_cache
    key: pycache-${{ runner.os }}-py311-${{ hashFiles('src/**/*.py') }}

# Cache 3: Benchmark outputs (skip API calls for unchanged inputs)
- uses: actions/cache@v4
  with:
    path: data/processed/benchmark_output
    key: bench-out-${{ runner.os }}-${{ hashFiles('data/processed/pipeline_output/text/**', 'prompts/**', 'schema/**') }}

- name: Run benchmark (all available models)
  run: |
    python src/extraction/llm_models_extraction.py \
      --use-cached-ocr \
      --skip-existing \
      --ground-truth data/ground_truth/diler-07-07-2025-rerun-41-44_gt.json \
      --output data/processed/benchmark_output
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  timeout-minutes: 45

- uses: quarto-dev/quarto-actions/setup@v2
- name: Render benchmark report
  run: quarto render benchmark_report.qmd
  continue-on-error: true

- uses: actions/upload-artifact@v4
  with:
    name: benchmark-results-${{ github.sha }}
    path: |
      data/processed/benchmark_output/
      benchmark_report.html
    retention-days: 90
```

**Cache layer rationale:**

| Layer | Cache key invalidation trigger | Effect when invalidated |
|---|---|---|
| pip | `requirements.txt` changes | Re-downloads packages |
| `__pycache__` | Any `src/**/*.py` change | Re-compiles bytecode |
| Benchmark outputs | OCR text, prompts, or schema changes | Re-runs all API calls |

When the benchmark output cache hits, `--skip-existing` loads every model's JSON from disk — **zero API calls, full scores recalculated** against potentially updated ground truth.

---

### 2.8 Quarto Benchmark Report (`benchmark_report.qmd`)

A new self-contained Quarto report generates visual comparisons across all benchmarked models. It renders to `benchmark_report.html` and is uploaded as a CI artifact alongside the raw JSON outputs.

**Report sections:**

| Section | Chart type | What it shows |
|---|---|---|
| Model Registry | Table | All models, providers, tiers, endpoints |
| Accuracy Overview | Grouped bar chart (6 metrics × models) | Precision, recall, F1 across document / chemical / mechanical / approval categories |
| Radar / Spider | Polar axes (one polygon per model) | Multi-axis performance fingerprint |
| Latency vs Quality | Scatter (bubble size = total tokens) | Time-quality tradeoff |
| Token Usage & Cost | Stacked bar + estimated cost | Prompt vs completion tokens, price estimate |
| Per-Field Heatmap | `imshow` with RdYlGn colormap | Field-by-field accuracy across models |
| Document Field Detail | Pivot table (✓/✗) | Per-document per-field extraction result |
| Error / Failure Breakdown | Table | Models that errored, failure reasons |

The report handles legacy benchmark runs (where `field_metrics` was not stored) by re-running `MTCEvaluator.evaluate()` on the saved per-model JSON files automatically.

---

### 2.9 Pre-Commit Hook (ruff format + check)

**File:** `.git/hooks/pre-commit` (local, not tracked)

```bash
#!/usr/bin/env bash
set -e

STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)
if [ -n "$STAGED" ]; then
    ruff format $STAGED          # auto-format (Black-compatible)
    ruff check --fix $STAGED     # auto-fix lint issues (e.g. unused imports)
    git add $STAGED              # re-stage reformatted files
fi
```

**Why:** The CI lint job runs two checks:

1. `ruff check src/` — catches rule violations (F541, E, W codes)
2. `ruff format --check src/` — rejects files that aren't formatted

Both caused CI failures on freshly committed code. The pre-commit hook runs both tools in auto-fix mode **before the commit is recorded**, so formatting issues never reach GitHub.

The hook only processes files that are already staged (`git diff --cached`), so it is fast and non-intrusive.

> **Note:** `.git/hooks/` is not committed to the repository. New contributors need to run the hook setup once:
>
> ```bash
> cp .git/hooks/pre-commit .git/hooks/pre-commit  # already done for you in this codespace
> chmod +x .git/hooks/pre-commit
> ```
>
> For a shareable alternative, see the `pre-commit` framework section below.

#### Optional: share via `pre-commit` framework

Install the `pre-commit` package and add `.pre-commit-config.yaml` to track the same rules:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Then any contributor runs `pre-commit install` once after cloning and gets the same protection.

---

## 3. Why Each Change Was Made

| Change | Primary motivation |
|---|---|
| `base_url` / `api_key_env` in model dicts | Single source of truth; eliminates scattered hardcoded strings |
| `_make_client` factory | Support multiple providers without a breaking refactor; lazy init avoids `EnvironmentError` for unused providers |
| `HF_MODELS` + Qwen2.5 | Extend benchmarking to open-source models; HuggingFace's OpenAI-compatible endpoint means zero new dependencies |
| Env-var gating in `main()` | CI must not crash when `HF_TOKEN` is absent; allows gradual secret rollout |
| `--skip-existing` | Avoid re-spending API tokens on unchanged documents; enables cheap re-scoring when ground truth is updated |
| `--providers` filter | Quick local runs against one provider without editing code |
| `--rate-limit-sleep` | HuggingFace free tier enforces per-minute rate limits; graceful sleep avoids 429 errors |
| CI 3-layer cache | Cuts repeat benchmark costs to near-zero when inputs are unchanged |
| CI `--models`-free invocation | Any model added to `ALL_MODELS` + its secret added to the repo runs automatically — no CI file edit needed |
| Quarto report | Stakeholders need visual comparisons, not raw JSON; report is generated as a CI artifact so it's available after every push |
| Pre-commit `ruff` hook | Formatting failures kept breaking CI; auto-fixing at commit time eliminates the entire class of problem |

---

## 4. How to Add a New Model

### Step 1 — Add the model to the registry

Open `src/extraction/llm_models_extraction.py` and append to the appropriate list:

**GitHub Models** (uses `GITHUB_TOKEN`):
```python
RANKED_MODELS.append({
    "id": "Mistral-large-2",
    "label": "Mistral Large 2",
    "provider": "Mistral AI",
    "tier": "top",
    "base_url": _GH_BASE,
    "api_key_env": "GITHUB_TOKEN",
})
```

**HuggingFace Inference API** (uses `HF_TOKEN`):
```python
HF_MODELS.append({
    "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "label": "Mixtral 8x7B",
    "provider": "Mistral AI",
    "tier": "top",
    "base_url": _HF_BASE,
    "api_key_env": "HF_TOKEN",
})
```

**Custom / self-hosted endpoint**:
```python
MY_MODELS = [
    {
        "id": "my-custom-model",
        "label": "My Model",
        "provider": "Internal",
        "tier": "top",
        "base_url": "https://my-api.example.com/v1",
        "api_key_env": "MY_API_KEY",
    }
]
ALL_MODELS = RANKED_MODELS + HF_MODELS + MY_MODELS
ALL_MODELS_REGISTRY = {m["id"]: m for m in ALL_MODELS}
```

### Step 2 — Add the secret to the repo

Go to **Settings → Secrets and variables → Actions** and add:

- `HF_TOKEN` for any model with `api_key_env: "HF_TOKEN"`
- `MY_API_KEY` for any custom provider

### Step 3 — Push

On the next push to `main`, CI automatically:
1. Detects the new model via `ALL_MODELS`
2. Checks the secret is available — if not, skips the model gracefully
3. Calls the API for the new model (no `--skip-existing` cache hit yet)
4. Saves `<model_id>_extracted.json` to `data/processed/benchmark_output/`
5. Re-runs all cached models from disk
6. Renders `benchmark_report.html` with the new model included

---

## 5. How to Run Locally

```bash
# All available models (auto-skips those missing env vars)
python src/extraction/llm_models_extraction.py \
    --use-cached-ocr \
    --skip-existing \
    --ground-truth data/ground_truth/diler-07-07-2025-rerun-41-44_gt.json \
    --output data/processed/benchmark_output

# GitHub Models only
python src/extraction/llm_models_extraction.py \
    --use-cached-ocr --providers github \
    --output data/processed/benchmark_output

# HuggingFace only, with rate-limit protection
python src/extraction/llm_models_extraction.py \
    --use-cached-ocr --providers huggingface \
    --rate-limit-sleep 2 \
    --output data/processed/benchmark_output

# Render the comparison report
quarto render benchmark_report.qmd
open benchmark_report.html
```

---

## 6. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                      ALL_MODELS registry                         │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐   │
│  │      RANKED_MODELS      │  │         HF_MODELS            │   │
│  │  gpt-4o                 │  │  Qwen/Qwen2.5-72B-Instruct   │   │
│  │  Meta-Llama-3.1-405B    │  │  Qwen/Qwen2.5-7B-Instruct    │   │
│  │  Meta-Llama-3.1-8B      │  │  (api_key_env: HF_TOKEN)     │   │
│  │  gpt-4o-mini            │  └──────────────────────────────┘   │
│  │  (api_key_env: GH_TOKEN)│                                     │
│  └─────────────────────────┘                                     │
└──────────────────────┬───────────────────────────────────────────┘
                       │ read by
          ┌────────────▼────────────┐
          │   main() — env-var gate │  ← auto-skips missing secrets
          └────────────┬────────────┘
                       │ filtered model_ids
          ┌────────────▼────────────────────────────┐
          │        benchmark() per-model loop        │
          │                                          │
          │  skip_existing? ──yes──► load JSON       │
          │       │                  re-score vs GT  │
          │       no                                 │
          │       ▼                                  │
          │  _make_client(model_id)                  │
          │  ┌──────────────────────────────────┐    │
          │  │ provider cache (base_url::key)   │    │
          │  │  GitHub endpoint  │  HF endpoint │    │
          │  └──────────────────────────────────┘    │
          │       ▼                                  │
          │  API call → save JSON → MTCEvaluator     │
          └────────────┬────────────────────────────-┘
                       │ benchmark_summary.json
          ┌────────────▼────────────┐
          │  quarto render          │
          │  benchmark_report.html  │
          └─────────────────────────┘

CI caches
  Layer 1: ~/.cache/pip           (key: requirements.txt hash)
  Layer 2: src/**/__pycache__     (key: src/**/*.py hash)
  Layer 3: benchmark_output/      (key: OCR text + prompts + schema hash)
```
