# CI/CD Pipeline & Repository Workflow

**Repository:** `GoSmarter-ai/mtc-extraction-benchmark`  
**Pipeline file:** `.github/workflows/ci.yml`  
**Last updated:** February 27, 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Trigger Conditions](#2-trigger-conditions)
3. [Permissions](#3-permissions)
4. [Pipeline Stages](#4-pipeline-stages)
   - 4.1 Stage 1 — Lint & Format
   - 4.2 Stage 2 — Unit Tests
   - 4.3 Stage 3 — Benchmark
5. [Execution Flow Diagram](#5-execution-flow-diagram)
6. [Artifacts](#6-artifacts)
7. [Secrets & Environment Variables](#7-secrets--environment-variables)
8. [Development Workflow](#8-development-workflow)
9. [Local Replication of CI Checks](#9-local-replication-of-ci-checks)
10. [Failure Reference](#10-failure-reference)

---

## 1. Overview

The CI/CD pipeline enforces three quality gates on every code change:

| Stage | Job | Runs on | Purpose |
|-------|-----|---------|---------|
| 1 | `lint` | All triggers | Enforce code style and formatting |
| 2 | `test` | All triggers (needs lint) | Run unit tests and collect coverage |
| 3 | `benchmark` | Push to `main` only (needs test) | Execute LLM benchmark against live models |

Stages are sequential — each stage must pass before the next begins. A PR cannot
proceed if linting fails; the benchmark will never run if tests fail.

---

## 2. Trigger Conditions

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
```

| Event | Stages Run |
|-------|-----------|
| Push to `main` | Lint → Test → Benchmark |
| Pull request targeting `main` | Lint → Test (benchmark skipped) |
| Manual trigger (`workflow_dispatch`) | Lint → Test (benchmark skipped unless manually on main) |

The benchmark stage has an additional guard:

```yaml
if: github.ref == 'refs/heads/main' && github.event_name == 'push'
```

This prevents the benchmark from running on PRs (avoiding unnecessary API calls and
costs) and on manual triggers from non-main branches.

---

## 3. Permissions

```yaml
permissions:
  contents: read
  models: read
```

The pipeline is granted the minimum permissions required:

| Permission | Reason |
|------------|--------|
| `contents: read` | Required to check out repository files |
| `models: read` | Required to call the GitHub Models inference endpoint (`models.inference.ai.azure.com`) |

No write permissions are granted. The pipeline cannot modify branches, create releases,
or write to the repository.

---

## 4. Pipeline Stages

### 4.1 Stage 1 — Lint & Format

**Job name:** `lint`  
**Runner:** `ubuntu-latest`  
**Depends on:** nothing (first stage)

```yaml
- name: Install linters
  run: pip install ruff

- name: Check linting
  run: ruff check src/

- name: Check formatting
  run: ruff format --check src/
```

**What it checks:**

`ruff check src/` applies the rules configured in `pyproject.toml`:

| Rule set | Code | Description |
|----------|------|-------------|
| pycodestyle errors | `E` | PEP 8 style errors |
| pyflakes | `F` | Unused imports, undefined names |
| pycodestyle warnings | `W` | Minor code quality warnings |
| isort | `I` | Import order and grouping |

`E501` (line too long) is explicitly ignored to avoid breaking long prompt strings and
schema path references.

`ruff format --check src/` validates that all files match `ruff`'s opinionated formatter
output. This is a read-only check — it exits with a non-zero code if any file would be
reformatted, without modifying files.

**Line length:** 100 characters (configured in `pyproject.toml`)  
**Target Python version:** 3.11

**Common failure reasons:**

- Unused imports left in after refactoring
- Import order not matching isort conventions
- Trailing whitespace or inconsistent quote style
- Long function signatures or deeply nested expressions that `ruff format` would rewrite

---

### 4.2 Stage 2 — Unit Tests

**Job name:** `test`  
**Runner:** `ubuntu-latest`  
**Depends on:** `lint`

```yaml
- name: Install system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y poppler-utils tesseract-ocr

- name: Install Python dependencies
  run: |
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install pytest pytest-cov

- name: Run tests
  run: pytest tests/ -v --cov=src --cov-report=xml

- name: Upload coverage report
  uses: actions/upload-artifact@v4
  with:
    name: coverage-report
    path: coverage.xml
```

**System dependencies:**

| Package | Purpose |
|---------|---------|
| `poppler-utils` | Required by `pdf2image` for PDF→image conversion |
| `tesseract-ocr` | Required by `pytesseract` for OCR operations |

**Python dependencies** are installed from `requirements.txt`, which includes:
`openai`, `jsonschema`, `paddleocr`, `pdf2image`, `pillow`, `opencv-python`, and others.

**Test suite (`tests/test_smoke.py`):**

The smoke tests validate fundamental structural invariants of the codebase:

```
test_imports             — Core modules import without error; LLMModelBenchmark class exists
test_ranked_models_structure — Every entry in RANKED_MODELS has id, label, provider, tier keys
test_ranked_models_order — First model in RANKED_MODELS is gpt-4o
```

These tests are intentionally lightweight — they do not make API calls or run extraction.
Their purpose is to catch import-time errors, broken refactors, or accidentally removed
constants.

**Coverage report:** Generated as `coverage.xml` (Cobertura format) and uploaded as
a workflow artifact named `coverage-report`, available for download from the Actions UI
for 90 days.

---

### 4.3 Stage 3 — Benchmark

**Job name:** `benchmark`  
**Runner:** `ubuntu-latest`  
**Depends on:** `test`  
**Condition:** Push to `main` only

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

- name: Upload benchmark results
  uses: actions/upload-artifact@v4
  with:
    name: benchmark-results-${{ github.sha }}
    path: data/processed/benchmark_output/
    retention-days: 90
```

**Flags explained:**

| Flag | Effect |
|------|--------|
| `--use-cached-ocr` | Reads pre-processed OCR text files from `data/processed/ocr_output/` instead of re-running PaddleOCR. Decouples OCR quality from LLM evaluation and makes the benchmark deterministic. |
| `--models ...` | Restricts the run to the 4 confirmed working models. The 405B Llama model is included but the benchmark will still pass within 20 minutes because all 4 models run concurrently (or sequentially within the timeout). |
| `--output ...` | Writes per-model extracted JSON files and `benchmark_summary.json` to `data/processed/benchmark_output/`. |

**Models run in CI:**

| Model | Provider | Tier |
|-------|----------|------|
| `gpt-4o` | OpenAI | Premium |
| `Meta-Llama-3.1-405B-Instruct` | Meta | Premium |
| `Meta-Llama-3.1-8B-Instruct` | Meta | Standard |
| `gpt-4o-mini` | OpenAI | Standard |

**Why Llama 3.1 405B is included despite being slow:**  
At ~472s in local testing, it fits within the 20-minute CI timeout when run with cached
OCR. It provides valuable data for tracking extraction quality over time. If it begins
causing timeouts, it should be moved to a separate scheduled workflow.

**Authentication:**  
`GITHUB_TOKEN` is automatically injected by GitHub Actions and forwarded to the script
as the `GITHUB_TOKEN` environment variable, which the OpenAI client uses as the API key
for `https://models.inference.ai.azure.com`.

**Benchmark output artifact:**  
Named `benchmark-results-{git SHA}` — tagged with the commit SHA so results can be
correlated with exact code versions. Retained for 90 days.

---

## 5. Execution Flow Diagram

```
Event: push to main
        │
        ▼
┌───────────────────────────────────┐
│  Stage 1 — lint                   │
│                                   │
│  ruff check src/                  │
│  ruff format --check src/         │
│                                   │
│  ✅ pass → continue               │
│  ❌ fail → pipeline stops         │
└─────────────────┬─────────────────┘
                  │
                  ▼
┌───────────────────────────────────┐
│  Stage 2 — test                   │
│                                   │
│  apt install poppler tesseract    │
│  pip install requirements.txt     │
│  pytest tests/ --cov=src          │
│  upload coverage.xml artifact     │
│                                   │
│  ✅ pass → continue               │
│  ❌ fail → pipeline stops         │
└─────────────────┬─────────────────┘
                  │
                  │  (only on push to main)
                  ▼
┌───────────────────────────────────┐
│  Stage 3 — benchmark              │
│                                   │
│  apt install poppler tesseract    │
│  pip install requirements.txt     │
│  python llm_models_extraction.py  │
│    --use-cached-ocr               │
│    --models gpt-4o llama405b      │
│             llama8b gpt-4o-mini   │
│  upload benchmark results         │
│                                   │
│  ✅ pass → done                   │
│  ❌ fail → investigate artifacts  │
└───────────────────────────────────┘


Event: pull request to main
        │
        ▼
  lint → test   (benchmark skipped)
```

---

## 6. Artifacts

All artifacts are uploaded via `actions/upload-artifact@v4`.

| Artifact Name | Produced By | Contents | Retention |
|---------------|-------------|----------|-----------|
| `coverage-report` | `test` stage | `coverage.xml` — Cobertura-format unit test coverage | 90 days |
| `benchmark-results-{sha}` | `benchmark` stage | `benchmark_summary.json`, per-model `*_extracted.json` files | 90 days |

Artifacts can be downloaded from the **Actions** tab in the GitHub repository UI by
selecting a workflow run and scrolling to the **Artifacts** section.

---

## 7. Secrets & Environment Variables

| Name | Type | Used by | Purpose |
|------|------|---------|---------|
| `GITHUB_TOKEN` | Auto-injected by Actions | `benchmark` stage | Authenticates against GitHub Models inference endpoint |

No additional secrets need to be configured. `GITHUB_TOKEN` is automatically available
in all GitHub Actions runs with the permissions declared in the workflow file.

---

## 8. Development Workflow

### Standard contribution flow

```
1. Create a feature branch from main
   git checkout -b feat/my-change

2. Make changes to src/ or tests/

3. Run lint and format locally before pushing
   ruff check src/ tests/
   ruff format src/ tests/

4. Run tests locally
   pytest tests/ -v

5. Push branch and open a pull request to main
   git push origin feat/my-change

6. CI runs: lint → test
   Both must pass for merge to be allowed

7. Merge PR to main → CI runs: lint → test → benchmark
   Benchmark results are uploaded as artifacts tagged with commit SHA
```

### Hotfix flow

Same as above. There is no separate hotfix branch — all changes go through the PR
process, which runs lint and test automatically.

### When the benchmark fails

1. Download the `benchmark-results-{sha}` artifact from the failed run
2. Inspect `benchmark_summary.json` for error messages per model
3. Check if the failure is a model availability issue (`unknown_model` HTTP 400)
   or an extraction/validation error
4. If a model has become unavailable, remove it from `RANKED_MODELS` and from
   the `--models` flag in `ci.yml`

---

## 9. Local Replication of CI Checks

Run the exact same checks as CI before pushing:

```bash
# Stage 1 — Lint
pip install ruff
ruff check src/ tests/
ruff format --check src/ tests/

# Auto-fix formatting if needed
ruff format src/ tests/

# Stage 2 — Tests
pip install -r requirements.txt pytest pytest-cov
pytest tests/ -v --cov=src --cov-report=term-missing

# Stage 3 — Benchmark (optional, requires GITHUB_TOKEN)
export GITHUB_TOKEN=your_token_here
python src/extraction/llm_models_extraction.py \
    --use-cached-ocr \
    --models gpt-4o gpt-4o-mini \
    --output data/processed/benchmark_output
```

---

## 10. Failure Reference

| Symptom | Stage | Likely Cause | Fix |
|---------|-------|-------------|-----|
| `ruff check` exits non-zero | lint | Unused import, undefined name, wrong import order | Run `ruff check --fix src/` to auto-fix, then review |
| `ruff format --check` exits non-zero | lint | File not formatted to ruff style | Run `ruff format src/` |
| `ImportError` in tests | test | Missing dependency in `requirements.txt` or broken `__init__.py` | Add missing package; check `src/extraction/__init__.py` exists |
| `AssertionError: RANKED_MODELS[0]["id"] == "gpt-4o"` | test | Wrong model placed first in the list | Restore `gpt-4o` as first entry in `RANKED_MODELS` |
| `unknown_model` HTTP 400 | benchmark | Model retired from GitHub Models endpoint | Remove model from `RANKED_MODELS` and `ci.yml --models` arg |
| `timeout-minutes: 20` exceeded | benchmark | A model is taking too long | Remove slow model from CI run; use scheduled workflow instead |
| `jsonschema.ValidationError` | benchmark | LLM returned invalid structure after 3 retries | Inspect model output in benchmark artifact; update prompt if needed |


parallelisation. all ten of them, forward pass, and output results, and compare to do the benchmark

add a caching step - python. 

multimodal sites. dont have to do the ocr seperately.

