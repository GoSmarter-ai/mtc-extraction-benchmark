# Implementation: Phases 1–4 — Why and How

## Background

The project started as a single Python script (`llm_models_extraction.py`) that ran from the command
line. It benchmarked LLM models for extracting structured data from Mill Test Certificates (MTC),
but had several fundamental gaps:

| Gap | Impact |
|---|---|
| No HTTP interface | External tools, frontends, and CI consumers had no way to call the pipeline |
| No standalone evaluator | Accuracy measurement was buried inside the benchmark run; could not be called in isolation |
| OCR path only | When OCR quality was low, there was no fallback; the LLM worked on bad text |
| No ground truth | Without annotated reference data, F1 scores were theoretical |
| No container build | Deployment required manually replicating the environment |

Phases 1–4 addressed each of these in order.

---

## Phase 1 — Evaluation Infrastructure

### Why

The benchmark produced raw counts (number of mechanical rows extracted, number of chemical heats)
but had no way to answer the question: *"Is row 14 in gpt-4o's output the same row as row 14 in the
real document?"* The root cause was that the existing metric matched rows by an exact composite key
`(heat_number, test_sample)`. In practice, `test_sample` is frequently `null`, which caused
identical physical rows to be treated as different, inflating false-positive and false-negative counts.

A standalone evaluator was also needed so that:
- Other code (the API, scripts, notebooks) could import it without pulling in the entire benchmark
- The tolerance parameters could be tuned per call without re-running the whole pipeline

### What was built

**`src/evaluation/evaluator.py`** — `MTCEvaluator` class with static methods:

```python
MTCEvaluator.evaluate(extracted, ground_truth, numeric_tolerance=0.001,
                      mech_weight_tol=0.01, mech_yield_tol=5.0)
```

Returns a structured dict with four sections and an `overall_f1`:

| Section | What it measures |
|---|---|
| `document` | Exact string match on certificate_number, issuing_date, standard, customer, order_number |
| `chemical` | Heat-level Precision / Recall / F1 by `heat_number`; element-level accuracy within matched heats |
| `mechanical` | Row-level Precision / Recall / F1 using **fuzzy key matching**; property-level accuracy within matched rows |
| `approval` | Exact match on certificate_of_approval_number, form_number, cares_approved |

#### Fuzzy mechanical key matching (key improvement)

The old approach:

```python
key = (heat_number, test_sample)  # fails when test_sample is None
```

The new approach finds the best GT match for every extracted row by minimising physical distance:

```python
# Match extracted row to GT row in the same heat where:
# |weight_kg_per_m difference| <= 0.01
# |yield_point_mpa difference| <= 5.0 MPa
```

This correctly handles rows where `test_sample` is `null`, collapsing rows with minor numeric
rounding, and avoids false mismatches caused by OCR transcription noise.

**`src/evaluation/__init__.py`** — makes evaluation a proper Python package:

```python
from src.evaluation import MTCEvaluator
```

**`data/ground_truth/diler-07-07-2025-rerun-41-44_gt.json`** — seeded from the gpt-4o extraction as
a starting template. Every value must be verified against the source PDF before use.

**`data/ground_truth/README.md`** — annotation guide explaining field priority order and how to use
the evaluator CLI.

The evaluator also has a CLI entry point:

```bash
python -m src.evaluation.evaluator \
  --prediction data/processed/benchmark_output/gpt-4o_extracted.json \
  --ground-truth data/ground_truth/diler-07-07-2025-rerun-41-44_gt.json \
  --output data/comparison/gpt-4o_vs_gt.json
```

---

## Phase 2 — Vision Extraction

### Why

All extraction up to this point used OCR-produced text as input. When PaddleOCR produces low
confidence output — for example on pages with rubber stamps, rotated annotations, or merged table
cells — the LLM is working from corrupted text and cannot recover the correct values regardless of
how good the model is.

The fix is to skip OCR entirely for problematic pages and pass the raw page image directly to a
multimodal LLM. gpt-4o natively understands document layout, table structure, and can read text
from images in a way that an OCR pipeline cannot.

### What was built

**`src/extraction/vision_extraction.py`** — `VisionExtractor` class:

```
PDF → pdf2image (JPEG per page) → base64 encode → OpenAI vision message → JSON + schema validation
```

Key design decisions:

| Decision | Reason |
|---|---|
| Downscale to 2048px max on long edge | Reduces image token cost by ~60–70% with negligible quality loss for typed text |
| JPEG quality=92 | Balance between file size and legibility of small text |
| 3× retry with schema validation | Same robustness pattern as the text pipeline |
| Per-page extraction then merge | Avoids exceeding context window on multi-page certificates |
| Composite-key deduplication on merge | Prevents duplicate chemical heats and mechanical rows appearing when the same heat spans two pages |

The extractor uses the same GitHub Models endpoint (`models.inference.ai.azure.com`) with `gpt-4o`
as the default, so no additional credentials are needed beyond `GITHUB_TOKEN`.

---

## Phase 3 — Hybrid Pipeline

### Why

Having two extraction paths (text and vision) creates a new problem: which one should run for a
given document? Running vision unconditionally is expensive (more tokens, higher latency). Running
text unconditionally fails on low-quality scans.

The hybrid pipeline solves this by making the routing decision automatically, based on measurable
signal (OCR confidence) rather than a manual per-document choice.

### What was built

**`src/extraction/hybrid_pipeline.py`** — `HybridPipeline` class with three-tier routing:

```
OCR confidence ≥ 0.85  →  Rule-based extraction
                            ↳ if completeness < 60%  →  Text LLM
                                                        ↳ if completeness < 60%  →  Vision LLM

OCR confidence 0.65–0.85  →  Text LLM
                               ↳ if completeness < 60%  →  Vision LLM

OCR confidence < 0.65  →  Vision LLM directly
```

**Thresholds (configurable):**

| Threshold | Default | Meaning |
|---|---|---|
| `high_confidence_threshold` | 0.85 | Above this, rule-based is tried first |
| `medium_confidence_threshold` | 0.65 | Below this, skip text-LLM entirely |
| `min_completeness` | 0.60 | Fraction of expected fields that must be non-null |

**Completeness check** counts six high-signal fields:
`certificate_number`, `issuing_date`, `standard`, `consignment_number`, `product.size`,
`product.quality` — plus the presence of any chemical or mechanical rows. A result scoring below
60% on this heuristic is considered incomplete and triggers escalation.

**`_pipeline_meta` in the response** provides full transparency:

```json
"_pipeline_meta": {
  "strategy_used": "text_llm",
  "ocr_confidence": 0.783,
  "elapsed_seconds": 18.4,
  "tokens_used": 12840,
  "text_model": "gpt-4o",
  "vision_model": "gpt-4o"
}
```

---

## Phase 4 — FastAPI REST Service

### Why

Before Phase 4, the only interface was the CLI. This meant:
- No programmatic access from other services or frontends
- No standard way to handle authentication, input validation, or structured error responses
- Manual file management to pass inputs and retrieve outputs
- Impossible to deploy as a standalone service

### What was built

```
src/api/
├── __init__.py
├── main.py            — FastAPI app, CORS, mounts routers
├── models.py          — Pydantic request/response contracts
└── routes/
    ├── __init__.py
    ├── extract.py     — POST /extract
    └── benchmark.py   — GET /models, POST /benchmark, POST /evaluate
```

#### Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok"}` |
| `GET` | `/models` | Lists all ranked models with id, label, provider, tier |
| `POST` | `/extract` | Accepts a PDF (multipart upload) → returns extracted JSON |
| `POST` | `/benchmark` | Runs extraction across multiple models against cached OCR |
| `POST` | `/evaluate` | Scores a prediction file against a ground-truth file |
| `GET` | `/docs` | Auto-generated Swagger UI |

#### `POST /extract` in detail

The endpoint accepts `multipart/form-data`:

```
pdf         file    (required) MTC certificate PDF
model       string  gpt-4o (default)
mode        string  text | vision | hybrid (default: text)
two_pass    bool    false (default)
```

Internally it:
1. Validates the mode and file extension before touching any pipeline code
2. Writes the PDF to a secure temp file (avoids path traversal)
3. Routes to `VisionExtractor`, `HybridPipeline`, or `LLMModelBenchmark` based on `mode`
4. Deletes the temp file in a `finally` block regardless of outcome
5. Returns a typed `ExtractionResponse` or an HTTP 500 with a message

#### Why Pydantic models (`models.py`)

| Old (CLI) | New (API + Pydantic) |
|---|---|
| Wrong flag → silent wrong default or Python traceback | Wrong field type → `422 Unprocessable Entity` with per-field error detail |
| No schema for the response | `ExtractionResponse` guarantees `run_id`, `elapsed_seconds`, `tokens_used`, `validation_errors` are always present |
| Tolerance flags were global per run | `EvaluationRequest` has independent `mech_weight_tol` and `mech_yield_tol` per call |
| Caller had to parse a file on disk | Caller gets the result in the HTTP response body |
| No API documentation | `/docs` Swagger UI auto-generated from `Field(description=...)` strings |

---

## CI/CD — Stage 4: Docker Build

### Why

Even with a working FastAPI service, there was no guarantee the containerised version was buildable.
A broken `Dockerfile` would only be discovered at deployment time, not at PR review time.

A dedicated CI stage that builds (but does not push) the Docker image on every main-branch push
gives an early signal if the Dockerfile is broken.

### What was built

**`Dockerfile`** — multi-stage build:

```
Stage 1 (builder):   python:3.11-slim + system deps + pip install requirements.txt
Stage 2 (runtime):   python:3.11-slim + system deps + copy site-packages from builder
                      → smaller final image, no build tools in production layer
```

The final image:
- Copies only `src/`, `schema/`, `prompts/` (data and notebooks excluded via `.dockerignore`)
- Sets `PYTHONPATH=/app` so imports work correctly
- Starts `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`

**`.dockerignore`** — excludes `data/`, `tests/`, `notebooks/`, `docs/`, `report_files/` from the
build context. Without this, the ~100 MB data directory would be sent to the Docker daemon on every
build even though none of it ends up in the image.

**CI Stage 4 in `.github/workflows/ci.yml`:**

```yaml
docker-build:
  needs: test
  if: github.ref == 'refs/heads/main'
  steps:
    - uses: docker/setup-buildx-action@v3
    - uses: docker/build-push-action@v5
      with:
        push: false
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

GitHub Actions cache (`type=gha`) stores BuildKit layer cache between runs. The base layer
(`python:3.11-slim` + system deps + pip install) is typically 3–5 minutes to build cold but
completes in ~30 seconds on a cache hit.

This stage runs **in parallel with the benchmark stage** (both depend on `test`, neither depends on
each other), so it adds no extra wall-clock time to the pipeline.

---

## Summary of Changes

| File | Type | Purpose |
|---|---|---|
| `src/evaluation/__init__.py` | New | Package init |
| `src/evaluation/evaluator.py` | New | Standalone evaluator with fuzzy matching |
| `data/ground_truth/diler-..._gt.json` | New | Seeded GT template (needs human verification) |
| `data/ground_truth/README.md` | New | Annotation guide |
| `src/extraction/vision_extraction.py` | New | PDF image → vision LLM extraction |
| `src/extraction/hybrid_pipeline.py` | New | OCR confidence-based routing pipeline |
| `src/api/__init__.py` | New | Package init |
| `src/api/main.py` | New | FastAPI app entry point |
| `src/api/models.py` | New | Pydantic request/response contracts |
| `src/api/routes/extract.py` | New | POST /extract endpoint |
| `src/api/routes/benchmark.py` | New | GET /models, POST /benchmark, POST /evaluate |
| `Dockerfile` | New (replaced stub) | Multi-stage production container |
| `.dockerignore` | New | Excludes data/tests/docs from build context |
| `requirements.txt` | Modified | Added fastapi, uvicorn, pydantic, python-multipart |
| `.github/workflows/ci.yml` | Modified | Added Stage 4 docker-build job |
