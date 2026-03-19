# MTC Extraction Benchmark — Complete Project History

> **Purpose:** A comprehensive, chronological record of how this repository was conceived, built, and evolved — covering the *what*, *why*, and *how* of every major decision and milestone.

---

## Table of Contents

1. [Project Origin & Goals](#1-project-origin--goals)
2. [Week 1 — Problem Definition & Schema Design](#2-week-1--problem-definition--schema-design)
3. [Week 2 — Baseline OCR Evaluation](#3-week-2--baseline-ocr-evaluation)
4. [Week 3 — Rule-Based Schema Extraction from OCR](#4-week-3--rule-based-schema-extraction-from-ocr)
5. [Week 4 — LLM vs Rule-Based: First Comparative Study (10 Feb 2026)](#5-week-4--llm-vs-rule-based-first-comparative-study-10-feb-2026)
6. [Phases 1–4 — Evaluation Infrastructure, Vision Extraction, Hybrid Pipeline, FastAPI Service](#6-phases-14--evaluation-infrastructure-vision-extraction-hybrid-pipeline-fastapi-service)
   - 6.1 Phase 1 — Evaluation Infrastructure
   - 6.2 Phase 2 — Vision Extraction
   - 6.3 Phase 3 — Hybrid Pipeline
   - 6.4 Phase 4 — FastAPI REST Service
7. [27 Feb 2026 — Pipeline Redesign, Full Benchmark Study & CI/CD Foundation](#7-27-feb-2026--pipeline-redesign-full-benchmark-study--cicd-foundation)
   - 7.1 Baseline Audit & Eight Failure Modes
   - 7.2 Eight-Point Redesign
   - 7.3 Benchmark Results
   - 7.4 Data Quality Investigation
   - 7.5 CI/CD Pipeline Established
8. [Model Expansion & CI Automation (commits d0cbeaa → 4cca2fb)](#8-model-expansion--ci-automation-commits-d0cbeaa--4cca2fb)
   - 8.1 Multi-Provider Model Registry
   - 8.2 HuggingFace / Qwen2.5 Integration
   - 8.3 Env-Var Gating & Skip-Existing Cache
   - 8.4 Quarto Benchmark Report
   - 8.5 Pre-Commit Hook
   - 8.6 CI Three-Layer Cache Overhaul
9. [17 March 2026 — HuggingFace Integration, CI Fixes & Finalisation](#9-17-march-2026--huggingface-integration-ci-fixes--finalisation)
10. [Infrastructure & Developer Experience](#10-infrastructure--developer-experience)
    - 10.1 DevContainer
    - 10.2 GitHub Codespaces Guide
    - 10.3 GitHub Models Integration Guide
11. [Architecture Overview — Current State](#11-architecture-overview--current-state)

---

## 1. Project Origin & Goals

### What

This project was initiated as a **6-week research and engineering effort** to explore automated extraction of structured data from **Mill Test Certificates (MTCs)** — quality assurance documents issued by steel manufacturers that record traceability information, chemical composition, and mechanical test results for each production batch.

MTCs are delivered as scanned PDFs with highly varied layouts. Manual data entry from these documents is time-consuming and error-prone. The objective was to build and benchmark an end-to-end pipeline that converts a raw PDF into a validated, machine-readable JSON structure.

### Why

Steel procurement, quality control, and traceability workflows all depend on MTC data. Automating extraction at high accuracy has direct business value for companies handling large volumes of these documents. Beyond the immediate use case, the project was designed as a **reusable benchmark template** for any semi-structured industrial document type.

### Planned Scope (from `project-plan.md`)

The 6-week plan covered:

| Week | Focus |
|------|-------|
| 1 | Problem definition, dataset review, schema design |
| 2 | Baseline OCR pipeline (Tesseract / PaddleOCR) |
| 3 | Structured extraction from OCR; initial ground truth |
| 4 | LLM-based & hybrid approaches |
| 5 | Evaluation & benchmarking |
| 6 | Documentation, generalisation, final report |

---

## 2. Week 1 — Problem Definition & Schema Design

### What

The first milestone was understanding the variability of **EN 10204 (3.1) Mill Test Certificates** and freezing a **Version 1 JSON extraction schema** before any automation was attempted. No OCR, ML, or extraction code was written in this phase.

### Why

Freezing the schema early creates a stable contract for all downstream work. Without it, every extraction method would produce different field names and data types, making comparison impossible. The schema defines:
- What information must be extracted
- Which fields are mandatory (business-critical and legally significant)
- Which fields are optional or nullable (layout-dependent)

Making optional fields explicitly nullable avoids forcing artificial values and enables cleaner evaluation logic.

### How — Schema Design Principles

Three principles governed the schema design:

1. **Structural consistency** — all outputs must follow an identical JSON structure regardless of which extraction method produced them.
2. **Minimal required fields** — only fields that are business-critical, legally significant, and consistently present were marked required. Unnecessary required fields cause false failures when non-essential data is absent.
3. **Explicit optionality** — fields that vary by mill, product type, or certificate format are included as optional with `null` allowed.

### Schema: Required Fields

| Field | Rationale |
|-------|-----------|
| `document.certificate_number` | Uniquely identifies the document |
| `document.issuing_date` | Legal validity anchor |
| `document.standard` | Specifies which norm governs the certificate |
| `traceability.heat_number` | Uniquely identifies the material batch |

If any required field is missing, the extraction output is considered invalid.

### Schema: Top-Level Structure

```json
{
  "document": {},
  "traceability": {},
  "product": {},
  "chemical_composition": [],
  "mechanical_properties": [],
  "approval": {}
}
```

The finalised schema was committed to `schema/mtc_extraction_schema_v1.json` and has not changed since — every extraction method and evaluation in this project targets it.

---

## 3. Week 2 — Baseline OCR Evaluation

### What

Two OCR engines were evaluated on sample MTC scanned PDFs: **Tesseract OCR** and **PaddleOCR**. The baseline OCR pipeline converted PDF pages to 300 DPI images, applied preprocessing (grayscale, median blur, Otsu thresholding), ran OCR, and exported plain text and word-level bounding boxes with confidence scores.

### Why

OCR is the unavoidable first step for scanned documents. Understanding the quality ceiling of available OCR tools sets realistic expectations for downstream extraction. If OCR accuracy is poor, no amount of sophisticated extraction logic can recover the correct values.

### Findings

#### Tesseract OCR

- Basic text extraction worked but with lower accuracy on complex table layouts
- Poor handling of dense tabular data with merged columns
- Inconsistent recognition of numeric values in tables
- Lost semantic relationships between headers and values
- Struggled with stamps, seals, and overlaid text

**Verdict:** Suitable as a diagnostic baseline only.

#### PaddleOCR

- Successfully extracted **226+ text blocks per page** with high confidence
- Character recognition accuracy: **95–99%** for most fields
- Excellent numeric precision (e.g., `0.9960`, `804.00`, `590`)
- Captured: certificate metadata, chemical composition values, mechanical properties, bounding box coordinates

**Technical bugs fixed during implementation:**
- `predict()` returns a generator; must be consumed with `list()`
- Dictionary keys must be plural: `rec_texts`, `rec_scores`, `rec_polys` (not singular forms)

**Example extraction (Page 4):**
```
CERTIFICATE NUMBER: 25-3133/01MNF/EXP (confidence: 0.9799)
ISSUING DATE: 07.07.2025 (confidence: 0.9846)
Yield Point (Re): 590 N/mm2 (confidence: 0.9998)
```

### Key Challenge Identified

PaddleOCR extracts text with bounding boxes but **does not understand document structure**. Chemical composition and mechanical property tables are extracted as individual disconnected text blocks. Reconstructing table structure from spatial coordinates requires layout-aware processing — the problem that Week 3 addressed.

### Outputs Committed

- Page-level extracted text files (`data/processed/paddle_ocr/`)
- Bounding-box JSON files
- Processed page images

---

## 4. Week 3 — Rule-Based Schema Extraction from OCR

### What

A full rule-based extraction system was built in `src/extraction/docling_extraction.py` (class `MTCPaddleExtractor`) that takes PaddleOCR JSON outputs and produces schema-compliant MTC JSON. The system used regex pattern matching and positional parsing to reconstruct table structure from the raw OCR text.

**Achievement: ~95% of all critical MTC data extracted**, including:
- 6 unique heat numbers
- 78 chemical element values (13 elements × 6 heats)
- 58 mechanical test samples
- Complete document metadata

### Why Rule-Based?

- **Deterministic** — predictable results for similar documents
- **Fast** — no ML model training required
- **Transparent** — easy to debug and understand
- **Sufficient** — works well for standardised MTC formats

### How — Extraction Pipeline

```
PaddleOCR JSON (Multi-page)
    ↓
Text Combination (merge all pages into single string)
    ↓
Pattern-Based Extraction
    │  ├── Document info (regex)
    │  ├── Traceability (regex)
    │  ├── Product info (regex)
    │  ├── Chemical composition (positional)
    │  ├── Mechanical properties (positional)
    │  └── Approval (keyword search)
    ↓
Validation & Deduplication
    │  ├── Range checks on numeric values
    │  ├── Remove duplicate heat numbers
    │  └── Remove duplicate test samples
    ↓
Structured JSON Output
```

### Key Extraction Logic

**Document fields** — regex patterns matched against the combined text:
```python
r"CERTIFICATE\s+NUMBER[:\s]+([A-Z0-9/-]+)"
r"ISSUING\s+DATE[:\s]+(\d{2}\.\d{2}\s+\d{4})"
r"EN\s+10204\s+[\d\.]+"
```

**Chemical composition** — window-based positional algorithm:
1. Find all heat numbers (8-digit pattern `\b(259900\d{2})\b`)
2. For each unique heat number, extract the next 20 lines
3. Find all decimal values matching `0[.,]\d{1,4}`
4. Map first 13 values to elements: C, Si, P, S, Mn, Ni, Cr, Mo, Cu, V, N, B, Ce

### Limitation Surfaced

The rule-based approach could only find **6 heat numbers** from the 4-page test certificate. The pattern relied on a specific heat number format, missing 14 other heats with different numeric patterns. This limitation directly motivated the LLM comparison in Week 4.

---

## 5. Week 4 — LLM vs Rule-Based: First Comparative Study (10 Feb 2026)

### What

A formal comparison was made between the rule-based PaddleOCR extractor and a **Meta-Llama-3.1-405B-Instruct** (via GitHub Models API) LLM extractor on a single 4-page Diler Demir Celik certificate (`25-3133/01MNF/EXP`).

### Why

The rule-based Week 3 results exposed a critical gap: only 6 of 20 heat numbers were captured. The question was whether an LLM, given the same OCR text, could do better without requiring hand-crafted patterns.

**Key optimisation discovered during implementation:** The initial LLM run was fed only page 1 text and failed. The fix was simple but crucial:
```bash
cat page1_text.txt page2_text.txt page3_text.txt page4_text.txt > all_pages.txt
```
Combining all pages before calling the LLM was the single change that enabled complete extraction.

### Results

| Metric | Rule-Based | LLM | Improvement |
|--------|------------|-----|-------------|
| Heat numbers captured | 6 | 20 | +233% |
| Test samples extracted | 58 | 80 | +38% |
| Approval/certification data | ❌ missed | ✅ found | — |
| Full lot identifier | Truncated (`1`) | Preserved (`2025-3133 LOT-1`) | — |
| Full customer address | Company name only | Full address including port | — |

**LLM schema structure was also superior** — nested `elements` dict within `chemical_composition` entries vs a flat key-value approach in the rule-based output, better matching the v1 schema design intent.

**Known LLM weaknesses:**
- ~5% minor data incompleteness on some heats (OCR quality dependent)
- Rule-based had no hallucinations but had duplicate entries requiring deduplication

### Recommendation from this study

> Use LLM as primary extraction method with a validation layer.

This recommendation became the architectural foundation for all subsequent work.

---

## 6. Phases 1–4 — Evaluation Infrastructure, Vision Extraction, Hybrid Pipeline, FastAPI Service

After the Week 4 LLM comparison, four structured engineering phases addressed the gaps between a research prototype and a production-ready system.

---

### 6.1 Phase 1 — Evaluation Infrastructure

#### What

The old benchmark counted raw numbers (rows extracted) but could not answer: *"Is row 14 in gpt-4o's output the same row as row 14 in the real document?"*

**Built:** `src/evaluation/evaluator.py` — the `MTCEvaluator` class with a standalone `evaluate()` method:

```python
MTCEvaluator.evaluate(extracted, ground_truth, numeric_tolerance=0.001,
                      mech_weight_tol=0.01, mech_yield_tol=5.0)
```

**Returns four scored sections + overall F1:**

| Section | What it measures |
|---------|-----------------|
| `document` | Exact string match on 5 certificate-level fields |
| `chemical` | Heat-level Precision/Recall/F1 by `heat_number`; element-level accuracy |
| `mechanical` | Row-level P/R/F1 using **fuzzy key matching**; property-level accuracy |
| `approval` | Exact match on approval fields |

#### Why

The root cause of the old metric's failure was that it matched mechanical rows by an exact composite key `(heat_number, test_sample)`. In practice, `test_sample` is frequently `null`, which caused identical physical rows to be treated as different — inflating false-positive and false-negative counts.

#### How — Fuzzy Mechanical Key Matching

The new approach finds the best ground-truth match for every extracted row by minimising physical distance:
- `|weight_kg_per_m difference| ≤ 0.01`
- `|yield_point_mpa difference| ≤ 5.0 MPa`

This correctly handles null `test_sample` values, absorbs minor OCR transcription rounding, and avoids false mismatches.

**CLI entry point:**
```bash
python -m src.evaluation.evaluator \
  --prediction data/processed/benchmark_output/gpt-4o_extracted.json \
  --ground-truth data/ground_truth/diler-07-07-2025-rerun-41-44_gt.json \
  --output data/comparison/gpt-4o_vs_gt.json
```

**Ground truth seeded:** `data/ground_truth/diler-07-07-2025-rerun-41-44_gt.json` was created from the gpt-4o extraction as a starting template, with a README annotation guide explaining field priority order.

---

### 6.2 Phase 2 — Vision Extraction

#### What

**Built:** `src/extraction/vision_extraction.py` — `VisionExtractor` class that bypasses OCR entirely and sends raw page images to a multimodal LLM:

```
PDF → pdf2image (JPEG per page) → base64 encode → OpenAI vision message → JSON + schema validation
```

#### Why

OCR-based extraction fails on pages with rubber stamps, rotated annotations, or merged table cells. When PaddleOCR produces low-confidence output, the LLM is working from corrupted text and cannot recover the correct values regardless of model quality. Sending the raw image directly to gpt-4o sidesteps this limitation, since gpt-4o natively understands document layout and table structure.

#### How — Key Design Decisions

| Decision | Reason |
|----------|--------|
| Downscale to 2048px max (long edge) | Reduces image token cost 60–70% with negligible quality loss for typed text |
| JPEG quality = 92 | Balance between file size and legibility of small text |
| 3× retry with schema validation | Same robustness pattern as the text pipeline |
| Per-page extraction then merge | Avoids exceeding context window on multi-page certificates |
| Composite-key deduplication on merge | Prevents duplicate heats/rows when the same heat spans two pages |

The extractor uses the same `models.inference.ai.azure.com` endpoint with `gpt-4o`, so no additional credentials beyond `GITHUB_TOKEN` are needed.

---

### 6.3 Phase 3 — Hybrid Pipeline

#### What

**Built:** `src/extraction/hybrid_pipeline.py` — `HybridPipeline` class with automatic three-tier routing based on OCR confidence:

```
OCR confidence ≥ 0.85  →  Rule-based extraction
                            └─ if completeness < 60%  →  Text LLM
                                                          └─ if completeness < 60%  →  Vision LLM

OCR confidence 0.65–0.85  →  Text LLM
                               └─ if completeness < 60%  →  Vision LLM

OCR confidence < 0.65  →  Vision LLM directly
```

#### Why

Having two extraction paths (text and vision) creates a new problem: which one should run? Running vision unconditionally is expensive. Running text unconditionally fails on low-quality scans. The hybrid pipeline solves this by routing automatically based on measurable signal (OCR confidence) rather than a manual per-document choice.

#### How — Thresholds & Completeness Check

| Threshold | Default | Meaning |
|-----------|---------|---------|
| `high_confidence_threshold` | 0.85 | Above this, try rule-based first |
| `medium_confidence_threshold` | 0.65 | Below this, skip text-LLM entirely |
| `min_completeness` | 0.60 | Fraction of expected fields that must be non-null |

The **completeness check** counts six high-signal fields — `certificate_number`, `issuing_date`, `standard`, `consignment_number`, `product.size`, `product.quality` — plus the presence of any chemical or mechanical rows. Scoring below 60% triggers escalation to the next tier.

**Full transparency via `_pipeline_meta`:**
```json
"_pipeline_meta": {
  "strategy_used": "text_llm",
  "ocr_confidence": 0.783,
  "elapsed_seconds": 18.4,
  "tokens_used": 12840
}
```

---

### 6.4 Phase 4 — FastAPI REST Service

#### What

**Built:** `src/api/` — a FastAPI application exposing the hybrid pipeline as an HTTP service.

#### Why

Before Phase 4, the only interface was the CLI. This meant:
- No programmatic access from other services or frontends
- No standard handling of authentication, input validation, or structured error responses
- Manual file management to pass inputs and retrieve outputs
- Impossible to deploy as a standalone service

#### Structure

```
src/api/
├── __init__.py
├── main.py       — FastAPI app, CORS, mounts routers
├── models.py     — Pydantic request/response models
└── routes/       — Endpoint implementations
```

---

## 7. 27 Feb 2026 — Pipeline Redesign, Full Benchmark Study & CI/CD Foundation

### What

The original `llm_models_extraction.py` (~598 lines) was fully audited, eight failure modes were identified, and the script was **completely rewritten** (growing to ~1,468 lines) with targeted fixes for every failure mode. A full 10-model benchmark was then run. Simultaneously, the first formal CI/CD pipeline was established in `.github/workflows/ci.yml`.

---

### 7.1 Baseline Audit — Eight Failure Modes

| # | Failure Mode | Root Cause | Impact |
|---|-------------|------------|--------|
| 1 | No OCR quality gating | PaddleOCR text accepted verbatim regardless of confidence | Low-quality pages silently corrupt extraction |
| 2 | No response validation | LLM output parsed as-is | Malformed JSON or schema violations accepted |
| 3 | No retry on failure | Single call per model per page | Transient hallucinations cause permanent data loss |
| 4 | Naive page merge | List concatenation without deduplication | Rows duplicated across pages |
| 5 | Single extraction pass | Pages processed independently | Fields spanning page boundaries never reconciled |
| 6 | No self-consistency sampling | One sample drawn per model | High variance on ambiguous table layouts |
| 7 | No cross-model ensemble | Models run in isolation | No mechanism to resolve model disagreements |
| 8 | Opaque cost tracking | No per-model token or time logging | Impossible to assess cost vs accuracy trade-offs |

---

### 7.2 Eight-Point Redesign

#### 4.1 OCR Confidence Parsing
`parse_ocr_with_confidence(text)` parses structured PaddleOCR output, extracts per-line confidence scores, and filters out lines below a configurable threshold before passing text to the prompt.

#### 4.2 JSON Schema Validation with Automatic Retry
`extract_with_validation()` validates each LLM response against `schema/mtc_extraction_schema_v1.json` using `jsonschema`. On validation failure it appends the error to the conversation and retries up to **3 times**. New dependency `jsonschema` added to `requirements.txt`.

#### 4.3 Self-Consistency Voting
`extract_with_consensus()` calls the same model *n* times with temperature > 0 and applies `_majority_vote()` to resolve each field independently by frequency. Controlled by `--consistency-samples N` (default: 1 = disabled).

#### 4.4 Smart Deduplication Merge
`merge_extractions_v2()` replaces naive concatenation with composite-key deduplication:
- **Mechanical properties:** keyed on `(heat_number, test_sample, weight_kg_per_m, yield_point_mpa)`
- **Chemical composition:** keyed on `heat_number` (one chemical record per melt)
- **Scalar fields:** last non-null value seen wins

#### 4.5 Two-Pass Extraction
`extract_two_pass()` runs each page independently first, then feeds all page texts and first-pass results into a single consolidation prompt. Enabled via `--two-pass`.

#### 4.6 Cross-Model Ensemble Extraction
`ensemble_extract()` runs the top-K models (configurable via `--ensemble-top-k`, default 3) and applies `_majority_vote()` across their outputs. Enabled via `--ensemble`.

#### 4.7 Rich Field-Level Evaluation Metrics
`compute_field_f1()` computes per-field **Precision, Recall, and F1**. For numeric fields, exact equality is replaced with tolerance-based matching. Configurable via `--numeric-tolerance F` (default: 0.01).

#### 4.8 Token and Latency Tracking
`extract_with_model()` now records `prompt_tokens`, `completion_tokens`, and `elapsed_seconds` per model call. All surfaced in `benchmark_summary.json`.

#### New CLI Surface

| Flag | Default | Description |
|------|---------|-------------|
| `--two-pass` | off | Enable two-pass extraction |
| `--consistency-samples N` | 1 | Self-consistency samples per model per page |
| `--ensemble` | off | Cross-model ensemble voting |
| `--ensemble-top-k N` | 3 | Number of top models for ensemble |
| `--numeric-tolerance F` | 0.01 | Tolerance for numeric field comparison |
| `--use-cached-ocr` | off | Use cached OCR text files, skip re-running PaddleOCR |
| `--models M [M ...]` | all | Restrict run to specific model names |

---

### 7.3 Benchmark Results

The rewritten script was run against all 10 originally targeted models on the 4-page Diler certificate using cached PaddleOCR text.

| Model | Status | Mech. Entries | Chem. Heats | Time (s) |
|-------|--------|:---:|:---:|:---:|
| gpt-4o | ✅ Success | 70 | 20 | 50.9 |
| Meta-Llama-3.1-405B-Instruct | ✅ Success | 77 | 20 | ~120 |
| Meta-Llama-3.1-8B-Instruct | ✅ Success | 74 | 20 | ~90 |
| gpt-4o-mini | ✅ Success | 79 | 20 | ~65 |
| Meta-Llama-3.1-70B-Instruct | ❌ Inference error | — | — | — |
| Meta-Llama-3-70B-Instruct | ❌ Inference error | — | — | — |
| Meta-Llama-3-8B-Instruct | ❌ Inference error | — | — | — |
| Mistral-large-2407 | ❌ Inference error | — | — | — |
| Mistral-Nemo | ❌ Inference error | — | — | — |
| AI21-Jamba-Instruct | ❌ Inference error | — | — | — |

Six of ten models returned `unknown_model` errors despite appearing in the GitHub Models catalogue. `RANKED_MODELS` was pruned from 10 to the 4 confirmed working models:
```python
RANKED_MODELS = ["gpt-4o", "Meta-Llama-3.1-405B-Instruct",
                 "Meta-Llama-3.1-8B-Instruct", "gpt-4o-mini"]
```

---

### 7.4 Data Quality Investigation

#### Question
Why did gpt-4o extract 70 mechanical property rows while gpt-4o-mini extracted 79?

#### Finding 1 — No Internal Duplicates
Both models produced zero duplicate composite keys — the discrepancy reflects genuine extraction differences, not an artefact of deduplication.

#### Finding 2 — The 9-Row Gap Isolated to 3 Heats

| Heat | gpt-4o | gpt-4o-mini | Difference |
|------|:---:|:---:|:---:|
| 2504095 | 1 | 3 | +2 |
| 25990024 | 1 | 4 | +3 |
| 25990031 | 1 | 5 | +4 |
| All other heats (17) | matched | matched | 0 |

gpt-4o **collapsed multiple test samples into a single row** for these three heats.

#### Finding 3 — Deeper Value-Level Disagreement

Even for heats where row counts matched, composite keys aligned on only ~39 of 70 entries due to:
- **Missing `test_sample` values in gpt-4o** — gpt-4o systematically failed to parse the sample number column, assigning `null` where gpt-4o-mini extracted a numeric value.
- **Minor numeric precision differences** — small rounding in `weight_kg_per_m` (e.g., 6.194 vs 6.195) caused physically identical rows to generate different composite keys.

#### Recommendations from Investigation

| Priority | Action |
|----------|--------|
| **High** | Annotate at least one certificate as human ground truth to enable F1-score evaluation |
| **High** | Implement fuzzy numeric matching in `merge_extractions_v2` (weight ±0.01 kg/m, yield ±5 MPa) |
| **Medium** | Reduce composite key from 4 fields to 3, dropping `test_sample` (frequently null) |
| **Medium** | Add post-extraction completeness check flagging heats with fewer rows than the mode |
| **Low** | Re-run 3 under-extracted heats with `--consistency-samples 3` |

---

### 7.5 CI/CD Pipeline Established

**File:** `.github/workflows/ci.yml`

Three sequential quality gates were established:

| Stage | Job | Runs on | Purpose |
|-------|-----|---------|---------|
| 1 | `lint` | All triggers | Enforce code style via `ruff check` + `ruff format --check` |
| 2 | `test` | All triggers | Run unit tests with `pytest`; collect coverage XML |
| 3 | `benchmark` | Push to `main` only | Run LLM benchmark against live models; upload results |

**Trigger conditions:**

| Event | Stages run |
|-------|-----------|
| Push to `main` | Lint → Test → Benchmark |
| Pull request to `main` | Lint → Test only (benchmark skipped) |
| Manual trigger | Lint → Test only |

**Permissions (minimum required):**
```yaml
permissions:
  contents: read   # checkout files
  models: read     # call GitHub Models inference endpoint
```

**Test suite (`tests/test_smoke.py`):**
Intentionally lightweight smoke tests — no API calls, no extraction. Validates:
- Core modules import without error; `LLMModelBenchmark` class exists
- Every entry in `RANKED_MODELS` has `id`, `label`, `provider`, `tier` keys
- First model in `RANKED_MODELS` is `gpt-4o`

**Benchmark job:**
- Uses `--use-cached-ocr` to decouple OCR quality from LLM evaluation and ensure determinism
- Restricts to 4 confirmed working models
- Uploads per-model JSON + `benchmark_summary.json` as `benchmark-results-{git SHA}` artifact (90-day retention)
- `GITHUB_TOKEN` auto-injected by GitHub Actions

**Local replication of CI checks:**
```bash
# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Tests
pytest tests/ -v --cov=src --cov-report=term-missing

# Benchmark (requires GITHUB_TOKEN)
python src/extraction/llm_models_extraction.py \
    --use-cached-ocr --models gpt-4o gpt-4o-mini \
    --output data/processed/benchmark_output
```

---

## 8. Model Expansion & CI Automation (commits d0cbeaa → 4cca2fb)

### Overview

The benchmark runner originally supported only four models, all served through GitHub Models. Every model was hardcoded in the CI job, and adding a new provider required changes in at least five places. This update transformed the runner into a **self-maintaining, multi-provider benchmark**.

---

### 8.1 Multi-Provider Model Registry

**What changed:** Each model dict was extended with two new keys:
- `base_url` — OpenAI-compatible API endpoint for this provider
- `api_key_env` — Name of the environment variable that holds the API key

```python
_GH_BASE = "https://models.inference.ai.azure.com"

RANKED_MODELS: List[Dict[str, str]] = [
    {
        "id": "gpt-4o",
        "label": "GPT-4o",
        "provider": "OpenAI",
        "tier": "top",
        "base_url": _GH_BASE,
        "api_key_env": "GITHUB_TOKEN",
    },
    # … three more GitHub Models entries
]
```

A fast lookup dict is derived automatically:
```python
ALL_MODELS_REGISTRY: Dict[str, dict] = {m["id"]: m for m in ALL_MODELS}
```

**Why:** Previously `base_url` and token names were scattered hardcoded strings. Centralising them makes every downstream consumer (client factory, env-var gating, CI) pull from a single source of truth.

**Per-provider client factory `_make_client()`:**

Replaced the single shared `OpenAI` client with a factory that creates and caches clients per `(base_url, api_key_env)` pair. Client creation is deferred until the model is actually called, so unused providers never raise `EnvironmentError`.

---

### 8.2 HuggingFace / Qwen2.5 Integration

**What changed:**

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

HuggingFace's Inference API exposes an OpenAI-compatible `/v1/chat/completions` endpoint — the same `openai` Python package works with no additional dependencies.

**Why:** Open-source models on HuggingFace provide a cost-efficient, non-proprietary comparison point. Qwen2.5-72B is competitive with GPT-4o on many structured-extraction tasks.

---

### 8.3 Env-Var Gating & Skip-Existing Cache

**Env-var gating (`main()`):**

Every model is checked against its required env var before running. Models whose secret is absent are silently skipped with a clear message:
```python
available_ids: List[str] = []
for mid in model_ids:
    key_env = ALL_MODELS_REGISTRY.get(mid, {}).get("api_key_env", "GITHUB_TOKEN")
    if os.environ.get(key_env):
        available_ids.append(mid)
    else:
        print(f"   ⏭️  Auto-skipping {mid!r} — {key_env} not set")
```

**Why:** Without this, adding HF models to `HF_MODELS` would crash CI whenever `HF_TOKEN` was absent. Gating allows gradual secret rollout.

**Skip-existing cache (`--skip-existing`):**

When passed, the benchmark loads per-model JSON from disk instead of calling the API:
```python
if skip_existing and output_dir:
    cached_file = output_dir / f"{safe_name}_extracted.json"
    if cached_file.exists():
        # load from disk, still re-run MTCEvaluator.evaluate() for updated scores
```

**Why:** Avoids re-spending API tokens when the document and prompt haven't changed. Still re-evaluates against potentially updated ground truth.

**New CLI flags added:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--providers github huggingface` | None (all) | Restrict which providers to include |
| `--rate-limit-sleep N` | 0.0 | Sleep N seconds between API calls (HuggingFace rate limits) |
| `--skip-existing` | False | Load cached JSON instead of calling API |

---

### 8.4 Quarto Benchmark Report

**File:** `benchmark_report.qmd` → renders to `benchmark_report.html`

A self-contained Quarto report generates visual comparisons across all benchmarked models and is uploaded as a CI artifact after every push to `main`.

**Report sections:**

| Section | Chart type | What it shows |
|---------|-----------|---------------|
| Model Registry | Table | All models, providers, tiers, endpoints |
| Accuracy Overview | Grouped bar chart | Precision, recall, F1 across 4 extraction categories |
| Radar / Spider | Polar axes | Multi-axis performance fingerprint per model |
| Latency vs Quality | Scatter (bubble = tokens) | Time-quality trade-off |
| Token Usage & Cost | Stacked bar | Prompt vs completion tokens, estimated cost |
| Per-Field Heatmap | `imshow` RdYlGn | Field-by-field accuracy across models |
| Document Field Detail | Pivot table | Per-document per-field extraction result (✓/✗) |
| Error / Failure Breakdown | Table | Models that errored and failure reasons |

The report handles legacy benchmark runs without `field_metrics` by re-running `MTCEvaluator.evaluate()` on saved JSON files automatically.

---

### 8.5 Pre-Commit Hook

**File:** `.git/hooks/pre-commit` (local, not tracked)

```bash
#!/usr/bin/env bash
STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)
if [ -n "$STAGED" ]; then
    ruff format $STAGED
    ruff check --fix $STAGED
    git add $STAGED
fi
```

**Why:** CI lint jobs were failing on freshly committed code that wasn't formatted. The hook runs both tools in auto-fix mode before the commit is recorded — eliminating the entire class of formatting-related CI failure.

---

### 8.6 CI Three-Layer Cache Overhaul

The `benchmark` job was rewritten to use a three-layer caching strategy:

```yaml
# Layer 1: pip packages (keyed on requirements.txt)
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: benchmark-${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

# Layer 2: Python __pycache__ (keyed on src/**/*.py)
- uses: actions/cache@v4
  with:
    path: src/**/__pycache__
    key: pycache-${{ runner.os }}-py311-${{ hashFiles('src/**/*.py') }}

# Layer 3: Benchmark outputs (keyed on OCR text, prompts, schema)
- uses: actions/cache@v4
  with:
    path: data/processed/benchmark_output
    key: bench-out-${{ runner.os }}-${{ hashFiles('data/processed/pipeline_output/text/**', 'prompts/**', 'schema/**') }}
```

| Layer | Invalidated by | Effect when cache misses |
|-------|---------------|------------------------|
| pip | `requirements.txt` change | Re-downloads all packages |
| `__pycache__` | Any `src/**/*.py` change | Re-compiles bytecode |
| Benchmark outputs | OCR text, prompt, or schema change | Re-runs all API calls |

When the benchmark output cache hits, `--skip-existing` loads every model's JSON from disk — **zero API calls, full scores recalculated** against potentially updated ground truth.

**Benchmark job also now:**
- Runs `quarto render benchmark_report.qmd` to generate the HTML report
- Passes `HF_TOKEN` in addition to `GITHUB_TOKEN` (when secret is set)
- Timeout increased from 20 to **45 minutes** to accommodate added models
- Model list is **no longer hardcoded** in CI — `--models` flag dropped; `main()` auto-discovers any model whose API key is set

---

## 9. 17 March 2026 — HuggingFace Integration, CI Fixes & Finalisation

Nine commits on 17 March 2026 addressed a cascade of issues that had emerged from the HF integration and Quarto report generation.

### Commit Timeline

| Time UTC | Commit | Area | Type |
|----------|--------|------|------|
| 08:30 | `b54ca9f` | Code Review Agent | fix |
| 08:39 | `9c91228` | CI Pipeline | fix |
| 08:45 | `fd01d5f` | Benchmark Script | fix |
| 08:50 | `c3b155c` | Quarto Report | fix |
| 09:00 | `0543b35` | Quarto Report | fix |
| 09:09 | `6a5dd8b` | CI Pipeline | ci |
| 14:07 | `1dc5011` | Benchmark Script | fix |
| 14:23 | `7cafbc6` | Benchmark Script | feat |
| 14:35 | `1c4fefc` | Benchmark Script + Results | feat |

---

### Fix: Code Review Agent — Token Limit Overrun (`b54ca9f`)

**Problem:** The weekly AI code review agent was failing with HTTP `413 tokens_limit_reached`. The agent was dumping the entire repository (~141K tokens) as context, far exceeding the GitHub Models free-tier budget (~8K tokens for GPT-4o).

**Fix:**
1. Replaced `collect_context()` (full-repo dump) with a two-step budget-aware function:
   - Key structural files (`README.md`, `project-plan.md`, `pyproject.toml`) truncated to 1,500 chars each
   - `collect_git_diff_context()` produces a focused diff of the last 7 days (3–15 commits), capped at 8,000 chars
2. `MAX_CONTEXT_CHARS = 14,000` hard cap on total context sent to any model
3. `MAX_OUTPUT_TOKENS` reduced from 16,384 → 2,000 (sufficient for a code review summary)
4. `CANDIDATE_MODELS` reordered: `gpt-4o` first (highest 8K free-tier budget), then `o3-mini`, `o4-mini`

The code review action was completely non-functional before this fix.

---

### Fix: CI — Missing Jupyter Dependencies for Quarto (`9c91228`)

**Problem:** `quarto render benchmark_report.qmd` was failing with `ModuleNotFoundError: No module named 'nbformat'`. Quarto requires `nbformat`, `nbclient`, and `ipykernel` to execute Python cells in `.qmd` files.

**Fix:** New step inserted before render:
```yaml
- name: Install Jupyter for Quarto
  run: pip install jupyter nbformat nbclient ipykernel
```

---

### Fix: Benchmark Script — Cached Results Shown as FAIL (`fd01d5f`)

**Problem:** When a model's output was loaded from a cached JSON file (`--skip-existing`), it was assigned `status='cached'`. The `_print_table()` method only treated `status='success'` as passing — so every cached model displayed `❌ FAIL` in the summary table even though extraction had succeeded.

**Fix:** `_print_table()` now treats `status == 'cached'` identically to `status == 'success'`. Cached runs display `⏭️ CACHE` instead of `✅ OK` for visual distinction, but no longer appear as failures.

---

### Fix: Quarto Report — `status='cached'` Excluded from Score Columns (`c3b155c`)

**Problem:** The `build-scores` cell filtered with `status != 'success'`, accidentally excluding all cached runs. The result was an empty DataFrame, causing a `KeyError` on `'doc_accuracy'`.

**Fix:**
```python
# Before
df = df[df['status'] == 'success']
# After
df = df[df['status'].isin(['success', 'cached'])]
```

---

### Fix: Quarto Report — Four Rendering Errors Resolved (`0543b35`)

| Error | Root cause | Fix |
|-------|-----------|-----|
| `AttributeError: 'Styler' has no attribute 'applymap'` | Pandas 2.1 renamed to `Styler.map` | Replaced all `applymap` → `map` |
| `IndexError` in latency scatter chart | Fragile `df_c.index.get_loc(model_name)` | Replaced with `enumerate()` for safe positional indexing |
| `KeyError` in registry table | Empty DataFrame when all models failed | Added `if df.empty:` guard |
| `ValueError` in heatmap | Empty DataFrame passed to `pd.pivot_table()` | Added guard before pivot call |

---

### CI: Split Benchmark Artifacts into Two Named Uploads (`6a5dd8b`)

**What changed:** Single `upload-artifact` step split into two:

| Artifact | Contents | Retention |
|----------|----------|-----------|
| `benchmark-results-<sha>` | All JSON in `benchmark_output/` | 90 days |
| `benchmark-report-<sha>` | `benchmark_report.html` | 90 days |

`continue-on-error: true` was **removed** from the `quarto render` step — render failures now cause visible job failures instead of being silently swallowed.

---

### Fix: HuggingFace Base URL Deprecated (`1dc5011`)

**Problem:** All HuggingFace model calls were failing because `https://api-inference.huggingface.co/v1` was deprecated.

**Fix (single-line change):**
```python
# Before
_HF_BASE = "https://api-inference.huggingface.co/v1"
# After
_HF_BASE = "https://router.huggingface.co/v1"
```

This restored connectivity to all HuggingFace models. The first HuggingFace extraction (`Qwen/Qwen2.5-7B-Instruct`) succeeded and its output was committed alongside this fix.

---

### Feat: Expand HuggingFace Model List, Drop 72B Due to Timeout (`7cafbc6`)

**Removed:** `Qwen/Qwen2.5-72B-Instruct` — caused consistent timeouts on HuggingFace free tier.

**Added:**

| Model ID | Label | Provider |
|----------|-------|---------|
| `microsoft/Phi-3.5-mini-instruct` | Phi-3.5 Mini | Microsoft |
| `google/gemma-2-9b-it` | Gemma 2 9B | Google |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | DeepSeek-R1 Distill 8B | DeepSeek |

**Per-model `max_tokens` override added:** `extract_with_model()` now reads `max_tokens` from the model registry entry rather than always using the global default — necessary because HuggingFace small models enforce stricter output token limits.

---

### Feat: Add HuggingFace Benchmark Results, Stabilise Model List (`1c4fefc`)

**Outcome of testing the new HF models:** Phi-3.5-mini, Gemma-2-9B, and DeepSeek-R1-Distill-8B either timed out or returned non-JSON responses consistently. HF model list consolidated to two working models:

| Model | Status |
|-------|--------|
| `Qwen/Qwen2.5-7B-Instruct` | ✅ Working |
| `meta-llama/Llama-3.1-8B-Instruct` | ✅ Working (new) |

**DeepSeek-R1 `<think>` tag stripping added:**

Reasoning models (DeepSeek-R1 and similar) wrap reasoning traces in `<think>…</think>` before the actual JSON answer. Without stripping, JSON parsing failed:
```python
if "<think>" in raw:
    think_end = raw.find("</think>")
    if think_end != -1:
        raw = raw[think_end + 8:].strip()
```

**Committed results:**
- `benchmark_summary.json` updated with all working models including new HF models
- `meta-llama_Llama-3_1-8B-Instruct_extracted.json` added as new extraction output
- `benchmark_report.html` regenerated

### Overall Impact on 17 March

| Area | Net Effect |
|------|-----------|
| CI pipeline | Reliable end-to-end CI with downloadable HTML report |
| Quarto report | Renders cleanly under all conditions (including all-model-fail) |
| HF integration | `Qwen2.5-7B` and `Llama-3.1-8B` via HuggingFace now working |
| Extraction script | More models produce usable JSON; table output is accurate; DeepSeek `<think>` handled |
| Code review agent | Weekly review functional again (context reduced from ~141K → ~14K tokens) |

---

## 10. Infrastructure & Developer Experience

### 10.1 DevContainer

The `.devcontainer/` directory provides a reproducible development environment for VS Code and GitHub Codespaces, consisting of a `Dockerfile` (system packages and Python dependencies) and `devcontainer.json` (VS Code settings, extensions, port forwarding).

**Key components installed in the container:**
- Python 3.11 (Ubuntu 22.04 base)
- PaddleOCR, pdf2image, Pillow, OpenCV
- poppler-utils (for PDF → image conversion)
- Tesseract OCR (baseline comparison)
- openai, jsonschema, fastapi, uvicorn, ruff, pytest, quarto

**Best practices applied:**
- RUN commands grouped to minimise image layers
- apt cache cleaned (`rm -rf /var/lib/apt/lists/*`)
- Commands ordered from least to most frequently changed for optimal Docker layer caching

**Container rebuild workflow:**
- Modify Dockerfile or `devcontainer.json` → "Dev Containers: Rebuild Container" in VS Code
- Full clean rebuild: "Dev Containers: Rebuild Container Without Cache"

### 10.2 GitHub Codespaces Guide

The project maintains full support for GitHub Codespaces cloud development. Key guidance:

- **`GITHUB_TOKEN`** is auto-available in Codespaces — no setup needed for GitHub Models API
- **`HF_TOKEN`** must be added as a Codespaces/Repository secret for HuggingFace models
- Recommended machine type: 4-core minimum for development, 8-core or higher for extraction runs
- Ports 8888 (Jupyter) and 5000 (FastAPI) are pre-forwarded

**Codespace lifecycle:**
- Auto-stops after 30 minutes of inactivity
- All files retained in stopped Codespaces
- CLI management: `gh codespace create/list/stop/delete`

### 10.3 GitHub Models Integration Guide

`docs/github-models-integration.md` documents how to use the GitHub Models endpoint (`https://models.inference.ai.azure.com`) with the OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)
```

Available models relevant to the project:
- `gpt-4o`, `gpt-4o-mini` (OpenAI — multimodal, strong structured extraction)
- `Meta-Llama-3.1-405B-Instruct`, `Meta-Llama-3.1-8B-Instruct` (Meta — open-weight)

Key extraction settings:
- `temperature=0.1` — low temperature for consistent, deterministic extraction
- `response_format={"type": "json_object"}` — enforces JSON output

---

## 11. Architecture Overview — Current State

### Repository Structure

```
mtc-extraction-benchmark/
├── schema/
│   └── mtc_extraction_schema_v1.json          # Frozen v1 extraction schema
├── prompts/
│   └── mtc_llm_extraction_prompt.txt          # LLM prompt template
├── src/
│   ├── extraction/
│   │   ├── llm_models_extraction.py           # Core benchmark runner (~1,500+ lines)
│   │   ├── docling_extraction.py              # Rule-based PaddleOCR extractor
│   │   ├── vision_extraction.py               # Vision (image) extractor
│   │   ├── hybrid_pipeline.py                 # Three-tier routing pipeline
│   │   ├── paddle_extraction.py               # PaddleOCR wrapper
│   │   └── complete_pipeline.py               # End-to-end pipeline
│   ├── evaluation/
│   │   └── evaluator.py                       # MTCEvaluator with fuzzy matching
│   └── api/
│       ├── main.py                            # FastAPI application
│       ├── models.py                          # Pydantic models
│       └── routes/                            # API endpoints
├── data/
│   ├── ground_truth/                          # Human-annotated reference data
│   ├── raw/                                   # Source PDF documents
│   └── processed/
│       ├── paddle_ocr/                        # PaddleOCR outputs (text + bbox JSON)
│       ├── benchmark_output/                  # Per-model extracted JSON + summary
│       └── pipeline_output/                   # Cached OCR text for CI
├── .github/
│   └── workflows/ci.yml                       # Three-stage CI: lint → test → benchmark
├── benchmark_report.qmd                       # Quarto report source
├── benchmark_report.html                      # Rendered benchmark report
└── tests/
    └── test_smoke.py                          # Lightweight structural smoke tests
```

### Active Model Registry

| Model | Provider | Tier | Endpoint |
|-------|----------|------|----------|
| gpt-4o | OpenAI | top | GitHub Models |
| Meta-Llama-3.1-405B-Instruct | Meta | top | GitHub Models |
| Meta-Llama-3.1-8B-Instruct | Meta | small | GitHub Models |
| gpt-4o-mini | OpenAI | small | GitHub Models |
| Qwen/Qwen2.5-7B-Instruct | Qwen/Alibaba | small | HuggingFace |
| meta-llama/Llama-3.1-8B-Instruct | Meta | small | HuggingFace |

### Extraction Strategy Tiers

```
OCR confidence ≥ 0.85  →  Rule-based  →  (fallback) Text LLM  →  (fallback) Vision LLM
OCR confidence 0.65–0.85  →  Text LLM  →  (fallback) Vision LLM
OCR confidence < 0.65  →  Vision LLM directly
```

### Evaluation Metrics

The `MTCEvaluator` produces structured F1 scores against human-annotated ground truth:
- **Document** — exact match on certificate-level fields
- **Chemical** — heat-level P/R/F1 + element-level accuracy within matched heats
- **Mechanical** — fuzzy-matched row-level P/R/F1 (weight ±0.01 kg/m, yield ±5 MPa)
- **Approval** — exact match on certification fields

### Adding a New Model

1. Append to the appropriate list in `llm_models_extraction.py`:
   ```python
   RANKED_MODELS.append({  # or HF_MODELS
       "id": "model-name",
       "label": "Display Name",
       "provider": "Provider",
       "tier": "top|small",
       "base_url": _GH_BASE,       # or _HF_BASE
       "api_key_env": "GITHUB_TOKEN",  # or "HF_TOKEN"
   })
   ```
2. Add the corresponding secret (`HF_TOKEN`) as a GitHub repository secret if using HuggingFace.
3. That's it — CI auto-discovers the model and runs it when the secret is present.

---

*Document compiled: 18 March 2026. Covers all 12 source files in `docs/` plus `README.md` and `project-plan.md`.*
