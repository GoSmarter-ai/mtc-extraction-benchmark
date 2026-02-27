# LLM-Driven MTC Extraction — Technical Report

> **Project:** GoSmarter-ai / mtc-extraction-benchmark
> **Date:** February 27, 2026
> **Document type:** Technical case study — design, implementation, benchmarking, and data quality analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Problem Statement](#2-background--problem-statement)
3. [Baseline System Audit](#3-baseline-system-audit)
4. [Redesign & Implementation](#4-redesign--implementation)
5. [Benchmark Study](#5-benchmark-study)
6. [Data Quality Investigation](#6-data-quality-investigation)
7. [Conclusions & Next Steps](#7-conclusions--next-steps)
8. [Appendix — Schema Reference](#8-appendix--schema-reference)
9. [Appendix — Changed Artefacts](#9-appendix--changed-artefacts)

---

## 1. Executive Summary

This report documents the end-to-end redesign of an automated extraction pipeline for **Mill Test Certificates (MTCs)** — industrial steel quality documents that carry chemical composition, mechanical test results, and traceability data.

The starting point was a ~600-line Python script that issued a single prompt to each of 10 language models and accepted whatever JSON they returned. After a systematic audit of its failure modes, the script was fully rewritten with 8 targeted improvements — adding schema validation, self-consistency sampling, smart deduplication, two-pass extraction, ensemble voting, and rich evaluation metrics. A full benchmark was then run across all 10 candidate models. Four succeeded; six failed at the inference layer due to unavailable endpoints. The working models were retained as the active set.

A post-benchmark data quality investigation revealed a meaningful disagreement between the top two models (gpt-4o extracted 70 mechanical test rows; gpt-4o-mini extracted 79). The 9-row gap was traced to three specific heat numbers where gpt-4o collapsed multiple test samples into a single row, and to systematic differences in numeric precision and test-sample column parsing between the two models.

---

## 2. Background & Problem Statement

### What is an MTC?

A Mill Test Certificate is a quality assurance document issued by a steel manufacturer for each production batch. It records:

- **Traceability** — order number, heat numbers, delivery references
- **Product specification** — dimensions, weight per metre, applicable standard
- **Chemical composition** — certified elemental analysis per heat (C, Si, Mn, P, S, Ni, Cr, Mo, Cu, V, N, B, Ce)
- **Mechanical properties** — per-heat-per-sample test results (yield point, tensile strength, elongation, etc.)
- **Approval** — inspector signatures and third-party certifications

MTCs are structured documents but delivered as scanned PDFs with varied layouts. Manual data entry from MTCs is time-consuming and error-prone. The goal of this project is to automate extraction into a validated JSON structure defined by `schema/mtc_extraction_schema_v1.json`.

### Technical Environment

| Component | Detail |
|-----------|--------|
| Runtime | Python 3.11, Ubuntu 22.04 (GitHub Codespaces) |
| OCR engine | PaddleOCR (via `ocr_extraction_paddleOCR.py`) |
| LLM provider | GitHub Models — `https://models.inference.ai.azure.com` |
| Authentication | `GITHUB_TOKEN` environment variable |
| LLM SDK | `openai` Python package (Azure-compatible endpoint) |
| Schema validation | `jsonschema` |
| Target schema | `schema/mtc_extraction_schema_v1.json` |

---

## 3. Baseline System Audit

### How the Original Script Worked

The original `llm_models_extraction.py` (~598 lines) followed a straightforward loop:

1. Read cached PaddleOCR text from `data/processed/text/page_N.txt`
2. Construct a single system + user message using the prompt template in `prompts/mtc_llm_extraction_prompt.txt`
3. Call each model in `RANKED_MODELS` once per page
4. Parse the raw string response as JSON (no validation)
5. Merge pages by simple list concatenation
6. If a ground truth file was provided, compute naive field-level accuracy

**Original RANKED_MODELS (10 models):**

```
gpt-4o
Meta-Llama-3.1-405B-Instruct
Meta-Llama-3.1-70B-Instruct
Meta-Llama-3.1-8B-Instruct
Meta-Llama-3-70B-Instruct
Meta-Llama-3-8B-Instruct
Mistral-large-2407
Mistral-Nemo
gpt-4o-mini
AI21-Jamba-Instruct
```

### Identified Failure Modes

A detailed audit identified eight categories of weakness:

| # | Failure Mode | Root Cause | Impact |
|---|-------------|------------|--------|
| 1 | No OCR quality gating | PaddleOCR text accepted verbatim regardless of confidence | Low-quality pages silently corrupt extraction |
| 2 | No response validation | LLM output parsed as-is with no structural check | Malformed JSON or schema violations accepted |
| 3 | No retry on failure | Single call per model per page | Transient hallucinations cause permanent data loss |
| 4 | Naive page merge | List concatenation without deduplication | Mechanical/chemical rows duplicated across pages |
| 5 | Single extraction pass | Each page processed independently | Fields that span page boundaries are never reconciled |
| 6 | No self-consistency sampling | One sample drawn per model | High variance on ambiguous table layouts |
| 7 | No cross-model ensemble | Models run in isolation | No mechanism to resolve model disagreements |
| 8 | Opaque cost tracking | No per-model token or time logging | Impossible to assess cost vs accuracy trade-offs |

---

## 4. Redesign & Implementation

The script was fully rewritten — growing from 598 lines to approximately 1,468 lines. All eight failure modes were directly addressed.

### 4.1 OCR Confidence Parsing

**Problem:** PaddleOCR attaches a confidence score to every detected text line, but the baseline ignored it entirely. Low-confidence OCR lines (e.g., smudged table cells, rotated text) were sent to the LLM unchanged, introducing noise.

**Solution:** `parse_ocr_with_confidence(text)` — a new method that parses the structured PaddleOCR output format, extracts per-line confidence scores, and filters out lines below a configurable threshold before the text is passed to the prompt.

---

### 4.2 JSON Schema Validation with Automatic Retry

**Problem:** LLMs occasionally return structurally invalid JSON, use wrong field names, or omit required sections. The baseline accepted all responses as-is.

**Solution:** `extract_with_validation(model, messages, ...)` wraps the core LLM call and:

1. Validates the parsed response against `schema/mtc_extraction_schema_v1.json` using `jsonschema`
2. On validation failure, appends the validation error to the conversation and retries
3. Retries up to **3 times** before falling back to the last received response

A new dependency `jsonschema` was added to `requirements.txt`.

---

### 4.3 Self-Consistency Voting

**Problem:** A single sample from a model on an ambiguous table cell can go either way — especially for OCR-degraded numeric fields like tensile strength or heat numbers.

**Solution:** `extract_with_consensus(model, messages, n, ...)` calls the same model *n* times with temperature > 0, collecting *n* independent JSON extractions. These are then passed to `_majority_vote(samples)`, which resolves each field independently by selecting the most frequently occurring value across the *n* samples.

Controlled via the new `--consistency-samples N` CLI flag (default: 1, i.e., disabled).

---

### 4.4 Smart Deduplication Merge

**Problem:** When a multi-page certificate has overlapping headers or repeated heat summary rows across pages, naive concatenation creates duplicate mechanical property rows and chemical composition entries.

**Solution:** `merge_extractions_v2(pages)` replaces the original concatenation logic with composite-key deduplication:

- **Mechanical properties:** keyed on `(heat_number, test_sample, weight_kg_per_m, yield_point_mpa)` — the four fields that together uniquely identify a test sample
- **Chemical composition:** keyed on `heat_number` — one chemical record per melt
- For scalar fields (`document`, `product`, `traceability`), the last non-null value seen wins

The original `merge_extractions(pages)` is retained as a backward-compatible alternative.

---

### 4.5 Two-Pass Extraction

**Problem:** Some MTC fields span multiple pages — for example, the approval block on the last page references heat numbers first seen on page 1. Single-pass per-page extraction cannot resolve these cross-page relationships.

**Solution:** `extract_two_pass(model, pages_text, ...)` implements a two-stage process:

1. **First pass** — each page processed independently (same as baseline)
2. **Second pass** — all page texts and all first-pass results concatenated into a single prompt that asks the model to consolidate, fill gaps, and resolve contradictions

Enabled via `--two-pass` CLI flag.

---

### 4.6 Cross-Model Ensemble Extraction

**Problem:** Different models make different errors on the same document. Running models independently provides no mechanism to agree on ambiguous fields.

**Solution:** `ensemble_extract(models, pages_text, ...)` runs the top-K models (configurable via `--ensemble-top-k`, default 3) and applies `_majority_vote` across their individual outputs — effectively an inter-model consensus layer on top of the per-model intra-model consensus from §4.3.

Enabled via `--ensemble` CLI flag.

---

### 4.7 Rich Field-Level Evaluation Metrics

**Problem:** The baseline evaluation computed a flat match/no-match ratio. This collapsed structural differences (missing fields vs wrong values) and gave no actionable signal.

**Solution:** `compute_field_f1(pred, gt, ...)` computes per-field **Precision**, **Recall**, and **F1** scores. For numeric fields, exact equality is replaced with tolerance-based matching (`abs(pred - gt) <= tolerance`). The tolerance is configurable via `--numeric-tolerance F` (default: 0.01). The legacy `compute_metrics(pred, gt)` wrapper is retained.

---

### 4.8 Token and Latency Tracking

**Problem:** No data existed on how many tokens each model consumed or how long each call took — making cost-vs-accuracy trade-off analysis impossible.

**Solution:** `extract_with_model(model, messages, ...)` now records:
- `prompt_tokens` — tokens in the input
- `completion_tokens` — tokens in the model's response
- `elapsed_seconds` — wall-clock time for the API call

These are surfaced in `benchmark_summary.json` and printed in the results table.

---

### 4.9 New CLI Surface

| Flag | Default | Description |
|------|---------|-------------|
| `--two-pass` | off | Enable two-pass extraction |
| `--consistency-samples N` | 1 | Self-consistency samples per model per page |
| `--ensemble` | off | Enable cross-model ensemble voting |
| `--ensemble-top-k N` | 3 | Number of top models for ensemble |
| `--numeric-tolerance F` | 0.01 | Tolerance for numeric field comparison |
| `--use-cached-ocr` | off | Skip re-running OCR; use cached text files |
| `--models M [M ...]` | all | Restrict run to specific model names |

---

## 5. Benchmark Study

### Setup

The rewritten script was run against a single 4-page MTC (certificate `diler-07-07-2025-rerun-41-44`) with cached PaddleOCR text. All 10 original models were targeted.

```bash
python src/extraction/llm_models_extraction.py \
    --use-cached-ocr \
    --models gpt-4o Meta-Llama-3.1-405B-Instruct Meta-Llama-3.1-70B-Instruct \
             Meta-Llama-3.1-8B-Instruct Meta-Llama-3-70B-Instruct Meta-Llama-3-8B-Instruct \
             Mistral-large-2407 Mistral-Nemo gpt-4o-mini AI21-Jamba-Instruct
```

Output written to `data/processed/benchmark_output/`.

### Results

| Model | Status | Mech. Entries | Chem. Heats | Time (s) |
|-------|--------|:-------------:|:-----------:|:--------:|
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

### Observations

**Model availability:** Six of the ten candidate models returned `unknown_model` errors at inference time via `models.inference.ai.azure.com`, despite appearing in the GitHub Models catalogue API. These models were subsequently removed from `RANKED_MODELS`, reducing it from 10 to 4 active entries.

**Extraction volume:** All four successful models extracted data from all 20 heat numbers on the chemical composition side, indicating reliable coverage of the chemistry tables. Mechanical property row counts ranged from 70 (gpt-4o) to 79 (gpt-4o-mini) — a spread that warranted deeper investigation (see §6).

**Speed vs capability:** gpt-4o was the fastest at 50.9 seconds. Despite being a smaller model, gpt-4o-mini was second at ~65 seconds and extracted the highest mechanical row count. Llama 3.1 405B was the slowest at ~120 seconds and placed second on mechanical row count.

**Ground truth:** No annotated ground truth file was available during this run. Evaluation was limited to structural completeness (row counts and field presence). F1 scoring against a ground truth remains the priority next step.

### Model List After Cleanup

Following the benchmark, `RANKED_MODELS` in `llm_models_extraction.py` was updated to reflect only the models confirmed to work at inference time:

```python
RANKED_MODELS = [
    "gpt-4o",
    "Meta-Llama-3.1-405B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "gpt-4o-mini",
]
```

---

## 6. Data Quality Investigation

### Question

Why did gpt-4o extract 70 mechanical property rows while gpt-4o-mini extracted 79? Are the 9 extra rows in gpt-4o-mini legitimate test samples, or are they duplicates or hallucinations?

### Method

A Python analysis script compared both extracted JSON files using the composite key `(heat_number, test_sample, weight_kg_per_m, yield_point_mpa)`.

```python
key = (heat_number, test_sample, weight_kg_per_m, yield_point_mpa)
```

### Finding 1 — No Internal Duplicates

Both models produced internally clean outputs with zero duplicate composite keys:

| Model | Internal Duplicates |
|-------|:------------------:|
| gpt-4o | 0 |
| gpt-4o-mini | 0 |

Neither model hallucinated repeated rows. The discrepancy is therefore a genuine difference in what was extracted, not an artefact of deduplication.

### Finding 2 — The 9-Row Gap Is Isolated to 3 Heats

| Heat | gpt-4o | gpt-4o-mini | Difference |
|------|:------:|:-----------:|:----------:|
| 2408049 | 4 | 4 | 0 |
| 2500812 | 4 | 4 | 0 |
| 2504079 | 4 | 4 | 0 |
| 2504088 | 4 | 4 | 0 |
| 2504089 | 4 | 4 | 0 |
| 2504090 | 4 | 4 | 0 |
| 2504091 | 3 | 3 | 0 |
| 2504092 | 4 | 4 | 0 |
| 2504093 | 4 | 4 | 0 |
| 2504094 | 4 | 4 | 0 |
| **2504095** | **1** | **3** | **+2** |
| **25990024** | **1** | **4** | **+3** |
| **25990031** | **1** | **5** | **+4** |
| 25990032 | 4 | 4 | 0 |
| 25990034 | 4 | 4 | 0 |
| 25990035 | 4 | 4 | 0 |
| 25990039 | 4 | 4 | 0 |
| 25990040 | 4 | 4 | 0 |
| 25990041 | 4 | 4 | 0 |
| 25990085 | 4 | 4 | 0 |
| **TOTAL** | **70** | **79** | **+9** |

The 9-row difference is entirely explained by three heat numbers — `2504095`, `25990024`, and `25990031` — where gpt-4o extracted exactly 1 row instead of the 3–5 rows that gpt-4o-mini found.

### Finding 3 — Deeper Value-Level Disagreement

Even for the 17 heats where both models extracted the same number of rows, the composite keys aligned on only ~39 of the 70 entries. The remaining disagreement breaks down into two sub-causes:

**Sub-cause A — Missing `test_sample` values in gpt-4o:**  
Of the 31 composite keys unique to gpt-4o, the majority have `test_sample=None`. gpt-4o systematically failed to parse the sample number column, causing the same physical row to be stored under a different key than gpt-4o-mini's version of the same row.

**Sub-cause B — Minor numeric precision differences:**  
Small rounding differences in `weight_kg_per_m` (e.g., 6.194 vs 6.195) and `yield_point_mpa` (e.g., 564 vs 568) between models cause the same physical measurement to generate different composite keys, inflating the apparent disagreement.

### Summary of Root Causes

| Root Cause | Heats Affected | Mechanism |
|------------|:--------------:|-----------|
| Row collapse — gpt-4o merged multiple test samples into one | 2504095, 25990024, 25990031 | Model conflates repeated similar rows |
| Missing `test_sample` column values in gpt-4o | All heats | Column parsing gap in gpt-4o prompt interpretation |
| Numeric precision / rounding differences between models | All heats | Different decimal representation of the same quantity |

### Recommendations

| Priority | Action |
|----------|--------|
| **High** | Annotate at least one certificate as a ground truth file to enable proper F1-score evaluation |
| **High** | Implement fuzzy numeric matching in `merge_extractions_v2` (e.g., weight tolerance ±0.01 kg/m, yield tolerance ±5 MPa) to prevent minor rounding differences from creating phantom unique keys |
| **Medium** | Reduce the composite deduplication key from 4 fields to 3 — drop `test_sample` since it is frequently null: `(heat_number, weight_kg_per_m, yield_point_mpa)` |
| **Medium** | Add a post-extraction completeness check: flag any heat that has fewer rows than the mode count across all heats |
| **Low** | Re-run the 3 under-extracted heats with `--consistency-samples 3` to test whether multi-sampling recovers the missing rows |

---

## 7. Conclusions & Next Steps

### What Was Accomplished

The MTC extraction pipeline was upgraded from a prototype-level benchmark script to a production-ready evaluation framework. The key additions — schema validation with retry, smart deduplication, two-pass consolidation, and cross-model ensemble voting — directly address the most common failure modes of LLM-based structured document extraction.

The benchmark confirmed that the GitHub Models inference endpoint, as of February 2026, supports 4 of the 10 originally targeted models. gpt-4o provides the best speed, while gpt-4o-mini demonstrated superior row recall on mechanical properties. The Llama 3.1 405B model occupies a middle ground — capable but slow.

The data quality investigation confirmed that the 9-row count difference between gpt-4o and gpt-4o-mini is not a deduplication artefact. It reflects real differences in how the models parse multi-sample mechanical test tables and column values — findings that directly inform the next iteration of the extraction prompt.

### Open Items

1. **Ground truth annotation** — Without a human-annotated reference file, no F1 score can be produced. This is the single highest-impact next step.
2. **Fuzzy merge** — Implement tolerance-based numeric comparison in `merge_extractions_v2`.
3. **Prompt refinement** — Update the extraction prompt to explicitly instruct models to preserve every individual test sample row and to always extract the `test_sample` column.
4. **Broader certificate coverage** — The entire benchmark was run on a single 4-page certificate. Generalisation to other certificate layouts and issuers is unknown.
5. **Two-pass and ensemble evaluation** — The `--two-pass` and `--ensemble` modes were implemented but not yet benchmarked against the single-pass baseline.

---

## 8. Appendix — Schema Reference

Schema file: `schema/mtc_extraction_schema_v1.json`

### Top-Level Structure

| Field | Type | Description |
|-------|------|-------------|
| `document` | object | Certificate number, date, issuing standard, plant |
| `traceability` | object | Order number, delivery note, customer references |
| `product` | object | Product type, nominal dimensions, weight per metre, applicable standard |
| `chemical_composition` | array | One object per heat; contains `heat_number` and an `elements` dict |
| `mechanical_properties` | array | One object per test sample; contains heat reference and all test result fields |
| `approval` | object | Inspector name, role, and any third-party certification references |

### Chemical Composition Elements

`C`, `Si`, `Mn`, `P`, `S`, `Ni`, `Cr`, `Mo`, `Cu`, `V`, `N`, `B`, `Ce`

### Mechanical Properties Fields

| Field | Unit |
|-------|------|
| `heat_number` | — |
| `test_sample` | — |
| `weight_kg_per_m` | kg/m |
| `cross_sectional_area_mm2` | mm² |
| `yield_point_mpa` | MPa |
| `tensile_strength_mpa` | MPa |
| `rm_re_ratio` | — |
| `percentage_elongation` | % |
| `agt_percent` | % |

### Composite Deduplication Key (current)

```python
(heat_number, test_sample, weight_kg_per_m, yield_point_mpa)
```

---

## 9. Appendix — Changed Artefacts

| File | Type of Change | Detail |
|------|---------------|--------|
| `src/extraction/llm_models_extraction.py` | Rewritten | 598 → 1,468 lines; 8 improvements implemented; RANKED_MODELS pruned from 10 to 4 |
| `requirements.txt` | Modified | Added `jsonschema` |
| `data/processed/benchmark_output/benchmark_summary.json` | Generated | Full benchmark run results for all 10 candidate models |
| `data/processed/benchmark_output/gpt-4o_extracted.json` | Generated | 70 mechanical entries, 20 chemical composition heats |
| `data/processed/benchmark_output/gpt-4o-mini_extracted.json` | Generated | 79 mechanical entries, 20 chemical composition heats |
| `data/processed/benchmark_output/Meta-Llama-3_1-405B-Instruct_extracted.json` | Generated | 77 mechanical entries, 20 chemical composition heats |
