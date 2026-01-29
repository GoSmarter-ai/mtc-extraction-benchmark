# Week 2 Findings: Baseline OCR for Mill Test Certificates (MTCs)

## Overview

During Week 2, the objective was to establish a baseline OCR pipeline for Mill / Material Test Certificates (MTCs) and to understand the limitations of traditional OCR-based approaches when applied to these documents.

Two OCR engines were evaluated, including **Tesseract OCR** and **PaddleOCR**, using scanned PDF certificates provided by the company. The focus was on text extraction quality, layout preservation, and suitability for downstream structured information extraction.

---

## Implemented OCR Pipelines

### Baseline OCR Workflow

The following baseline pipeline was implemented and committed to the repository:

1. Convert scanned PDF pages to images (300 DPI)
2. Apply basic image preprocessing:
   - Grayscale conversion
   - Median blur
   - Otsu thresholding
3. Run OCR engine (Tesseract / PaddleOCR)
4. Export:
   - Plain text output per page
   - Word-level bounding boxes with confidence scores

Artifacts saved to the repository include:
- Page-level extracted text files
- Bounding-box JSON files
- Processed page images

This pipeline provides a reproducible baseline for comparison with more advanced document AI models.

---

## OCR Engines Evaluated

### 1. Tesseract OCR

**Observations:**
- Poor handling of dense tabular data
- Frequent merging of adjacent columns
- Inconsistent recognition of numeric values
- Loss of semantic relationships between headers and values
- Struggles with stamps, seals, and overlaid text

**Conclusion:**
Tesseract performs poorly on MTCs and is unsuitable as a standalone extraction solution. Its outputs are useful only as a diagnostic baseline.

---

### 2. PaddleOCR

**Observations:**
- Better character recognition than Tesseract
- Slightly improved handling of tables
- Still unreliable column alignment
- Numeric precision errors remain common
- Limited understanding of multi-heat tables

**Conclusion:**
PaddleOCR shows marginal improvement over Tesseract but still fails to robustly preserve document structure required for accurate schema mapping.

---

## Key OCR Failure Modes Identified

### 1. Table Structure Loss
- Chemical composition and mechanical property tables are flattened into text
- Column boundaries are not preserved
- Values are frequently associated with the wrong headers

### 2. Multi-Heat Number Ambiguity
- OCR cannot reliably associate rows or columns with specific heat numbers
- Critical traceability relationships are lost

### 3. Unit Separation
- Units (e.g. %, MPa) are often detached from their values
- Requires heuristic post-processing, increasing brittleness

### 4. Rotation and Skew
- Slight page skew causes significant degradation
- OCR confidence drops sharply for rotated or scanned documents

---

## Why OCR Alone Is Insufficient for MTC Extraction

Mill Test Certificates are **layout-heavy, semi-structured documents** where meaning is conveyed through:
- Table geometry
- Row/column alignment
- Spatial grouping
- Section headers

Traditional OCR systems:
- Extract characters, not structure
- Do not reason about layout
- Cannot reliably infer relationships between values

As a result, OCR-only pipelines produce outputs that are **not reliable for direct schema population**.

---

## Rationale for Moving to Layout-Aware Document AI Models

Given the limitations observed, the project will transition to evaluating **layout-aware document AI models**, including:

- **LayoutLM**
- **Docling**
- Other document understanding transformers (e.g. Donut)

These models:
- Use OCR outputs as input tokens
- Incorporate spatial layout and visual context
- Are designed to understand tables, forms, and key-value relationships

OCR outputs from Week 2 will be retained as:
- A diagnostic baseline
- A control for comparative evaluation
- Evidence of improvement achieved by advanced models

---

## Summary

- A working OCR baseline pipeline has been implemented
- OCR outputs have been committed to the repository
- Significant limitations were identified when applying OCR to MTCs
- These findings justify the transition to layout-aware document AI models in Week 3

The results from this week establish a clear baseline and provide a strong motivation for exploring advanced document understanding approaches.