# MTC Extraction Benchmark — Open-Source Document AI for Mill/Material Test Certificates

**Goal**  
Build a reproducible, extensible benchmark and pipeline repository that extracts **structured data** from mill/material test certificates (MTCs) into a defined schema, comparing multiple open‑source OCR/document‑AI stacks. Implement, run, and compare these solutions; report accuracy/performance versus a manually curated ground truth; and package everything as a **template** for other document types.

---

## Outcomes & Deliverables

1. **Repository:** Modular pipelines with pluggable backends (Docling, Tesseract, PaddleOCR, Donut, TrOCR, etc.), Docker, Makefile, CI, YAML configs, and literate docs.
2. **Schema & Samples:** JSON Schema for MTCs with examples and converters to CSV/Parquet.
3. **Evaluation Report:** Accuracy (text & table), entity-level F1, normalized edit distance, document-level exact match, throughput, and resource use—plus error analysis.
4. **Template Mode:** Clear contribution guide to compare solutions for other document types.
5. **Ground Truth:** Manually curated dataset aligned to the schema with provenance.

---

## Scope

- Focus on **steel & aluminium** MTCs (PDFs/images; scanned and text), multi-page, with **chemistry tables**, **mechanical properties**, **grades**, **heat/lot**, **dimensions**, and **test methods/standards** (EN/ASTM).
- Compare **5–7 solutions** across OCR-first and OCR-free/document-AI approaches.
- Include **Docling** and **Mistral (LLM)** as specified; integrate **OCRmyPDF** for scans.

---

## Candidate Open‑Source Solutions

- **Docling (IBM/LF AI & Data)** — advanced document parsing, OCR support, structured outputs.  
- **Tesseract OCR** — classic OCR, many languages; pair with OCRmyPDF for scans.  
- **PaddleOCR** — end-to-end OCR & document AI (PP-Structure, PaddleOCR‑VL), multilingual.  
- **Donut (NAVER CLOVA)** — OCR‑free document understanding, direct JSON generation.  
- **TrOCR (Microsoft)** — Transformer-based OCR (printed/handwritten text-line).  
- **pdfplumber/Camelot/Tabula** — table extraction for text PDFs.  
- **Mistral (open‑weight LLMs)** — post‑OCR schema mapping and normalization.

> Note: You can optionally test **docTR/MMOCR**; in 6 weeks choose **one** to keep scope lean.

---

## MTC JSON Schema (v1)

See `schemas/mtc.schema.json` for full specification. Key fields:

- `issuer`, `supplier`, `heat_number`, `lot_number`, `material_grade`, `standard`
- `dimensions` (shape, thickness/width/length/diameter)
- `chemistry` array of `{element, percent}`
- `mechanical_properties` (YS, UTS, elongation, hardness)
- `tests` array with `{test_method, spec_ref, location, result}`
- `dates` (manufacture, test, certificate)
- `notes`, `attachments`, `source_file`

---

## Repository Structure

```text
data/            # PDFs/images and annotations
schemas/         # JSON Schema(s)
configs/         # per-backend & global YAML configs
backends/        # backend modules providing run_backend()
pipeline/        # shared steps: ingest/ocr/layout/tables/entities/export
eval/            # metrics & evaluation harness
docs/            # Quarto/nbdev notebooks and rendered reports
```
