# Project Plan: 6-Week MTC OCR AI Research Project

## Week 1: Onboarding & Problem Definition

### Goals
- Understand Mill / Material Test Certificates
- Review provided dataset
- Define extraction scope and schema

### Tasks
- Review sample MTCs and identify common fields
- Design an initial JSON schema for extracted data
- Survey candidate OCR and document AI solutions
- Set up repository structure and development environment

### Outputs
- Documented extraction schema
- Initial repository scaffold
- Short write-up on chosen tools and rationale

## Week 2: Baseline OCR & Text Extraction

### Goals
- Establish a baseline OCR pipeline
- Identify document quality challenges

### Tasks
- Implement at least one baseline OCR solution (e.g. Tesseract or PaddleOCR)
- Normalize OCR outputs (text, layout, coordinates if available)
- Inspect failure modes (tables, stamps, rotated text)

### Outputs
- Working OCR pipeline
- Example OCR outputs committed to repo
- Notes on OCR limitations for MTCs

## Week 3: Structured Information Extraction

### Goals
- Convert raw text into structured schema fields

### Tasks
- Implement rule-based and/or ML-based extraction
- Integrate at least one document AI model (e.g. Docling, LayoutLM, Donut)
- Map extracted fields to the schema
- Begin manual ground-truth extraction for evaluation

### Outputs
- End-to-end extraction pipeline for at least one model
- Ground-truth dataset for evaluation
- Initial accuracy measurements


## Week 4: LLM-Based & Hybrid Approaches

### Goals
- Evaluate LLM-assisted extraction strategies

### Tasks
- Integrate open-source LLMs (e.g. Mistral) for structured extraction
- Compare prompt-based vs fine-tuned approaches (if feasible)
- Standardize input/output format across all solutions

### Outputs
- LLM-based extraction pipeline
- Comparable outputs across multiple approaches
- Intermediate evaluation results

## Week 5: Evaluation & Benchmarking

### Goals
- Quantitatively compare all solutions

### Tasks
- Define and implement evaluation metrics:
  - Field-level accuracy
  - Precision / recall / F1
  - Document-level completeness
- Measure runtime and resource usage (where possible)
- Analyze error patterns by field and document type

### Outputs
- Evaluation scripts
- Results tables and plots
- Written analysis of trade-offs and performance

## Week 6: Documentation, Generalization & Final Report

### Goals
- Finalize documentation and make solution reusable

### Tasks
- Complete literate programming documentation
- Refactor pipelines for clarity and extensibility
- Demonstrate how the framework could be adapted to another document type
- Prepare final summary and recommendations

### Outputs
- Polished README and documentation
- Clean, reproducible repository
- Final report summarizing findings and next steps
  
---

## Risks & Mitigations

- **Inconsistent document formats**  
  → Focus on schema flexibility and robust evaluation

- **OCR quality issues**  
  → Compare multiple OCR engines and hybrid approaches

- **Time constraints**  
  → Prioritize reproducibility and clear comparisons over model complexity


## Final Outcome

A reusable, well-documented benchmarking framework for comparing open-source AI solutions for structured information extraction from industrial documents, starting with Mill / Material Test Certificates.
