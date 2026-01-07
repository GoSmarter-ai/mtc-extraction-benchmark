# MTC Extraction Benchmark — Open-Source Document AI for Mill/Material Test Certificates

## Overview

This project explores the **implementation, accuracy, and comparative performance** of multiple **open-source AI solutions** for extracting structured information from **Mill Certificates / Material Test Certificates (MTCs)**.

The goal is to design and evaluate end-to-end pipelines that:
1. Extract text from certificate documents (OCR / document understanding),
2. Parse and normalize that information into a predefined schema,
3. Measure accuracy and performance against manually extracted ground-truth values.

The final output should serve not only as an evaluation of current solutions for MTCs, but also as a **reusable template** for benchmarking document-extraction approaches across other document types in the future.

## Objectives

By the end of the 6-week project, the intern should deliver:

- A **working repository** containing:
  - Modular pipelines for training and running multiple AI extraction solutions
  - Reproducible experiments and evaluation scripts
- A **well-defined schema** for Mill / Material Test Certificates
- **Quantitative comparisons** between solutions using consistent metrics
- **Clear documentation** explaining:
  - Design decisions
  - Implementation details
  - Results and trade-offs
- A solution architecture that can be easily adapted to **other document types**

## Scope of Work

### 1. Document Understanding & OCR
Evaluate and integrate multiple open-source OCR and document-AI solutions capable of handling scanned PDFs and images of MTCs.

### 2. Information Extraction
Extract key fields from certificates into a structured schema (e.g. JSON), such as:
- Heat number
- Material grade / specification
- Chemical composition
- Mechanical properties
- Dimensions
- Manufacturer
- Certificate date
- Standards / norms referenced

### 3. Evaluation & Benchmarking
- Manually extract ground-truth values from a subset of documents
- Compare automated outputs against ground truth
- Compute and report performance metrics


## Candidate Open-Source Solutions

We need to evaluate several of the following (final selection may evolve during the project):

### OCR / Document Parsing
- **Tesseract OCR**
- **DocTR**
- **Docling**
- **PaddleOCR**
- **TrOCR**
- **Kraken**

### LLM / Document AI / Hybrid Approaches
- **Mistral (open-weight models, e.g. Mistral-7B / Mixtral)**
- **LayoutLM / LayoutLMv3**
- **Donut (Document Understanding Transformer)**
- **Nougat**
- **Open-source LLaMA-based models for extraction / parsing**

## Deliverables

### Code
- A structured Git repository containing:
  - Data ingestion and preprocessing
  - Model training and inference pipelines
  - Evaluation scripts
  - Configuration files for running experiments

### Documentation
- Literate programming–style documentation (e.g. Markdown + notebooks or Markdown + embedded results) describing:
  - What was built
  - How each model was trained or configured
  - Results and observed strengths / weaknesses
  - Recommendations for future work

### Results
- Tables and plots comparing:
  - Field-level accuracy
  - Precision / recall / F1
  - End-to-end document accuracy
  - Runtime and resource usage (where feasible)


## Provided Inputs

- A dataset of **Mill / Material Test Certificates** for:
  - Training / configuration
  - Validation
  - Manual ground-truth extraction


## Success Criteria

This project will be considered successful if:
- Multiple solutions are implemented and evaluated consistently
- Results are reproducible from the repository
- Accuracy is quantified and clearly explained
- The repository structure and documentation make it easy to:
  - Add a new model
  - Swap in a new document type
  - Re-run the evaluation pipeline

## Expected Skills & Learning Outcomes

- Practical experience with open-source document AI and OCR tools
- Building reproducible ML pipelines
- Designing evaluation frameworks for unstructured data extraction
- Clear technical communication and documentation
