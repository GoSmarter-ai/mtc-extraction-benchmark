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

## Getting Started

### Development Environment

This project includes a complete development container configuration for consistent and reproducible development environments.

**Quick Start with GitHub Codespaces:**
1. Click the "Code" button above
2. Select "Codespaces" tab
3. Click "Create codespace on main"

**Or use VS Code locally:**
1. Install [Docker](https://www.docker.com/products/docker-desktop) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Clone this repository
4. Open in VS Code and click "Reopen in Container" when prompted

The devcontainer includes:
- Python 3.11 with all required ML/AI packages
- OCR tools (Tesseract, PaddleOCR, DocTR)
- PyTorch with CUDA support for GPU acceleration
- Jupyter Lab for notebook-based development
- All document processing libraries

### Documentation

- **[DevContainer Iteration Guide](docs/devcontainer-iteration.md)**: Learn how to modify and customize the development environment
- **[GitHub Codespaces Guide](docs/codespaces-guide.md)**: Complete guide to working with GitHub Codespaces
- **[GitHub Models Integration](docs/github-models-integration.md)**: How to use GitHub Models API for LLM-based extraction

### Project Structure

```
mtc-extraction-benchmark/
├── .devcontainer/          # Development container configuration
│   ├── devcontainer.json   # VS Code devcontainer settings
│   └── Dockerfile          # Container image with dependencies
├── docs/                   # Documentation
│   ├── devcontainer-iteration.md
│   ├── codespaces-guide.md
│   └── github-models-integration.md
├── data/                   # Dataset directory (not in repo)
├── src/                    # Source code (to be created)
├── notebooks/              # Jupyter notebooks (to be created)
├── scripts/                # Utility scripts (to be created)
├── tests/                  # Test suite (to be created)
└── README.md              # This file
```
