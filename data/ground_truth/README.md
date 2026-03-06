# Ground Truth Annotations

This directory contains human-annotated reference JSON files used to evaluate extraction accuracy.

## File Naming Convention

```
{document-id}_gt.json
```

## How to Use

1. Run the benchmark first to generate a model extraction as a starting point:
   ```bash
   python src/extraction/llm_models_extraction.py --use-cached-ocr --models gpt-4o --output data/processed/
   ```

2. Copy the extraction as a template:
   ```bash
   cp data/processed/benchmark_output/gpt-4o_extracted.json data/ground_truth/{document-id}_gt.json
   ```

3. Open the `_gt.json` file and **manually verify / correct every value** against the original PDF:
   - `data/raw/diler/` — contains the original source PDFs

4. Fields that are `null` or missing in the model output should be filled in from the physical document.

## Annotation Priority

Review in this order (highest impact first):

| Section | Key Fields | notes |
|---|---|---|
| `chemical_composition` | `elements.*` per heat | Float values, verify to 4 decimal places |
| `mechanical_properties` | all numeric fields (yield, tensile, etc.) | Verify row count matches original |
| `document` | `certificate_number`, `issuing_date`, `standard` | Exact string match |
| `approval` | `cares_approved`, `certificate_of_approval_number` | Boolean + strings |
| `traceability` | `consignment_number`, `lot_number` | Exact strings |

## Seeded Files

| File | Source | Status |
|---|---|---|
| `diler-07-07-2025-rerun-41-44_gt.json` | Seeded from gpt-4o extraction | **⚠ Needs human verification** |

> **Warning**: Seeded files are NOT validated ground truth. They are only a convenient starting
> point. Every value must be verified against the original PDF before use in evaluation.

## Evaluating Against Ground Truth

```bash
python -m src.evaluation.evaluator \
  --prediction data/processed/benchmark_output/gpt-4o_extracted.json \
  --ground-truth data/ground_truth/diler-07-07-2025-rerun-41-44_gt.json \
  --output data/comparison/gpt-4o_vs_gt.json
```
