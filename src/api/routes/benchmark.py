"""
/benchmark and /evaluate endpoints.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from src.api.models import (
    BenchmarkRequest,
    BenchmarkResponse,
    EvaluationRequest,
    EvaluationResponse,
    ModelInfo,
    ModelResult,
    ModelsResponse,
)

router = APIRouter(tags=["benchmark"])

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """Return all ranked models available for extraction."""
    from src.extraction.llm_models_extraction import RANKED_MODELS

    return ModelsResponse(
        models=[ModelInfo(**m) for m in RANKED_MODELS],
        count=len(RANKED_MODELS),
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@router.post("/benchmark", response_model=BenchmarkResponse)
def run_benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    """
    Run the full benchmark across one or more models.

    This is a synchronous endpoint — it blocks until all models finish.
    For large benchmarks consider running the CLI directly.
    """
    from src.extraction.llm_models_extraction import RANKED_MODELS, LLMModelBenchmark

    schema_path = REPO_ROOT / "schema" / "mtc_extraction_schema_v1.json"
    prompt_path = REPO_ROOT / "prompts" / "mtc_llm_extraction_prompt.txt"
    ocr_dir = REPO_ROOT / "data" / "processed" / "paddle_ocr"
    output_dir = REPO_ROOT / "data" / "processed" / "benchmark_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_ids = req.models or [m["id"] for m in RANKED_MODELS]
    run_id = str(uuid.uuid4())[:8]

    try:
        bench = LLMModelBenchmark(schema_path=schema_path, prompt_path=prompt_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialise benchmark: {exc}",
        ) from exc

    if req.use_cached_ocr:
        pages = bench.load_ocr_pages(ocr_dir)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="use_cached_ocr=false requires a PDF path; use the CLI for fresh OCR.",
        )

    gt_data = None
    if req.ground_truth_path:
        gt_path = REPO_ROOT / req.ground_truth_path
        if not gt_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ground truth not found: {req.ground_truth_path}",
            )
        gt_data = json.loads(gt_path.read_text())

    all_runs = bench.benchmark(
        page_texts=pages,
        models=model_ids,
        ground_truth=gt_data,
        output_dir=output_dir,
        two_pass=req.two_pass,
    )

    results: list[ModelResult] = []
    for model_id, run in all_runs.items():
        if model_id == "__ensemble__":
            continue
        extracted = run.get("result") or {}
        field_metrics = run.get("field_metrics") or {}
        tok = run.get("token_usage") or {}
        results.append(
            ModelResult(
                model=model_id,
                status=run.get("status", "error"),
                elapsed_seconds=run.get("elapsed"),
                tokens_used=tok.get("total_tokens"),
                mech_count=len(extracted.get("mechanical_properties") or []),
                chem_heats=len(extracted.get("chemical_composition") or []),
                overall_f1=field_metrics.get("overall_f1"),
                error=run.get("error"),
            )
        )

    return BenchmarkResponse(
        run_id=run_id,
        models_run=len(results),
        results=results,
        output_dir=str(output_dir),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_extraction(req: EvaluationRequest) -> EvaluationResponse:
    """
    Evaluate a model extraction against a ground-truth annotation.

    Both paths are relative to the repository root.
    """
    from src.evaluation import MTCEvaluator

    pred_path = REPO_ROOT / req.prediction_path
    gt_path = REPO_ROOT / req.ground_truth_path

    if not pred_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction file not found: {req.prediction_path}",
        )
    if not gt_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ground truth file not found: {req.ground_truth_path}",
        )

    extracted = json.loads(pred_path.read_text())
    gt = json.loads(gt_path.read_text())

    report = MTCEvaluator.evaluate(
        extracted=extracted,
        ground_truth=gt,
        numeric_tolerance=req.numeric_tolerance,
        mech_weight_tol=req.mech_weight_tol,
        mech_yield_tol=req.mech_yield_tol,
    )

    return EvaluationResponse(**report)
