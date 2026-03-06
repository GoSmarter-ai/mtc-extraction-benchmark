"""Pydantic models for the MTC Extraction REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ExtractionRequest(BaseModel):
    model: str = Field(
        default="gpt-4o",
        description="LLM model ID to use for extraction.",
    )
    mode: str = Field(
        default="text",
        description="Extraction mode: 'text', 'vision', or 'hybrid'.",
    )
    two_pass: bool = Field(
        default=False,
        description="Enable two-pass consolidation.",
    )
    numeric_tolerance: float = Field(
        default=0.001,
        ge=0,
        description="Numeric tolerance for post-extraction evaluation.",
    )


class ExtractionResponse(BaseModel):
    run_id: str
    model: str
    mode: str
    elapsed_seconds: float
    tokens_used: int
    extracted: Dict[str, Any]
    validation_errors: List[str] = []


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    models: Optional[List[str]] = Field(
        default=None,
        description="List of model IDs to benchmark (defaults to all ranked models).",
    )
    use_cached_ocr: bool = Field(default=True)
    two_pass: bool = Field(default=False)
    ground_truth_path: Optional[str] = Field(
        default=None,
        description="Relative path to a ground-truth JSON for evaluation.",
    )
    numeric_tolerance: float = Field(default=0.001, ge=0)


class ModelResult(BaseModel):
    model: str
    status: str  # "success" | "error"
    elapsed_seconds: Optional[float] = None
    tokens_used: Optional[int] = None
    mech_count: Optional[int] = None
    chem_heats: Optional[int] = None
    overall_f1: Optional[float] = None
    error: Optional[str] = None


class BenchmarkResponse(BaseModel):
    run_id: str
    models_run: int
    results: List[ModelResult]
    output_dir: str


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationRequest(BaseModel):
    prediction_path: str = Field(
        description="Relative path to the model's extracted JSON file.",
    )
    ground_truth_path: str = Field(
        description="Relative path to the ground-truth annotation file.",
    )
    numeric_tolerance: float = Field(default=0.001, ge=0)
    mech_weight_tol: float = Field(default=0.01, ge=0)
    mech_yield_tol: float = Field(default=5.0, ge=0)


class EvaluationResponse(BaseModel):
    overall_f1: float
    document: Dict[str, Any]
    chemical: Dict[str, Any]
    mechanical: Dict[str, Any]
    approval: Dict[str, Any]
    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    id: str
    label: str
    provider: str
    tier: str


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    count: int
