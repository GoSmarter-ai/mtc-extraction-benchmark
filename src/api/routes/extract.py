"""
/extract endpoint — upload a PDF and extract structured MTC data.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from src.api.models import ExtractionResponse

router = APIRouter(prefix="/extract", tags=["extraction"])

REPO_ROOT = Path(__file__).resolve().parents[3]


@router.post("/", response_model=ExtractionResponse, status_code=status.HTTP_200_OK)
async def extract(
    pdf: UploadFile = File(..., description="MTC certificate PDF"),
    model: str = Form(default="gpt-4o"),
    mode: str = Form(default="text"),
    two_pass: bool = Form(default=False),
    numeric_tolerance: float = Form(default=0.001),
) -> ExtractionResponse:
    """
    Extract structured data from an MTC certificate PDF.

    - **mode=text**: OCR → LLM text extraction (default, fast)
    - **mode=vision**: direct image → vision LLM (no OCR step)
    - **mode=hybrid**: auto-route based on OCR confidence
    """
    if mode not in ("text", "vision", "hybrid"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid mode '{mode}'. Choose from: text, vision, hybrid.",
        )
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must be a PDF.",
        )

    run_id = str(uuid.uuid4())[:8]
    schema_path = REPO_ROOT / "schema" / "mtc_extraction_schema_v1.json"
    prompt_path = REPO_ROOT / "prompts" / "mtc_llm_extraction_prompt.txt"

    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await pdf.read())
        tmp_path = Path(tmp.name)

    t0 = time.time()
    tokens_used = 0
    extracted: dict = {}
    validation_errors: list[str] = []

    try:
        if mode == "vision":
            from src.extraction.vision_extraction import VisionExtractor

            extractor = VisionExtractor(
                schema_path=schema_path,
                prompt_path=prompt_path,
                model=model,
            )
            extracted = extractor.extract(pdf_path=tmp_path)

        elif mode == "hybrid":
            from src.extraction.hybrid_pipeline import HybridPipeline

            pipeline = HybridPipeline(
                schema_path=schema_path,
                prompt_path=prompt_path,
                text_model=model,
                vision_model=model,
            )
            result = pipeline.run(pdf_path=tmp_path)
            meta = result.pop("_pipeline_meta", {})
            tokens_used += meta.get("tokens_used", 0)
            extracted = result

        else:  # text mode
            from src.extraction.llm_models_extraction import LLMModelBenchmark

            bench = LLMModelBenchmark(
                schema_path=schema_path,
                prompt_path=prompt_path,
            )
            pages = bench.run_ocr_fresh(pdf_path=tmp_path)
            if two_pass:
                result = bench.extract_two_pass(pages, model_id=model)
            else:
                result = bench.extract_with_validation(page_texts=pages, model_id=model)
            tokens_used = result.pop("_tokens_used", 0)
            validation_errors = result.pop("_validation_errors", [])
            extracted = result

    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {exc}",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return ExtractionResponse(
        run_id=run_id,
        model=model,
        mode=mode,
        elapsed_seconds=round(time.time() - t0, 2),
        tokens_used=tokens_used,
        extracted=extracted,
        validation_errors=validation_errors,
    )
