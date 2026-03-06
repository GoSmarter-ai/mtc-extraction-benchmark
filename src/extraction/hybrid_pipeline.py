"""
HybridPipeline
==============
Intelligent routing pipeline that selects the extraction strategy based on
OCR confidence, document complexity, and cost constraints.

Routing logic
-------------
                        ┌─────────────────┐
                        │   Input PDF     │
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Run PaddleOCR         │
                    │   Compute avg confidence │
                    └────────────┬────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               │ conf ≥ 0.85     │ 0.65 ≤ conf      │ conf < 0.65
               │ (high quality)  │ < 0.85 (medium)  │ (low quality)
               ▼                 ▼                   ▼
         Rule-based          LLM text path      Vision LLM path
         extraction          (text OCR →        (images → gpt-4o
                             best LLM)          vision)
               │                 │                   │
               └────────┬────────┘                   │
                         │                            │
               ┌─────────▼──────────┐                │
               │  Completeness check│◄───────────────┘
               │  (>= MIN_FIELDS)   │
               └─────────┬──────────┘
                          │ incomplete
                          ▼
                   Escalate to vision

Usage (Python)
--------------
    from src.extraction.hybrid_pipeline import HybridPipeline

    pipeline = HybridPipeline(
        schema_path=Path("schema/mtc_extraction_schema_v1.json"),
        prompt_path=Path("prompts/mtc_llm_extraction_prompt.txt"),
    )
    result = pipeline.run(pdf_path=Path("data/raw/diler/cert.pdf"))
    print(result["strategy_used"], result["ocr_confidence"])

Usage (CLI)
-----------
    python -m src.extraction.hybrid_pipeline \\
        --pdf data/raw/diler/diler-07-07-2025-rerun-41-44.pdf \\
        --output data/processed/pipeline_output/hybrid_extracted.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]

# Confidence thresholds
HIGH_CONFIDENCE = 0.85  # → rule-based or text-LLM
MEDIUM_CONFIDENCE = 0.65  # → text-LLM
# < MEDIUM_CONFIDENCE    → vision-LLM directly

# Minimum populated field ratio to consider extraction complete
MIN_COMPLETENESS = 0.60

# Default fallback LLM model for the text path
DEFAULT_TEXT_MODEL = "gpt-4o"
DEFAULT_VISION_MODEL = "gpt-4o"


class HybridPipeline:
    """
    Intelligent routing pipeline for MTC extraction.

    Picks the cheapest extraction strategy that achieves acceptable quality,
    escalating to vision LLM only when necessary.
    """

    def __init__(
        self,
        schema_path: Path,
        prompt_path: Path,
        text_model: str = DEFAULT_TEXT_MODEL,
        vision_model: str = DEFAULT_VISION_MODEL,
        high_confidence_threshold: float = HIGH_CONFIDENCE,
        medium_confidence_threshold: float = MEDIUM_CONFIDENCE,
        min_completeness: float = MIN_COMPLETENESS,
    ):
        self.schema_path = schema_path
        self.prompt_path = prompt_path
        self.text_model = text_model
        self.vision_model = vision_model
        self.high_threshold = high_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.min_completeness = min_completeness

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Run the full hybrid pipeline on a PDF.

        Returns a dict with:
          - All extracted fields (document, chemical_composition, etc.)
          - 'strategy_used': one of "rule_based", "text_llm", "vision_llm"
          - 'ocr_confidence': float (mean confidence over all OCR tokens)
          - 'elapsed_seconds': total wall-clock time
          - 'tokens_used': total LLM tokens consumed (0 for rule-based)
        """
        t0 = time.time()

        print(f"\n🔀 HybridPipeline: {pdf_path.name}")
        ocr_texts, avg_conf = self._run_ocr(pdf_path)
        print(f"   OCR confidence: {avg_conf:.3f}")

        strategy: str
        extracted: Dict[str, Any]
        tokens_used: int = 0

        if avg_conf >= self.high_threshold:
            print(f"   → HIGH confidence ({avg_conf:.2f}) — trying rule-based …")
            extracted = self._rule_based_extract(ocr_texts)
            completeness = self._completeness(extracted)
            print(f"   Completeness: {completeness:.1%}")

            if completeness >= self.min_completeness:
                strategy = "rule_based"
            else:
                print("   Rule-based incomplete — escalating to text LLM …")
                extracted, tokens_used = self._text_llm_extract(ocr_texts)
                completeness = self._completeness(extracted)
                strategy = "text_llm"
                if completeness < self.min_completeness:
                    print("   Text LLM incomplete — escalating to vision LLM …")
                    extracted, tokens_used = self._vision_llm_extract(pdf_path)
                    strategy = "vision_llm"

        elif avg_conf >= self.medium_threshold:
            print(f"   → MEDIUM confidence ({avg_conf:.2f}) — using text LLM …")
            extracted, tokens_used = self._text_llm_extract(ocr_texts)
            completeness = self._completeness(extracted)
            strategy = "text_llm"
            if completeness < self.min_completeness:
                print("   Text LLM incomplete — escalating to vision LLM …")
                extracted, tokens_used = self._vision_llm_extract(pdf_path)
                strategy = "vision_llm"

        else:
            print(f"   → LOW confidence ({avg_conf:.2f}) — using vision LLM directly …")
            extracted, tokens_used = self._vision_llm_extract(pdf_path)
            strategy = "vision_llm"

        elapsed = round(time.time() - t0, 2)
        print(f"   ✅ Done  strategy={strategy}  time={elapsed}s  tokens={tokens_used:,}")

        return {
            **extracted,
            "_pipeline_meta": {
                "strategy_used": strategy,
                "ocr_confidence": round(avg_conf, 4),
                "elapsed_seconds": elapsed,
                "tokens_used": tokens_used,
                "text_model": self.text_model,
                "vision_model": self.vision_model,
            },
        }

    # ------------------------------------------------------------------
    # OCR layer
    # ------------------------------------------------------------------

    def _run_ocr(self, pdf_path: Path) -> tuple[List[str], float]:
        """Run PaddleOCR and return (page_texts, mean_confidence)."""
        try:
            import cv2
            import numpy as np
            from paddleocr import PaddleOCR
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR and pdf2image are required for hybrid pipeline OCR. "
                "Install with: pip install paddleocr pdf2image"
            ) from exc

        images = convert_from_path(str(pdf_path), dpi=200)
        ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
        page_texts: List[str] = []
        all_confs: List[float] = []

        for pil_img in images:
            arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            results = list(ocr_engine.predict(arr))
            blocks: List[str] = []
            if results:
                r = results[0]
                for txt, conf in zip(r.get("rec_texts", []), r.get("rec_scores", [])):
                    blocks.append(txt)
                    all_confs.append(float(conf))
            page_texts.append("\n".join(blocks))

        avg_conf = sum(all_confs) / max(len(all_confs), 1)
        return page_texts, avg_conf

    # ------------------------------------------------------------------
    # Rule-based extraction layer
    # ------------------------------------------------------------------

    def _rule_based_extract(self, page_texts: List[str]) -> Dict[str, Any]:
        """
        Lightweight regex-based extractor.  Returns a partial result dict;
        blank / None fields are acceptable and trigger LLM escalation.
        """
        import re

        full_text = "\n".join(page_texts)

        # ── document ────────────────────────────────────────────────────
        cert_match = re.search(
            r"(?:certificate\s*(?:no|number)[.:\s]*)([\w\-/]+)",
            full_text,
            re.IGNORECASE,
        )
        date_match = re.search(
            r"(?:date[.:\s]*)(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[./-]\d{2}[./-]\d{2})",
            full_text,
            re.IGNORECASE,
        )

        return {
            "document": {
                "certificate_number": (cert_match.group(1).strip() if cert_match else None),
                "issuing_date": date_match.group(1).strip() if date_match else None,
                "standard": None,
                "customer": None,
                "order_number": None,
            },
            "traceability": {},
            "product": {},
            "chemical_composition": [],
            "mechanical_properties": [],
            "approval": {},
        }

    # ------------------------------------------------------------------
    # Text LLM extraction layer
    # ------------------------------------------------------------------

    def _text_llm_extract(self, page_texts: List[str]) -> tuple[Dict[str, Any], int]:
        """Send concatenated OCR text to the text-only LLM and return JSON."""
        # Lazy import to avoid circular dependency
        from src.extraction.llm_models_extraction import LLMModelBenchmark

        bench = LLMModelBenchmark(
            schema_path=self.schema_path,
            prompt_path=self.prompt_path,
        )
        result = bench.extract_with_validation(
            page_texts=page_texts,
            model_id=self.text_model,
        )
        tokens = result.pop("_tokens_used", 0)
        return result, tokens

    # ------------------------------------------------------------------
    # Vision LLM extraction layer
    # ------------------------------------------------------------------

    def _vision_llm_extract(self, pdf_path: Path) -> tuple[Dict[str, Any], int]:
        """Delegate to VisionExtractor and return (result, tokens_used)."""
        from src.extraction.vision_extraction import VisionExtractor

        extractor = VisionExtractor(
            schema_path=self.schema_path,
            prompt_path=self.prompt_path,
            model=self.vision_model,
        )
        # VisionExtractor.extract is already instrumented with token printing;
        # capture token count from stdout is fragile, so we call it directly.
        result = extractor.extract(pdf_path=pdf_path)
        # Token count isn't tracked separately at the pipeline level here;
        # the VisionExtractor already prints it.  Return 0 as placeholder.
        return result, 0

    # ------------------------------------------------------------------
    # Completeness check
    # ------------------------------------------------------------------

    @staticmethod
    def _completeness(extracted: Dict[str, Any]) -> float:
        """
        Heuristic completeness score in [0, 1].

        Counts populated non-None leaves against an expected minimum set.
        """
        expected_keys = [
            ("document", "certificate_number"),
            ("document", "issuing_date"),
            ("document", "standard"),
            ("traceability", "consignment_number"),
            ("product", "size"),
            ("product", "quality"),
        ]
        filled = sum(
            1 for section, key in expected_keys if extracted.get(section, {}).get(key) is not None
        )
        scalar_score = filled / len(expected_keys)

        has_chem = 1.0 if extracted.get("chemical_composition") else 0.0
        has_mech = 1.0 if extracted.get("mechanical_properties") else 0.0

        return (scalar_score + has_chem + has_mech) / 3.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the hybrid MTC extraction pipeline on a PDF.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the source PDF.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "schema" / "mtc_extraction_schema_v1.json",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=REPO_ROOT / "prompts" / "mtc_llm_extraction_prompt.txt",
    )
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Save extracted JSON here (default: print to stdout).",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"❌ PDF not found: {args.pdf}")
        return 1

    pipeline = HybridPipeline(
        schema_path=args.schema,
        prompt_path=args.prompt,
        text_model=args.text_model,
        vision_model=args.vision_model,
    )
    result = pipeline.run(pdf_path=args.pdf)

    output_json = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)
        print(f"💾 Saved → {args.output}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    exit(main())
