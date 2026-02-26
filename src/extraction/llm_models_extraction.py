"""
Multi-Model LLM Extraction Benchmark
======================================
Benchmarks all available GitHub Models for MTC structured data extraction.
Re-uses existing OCR text from the pipeline to isolate LLM performance.

Models ranked best-to-last for structured extraction:
  1. gpt-4o                        – best reasoning & JSON compliance
  2. Meta-Llama-3.1-405B-Instruct  – current default, very strong open-source
  3. Mistral-large-2407            – strong JSON generation
  4. Meta-Llama-3.1-70B-Instruct   – solid mid-tier
  5. Meta-Llama-3-70B-Instruct     – older generation
  6. Mistral-Nemo                  – lightweight Mistral
  7. Meta-Llama-3.1-8B-Instruct    – small model
  8. Meta-Llama-3-8B-Instruct      – small, older
  9. gpt-4o-mini                   – cost-optimized OpenAI
 10. AI21-Jamba-Instruct           – experimental Mamba architecture

Usage:
    # Run all models (best to last):
    python src/extraction/llm_models_extraction.py

    # Run specific models:
    python src/extraction/llm_models_extraction.py --models gpt-4o Meta-Llama-3.1-405B-Instruct

    # Use pre-extracted OCR text (skip PDF→OCR step):
    python src/extraction/llm_models_extraction.py --use-cached-ocr

    # Compare against ground truth:
    python src/extraction/llm_models_extraction.py --ground-truth data/ground_truth/cert.json
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Available models on GitHub Models – ordered best → last
# ---------------------------------------------------------------------------
RANKED_MODELS: List[Dict[str, str]] = [
    {
        "id": "gpt-4o",
        "label": "GPT-4o",
        "provider": "OpenAI",
        "tier": "top",
    },
    {
        "id": "Meta-Llama-3.1-405B-Instruct",
        "label": "Llama 3.1 405B",
        "provider": "Meta",
        "tier": "top",
    },
    {
        "id": "Mistral-large-2407",
        "label": "Mistral Large",
        "provider": "Mistral",
        "tier": "high",
    },
    {
        "id": "Meta-Llama-3.1-70B-Instruct",
        "label": "Llama 3.1 70B",
        "provider": "Meta",
        "tier": "high",
    },
    {
        "id": "Meta-Llama-3-70B-Instruct",
        "label": "Llama 3 70B",
        "provider": "Meta",
        "tier": "mid",
    },
    {
        "id": "Mistral-Nemo",
        "label": "Mistral Nemo",
        "provider": "Mistral",
        "tier": "mid",
    },
    {
        "id": "Meta-Llama-3.1-8B-Instruct",
        "label": "Llama 3.1 8B",
        "provider": "Meta",
        "tier": "small",
    },
    {
        "id": "Meta-Llama-3-8B-Instruct",
        "label": "Llama 3 8B",
        "provider": "Meta",
        "tier": "small",
    },
    {
        "id": "gpt-4o-mini",
        "label": "GPT-4o Mini",
        "provider": "OpenAI",
        "tier": "small",
    },
    {
        "id": "AI21-Jamba-Instruct",
        "label": "AI21 Jamba",
        "provider": "AI21",
        "tier": "experimental",
    },
]


# ---------------------------------------------------------------------------
# Core extraction class
# ---------------------------------------------------------------------------
class LLMModelBenchmark:
    """Benchmark multiple LLM models for MTC extraction quality."""

    def __init__(
        self,
        schema_path: Path,
        prompt_path: Path,
        max_tokens: int = 16384,
    ):
        self.schema = json.loads(schema_path.read_text())
        self.system_prompt = prompt_path.read_text()
        self.max_tokens = max_tokens

        # Single client – all models share the same endpoint
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ["GITHUB_TOKEN"],
        )
        print("✅ LLM client initialised (models.inference.ai.azure.com)\n")

    # ------------------------------------------------------------------
    # OCR text helpers
    # ------------------------------------------------------------------
    @staticmethod
    def load_ocr_pages(text_dir: Path) -> List[str]:
        """Load pre-extracted OCR page texts from a directory."""
        pages = []
        for txt_file in sorted(text_dir.glob("*.txt")):
            content = txt_file.read_text()
            # Strip the header lines added by the pipeline
            lines = content.split("\n")
            if lines and lines[0].startswith("Page"):
                # Skip "Page N" and "===..." header
                content = "\n".join(lines[2:]).strip()
            pages.append(content)

        if not pages:
            raise FileNotFoundError(f"No .txt files found in {text_dir}")

        print(f"📄 Loaded {len(pages)} cached OCR pages from {text_dir}")
        return pages

    @staticmethod
    def run_ocr_fresh(pdf_path: Path, dpi: int = 200, max_width: int = 2000) -> List[str]:
        """Run PaddleOCR on a PDF and return page texts (imports heavy deps lazily)."""
        import gc

        import cv2
        import numpy as np
        from paddleocr import PaddleOCR
        from pdf2image import convert_from_path
        from PIL import Image as PILImage

        print(f"📄 Converting PDF → images (DPI={dpi}) ...")
        images = convert_from_path(str(pdf_path), dpi=dpi)
        print(f"   ✓ {len(images)} page(s)")

        ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
        page_texts: List[str] = []

        for page_num, pil_img in enumerate(images, 1):
            print(f"   OCR page {page_num} …")
            if pil_img.width > max_width:
                ratio = max_width / pil_img.width
                pil_img = pil_img.resize((max_width, int(pil_img.height * ratio)), PILImage.LANCZOS)
            arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            results = list(ocr_engine.predict(arr))

            blocks = []
            if results:
                r = results[0]
                for txt, conf in zip(r.get("rec_texts", []), r.get("rec_scores", [])):
                    blocks.append(f"{txt} (confidence: {conf:.4f})")

            page_texts.append("\n".join(blocks))
            del pil_img, arr, results
            gc.collect()

        del images
        gc.collect()
        return page_texts

    # ------------------------------------------------------------------
    # Single-model extraction
    # ------------------------------------------------------------------
    def extract_with_model(
        self,
        model_id: str,
        ocr_text: str,
        page_info: str = "",
    ) -> dict:
        """Call a single model and return parsed JSON."""
        user_prompt = (
            f"SCHEMA:\n{json.dumps(self.schema, indent=2)}\n\n"
            f'OCR TEXT{page_info}:\n"""\n{ocr_text}\n"""'
        )

        response = self.client.chat.completions.create(
            model=model_id,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if "```json" in raw:
            raw = raw[raw.find("```json") + 7 :]
            raw = raw[: raw.find("```")].strip()
        elif "```" in raw:
            raw = raw[raw.find("```") + 3 :]
            raw = raw[: raw.find("```")].strip()

        return json.loads(raw)

    # ------------------------------------------------------------------
    # Merge per-page results (same logic as complete_pipeline)
    # ------------------------------------------------------------------
    @staticmethod
    def merge_extractions(results: List[dict]) -> dict:
        if not results:
            return {}
        if len(results) == 1:
            return results[0]

        merged = results[0].copy()

        seen_heats = {item["heat_number"] for item in merged.get("chemical_composition", [])}
        for r in results[1:]:
            for chem in r.get("chemical_composition", []):
                if chem["heat_number"] not in seen_heats:
                    merged.setdefault("chemical_composition", []).append(chem)
                    seen_heats.add(chem["heat_number"])

        for r in results[1:]:
            for mech in r.get("mechanical_properties", []):
                merged.setdefault("mechanical_properties", []).append(mech)

        for r in results:
            if r.get("approval", {}).get("certificate_of_approval_number"):
                merged["approval"] = r["approval"]
                break

        return merged

    # ------------------------------------------------------------------
    # Run one model end-to-end (chunked by page, then merged)
    # ------------------------------------------------------------------
    def run_model(
        self,
        model_id: str,
        page_texts: List[str],
    ) -> Dict:
        """
        Run extraction for a single model across all pages.

        Returns dict with keys: result, elapsed, status, error
        """
        start = time.time()
        try:
            page_results = []
            for i, text in enumerate(page_texts, 1):
                if not text.strip():
                    continue
                result = self.extract_with_model(
                    model_id, text, page_info=f" (Page {i}/{len(page_texts)})"
                )
                chem = len(result.get("chemical_composition", []))
                mech = len(result.get("mechanical_properties", []))
                print(f"      Page {i}: {chem} heats, {mech} samples")
                page_results.append(result)

            merged = self.merge_extractions(page_results)
            elapsed = time.time() - start
            return {
                "result": merged,
                "elapsed": round(elapsed, 2),
                "status": "success",
                "error": None,
            }

        except Exception as exc:
            elapsed = time.time() - start
            return {
                "result": None,
                "elapsed": round(elapsed, 2),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------------
    # Comparison metrics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_metrics(
        extracted: dict,
        ground_truth: Optional[dict] = None,
    ) -> Dict:
        """Compute quick quality metrics for an extraction result."""
        m: Dict = {}

        m["certificate_number"] = extracted.get("document", {}).get("certificate_number", "N/A")
        m["chemical_count"] = len(extracted.get("chemical_composition", []))
        m["mechanical_count"] = len(extracted.get("mechanical_properties", []))
        m["has_approval"] = bool(
            extracted.get("approval", {}).get("certificate_of_approval_number")
        )

        # Collect unique heat numbers from both sections
        chem_heats = {
            c["heat_number"]
            for c in extracted.get("chemical_composition", [])
            if c.get("heat_number")
        }
        mech_heats = {
            mp["heat_number"]
            for mp in extracted.get("mechanical_properties", [])
            if mp.get("heat_number")
        }
        m["unique_chem_heats"] = len(chem_heats)
        m["unique_mech_heats"] = len(mech_heats)

        # Check JSON validity / completeness flags
        m["has_document"] = bool(extracted.get("document"))
        m["has_traceability"] = bool(extracted.get("traceability"))
        m["has_product"] = bool(extracted.get("product"))

        if ground_truth:
            gt_chem = len(ground_truth.get("chemical_composition", []))
            gt_mech = len(ground_truth.get("mechanical_properties", []))
            m["gt_chemical_count"] = gt_chem
            m["gt_mechanical_count"] = gt_mech
            m["chemical_match"] = m["chemical_count"] == gt_chem
            m["mechanical_match"] = m["mechanical_count"] == gt_mech

            # Field-level comparison for document section
            gt_doc = ground_truth.get("document", {})
            ex_doc = extracted.get("document", {})
            m["cert_number_match"] = gt_doc.get("certificate_number") == ex_doc.get(
                "certificate_number"
            )
            m["date_match"] = gt_doc.get("issuing_date") == ex_doc.get("issuing_date")

        return m

    # ------------------------------------------------------------------
    # Full benchmark run
    # ------------------------------------------------------------------
    def benchmark(
        self,
        page_texts: List[str],
        models: Optional[List[str]] = None,
        ground_truth: Optional[dict] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Dict]:
        """
        Run benchmark across multiple models.

        Args:
            page_texts: List of OCR text strings (one per page)
            models:     List of model IDs to test (default: all RANKED_MODELS)
            ground_truth: Optional ground-truth dict for comparison
            output_dir:   Directory to save per-model JSON outputs

        Returns:
            Dict mapping model_id → { result, elapsed, status, metrics }
        """
        if models is None:
            model_ids = [m["id"] for m in RANKED_MODELS]
        else:
            model_ids = models

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict] = {}
        total = len(model_ids)

        for idx, model_id in enumerate(model_ids, 1):
            print(f"\n{'=' * 70}")
            print(f"  [{idx}/{total}] 🧪 {model_id}")
            print(f"{'=' * 70}")

            run = self.run_model(model_id, page_texts)

            if run["status"] == "success" and run["result"]:
                run["metrics"] = self.compute_metrics(run["result"], ground_truth)

                # Save individual model output
                if output_dir:
                    safe_name = model_id.replace("/", "_").replace(".", "_")
                    out_file = output_dir / f"{safe_name}_extracted.json"
                    out_file.write_text(json.dumps(run["result"], indent=2))
                    print(f"   💾 Saved → {out_file.name}")
            else:
                run["metrics"] = {}
                print(f"   ❌ {run['error']}")

            results[model_id] = run

        # ----- Print comparison table -----
        self._print_table(results)

        # ----- Save summary -----
        if output_dir:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "models_tested": total,
                "results": {
                    mid: {
                        "status": r["status"],
                        "elapsed_seconds": r["elapsed"],
                        "error": r["error"],
                        "metrics": r["metrics"],
                    }
                    for mid, r in results.items()
                },
            }
            summary_file = output_dir / "benchmark_summary.json"
            summary_file.write_text(json.dumps(summary, indent=2))
            print(f"\n💾 Summary saved → {summary_file}")

        return results

    # ------------------------------------------------------------------
    # Pretty-print results table
    # ------------------------------------------------------------------
    @staticmethod
    def _print_table(results: Dict[str, Dict]) -> None:
        hdr = (
            f"\n{'Model':<38} {'Status':<9} {'Time':>7} "
            f"{'Chem#':>6} {'Mech#':>6} {'Heats':>6} {'Cert#':<28}"
        )
        print("\n" + "=" * 110)
        print("📊  BENCHMARK RESULTS")
        print("=" * 110)
        print(hdr)
        print("-" * 110)

        for mid, r in results.items():
            m = r.get("metrics", {})
            if r["status"] == "success":
                print(
                    f"  {mid:<36} {'✅ OK':<9} {r['elapsed']:>6.1f}s "
                    f"{m.get('chemical_count', '—'):>6} "
                    f"{m.get('mechanical_count', '—'):>6} "
                    f"{m.get('unique_chem_heats', '—'):>6} "
                    f"{m.get('certificate_number', 'N/A'):<28}"
                )
            else:
                err_short = (r["error"] or "")[:40]
                print(
                    f"  {mid:<36} {'❌ FAIL':<9} {r['elapsed']:>6.1f}s "
                    f"{'—':>6} {'—':>6} {'—':>6} {err_short:<28}"
                )

        print("=" * 110)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark GitHub Models for MTC extraction (best → last)"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(
            "/workspaces/mtc-extraction-benchmark/data/raw/diler/diler-07-07-2025-rerun-41-44.pdf"
        ),
        help="Input PDF (used only when --use-cached-ocr is not set)",
    )
    parser.add_argument(
        "--use-cached-ocr",
        action="store_true",
        help="Use pre-extracted OCR text from pipeline_output/text instead of re-running OCR",
    )
    parser.add_argument(
        "--ocr-text-dir",
        type=Path,
        default=Path("/workspaces/mtc-extraction-benchmark/data/processed/pipeline_output/text"),
        help="Directory containing cached OCR page .txt files",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("/workspaces/mtc-extraction-benchmark/schema/mtc_extraction_schema_v1.json"),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("/workspaces/mtc-extraction-benchmark/prompts/mtc_llm_extraction_prompt.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspaces/mtc-extraction-benchmark/data/processed/benchmark_output"),
        help="Directory to save per-model outputs and summary",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional ground-truth JSON for comparison metrics",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model IDs to test (default: all ranked models)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only test the top N models from the ranked list",
    )
    args = parser.parse_args()

    # ---- Validate environment ----
    if "GITHUB_TOKEN" not in os.environ:
        print("❌ GITHUB_TOKEN not set. Required for GitHub Models API.")
        return 1

    if not args.schema.exists():
        print(f"❌ Schema not found: {args.schema}")
        return 1
    if not args.prompt.exists():
        print(f"❌ Prompt not found: {args.prompt}")
        return 1

    # ---- Determine which models to run ----
    if args.models:
        model_ids = args.models
    elif args.top_n:
        model_ids = [m["id"] for m in RANKED_MODELS[: args.top_n]]
    else:
        model_ids = [m["id"] for m in RANKED_MODELS]

    print(f"🏁 Models to benchmark ({len(model_ids)}):")
    for i, mid in enumerate(model_ids, 1):
        print(f"   {i:>2}. {mid}")

    # ---- Get OCR text ----
    if args.use_cached_ocr:
        bench = LLMModelBenchmark(args.schema, args.prompt)
        page_texts = bench.load_ocr_pages(args.ocr_text_dir)
    else:
        if not args.pdf.exists():
            print(f"❌ PDF not found: {args.pdf}")
            return 1
        # Initialise benchmark (LLM client) first, then OCR
        bench = LLMModelBenchmark(args.schema, args.prompt)
        page_texts = bench.run_ocr_fresh(args.pdf)

    # ---- Load ground truth if provided ----
    ground_truth = None
    if args.ground_truth and args.ground_truth.exists():
        ground_truth = json.loads(args.ground_truth.read_text())
        print(f"📏 Ground truth loaded from {args.ground_truth}")

    # ---- Run benchmark ----
    bench.benchmark(
        page_texts=page_texts,
        models=model_ids,
        ground_truth=ground_truth,
        output_dir=args.output,
    )

    return 0


if __name__ == "__main__":
    exit(main())
