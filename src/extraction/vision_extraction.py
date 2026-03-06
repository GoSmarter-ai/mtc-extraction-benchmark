"""
VisionExtractor
===============
Sends base64-encoded page images directly to a vision-capable LLM
(e.g. gpt-4o) for MTC structured data extraction — bypassing OCR entirely.

This is the "vision path" of the hybrid pipeline and is most useful when:
  - OCR confidence is low (e.g. stamps, rotated text, hand-written annotations)
  - Document layout is complex (tables with merged cells)
  - Fast bootstrapping without an OCR engine installed

Usage (Python)
--------------
    from src.extraction.vision_extraction import VisionExtractor

    extractor = VisionExtractor(
        schema_path=Path("schema/mtc_extraction_schema_v1.json"),
        prompt_path=Path("prompts/mtc_llm_extraction_prompt.txt"),
    )
    result = extractor.extract(pdf_path=Path("data/raw/diler/cert.pdf"))

Usage (CLI)
-----------
    python -m src.extraction.vision_extraction \\
        --pdf data/raw/diler/diler-07-07-2025-rerun-41-44.pdf \\
        --model gpt-4o \\
        --output data/processed/pipeline_output/vision_extracted.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
from openai import OpenAI

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# Default vision model — must support image content parts
DEFAULT_VISION_MODEL = "gpt-4o"

# Maximum pixels on the long edge before downscaling (reduces token cost)
_MAX_LONG_EDGE = 2048


class VisionExtractor:
    """
    Extract structured MTC data from PDF pages using a vision LLM.

    The extractor converts each PDF page to a JPEG, encodes it as base64,
    and sends it alongside the system prompt to the model.  Page results
    are merged using the same composite-key deduplication strategy as the
    text-based pipeline.
    """

    def __init__(
        self,
        schema_path: Path,
        prompt_path: Path,
        model: str = DEFAULT_VISION_MODEL,
        max_tokens: int = 16384,
        dpi: int = 200,
    ):
        self.schema = json.loads(schema_path.read_text())
        self.system_prompt = prompt_path.read_text()
        self.model = model
        self.max_tokens = max_tokens
        self.dpi = dpi
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ["GITHUB_TOKEN"],
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Full pipeline: PDF → images → per-page vision extraction → merged output.

        Returns the merged structured JSON dict.
        """
        images_b64 = self._pdf_to_base64_jpegs(pdf_path)
        page_results = []
        total_tokens = 0

        for page_num, img_b64 in enumerate(images_b64, 1):
            print(f"  🔍 Vision extracting page {page_num}/{len(images_b64)} …")
            result, tokens = self._extract_single_page(img_b64, page_num)
            if result:
                page_results.append(result)
            total_tokens += tokens

        print(f"  📊 Total tokens used: {total_tokens:,}")

        if not page_results:
            return {}

        merged = page_results[0]
        for page_result in page_results[1:]:
            merged = self._merge(merged, page_result)

        return merged

    def extract_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Return individual per-page extraction results (before merging).
        Useful for debugging and per-page quality inspection.
        """
        images_b64 = self._pdf_to_base64_jpegs(pdf_path)
        return [self._extract_single_page(img, i + 1)[0] for i, img in enumerate(images_b64)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pdf_to_base64_jpegs(self, pdf_path: Path) -> List[str]:
        """Convert each PDF page to a JPEG and base64-encode it."""
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise ImportError(
                "pdf2image is required for vision extraction. Install with: pip install pdf2image"
            ) from exc

        print(f"  📄 Converting {pdf_path.name} → images (DPI={self.dpi}) …")
        pil_images = convert_from_path(str(pdf_path), dpi=self.dpi)
        print(f"     ✓ {len(pil_images)} page(s)")

        results: List[str] = []
        for pil_img in pil_images:
            # Downscale if needed to keep tokens low
            w, h = pil_img.size
            long_edge = max(w, h)
            if long_edge > _MAX_LONG_EDGE:
                scale = _MAX_LONG_EDGE / long_edge
                pil_img = pil_img.resize((int(w * scale), int(h * scale)))

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=92)
            results.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        return results

    def _extract_single_page(
        self, img_b64: str, page_num: int
    ) -> tuple[Optional[Dict[str, Any]], int]:
        """
        Send one page image to the vision model and return (parsed_json, tokens).

        Retries up to 3 times on schema-validation failure.
        """
        user_content = [
            {
                "type": "text",
                "text": (
                    f"This is page {page_num} of an MTC certificate. "
                    "Extract ALL structured data visible on this page and return "
                    "it as JSON matching the schema exactly."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high",
                },
            },
        ]

        for attempt in range(1, 4):
            try:
                t0 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                elapsed = time.time() - t0
                tokens_used = response.usage.total_tokens if response.usage else 0

                raw = response.choices[0].message.content or ""
                parsed = self._parse_json(raw)

                if parsed is None:
                    print(f"     ⚠  Page {page_num} attempt {attempt}: JSON parse failed")
                    continue

                try:
                    jsonschema.validate(instance=parsed, schema=self.schema)
                    print(
                        f"     ✓ Page {page_num} extracted in {elapsed:.1f}s "
                        f"({tokens_used:,} tokens)"
                    )
                    return parsed, tokens_used
                except jsonschema.ValidationError as e:
                    print(f"     ⚠  Page {page_num} attempt {attempt}: schema error — {e.message}")

            except Exception as exc:  # noqa: BLE001
                print(f"     ❌ Page {page_num} attempt {attempt}: {exc}")

        return None, 0

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from model output (handles markdown fences)."""
        text = text.strip()
        # Strip markdown code fences
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        elif text.startswith("{"):
            # Find the matching closing brace
            depth = 0
            end = 0
            for i, ch in enumerate(text):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            text = text[: end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two per-page extractions into one, deduplicating on composite keys.

        - document / traceability / product / approval: overlay wins if not None
        - chemical_composition: deduplicated on heat_number
        - mechanical_properties: deduplicated on (heat_number, weight_kg_per_m, yield_point_mpa)
        """
        result = dict(base)

        # Scalar sections — overlay wins for non-null fields
        for section in ("document", "traceability", "product", "approval"):
            base_sec = base.get(section, {}) or {}
            overlay_sec = overlay.get(section, {}) or {}
            merged_sec = dict(base_sec)
            for k, v in overlay_sec.items():
                if v is not None:
                    merged_sec[k] = v
            result[section] = merged_sec

        # Chemical composition — dedupe on heat_number
        chem_map: Dict[str, Any] = {
            c["heat_number"]: c for c in (base.get("chemical_composition") or [])
        }
        for c in overlay.get("chemical_composition") or []:
            hn = c.get("heat_number")
            if hn and hn not in chem_map:
                chem_map[hn] = c
            elif hn:
                # Merge element-level (overlay wins per element)
                existing = dict(chem_map[hn])
                overlay_elems = c.get("elements", {}) or {}
                merged_elems = dict(existing.get("elements", {}) or {})
                merged_elems.update({k: v for k, v in overlay_elems.items() if v is not None})
                existing["elements"] = merged_elems
                chem_map[hn] = existing
        result["chemical_composition"] = list(chem_map.values())

        # Mechanical properties — dedupe on (heat_number, weight_kg_per_m, yield_point_mpa)
        def _mech_key(row: dict) -> tuple:
            return (
                row.get("heat_number"),
                row.get("weight_kg_per_m"),
                row.get("yield_point_mpa"),
            )

        existing_keys = {_mech_key(r) for r in (base.get("mechanical_properties") or [])}
        extra = [
            r
            for r in (overlay.get("mechanical_properties") or [])
            if _mech_key(r) not in existing_keys
        ]
        result["mechanical_properties"] = list(base.get("mechanical_properties") or []) + extra

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract MTC data from a PDF using vision LLM.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the source PDF.")
    parser.add_argument(
        "--model",
        default=DEFAULT_VISION_MODEL,
        help=f"Vision model to use (default: {DEFAULT_VISION_MODEL}).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "schema" / "mtc_extraction_schema_v1.json",
        help="Path to the JSON schema file.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=REPO_ROOT / "prompts" / "mtc_llm_extraction_prompt.txt",
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution for PDF→image conversion (default: 200).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save the extracted JSON (default: prints to stdout).",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"❌ PDF not found: {args.pdf}")
        return 1

    extractor = VisionExtractor(
        schema_path=args.schema,
        prompt_path=args.prompt,
        model=args.model,
        dpi=args.dpi,
    )

    print(f"🔬 Vision extraction: {args.pdf.name}  model={args.model}")
    result = extractor.extract(pdf_path=args.pdf)

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
