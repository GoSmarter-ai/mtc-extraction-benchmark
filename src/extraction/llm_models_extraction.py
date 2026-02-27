"""
Multi-Model LLM Extraction Benchmark
======================================
Benchmarks all available GitHub Models for MTC structured data extraction.
Re-uses existing OCR text from the pipeline to isolate LLM performance.

Improvements over baseline:
  - Schema validation with retry on malformed JSON
  - Token usage & cost tracking per model
  - OCR confidence parsing and low-confidence flagging
  - Two-pass extraction with cross-page consolidation
  - Self-consistency via multi-sample voting
  - Smart merge with full deduplication (chemical + mechanical)
  - Cross-model ensemble extraction
  - Rich field-level precision / recall / F1 metrics with numeric tolerance

Models ranked best-to-last for structured extraction:
  1. gpt-4o                        – best reasoning & JSON compliance
  2. Meta-Llama-3.1-405B-Instruct  – very strong open-source
  3. Meta-Llama-3.1-8B-Instruct    – small but capable
  4. gpt-4o-mini                   – cost-optimized OpenAI

Usage:
    # Run all models (best to last):
    python src/extraction/llm_models_extraction.py

    # Run specific models:
    python src/extraction/llm_models_extraction.py --models gpt-4o Meta-Llama-3.1-405B-Instruct

    # Use pre-extracted OCR text (skip PDF→OCR step):
    python src/extraction/llm_models_extraction.py --use-cached-ocr

    # Compare against ground truth:
    python src/extraction/llm_models_extraction.py --ground-truth data/ground_truth/cert.json

    # Enable two-pass consolidation:
    python src/extraction/llm_models_extraction.py --two-pass

    # Run self-consistency voting (3 samples per page):
    python src/extraction/llm_models_extraction.py --consistency-samples 3

    # Ensemble top-3 models:
    python src/extraction/llm_models_extraction.py --ensemble --top-n 3
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from openai import OpenAI

# ---------------------------------------------------------------------------
# Repository root (works in Codespaces, CI runners, local checkouts)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

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
        "id": "Meta-Llama-3.1-8B-Instruct",
        "label": "Llama 3.1 8B",
        "provider": "Meta",
        "tier": "small",
    },
    {
        "id": "gpt-4o-mini",
        "label": "GPT-4o Mini",
        "provider": "OpenAI",
        "tier": "small",
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
    # OCR confidence parsing
    # ------------------------------------------------------------------
    @staticmethod
    def parse_ocr_with_confidence(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Separate clean text from confidence annotations.

        Returns:
            (clean_text, low_confidence_items) where low_confidence_items
            is a list of dicts with keys 'text' and 'confidence' for any
            line with confidence < 0.90.
        """
        lines = raw_text.split("\n")
        clean_lines: List[str] = []
        low_confidence: List[Dict[str, Any]] = []

        for line in lines:
            match = re.match(r"^(.+?)\s*\(confidence:\s*([\d.]+)\)\s*$", line)
            if match:
                text, conf = match.group(1), float(match.group(2))
                clean_lines.append(text)
                if conf < 0.90:
                    low_confidence.append({"text": text, "confidence": round(conf, 4)})
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines), low_confidence

    # ------------------------------------------------------------------
    # Single-model extraction (with token tracking)
    # ------------------------------------------------------------------
    def extract_with_model(
        self,
        model_id: str,
        ocr_text: str,
        page_info: str = "",
        temperature: float = 0,
    ) -> Tuple[dict, Dict[str, int]]:
        """
        Call a single model and return (parsed_json, token_usage).

        Token usage dict has keys: prompt_tokens, completion_tokens, total_tokens.
        """
        user_prompt = (
            f"SCHEMA:\n{json.dumps(self.schema, indent=2)}\n\n"
            f'OCR TEXT{page_info}:\n"""\n{ocr_text}\n"""'
        )

        response = self.client.chat.completions.create(
            model=model_id,
            temperature=temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Track token usage
        usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if hasattr(response, "usage") and response.usage is not None:
            usage["prompt_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
            usage["completion_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
            usage["total_tokens"] = getattr(response.usage, "total_tokens", 0) or 0

        # Strip markdown fences
        if "```json" in raw:
            raw = raw[raw.find("```json") + 7 :]
            raw = raw[: raw.find("```")].strip()
        elif "```" in raw:
            raw = raw[raw.find("```") + 3 :]
            raw = raw[: raw.find("```")].strip()

        return json.loads(raw), usage

    # ------------------------------------------------------------------
    # Schema-validated extraction with retry
    # ------------------------------------------------------------------
    def extract_with_validation(
        self,
        model_id: str,
        ocr_text: str,
        page_info: str = "",
        max_retries: int = 3,
        temperature: float = 0,
    ) -> Tuple[dict, Dict[str, int]]:
        """
        Extract with JSON Schema validation and retry on failure.

        On each failed attempt the error message is logged so the user
        can see what went wrong.  Retries use the same prompt.
        """
        last_error: Optional[Exception] = None
        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for attempt in range(1, max_retries + 1):
            try:
                result, usage = self.extract_with_model(
                    model_id,
                    ocr_text,
                    page_info=page_info,
                    temperature=temperature,
                )
                # Accumulate tokens across retries
                for k in cumulative_usage:
                    cumulative_usage[k] += usage.get(k, 0)

                # Validate against the JSON schema
                jsonschema.validate(instance=result, schema=self.schema)
                return result, cumulative_usage

            except json.JSONDecodeError as exc:
                last_error = exc
                print(f"      ⚠️  Attempt {attempt}/{max_retries} – invalid JSON: {exc}")
            except jsonschema.ValidationError as exc:
                last_error = exc
                short_msg = str(exc.message)[:120]
                print(f"      ⚠️  Attempt {attempt}/{max_retries} – schema violation: {short_msg}")

        # All retries exhausted — raise the last error
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Self-consistency via multi-sample voting
    # ------------------------------------------------------------------
    def extract_with_consensus(
        self,
        model_id: str,
        ocr_text: str,
        page_info: str = "",
        n_samples: int = 3,
    ) -> Tuple[dict, Dict[str, int]]:
        """
        Run *n_samples* extractions (temperature > 0) and merge via
        majority voting on each field value.

        Returns (consensus_result, cumulative_token_usage).
        """
        samples: List[dict] = []
        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for i in range(n_samples):
            try:
                result, usage = self.extract_with_validation(
                    model_id,
                    ocr_text,
                    page_info=page_info,
                    max_retries=2,
                    temperature=0.3,
                )
                for k in cumulative_usage:
                    cumulative_usage[k] += usage.get(k, 0)
                samples.append(result)
            except Exception as exc:
                print(f"      ⚠️  Consensus sample {i + 1} failed: {exc}")

        if not samples:
            raise RuntimeError("All consensus samples failed")

        consensus = self._majority_vote(samples)
        return consensus, cumulative_usage

    @staticmethod
    def _majority_vote(samples: List[dict]) -> dict:
        """
        Field-level majority voting across multiple extraction samples.

        For scalar fields the most common value wins.
        For list fields (chemical_composition, mechanical_properties),
        items are matched by heat_number and each sub-field is voted on.
        """
        if len(samples) == 1:
            return samples[0]

        def _vote_scalar(values: List[Any]) -> Any:
            """Return the most common non-None value, or None."""
            filtered = [v for v in values if v is not None]
            if not filtered:
                return None
            counter = Counter(
                json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for v in filtered
            )
            winner_raw = counter.most_common(1)[0][0]
            # Deserialise if it was a dict/list
            if isinstance(winner_raw, str):
                try:
                    return json.loads(winner_raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            return winner_raw

        # Vote on top-level scalar / object sections
        voted: dict = {}
        all_keys: set = set()
        for s in samples:
            all_keys.update(s.keys())

        list_sections = {"chemical_composition", "mechanical_properties"}
        for key in all_keys:
            if key in list_sections:
                continue  # handled below
            values = [s.get(key) for s in samples]
            # For dicts, vote per sub-key
            if all(isinstance(v, dict) for v in values if v is not None):
                sub_keys: set = set()
                for v in values:
                    if isinstance(v, dict):
                        sub_keys.update(v.keys())
                sub_voted: dict = {}
                for sk in sub_keys:
                    sub_vals = [v.get(sk) for v in values if isinstance(v, dict)]
                    sub_voted[sk] = _vote_scalar(sub_vals)
                voted[key] = sub_voted
            else:
                voted[key] = _vote_scalar(values)

        # Vote on list sections (chemical_composition, mechanical_properties)
        for section in list_sections:
            all_items: Dict[str, List[dict]] = {}
            for s in samples:
                for item in s.get(section, []):
                    hn = item.get("heat_number", "")
                    all_items.setdefault(hn, []).append(item)

            voted_items: list = []
            for _hn, items in all_items.items():
                item_keys: set = set()
                for it in items:
                    item_keys.update(it.keys())
                voted_item: dict = {}
                for ik in item_keys:
                    sub_values = [it.get(ik) for it in items]
                    if all(isinstance(v, dict) for v in sub_values if v is not None):
                        # e.g. "elements" sub-dict
                        elem_keys: set = set()
                        for v in sub_values:
                            if isinstance(v, dict):
                                elem_keys.update(v.keys())
                        voted_sub: dict = {}
                        for ek in elem_keys:
                            ek_vals = [v.get(ek) for v in sub_values if isinstance(v, dict)]
                            voted_sub[ek] = _vote_scalar(ek_vals)
                        voted_item[ik] = voted_sub
                    else:
                        voted_item[ik] = _vote_scalar(sub_values)
                voted_items.append(voted_item)
            voted[section] = voted_items

        return voted

    # ------------------------------------------------------------------
    # Merge per-page results (original – kept for backward compat)
    # ------------------------------------------------------------------
    @staticmethod
    def merge_extractions(results: List[dict]) -> dict:
        """Legacy merge: deduplicates chemical by heat_number, appends mechanical."""
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
    # Smart merge v2 – full deduplication for ALL sections
    # ------------------------------------------------------------------
    @staticmethod
    def merge_extractions_v2(results: List[dict]) -> dict:
        """
        Improved merge with deduplication for *all* repeated sections.

        Chemical composition: deduplicated by heat_number.
        Mechanical properties: deduplicated by composite key
            (heat_number, test_sample, weight_kg_per_m, yield_point_mpa).
        Document / traceability / product / approval: first non-empty wins,
            with field-level merge for partial data.
        """
        if not results:
            return {}
        if len(results) == 1:
            return results[0]

        merged: dict = {}

        # --- Merge scalar / object sections (first non-empty, field-level) ---
        object_sections = ["document", "traceability", "product", "approval"]
        for section in object_sections:
            combined: dict = {}
            for r in results:
                obj = r.get(section)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        # Keep first non-null / non-empty value
                        if k not in combined or combined[k] is None or combined[k] == "":
                            combined[k] = v
            if combined:
                merged[section] = combined

        # --- Chemical composition: deduplicate by heat_number ---
        seen_chem_heats: set = set()
        unique_chem: List[dict] = []
        for r in results:
            for chem in r.get("chemical_composition", []):
                hn = chem.get("heat_number", "")
                if hn not in seen_chem_heats:
                    seen_chem_heats.add(hn)
                    unique_chem.append(chem)
        merged["chemical_composition"] = unique_chem

        # --- Mechanical properties: deduplicate by composite key ---
        seen_mech: set = set()
        unique_mech: List[dict] = []
        for r in results:
            for mech in r.get("mechanical_properties", []):
                key = (
                    mech.get("heat_number", ""),
                    mech.get("test_sample"),
                    mech.get("weight_kg_per_m"),
                    mech.get("yield_point_mpa"),
                )
                if key not in seen_mech:
                    seen_mech.add(key)
                    unique_mech.append(mech)
        merged["mechanical_properties"] = unique_mech

        return merged

    # ------------------------------------------------------------------
    # Two-pass extraction with cross-page consolidation
    # ------------------------------------------------------------------
    def extract_two_pass(
        self,
        model_id: str,
        page_texts: List[str],
        use_validation: bool = True,
    ) -> Tuple[dict, Dict[str, int]]:
        """
        Pass 1: per-page extraction.
        Pass 2: send merged draft + all text for LLM consolidation.

        The consolidation pass allows the model to:
        - Resolve contradictions across pages
        - Fill in missing fields using cross-page context
        - Deduplicate entries the merge missed
        """
        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # --- Pass 1: per-page ---
        page_results: List[dict] = []
        for i, text in enumerate(page_texts, 1):
            if not text.strip():
                continue
            clean_text, _low_conf = self.parse_ocr_with_confidence(text)
            page_info = f" (Page {i}/{len(page_texts)})"

            if use_validation:
                result, usage = self.extract_with_validation(
                    model_id, clean_text, page_info=page_info
                )
            else:
                result, usage = self.extract_with_model(model_id, clean_text, page_info=page_info)

            for k in cumulative_usage:
                cumulative_usage[k] += usage.get(k, 0)

            chem = len(result.get("chemical_composition", []))
            mech = len(result.get("mechanical_properties", []))
            print(f"      Pass 1 – Page {i}: {chem} heats, {mech} samples")
            page_results.append(result)

        draft = self.merge_extractions_v2(page_results)

        # --- Pass 2: consolidation ---
        full_ocr = "\n\n---PAGE BREAK---\n\n".join(page_texts)
        consolidation_prompt = (
            f"You are reviewing a structured extraction from a "
            f"{len(page_texts)}-page Mill Test Certificate.\n\n"
            f"DRAFT EXTRACTION (may contain errors or duplicates):\n"
            f"```json\n{json.dumps(draft, indent=2)}\n```\n\n"
            f'FULL OCR TEXT:\n"""\n{full_ocr}\n"""\n\n'
            f"SCHEMA:\n{json.dumps(self.schema, indent=2)}\n\n"
            f"Instructions:\n"
            f"1. Review the draft against the full OCR text\n"
            f"2. Correct any extraction errors\n"
            f"3. Remove duplicate entries\n"
            f"4. Fill in any fields that were missed\n"
            f"5. Return the corrected JSON (schema-compliant)"
        )

        try:
            response = self.client.chat.completions.create(
                model=model_id,
                temperature=0,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": consolidation_prompt},
                ],
            )

            raw = response.choices[0].message.content.strip()

            if hasattr(response, "usage") and response.usage is not None:
                cumulative_usage["prompt_tokens"] += (
                    getattr(response.usage, "prompt_tokens", 0) or 0
                )
                cumulative_usage["completion_tokens"] += (
                    getattr(response.usage, "completion_tokens", 0) or 0
                )
                cumulative_usage["total_tokens"] += getattr(response.usage, "total_tokens", 0) or 0

            # Strip markdown fences
            if "```json" in raw:
                raw = raw[raw.find("```json") + 7 :]
                raw = raw[: raw.find("```")].strip()
            elif "```" in raw:
                raw = raw[raw.find("```") + 3 :]
                raw = raw[: raw.find("```")].strip()

            consolidated = json.loads(raw)
            print("      Pass 2 – Consolidation ✓")
            return consolidated, cumulative_usage

        except Exception as exc:
            print(f"      ⚠️  Pass 2 consolidation failed ({exc}), using Pass 1 merge")
            return draft, cumulative_usage

    # ------------------------------------------------------------------
    # Run one model end-to-end
    # ------------------------------------------------------------------
    def run_model(
        self,
        model_id: str,
        page_texts: List[str],
        two_pass: bool = False,
        consistency_samples: int = 0,
    ) -> Dict:
        """
        Run extraction for a single model across all pages.

        Args:
            model_id: The model identifier string.
            page_texts: List of OCR text strings (one per page).
            two_pass: If True, use two-pass extraction with consolidation.
            consistency_samples: If > 1, use self-consistency voting with
                this many samples per page.

        Returns:
            dict with keys: result, elapsed, status, error, token_usage
        """
        start = time.time()
        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
            # --- Two-pass mode ---
            if two_pass:
                merged, usage = self.extract_two_pass(model_id, page_texts)
                for k in cumulative_usage:
                    cumulative_usage[k] += usage.get(k, 0)

            # --- Self-consistency mode ---
            elif consistency_samples > 1:
                page_results: List[dict] = []
                for i, text in enumerate(page_texts, 1):
                    if not text.strip():
                        continue
                    clean_text, _low_conf = self.parse_ocr_with_confidence(text)
                    result, usage = self.extract_with_consensus(
                        model_id,
                        clean_text,
                        page_info=f" (Page {i}/{len(page_texts)})",
                        n_samples=consistency_samples,
                    )
                    for k in cumulative_usage:
                        cumulative_usage[k] += usage.get(k, 0)
                    chem = len(result.get("chemical_composition", []))
                    mech = len(result.get("mechanical_properties", []))
                    print(f"      Page {i} (consensus): {chem} heats, {mech} samples")
                    page_results.append(result)
                merged = self.merge_extractions_v2(page_results)

            # --- Standard mode (with validation) ---
            else:
                page_results = []
                for i, text in enumerate(page_texts, 1):
                    if not text.strip():
                        continue
                    clean_text, low_conf = self.parse_ocr_with_confidence(text)
                    if low_conf:
                        print(f"      ⚠️  Page {i}: {len(low_conf)} low-confidence OCR tokens")

                    result, usage = self.extract_with_validation(
                        model_id,
                        clean_text,
                        page_info=f" (Page {i}/{len(page_texts)})",
                    )
                    for k in cumulative_usage:
                        cumulative_usage[k] += usage.get(k, 0)
                    chem = len(result.get("chemical_composition", []))
                    mech = len(result.get("mechanical_properties", []))
                    print(f"      Page {i}: {chem} heats, {mech} samples")
                    page_results.append(result)
                merged = self.merge_extractions_v2(page_results)

            elapsed = time.time() - start
            return {
                "result": merged,
                "elapsed": round(elapsed, 2),
                "status": "success",
                "error": None,
                "token_usage": cumulative_usage,
            }

        except Exception as exc:
            elapsed = time.time() - start
            return {
                "result": None,
                "elapsed": round(elapsed, 2),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "token_usage": cumulative_usage,
            }

    # ------------------------------------------------------------------
    # Ensemble extraction across multiple models
    # ------------------------------------------------------------------
    def ensemble_extract(
        self,
        page_texts: List[str],
        model_ids: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> Dict:
        """
        Run top-k models and ensemble their outputs via field-level
        majority voting.

        Returns dict with keys: result, elapsed, status, model_results,
            token_usage
        """
        if model_ids is None:
            model_ids = [m["id"] for m in RANKED_MODELS[:top_k]]
        else:
            model_ids = model_ids[:top_k]

        start = time.time()
        all_results: Dict[str, dict] = {}
        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for mid in model_ids:
            print(f"\n   🔄 Ensemble member: {mid}")
            run = self.run_model(mid, page_texts)
            if run["status"] == "success" and run["result"]:
                all_results[mid] = run["result"]
                for k in cumulative_usage:
                    cumulative_usage[k] += run.get("token_usage", {}).get(k, 0)
            else:
                print(f"      ❌ {mid} failed: {run['error']}")

        if not all_results:
            elapsed = time.time() - start
            return {
                "result": None,
                "elapsed": round(elapsed, 2),
                "status": "error",
                "error": "All ensemble models failed",
                "model_results": {},
                "token_usage": cumulative_usage,
            }

        # Cross-model majority voting
        consensus = self._majority_vote(list(all_results.values()))
        elapsed = time.time() - start

        return {
            "result": consensus,
            "elapsed": round(elapsed, 2),
            "status": "success",
            "error": None,
            "model_results": {mid: "success" for mid in all_results},
            "token_usage": cumulative_usage,
        }

    # ------------------------------------------------------------------
    # Rich field-level metrics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_field_f1(
        extracted: dict,
        ground_truth: dict,
        numeric_tolerance: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Compute per-field precision, recall, and F1 with numeric tolerance.

        Evaluates:
          - Document-level scalar fields
          - Chemical composition matched by heat_number
          - Per-element numeric accuracy with tolerance
          - Mechanical properties matched by heat_number + test_sample
          - Per-property numeric accuracy
        """
        scores: Dict[str, Any] = {}

        # ---------- Document fields ----------
        gt_doc = ground_truth.get("document", {})
        ex_doc = extracted.get("document", {})
        doc_fields = [
            "certificate_number",
            "issuing_date",
            "standard",
            "customer",
            "order_number",
        ]
        doc_correct = 0
        doc_total = 0
        doc_detail: Dict[str, bool] = {}
        for field in doc_fields:
            gt_val = gt_doc.get(field)
            ex_val = ex_doc.get(field)
            if gt_val is not None:
                doc_total += 1
                match = str(gt_val).strip() == str(ex_val).strip() if ex_val is not None else False
                if match:
                    doc_correct += 1
                doc_detail[field] = match
        scores["document_accuracy"] = round(doc_correct / max(doc_total, 1), 4)
        scores["document_fields"] = doc_detail

        # ---------- Chemical composition ----------
        gt_chem = {c["heat_number"]: c for c in ground_truth.get("chemical_composition", [])}
        ex_chem = {c["heat_number"]: c for c in extracted.get("chemical_composition", [])}

        chem_tp = len(set(ex_chem) & set(gt_chem))
        chem_fp = len(set(ex_chem) - set(gt_chem))
        chem_fn = len(set(gt_chem) - set(ex_chem))

        scores["heat_precision"] = round(chem_tp / max(chem_tp + chem_fp, 1), 4)
        scores["heat_recall"] = round(chem_tp / max(chem_tp + chem_fn, 1), 4)
        if scores["heat_precision"] + scores["heat_recall"] > 0:
            scores["heat_f1"] = round(
                2
                * scores["heat_precision"]
                * scores["heat_recall"]
                / (scores["heat_precision"] + scores["heat_recall"]),
                4,
            )
        else:
            scores["heat_f1"] = 0.0

        # Per-element accuracy (within matched heats)
        element_correct = 0
        element_total = 0
        element_detail: Dict[str, Dict[str, bool]] = {}
        for heat in set(ex_chem) & set(gt_chem):
            gt_elements = gt_chem[heat].get("elements", {})
            ex_elements = ex_chem[heat].get("elements", {})
            heat_detail: Dict[str, bool] = {}
            for elem, gt_val in gt_elements.items():
                if gt_val is None:
                    continue
                element_total += 1
                ex_val = ex_elements.get(elem)
                if ex_val is not None:
                    try:
                        match = abs(float(ex_val) - float(gt_val)) <= numeric_tolerance
                    except (ValueError, TypeError):
                        match = str(ex_val).strip() == str(gt_val).strip()
                else:
                    match = False
                if match:
                    element_correct += 1
                heat_detail[elem] = match
            element_detail[heat] = heat_detail

        scores["element_accuracy"] = round(element_correct / max(element_total, 1), 4)
        scores["element_detail"] = element_detail

        # ---------- Mechanical properties ----------
        def _mech_key(m: dict) -> Tuple:
            return (m.get("heat_number", ""), m.get("test_sample"))

        gt_mech = {_mech_key(m): m for m in ground_truth.get("mechanical_properties", [])}
        ex_mech = {_mech_key(m): m for m in extracted.get("mechanical_properties", [])}

        mech_tp = len(set(ex_mech) & set(gt_mech))
        mech_fp = len(set(ex_mech) - set(gt_mech))
        mech_fn = len(set(gt_mech) - set(ex_mech))

        scores["mech_precision"] = round(mech_tp / max(mech_tp + mech_fp, 1), 4)
        scores["mech_recall"] = round(mech_tp / max(mech_tp + mech_fn, 1), 4)
        if scores["mech_precision"] + scores["mech_recall"] > 0:
            scores["mech_f1"] = round(
                2
                * scores["mech_precision"]
                * scores["mech_recall"]
                / (scores["mech_precision"] + scores["mech_recall"]),
                4,
            )
        else:
            scores["mech_f1"] = 0.0

        # Per-property numeric accuracy (within matched samples)
        num_props = [
            "weight_kg_per_m",
            "cross_sectional_area_mm2",
            "yield_point_mpa",
            "tensile_strength_mpa",
            "rm_re_ratio",
            "percentage_elongation",
            "agt_percent",
        ]
        prop_correct = 0
        prop_total = 0
        for mk in set(ex_mech) & set(gt_mech):
            gt_m = gt_mech[mk]
            ex_m = ex_mech[mk]
            for prop in num_props:
                gt_val = gt_m.get(prop)
                if gt_val is None:
                    continue
                prop_total += 1
                ex_val = ex_m.get(prop)
                if ex_val is not None:
                    try:
                        if abs(float(ex_val) - float(gt_val)) <= numeric_tolerance:
                            prop_correct += 1
                    except (ValueError, TypeError):
                        if str(ex_val).strip() == str(gt_val).strip():
                            prop_correct += 1
        scores["mech_property_accuracy"] = round(prop_correct / max(prop_total, 1), 4)

        # ---------- Overall F1 ----------
        all_f1 = [
            scores.get("heat_f1", 0),
            scores.get("mech_f1", 0),
            scores.get("document_accuracy", 0),
            scores.get("element_accuracy", 0),
            scores.get("mech_property_accuracy", 0),
        ]
        scores["overall_f1"] = round(sum(all_f1) / len(all_f1), 4)

        return scores

    # ------------------------------------------------------------------
    # Quick metrics (backward compatible)
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
        two_pass: bool = False,
        consistency_samples: int = 0,
        ensemble: bool = False,
        ensemble_top_k: int = 3,
    ) -> Dict[str, Dict]:
        """
        Run benchmark across multiple models.

        Args:
            page_texts:            List of OCR text strings (one per page)
            models:                List of model IDs to test (default: all)
            ground_truth:          Optional ground-truth dict for comparison
            output_dir:            Directory to save per-model JSON outputs
            two_pass:              Enable two-pass consolidation
            consistency_samples:   Number of samples for self-consistency
                                   (0 = off)
            ensemble:              Enable cross-model ensemble
            ensemble_top_k:        How many top models to use for ensemble

        Returns:
            Dict mapping model_id → { result, elapsed, status, metrics, … }
        """
        if models is None:
            model_ids = [m["id"] for m in RANKED_MODELS]
        else:
            model_ids = models

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict] = {}
        total = len(model_ids)

        # --- Per-model benchmark ---
        for idx, model_id in enumerate(model_ids, 1):
            print(f"\n{'=' * 70}")
            mode_label = ""
            if two_pass:
                mode_label = " [two-pass]"
            elif consistency_samples > 1:
                mode_label = f" [consensus×{consistency_samples}]"
            print(f"  [{idx}/{total}] 🧪 {model_id}{mode_label}")
            print(f"{'=' * 70}")

            run = self.run_model(
                model_id,
                page_texts,
                two_pass=two_pass,
                consistency_samples=consistency_samples,
            )

            if run["status"] == "success" and run["result"]:
                run["metrics"] = self.compute_metrics(run["result"], ground_truth)

                # Rich field-level metrics when ground truth is available
                if ground_truth:
                    run["field_metrics"] = self.compute_field_f1(run["result"], ground_truth)
                    f1 = run["field_metrics"].get("overall_f1", "N/A")
                    print(f"   📊 Overall F1: {f1}")

                # Save individual model output
                if output_dir:
                    safe_name = model_id.replace("/", "_").replace(".", "_")
                    out_file = output_dir / f"{safe_name}_extracted.json"
                    out_file.write_text(json.dumps(run["result"], indent=2))
                    print(f"   💾 Saved → {out_file.name}")

                # Print token usage
                tok = run.get("token_usage", {})
                if tok.get("total_tokens"):
                    print(
                        f"   🪙 Tokens: {tok['prompt_tokens']:,} prompt + "
                        f"{tok['completion_tokens']:,} completion = "
                        f"{tok['total_tokens']:,} total"
                    )
            else:
                run["metrics"] = {}
                run["field_metrics"] = {}
                print(f"   ❌ {run['error']}")

            results[model_id] = run

        # --- Ensemble (optional) ---
        if ensemble and len(model_ids) >= 2:
            print(f"\n{'=' * 70}")
            print(f"  🏆 ENSEMBLE (top {ensemble_top_k} models)")
            print(f"{'=' * 70}")

            ens = self.ensemble_extract(page_texts, model_ids=model_ids, top_k=ensemble_top_k)
            if ens["status"] == "success" and ens["result"]:
                ens["metrics"] = self.compute_metrics(ens["result"], ground_truth)
                if ground_truth:
                    ens["field_metrics"] = self.compute_field_f1(ens["result"], ground_truth)
                    print(f"   📊 Ensemble Overall F1: {ens['field_metrics'].get('overall_f1')}")
                if output_dir:
                    out_file = output_dir / "ensemble_extracted.json"
                    out_file.write_text(json.dumps(ens["result"], indent=2))
                    print(f"   💾 Saved → {out_file.name}")
            else:
                ens["metrics"] = {}
                ens["field_metrics"] = {}

            results["__ensemble__"] = ens

        # ----- Print comparison table -----
        self._print_table(results, has_ground_truth=ground_truth is not None)

        # ----- Save summary -----
        if output_dir:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "models_tested": total,
                "config": {
                    "two_pass": two_pass,
                    "consistency_samples": consistency_samples,
                    "ensemble": ensemble,
                    "ensemble_top_k": ensemble_top_k,
                },
                "results": {
                    mid: {
                        "status": r["status"],
                        "elapsed_seconds": r["elapsed"],
                        "error": r["error"],
                        "metrics": r["metrics"],
                        "field_metrics": r.get("field_metrics", {}),
                        "token_usage": r.get("token_usage", {}),
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
    def _print_table(results: Dict[str, Dict], has_ground_truth: bool = False) -> None:
        if has_ground_truth:
            hdr = (
                f"\n{'Model':<38} {'Status':<9} {'Time':>7} "
                f"{'Chem#':>6} {'Mech#':>6} {'Heats':>6} "
                f"{'F1':>6} {'Tokens':>8} {'Cert#':<28}"
            )
            sep_len = 125
        else:
            hdr = (
                f"\n{'Model':<38} {'Status':<9} {'Time':>7} "
                f"{'Chem#':>6} {'Mech#':>6} {'Heats':>6} "
                f"{'Tokens':>8} {'Cert#':<28}"
            )
            sep_len = 118

        print("\n" + "=" * sep_len)
        print("📊  BENCHMARK RESULTS")
        print("=" * sep_len)
        print(hdr)
        print("-" * sep_len)

        for mid, r in results.items():
            display_name = "🏆 ENSEMBLE" if mid == "__ensemble__" else mid
            m = r.get("metrics", {})
            tok = r.get("token_usage", {})
            tok_str = f"{tok.get('total_tokens', 0):,}" if tok.get("total_tokens") else "—"

            if r["status"] == "success":
                f1_str = ""
                if has_ground_truth:
                    fm = r.get("field_metrics", {})
                    f1_val = fm.get("overall_f1", "—")
                    f1_str = f"{f1_val:>6} " if isinstance(f1_val, float) else f"{'—':>6} "

                print(
                    f"  {display_name:<36} {'✅ OK':<9} "
                    f"{r['elapsed']:>6.1f}s "
                    f"{m.get('chemical_count', '—'):>6} "
                    f"{m.get('mechanical_count', '—'):>6} "
                    f"{m.get('unique_chem_heats', '—'):>6} "
                    f"{f1_str}"
                    f"{tok_str:>8} "
                    f"{m.get('certificate_number', 'N/A'):<28}"
                )
            else:
                err_short = (r["error"] or "")[:40]
                f1_col = f"{'—':>6} " if has_ground_truth else ""
                print(
                    f"  {display_name:<36} {'❌ FAIL':<9} "
                    f"{r['elapsed']:>6.1f}s "
                    f"{'—':>6} {'—':>6} {'—':>6} "
                    f"{f1_col}"
                    f"{tok_str:>8} "
                    f"{err_short:<28}"
                )

        print("=" * sep_len)


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
        help=("Use pre-extracted OCR text from pipeline_output/text instead of re-running OCR"),
    )
    parser.add_argument(
        "--ocr-text-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "pipeline_output" / "text",
        help="Directory containing cached OCR page .txt files",
    )
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "benchmark_output",
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
    # --- New flags ---
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Enable two-pass extraction with cross-page consolidation",
    )
    parser.add_argument(
        "--consistency-samples",
        type=int,
        default=0,
        help=("Number of samples for self-consistency voting (0=disabled, recommended: 3)"),
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help=("Enable cross-model ensemble (majority vote across top-k models)"),
    )
    parser.add_argument(
        "--ensemble-top-k",
        type=int,
        default=3,
        help="Number of top models to include in ensemble (default: 3)",
    )
    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=0.001,
        help=("Numeric tolerance for field-level metric comparisons (default: 0.001)"),
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

    # Print active modes
    modes: list = []
    if args.two_pass:
        modes.append("two-pass consolidation")
    if args.consistency_samples > 1:
        modes.append(f"self-consistency (×{args.consistency_samples})")
    if args.ensemble:
        modes.append(f"ensemble (top-{args.ensemble_top_k})")
    if modes:
        print(f"   🔧 Active modes: {', '.join(modes)}")

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
        two_pass=args.two_pass,
        consistency_samples=args.consistency_samples,
        ensemble=args.ensemble,
        ensemble_top_k=args.ensemble_top_k,
    )

    return 0


if __name__ == "__main__":
    exit(main())
