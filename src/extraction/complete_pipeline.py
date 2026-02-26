"""
Complete MTC Extraction Pipeline
=================================
End-to-end pipeline that:
1. Converts PDF to images
2. Extracts text using PaddleOCR
3. Extracts structured data using LLM (GitHub Models)
4. Outputs validated JSON conforming to the MTC schema

Usage:
    python complete_pipeline.py --pdf <path/to/pdf> --output <output_dir>

    Or run with defaults (processes sample PDF)
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from openai import OpenAI
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image

# Repository root (works in Codespaces, CI runners, local checkouts)
REPO_ROOT = Path(__file__).resolve().parents[2]


class MTCPipeline:
    """Complete pipeline for MTC extraction from PDF to structured JSON."""

    def __init__(
        self,
        schema_path: Path,
        prompt_path: Path,
        dpi: int = 200,
        max_width: int = 2000,
        llm_model: str = "Meta-Llama-3.1-405B-Instruct",
        max_tokens: int = 16384,
    ):
        """
        Initialize the pipeline.

        Args:
            schema_path: Path to JSON schema file
            prompt_path: Path to system prompt file
            dpi: DPI for PDF to image conversion
            max_width: Maximum image width (for memory optimization)
            llm_model: LLM model to use for extraction
            max_tokens: Maximum tokens for LLM output
        """
        self.dpi = dpi
        self.max_width = max_width
        self.llm_model = llm_model
        self.max_tokens = max_tokens

        # Load schema and prompt
        print("📋 Loading schema and prompt...")
        self.schema = json.loads(schema_path.read_text())
        self.system_prompt = prompt_path.read_text()

        # Initialize PaddleOCR
        print("🔧 Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_textline_orientation=True, lang="en")

        # Initialize LLM client
        print("🔑 Initializing LLM client...")
        self.llm_client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ["GITHUB_TOKEN"],
        )

        print("✅ Pipeline initialized!\n")

    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to list of PIL images."""
        print(f"📄 Converting PDF to images (DPI={self.dpi})...")
        try:
            images = convert_from_path(str(pdf_path), dpi=self.dpi)
            print(f"   ✓ Converted to {len(images)} page(s)")
            return images
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF: {e}")

    def extract_text_from_image(
        self, pil_image: Image.Image, page_num: int
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """
        Extract text from a single image using PaddleOCR.

        Returns:
            tuple: (extracted_text, original_image_array, annotated_image_array)
        """
        print(f"   Processing page {page_num}...")

        # Resize if too large
        if pil_image.width > self.max_width:
            ratio = self.max_width / pil_image.width
            new_height = int(pil_image.height * ratio)
            pil_image = pil_image.resize((self.max_width, new_height), Image.LANCZOS)
            print(f"     Resized to {self.max_width}x{new_height}")

        # Convert to OpenCV format
        img_array = np.array(pil_image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Run OCR
        print("     Running OCR...")
        results = list(self.ocr.predict(img_array))

        # Extract text and create annotated image
        img_with_boxes = img_array.copy()
        text_blocks = []

        if results and len(results) > 0:
            result = results[0]
            rec_texts = result.get("rec_texts", [])
            rec_scores = result.get("rec_scores", [])
            rec_polys = result.get("rec_polys", [])

            print(f"     Found {len(rec_texts)} text blocks")

            for idx in range(len(rec_texts)):
                try:
                    text = rec_texts[idx]
                    box = rec_polys[idx]
                    confidence = rec_scores[idx]

                    # Draw bounding box
                    box = np.array(box, dtype=np.int32)
                    cv2.polylines(img_with_boxes, [box], True, (0, 255, 0), 2)

                    # Add text label
                    display_text = text[:30] if len(text) > 30 else text
                    cv2.putText(
                        img_with_boxes,
                        display_text,
                        (box[0][0], max(box[0][1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )

                    text_blocks.append(f"{text} (confidence: {confidence:.4f})")

                except (IndexError, TypeError, KeyError) as e:
                    print(f"     Warning: Error at index {idx}: {e}")
                    continue

        # Combine text blocks
        extracted_text = "\n".join(text_blocks) if text_blocks else ""

        # Clean up
        del pil_image, results
        gc.collect()

        return extracted_text, img_array, img_with_boxes

    def extract_with_llm(self, ocr_text: str, page_info: str = "") -> dict:
        """Extract structured data from OCR text using LLM."""
        print(f"🤖 Calling LLM for extraction{page_info}...")

        user_prompt = f"""
SCHEMA:
{json.dumps(self.schema, indent=2)}

OCR TEXT{page_info}:
\"\"\"
{ocr_text}
\"\"\"
"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_output = response.choices[0].message.content.strip()

        # Extract JSON from markdown if needed
        if "```json" in raw_output:
            json_start = raw_output.find("```json") + 7
            json_end = raw_output.find("```", json_start)
            raw_output = raw_output[json_start:json_end].strip()
        elif "```" in raw_output:
            json_start = raw_output.find("```") + 3
            json_end = raw_output.find("```", json_start)
            raw_output = raw_output[json_start:json_end].strip()

        return json.loads(raw_output)

    def merge_extractions(self, results: List[dict]) -> dict:
        """Merge multiple page extraction results."""
        if not results:
            return {}

        if len(results) == 1:
            return results[0]

        print(f"🔀 Merging {len(results)} page results...")

        # Start with first result as base
        merged = results[0].copy()

        # Merge chemical composition (deduplicate by heat_number)
        seen_heats = {item["heat_number"] for item in merged.get("chemical_composition", [])}
        for result in results[1:]:
            for chem in result.get("chemical_composition", []):
                if chem["heat_number"] not in seen_heats:
                    merged.setdefault("chemical_composition", []).append(chem)
                    seen_heats.add(chem["heat_number"])

        # Merge mechanical properties
        for result in results[1:]:
            for mech in result.get("mechanical_properties", []):
                merged.setdefault("mechanical_properties", []).append(mech)

        # Take approval from any result that has it
        for result in results:
            if result.get("approval", {}).get("certificate_of_approval_number"):
                merged["approval"] = result["approval"]
                break

        return merged

    def process_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
        save_intermediates: bool = True,
        chunked_processing: bool = True,
    ) -> dict:
        """
        Process a PDF through the complete pipeline.

        Args:
            pdf_path: Path to input PDF
            output_dir: Directory to save outputs
            save_intermediates: Save OCR text and annotated images
            chunked_processing: Process pages individually (recommended for large docs)

        Returns:
            dict: Extracted structured data
        """
        print("\n" + "=" * 70)
        print(f"🚀 PROCESSING: {pdf_path.name}")
        print("=" * 70 + "\n")

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_intermediates:
            (output_dir / "images").mkdir(exist_ok=True)
            (output_dir / "text").mkdir(exist_ok=True)

        base_name = pdf_path.stem

        # Step 1: Convert PDF to images
        images = self.pdf_to_images(pdf_path)

        # Step 2: Extract text from each page
        print(f"\n📝 Extracting text from {len(images)} pages...")
        page_texts = []

        for page_num, pil_image in enumerate(images, 1):
            text, img_original, img_annotated = self.extract_text_from_image(pil_image, page_num)
            page_texts.append(text)

            # Save intermediates if requested
            if save_intermediates:
                page_suffix = f"_page{page_num}"

                # Save images
                cv2.imwrite(
                    str(output_dir / "images" / f"{base_name}{page_suffix}_original.jpg"),
                    img_original,
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                cv2.imwrite(
                    str(output_dir / "images" / f"{base_name}{page_suffix}_annotated.jpg"),
                    img_annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )

                # Save text
                with open(output_dir / "text" / f"{base_name}{page_suffix}.txt", "w") as f:
                    f.write(f"Page {page_num}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(text if text else "No text detected.\n")

            # Clean up
            del img_original, img_annotated
            gc.collect()

        # Clean up images
        del images
        gc.collect()

        # Step 3: LLM extraction
        print("\n🧠 Extracting structured data with LLM...")

        if chunked_processing:
            # Process each page separately and merge
            results = []
            for i, page_text in enumerate(page_texts, 1):
                if not page_text.strip():
                    print(f"   ⚠️  Skipping empty page {i}")
                    continue

                try:
                    result = self.extract_with_llm(
                        page_text, page_info=f" (Page {i}/{len(page_texts)})"
                    )
                    results.append(result)

                    # Show what was extracted
                    chem_count = len(result.get("chemical_composition", []))
                    mech_count = len(result.get("mechanical_properties", []))
                    print(f"   ✓ Page {i}: {chem_count} heat numbers, {mech_count} test samples")

                except Exception as e:
                    print(f"   ⚠️  Error on page {i}: {e}")
                    continue

            extracted_data = self.merge_extractions(results)
        else:
            # Process all pages together
            all_text = "\n\n".join(
                f"=== PAGE {i} ===\n{text}" for i, text in enumerate(page_texts, 1)
            )
            extracted_data = self.extract_with_llm(all_text, page_info=" (ALL PAGES)")

        # Step 4: Save output
        output_file = output_dir / f"{base_name}_extracted.json"
        with open(output_file, "w") as f:
            json.dump(extracted_data, f, indent=2)

        # Print summary
        print("\n" + "=" * 70)
        print("📊 EXTRACTION SUMMARY")
        print("=" * 70)
        print(
            f"Certificate Number: {extracted_data.get('document', {}).get('certificate_number', 'N/A')}"
        )
        print(f"Heat Numbers: {len(extracted_data.get('chemical_composition', []))}")
        print(f"Mechanical Test Samples: {len(extracted_data.get('mechanical_properties', []))}")
        print(
            f"Approval Number: {extracted_data.get('approval', {}).get('certificate_of_approval_number', 'N/A')}"
        )
        print("=" * 70)
        print("\n✅ Pipeline completed successfully!")
        print(f"💾 Output saved to: {output_file}\n")

        return extracted_data


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Complete MTC extraction pipeline from PDF to structured JSON"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "diler" / "diler-07-07-2025-rerun-41-44.pdf",
        help="Path to input PDF file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "pipeline_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "schema" / "mtc_extraction_schema_v1.json",
        help="Path to JSON schema file",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=REPO_ROOT / "prompts" / "mtc_llm_extraction_prompt.txt",
        help="Path to system prompt file",
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Don't save intermediate OCR outputs and images",
    )
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Process all pages together (not recommended for large docs)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion (default: 200)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Meta-Llama-3.1-405B-Instruct",
        help="LLM model to use (default: Meta-Llama-3.1-405B-Instruct)",
    )

    args = parser.parse_args()

    # Verify input file exists
    if not args.pdf.exists():
        print(f"❌ Error: PDF file not found: {args.pdf}")
        return 1

    # Verify schema and prompt exist
    if not args.schema.exists():
        print(f"❌ Error: Schema file not found: {args.schema}")
        return 1

    if not args.prompt.exists():
        print(f"❌ Error: Prompt file not found: {args.prompt}")
        return 1

    # Verify GITHUB_TOKEN is available
    if "GITHUB_TOKEN" not in os.environ:
        print("❌ Error: GITHUB_TOKEN environment variable not set")
        print("   In Codespaces, this should be automatically available.")
        return 1

    # Initialize pipeline
    try:
        pipeline = MTCPipeline(
            schema_path=args.schema,
            prompt_path=args.prompt,
            dpi=args.dpi,
            llm_model=args.model,
        )
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return 1

    # Process PDF
    try:
        pipeline.process_pdf(
            pdf_path=args.pdf,
            output_dir=args.output,
            save_intermediates=not args.no_intermediates,
            chunked_processing=not args.no_chunking,
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error processing PDF: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
