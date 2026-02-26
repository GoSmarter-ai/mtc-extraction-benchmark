import json
import os
from pathlib import Path

from openai import OpenAI

"""
LLM Extraction using GitHub Models (Free Open Source)

This script extracts structured data from Mill Test Certificates using LLM.
It processes pages in chunks if needed to stay within token limits.

Available open-source models:
- Meta-Llama-3-8B-Instruct: Fast, efficient (8B parameters)
- Meta-Llama-3-70B-Instruct: More capable (70B parameters)
- Meta-Llama-3.1-405B-Instruct: Most powerful (405B parameters)

Note: Some models have input token limits (~8000 tokens). For large documents,
we chunk the processing by pages and merge results.

Requires GITHUB_TOKEN environment variable (automatically available in Codespaces)
"""

# ---------- Configuration ----------
USE_CHUNKED_PROCESSING = True  # Set to True if hitting token limits
MAX_TOKENS_OUTPUT = 16384

# ---------- Paths ----------
TEXT_PATH = Path(
    "/workspaces/mtc-extraction-benchmark/data/processed/paddle_ocr/diler-07-07-2025-rerun-41-44_all_pages.txt"
)
PAGE_FILES = [
    Path(
        "/workspaces/mtc-extraction-benchmark/data/processed/paddle_ocr/diler-07-07-2025-rerun-41-44_page1_text.txt"
    ),
    Path(
        "/workspaces/mtc-extraction-benchmark/data/processed/paddle_ocr/diler-07-07-2025-rerun-41-44_page2_text.txt"
    ),
    Path(
        "/workspaces/mtc-extraction-benchmark/data/processed/paddle_ocr/diler-07-07-2025-rerun-41-44_page3_text.txt"
    ),
    Path(
        "/workspaces/mtc-extraction-benchmark/data/processed/paddle_ocr/diler-07-07-2025-rerun-41-44_page4_text.txt"
    ),
]
SCHEMA_PATH = Path("/workspaces/mtc-extraction-benchmark/schema/mtc_extraction_schema_v1.json")
PROMPT_PATH = Path("/workspaces/mtc-extraction-benchmark/prompts/mtc_llm_extraction_prompt.txt")
OUTPUT_PATH = Path(
    "/workspaces/mtc-extraction-benchmark/data/processed/schema_output/llm_certificate_full.json"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------- Helper Functions ----------
def extract_from_text(
    ocr_text: str, schema: dict, system_prompt: str, client, page_info: str = ""
) -> dict:
    """Extract structured data from OCR text using LLM."""

    user_prompt = f"""
SCHEMA:
{json.dumps(schema, indent=2)}

OCR TEXT {page_info}:
\"\"\"
{ocr_text}
\"\"\"
"""

    response = client.chat.completions.create(
        model="Meta-Llama-3.1-405B-Instruct",
        temperature=0,
        max_tokens=MAX_TOKENS_OUTPUT,
        messages=[
            {"role": "system", "content": system_prompt},
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


def merge_extractions(results: list) -> dict:
    """Merge multiple extraction results into one complete extraction."""
    if not results:
        return {}

    if len(results) == 1:
        return results[0]

    # Start with first result as base
    merged = results[0].copy()

    # Merge chemical composition (deduplicate by heat_number)
    seen_heats = {item["heat_number"] for item in merged.get("chemical_composition", [])}
    for result in results[1:]:
        for chem in result.get("chemical_composition", []):
            if chem["heat_number"] not in seen_heats:
                merged.setdefault("chemical_composition", []).append(chem)
                seen_heats.add(chem["heat_number"])

    # Merge mechanical properties (append all, they should be unique by sample)
    for result in results[1:]:
        for mech in result.get("mechanical_properties", []):
            merged.setdefault("mechanical_properties", []).append(mech)

    # Take approval from any result that has it
    for result in results:
        if result.get("approval", {}).get("certificate_of_approval_number"):
            merged["approval"] = result["approval"]
            break

    return merged


# ---------- Main Execution ----------
def main():
    print("📋 Loading schema...")
    schema = json.loads(SCHEMA_PATH.read_text())

    print("💬 Loading prompt...")
    system_prompt = PROMPT_PATH.read_text()

    print("🔑 Initializing LLM client...")
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ["GITHUB_TOKEN"],
    )

    # ---------- Process pages ----------
    if USE_CHUNKED_PROCESSING:
        print("📄 Processing pages individually (chunked mode)...")
        results = []

        for i, page_file in enumerate(PAGE_FILES, 1):
            print(f"\n   Page {i}/4...")
            ocr_text = page_file.read_text(errors="ignore")
            print(f"   - Text length: {len(ocr_text)} characters")

            try:
                result = extract_from_text(
                    ocr_text, schema, system_prompt, client, page_info=f"(Page {i}/4)"
                )
                results.append(result)

                # Show what was extracted
                chem_count = len(result.get("chemical_composition", []))
                mech_count = len(result.get("mechanical_properties", []))
                print(f"   - Extracted: {chem_count} heat numbers, {mech_count} test samples")

            except Exception as e:
                print(f"   ⚠️  Error on page {i}: {e}")
                # Save debug info
                debug_path = OUTPUT_PATH.parent / f"llm_failed_page{i}.txt"
                print(f"   💾 Error details saved to: {debug_path}")
                continue

        print(f"\n🔀 Merging {len(results)} page results...")
        parsed = merge_extractions(results)

    else:
        print("📄 Loading OCR text from all pages...")
        ocr_text = TEXT_PATH.read_text(errors="ignore")
        print(f"   Text length: {len(ocr_text)} characters")

        print("🤖 Calling LLM...")
        parsed = extract_from_text(ocr_text, schema, system_prompt, client, page_info="(ALL PAGES)")

    # ---------- Validation & Summary ----------
    print("\n📊 Extraction Summary:")
    print(f"   Certificate: {parsed.get('document', {}).get('certificate_number', 'N/A')}")
    print(f"   Heat numbers in chem comp: {len(parsed.get('chemical_composition', []))}")
    print(f"   Mechanical test samples: {len(parsed.get('mechanical_properties', []))}")
    print(
        f"   Approval number: {parsed.get('approval', {}).get('certificate_of_approval_number', 'N/A')}"
    )

    # ---------- Save output ----------
    with open(OUTPUT_PATH, "w") as f:
        json.dump(parsed, f, indent=2)

    print("\n✅ LLM extraction completed successfully!")
    print(f"💾 Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
