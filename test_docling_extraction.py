#!/usr/bin/env python3
"""
Test script for Docling extraction on a single MTC document.
"""

from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from extraction.docling_extraction import MTCDoclingExtractor


def main():
    # Setup paths
    project_root = Path(__file__).parent
    schema_path = project_root / "schema" / "mtc_extraction_schema_v1.json"
    pdf_path = (
        project_root / "data" / "raw" / "diler" / "diler-07-07-2025-rerun-41-44.pdf"
    )
    output_dir = project_root / "data" / "processed" / "docling_output"

    print("=" * 80)
    print("🧪 Testing Docling MTC Extraction")
    print("=" * 80)

    # Check if PDF exists
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return

    print(f"📄 Input PDF: {pdf_path.name}")
    print(f"📋 Schema: {schema_path.name}")
    print(f"📁 Output: {output_dir}")

    # Initialize extractor
    print("\n🔧 Initializing Docling extractor...")
    extractor = MTCDoclingExtractor(schema_path=str(schema_path))

    # Extract data
    print("\n🚀 Starting extraction...")
    try:
        data = extractor.extract_from_pdf(str(pdf_path))

        # Save output
        output_filename = pdf_path.stem + "_extracted.json"
        output_path = output_dir / output_filename
        extractor.save_output(data, str(output_path))

        # Print detailed results
        print("\n" + "=" * 80)
        print("📊 EXTRACTION RESULTS")
        print("=" * 80)

        print("\n📄 DOCUMENT INFO:")
        for key, value in data["document"].items():
            print(f"   {key:20s}: {value}")

        print("\n🔗 TRACEABILITY:")
        for key, value in data["traceability"].items():
            print(f"   {key:20s}: {value}")

        print("\n🏭 PRODUCT:")
        for key, value in data["product"].items():
            print(f"   {key:20s}: {value}")

        print(
            f"\n🧪 CHEMICAL COMPOSITION: ({len(data['chemical_composition'])} elements)"
        )
        for element in data["chemical_composition"]:
            print(f"   {element['element']:5s}: {element['actual']}%")

        print(
            f"\n⚙️  MECHANICAL PROPERTIES: ({len(data['mechanical_properties'])} properties)"
        )
        for prop in data["mechanical_properties"]:
            unit = f" {prop['unit']}" if prop["unit"] else ""
            print(f"   {prop['property']:20s}: {prop['value']}{unit}")

        print("\n✅ APPROVAL:")
        for key, value in data["approval"].items():
            print(f"   {key:30s}: {value}")

        print("\n" + "=" * 80)
        print(f"✅ SUCCESS! Output saved to:")
        print(f"   {output_path}")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
