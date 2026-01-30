#!/usr/bin/env python3
"""
Docling-based extraction for Mill Test Certificates (MTC)
Converts PDF documents to structured JSON following the MTC schema.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

from docling.document_converter import DocumentConverter


class MTCDoclingExtractor:
    """Extract MTC data using Docling document understanding."""

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            schema_path: Path to MTC JSON schema for validation
        """
        # Initialize Docling converter with default settings
        self.converter = DocumentConverter()

        # Load schema if provided
        self.schema = None
        if schema_path and Path(schema_path).exists():
            with open(schema_path, "r") as f:
                self.schema = json.load(f)

        print("✓ Docling extractor initialized")

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract structured data from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted data as a dictionary
        """
        print(f"\n📄 Processing: {pdf_path}")

        # Convert document
        result = self.converter.convert(pdf_path)

        # Get the document object
        doc = result.document

        # Extract data
        extracted_data = {
            "document": self._extract_document_info(doc),
            "traceability": self._extract_traceability(doc),
            "product": self._extract_product_info(doc),
            "chemical_composition": self._extract_chemical_composition(doc),
            "mechanical_properties": self._extract_mechanical_properties(doc),
            "approval": self._extract_approval(doc),
        }

        return extracted_data

    def _extract_document_info(self, doc) -> Dict[str, Optional[str]]:
        """Extract document-level information."""
        text = doc.export_to_text()

        # Extract certificate number
        cert_match = re.search(
            r"CERTIFICATE\s+NUMBER[:\s]+([A-Z0-9/-]+)", text, re.IGNORECASE
        )
        certificate_number = cert_match.group(1) if cert_match else None

        # Extract issuing date
        date_match = re.search(
            r"ISSUING\s+DATE[:\s]+(\d{2}\.\d{2}\s+\d{4})", text, re.IGNORECASE
        )
        issuing_date = None
        if date_match:
            # Convert DD.MM YYYY to YYYY-MM-DD
            date_str = date_match.group(1)
            try:
                dt = datetime.strptime(date_str, "%d.%m %Y")
                issuing_date = dt.strftime("%Y-%m-%d")
            except:
                issuing_date = date_str

        # Extract standard
        standard_match = re.search(r"EN\s+10204\s+[\d\.]+", text)
        standard = standard_match.group(0) if standard_match else None

        # Extract customer
        customer_match = re.search(
            r"CUSTOMER[:\s]+([A-Z\s]+(?:LIMITED|LTD|INC|LLC)?[^\n]*)",
            text,
            re.IGNORECASE,
        )
        customer = customer_match.group(1).strip() if customer_match else None

        # Extract order number
        order_match = re.search(r"ORDER\s+NO[:\s]+([A-Z0-9-]+)", text, re.IGNORECASE)
        order_number = order_match.group(1) if order_match else None

        return {
            "certificate_number": certificate_number,
            "issuing_date": issuing_date,
            "standard": standard,
            "customer": customer,
            "order_number": order_number,
        }

    def _extract_traceability(self, doc) -> Dict[str, Optional[str]]:
        """Extract traceability information."""
        text = doc.export_to_text()

        # Extract heat/lot number
        heat_patterns = [
            r"Heat\s+Number[:\s]+([A-Z0-9-]+)",
            r"LOT[:\s-]+([A-Z0-9-]+)",
            r"ORDER\s+NO:\s+([0-9-]+)\s+LOT[:\s-]+(\d+)",
        ]

        heat_number = None
        for pattern in heat_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                heat_number = (
                    match.group(1)
                    if match.lastindex == 1
                    else f"{match.group(1)}-LOT{match.group(2)}"
                )
                break

        # Extract consignment number
        consignment_match = re.search(
            r"CONSIGNMENT\s+NO[:\s]+([A-Z0-9/-]+)", text, re.IGNORECASE
        )
        consignment_number = consignment_match.group(1) if consignment_match else None

        # Extract vessel name
        vessel_match = re.search(r"VESSEL\s+NAME[:\s]+([A-Z\s]+)", text, re.IGNORECASE)
        vessel_name = vessel_match.group(1).strip() if vessel_match else None

        return {
            "heat_number": heat_number,
            "consignment_number": consignment_number,
            "vessel_name": vessel_name,
        }

    def _extract_product_info(self, doc) -> Dict[str, Optional[str]]:
        """Extract product information."""
        text = doc.export_to_text()

        # Extract size
        size_match = re.search(
            r"SIZE[:\s]+([0-9]+MM[X×][0-9]+M\.?)", text, re.IGNORECASE
        )
        size = size_match.group(1) if size_match else None

        # Extract quality/grade
        quality_match = re.search(
            r"QUALITY[:\s]+([A-Z0-9:]+\s+[A-Z0-9\s]+)", text, re.IGNORECASE
        )
        quality = quality_match.group(1).strip() if quality_match else None

        # Extract production process
        process_match = re.search(
            r"PRODUCTION\s+PROSES?[:\s]+([A-Z]+)", text, re.IGNORECASE
        )
        production_process = process_match.group(1) if process_match else None

        return {
            "size": size,
            "quality": quality,
            "production_process": production_process,
        }

    def _extract_chemical_composition(self, doc) -> List[Dict[str, Any]]:
        """Extract chemical composition data."""
        text = doc.export_to_text()
        composition = []

        # Common elements in steel
        elements = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V", "N", "B"]

        # Search for element values in text
        # This is simplified - Docling's table extraction would be better
        for element in elements:
            # Look for patterns like "C 0.25" or "carbon 0.25%"
            pattern = rf"{element}\s+([0-9]+\.[0-9]+)"
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                value_str = match.group(1)
                try:
                    value = float(value_str)
                    composition.append(
                        {"element": element, "actual": value, "unit": "%"}
                    )
                except ValueError:
                    pass

        return composition

    def _extract_mechanical_properties(self, doc) -> List[Dict[str, Any]]:
        """Extract mechanical properties."""
        text = doc.export_to_text()
        properties = []

        # Property patterns
        patterns = {
            "yield_strength": r"Yield\s+(?:Point|Strength)[:\s]+([0-9]+)",
            "tensile_strength": r"Tensile\s+Strength[:\s]+([0-9]+)",
            "elongation": r"(?:Elongation|Percentage)[:\s]+([0-9]+)",
            "carbon_equivalent": r"(?:C|Carbon)\s+equivalent[:\s]+([0-9]+\.[0-9]+)",
        }

        for prop_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                try:
                    value = float(value_str)

                    # Determine unit
                    unit = None
                    if "strength" in prop_name.lower():
                        unit = "MPa"
                    elif "elongation" in prop_name.lower():
                        unit = "%"
                    elif "equivalent" in prop_name.lower():
                        unit = None

                    properties.append(
                        {"property": prop_name, "value": value, "unit": unit}
                    )
                except ValueError:
                    pass

        return properties

    def _extract_approval(self, doc) -> Dict[str, Optional[Any]]:
        """Extract approval information."""
        text = doc.export_to_text()

        # Check for CARES approval
        cares_approved = "CARES APPROVED" in text.upper()

        return {
            "certificate_of_approval_number": None,
            "form_number": None,
            "cares_approved": cares_approved if cares_approved else None,
        }

    def save_output(self, data: Dict[str, Any], output_path: str):
        """Save extracted data to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved to: {output_path}")


def main():
    """Example usage."""
    import sys

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    schema_path = project_root / "schema" / "mtc_extraction_schema_v1.json"
    raw_data_dir = project_root / "data" / "raw" / "diler"
    output_dir = project_root / "data" / "processed" / "docling_output"

    # Initialize extractor
    extractor = MTCDoclingExtractor(schema_path=str(schema_path))

    # Find PDF files
    pdf_files = list(raw_data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {raw_data_dir}")
        return

    print(f"\n🔍 Found {len(pdf_files)} PDF file(s)")

    # Process each PDF
    for pdf_file in pdf_files:
        try:
            # Extract data
            data = extractor.extract_from_pdf(str(pdf_file))

            # Save output
            output_filename = pdf_file.stem + "_extracted.json"
            output_path = output_dir / output_filename
            extractor.save_output(data, str(output_path))

            # Print summary
            print(f"\n📊 Extraction Summary:")
            print(f"   Certificate: {data['document'].get('certificate_number')}")
            print(f"   Heat Number: {data['traceability'].get('heat_number')}")
            print(f"   Chemical Elements: {len(data['chemical_composition'])}")
            print(f"   Mechanical Props: {len(data['mechanical_properties'])}")

        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()
