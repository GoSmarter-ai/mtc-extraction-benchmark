#!/usr/bin/env python3
"""
Schema-compliant extraction from PaddleOCR text outputs
Converts PaddleOCR text files to MTC JSON schema
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import re


class MTCSchemaExtractor:
    """Extract MTC data from PaddleOCR text and map to schema."""

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            schema_path: Path to MTC JSON schema for validation
        """
        # Load schema if provided
        self.schema = None
        if schema_path and Path(schema_path).exists():
            with open(schema_path, "r") as f:
                self.schema = json.load(f)

        print("✓ Schema extractor initialized")

    def extract_from_text(self, text_path: str) -> Dict[str, Any]:
        """
        Extract structured data from a PaddleOCR text file.

        Args:
            text_path: Path to the PaddleOCR text output file

        Returns:
            Extracted data as a dictionary
        """
        print(f"\n📄 Processing: {text_path}")

        # Read text file
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract data
        extracted_data = {
            "document": self._extract_document_info(text),
            "traceability": self._extract_traceability(text),
            "product": self._extract_product_info(text),
            "chemical_composition": self._extract_chemical_composition(text),
            "mechanical_properties": self._extract_mechanical_properties(text),
            "approval": self._extract_approval(text),
        }

        return extracted_data

    def extract_from_multi_page(self, text_files: List[str]) -> Dict[str, Any]:
        """
        Extract from multiple pages and combine results.

        Args:
            text_files: List of paths to PaddleOCR text files

        Returns:
            Combined extracted data
        """
        print(f"\n🔗 Combining data from {len(text_files)} pages")

        # Combine all text
        combined_text = ""
        for text_file in sorted(text_files):
            with open(text_file, "r", encoding="utf-8") as f:
                combined_text += f.read() + "\n\n"

        # Extract from combined text
        return self.extract_from_text_content(combined_text)

    def extract_from_text_content(self, text: str) -> Dict[str, Any]:
        """Extract from text content string."""
        extracted_data = {
            "document": self._extract_document_info(text),
            "traceability": self._extract_traceability(text),
            "product": self._extract_product_info(text),
            "chemical_composition": self._extract_chemical_composition(text),
            "mechanical_properties": self._extract_mechanical_properties(text),
            "approval": self._extract_approval(text),
        }
        return extracted_data

    def _extract_document_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract document-level information."""

        # Extract certificate number
        cert_patterns = [
            r"CERTIFICATE\s+NUMBER[:\s]+([A-Z0-9/-]+)",
            r"Certificate\s+Number[:\s]+([A-Z0-9/-]+)",
        ]
        certificate_number = self._find_first_match(text, cert_patterns)

        # Extract issuing date
        date_patterns = [
            r"ISSUING\s+DATE[:\s]+(\d{2}\.\d{2}\s+\d{4})",
            r"Issuing\s+Date[:\s]+(\d{2}\.\d{2}\s+\d{4})",
        ]
        issuing_date = None
        date_match = self._find_first_match(text, date_patterns)
        if date_match:
            # Convert DD.MM YYYY to YYYY-MM-DD
            try:
                dt = datetime.strptime(date_match, "%d.%m %Y")
                issuing_date = dt.strftime("%Y-%m-%d")
            except:
                issuing_date = date_match

        # Extract standard
        standard_patterns = [
            r"EN\s+10204\s+[\d\.]+",
            r"TEST\s+CERTIFICATE\s+\(([^)]+)\)",
        ]
        standard = self._find_first_match(text, standard_patterns)

        # Extract customer
        customer_patterns = [
            r"CUSTOMER[:\s]+([A-Z][A-Z\s]+(?:LIMITED|LTD|INC|LLC)?[^\n]*?)(?:\n|$)",
        ]
        customer = self._find_first_match(text, customer_patterns)
        if customer:
            customer = customer.strip()

        # Extract order number
        order_patterns = [
            r"ORDER\s+NO[:\s]+([A-Z0-9-]+)",
            r"Order\s+No[:\s]+([A-Z0-9-]+)",
        ]
        order_number = self._find_first_match(text, order_patterns)

        return {
            "certificate_number": certificate_number,
            "issuing_date": issuing_date,
            "standard": standard,
            "customer": customer,
            "order_number": order_number,
        }

    def _extract_traceability(self, text: str) -> Dict[str, Optional[str]]:
        """Extract traceability information."""

        # Extract heat/lot number
        heat_patterns = [
            r"Heat\s+Number[:\s]+([A-Z0-9-]+)",
            r"ORDER\s+NO:\s+([0-9-]+)\s+LOT[:\s-]+(\d+)",
            r"LOT[:\s-]+([A-Z0-9-]+)",
        ]

        heat_number = None
        for pattern in heat_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.lastindex == 1:
                    heat_number = match.group(1)
                elif match.lastindex == 2:
                    heat_number = f"{match.group(1)}-LOT-{match.group(2)}"
                break

        # Extract consignment number
        consignment_patterns = [
            r"CONSIGNMENT\s+NO[:\s]+([A-Z0-9/-]+)",
            r"Consignment\s+No[:\s]+([A-Z0-9/-]+)",
        ]
        consignment_number = self._find_first_match(text, consignment_patterns)

        # Extract vessel name
        vessel_patterns = [
            r"VESSEL\s+NAME[:\s]+([A-Z\s]+)",
            r"Vessel\s+Name[:\s]+([A-Z\s]+)",
        ]
        vessel_name = self._find_first_match(text, vessel_patterns)
        if vessel_name:
            vessel_name = vessel_name.strip()

        return {
            "heat_number": heat_number,
            "consignment_number": consignment_number,
            "vessel_name": vessel_name,
        }

    def _extract_product_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract product information."""

        # Extract size
        size_patterns = [
            r"SIZE[:\s]+([0-9]+MM[X×xXx][0-9]+M\.?)",
            r"(\d+MM[X×xXx]\d+M\.?)",
        ]
        size = self._find_first_match(text, size_patterns)

        # Extract quality/grade
        quality_patterns = [
            r"QUALITY[:\s]+([A-Z0-9:]+\s+[A-Z0-9\s]+)",
            r"BS4449:\d+\s+GR\s+[AB]\d+\s+[AB]",
        ]
        quality = self._find_first_match(text, quality_patterns)
        if quality:
            quality = quality.strip()

        # Extract production process
        process_patterns = [
            r"PRODUCTION\s+PROSES?[:\s]+([A-Z]+)",
            r"Process[:\s]+([A-Z]+)",
        ]
        production_process = self._find_first_match(text, process_patterns)

        return {
            "size": size,
            "quality": quality,
            "production_process": production_process,
        }

    def _extract_chemical_composition(self, text: str) -> List[Dict[str, Any]]:
        """Extract chemical composition data."""
        composition = []

        # Elements to search for
        elements = {
            "C": ["carbon", "C"],
            "Si": ["Si", "Silicon"],
            "Mn": ["Mn", "Manganese"],
            "P": ["P", "Phosphorus"],
            "S": ["S", "Sulphur", "Sulfur"],
            "Ni": ["Ni", "Nickel"],
            "Cr": ["Cr", "Chromium"],
            "Mo": ["Mo", "Molybdenum"],
            "Cu": ["Cu", "Copper"],
            "V": ["V", "Vanadium"],
            "N": ["N", "Nitrogen"],
            "B": ["B", "Boron"],
        }

        for symbol, names in elements.items():
            for name in names:
                # Try multiple patterns
                patterns = [
                    rf"{name}\s+(?:\(confidence: [0-9\.]+\)\s*)?([0-9]+\.[0-9]+)",
                    rf"{name}[:\s]+([0-9]+\.[0-9]+)",
                ]

                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1))
                            composition.append(
                                {"element": symbol, "actual": value, "unit": "%"}
                            )
                            break
                        except ValueError:
                            pass
                if any(c["element"] == symbol for c in composition):
                    break  # Found this element, move to next

        return composition

    def _extract_mechanical_properties(self, text: str) -> List[Dict[str, Any]]:
        """Extract mechanical properties."""
        properties = []

        # Property patterns with variations
        patterns = {
            "yield_strength": [
                r"Yield\s+(?:Point|Strength)[:\s]+([0-9]+)",
                r"Yleld\s+Point[:\s]+([0-9]+)",  # OCR typo
            ],
            "tensile_strength": [
                r"Tensile\s+Strength[:\s]+([0-9]+)",
            ],
            "elongation": [
                r"Elongation[:\s]+([0-9]+)",
                r"Percentage[:\s]+([0-9]+)",
            ],
            "carbon_equivalent": [
                r"(?:carbon|C)\s+equivalent[:\s]+([0-9]+\.[0-9]+)",
                r"equvalent[:\s]+([0-9]+\.[0-9]+)",  # OCR typo
            ],
        }

        for prop_name, prop_patterns in patterns.items():
            for pattern in prop_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value_str = match.group(1)
                        value = float(value_str) if "." in value_str else int(value_str)

                        # Determine unit
                        unit = None
                        if "strength" in prop_name.lower():
                            unit = "MPa"
                        elif "elongation" in prop_name.lower():
                            unit = "%"

                        properties.append(
                            {"property": prop_name, "value": value, "unit": unit}
                        )
                        break
                    except ValueError:
                        pass

        return properties

    def _extract_approval(self, text: str) -> Dict[str, Optional[Any]]:
        """Extract approval information."""

        # Check for CARES approval
        cares_approved = "CARES APPROVED" in text.upper()

        return {
            "certificate_of_approval_number": None,
            "form_number": None,
            "cares_approved": cares_approved if cares_approved else None,
        }

    def _find_first_match(self, text: str, patterns: List[str]) -> Optional[str]:
        """Try multiple patterns and return first match."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def save_output(self, data: Dict[str, Any], output_path: str):
        """Save extracted data to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved to: {output_path}")


def main():
    """Example usage."""
    # Setup paths
    project_root = Path(__file__).parent
    schema_path = project_root / "schema" / "mtc_extraction_schema_v1.json"
    paddle_ocr_dir = project_root / "data" / "processed" / "paddle_ocr"
    output_dir = project_root / "data" / "processed" / "schema_output"

    print("=" * 80)
    print("🚀 MTC Schema Extraction from PaddleOCR Outputs")
    print("=" * 80)

    # Initialize extractor
    extractor = MTCSchemaExtractor(schema_path=str(schema_path))

    # Find all PaddleOCR text files
    text_files = sorted(paddle_ocr_dir.glob("*_text.txt"))

    if not text_files:
        print(f"❌ No text files found in {paddle_ocr_dir}")
        return

    print(f"\n🔍 Found {len(text_files)} PaddleOCR text file(s)")
    for f in text_files:
        print(f"   - {f.name}")

    # Extract from combined pages
    data = extractor.extract_from_multi_page([str(f) for f in text_files])

    # Save output
    output_filename = "diler-07-07-2025-mtc-extracted.json"
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

    print(f"\n🧪 CHEMICAL COMPOSITION: ({len(data['chemical_composition'])} elements)")
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


if __name__ == "__main__":
    main()
