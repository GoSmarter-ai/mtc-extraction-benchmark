"""
Docling-based MTC extraction using the Docling library for document understanding.
This serves as a comparison to the rule-based PaddleOCR extraction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import re

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("⚠️  Docling not installed. Install with: pip install docling")
    raise


class MTCDoclingExtractor:
    """Extract MTC data using Docling's document understanding capabilities."""

    def __init__(self, schema_path: Optional[str] = None):
        """Initialize Docling extractor."""

        # Initialize document converter with default settings
        self.converter = DocumentConverter()

        # Load schema if provided
        self.schema = None
        if schema_path and Path(schema_path).exists():
            with open(schema_path, "r") as f:
                self.schema = json.load(f)

        print("✓ Docling extractor initialized")

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract MTC data from PDF using Docling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Structured MTC data matching schema
        """
        print(f"\n📄 Processing with Docling: {pdf_path}")

        # Convert PDF to Docling document
        result = self.converter.convert(pdf_path)
        doc = result.document

        # Extract full text
        full_text = doc.export_to_markdown()
        plain_text = doc.export_to_text()

        # Extract tables from document
        tables = []
        for table_ix, table in enumerate(doc.tables):
            table_data = self._parse_table(table)
            if table_data:
                tables.append({"index": table_ix, "data": table_data})

        print(f"   ✓ Extracted {len(tables)} tables")
        print(f"   ✓ Text length: {len(plain_text)} characters")

        # Extract structured data
        extracted_data = {
            "document": self._extract_document_info(plain_text),
            "traceability": self._extract_traceability(plain_text),
            "product": self._extract_product_info(plain_text),
            "chemical_composition": self._extract_chemical_composition(
                tables, plain_text
            ),
            "mechanical_properties": self._extract_mechanical_properties(
                tables, plain_text
            ),
            "approval": self._extract_approval(plain_text),
            "extraction_metadata": {
                "method": "docling",
                "num_pages": len(doc.pages),
                "num_tables": len(tables),
                "extraction_timestamp": datetime.now().isoformat(),
            },
        }

        return extracted_data

    def _parse_table(self, table) -> List[List[str]]:
        """Parse Docling table into 2D array."""
        try:
            # Get table data from Docling's table structure
            table_data = []

            # Try to export table as markdown and parse it
            if hasattr(table, "export_to_dataframe"):
                df = table.export_to_dataframe()
                # Convert dataframe to list of lists
                table_data = [df.columns.tolist()] + df.values.tolist()
            elif hasattr(table, "data"):
                # Direct access to table data
                for row in table.data:
                    row_data = [str(cell) for cell in row]
                    table_data.append(row_data)

            return table_data
        except Exception as e:
            print(f"   ⚠️  Error parsing table: {e}")
            return []

    def _extract_document_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract document metadata using regex patterns."""
        # Certificate number
        cert_match = re.search(
            r"CERTIFICATE\s+NUMBER[:\s]+([A-Z0-9/-]+)", text, re.IGNORECASE
        )
        certificate_number = cert_match.group(1) if cert_match else None

        # Issuing date
        date_match = re.search(
            r"ISSUING\s+DATE[:\s]+(\d{2}\.\d{2}\s+\d{4})", text, re.IGNORECASE
        )
        issuing_date = None
        if date_match:
            date_str = date_match.group(1)
            try:
                dt = datetime.strptime(date_str, "%d.%m %Y")
                issuing_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        # Standard
        standard_match = re.search(r"EN\s+10204\s+[\d\.]+", text)
        standard = standard_match.group(0) if standard_match else None

        # Customer
        customer_match = re.search(
            r"CUSTOMER[:\s]+([A-Z\s]+(?:LIMITED|LTD|INC|LLC))", text, re.IGNORECASE
        )
        customer = customer_match.group(1).strip() if customer_match else None

        # Order number
        order_match = re.search(r"ORDER\s+NO[:\s]+([A-Z0-9-]+)", text, re.IGNORECASE)
        order_number = order_match.group(1) if order_match else None

        return {
            "certificate_number": certificate_number,
            "issuing_date": issuing_date,
            "standard": standard,
            "customer": customer,
            "order_number": order_number,
        }

    def _extract_traceability(self, text: str) -> Dict[str, Optional[str]]:
        """Extract traceability information."""
        # Consignment number
        consignment_match = re.search(
            r"CONSIGNMENT\s+NO[:\s]+([A-Z0-9/-]+)", text, re.IGNORECASE
        )
        consignment_number = consignment_match.group(1) if consignment_match else None

        # Vessel name
        vessel_match = re.search(
            r"VESSEL\s+NAME[:\s]+([A-Z\s]+?)(?=\n)", text, re.IGNORECASE
        )
        vessel_name = vessel_match.group(1).strip() if vessel_match else None

        # Lot number
        lot_match = re.search(
            r"ORDER\s+NO:\s+[0-9-]+\s+LOT[:\s-]+(\d+)", text, re.IGNORECASE
        )
        lot_number = lot_match.group(1) if lot_match else None

        return {
            "consignment_number": consignment_number,
            "vessel_name": vessel_name,
            "lot_number": lot_number,
        }

    def _extract_product_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract product specifications."""
        # Size
        size_match = re.search(
            r"SIZE[:\s]+([0-9]+MM[X×][0-9]+M\.?)", text, re.IGNORECASE
        )
        size = size_match.group(1) if size_match else None

        # Quality
        quality_match = re.search(
            r"QUALITY[:\s\n]+([A-Z0-9:]+\s+[A-Z]+\s+[AB][0-9]+\s*[AB]?)",
            text,
            re.IGNORECASE,
        )
        if not quality_match:
            quality_match = re.search(
                r"(BS[0-9]+:[0-9]+\s+GR\s+[AB][0-9]+\s*[AB]?)", text
            )
        quality = quality_match.group(1).strip() if quality_match else None

        # Production process
        process_match = re.search(
            r"PRODUCTION\s+PROSES?[:\s]+([A-Z]+)", text, re.IGNORECASE
        )
        production_process = process_match.group(1) if process_match else None

        return {
            "size": size,
            "quality": quality,
            "production_process": production_process,
        }

    def _extract_chemical_composition(
        self, tables: List[Dict], text: str
    ) -> List[Dict[str, Any]]:
        """
        Extract chemical composition from tables or text.
        """
        compositions = []

        # Standard elements in order
        elements = [
            "C",
            "Si",
            "P",
            "S",
            "Mn",
            "Ni",
            "Cr",
            "Mo",
            "Cu",
            "V",
            "N",
            "B",
            "Ce",
        ]

        # Try table-based extraction first
        for table in tables:
            table_data = table.get("data", [])
            if not table_data or len(table_data) < 2:
                continue

            # Look for chemical composition patterns
            for row in table_data:
                if len(row) < 2:
                    continue

                first_cell = str(row[0]).strip()

                # Check if this looks like a heat number
                heat_match = re.search(r"\b(259900\d{2})\b", first_cell)
                if heat_match:
                    heat_number = heat_match.group(1)

                    # Extract chemical values from remaining cells
                    values = []
                    for cell in row[1:]:
                        cell_str = str(cell).strip().replace(",", ".")
                        # Match decimal values like 0.XX
                        val_match = re.search(r"0[.,]\d{1,4}", cell_str)
                        if val_match:
                            try:
                                values.append(
                                    float(val_match.group(0).replace(",", "."))
                                )
                            except ValueError:
                                pass

                    # Create composition entry if we have enough values
                    if len(values) >= 13:
                        composition = {"heat_number": heat_number, "elements": {}}

                        for i, element in enumerate(elements):
                            if i < len(values):
                                composition["elements"][element] = values[i]

                        compositions.append(composition)

        # Fallback to text-based extraction if no tables worked
        if not compositions:
            print("   ⚠️  Table extraction failed, using text-based extraction")
            compositions = self._extract_chemical_from_text(text)

        # Deduplicate
        seen = set()
        unique_compositions = []
        for comp in compositions:
            heat_num = comp["heat_number"]
            if heat_num not in seen:
                seen.add(heat_num)
                unique_compositions.append(comp)

        return unique_compositions

    def _extract_chemical_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Fallback text-based chemical composition extraction."""
        compositions = []
        elements = [
            "C",
            "Si",
            "P",
            "S",
            "Mn",
            "Ni",
            "Cr",
            "Mo",
            "Cu",
            "V",
            "N",
            "B",
            "Ce",
        ]

        lines = text.split("\n")
        heat_numbers = list(set(re.findall(r"\b(259900\d{2})\b", text)))
        seen_heat_numbers = set()

        for heat_num in heat_numbers:
            if heat_num in seen_heat_numbers:
                continue
            seen_heat_numbers.add(heat_num)

            # Find line index
            heat_idx = None
            for idx, line in enumerate(lines):
                if heat_num in line:
                    heat_idx = idx
                    break

            if heat_idx is None:
                continue

            # Extract values from next 20 lines
            values_window = "\n".join(lines[heat_idx : heat_idx + 20])
            values = re.findall(r"\b0[.,]\d{1,4}\b", values_window)
            values = [v.replace(",", ".") for v in values]

            if len(values) >= 13:
                composition = {"heat_number": heat_num, "elements": {}}

                for i, element in enumerate(elements):
                    if i < len(values):
                        composition["elements"][element] = float(values[i])

                compositions.append(composition)

        return compositions

    def _extract_mechanical_properties(
        self, tables: List[Dict], text: str
    ) -> List[Dict[str, Any]]:
        """
        Extract mechanical properties from tables or text.
        """
        properties = []

        # Try table-based extraction
        for table in tables:
            table_data = table.get("data", [])
            if not table_data:
                continue

            for row in table_data:
                if len(row) < 7:
                    continue

                try:
                    # Look for heat numbers
                    heat_match = re.search(r"\b(259900\d{2})\b", str(row[0]))
                    if not heat_match:
                        continue

                    heat_number = heat_match.group(1)

                    # Parse numeric values from remaining cells
                    values = []
                    for cell in row[1:]:
                        cell_str = str(cell).strip().replace(",", ".")
                        num_match = re.search(r"\d+\.?\d*", cell_str)
                        if num_match:
                            try:
                                values.append(float(num_match.group(0)))
                            except ValueError:
                                pass

                    if len(values) >= 6:
                        weight = values[0]
                        area = values[1]
                        yield_val = values[2]
                        tensile_val = values[3]
                        ratio = values[4]
                        elongation = values[5]

                        # Validate ranges
                        if (
                            6.0 <= weight <= 7.0
                            and 800.0 <= area <= 810.0
                            and 500 <= yield_val <= 700
                            and 600 <= tensile_val <= 800
                        ):

                            prop = {
                                "heat_number": heat_number,
                                "weight_kg_per_m": weight,
                                "cross_sectional_area_mm2": area,
                                "yield_point_mpa": yield_val,
                                "tensile_strength_mpa": tensile_val,
                                "rm_re_ratio": ratio,
                                "percentage_elongation": elongation,
                                "agt_percent": values[6] if len(values) >= 7 else None,
                            }
                            properties.append(prop)

                except (ValueError, IndexError):
                    continue

        # Fallback to text-based extraction
        if not properties:
            print("   ⚠️  Table extraction failed, using text-based extraction")
            properties = self._extract_mechanical_from_text(text)

        # Deduplicate
        seen = set()
        unique_properties = []
        for prop in properties:
            key = (
                prop["heat_number"],
                prop["weight_kg_per_m"],
                prop["yield_point_mpa"],
            )
            if key not in seen:
                seen.add(key)
                unique_properties.append(prop)

        return unique_properties

    def _extract_mechanical_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Fallback text-based mechanical properties extraction."""
        properties = []
        lines = text.split("\n")
        heat_numbers = list(set(re.findall(r"\b(259900\d{2})\b", text)))

        for heat_num in heat_numbers:
            heat_indices = [i for i, line in enumerate(lines) if heat_num in line]

            for heat_idx in heat_indices:
                i = heat_idx + 14  # Skip chemical composition
                test_count = 0

                while i < len(lines) and i < heat_idx + 50:
                    line = lines[i].strip()

                    if re.match(r"^6\.\d{3}$", line):
                        values = []
                        for offset in range(8):
                            if i + offset < len(lines):
                                val_line = lines[i + offset].strip().replace(",", ".")
                                if val_line and val_line not in ["OK", "NG"]:
                                    values.append(val_line)

                        if len(values) >= 6:
                            try:
                                weight = float(values[0])
                                area = float(values[1])
                                yield_val = float(values[2])
                                tensile_val = float(values[3])
                                ratio = float(values[4])
                                elongation = float(values[5])

                                if (
                                    6.0 <= weight <= 7.0
                                    and 800.0 <= area <= 810.0
                                    and 500 <= yield_val <= 700
                                    and 600 <= tensile_val <= 800
                                ):

                                    test_count += 1
                                    properties.append(
                                        {
                                            "heat_number": heat_num,
                                            "test_sample": test_count,
                                            "weight_kg_per_m": weight,
                                            "cross_sectional_area_mm2": area,
                                            "yield_point_mpa": yield_val,
                                            "tensile_strength_mpa": tensile_val,
                                            "rm_re_ratio": ratio,
                                            "percentage_elongation": elongation,
                                            "agt_percent": (
                                                float(values[6])
                                                if len(values) >= 7
                                                else None
                                            ),
                                        }
                                    )
                            except ValueError:
                                pass

                    i += 1

        return properties

    def _extract_approval(self, text: str) -> Dict[str, Optional[Any]]:
        """Extract approval information."""
        cares_approved = "CARES APPROVED" in text.upper()

        cert_approval_match = re.search(
            r"CERTIFICATE\s+OF\s+APPROVAL\s+NO[:\s]+([A-Z0-9/-]+)", text, re.IGNORECASE
        )
        certificate_of_approval_number = (
            cert_approval_match.group(1) if cert_approval_match else None
        )

        form_match = re.search(r"FORM\s+NO[:\s]+([A-Z0-9/-]+)", text, re.IGNORECASE)
        form_number = form_match.group(1) if form_match else None

        return {
            "certificate_of_approval_number": certificate_of_approval_number,
            "form_number": form_number,
            "cares_approved": cares_approved,
        }

    def save_output(self, data: Dict[str, Any], output_path: str):
        """Save extracted data to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved to: {output_file}")


def main():
    """Run Docling extraction on MTC PDFs."""

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    schema_path = project_root / "schema" / "mtc_extraction_schema_v1.json"
    pdf_dir = project_root / "data" / "raw" / "diler"
    output_dir = project_root / "data" / "processed" / "docling_output"

    # Initialize extractor
    extractor = MTCDoclingExtractor(schema_path=str(schema_path))

    # Find PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        return

    print(f"\n🔍 Found {len(pdf_files)} PDF file(s)")

    # Process each PDF
    for pdf_file in pdf_files:
        try:
            # Extract data
            data = extractor.extract_from_pdf(str(pdf_file))

            # Generate output filename
            output_filename = f"{pdf_file.stem}_docling_output.json"
            output_path = output_dir / output_filename

            # Save output
            extractor.save_output(data, str(output_path))

            # Print summary
            print(f"\n📊 Extraction Summary:")
            print(
                f"   Certificate: {data['document'].get('certificate_number', 'N/A')}"
            )
            print(f"   Lot Number: {data['traceability'].get('lot_number', 'N/A')}")
            print(f"   Chemical Compositions: {len(data['chemical_composition'])}")
            print(f"   Mechanical Props: {len(data['mechanical_properties'])}")
            print(f"   Extraction Method: {data['extraction_metadata']['method']}")

        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n✅ Docling extraction complete!")


if __name__ == "__main__":
    main()
