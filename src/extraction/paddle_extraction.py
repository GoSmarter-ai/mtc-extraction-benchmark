import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MTCPaddleExtractor:
    def __init__(self, schema_path: Optional[str] = None):
        # Load schema if provided
        self.schema = None
        if schema_path and Path(schema_path).exists():
            with open(schema_path, "r") as f:
                self.schema = json.load(f)

        print("✓ PaddleOCR extractor initialized")

    def extract_from_paddle_json(self, json_path: str) -> Dict[str, Any]:

        print(f"\n📄 Processing: {json_path}")

        # Load PaddleOCR JSON
        with open(json_path, "r", encoding="utf-8") as f:
            paddle_data = json.load(f)

        # Combine all text from all pages
        full_text = self._combine_paddle_text(paddle_data)

        # Extract data
        extracted_data = {
            "document": self._extract_document_info(full_text),
            "traceability": self._extract_traceability(full_text),
            "product": self._extract_product_info(full_text),
            "chemical_composition": self._extract_chemical_composition(full_text),
            "mechanical_properties": self._extract_mechanical_properties(full_text),
            "approval": self._extract_approval(full_text),
        }

        return extracted_data

    def _combine_paddle_text(self, paddle_data: List[Dict]) -> str:

        all_text = []

        for page_data in paddle_data:
            # Get text lines from each page
            text_lines = page_data.get("text_lines", [])
            for line in text_lines:
                if isinstance(line, dict) and "text" in line:
                    all_text.append(line["text"])
                elif isinstance(line, str):
                    all_text.append(line)

            # Also get full_text if available
            full_text = page_data.get("full_text", [])
            if isinstance(full_text, list):
                all_text.extend([t for t in full_text if isinstance(t, str)])

        return "\n".join(all_text)

    def _extract_document_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract document-level information."""
        # Extract certificate number
        cert_match = re.search(r"CERTIFICATE\s+NUMBER[:\s]+([A-Z0-9/-]+)", text, re.IGNORECASE)
        certificate_number = cert_match.group(1) if cert_match else None

        # Extract issuing date
        date_match = re.search(r"ISSUING\s+DATE[:\s]+(\d{2}\.\d{2}\s+\d{4})", text, re.IGNORECASE)
        issuing_date = None
        if date_match:
            # Convert DD.MM YYYY to YYYY-MM-DD
            date_str = date_match.group(1)
            try:
                dt = datetime.strptime(date_str, "%d.%m %Y")
                issuing_date = dt.strftime("%Y-%m-%d")
            except Exception:
                issuing_date = date_str

        # Extract standard
        standard_match = re.search(r"EN\s+10204\s+[\d\.]+", text)
        standard = standard_match.group(0) if standard_match else None

        # Extract customer - stop at newline or MILL
        customer_match = re.search(
            r"CUSTOMER[:\s]+([A-Z\s]+(?:LIMITED|LTD|INC|LLC))",
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

    def _extract_traceability(self, text: str) -> Dict[str, Optional[str]]:
        """Extract traceability information."""
        # Extract consignment number
        consignment_match = re.search(r"CONSIGNMENT\s+NO[:\s]+([A-Z0-9/-]+)", text, re.IGNORECASE)
        consignment_number = consignment_match.group(1) if consignment_match else None

        # Extract vessel name - stop at newline
        vessel_match = re.search(r"VESSEL\s+NAME[:\s]+([A-Z\s]+?)(?=\n)", text, re.IGNORECASE)
        vessel_name = vessel_match.group(1).strip() if vessel_match else None

        # Extract lot number
        lot_match = re.search(r"ORDER\s+NO:\s+[0-9-]+\s+LOT[:\s-]+(\d+)", text, re.IGNORECASE)
        lot_number = lot_match.group(1) if lot_match else None

        return {
            "consignment_number": consignment_number,
            "vessel_name": vessel_name,
            "lot_number": lot_number,
        }

    def _extract_product_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract product information."""
        # Extract size
        size_match = re.search(r"SIZE[:\s]+([0-9]+MM[X×][0-9]+M\.?)", text, re.IGNORECASE)
        size = size_match.group(1) if size_match else None

        # Extract quality/grade - look for BS standard
        quality_match = re.search(
            r"QUALITY[:\s\n]+([A-Z0-9:]+\s+[A-Z]+\s+[AB][0-9]+\s*[AB]?)",
            text,
            re.IGNORECASE,
        )
        if not quality_match:
            quality_match = re.search(
                r"(BS[0-9]+:[0-9]+\s+GR\s+[AB][0-9]+\s*[AB]?)", text, re.IGNORECASE
            )
        quality = quality_match.group(1).strip() if quality_match else None

        # Extract production process
        process_match = re.search(r"PRODUCTION\s+PROSES?[:\s]+([A-Z]+)", text, re.IGNORECASE)
        production_process = process_match.group(1) if process_match else None

        return {
            "size": size,
            "quality": quality,
            "production_process": production_process,
        }

    def _extract_chemical_composition(self, text: str) -> List[Dict[str, Any]]:
        """Extract chemical composition data by parsing table rows."""
        compositions = []
        seen_heat_numbers = set()

        # Find all heat numbers (8-digit numbers starting with 25990)
        heat_numbers = re.findall(r"\b(259900\d{2})\b", text)

        if not heat_numbers:
            return compositions

        # Split text into lines for easier parsing
        lines = text.split("\n")

        # For each heat number, find the corresponding chemical composition values
        for heat_num in heat_numbers:
            # Skip if already processed (deduplication)
            if heat_num in seen_heat_numbers:
                continue
            seen_heat_numbers.add(heat_num)
            # Find the line index where this heat number appears
            heat_idx = None
            for i, line in enumerate(lines):
                if heat_num in line:
                    heat_idx = i
                    break

            if heat_idx is None:
                continue

            # The chemical composition values should be in the same and following few lines
            # Pattern: heat_num, C, Si, P, S, Mn, Ni, Cr, Mo, Cu, V, N, B, Ce
            values_window = lines[heat_idx : heat_idx + 20]
            values_text = " ".join(values_window)

            # Extract decimal values after the heat number
            decimal_pattern = r"\b0[.,]\d{1,4}\b"
            values = re.findall(decimal_pattern, values_text)

            # Clean values (replace comma with period)
            values = [v.replace(",", ".") for v in values]

            if len(values) >= 13:  # We expect at least 13 chemical element values
                try:
                    composition_entry = {
                        "heat_number": heat_num,
                        "elements": {
                            "C": float(values[0]),
                            "Si": float(values[1]),
                            "P": float(values[2]),
                            "S": float(values[3]),
                            "Mn": float(values[4]),
                            "Ni": float(values[5]),
                            "Cr": float(values[6]),
                            "Mo": float(values[7]),
                            "Cu": float(values[8]),
                            "V": float(values[9]),
                            "N": float(values[10]),
                            "B": float(values[11]),
                            "Ce": float(values[12]),
                        },
                    }
                    compositions.append(composition_entry)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse chemical composition for heat {heat_num}: {e}")
                    continue

        return compositions

    def _extract_mechanical_properties(self, text: str) -> List[Dict[str, Any]]:
        """Extract mechanical properties by parsing table rows."""
        properties = []
        seen_tests = set()  # Track (heat_num, sample) to avoid duplicates

        # Find all heat numbers
        heat_numbers = list(set(re.findall(r"\b(259900\d{2})\b", text)))  # Deduplicate

        if not heat_numbers:
            return properties

        lines = text.split("\n")

        # For each heat number, find its associated mechanical property test results
        for heat_num in heat_numbers:
            # Find where this heat number appears
            heat_indices = [i for i, line in enumerate(lines) if heat_num in line]

            for heat_idx in heat_indices:
                # After the heat number and chemical composition (usually 13-14 values),
                # mechanical test data appears. Look for weight pattern: 6.xxx
                test_count = 0
                i = heat_idx + 14  # Skip chemical composition values

                while i < len(lines) and i < heat_idx + 50:  # Look within reasonable range
                    line = lines[i].strip()

                    # Check if this line starts a test sample (weight value ~6.xxx)
                    if re.match(r"^6\.\d{3}$", line):
                        # Collect next 7-8 values: weight, area, yield, tensile, ratio, elongation, agt, rebend
                        try:
                            values = []
                            for offset in range(8):  # Collect next 8 lines
                                if i + offset < len(lines):
                                    val_line = lines[i + offset].strip()
                                    # Clean and convert
                                    val_line = val_line.replace(",", ".")
                                    if val_line and val_line not in ["OK", "NG"]:
                                        values.append(val_line)

                            # Need at least 6 values: weight, area, yield, tensile, ratio, elongation
                            if len(values) >= 6:
                                weight = float(values[0])
                                area = float(values[1])
                                yield_val = float(values[2])
                                tensile_val = float(values[3])
                                ratio = float(values[4])
                                elongation = float(values[5])

                                # Validate ranges
                                if (
                                    6.0 <= weight <= 7.0
                                    and 800.0 <= area <= 810.0
                                    and 500 <= yield_val <= 700
                                    and 600 <= tensile_val <= 800
                                ):
                                    test_count += 1
                                    test_key = (heat_num, heat_idx, test_count)

                                    if test_key not in seen_tests:
                                        seen_tests.add(test_key)

                                        prop_entry = {
                                            "heat_number": heat_num,
                                            "test_sample": test_count,
                                            "weight_kg_per_m": weight,
                                            "cross_sectional_area_mm2": area,
                                            "yield_point_mpa": yield_val,
                                            "tensile_strength_mpa": tensile_val,
                                            "rm_re_ratio": ratio,
                                            "percentage_elongation": elongation,
                                        }

                                        # Add optional Agt if available
                                        if len(values) >= 7:
                                            try:
                                                prop_entry["agt_percent"] = float(values[6])
                                            except Exception:
                                                pass

                                        properties.append(prop_entry)

                                        # Skip past this test sample
                                        i += 8
                                        continue
                        except (ValueError, IndexError):
                            pass

                    i += 1

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
    project_root = Path(__file__).parent.parent.parent
    schema_path = project_root / "schema" / "mtc_extraction_schema_v1.json"
    paddle_data_dir = project_root / "data" / "processed" / "docling" / "paddle_combined"
    output_dir = project_root / "data" / "processed" / "schema_output"

    # Initialize extractor
    extractor = MTCPaddleExtractor(schema_path=str(schema_path))

    # Find PaddleOCR JSON files
    json_files = list(paddle_data_dir.glob("*_combined.json"))

    if not json_files:
        print(f"No JSON files found in {paddle_data_dir}")
        return

    print(f"\n Found {len(json_files)} JSON file(s)")

    # Process each JSON
    for json_file in json_files:
        try:
            # Extract data
            data = extractor.extract_from_paddle_json(str(json_file))

            # Save output
            output_filename = json_file.stem.replace("_combined", "_schema_output") + ".json"
            output_path = output_dir / output_filename
            extractor.save_output(data, str(output_path))

            # Print summary
            print("\n Extraction Summary:")
            print(f"   Certificate: {data['document'].get('certificate_number')}")
            print(f"   Lot Number: {data['traceability'].get('lot_number')}")
            print(f"   Chemical Compositions: {len(data['chemical_composition'])}")
            print(f"   Mechanical Props: {len(data['mechanical_properties'])}")

        except Exception as e:
            print(f" Error processing {json_file.name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n Processing complete!")


if __name__ == "__main__":
    main()
