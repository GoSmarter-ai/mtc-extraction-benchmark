"""
MTCEvaluator
============
Standalone evaluation module for MTC extraction quality measurement.

Compares a model's extracted JSON against a ground-truth annotation and
produces per-field Precision / Recall / F1 scores with numeric tolerance.

Usage
-----
From Python:
    from src.evaluation import MTCEvaluator

    report = MTCEvaluator.evaluate(
        extracted=json.loads(Path("model_output.json").read_text()),
        ground_truth=json.loads(Path("ground_truth.json").read_text()),
    )
    print(MTCEvaluator.format_report(report))

From CLI:
    python -m src.evaluation.evaluator \\
        --prediction data/processed/benchmark_output/gpt-4o_extracted.json \\
        --ground-truth data/ground_truth/diler-07-07-2025-rerun-41-44_gt.json \\
        --tolerance 0.001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class MTCEvaluator:
    """
    Evaluation engine for MTC structured data extraction.

    All public methods are static so the class works without instantiation.
    """

    # Numeric fields in mechanical_properties
    MECH_NUMERIC_FIELDS: List[str] = [
        "weight_kg_per_m",
        "cross_sectional_area_mm2",
        "yield_point_mpa",
        "tensile_strength_mpa",
        "rm_re_ratio",
        "percentage_elongation",
        "agt_percent",
    ]

    # Scalar fields in document section
    DOCUMENT_FIELDS: List[str] = [
        "certificate_number",
        "issuing_date",
        "standard",
        "customer",
        "order_number",
    ]

    # Chemical element symbols
    CHEMICAL_ELEMENTS: List[str] = [
        "C",
        "Si",
        "Mn",
        "P",
        "S",
        "Ni",
        "Cr",
        "Mo",
        "Cu",
        "V",
        "N",
        "B",
        "Ce",
    ]

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    @classmethod
    def evaluate(
        cls,
        extracted: dict,
        ground_truth: dict,
        numeric_tolerance: float = 0.001,
        mech_weight_tol: float = 0.01,
        mech_yield_tol: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Full evaluation of one extraction against a ground-truth annotation.

        Args:
            extracted:         Model's extracted JSON.
            ground_truth:      Human-annotated reference JSON.
            numeric_tolerance: Absolute tolerance for chemical element values
                               and exact numeric field matching (default 0.001).
            mech_weight_tol:   Tolerance for weight_kg_per_m in fuzzy key
                               matching (default 0.01 kg/m).
            mech_yield_tol:    Tolerance for yield_point_mpa in fuzzy key
                               matching (default 5 MPa).

        Returns:
            Dict with keys: document, chemical, mechanical, approval, overall.
        """
        doc_scores = cls._eval_document(extracted, ground_truth)
        chem_scores = cls._eval_chemical(extracted, ground_truth, numeric_tolerance)
        mech_scores = cls._eval_mechanical(
            extracted,
            ground_truth,
            numeric_tolerance,
            mech_weight_tol,
            mech_yield_tol,
        )
        approval_scores = cls._eval_approval(extracted, ground_truth)

        # Weighted overall F1 — chemistry and mechanics count for more
        component_f1s = [
            doc_scores["document_accuracy"],
            chem_scores["heat_f1"],
            chem_scores["element_accuracy"],
            mech_scores["mech_f1"],
            mech_scores["mech_property_accuracy"],
            approval_scores["approval_accuracy"],
        ]
        overall_f1 = round(sum(component_f1s) / len(component_f1s), 4)

        return {
            "document": doc_scores,
            "chemical": chem_scores,
            "mechanical": mech_scores,
            "approval": approval_scores,
            "overall_f1": overall_f1,
            "config": {
                "numeric_tolerance": numeric_tolerance,
                "mech_weight_tol": mech_weight_tol,
                "mech_yield_tol": mech_yield_tol,
            },
        }

    @staticmethod
    def format_report(report: Dict[str, Any]) -> str:
        """Render an evaluation report as a human-readable ASCII table."""
        lines = [
            "",
            "╔══════════════════════════════════════════════════╗",
            "║          MTC EXTRACTION EVALUATION REPORT        ║",
            "╠══════════════════════════════════════════════════╣",
        ]

        def row(label: str, value: Any, width: int = 48) -> str:
            label_s = f"  {label}"
            value_s = f"{value}"
            pad = width - len(label_s) - len(value_s)
            return f"║{label_s}{' ' * max(pad, 1)}{value_s}║"

        # Document
        doc = report["document"]
        lines += [
            "║  DOCUMENT FIELDS                                 ║",
            row("  Accuracy", f"{doc['document_accuracy']:.1%}"),
        ]
        for field, ok in doc.get("document_fields", {}).items():
            mark = "✓" if ok else "✗"
            lines.append(row(f"    {mark} {field}", ""))

        # Chemical
        chem = report["chemical"]
        lines += [
            "╠══════════════════════════════════════════════════╣",
            "║  CHEMICAL COMPOSITION                            ║",
            row("  Heat Precision", f"{chem['heat_precision']:.1%}"),
            row("  Heat Recall", f"{chem['heat_recall']:.1%}"),
            row("  Heat F1", f"{chem['heat_f1']:.1%}"),
            row("  Element Accuracy", f"{chem['element_accuracy']:.1%}"),
            row("  Extracted Heats", chem.get("extracted_heats", "?")),
            row("  GT Heats", chem.get("gt_heats", "?")),
        ]

        # Mechanical
        mech = report["mechanical"]
        lines += [
            "╠══════════════════════════════════════════════════╣",
            "║  MECHANICAL PROPERTIES                           ║",
            row("  Row Precision", f"{mech['mech_precision']:.1%}"),
            row("  Row Recall", f"{mech['mech_recall']:.1%}"),
            row("  Row F1", f"{mech['mech_f1']:.1%}"),
            row("  Property Accuracy", f"{mech['mech_property_accuracy']:.1%}"),
            row("  Extracted Rows", mech.get("extracted_rows", "?")),
            row("  GT Rows", mech.get("gt_rows", "?")),
            row("  Fuzzy Matched", mech.get("fuzzy_matched", "?")),
        ]

        # Approval
        appr = report["approval"]
        lines += [
            "╠══════════════════════════════════════════════════╣",
            "║  APPROVAL                                        ║",
            row("  Accuracy", f"{appr['approval_accuracy']:.1%}"),
        ]

        # Overall
        lines += [
            "╠══════════════════════════════════════════════════╣",
            row("  ★ OVERALL F1", f"{report['overall_f1']:.1%}"),
            "╚══════════════════════════════════════════════════╝",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def save_report(report: Dict[str, Any], output_path: Path) -> None:
        """Save the evaluation report as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    @classmethod
    def _eval_document(cls, extracted: dict, gt: dict) -> Dict[str, Any]:
        gt_doc = gt.get("document", {})
        ex_doc = extracted.get("document", {})
        correct = 0
        total = 0
        details: Dict[str, bool] = {}
        for field in cls.DOCUMENT_FIELDS:
            gt_val = gt_doc.get(field)
            if gt_val is None:
                continue
            total += 1
            ex_val = ex_doc.get(field)
            match = (
                str(gt_val).strip().lower() == str(ex_val).strip().lower()
                if ex_val is not None
                else False
            )
            if match:
                correct += 1
            details[field] = match
        return {
            "document_accuracy": round(correct / max(total, 1), 4),
            "document_fields": details,
            "correct": correct,
            "total": total,
        }

    @classmethod
    def _eval_chemical(
        cls,
        extracted: dict,
        gt: dict,
        numeric_tolerance: float,
    ) -> Dict[str, Any]:
        gt_chem = {c["heat_number"]: c for c in gt.get("chemical_composition", [])}
        ex_chem = {c["heat_number"]: c for c in extracted.get("chemical_composition", [])}

        gt_heats = set(gt_chem)
        ex_heats = set(ex_chem)
        tp = len(ex_heats & gt_heats)
        fp = len(ex_heats - gt_heats)
        fn = len(gt_heats - ex_heats)

        precision = round(tp / max(tp + fp, 1), 4)
        recall = round(tp / max(tp + fn, 1), 4)
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if precision + recall > 0
            else 0.0
        )

        # Element-level accuracy within matched heats
        elem_correct = 0
        elem_total = 0
        elem_detail: Dict[str, Dict[str, bool]] = {}
        for heat in ex_heats & gt_heats:
            gt_elems = gt_chem[heat].get("elements", {})
            ex_elems = ex_chem[heat].get("elements", {})
            heat_detail: Dict[str, bool] = {}
            for elem in cls.CHEMICAL_ELEMENTS:
                gt_val = gt_elems.get(elem)
                if gt_val is None:
                    continue
                elem_total += 1
                ex_val = ex_elems.get(elem)
                try:
                    match = (
                        abs(float(ex_val) - float(gt_val)) <= numeric_tolerance
                        if ex_val is not None
                        else False
                    )
                except (ValueError, TypeError):
                    match = str(ex_val).strip() == str(gt_val).strip() if ex_val else False
                if match:
                    elem_correct += 1
                heat_detail[elem] = match
            elem_detail[heat] = heat_detail

        return {
            "heat_precision": precision,
            "heat_recall": recall,
            "heat_f1": f1,
            "element_accuracy": round(elem_correct / max(elem_total, 1), 4),
            "element_correct": elem_correct,
            "element_total": elem_total,
            "element_detail": elem_detail,
            "extracted_heats": len(ex_heats),
            "gt_heats": len(gt_heats),
            "missing_heats": sorted(gt_heats - ex_heats),
            "extra_heats": sorted(ex_heats - gt_heats),
        }

    @classmethod
    def _eval_mechanical(
        cls,
        extracted: dict,
        gt: dict,
        numeric_tolerance: float,
        weight_tol: float,
        yield_tol: float,
    ) -> Dict[str, Any]:
        gt_rows = gt.get("mechanical_properties", [])
        ex_rows = extracted.get("mechanical_properties", [])

        # Fuzzy matching: find the best GT match for each extracted row
        # by minimising abs difference on (weight_kg_per_m, yield_point_mpa)
        # within the same heat_number.
        matched_pairs: List[Tuple[dict, dict]] = []
        unmatched_ex: List[dict] = []
        used_gt_idx: set = set()

        for ex_row in ex_rows:
            best_idx: Optional[int] = None
            best_dist = float("inf")
            ex_heat = ex_row.get("heat_number", "")
            ex_weight = ex_row.get("weight_kg_per_m")
            ex_yield = ex_row.get("yield_point_mpa")

            for gi, gt_row in enumerate(gt_rows):
                if gi in used_gt_idx:
                    continue
                if gt_row.get("heat_number", "") != ex_heat:
                    continue
                gt_weight = gt_row.get("weight_kg_per_m")
                gt_yield = gt_row.get("yield_point_mpa")
                try:
                    w_diff = abs(float(ex_weight or 0) - float(gt_weight or 0))
                    y_diff = abs(float(ex_yield or 0) - float(gt_yield or 0))
                except (TypeError, ValueError):
                    continue
                if w_diff <= weight_tol and y_diff <= yield_tol:
                    dist = w_diff + y_diff * 0.01
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = gi

            if best_idx is not None:
                matched_pairs.append((ex_rows[ex_rows.index(ex_row)], gt_rows[best_idx]))
                used_gt_idx.add(best_idx)
            else:
                unmatched_ex.append(ex_row)

        unmatched_gt = [gt_rows[i] for i in range(len(gt_rows)) if i not in used_gt_idx]

        tp = len(matched_pairs)
        fp = len(unmatched_ex)
        fn = len(unmatched_gt)

        precision = round(tp / max(tp + fp, 1), 4)
        recall = round(tp / max(tp + fn, 1), 4)
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if precision + recall > 0
            else 0.0
        )

        # Property-level accuracy within fuzzy-matched pairs
        prop_correct = 0
        prop_total = 0
        for ex_m, gt_m in matched_pairs:
            for prop in cls.MECH_NUMERIC_FIELDS:
                gt_val = gt_m.get(prop)
                if gt_val is None:
                    continue
                prop_total += 1
                ex_val = ex_m.get(prop)
                try:
                    if (
                        ex_val is not None
                        and abs(float(ex_val) - float(gt_val)) <= numeric_tolerance
                    ):
                        prop_correct += 1
                except (TypeError, ValueError):
                    if str(ex_val).strip() == str(gt_val).strip():
                        prop_correct += 1

        return {
            "mech_precision": precision,
            "mech_recall": recall,
            "mech_f1": f1,
            "mech_property_accuracy": round(prop_correct / max(prop_total, 1), 4),
            "fuzzy_matched": tp,
            "unmatched_extracted": fp,
            "unmatched_gt": fn,
            "extracted_rows": len(ex_rows),
            "gt_rows": len(gt_rows),
        }

    @staticmethod
    def _eval_approval(extracted: dict, gt: dict) -> Dict[str, Any]:
        gt_appr = gt.get("approval", {})
        ex_appr = extracted.get("approval", {})
        appr_fields = [
            "certificate_of_approval_number",
            "form_number",
            "cares_approved",
        ]
        correct = 0
        total = 0
        details: Dict[str, bool] = {}
        for field in appr_fields:
            gt_val = gt_appr.get(field)
            if gt_val is None:
                continue
            total += 1
            ex_val = ex_appr.get(field)
            match = gt_val == ex_val
            if match:
                correct += 1
            details[field] = match
        return {
            "approval_accuracy": round(correct / max(total, 1), 4),
            "approval_fields": details,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an MTC extraction against a ground-truth annotation."
    )
    parser.add_argument(
        "--prediction",
        "-p",
        type=Path,
        required=True,
        help="Path to the model's extracted JSON file.",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        type=Path,
        required=True,
        help="Path to the ground-truth JSON annotation.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="Absolute numeric tolerance for chemical elements (default: 0.001).",
    )
    parser.add_argument(
        "--weight-tol",
        type=float,
        default=0.01,
        help="Tolerance for weight_kg_per_m fuzzy matching (default: 0.01).",
    )
    parser.add_argument(
        "--yield-tol",
        type=float,
        default=5.0,
        help="Tolerance for yield_point_mpa fuzzy matching (default: 5.0).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to save the evaluation report as JSON.",
    )
    args = parser.parse_args()

    if not args.prediction.exists():
        print(f"❌ Prediction file not found: {args.prediction}")
        return 1
    if not args.ground_truth.exists():
        print(f"❌ Ground truth file not found: {args.ground_truth}")
        return 1

    extracted = json.loads(args.prediction.read_text())
    gt = json.loads(args.ground_truth.read_text())

    report = MTCEvaluator.evaluate(
        extracted=extracted,
        ground_truth=gt,
        numeric_tolerance=args.tolerance,
        mech_weight_tol=args.weight_tol,
        mech_yield_tol=args.yield_tol,
    )

    print(MTCEvaluator.format_report(report))

    if args.output:
        MTCEvaluator.save_report(report, args.output)
        print(f"💾 Report saved → {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
