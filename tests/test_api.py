"""
API tests — verify all endpoints work end-to-end using FastAPI's TestClient.

These tests use httpx/TestClient (no real server needed) and mock the
extraction pipeline so they run without GITHUB_TOKEN or OCR engines.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def minimal_extracted() -> dict:
    """A valid minimal extraction result (matches schema structure)."""
    return {
        "document": {
            "certificate_number": "TEST-001",
            "issuing_date": "2025-01-01",
            "standard": "EN 10204 3.1",
            "customer": "ACME Ltd",
            "order_number": "ORD-123",
        },
        "traceability": {
            "heat_number": None,
            "consignment_number": "C-001",
            "vessel_name": "MV TEST",
            "lot_number": "LOT-1",
        },
        "product": {
            "size": "32MM",
            "quality": "BS4449:2005 GR B500 B",
            "production_process": "QST",
        },
        "approval": {
            "certificate_of_approval_number": "011001",
            "form_number": "C8.03",
            "cares_approved": True,
        },
        "chemical_composition": [
            {
                "heat_number": "TEST001",
                "elements": {"C": 0.19, "Si": 0.2, "Mn": 1.0, "P": 0.02, "S": 0.01},
            }
        ],
        "mechanical_properties": [
            {
                "heat_number": "TEST001",
                "test_sample": "A",
                "weight_kg_per_m": 6.31,
                "cross_sectional_area_mm2": 804.0,
                "yield_point_mpa": 550.0,
                "tensile_strength_mpa": 640.0,
                "rm_re_ratio": 1.16,
                "percentage_elongation": 16.0,
                "agt_percent": 7.5,
            }
        ],
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------


def test_list_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    assert body["count"] == len(body["models"])
    assert body["count"] > 0
    first = body["models"][0]
    for key in ("id", "label", "provider", "tier"):
        assert key in first, f"Missing key '{key}' in model entry"


def test_list_models_first_is_gpt4o(client):
    resp = client.get("/models")
    assert resp.json()["models"][0]["id"] == "gpt-4o"


# ---------------------------------------------------------------------------
# /extract — validation errors (no mocking needed)
# ---------------------------------------------------------------------------


def test_extract_rejects_non_pdf(client):
    resp = client.post(
        "/extract/",
        data={"model": "gpt-4o", "mode": "text"},
        files={"pdf": ("cert.txt", io.BytesIO(b"not a pdf"), "text/plain")},
    )
    assert resp.status_code == 422
    assert "PDF" in resp.json()["detail"]


def test_extract_rejects_invalid_mode(client):
    resp = client.post(
        "/extract/",
        data={"model": "gpt-4o", "mode": "invalid"},
        files={"pdf": ("cert.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert resp.status_code == 422
    assert "invalid" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /extract — text mode (mocked pipeline)
# ---------------------------------------------------------------------------


def test_extract_text_mode_success(client, minimal_extracted):
    mock_run_result = {
        "result": minimal_extracted,
        "status": "success",
        "error": None,
        "elapsed": 1.5,
        "token_usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        },
    }

    mock_bench = MagicMock()
    mock_bench.run_ocr_fresh.return_value = ["Page 1 OCR text"]
    mock_bench.run_model.return_value = mock_run_result

    with patch(
        "src.extraction.llm_models_extraction.LLMModelBenchmark",
        return_value=mock_bench,
    ):
        resp = client.post(
            "/extract/",
            data={"model": "gpt-4o", "mode": "text"},
            files={"pdf": ("cert.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "gpt-4o"
    assert body["mode"] == "text"
    assert "run_id" in body
    assert "elapsed_seconds" in body
    assert body["tokens_used"] == 1500
    assert body["extracted"]["document"]["certificate_number"] == "TEST-001"


def test_extract_text_mode_two_pass(client, minimal_extracted):
    mock_run_result = {
        "result": minimal_extracted,
        "status": "success",
        "error": None,
        "elapsed": 3.0,
        "token_usage": {"total_tokens": 3000},
    }

    mock_bench = MagicMock()
    mock_bench.run_ocr_fresh.return_value = ["Page 1", "Page 2"]
    mock_bench.run_model.return_value = mock_run_result

    with patch(
        "src.extraction.llm_models_extraction.LLMModelBenchmark",
        return_value=mock_bench,
    ):
        resp = client.post(
            "/extract/",
            data={"model": "gpt-4o", "mode": "text", "two_pass": "true"},
            files={"pdf": ("cert.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

    assert resp.status_code == 200
    # Verify two_pass=True was forwarded to run_model
    call_kwargs = mock_bench.run_model.call_args
    assert call_kwargs.kwargs.get("two_pass") is True or (
        len(call_kwargs.args) > 2 and call_kwargs.args[2] is True
    )


def test_extract_pipeline_error_returns_500(client):
    mock_bench = MagicMock()
    mock_bench.run_ocr_fresh.side_effect = RuntimeError("OCR engine crashed")

    with patch(
        "src.extraction.llm_models_extraction.LLMModelBenchmark",
        return_value=mock_bench,
    ):
        resp = client.post(
            "/extract/",
            data={"model": "gpt-4o", "mode": "text"},
            files={"pdf": ("cert.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

    assert resp.status_code == 500
    assert "OCR engine crashed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /extract — vision mode (mocked)
# ---------------------------------------------------------------------------


def test_extract_vision_mode_success(client, minimal_extracted):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = minimal_extracted

    with patch(
        "src.extraction.vision_extraction.VisionExtractor", return_value=mock_extractor
    ):
        resp = client.post(
            "/extract/",
            data={"model": "gpt-4o", "mode": "vision"},
            files={"pdf": ("cert.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "vision"
    assert body["extracted"]["document"]["certificate_number"] == "TEST-001"


# ---------------------------------------------------------------------------
# /extract — hybrid mode (mocked)
# ---------------------------------------------------------------------------


def test_extract_hybrid_mode_success(client, minimal_extracted):
    hybrid_result = {
        **minimal_extracted,
        "_pipeline_meta": {
            "strategy_used": "text_llm",
            "ocr_confidence": 0.87,
            "elapsed_seconds": 12.3,
            "tokens_used": 8000,
            "text_model": "gpt-4o",
            "vision_model": "gpt-4o",
        },
    }

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = hybrid_result

    with patch(
        "src.extraction.hybrid_pipeline.HybridPipeline", return_value=mock_pipeline
    ):
        resp = client.post(
            "/extract/",
            data={"model": "gpt-4o", "mode": "hybrid"},
            files={"pdf": ("cert.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "hybrid"
    # _pipeline_meta must be stripped from the response extracted dict
    assert "_pipeline_meta" not in body["extracted"]
    assert body["tokens_used"] == 8000


# ---------------------------------------------------------------------------
# /evaluate
# ---------------------------------------------------------------------------


def test_evaluate_missing_prediction(client):
    resp = client.post(
        "/evaluate",
        json={
            "prediction_path": "data/does_not_exist.json",
            "ground_truth_path": "data/ground_truth/diler-07-07-2025-rerun-41-44_gt.json",
        },
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_evaluate_missing_ground_truth(client):
    resp = client.post(
        "/evaluate",
        json={
            "prediction_path": "data/processed/benchmark_output/gpt-4o_extracted.json",
            "ground_truth_path": "data/ground_truth/does_not_exist.json",
        },
    )
    assert resp.status_code == 404


def test_evaluate_identical_files_gives_perfect_score(
    client, minimal_extracted, tmp_path
):
    """Evaluating a file against itself should return overall_f1 = 1.0."""
    pred_file = tmp_path / "pred.json"
    gt_file = tmp_path / "gt.json"
    pred_file.write_text(json.dumps(minimal_extracted))
    gt_file.write_text(json.dumps(minimal_extracted))

    pred_rel = (
        str(pred_file.relative_to(REPO_ROOT))
        if pred_file.is_relative_to(REPO_ROOT)
        else str(pred_file)
    )
    gt_rel = (
        str(gt_file.relative_to(REPO_ROOT))
        if gt_file.is_relative_to(REPO_ROOT)
        else str(gt_file)
    )

    # Patch REPO_ROOT resolution so the endpoint finds our tmp files
    with patch("src.api.routes.benchmark.REPO_ROOT", tmp_path):
        resp = client.post(
            "/evaluate",
            json={
                "prediction_path": "pred.json",
                "ground_truth_path": "gt.json",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "overall_f1" in body
    assert body["overall_f1"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# MTCEvaluator unit tests
# ---------------------------------------------------------------------------


def test_evaluator_document_accuracy(minimal_extracted):
    from src.evaluation import MTCEvaluator

    report = MTCEvaluator.evaluate(minimal_extracted, minimal_extracted)
    assert report["document"]["document_accuracy"] == pytest.approx(1.0)


def test_evaluator_chemical_f1_perfect(minimal_extracted):
    from src.evaluation import MTCEvaluator

    report = MTCEvaluator.evaluate(minimal_extracted, minimal_extracted)
    assert report["chemical"]["heat_f1"] == pytest.approx(1.0)
    assert report["chemical"]["element_accuracy"] == pytest.approx(1.0)


def test_evaluator_mechanical_f1_perfect(minimal_extracted):
    from src.evaluation import MTCEvaluator

    report = MTCEvaluator.evaluate(minimal_extracted, minimal_extracted)
    assert report["mechanical"]["mech_f1"] == pytest.approx(1.0)
    assert report["mechanical"]["mech_property_accuracy"] == pytest.approx(1.0)


def test_evaluator_missing_heat_reduces_recall(minimal_extracted):
    from src.evaluation import MTCEvaluator
    import copy

    extracted_missing = copy.deepcopy(minimal_extracted)
    extracted_missing["chemical_composition"] = []  # extracted nothing

    report = MTCEvaluator.evaluate(extracted_missing, minimal_extracted)
    assert report["chemical"]["heat_recall"] == pytest.approx(0.0)
    # tp=0, fp=0 → evaluator returns 0.0 on empty denominator
    assert report["chemical"]["heat_precision"] == pytest.approx(0.0)


def test_evaluator_extra_heat_reduces_precision(minimal_extracted):
    from src.evaluation import MTCEvaluator
    import copy

    extracted_extra = copy.deepcopy(minimal_extracted)
    extracted_extra["chemical_composition"].append(
        {"heat_number": "EXTRA999", "elements": {"C": 0.1}}
    )

    report = MTCEvaluator.evaluate(extracted_extra, minimal_extracted)
    assert report["chemical"]["heat_precision"] < 1.0
    assert report["chemical"]["heat_recall"] == pytest.approx(1.0)


def test_evaluator_fuzzy_mechanical_match_within_tolerance(minimal_extracted):
    """A row with yield ±3 MPa should still match."""
    from src.evaluation import MTCEvaluator
    import copy

    slightly_off = copy.deepcopy(minimal_extracted)
    slightly_off["mechanical_properties"][0]["yield_point_mpa"] = 553.0  # +3 MPa

    report = MTCEvaluator.evaluate(slightly_off, minimal_extracted, mech_yield_tol=5.0)
    assert report["mechanical"]["mech_f1"] == pytest.approx(1.0)


def test_evaluator_fuzzy_mechanical_no_match_outside_tolerance(minimal_extracted):
    """A row with yield ±50 MPa should NOT match with default tolerance of 5."""
    from src.evaluation import MTCEvaluator
    import copy

    far_off = copy.deepcopy(minimal_extracted)
    far_off["mechanical_properties"][0]["yield_point_mpa"] = 600.0  # +50 MPa

    report = MTCEvaluator.evaluate(far_off, minimal_extracted, mech_yield_tol=5.0)
    assert report["mechanical"]["mech_f1"] < 1.0
