"""
MTC Extraction REST API
=======================
FastAPI application that exposes the MTC extraction pipeline over HTTP.

Start the server
----------------
    uvicorn src.api.main:app --reload --port 8000

Or via Docker:
    docker build -t mtc-api .
    docker run -p 8000:8000 -e GITHUB_TOKEN=$GITHUB_TOKEN mtc-api

Endpoints
---------
    GET  /health            — liveness check
    GET  /models            — list available models
    POST /extract           — extract from uploaded PDF (multipart/form-data)
    POST /benchmark         — run model benchmark (uses cached OCR)
    POST /evaluate          — evaluate prediction vs ground truth
    GET  /docs              — OpenAPI Swagger UI (auto-generated)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.benchmark import router as benchmark_router
from src.api.routes.extract import router as extract_router

app = FastAPI(
    title="MTC Extraction API",
    description=(
        "REST API for structured data extraction from Mill Test Certificates (MTC). "
        "Supports text-based LLM extraction, vision LLM extraction, and a hybrid "
        "confidence-routing pipeline."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS from any origin for development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(extract_router)
app.include_router(benchmark_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe — returns 200 if the service is running."""
    return {"status": "ok", "service": "mtc-extraction-api", "version": "0.1.0"}
