"""Yazici Demir Celik MTC — image-based splitter.

Each page is rendered to a rotation-corrected JPEG and saved as:
    <heat1>_<heat2>_...-YAZICI.jpg

The heat numbers are read from the HEAT NUMBER column via the GitHub Models
vision API (gpt-4o).  Pages where no heat number can be found are saved as
    PAGE<n>-YAZICI.jpg

Usage
-----
    python src/extraction/yazici-extraction/yazici_image_pipeline.py \\
        --input  data/raw/Yazici/Scanned_05-03-2026-100305.pdf \\
                 data/raw/Yazici/Scanned_05-03-2026-100432.pdf \\
        --output data/processed/yazici_images

    # higher DPI for cleaner images (uses more tokens)
    python src/extraction/yazici-extraction/yazici_image_pipeline.py \\
        --input  data/raw/Yazici/Scanned_05-03-2026-100305.pdf \\
        --output data/processed/yazici_images \\
        --dpi 300

Requires
--------
    pip install pymupdf pillow openai
    export GITHUB_TOKEN=<your token>
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pymupdf as fitz  # PyMuPDF >= 1.24
from openai import OpenAI
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GH_BASE = "https://models.inference.ai.azure.com"
DEFAULT_MODEL = "gpt-4o-mini"

# Yazici pages are stored at −270° in the PDF; correct by +90° before saving
_PAGE_ROTATION_DEGREES = 90

# Render DPI — 200 gives clear text with manageable file size
DEFAULT_DPI = 200

# Vision API: cap long edge so we don't blow the token budget
_MAX_LONG_EDGE = 2048

# Only keep heat-number strings that look plausible (4–20 uppercase alphanumeric)
_HEAT_RE = re.compile(r"[A-Z0-9]{4,20}")

_SYSTEM_PROMPT = (
    "You are an expert reader of steel mill test certificates (MTCs). "
    "Your only task is to extract every HEAT NUMBER (also called cast number, "
    "charge number, or ısı numarası) visible on the page. "
    'Return ONLY a JSON array of strings, e.g. ["ABC123", "XYZ456"]. '
    "If no heat number is visible, return an empty array []. "
    "Do not include any other text."
)

_USER_TEXT = (
    "Look at the HEAT NUMBER column (or equivalent) in this mill certificate page. "
    "List every unique heat / cast number you can read. "
    "Return only a JSON array of strings."
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _make_client() -> OpenAI:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN is not set. Export your GitHub personal access token before running."
        )
    return OpenAI(base_url=_GH_BASE, api_key=token)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_page(page: fitz.Page, dpi: int = DEFAULT_DPI) -> Tuple[Image.Image, str]:
    """Render *page* to a rotation-corrected PIL Image and a base64 JPEG string.

    Returns ``(pil_image, base64_string)``.
    The same image object is reused for both the API call and writing to disk —
    no double-rendering.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72).prerotate(_PAGE_ROTATION_DEGREES)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Downscale for the API call only (keeps a separate copy for saving at full DPI)
    api_img = img
    long_edge = max(img.width, img.height)
    if long_edge > _MAX_LONG_EDGE:
        scale = _MAX_LONG_EDGE / long_edge
        api_img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )

    buf = io.BytesIO()
    api_img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()

    # Return the full-DPI image for saving, but the compressed b64 for the API
    return img, b64


# ---------------------------------------------------------------------------
# Heat-number extraction
# ---------------------------------------------------------------------------


def _parse_retry_after(exc: Exception) -> tuple[int, bool]:
    """Return (wait_seconds, is_daily_limit) from a 429 exception message."""
    msg = str(exc)
    # Extract suggested wait seconds from the error message
    m = re.search(r"Please wait (\d+) seconds", msg)
    wait = int(m.group(1)) if m else 60
    daily = "ByDay" in msg or "86400" in msg
    return wait, daily


def extract_heat_numbers(
    img_b64: str,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    retries: int = 3,
) -> List[str]:
    """Ask the vision model for all heat numbers on the page."""
    user_content = [
        {"type": "text", "text": _USER_TEXT},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "high",
            },
        },
    ]

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            heats = _parse_heat_response(raw)
            if heats or attempt == retries:
                return heats
            print(f"      ⚠  Attempt {attempt}: no heats parsed — raw: {raw!r}")
        except Exception as exc:  # noqa: BLE001
            wait, daily = _parse_retry_after(exc)
            if daily:
                print(
                    f"\n❌  Daily rate limit exhausted for model '{model}'. "
                    f"Try again in {wait // 3600:.0f}h {(wait % 3600) // 60:.0f}m, "
                    f"or switch to a different model with --model."
                )
                raise SystemExit(1) from exc
            # Minute-level limit: sleep the required time then retry
            print(f"      ⚠  Attempt {attempt} rate-limited (minute) — sleeping {wait}s …")
            time.sleep(wait + 1)

    return []


def _parse_heat_response(raw: str) -> List[str]:
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        candidates = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    heats: List[str] = []
    seen: set = set()
    for item in candidates:
        if not isinstance(item, str):
            continue
        val = item.strip().upper()
        if _HEAT_RE.fullmatch(val) and val not in seen:
            heats.append(val)
            seen.add(val)
    return heats


# ---------------------------------------------------------------------------
# Main splitting logic
# ---------------------------------------------------------------------------


def _stem(heats: List[str], page_idx: int) -> str:
    base = "_".join(heats) if heats else f"PAGE{page_idx + 1}"
    return f"{base}-YAZICI"


def split_to_images(
    pdf_path: Path,
    output_dir: Path,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    dpi: int = DEFAULT_DPI,
    quality: int = 95,
) -> List[Path]:
    """Render every page of *pdf_path* to a JPEG named by its heat numbers.

    Args:
        pdf_path:   Source PDF.
        output_dir: Destination directory for JPEG files.
        client:     Initialised OpenAI client.
        model:      Vision model ID.
        dpi:        Render resolution (default 200).
        quality:    JPEG quality for saved files (1–95, default 95).

    Returns:
        List of Paths to the written JPEG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    n = len(doc)
    print(f"\n📄  {pdf_path.name}  ({n} page{'s' if n != 1 else ''})")

    written: List[Path] = []

    for idx in range(n):
        page = doc[idx]
        print(f"  ▶  Page {idx + 1}/{n} — rendering at {dpi} DPI …", end=" ", flush=True)

        # Single render call → full-DPI image + downsized b64 for API
        full_img, api_b64 = render_page(page, dpi=dpi)

        print("asking model …", end=" ", flush=True)
        heats = extract_heat_numbers(api_b64, client, model)

        stem = _stem(heats, idx)
        out_path = output_dir / f"{stem}.jpg"

        # Deduplicate when two pages share the same heat set
        if out_path.exists():
            counter = 2
            while out_path.exists():
                out_path = output_dir / f"{stem}_v{counter}.jpg"
                counter += 1

        # Save the full-DPI image — this is the actual certificate content
        full_img.save(str(out_path), format="JPEG", quality=quality)

        size_kb = out_path.stat().st_size // 1024
        heat_display = ", ".join(heats) if heats else "⚠  none found"
        print(f"✓  [{heat_display}]  →  {out_path.name}  ({size_kb} KB)")
        written.append(out_path)

    doc.close()
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render each page of a Yazici MTC PDF to a JPEG "
            "named by its heat numbers: <heat1>_<heat2>_...-YAZICI.jpg"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="PDF",
        help="One or more PDF paths (or glob patterns) to process.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory to write the JPEG files into.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL_ID",
        help="Vision model via GitHub Models API.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        metavar="N",
        help="Render resolution in DPI.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        metavar="1-95",
        help="JPEG save quality for output images.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    pdf_paths: List[Path] = []
    for pattern in args.input:
        matches = glob.glob(pattern)
        if matches:
            pdf_paths.extend(Path(m) for m in sorted(matches))
        else:
            p = Path(pattern)
            if p.exists():
                pdf_paths.append(p)
            else:
                print(f"⚠  No file found for: {pattern!r}", file=sys.stderr)

    if not pdf_paths:
        print("❌  No input PDFs found.", file=sys.stderr)
        return 1

    client = _make_client()
    all_written: List[Path] = []

    for pdf_path in pdf_paths:
        written = split_to_images(
            pdf_path,
            args.output,
            client,
            model=args.model,
            dpi=args.dpi,
            quality=args.quality,
        )
        all_written.extend(written)

    print(f"\n✅  Done — {len(all_written)} image(s) written to {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
