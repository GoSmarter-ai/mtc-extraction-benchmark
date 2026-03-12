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
from typing import List, Optional

import pymupdf as fitz  # PyMuPDF >= 1.24 (replaces legacy `import fitz`)
from openai import OpenAI
from PIL import Image

_GH_BASE = "https://models.inference.ai.azure.com"
DEFAULT_MODEL = "gpt-4o-mini"

_RENDER_DPI = 200
_MAX_LONG_EDGE = 2048

_PAGE_ROTATION_DEGREES = 90

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


def _make_client() -> OpenAI:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise EnvironmentError("GITHUB TOKEN is not set")
    return OpenAI(base_url=_GH_BASE, api_key=token)


def _image_to_base64(page: fitz.Page, dpi: int = _RENDER_DPI) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72).prerotate(_PAGE_ROTATION_DEGREES)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    long_edge = max(img.width, img.height)

    if long_edge > _MAX_LONG_EDGE:
        scale = _MAX_LONG_EDGE / long_edge
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_retry_after(exc: Exception) -> tuple[int, bool]:
    """Return (wait_seconds, is_daily_limit) from a 429 exception message."""
    msg = str(exc)
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
            print(f" Attempt {attempt}: no heats parsed - raw: {raw!r}")
        except Exception as exc:
            wait, daily = _parse_retry_after(exc)
            if daily:
                print(
                    f"\n❌  Daily rate limit exhausted for model '{model}'. "
                    f"Try again in {wait // 3600:.0f}h {(wait % 3600) // 60:.0f}m, "
                    f"or switch to a different model with --model."
                )
                raise SystemExit(1) from exc
            print(f" Attempt {attempt} rate-limited (minute) — sleeping {wait}s …")
            time.sleep(wait + 1)
    return []


def _parse_heat_response(raw: str) -> List[str]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
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


def _safe_filename(heats: List[str], page_idx: int) -> str:
    stem = "_".join(heats) if heats else f"PAGE{page_idx + 1}"
    return f"{stem}-YAZICI"


def split_pdf(
    pdf_path: Path,
    output_dir: Path,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
) -> List[Path]:

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    print(f"\n {pdf_path.name} ({n_pages} page{'s' if n_pages != 1 else ''})")

    written: List[Path] = []

    for page_idx in range(n_pages):
        page = doc[page_idx]
        print(f"Page {page_idx + 1}/{n_pages} - rendering...", end=" ", flush=True)

        img_b64 = _image_to_base64(page)
        print("asking model...", end=" ", flush=True)
        heats = extract_heat_numbers(img_b64, client, model)

        stem = _safe_filename(heats, page_idx)
        out_path = output_dir / f"{stem}.pdf"

        # Overwrite any existing file with the same name
        single = fitz.open()
        single.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        single.save(str(out_path))
        single.close()

        heat_display = ", ".join(heats) if heats else "None Found"
        print(f"Heats=[{heat_display}] -> {out_path.name}")
        written.append(out_path)

    doc.close()
    return written


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split Yazici Demir Celik mill certificate PDFs by page, "
            "renaming each page file by its heat numbers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="PDF",
        help="one or more PDF paths (or global patterns) to process",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="directory to write the per-page PDFs files into.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Vision model via GitHub Models API (default: {DEFAULT_MODEL})",
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
                print(f"Warning: no matches found for {pattern!r}", file=sys.stderr)

    if not pdf_paths:
        print("Error: no valid PDF paths provided.", file=sys.stderr)
        return 1

    client = _make_client()
    all_written: List[Path] = []

    for pdf_path in pdf_paths:
        written = split_pdf(pdf_path, args.output, client, args.model)
        all_written.extend(written)

    print(
        f"\nDone! Processed {len(pdf_paths)} PDFs, wrote {len(all_written)} page files to {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
