#!/usr/bin/env python3
"""
AI Code Review Agent for MTC Extraction Benchmark.

Runs weekly (Monday) and creates a GitHub issue with prioritised findings,
rationales, and links to further reading — acting as a senior AI engineer
and cloud architect mentoring the contributors.
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Repository root (two levels up from .github/scripts/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Files included in the review context (ordered by importance)
# ---------------------------------------------------------------------------
CONTEXT_FILES = [
    # Project definition
    "README.md",
    "project-plan.md",
    "QUICKSTART.md",
    # Configuration / dependencies
    "pyproject.toml",
    "requirements.txt",
    "requirements-test.txt",
    # CI / DevOps
    ".github/workflows/ci.yml",
    # Schema
    "schema/mtc_extraction_schema_v1.json",
    # Source — evaluation
    "src/evaluation/evaluator.py",
    # Source — extraction
    "src/extraction/llm_models_extraction.py",
    "src/extraction/complete_pipeline.py",
    "src/extraction/hybrid_pipeline.py",
    "src/extraction/docling_extraction.py",
    "src/extraction/paddle_extraction.py",
    # Source — API
    "src/api/main.py",
    "src/api/models.py",
    "src/api/routes/extract.py",
    "src/api/routes/benchmark.py",
    # Tests
    "tests/test_api.py",
    "tests/test_smoke.py",
    # Selected docs
    "docs/implementation_phase1_to_4.md",
    "docs/cicd_and_workflow.md",
    "docs/LLM_approach.md",
    "docs/model_expansion_and_ci_automation.md",
    "docs/week2_ocr_report.md",
]

# Maximum characters read per file — keeps the context within model limits
MAX_FILE_CHARS = 10_000

# Approximate character-to-token ratio; used for a soft guard only
_CHARS_PER_TOKEN = 3.5
# Leave headroom for system prompt + response; gpt-4o supports 128 k tokens
MAX_CONTEXT_TOKENS = 90_000

# GitHub Models endpoint and model
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
REVIEW_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_file(path: Path, max_chars: int = MAX_FILE_CHARS) -> str:
    """Return file contents, truncating if necessary."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return f"[Could not read file: {exc}]"
    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n… [truncated — {len(content)} chars total]"
    return content


def collect_context() -> str:
    """Assemble the repository context that will be sent to the model."""
    parts: list[str] = []
    for rel in CONTEXT_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        content = read_file(path)
        parts.append(f"### {rel}\n```\n{content}\n```")

    context = "\n\n".join(parts)

    # Soft guard: warn if context is unexpectedly large
    estimated_tokens = len(context) / _CHARS_PER_TOKEN
    if estimated_tokens > MAX_CONTEXT_TOKENS:
        print(
            f"⚠️  Context is ~{estimated_tokens:,.0f} estimated tokens "
            f"(limit ~{MAX_CONTEXT_TOKENS:,}). "
            "Consider reducing MAX_FILE_CHARS or CONTEXT_FILES."
        )

    return context


def create_github_issue(title: str, body: str, repo: str, token: str) -> dict:
    """Create a GitHub issue via the REST API and return the response JSON."""
    url = f"https://api.github.com/repos/{repo}/issues"
    payload = json.dumps({"title": title, "body": body}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        raise RuntimeError(
            f"GitHub API returned HTTP {exc.code}: {body_bytes.decode(errors='replace')}"
        ) from exc


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior AI engineer and cloud architect with deep expertise in:
- Machine learning pipelines and MLOps best practices
- Document intelligence, OCR, and information extraction systems
- Python software engineering and clean-code principles
- Cloud-native architectures (Azure, AWS, GCP) and DevOps / CI/CD
- Open-source AI/ML frameworks (Hugging Face, LangChain, ONNX, etc.)

You are mentoring contributors to an open-source project called
"MTC Extraction Benchmark" — a reusable training and evaluation framework
for a document intelligence processor that extracts structured information
from Mill / Material Test Certificates (MTCs).

Your goal is to produce a thorough, educational code-review report that:
1. Evaluates progress against the 6-week project plan.
2. Identifies code-quality, architecture, and process improvements.
3. Prioritises each finding and explains WHY it matters.
4. Points contributors to curated resources so they can learn and grow.

OUTPUT FORMAT — write a GitHub issue body in Markdown:

## Executive Summary
2–3 sentences on overall status and the single most important theme.

## Progress Against Project Plan
A concise table or bullet list mapping each week's goals to "✅ Done /
🔄 In Progress / ❌ Not Started", followed by 2–3 sentences of commentary.

## Key Findings & Recommendations
Numbered, prioritised list. For each item use this exact structure:

**N. [Category] Title**
| | |
|---|---|
| **Priority** | 🔴 Critical / 🟡 High / 🟢 Medium / 🔵 Low |
| **Category** | Code Quality / Testing / Architecture / MLOps / Documentation / Security / Performance |

**Finding:** One paragraph describing the issue with specific file/line references where applicable.

**Why it matters:** The rationale — what breaks, slows down, or limits contributors if this is not addressed.

**Recommendation:** Concrete, actionable steps (use a short bullet list).

**Further reading:**
- [Title](URL)
- [Title](URL)

---

Aim for 10–15 findings total, ordered from highest to lowest priority.

## Suggested Next Steps This Week
Top 3–5 actions the team should tackle immediately, as a numbered list.

---

Tone: educational and encouraging — you are mentoring, not criticising.
Be specific (reference actual file names and code patterns you observed).
"""


def build_user_prompt(context: str, today: str) -> str:
    return f"""\
Please review the repository context below and produce your full code-review \
report as described in the system prompt.

Today's date: {today}

---

{context}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    issue_title = f"{today} code review"

    print("📚 Collecting repository context …")
    context = collect_context()
    print(f"   Context size: {len(context):,} characters")

    print(f"🤖 Calling {REVIEW_MODEL} via GitHub Models …")
    client = OpenAI(base_url=GITHUB_MODELS_BASE_URL, api_key=token)

    try:
        response = client.chat.completions.create(
            model=REVIEW_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(context, today)},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
    except Exception as exc:
        print(f"❌ AI model call failed: {exc}")
        raise

    review_body = response.choices[0].message.content.strip()
    print(f"   Review generated ({len(review_body):,} characters)")

    footer = (
        "\n\n---\n"
        f"*Automatically generated by the "
        f"[Weekly Code Review Agent]"
        f"(../../.github/workflows/code-review.yml) "
        f"on {today} using {REVIEW_MODEL} via GitHub Models.*"
    )
    issue_body = review_body + footer

    print(f"📝 Creating GitHub issue: '{issue_title}' …")
    try:
        result = create_github_issue(issue_title, issue_body, repo, token)
    except Exception as exc:
        print(f"❌ Failed to create issue: {exc}")
        raise
    print(f"✅ Issue created: {result.get('html_url', '(no URL returned)')}")


if __name__ == "__main__":
    main()
