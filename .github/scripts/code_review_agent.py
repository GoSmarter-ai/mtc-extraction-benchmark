#!/usr/bin/env python3
"""
AI Code Review Agent for MTC Extraction Benchmark.

Runs weekly (Monday) and creates a GitHub issue with prioritised findings,
rationales, and links to further reading — acting as a senior AI engineer
and cloud architect mentoring the contributors.

Uses the GitHub Copilot API (https://api.githubcopilot.com) which provides
access to the latest models (Claude, GPT-4o, etc.) with large context windows
and no restrictive free-tier token limits.
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
# Dynamic file discovery
# ---------------------------------------------------------------------------

# Directories to scan recursively for reviewable source files
SCAN_DIRS = [
    "src",
    "tests",
    "docs",
    "schema",
    "prompts",
    "scripts",
    ".github/workflows",
    ".github/scripts",
]

# Individual root-level files always included when present
ROOT_FILES = [
    "README.md",
    "project-plan.md",
    "QUICKSTART.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-test.txt",
    "Dockerfile",
    ".gitignore",
    "benchmark_report.qmd",
    "report.qmd",
]

# Text-based extensions eligible for review
INCLUDE_EXTENSIONS = {".py", ".md", ".yml", ".yaml", ".json", ".txt", ".qmd", ".sh"}

# Explicitly excluded extensions — PDFs and all binary/generated formats
EXCLUDE_EXTENSIONS = {
    ".pdf",
    ".html",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".woff",
    ".css",
    ".js",
    ".bak",
    ".ipynb",
    ".woff2",
    ".ico",
    ".svg",
}

# Directory names skipped during recursive scan
SKIP_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    "data",          # raw/processed certificate data — not source code
    "report_files",  # generated Quarto HTML assets
    "Misc",          # ad-hoc scratch files
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".venv",
    "venv",
}

# GitHub Copilot API supports large context windows (128K–200K tokens depending
# on the model), so we apply generous limits rather than the tight ones required
# by the GitHub Models free-tier playground.
MAX_FILE_CHARS = 5_000    # chars per file before truncation
MAX_CONTEXT_CHARS = 100_000  # total chars across all files (~28K tokens)


def _is_likely_binary(path: Path) -> bool:
    """Return True if the file appears to be binary (quick heuristic)."""
    try:
        chunk = path.read_bytes()[:512]
    except OSError:
        return True
    return b"\x00" in chunk


def discover_files() -> list[Path]:
    """
    Discover all reviewable files in the repository.

    Returns an ordered list: root config files first, then scanned
    directories in definition order, files sorted alphabetically within
    each directory.
    """
    seen: set[Path] = set()
    files: list[Path] = []

    def add(path: Path, *, bypass_extension_check: bool = False) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        if path.suffix.lower() in EXCLUDE_EXTENSIONS:
            return
        if not bypass_extension_check and path.suffix.lower() not in INCLUDE_EXTENSIONS:
            return
        if _is_likely_binary(path):
            return
        seen.add(resolved)
        files.append(path)

    # 1. Root-level files (explicit, in declared order) — bypass extension check
    #    because files like Dockerfile, .gitignore, pyproject.toml are always
    #    useful for review regardless of extension.
    for rel in ROOT_FILES:
        p = REPO_ROOT / rel
        if p.exists() and p.is_file():
            add(p, bypass_extension_check=True)

    # 2. Scanned directories (recursive, alphabetical within each dir)
    for dir_rel in SCAN_DIRS:
        dir_path = REPO_ROOT / dir_rel
        if not dir_path.is_dir():
            continue
        for path in sorted(dir_path.rglob("*")):
            if not path.is_file():
                continue
            # Skip if any ancestor directory is in SKIP_DIRS
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            add(path)

    return files


def read_file(path: Path) -> str:
    """Return the full contents of a file as text."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return f"[Could not read file: {exc}]"


def collect_context() -> str:
    """Assemble the repository context to send to the model.

    Each file is truncated to MAX_FILE_CHARS characters and the total
    context is capped at MAX_CONTEXT_CHARS to stay within a reasonable
    budget for the GitHub Copilot API.
    """
    files = discover_files()
    parts: list[str] = []
    total_chars = 0
    included = 0
    skipped_budget = 0

    for path in files:
        rel = path.relative_to(REPO_ROOT)
        content = read_file(path)

        # Truncate individual files that are too long.
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + f"\n... [truncated at {MAX_FILE_CHARS} chars]"

        chunk = f"### {rel}\n```\n{content}\n```"

        # Stop adding files once we would exceed the total budget.
        if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
            skipped_budget += 1
            continue

        parts.append(chunk)
        total_chars += len(chunk)
        included += 1

    context = "\n\n".join(parts)

    _CHARS_PER_TOKEN = 3.5
    estimated_tokens = len(context) / _CHARS_PER_TOKEN
    print(
        f"   Included {included}/{len(files)} files, "
        f"~{estimated_tokens:,.0f} estimated tokens "
        f"({len(context):,} chars)"
    )
    if skipped_budget:
        print(
            f"   ℹ️  {skipped_budget} file(s) omitted to stay within the "
            f"{MAX_CONTEXT_CHARS:,}-char context budget."
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
# Model configuration
# ---------------------------------------------------------------------------

# GitHub Copilot API endpoint — provides access to the latest models with large
# context windows and no restrictive free-tier token limits.
COPILOT_BASE_URL = "https://api.githubcopilot.com"

# Preferred models in priority order — the first one that responds successfully
# is used.  Claude 3.5 Sonnet is optimised for code and supports 200K tokens;
# GPT-4o is the reliable fallback with 128K token context.
CANDIDATE_MODELS = [
    "claude-3.5-sonnet",  # Anthropic Claude 3.5 Sonnet — strong code analysis, 200K ctx
    "gpt-4o",             # OpenAI GPT-4o — reliable fallback, 128K ctx
    "gpt-4o-mini",        # Smaller / faster fallback
]

# Models that use the OpenAI "reasoning" API contract:
#   - system role → developer role
#   - temperature param not supported
#   - max_tokens → max_completion_tokens
# Retained for forward-compatibility if o-series models are added to
# CANDIDATE_MODELS in future.
REASONING_MODELS = {"o4-mini", "o3-mini", "o1", "o1-mini", "o3"}

# Token budget for the generated review.
# Reasoning models use max_completion_tokens; standard models use max_tokens.
# 16,384 gives enough headroom for 10–15 detailed findings with tables/links.
MAX_OUTPUT_TOKENS = 16_384


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

    client = OpenAI(base_url=COPILOT_BASE_URL, api_key=token)

    # Try each candidate model in priority order; use the first that succeeds.
    response = None
    used_model = None
    for model in CANDIDATE_MODELS:
        print(f"🤖 Trying model '{model}' via GitHub Copilot …")

        is_reasoning = model in REASONING_MODELS

        # Reasoning models use the "developer" role instead of "system".
        system_role = "developer" if is_reasoning else "system"
        messages = [
            {"role": system_role, "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(context, today)},
        ]

        # Reasoning models do not support `temperature` and use
        # `max_completion_tokens` instead of `max_tokens`.
        call_kwargs: dict = {"model": model, "messages": messages}
        if is_reasoning:
            call_kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
        else:
            call_kwargs["temperature"] = 0.3
            call_kwargs["max_tokens"] = MAX_OUTPUT_TOKENS

        try:
            response = client.chat.completions.create(**call_kwargs)
            used_model = model
            print(f"   ✅ Using model: {used_model}")
            break
        except Exception as exc:
            print(f"   ⚠️  Model '{model}' unavailable: {exc}")

    if response is None:
        raise RuntimeError(
            f"All candidate models failed: {CANDIDATE_MODELS}. "
            "Check GitHub Models availability."
        )

    review_body = response.choices[0].message.content.strip()
    print(f"   Review generated ({len(review_body):,} characters)")

    footer = (
        "\n\n---\n"
        f"*Automatically generated by the "
        f"[Weekly Code Review Agent]"
        f"(../../.github/workflows/code-review.yml) "
        f"on {today} using {used_model} via GitHub Copilot.*"
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
