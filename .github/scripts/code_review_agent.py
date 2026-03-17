#!/usr/bin/env python3
"""
AI Code Review Agent for MTC Extraction Benchmark.

Runs weekly (Monday) and creates a GitHub issue with prioritised findings,
rationales, and links to further reading — acting as a senior AI engineer
and cloud architect mentoring the contributors.
"""

import json
import os
import subprocess
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
    "data",  # raw/processed certificate data — not source code
    "report_files",  # generated Quarto HTML assets
    "Misc",  # ad-hoc scratch files
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".venv",
    "venv",
}


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


def _git(*args: str, timeout: int = 30) -> str:
    """Run a git command and return stdout, or empty string on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def collect_git_diff_context() -> str:
    """
    Build a compact diff-based context covering the last 7 days of commits
    (or the last 5 commits if there were no recent ones).
    Returns a formatted string ready to embed in the user prompt.
    """
    # Count commits in last 7 days
    log_out = _git("log", "--since=7 days ago", "--oneline")
    recent_count = len([line for line in log_out.splitlines() if line.strip()])
    n = max(min(recent_count, 15), 3)  # at least 3, at most 15

    parts: list[str] = []

    stat = _git("diff", f"HEAD~{n}", "HEAD", "--stat")
    if stat.strip():
        parts.append(f"**Changed files (last {n} commits / ~7 days):**\n```\n{stat.strip()}\n```")

    diff = _git(
        "diff",
        f"HEAD~{n}",
        "HEAD",
        "--",
        "*.py",
        "*.yml",
        "*.yaml",
        "*.toml",
        "*.md",
        "*.txt",
    )
    if diff.strip():
        # Hard cap on diff size to stay within model token budget
        diff_capped = diff[:8_000] + ("\n… [diff truncated]" if len(diff) > 8_000 else "")
        parts.append(f"**Code diff:**\n```diff\n{diff_capped}\n```")

    return "\n\n".join(parts)


def collect_context() -> str:
    """
    Assemble a token-budget-aware repository context.

    Strategy (in priority order, stops when MAX_CONTEXT_CHARS is reached):
      1. A handful of key structural files (README, project-plan, pyproject.toml)
         each truncated to 1 500 chars.
      2. Git diff of the last 7 days — the most actionable signal for a
         weekly code review.
    """
    _CHARS_PER_TOKEN = 3.5
    budget = MAX_CONTEXT_CHARS
    parts: list[str] = []

    # ── 1. Key structural files (truncated) ─────────────────────────────────
    PRIORITY_FILES = ["README.md", "project-plan.md", "pyproject.toml"]
    FILE_CHAR_LIMIT = 1_500
    for rel in PRIORITY_FILES:
        p = REPO_ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if len(text) > FILE_CHAR_LIMIT:
            text = text[:FILE_CHAR_LIMIT] + "\n… [truncated]"
        entry = f"### {rel}\n```\n{text}\n```"
        if len(entry) > budget:
            break
        parts.append(entry)
        budget -= len(entry)

    # ── 2. Git diff (recent changes) ────────────────────────────────────────
    print("   📂 Collecting git diff for last 7 days …")
    diff_ctx = collect_git_diff_context()
    if diff_ctx and len(diff_ctx) <= budget:
        parts.append(f"### Recent changes\n{diff_ctx}")
        budget -= len(diff_ctx)
    elif diff_ctx:
        # Still include as much as fits
        truncated = diff_ctx[: budget - 50] + "\n… [truncated]"
        parts.append(f"### Recent changes\n{truncated}")
        budget = 0

    context = "\n\n".join(parts)
    estimated_tokens = len(context) / _CHARS_PER_TOKEN
    print(
        f"   Included {len(parts)} sections, "
        f"~{estimated_tokens:,.0f} estimated tokens "
        f"({len(context):,} chars)"
    )
    if estimated_tokens > 4_500:
        print(
            "⚠️  Context may still exceed model token limits. Consider reducing MAX_CONTEXT_CHARS."
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

# GitHub Models endpoint
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"

# Preferred models in priority order — the first one that responds successfully
# is used.  gpt-4o is listed first because the GitHub Models free tier grants
# it the highest per-request token budget (~8 K tokens) vs. ~4 K for the
# reasoning models.
CANDIDATE_MODELS = [
    "gpt-4o",  # Standard chat model — highest free-tier token budget
    "o3-mini",  # Efficient reasoning model, strong code analysis
    "o4-mini",  # OpenAI reasoning model, optimised for code
]

# Models that use the OpenAI "reasoning" API contract:
#   - system role → developer role
#   - temperature param not supported
#   - max_tokens → max_completion_tokens
REASONING_MODELS = {"o4-mini", "o3-mini", "o1", "o1-mini", "o3"}

# Hard character cap for the context passed to the model.  Derived from the
# tightest GitHub Models free-tier limit (gpt-4o ≈ 8 K total tokens):
#   8 000 tokens  −  ~450 system-prompt tokens  −  ~50 preamble tokens
#   − MAX_OUTPUT_TOKENS  ≈  5 500 input tokens  ×  3.5 chars/token ≈ 19 000 chars
# We keep a comfortable margin here.
MAX_CONTEXT_CHARS = 14_000

# Token budget for the generated review.
# Reasoning models use max_completion_tokens; standard models use max_tokens.
# Kept at 2 000 so the total request stays within GitHub Models free-tier
# limits (4 K–8 K tokens depending on model).
MAX_OUTPUT_TOKENS = 2_000


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

    client = OpenAI(base_url=GITHUB_MODELS_BASE_URL, api_key=token)

    # Try each candidate model in priority order; use the first that succeeds.
    response = None
    used_model = None
    for model in CANDIDATE_MODELS:
        print(f"🤖 Trying model '{model}' via GitHub Models …")

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
            f"All candidate models failed: {CANDIDATE_MODELS}. Check GitHub Models availability."
        )

    review_body = response.choices[0].message.content.strip()
    print(f"   Review generated ({len(review_body):,} characters)")

    footer = (
        "\n\n---\n"
        f"*Automatically generated by the "
        f"[Weekly Code Review Agent]"
        f"(../../.github/workflows/code-review.yml) "
        f"on {today} using {used_model} via GitHub Models.*"
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
