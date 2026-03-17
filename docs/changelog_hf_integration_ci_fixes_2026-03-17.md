# Changes Log — March 17, 2026

This document details all changes committed on March 17, 2026, covering nine commits across three areas: the CI/CD pipeline, the Quarto benchmark report, the LLM extraction benchmark script, and the AI code review agent.

---

## Summary of Commits (Chronological)

| Time (UTC) | Commit | Scope | Type |
|---|---|---|---|
| 08:30 | `b54ca9f` | Code Review Agent | fix |
| 08:39 | `9c91228` | CI Pipeline | fix |
| 08:45 | `fd01d5f` | Benchmark Script | fix |
| 08:50 | `c3b155c` | Quarto Report | fix |
| 09:00 | `0543b35` | Quarto Report | fix |
| 09:09 | `6a5dd8b` | CI Pipeline | ci |
| 14:07 | `1dc5011` | Benchmark Script | fix |
| 14:23 | `7cafbc6` | Benchmark Script | feat |
| 14:35 | `1c4fefc` | Benchmark Script + Results | feat |

---

## 1. Fix: Code Review Agent — Token Limit Overrun (`b54ca9f`)

**File changed:** `.github/scripts/code_review_agent.py`

### Problem
The weekly AI code review agent was failing for all models with HTTP `413 tokens_limit_reached`. The root cause was that the agent was dumping every file in the entire repository (~141 K tokens) as context, far exceeding the GitHub Models free-tier budget (max ~8 K tokens for GPT-4o).

### What Changed

- **Replaced `collect_context()` (full-repo dump)** with a two-step budget-aware function:
  1. **Key structural files first**: `README.md`, `project-plan.md`, and `pyproject.toml` are loaded, each truncated to 1,500 characters.
  2. **Git diff context**: A new helper `collect_git_diff_context()` uses `git diff HEAD~N HEAD` to produce a focused diff covering the last 7 days of commits (3–15 commits), capped at 8,000 characters.
- **`MAX_CONTEXT_CHARS = 14,000`** hard cap added — the total context sent to any model will never exceed this.
- **`MAX_OUTPUT_TOKENS` reduced** from 16,384 → 2,000, which is sufficient for a code review summary.
- **`CANDIDATE_MODELS` reordered**: `gpt-4o` is now first (highest 8 K free-tier budget), followed by `o3-mini`, then `o4-mini`.
- **`import subprocess` added** to support the `_git()` helper that shells out to `git`.

### Why This Matters
The code review action was completely non-functional before this fix. By switching from a static file dump to a targeted diff, the context is both smaller and more actionable — the model sees exactly what changed recently rather than the entire codebase.

---

## 2. Fix: CI — Missing Jupyter Dependencies for Quarto (`9c91228`)

**File changed:** `.github/workflows/ci.yml`

### Problem
The `quarto render benchmark_report.qmd` step was failing in CI with:
```
ModuleNotFoundError: No module named 'nbformat'
```
Quarto requires `nbformat`, `nbclient`, and `ipykernel` to execute Python code cells inside `.qmd` files, but these were not installed before the render step.

### What Changed
A new step was inserted immediately before `quarto render`:

```yaml
- name: Install Jupyter for Quarto
  run: pip install jupyter nbformat nbclient ipykernel
```

### Why This Matters
Without these packages, the benchmark report could never be rendered in CI. This step ensures the full rendering pipeline works end-to-end.

---

## 3. Fix: Benchmark Script — Cached Results Shown as FAIL (`fd01d5f`)

**File changed:** `src/extraction/llm_models_extraction.py`

### Problem
When a model's output was loaded from a previously cached JSON file (via `--skip-existing`), the benchmark system assigned it `status='cached'`. However, the `_print_table()` method only treated `status='success'` as a passing result, causing every cached model to display `❌ FAIL` in the summary table — even though the extraction had actually succeeded.

### What Changed
- `_print_table()` now treats `status == 'cached'` the same as `status == 'success'` (i.e., the model is counted as having run successfully).
- Cached runs display with label `⏭️ CACHE` instead of `✅ OK` so they are visually distinguishable, but no longer shown as failures.

---

## 4. Fix: Quarto Report — `status='cached'` Excluded from Score Columns (`c3b155c`)

**File changed:** `benchmark_report.qmd`

### Problem
In the `build-scores` code cell inside the Quarto report, a filter of `status != 'success'` was used to exclude failed runs. This accidentally excluded all cached runs too (since their status was `'cached'`), resulting in an empty DataFrame. Downstream code then raised a `KeyError` on `'doc_accuracy'` because the expected columns did not exist.

### What Changed
The filter was updated to pass through both `'success'` and `'cached'` statuses:

```python
# Before
df = df[df['status'] == 'success']

# After
df = df[df['status'].isin(['success', 'cached'])]
```

---

## 5. Fix: Quarto Report — Multiple Rendering Errors Resolved (`0543b35`)

**File changed:** `benchmark_report.qmd` (40 insertions, 34 deletions)

### Problems Fixed
Four separate rendering errors were resolved in a single commit:

| Error | Root Cause | Fix |
|---|---|---|
| `AttributeError: 'Styler' has no attribute 'applymap'` | Pandas 2.1 renamed `Styler.applymap` to `Styler.map` | Replaced all `applymap` calls with `map` |
| `IndexError` in latency scatter chart | `df_c.index.get_loc(model_name)` was fragile when the DataFrame index was not a simple integer range | Replaced with `enumerate()` for safe positional bubble-size indexing |
| `KeyError` in registry table | When all models failed, the DataFrame was empty and column access raised a `KeyError` | Added a guard (`if df.empty: ...`) before accessing columns |
| `ValueError` in doc-field detail heatmap | Empty DataFrame being passed to `pd.pivot_table()` raised an error | Added a guard before the pivot call |

### Why This Matters
The Quarto report is the primary deliverable of the benchmark pipeline. All four of these errors caused the HTML report to fail to render. After this fix the report renders cleanly under all conditions, including when some or all models fail.

---

## 6. CI: Split Benchmark Artifacts into Two Named Uploads (`6a5dd8b`)

**File changed:** `.github/workflows/ci.yml`

### What Changed
The single `upload-artifact` step at the end of the benchmark job was split into two separate uploads:

| Artifact | Contents | Retention |
|---|---|---|
| `benchmark-results-<sha>` | All JSON files in `data/processed/benchmark_output/` | 90 days |
| `benchmark-report-<sha>` | `benchmark_report.html` | 90 days |

Additionally, `continue-on-error: true` was **removed** from the `quarto render` step. Previously, render failures were silently swallowed and the job appeared green. Now a render failure causes a visible job failure.

### Why This Matters
Having the HTML report as a standalone artifact makes it easy to download and view in a browser directly from the GitHub Actions run page, without having to extract it from the larger JSON bundle.

---

## 7. Fix: HuggingFace Base URL Deprecated (`1dc5011`)

**File changed:** `src/extraction/llm_models_extraction.py`

### Problem
All HuggingFace model calls were failing because the endpoint `https://api-inference.huggingface.co/v1` was deprecated by HuggingFace.

### What Changed
```python
# Before
_HF_BASE = "https://api-inference.huggingface.co/v1"

# After
_HF_BASE = "https://router.huggingface.co/v1"
```

This single-line change restores connectivity to all HuggingFace models. A side effect of the fix: the first HuggingFace model extraction (`Qwen/Qwen2.5-7B-Instruct`) succeeded and its output was committed alongside this change.

---

## 8. Feat: Expand HuggingFace Model List, Drop 72B Due to Timeout (`7cafbc6`)

**File changed:** `src/extraction/llm_models_extraction.py`

### What Changed
The `HF_MODELS` list was restructured:

**Removed:**
- `Qwen/Qwen2.5-72B-Instruct` (72B) — caused consistent timeouts on the HuggingFace free tier

**Added (three new models):**
| Model ID | Label | Provider |
|---|---|---|
| `microsoft/Phi-3.5-mini-instruct` | Phi-3.5 Mini | Microsoft |
| `google/gemma-2-9b-it` | Gemma 2 9B | Google |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | DeepSeek-R1 Distill 8B | DeepSeek |

All three new models are assigned `"tier": "small"` and `"max_tokens": 8192`.

**Per-model `max_tokens` override:**
The `extract_with_model()` method was updated to read `max_tokens` from the model's registry entry rather than always using the global default:

```python
# Before — always used self.max_tokens (16,384)
response = client.chat.completions.create(
    model=model_id,
    max_tokens=self.max_tokens,
    ...
)

# After — respects per-model cap
effective_max_tokens = self._model_registry.get(model_id, {}).get(
    "max_tokens", self.max_tokens
)
response = client.chat.completions.create(
    model=model_id,
    max_tokens=effective_max_tokens,
    ...
)
```

This is important for HF models that enforce stricter output token limits than GitHub Models.

---

## 9. Feat: Add HuggingFace Benchmark Results & Update Report (`1c4fefc`)

**Files changed:**
- `src/extraction/llm_models_extraction.py`
- `data/processed/benchmark_output/benchmark_summary.json`
- `data/processed/benchmark_output/meta-llama_Llama-3_1-8B-Instruct_extracted.json` *(new)*
- `benchmark_report.html`

### What Changed

**Model list finalised:**
After testing `Phi-3.5-mini`, `Gemma-2-9B`, and `DeepSeek-R1-Distill-8B` and finding them either timing out or returning non-JSON responses, the HF Models list was consolidated to two working models:

| Model | Status |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | ✅ Working |
| `meta-llama/Llama-3.1-8B-Instruct` | ✅ Working (new) |

**Reasoning model output stripping:**
DeepSeek-R1 and other "thinking" models wrap their reasoning traces in `<think>…</think>` tags before the actual JSON answer. These blocks were being parsed as JSON, causing extraction failures. A stripping step was added:

```python
# Strip DeepSeek-R1 / reasoning model <think>...</think> blocks
if "<think>" in raw:
    think_end = raw.find("</think>")
    if think_end != -1:
        raw = raw[think_end + 8 :].strip()
```

**Benchmark results updated:**
- `benchmark_summary.json` updated with results for all models including the newly working HF models
- `meta-llama_Llama-3_1-8B-Instruct_extracted.json` committed as a new extraction output
- `benchmark_report.html` regenerated with the full updated benchmark summary

---

## Overall Impact Summary

| Area | Changes | Net Effect |
|---|---|---|
| **CI pipeline** | Jupyter deps installed; artifact split; render errors are now fatal | Reliable end-to-end CI with downloadable report |
| **Quarto report** | 4 rendering bugs fixed; cached-status handling corrected | Report renders under all conditions |
| **HF integration** | Base URL updated; model list expanded then stabilised | `Qwen2.5-7B` and `Llama-3.1-8B` via HuggingFace now working |
| **Extraction script** | DeepSeek `<think>` stripping; per-model `max_tokens`; cached-results display | More models produce usable JSON; table output is accurate |
| **Code review agent** | Context reduced from ~141 K → ~14 K tokens; diff-focused context | Weekly review action functional again |
