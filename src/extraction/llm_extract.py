import json
import os
from pathlib import Path
from openai import OpenAI

TEXT_PATH = Path(
    "/workspaces/mtc-extraction-benchmark/data/processed/paddle_ocr/diler-07-07-2025-rerun-41-44_page1_text.txt"
)
SCHEMA_PATH = Path(
    "/workspaces/mtc-extraction-benchmark/schema/mtc_extraction_schema_v1.json"
)
PROMPT_PATH = Path(
    "/workspaces/mtc-extraction-benchmark/prompts/mtc_llm_extraction_prompt.txt"
)
OUTPUT_PATH = Path(
    "/workspaces/mtc-extraction-benchmark/data/processed/schema_output/certificate_001.json"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

ocr_text = TEXT_PATH.read_text(errors="ignore")
schema = json.loads(SCHEMA_PATH.read_text())
system_prompt = PROMPT_PATH.read_text()

user_prompt = f"""
SCHEMA:
{json.dumps(schema, indent=2)}

OCR TEXT:
\"\"\"
{ocr_text}
\"\"\"
"""

# Using GitHub Models with free open-source LLM (Meta Llama 3.1)
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

response = client.chat.completions.create(
    model="Meta-Llama-3.1-405B-Instruct",  # Using powerful open-source Llama 3.1 405B model
    temperature=0,
    max_tokens=4096,  # Ensure sufficient tokens for complete JSON output
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)

raw_output = response.choices[0].message.content.strip()

print(f"Raw LLM output (first 500 chars):\n{raw_output[:500]}\n")

try:

    if "```json" in raw_output:
        json_start = raw_output.find("```json") + 7
        json_end = raw_output.find("```", json_start)
        raw_output = raw_output[json_start:json_end].strip()
    elif "```" in raw_output:
        json_start = raw_output.find("```") + 3
        json_end = raw_output.find("```", json_start)
        raw_output = raw_output[json_start:json_end].strip()

    parsed = json.loads(raw_output)
except json.JSONDecodeError as e:
    print(f"JSONDecodeError: {e}")
    print(f"Full output:\n{raw_output}")
    raise ValueError("LLM output is not valid JSON")

with open(OUTPUT_PATH, "w") as f:
    json.dump(parsed, f, indent=2)

print("LLM extraction completed.")
