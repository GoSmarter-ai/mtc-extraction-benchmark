import os
import json
from pathlib import Path

# Directories
paddle_dir = "/workspaces/mtc-extraction-benchmark/data/processed/paddle_ocr"
output_dir = "/workspaces/mtc-extraction-benchmark/data/processed/paddle_combined"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Find all text files
text_files = sorted([f for f in os.listdir(paddle_dir) if f.endswith("_text.txt")])

print(f"Found {len(text_files)} PaddleOCR text files\n")

# Process each page
all_pages_text = []
all_pages_data = []

for text_file in text_files:
    file_path = os.path.join(paddle_dir, text_file)
    page_num = text_file.split("_page")[1].split("_")[0]

    print(f"Processing page {page_num}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the content to extract text and confidence scores
    lines = content.split("\n")
    page_data = {
        "page": int(page_num),
        "file": text_file,
        "text_lines": [],
        "full_text": [],
    }

    # Skip header lines
    in_content = False
    for line in lines:
        if "=" * 20 in line:
            in_content = True
            continue
        if in_content and line.strip():
            # Extract text and confidence if present
            if "(confidence:" in line:
                text_part = line.split("(confidence:")[0].strip()
                conf_part = line.split("(confidence:")[1].strip(")")
                page_data["text_lines"].append(
                    {"text": text_part, "confidence": float(conf_part)}
                )
                page_data["full_text"].append(text_part)
            else:
                # Lines without confidence scores
                if line.strip():
                    page_data["text_lines"].append(
                        {"text": line.strip(), "confidence": None}
                    )
                    page_data["full_text"].append(line.strip())

    all_pages_data.append(page_data)
    all_pages_text.extend(page_data["full_text"])

# Save combined results
output_json = os.path.join(output_dir, "diler-07-07-2025-rerun-41-44_combined.json")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_pages_data, f, indent=2, ensure_ascii=False)

# Save as simple markdown
output_md = os.path.join(output_dir, "diler-07-07-2025-rerun-41-44_combined.md")
with open(output_md, "w", encoding="utf-8") as f:
    f.write("# Material Test Certificate - Combined OCR Results\n\n")
    f.write("**Source:** PaddleOCR Pre-extracted Results\n\n")
    f.write("---\n\n")
    for page_data in all_pages_data:
        f.write(f"## Page {page_data['page']}\n\n")
        f.write("\n".join(page_data["full_text"]))
        f.write("\n\n")

# Save as plain text
output_txt = os.path.join(output_dir, "diler-07-07-2025-rerun-41-44_combined.txt")
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(all_pages_text))

print(f"\nCombined {len(all_pages_data)} pages")
print(f"JSON: {output_json}")
print(f"Markdown: {output_md}")
print(f"Text: {output_txt}")
print(
    f"\nTotal text lines extracted: {sum(len(p['text_lines']) for p in all_pages_data)}"
)
