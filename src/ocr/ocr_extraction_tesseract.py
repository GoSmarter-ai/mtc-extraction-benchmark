import os
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path

PDF_PATH = "/workspaces/mtc-extraction-benchmark/data/raw/diler/diler-07-07-2025-rerun-41-44.pdf"
OUTPUT_DIR = "/workspaces/mtc-extraction-benchmark/data/processed"
DPI = 300
TESSERACT_CONFIG = "--oem 3 --psm 6"

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/text", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/boxes", exist_ok=True)

pages = convert_from_path(PDF_PATH, dpi=DPI)

print(f"Converted {len(pages)} pages")

for i, page in enumerate(pages):
    img_path = f"{OUTPUT_DIR}/images/page_{i+1}.png"
    page.save(img_path, "PNG")

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(
        thresh,
        lang="eng",
        config=TESSERACT_CONFIG
    )

    with open(f"{OUTPUT_DIR}/text/page_{i+1}.txt", "w") as f:
        f.write(text)

    data = pytesseract.image_to_data(
        thresh,
        lang="eng",
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    boxes=[]
    for j in range(len(data["text"])):
        if int(data["conf"][j] > 0):
            boxes.append({
                "text": data["text"][j],
                "conf": data["conf"][j],
                "x": data["left"][j],
                "y": data["top"][j],
                "w": data["width"][j],
                "h": data["height"][j],
            })

    with open(f"{OUTPUT_DIR}/boxes/page_{i+1}.json", "w") as f:
        import json
        json.dump(boxes, f, indent=2)

    print(f"Page {i+1}: OCR Complete")

print("Baseline OCR Pipeline finished.")