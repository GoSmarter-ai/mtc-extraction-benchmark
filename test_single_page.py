#!/usr/bin/env python3
"""Quick test on a single page image"""
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2

print("Initializing PaddleOCR...")
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

print("Loading image...")
img_path = "/workspaces/mtc-extraction-benchmark/data/processed/images/page_1.png"
pil_img = Image.open(img_path)
img_array = np.array(pil_img)
img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

print("Running OCR...")
results = list(ocr.predict(img_bgr))

print(f"Number of results: {len(results)}")
if results and len(results) > 0:
    result = results[0]
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys())}")
    print(f"rec_texts length: {len(result.get('rec_texts', []))}")
    print(f"rec_scores length: {len(result.get('rec_scores', []))}")
    print(f"rec_polys length: {len(result.get('rec_polys', []))}")
    print(f"dt_polys length: {len(result.get('dt_polys', []))}")
    
    if result.get('rec_texts'):
        print(f"\nFirst 5 texts:")
        for i, text in enumerate(result['rec_texts'][:5]):
            print(f"  {i+1}: {text}")
