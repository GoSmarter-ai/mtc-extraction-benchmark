#!/usr/bin/env python3
"""
Debug script to test PaddleOCR on existing images
"""
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

# Test with one of the existing images
test_image_path = "/workspaces/mtc-extraction-benchmark/data/processed/images/page_1.png"

print("Loading image...")
# Try different loading methods
pil_img = Image.open(test_image_path)
print(f"PIL Image size: {pil_img.size}")
print(f"PIL Image mode: {pil_img.mode}")

# Convert to numpy array
img_array = np.array(pil_img)
print(f"Numpy array shape: {img_array.shape}")
print(f"Numpy array dtype: {img_array.dtype}")

# Convert RGB to BGR for OpenCV
if len(img_array.shape) == 3 and img_array.shape[2] == 3:
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
else:
    img_bgr = img_array

print("\nInitializing PaddleOCR...")
# Test with correct parameters
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

print("\nRunning OCR with ocr() method...")
result1 = ocr.ocr(test_image_path)
print(f"Result from ocr() method: {type(result1)}")
if result1:
    print(f"Result length: {len(result1)}")
    if len(result1) > 0:
        print(f"First page result type: {type(result1[0])}")
        if result1[0]:
            print(f"First page result length: {len(result1[0])}")
            if len(result1[0]) > 0:
                print(f"Sample result structure:")
                print(f"  Type: {type(result1[0][0])}")
                print(f"  Content: {result1[0][0]}")
                print(f"\nFirst 5 results:")
                for i, item in enumerate(result1[0][:5]):
                    print(f"  {i}: {item}")

print("\n" + "="*60)
print("Testing with predict() method...")
result2 = ocr.predict(test_image_path)
print(f"Result from predict(): {type(result2)}")
print(f"Keys in result: {result2.keys() if isinstance(result2, dict) else 'Not a dict'}")
if isinstance(result2, dict):
    for key, value in result2.items():
        print(f"  {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")

