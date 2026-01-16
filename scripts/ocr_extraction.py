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

