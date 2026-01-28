import os
from pathlib import Path
from typing import List, Optional
import logging

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not installed. Run: pip install pdf2image")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> List[str]:
    """
    Convert PDF pages to images
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save images
        dpi: Resolution (higher = better quality, default 300)
        
    Returns:
        List of paths to saved images
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image not installed. Install with: pip install pdf2image")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Converting PDF: {pdf_path}")
    
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)
    
    # Save images
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_name = Path(pdf_path).stem
    saved_paths = []
    
    for i, image in enumerate(images, start=1):
        output_path = os.path.join(output_dir, f'{pdf_name}_page_{i}.png')
        image.save(output_path, 'PNG')
        saved_paths.append(output_path)
        logger.info(f"Saved page {i}: {output_path}")
    
    logger.info(f"Converted {len(saved_paths)} pages from {pdf_path}")
    
    return saved_paths


def get_pdf_page_count(pdf_path: str) -> int:
    """Get number of pages in PDF"""
    if PYPDF2_AVAILABLE:
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            return len(pdf.pages)
    else:
        # Fallback: convert and count
        images = convert_from_path(pdf_path, dpi=72)  # Low res just for counting
        return len(images)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("PDF Utils ready!")
    print("\nUsage:")
    print("  from src.OCR.pdf_utils import convert_pdf_to_images")
    print("  images = convert_pdf_to_images('data/raw/dataset/certificate.pdf', 'data/processed/images')")