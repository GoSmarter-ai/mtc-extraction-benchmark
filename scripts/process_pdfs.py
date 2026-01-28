import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append('/workspaces/mtc-extraction-benchmark/src/ocr')

from src.OCR.pdf_utils import convert_pdf_to_images
from src.OCR.paddle_ocr import PaddleOCRExtractor


def main():
    print("="*70)
    print("PDF PROCESSING PIPELINE")
    print("="*70)
    
    # Convert all PDFs to images
    input_dir = '/workspaces/mtc-extraction-benchmark/data/raw/diler'
    image_dir = 'data/processed/images'
    
    # Find all PDFs
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    
    if not pdf_files:
        print(f"\nNo PDF files found in {input_dir}")
        print("Make sure your PDFs are in the correct directory!")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    # Convert PDFs to images
    print(f"\n🔄 Converting PDFs to images...")
    all_images = []
    
    for pdf_file in pdf_files:
        print(f"\n   Processing: {pdf_file.name}")
        try:
            images = convert_pdf_to_images(str(pdf_file), image_dir, dpi=300)
            all_images.extend(images)
            print(f"Converted {len(images)} pages")
        except Exception as e:
            print(f"Failed: {e}")
    
    if not all_images:
        print("\n No images were created!")
        return
    
    print(f"\nTotal: Converted {len(all_images)} pages to images")
    print(f"   Saved to: {image_dir}/")
    
    # Run OCR on all images
    print(f"\nRunning OCR extraction...")
    
    try:
        extractor = PaddleOCRExtractor()
        stats = extractor.process_directory(image_dir, 'data/processed')
        
        print(f"\nOCR COMPLETE!")
        print(f"   Processed: {stats['processed']} images")
        print(f"   Failed: {stats['failed']} images")
        print(f"   Total text regions extracted: {stats['total_regions']}")
        
        print(f"\n📁 Output files saved to:")
        print(f"   - data/processed/boxes/     (bounding box data)")
        print(f"   - data/processed/text/      (extracted text)")
        print(f"   - data/processed/metadata/  (confidence scores)")
        
    except Exception as e:
        print(f"\nOCR failed: {e}")
        print("You may need to install: pip install paddlepaddle paddleocr")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()