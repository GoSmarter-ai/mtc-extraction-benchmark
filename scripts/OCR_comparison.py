import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from OCR.paddle_ocr import PaddleOCRExtractor
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")

try:
    from OCR.pdf_utils import convert_pdf_to_images
    PDF_UTILS_AVAILABLE = True
except ImportError:
    PDF_UTILS_AVAILABLE = False
    print("pdf2image not available. Install with: pip install pdf2image")

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Pytesseract not available. Install with: pip install pytesseract pillow")


def run_tesseract(image_path):
    """Run pytesseract on an image"""
    print(f"\nRunning Pytesseract on {Path(image_path).name}...")
    
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    results = []
    confidences = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = data['conf'][i]
        if text and conf > 0:
            results.append({
                'text': text,
                'confidence': conf / 100,  # Normalize to 0-1
            })
            confidences.append(conf / 100)
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    high_conf = len([c for c in confidences if c > 0.9])
    
    print(f"Pytesseract: {len(results)} regions, avg confidence: {avg_conf:.2%}")
    
    return {
        'engine': 'Pytesseract',
        'regions_detected': len(results),
        'avg_confidence': avg_conf,
        'high_confidence_count': high_conf,
        'high_confidence_pct': high_conf / len(results) if results else 0,
        'results': results
    }


def run_paddleocr(image_path):
    """Run PaddleOCR on an image"""
    print(f"\nRunning PaddleOCR on {Path(image_path).name}...")
    
    extractor = PaddleOCRExtractor(show_log=False)
    extracted = extractor.extract_from_image(image_path)
    
    print(f"PaddleOCR: {extracted['metadata']['total_regions']} regions, "
          f"avg confidence: {extracted['metadata']['avg_confidence']:.2%}")
    
    return {
        'engine': 'PaddleOCR',
        'regions_detected': extracted['metadata']['total_regions'],
        'avg_confidence': extracted['metadata']['avg_confidence'],
        'high_confidence_count': extracted['metadata']['high_confidence_count'],
        'high_confidence_pct': extracted['metadata']['high_confidence_count'] / extracted['metadata']['total_regions'] 
                               if extracted['metadata']['total_regions'] > 0 else 0,
        'results': extracted['text_regions']
    }


def compare_engines(tesseract_results, paddle_results):
    """Print comparison between engines"""
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Pytesseract':<20} {'PaddleOCR':<20}")
    print("-"*70)
    
    # Regions detected
    t_regions = tesseract_results['regions_detected']
    p_regions = paddle_results['regions_detected']
    diff_regions = p_regions - t_regions
    print(f"{'Text Regions Detected':<30} {t_regions:<20} {p_regions:<20} ({diff_regions:+d})")
    
    # Average confidence
    t_conf = tesseract_results['avg_confidence']
    p_conf = paddle_results['avg_confidence']
    diff_conf = (p_conf - t_conf) * 100
    print(f"{'Average Confidence':<30} {t_conf:<20.2%} {p_conf:<20.2%} ({diff_conf:+.1f}%)")
    
    # High confidence regions
    t_high = tesseract_results['high_confidence_count']
    p_high = paddle_results['high_confidence_count']
    t_high_pct = tesseract_results['high_confidence_pct']
    p_high_pct = paddle_results['high_confidence_pct']
    print(f"{'High Conf Regions (>90%)':<30} {t_high} ({t_high_pct:.1%}){'':<8} {p_high} ({p_high_pct:.1%})")
    
    print("\n" + "="*70)
    
    # Recommendation
    print("\nRECOMMENDATION:")
    if p_conf > t_conf and p_regions >= t_regions:
        print(f"PaddleOCR shows better performance:")
        print(f"{diff_conf:+.1f}% higher average confidence")
        print(f"{diff_regions:+d} more text regions detected")
        print(f"{p_high_pct:.1%} high-confidence extractions vs {t_high_pct:.1%}")
        print(f"\n   Recommend using PaddleOCR as the baseline OCR engine.")
    elif t_conf > p_conf:
        print(f"Pytesseract performed better on this sample.")
        print(f"Consider testing on more documents before deciding.")
    else:
        print(f"Both engines performed similarly.")
        print(f"PaddleOCR may still be better for complex layouts and tables.")


def main():
    parser = argparse.ArgumentParser(description='Compare OCR engines on MTC documents')
    parser.add_argument('--input', '-i', required=True, help='Path to MTC image or PDF to test')
    parser.add_argument('--output', '-o', default='data/comparison', help='Output directory for results')
    args = parser.parse_args()
    
    # Validate input exists
    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return
    
    print("="*70)
    print("OCR ENGINE COMPARISON TEST")
    print("="*70)
    print(f"\nTesting on: {args.input}")
    
    # Handle PDF files
    input_path = args.input
    is_pdf = Path(args.input).suffix.lower() == '.pdf'
    temp_images = []
    
    if is_pdf:
        if not PDF_UTILS_AVAILABLE:
            print("\nPDF support not available!")
            print("Install with: pip install pdf2image")
            print("\nOn Linux, you may also need: sudo apt-get install poppler-utils")
            print("On Mac: brew install poppler")
            return
        
        print("\n📄 PDF detected - converting to images...")
        temp_dir = os.path.join(args.output, 'temp_images')
        temp_images = convert_pdf_to_images(args.input, temp_dir)
        
        if not temp_images:
            print("Failed to convert PDF")
            return
        
        print(f"Converted to {len(temp_images)} images")
        
        # Use first page for comparison
        input_path = temp_images[0]
        print(f"Testing on first page: {Path(input_path).name}")
    
    # Check availability
    if not TESSERACT_AVAILABLE and not PADDLE_AVAILABLE:
        print("\nNo OCR engines available!")
        print("Install at least one:")
        print("  - Pytesseract: pip install pytesseract pillow")
        print("  - PaddleOCR: pip install paddlepaddle paddleocr")
        return
    
    results = {}
    
    # Run available engines
    if TESSERACT_AVAILABLE:
        try:
            results['tesseract'] = run_tesseract(input_path)
        except Exception as e:
            print(f"Pytesseract failed: {e}")
    
    if PADDLE_AVAILABLE:
        try:
            results['paddle'] = run_paddleocr(input_path)
        except Exception as e:
            print(f"PaddleOCR failed: {e}")
    
    # Compare if both ran
    if 'tesseract' in results and 'paddle' in results:
        compare_engines(results['tesseract'], results['paddle'])
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, 'comparison_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to: {output_file}")
    
    if is_pdf:
        print(f"\nNote: Tested only the first page. Full PDF has {len(temp_images)} pages.")
        print(f"   Temp images saved in: {os.path.dirname(temp_images[0])}")
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()