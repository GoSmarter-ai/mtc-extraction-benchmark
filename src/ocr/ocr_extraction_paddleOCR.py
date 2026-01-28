import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not installed. Run: pip install paddlepaddle paddleocr")


class PaddleOCRExtractor:
    def __init__(self, lang='en', use_angle_cls=True, show_log=False):

        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
        
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=show_log)
        self.logger = logging.getLogger(__name__)
    
    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.logger.info(f"Processing: {image_path}")
        
        # Run OCR
        result = self.ocr.ocr(image_path, cls=True)
        
        # Parse results
        extracted_data = {
            'source_file': os.path.basename(image_path),
            'text_regions': [],
            'full_text': [],
            'metadata': {
                'total_regions': 0,
                'avg_confidence': 0.0,
                'high_confidence_count': 0  # confidence > 0.9
            }
        }
        
        if not result or not result[0]:
            self.logger.warning(f"No text detected in {image_path}")
            return extracted_data
        
        confidences = []
        
        for line in result[0]:
            # line[0] = bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # line[1] = (text, confidence)
            bbox = line[0]
            text, confidence = line[1]
            
            # Store structured data
            region = {
                'text': text,
                'confidence': round(confidence, 4),
                'bbox': bbox,
                'bbox_simplified': {
                    'x': int(min(p[0] for p in bbox)),
                    'y': int(min(p[1] for p in bbox)),
                    'width': int(max(p[0] for p in bbox) - min(p[0] for p in bbox)),
                    'height': int(max(p[1] for p in bbox) - min(p[1] for p in bbox))
                }
            }
            
            extracted_data['text_regions'].append(region)
            extracted_data['full_text'].append(text)
            confidences.append(confidence)
            
            if confidence > 0.9:
                extracted_data['metadata']['high_confidence_count'] += 1
        
        # Calculate metadata
        extracted_data['metadata']['total_regions'] = len(extracted_data['text_regions'])
        if confidences:
            extracted_data['metadata']['avg_confidence'] = round(sum(confidences) / len(confidences), 4)
        
        self.logger.info(f"Extracted {len(extracted_data['text_regions'])} text regions "
                        f"(avg confidence: {extracted_data['metadata']['avg_confidence']:.2%})")
        
        return extracted_data
    
    def save_results(self, extracted_data: Dict[str, Any], output_dir: str, 
                     save_boxes: bool = True, save_text: bool = True) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        
        source_file = extracted_data['source_file']
        base_name = Path(source_file).stem
        
        saved_files = {}
        
        # Save bounding boxes as JSON
        if save_boxes:
            boxes_path = os.path.join(output_dir, 'boxes', f'{base_name}.json')
            os.makedirs(os.path.dirname(boxes_path), exist_ok=True)
            
            with open(boxes_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data['text_regions'], f, indent=2, ensure_ascii=False)
            
            saved_files['boxes'] = boxes_path
            self.logger.info(f"Saved boxes: {boxes_path}")
        
        # Save plain text
        if save_text:
            text_path = os.path.join(output_dir, 'text', f'{base_name}.txt')
            os.makedirs(os.path.dirname(text_path), exist_ok=True)
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(extracted_data['full_text']))
            
            saved_files['text'] = text_path
            self.logger.info(f"Saved text: {text_path}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata', f'{base_name}_meta.json')
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data['metadata'], f, indent=2)
        
        saved_files['metadata'] = metadata_path
        
        return saved_files
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         file_extensions: List[str] = ['.png', '.jpg', '.jpeg', '.pdf']) -> Dict[str, Any]:

        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
        
        if not image_files:
            self.logger.warning(f"No images found in {input_dir}")
            return {}
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = {
            'processed': 0,
            'failed': 0,
            'total_regions': 0,
            'files': []
        }
        
        for image_file in image_files:
            try:
                extracted = self.extract_from_image(str(image_file))
                self.save_results(extracted, output_dir)
                
                results['processed'] += 1
                results['total_regions'] += extracted['metadata']['total_regions']
                results['files'].append(str(image_file))
                
            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {e}")
                results['failed'] += 1
        
        self.logger.info(f"Processing complete: {results['processed']} successful, {results['failed']} failed")
        
        return results


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    extractor = PaddleOCRExtractor()
    
    # Example usage
    print("PaddleOCR Extractor initialized successfully!")
    print("\nUsage example:")
    print("  extractor = PaddleOCRExtractor()")
    print("  data = extractor.extract_from_image('data/raw/dataset/page_1.png')")
    print("  extractor.save_results(data, 'data/processed')")