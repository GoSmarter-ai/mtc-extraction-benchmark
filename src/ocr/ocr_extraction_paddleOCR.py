import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import gc

def process_pdf_with_paddleocr(input_dir, output_dir):
    """
    Process PDF files using PaddleOCR and save results with bounding boxes.
    Memory-optimized version.
    
    Args:
        input_dir: Directory containing input PDF files
        output_dir: Directory to save output images and text files
    """
    # Check if directories exist
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist!")
        return
    
    # Initialize PaddleOCR with optimized settings
    print("Initializing PaddleOCR (this may take a moment)...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    print("PaddleOCR initialized!\n")
    
    # Get all PDF files from input directory
    input_path = Path(input_dir)
    pdf_files = list(input_path.glob('*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)\n")
    
    # Process each PDF
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        
        # Convert PDF to images with reduced DPI to save memory
        try:
            # Use 200 DPI instead of 300 to reduce memory usage
            print(f"  Converting PDF to images (DPI=200)...")
            images = convert_from_path(str(pdf_file), dpi=200)
            print(f"  Converted to {len(images)} page(s)")
        except Exception as e:
            print(f"  Error converting PDF: {e}")
            continue
        
        # Process each page
        for page_num, pil_image in enumerate(images, 1):
            print(f"  Processing page {page_num}/{len(images)}...")
            
            try:
                # Resize image if too large (max width 2000px to save memory)
                max_width = 2000
                if pil_image.width > max_width:
                    ratio = max_width / pil_image.width
                    new_height = int(pil_image.height * ratio)
                    pil_image = pil_image.resize((max_width, new_height), Image.LANCZOS)
                    print(f"    Resized image to {max_width}x{new_height}")
                
                # Convert PIL Image to numpy array for OpenCV
                img_array = np.array(pil_image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Clear PIL image to free memory
                del pil_image
                gc.collect()
                
                # Run OCR
                print(f"    Running OCR...")
                result = ocr.predict(img_array)
                print(f"    OCR complete!")
                
                # Create output filename
                base_name = pdf_file.stem
                page_suffix = f"_page{page_num}" if len(images) > 1 else ""
                
                # Draw bounding boxes and save image
                img_with_boxes = img_array.copy()
                text_output = []
                text_count = 0
                
                # Handle result
                if result and len(result) > 0 and result[0]:
                    ocr_result = result[0] if isinstance(result[0], list) else result
                    
                    for line in ocr_result:
                        try:
                            # Each line contains: [box_coordinates, (text, confidence)]
                            box = line[0]
                            text = line[1][0]
                            confidence = line[1][1]
                            
                            # Convert box coordinates to integer
                            box = np.array(box, dtype=np.int32)
                            
                            # Draw bounding box
                            cv2.polylines(img_with_boxes, [box], True, (0, 255, 0), 2)
                            
                            # Add text above the box (truncate if too long)
                            display_text = text[:30] if len(text) > 30 else text
                            cv2.putText(img_with_boxes, display_text, 
                                       (box[0][0], max(box[0][1] - 10, 20)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                            
                            # Store text with confidence
                            text_output.append(f"{text} (confidence: {confidence:.4f})")
                            text_count += 1
                            
                        except (IndexError, TypeError) as e:
                            print(f"    Warning: Skipping malformed result: {e}")
                            continue
                
                print(f"    Extracted {text_count} text blocks")
                
                # Save annotated image
                output_img_path = os.path.join(output_dir, f"{base_name}{page_suffix}_annotated.jpg")
                cv2.imwrite(output_img_path, img_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 85])
                print(f"    ✓ Saved: {base_name}{page_suffix}_annotated.jpg")
                
                # Save original image
                output_orig_path = os.path.join(output_dir, f"{base_name}{page_suffix}_original.jpg")
                cv2.imwrite(output_orig_path, img_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
                print(f"    ✓ Saved: {base_name}{page_suffix}_original.jpg")
                
                # Save extracted text
                output_txt_path = os.path.join(output_dir, f"{base_name}{page_suffix}_text.txt")
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"OCR Results for: {pdf_file.name} - Page {page_num}\n")
                    f.write("=" * 80 + "\n\n")
                    for text in text_output:
                        f.write(text + "\n")
                print(f"    ✓ Saved: {base_name}{page_suffix}_text.txt")
                
                # Clean up memory after each page
                del img_array, img_with_boxes, result, text_output
                gc.collect()
                
            except Exception as e:
                print(f"    Error processing page {page_num}: {e}")
                continue
        
        # Clean up after processing all pages of this PDF
        del images
        gc.collect()
        print(f"  ✓ Completed: {pdf_file.name}\n")
    
    print(f"✓ All processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    # Set your input and output directories
    INPUT_DIR = "/workspaces/mtc-extraction-benchmark/data/raw/diler"
    OUTPUT_DIR = "/workspaces/mtc-extraction-benchmark/data/processed"
    
    print("=" * 60)
    print("PaddleOCR PDF Processor (Memory Optimized)")
    print("=" * 60)
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60 + "\n")
    
    # Process PDFs
    process_pdf_with_paddleocr(INPUT_DIR, OUTPUT_DIR)