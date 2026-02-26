import gc
import os
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image


def process_pdf_with_paddleocr(input_dir, output_dir):
    # Check if directories exist
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return

    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist!")
        return

    # Initialize PaddleOCR with optimized settings
    print("Initializing PaddleOCR (this may take a moment)...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="en")
    print("PaddleOCR initialized!\n")

    # Get all PDF files from input directory
    input_path = Path(input_dir)
    pdf_files = list(input_path.glob("*.pdf"))

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
            print("  Converting PDF to images (DPI=200)...")
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
                print("    Running OCR...")
                # predict() returns a generator, need to convert to list
                results = list(ocr.predict(img_array))
                print("    OCR complete!")
                print(f"    DEBUG: Number of results: {len(results)}")
                if results and len(results) > 0:
                    print(f"    DEBUG: Result type: {type(results[0])}")
                    if hasattr(results[0], "keys"):
                        print(f"    DEBUG: Result keys: {list(results[0].keys())}")
                        if "rec_texts" in results[0]:
                            print(f"    DEBUG: rec_texts length: {len(results[0]['rec_texts'])}")
                            if len(results[0]["rec_texts"]) > 0:
                                print(f"    DEBUG: First text: {results[0]['rec_texts'][0]}")
                        if "dt_polys" in results[0]:
                            print(f"    DEBUG: dt_polys length: {len(results[0]['dt_polys'])}")

                # Create output filename
                base_name = pdf_file.stem
                page_suffix = f"_page{page_num}" if len(images) > 1 else ""

                # Draw bounding boxes and save image
                img_with_boxes = img_array.copy()
                text_output = []
                text_count = 0

                # Process results - the result dict has 'rec_texts', 'rec_scores', 'rec_polys'
                if results and len(results) > 0:
                    result = results[0]  # Get first (and usually only) result

                    # Extract data from result - use correct plural key names
                    rec_texts = result.get("rec_texts", [])
                    rec_scores = result.get("rec_scores", [])
                    rec_polys = result.get("rec_polys", [])

                    print(f"    Found {len(rec_texts)} text blocks")

                    # Process each detected text box
                    for idx in range(len(rec_texts)):
                        try:
                            text = rec_texts[idx]
                            box = rec_polys[idx]
                            confidence = rec_scores[idx]

                            # Convert box coordinates to integer
                            box = np.array(box, dtype=np.int32)

                            # Draw bounding box
                            cv2.polylines(img_with_boxes, [box], True, (0, 255, 0), 2)

                            # Add text above the box (truncate if too long)
                            display_text = text[:30] if len(text) > 30 else text
                            cv2.putText(
                                img_with_boxes,
                                display_text,
                                (box[0][0], max(box[0][1] - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                1,
                            )

                            # Store text with confidence
                            text_output.append(f"{text} (confidence: {confidence:.4f})")
                            text_count += 1

                        except (IndexError, TypeError, KeyError) as e:
                            print(f"    Warning: Error at index {idx}: {e}")
                            continue
                else:
                    print("    Warning: No OCR results found or unexpected result format")

                print(f"    Extracted {text_count} text blocks")

                # Always save files even if no text detected
                # Save annotated image
                output_img_path = os.path.join(
                    output_dir, f"{base_name}{page_suffix}_annotated.jpg"
                )
                success = cv2.imwrite(
                    output_img_path, img_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if success:
                    print(f"    ✓ Saved: {base_name}{page_suffix}_annotated.jpg")
                else:
                    print("    ✗ Failed to save annotated image")

                # Save original image
                output_orig_path = os.path.join(
                    output_dir, f"{base_name}{page_suffix}_original.jpg"
                )
                success = cv2.imwrite(output_orig_path, img_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    print(f"    ✓ Saved: {base_name}{page_suffix}_original.jpg")
                else:
                    print("    ✗ Failed to save original image")

                # Save extracted text
                output_txt_path = os.path.join(output_dir, f"{base_name}{page_suffix}_text.txt")
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(f"OCR Results for: {pdf_file.name} - Page {page_num}\n")
                    f.write("=" * 80 + "\n\n")
                    if text_output:
                        for text in text_output:
                            f.write(text + "\n")
                    else:
                        f.write("No text detected on this page.\n")
                print(f"    ✓ Saved: {base_name}{page_suffix}_text.txt")

                # Clean up memory after each page
                del img_array, img_with_boxes, results, text_output
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
    INPUT_DIR = "/workspaces/mtc-extraction-benchmark/data/raw/diler"
    OUTPUT_DIR = "/workspaces/mtc-extraction-benchmark/data/processed/paddle_ocr"

    print("=" * 60)
    print("PaddleOCR PDF Processor (Memory Optimized)")
    print("=" * 60)
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    # Process PDFs
    process_pdf_with_paddleocr(INPUT_DIR, OUTPUT_DIR)
