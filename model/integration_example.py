#!/usr/bin/env python3
"""
Integration Example: How to use YOLO FER Parser with ONNX Runtime
Shows how to integrate the parser with your TypeScript/Node.js CLI
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import sys
import argparse
from yolo_parser import YOLOFERParser

def preprocess_image(image_path: str, target_size: tuple = (320, 320)) -> np.ndarray:
    """
    Preprocess image for YOLO model input
    
    Args:
        image_path: Path to input image
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image array of shape [1, 3, 320, 320]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Change from HWC to CHW format
    image_array = image_array.transpose(2, 0, 1)
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def run_fer_inference(model_path: str, image_path: str, confidence_threshold: float = 0.01, debug: bool = False) -> dict:
    """
    Run facial expression recognition inference
    
    Args:
        model_path: Path to ONNX model
        image_path: Path to input image
    
    Returns:
        FER results dictionary
    """
    # Load ONNX model
    session = ort.InferenceSession(model_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    
    # Parse YOLO output
    parser = YOLOFERParser(confidence_threshold=confidence_threshold, debug=debug)
    result = parser.parse_yolo_output(outputs[0])
    
    return result

def format_csv_output(filename: str, result: dict) -> str:
    """
    Format result for CSV output as specified in CLAUDE.md
    
    Args:
        filename: Image filename
        result: FER result dictionary
    
    Returns:
        CSV line string
    """
    if result['detected_faces'] == 0:
        return f"{filename}, no_face_detected"
    
    # Sort expressions by probability (highest first)
    expressions = []
    for expr, prob_str in result['all_expressions'].items():
        prob_value = float(prob_str.replace('%', ''))
        expressions.append((expr, prob_value, prob_str))
    
    expressions.sort(key=lambda x: x[1], reverse=True)
    
    # Format as: filename, expr1/prob1%, expr2/prob2%, ...
    expr_strings = [f"{expr}/{prob_str}" for expr, _, prob_str in expressions[:6]]  # Top 6
    
    return f"{filename}, {', '.join(expr_strings)}"

def process_folder(model_path: str, input_folder: str, output_file: str, confidence_threshold: float = 0.01, debug: bool = False):
    """
    Process a folder of images for facial expression recognition
    
    Args:
        model_path: Path to ONNX model file
        input_folder: Path to folder containing images
        output_file: Path to output CSV file
        confidence_threshold: Confidence threshold for detections
    """
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        return
    
    # Get all image files
    image_files = []
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            image_files.append(file)
    
    # Sort alphabetically
    image_files.sort()
    
    results = []
    
    print(f"Processing {len(image_files)} images...")
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(input_folder, filename)
        
        try:
            # Run FER inference
            result = run_fer_inference(model_path, image_path, confidence_threshold, debug)
            
            # Format for CSV
            csv_line = format_csv_output(filename, result)
            results.append(csv_line)
            
            print(f"[{i+1}/{len(image_files)}] {filename}: {result['primary_expression']} ({result['confidence']:.2f})")
            
        except Exception as e:
            error_line = f"{filename}, error: {str(e)}"
            results.append(error_line)
            print(f"[{i+1}/{len(image_files)}] {filename}: ERROR - {str(e)}")
    
    # Write results to CSV
    with open(output_file, 'w') as f:
        f.write("filename, expressions\n")  # Header
        for line in results:
            f.write(line + "\n")
    
    print(f"\nResults saved to: {output_file}")

def main():
    """Main function to handle command line arguments and run FER processing"""
    parser = argparse.ArgumentParser(
        description="YOLO Facial Expression Recognition - Process images and output CSV results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python integration_example.py -i ./images -o results.csv
  python integration_example.py -i /path/to/pics -o /path/to/output.csv -m model.onnx -c 0.1
  python integration_example.py --input-folder ./test_images --output-file fer_results.csv --confidence 0.05 --debug
  python integration_example.py -i ./images -o results.csv -d  # Enable debug mode"""
    )
    
    parser.add_argument(
        "-i", "--input-folder",
        type=str,
        required=True,
        help="Path to folder containing image files (.jpg, .jpeg, .png, .bmp, .tiff, .webp)"
    )
    
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        default="yolo.onnx",
        help="Path to ONNX model file (default: yolo.onnx)"
    )
    
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.01,
        help="Confidence threshold for face detection (default: 0.01)"
    )
    
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode to show detailed parsing information"
    )
    
    args = parser.parse_args()
    
    print("🚀 YOLO Facial Expression Recognition")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Input folder: {args.input_folder}")
    print(f"Output file: {args.output_file}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Debug mode: {args.debug}")
    print("=" * 50)
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.input_folder):
        print(f"❌ Error: Input folder not found: {args.input_folder}")
        sys.exit(1)
    
    if not os.path.isdir(args.input_folder):
        print(f"❌ Error: Input path is not a directory: {args.input_folder}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")
    
    try:
        # Process the folder
        process_folder(args.model_path, args.input_folder, args.output_file, args.confidence, args.debug)
        print("\n✅ Processing completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()