#!/usr/bin/env python3
"""
YOLO11s FER Output Parser for Facial Expression Recognition
Parses YOLO11s model output based on YOLO11s_CS.yaml configuration
"""

import numpy as np
import yaml
import os
from typing import List, Tuple, Dict
from pathlib import Path

class YOLO11sFERParser:
    def __init__(self, config_path: str = None, confidence_threshold: float = None, nms_threshold: float = None, debug: bool = False):
        """
        Initialize YOLO11s FER Parser with configuration

        Args:
            config_path: Path to YOLO11s_CS.yaml config file
            confidence_threshold: Override confidence threshold (optional)
            nms_threshold: Override NMS threshold (optional) 
            debug: Enable debug output
        """
        self.debug = debug
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "YOLO11s_CS.yaml"
        
        self.config = self._load_config(config_path)
        
        # Extract configuration parameters with optional overrides
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else self.config.get('conf_threshold', 0.25)
        self.nms_threshold = nms_threshold if nms_threshold is not None else self.config.get('iou_threshold', 0.45)
        self.model_type = self.config.get('type', 'yolo11')
        self.model_name = self.config.get('name', 'yolo11s')
        
        # Load class names from config
        self.fer_classes = self.config.get('classes', [
            "angry", "disgust", "fear", "happy", "neutral", "sad"
        ])
        
        if self.debug:
            print(f"🔧 YOLO11s FER Parser initialized:")
            print(f"   Model: {self.model_name} ({self.model_type})")
            print(f"   Classes: {self.fer_classes}")
            print(f"   Confidence threshold: {self.confidence_threshold}")
            print(f"   IoU threshold: {self.nms_threshold}")

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️  Warning: Could not load config from {config_path}: {e}")
            # Return default configuration
            return {
                'type': 'yolo11',
                'name': 'yolo11s-default',
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'classes': ["angry", "disgust", "fear", "happy", "neutral", "sad"]
            }

    def parse_yolo_output(self, output: np.ndarray, input_width: int = 320, input_height: int = 320) -> Dict:
        """
        Parse YOLO11s output tensor to extract facial expressions

        Args:
            output: Model output tensor (expected shape varies by model)
            input_width: Input image width
            input_height: Input image height

        Returns:
            Dictionary with detected faces and their expressions
        """
        if self.debug:
            print(f"\n🔍 YOLO11s Parser Debug Info:")
            print(f"Input shape: {output.shape}")
            print(f"Input min/max: {output.min():.6f} / {output.max():.6f}")
            print(f"Input mean/std: {output.mean():.6f} / {output.std():.6f}")

        # Handle different output shapes
        if len(output.shape) == 3 and output.shape[0] == 1:
            # Remove batch dimension: [1, C, N] -> [C, N]
            output = output.squeeze(0)
        
        if self.debug:
            print(f"After preprocessing: {output.shape}")

        # Parse based on YOLO11s format
        detections = self._parse_yolo11s_format(output, input_width, input_height)
        
        # Apply NMS if multiple detections
        if len(detections) > 1:
            detections = self.apply_nms(detections)

        result = self.format_results(detections)

        if self.debug:
            print(f"\n📋 Final Result:")
            print(f"Detected faces: {result['detected_faces']}")
            print(f"Primary expression: {result['primary_expression']}")
            print(f"Confidence: {result['confidence']:.6f}")

        return result

    def _parse_yolo11s_format(self, output: np.ndarray, input_width: int, input_height: int) -> List[Dict]:
        """
        Parse YOLO11s specific output format
        
        YOLO11s typically outputs: [num_classes + 4, num_detections]
        Where the first 4 values are [x, y, w, h] and the rest are class probabilities
        """
        detections = []
        
        # Expected format: [num_classes + 4, num_detections]
        num_classes = len(self.fer_classes)
        expected_channels = num_classes + 4  # 4 for bbox coordinates
        
        if output.shape[0] != expected_channels:
            if self.debug:
                print(f"⚠️  Unexpected output format. Expected {expected_channels} channels, got {output.shape[0]}")
            # Try to adapt
            if output.shape[0] == 10 and num_classes == 6:
                # Assume [4 bbox + 6 classes] format
                output = output.T  # Transpose to [num_detections, 10]
                return self._parse_bbox_classes_format(output, input_width, input_height)
        
        # Standard YOLO11s format: [num_classes + 4, num_detections]
        num_detections = output.shape[1]
        
        for i in range(num_detections):
            detection = output[:, i]  # Shape: [num_classes + 4]
            
            # Extract bbox coordinates (first 4 values)
            x_center, y_center, width, height = detection[:4]
            
            # Extract class probabilities
            class_probs = detection[4:4+num_classes]
            
            # Find the class with highest probability
            if len(class_probs) > 0:
                class_id = np.argmax(class_probs)
                class_confidence = class_probs[class_id]
                
                if class_confidence > self.confidence_threshold:
                    # Convert normalized coordinates to absolute coordinates
                    x1 = (x_center - width/2) * input_width
                    y1 = (y_center - height/2) * input_height
                    x2 = (x_center + width/2) * input_width
                    y2 = (y_center + height/2) * input_height
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(x1, input_width))
                    y1 = max(0, min(y1, input_height))
                    x2 = max(0, min(x2, input_width))
                    y2 = max(0, min(y2, input_height))
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(class_confidence),
                        'class_id': int(class_id),
                        'class_name': self.fer_classes[class_id] if class_id < len(self.fer_classes) else f'class_{class_id}',
                        'class_probs': class_probs.tolist(),
                        'method': 'yolo11s_standard'
                    })
        
        return detections

    def _parse_bbox_classes_format(self, output: np.ndarray, input_width: int, input_height: int) -> List[Dict]:
        """
        Parse alternative format: [num_detections, bbox + classes]
        """
        detections = []
        
        for i in range(output.shape[0]):
            detection = output[i]  # Shape: [10] for 4 bbox + 6 classes
            
            if len(detection) >= 10:
                # Extract bbox and class probabilities
                x_center, y_center, width, height = detection[:4]
                class_probs = detection[4:4+len(self.fer_classes)]
                
                if len(class_probs) > 0:
                    class_id = np.argmax(class_probs)
                    class_confidence = class_probs[class_id]
                    
                    if class_confidence > self.confidence_threshold:
                        # Convert to absolute coordinates
                        x1 = (x_center - width/2) * input_width
                        y1 = (y_center - height/2) * input_height
                        x2 = (x_center + width/2) * input_width
                        y2 = (y_center + height/2) * input_height
                        
                        # Ensure coordinates are within bounds
                        x1 = max(0, min(x1, input_width))
                        y1 = max(0, min(y1, input_height))
                        x2 = max(0, min(x2, input_width))
                        y2 = max(0, min(y2, input_height))
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(class_confidence),
                            'class_id': int(class_id),
                            'class_name': self.fer_classes[class_id] if class_id < len(self.fer_classes) else f'class_{class_id}',
                            'class_probs': class_probs.tolist(),
                            'method': 'bbox_classes_alt'
                        })
        
        return detections

    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression using configured IoU threshold"""
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while detections:
            # Keep the detection with highest confidence
            best = detections.pop(0)
            keep.append(best)

            # Remove detections that overlap significantly with the best one
            detections = [det for det in detections
                         if self.calculate_iou(best['bbox'], det['bbox']) < self.nms_threshold]

        return keep

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def format_results(self, detections: List[Dict]) -> Dict:
        """Format detection results for FER output"""
        if not detections:
            # Return neutral as default when no face detected
            default_expressions = {class_name: "0.00%" for class_name in self.fer_classes}
            default_expressions['neutral'] = "100.00%"
            
            return {
                'detected_faces': 0,
                'primary_expression': 'neutral',
                'confidence': 0.0,
                'all_expressions': default_expressions,
                'raw_detections': []
            }

        # Use the detection with highest confidence as primary
        primary_detection = max(detections, key=lambda x: x['confidence'])

        # Calculate expression probabilities
        all_expressions = {}
        for i, class_name in enumerate(self.fer_classes):
            if i < len(primary_detection['class_probs']):
                prob = primary_detection['class_probs'][i]
                # Convert to percentage
                all_expressions[class_name] = f"{prob * 100:.2f}%"
            else:
                all_expressions[class_name] = "0.00%"

        return {
            'detected_faces': len(detections),
            'primary_expression': primary_detection['class_name'],
            'confidence': primary_detection['confidence'],
            'all_expressions': all_expressions,
            'raw_detections': detections
        }

# Backward compatibility alias
YOLOFERParser = YOLO11sFERParser

def test_parser():
    """Test the YOLO11s FER Parser with dummy data"""
    parser = YOLO11sFERParser(debug=True)

    # Create dummy output data matching expected YOLO11s format
    # [num_classes + 4, num_detections] = [10, 2100]
    dummy_output = np.random.rand(1, 10, 2100) * 0.1  # Low random values

    # Make one detection more prominent (happy face)
    # Format: [x_center, y_center, width, height, angry, disgust, fear, happy, neutral, sad]
    dummy_output[0, :, 0] = [0.5, 0.5, 0.3, 0.3, 0.1, 0.05, 0.05, 0.8, 0.1, 0.05]  # Happy face
    dummy_output[0, :, 1] = [0.3, 0.3, 0.2, 0.2, 0.7, 0.1, 0.1, 0.05, 0.05, 0.05]  # Angry face

    result = parser.parse_yolo_output(dummy_output)

    print("\n🧪 YOLO11s FER Parser Test Results:")
    print("=" * 50)
    print(f"Detected faces: {result['detected_faces']}")
    print(f"Primary expression: {result['primary_expression']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nAll expressions:")
    for expr, prob in result['all_expressions'].items():
        print(f"  {expr}: {prob}")
    
    if result['raw_detections']:
        print("\nRaw detections:")
        for i, det in enumerate(result['raw_detections']):
            print(f"  Detection {i+1}: {det['class_name']} ({det['confidence']:.3f}) - {det['method']}")

if __name__ == "__main__":
    test_parser()
