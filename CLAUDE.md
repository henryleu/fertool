# fertool
this is facial expression recognition cli is mainly developed in bunjs/nodejs/typescript with npm moduels of onnxruntime-web and jimp.
## Cli flags:
- -i the input folder path where picture files are located in supporting .jpg, jpeg, png, .bmp. by default, $cwd is used.
- -o the output filename which is the recognized results as a csv format file. by default. $cwd/results.csv is used.
- -m the model file path which is the facial expression recognition model file in onnx format. by default, $cwd/model.onnx is used. It also supports FER_MODEL environment variable but it will be overwritten by -m flag.

## Terms
- fer: facial expression recognition

## Features
- Supported picture file types: *.jpg, *.jpeg, *.png, *.bmp, *.tiff, *.webp
- fer a picture folder and output the results as a csv file
  - list and filter picture files in input folder order by filename asc (Alphabetically). such as: 001.jpg, 002.jpg, 003.png, etc.
  - fer a picture file and cache the result one by one
  - output all results to a csv file.
  - each picture file will be recognized as a csv line in the output file.
  - each line of the output file will be: picture filename, recognized result.
  - recognized result is like: 001.png,  fear/60.12%, happy/30.2%, sad/6.6%, angry/0.5%, disgust/1.2%, neutral/9.1%

## Model Shape Specifications

### Input Shape
- **Format**: `[batch, channels, height, width]`
- **Shape**: `[1, 3, 320, 320]`
- **Data Type**: `float32`
- **Normalization**: RGB values normalized to [0.0, 1.0] range
- **Channel Order**: RGB (Red, Green, Blue)
- **Preprocessing**: Images are resized to 320x320 and converted to CHW format

### Output Shape Specifications

The tool supports two model output formats:

#### 1. Simple Classification Format (Legacy)
- **Shape**: `[1, 6]` or `[6]`
- **Content**: Direct class probabilities for 6 emotions
- **Order**: [angry, disgust, fear, happy, neutral, sad]
- **Data Type**: `float32`
- **Range**: [0.0, 1.0] probability values

#### 2. YOLO11s Detection Format (Current)
- **Shape**: `[1, 10, N]` where N is number of detections (e.g., 2100)
- **Content**: Each detection contains [bbox + class_probs]
  - **Bounding Box**: 4 values [x_center, y_center, width, height] (normalized coordinates)
  - **Class Probabilities**: 6 values [angry, disgust, fear, happy, neutral, sad]
- **Alternative Shape**: `[1, N, 10]` (transposed format)
- **Data Type**: `float32`
- **Coordinate System**: Normalized coordinates [0.0, 1.0]

### Model Configuration (YOLO11s_CS.yaml)
- **Model Type**: `yolo11`
- **Model Name**: `yolo11s-r20240930`
- **Confidence Threshold**: `0.25` (minimum confidence for valid detection)
- **IoU Threshold**: `0.45` (for Non-Maximum Suppression)
- **Classes**: `[angry, disgust, fear, happy, neutral, sad]`

## Facial Expression Recognition Implementation

### 1. Inference Pipeline
```
Image Input → Preprocessing → ONNX Model → Output Parsing → NMS → Result Formatting
```

### 2. Output Parsing Logic

#### Format Detection
- If output length equals 6: Simple classification format
- If output length is divisible by 10: YOLO11s detection format
- Auto-detection based on output tensor dimensions

#### YOLO11s Parsing Process
1. **Detection Extraction**: For each detection in the output:
   - Extract bounding box coordinates: `[x_center, y_center, width, height]`
   - Extract class probabilities: `[prob_angry, prob_disgust, prob_fear, prob_happy, prob_neutral, prob_sad]`
   
2. **Confidence Filtering**: 
   - Find class with highest probability for each detection
   - Filter detections below confidence threshold (default: 0.25)
   
3. **Coordinate Conversion**:
   - Convert normalized coordinates to absolute pixel coordinates
   - Ensure coordinates are within image bounds [0, image_size]

4. **Non-Maximum Suppression (NMS)**:
   - Calculate IoU (Intersection over Union) between overlapping detections
   - Remove detections with IoU ≥ threshold (default: 0.45)
   - Keep detection with highest confidence per face

### 3. Result Structure
```typescript
interface FERResult {
  detected_faces: number;           // Number of faces detected
  primary_expression: string;       // Dominant emotion
  confidence: number;               // Confidence of primary expression
  all_expressions: Record<string, string>; // All emotion percentages
  raw_detections: Detection[];      // Raw detection data with bounding boxes
}
```

### 4. Error Handling
- **No Face Detected**: Returns neutral emotion (100%) with 0 confidence
- **Model Loading Errors**: Descriptive error messages with fallback behavior
- **Invalid Output Format**: Warns and attempts format adaptation
- **Processing Errors**: Individual image errors don't stop batch processing

### 5. Backward Compatibility
- Automatically detects and handles both simple and YOLO11s model formats
- Maintains existing CSV output format regardless of model type
- Configuration-driven thresholds with sensible defaults

## Build and Distribution
- The CLI tool will be a binary file compiled and built by bun command
- Supports cross-platform distribution for Windows, macOS with architectures amd64 and arm64
