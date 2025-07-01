#!/usr/bin/env bun

import { Command } from 'commander';
import { resolve } from 'path';
import { existsSync, readdirSync, writeFileSync, readFileSync } from 'fs';
import { Jimp } from 'jimp';
import * as ort from 'onnxruntime-web';
import * as yaml from 'js-yaml';

const program = new Command();

function getDefaultModelPath(): string {
  return process.env.FER_MODEL || resolve(process.cwd(), 'model.onnx');
}

function loadModelConfig(configPath?: string): ModelConfig {
  const defaultConfig: ModelConfig = {
    type: 'yolo11',
    name: 'yolo11s-default',
    conf_threshold: 0.01, // Match Python integration example default
    iou_threshold: 0.45,
    classes: DEFAULT_EMOTION_LABELS
  };

  if (!configPath) {
    configPath = resolve(process.cwd(), 'model', 'YOLO11s_CS.yaml');
  }

  if (!existsSync(configPath)) {
    console.warn(`Warning: Config file not found at ${configPath}, using defaults`);
    return defaultConfig;
  }

  try {
    const fileContent = readFileSync(configPath, 'utf8');
    const config = yaml.load(fileContent) as any;

    return {
      type: config.type || defaultConfig.type,
      name: config.name || defaultConfig.name,
      conf_threshold: config.conf_threshold || defaultConfig.conf_threshold,
      iou_threshold: config.iou_threshold || defaultConfig.iou_threshold,
      classes: config.classes || defaultConfig.classes
    };
  } catch (error) {
    console.warn(`Warning: Failed to load config from ${configPath}: ${error}. Using defaults.`);
    return defaultConfig;
  }
}

program
  .name('fertool')
  .description('Facial Expression Recognition CLI Tool')
  .version('1.0.0')
  .option('-i, --input <folder>', 'input folder path where picture files are located (.jpg, .jpeg, .png, .bmp)', process.cwd())
  .option('-o, --output <file>', 'output filename for recognized results as CSV format', resolve(process.cwd(), 'results.csv'))
  .option('-m, --model <file>', 'model file path for facial expression recognition model in ONNX format')
  .option('-d, --debug', 'enable debug mode to show detailed parsing information', false)
  .action(async (options) => {
    console.log('Facial Expression Recognition Tool');
    console.log('Input folder:', options.input);
    console.log('Output file:', options.output);

    const modelPath = options.model || getDefaultModelPath();
    console.log('Model file:', modelPath);

    if (!existsSync(options.input)) {
      console.error(`Error: Input folder "${options.input}" does not exist.`);
      process.exit(1);
    }

    if (!existsSync(modelPath)) {
      console.error(`Error: Model file "${modelPath}" does not exist.`);
      process.exit(1);
    }

    try {
      await processImages(options.input, options.output, modelPath, options.debug);
    } catch (error) {
      console.error('Error processing images:', error);
      process.exit(1);
    }
  });

const SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'];
const DEFAULT_EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'];

interface ModelConfig {
  type: string;
  name: string;
  conf_threshold: number;
  iou_threshold: number;
  classes: string[];
}

interface Detection {
  bbox: [number, number, number, number];
  confidence: number;
  class_id: number;
  class_name: string;
  class_probs: number[];
}

interface FERResult {
  detected_faces: number;
  primary_expression: string;
  confidence: number;
  all_expressions: Record<string, string>;
  raw_detections: Detection[];
}

function calculateIoU(box1: [number, number, number, number], box2: [number, number, number, number]): number {
  const [x1_1, y1_1, x2_1, y2_1] = box1;
  const [x1_2, y1_2, x2_2, y2_2] = box2;

  const x1_i = Math.max(x1_1, x1_2);
  const y1_i = Math.max(y1_1, y1_2);
  const x2_i = Math.min(x2_1, x2_2);
  const y2_i = Math.min(y2_1, y2_2);

  if (x2_i <= x1_i || y2_i <= y1_i) {
    return 0.0;
  }

  const intersection = (x2_i - x1_i) * (y2_i - y1_i);
  const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
  const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
  const union = area1 + area2 - intersection;

  return union > 0 ? intersection / union : 0.0;
}

function applyNMS(detections: Detection[], iouThreshold: number): Detection[] {
  if (detections.length <= 1) {
    return detections;
  }

  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const keep: Detection[] = [];

  while (sorted.length > 0) {
    const best = sorted.shift()!;
    keep.push(best);

    for (let i = sorted.length - 1; i >= 0; i--) {
      const detection = sorted[i];
      if (detection && calculateIoU(best.bbox, detection.bbox) >= iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return keep;
}

function parseYOLO11sOutput(output: Float32Array, config: ModelConfig, inputWidth: number = 320, inputHeight: number = 320, debug: boolean = false): Detection[] {
  const detections: Detection[] = [];
  const numClasses = config.classes.length;
  
  if (debug) {
    console.log(`\n🔍 YOLO11s Parser Debug Info:`);
    console.log(`Original output length: ${output.length}`);
    console.log(`Min/Max values: ${Math.min(...output).toFixed(6)} / ${Math.max(...output).toFixed(6)}`);
  }

  // Handle tensor shape - need to determine the actual tensor dimensions
  // Common formats: [1, 10, 2100] or [10, 2100] or flat [21000]
  let tensorOutput: number[][];
  let actualShape: number[] = [];
  
  // Try to infer tensor shape from length
  const expectedChannels = numClasses + 4; // 10 for 6 classes + 4 bbox
  
  if (output.length === numClasses) {
    // Simple classification format [6]
    const outputArray = Array.from(output);
    return [{
      bbox: [0, 0, inputWidth, inputHeight],
      confidence: Math.max(...outputArray),
      class_id: outputArray.indexOf(Math.max(...outputArray)),
      class_name: config.classes[outputArray.indexOf(Math.max(...outputArray))] || 'unknown',
      class_probs: outputArray
    }];
  }
  
  // Check if it's a common YOLO output size
  const commonSizes = [2100, 8400, 16800]; // Common YOLO detection counts
  let numDetections = 0;
  
  for (const size of commonSizes) {
    if (output.length === expectedChannels * size) {
      numDetections = size;
      actualShape = [expectedChannels, numDetections];
      break;
    } else if (output.length === size * expectedChannels) {
      numDetections = size;
      actualShape = [numDetections, expectedChannels];
      break;
    }
  }
  
  if (numDetections === 0) {
    // Fallback: try to infer from length
    if (output.length % expectedChannels === 0) {
      numDetections = output.length / expectedChannels;
      actualShape = [expectedChannels, numDetections];
    } else {
      console.warn(`Cannot infer tensor shape from length ${output.length}`);
      return detections;
    }
  }
  
  if (debug) {
    console.log(`Inferred shape: [${actualShape.join(', ')}]`);
    console.log(`Num detections: ${numDetections}`);
  }
  
  // Convert flat array to 2D tensor based on inferred shape
  const outputArray = Array.from(output);
  
  if (actualShape[0] === expectedChannels) {
    // Format: [10, N] - channels first (standard YOLO11s)
    tensorOutput = Array.from({ length: expectedChannels }, () => 
      Array.from({ length: numDetections }, () => 0)
    );
    for (let c = 0; c < expectedChannels; c++) {
      for (let n = 0; n < numDetections; n++) {
        const index = c * numDetections + n;
        tensorOutput[c]![n] = index < outputArray.length ? (outputArray[index] ?? 0) : 0;
      }
    }
  } else {
    // Format: [N, 10] - detections first (transposed)
    tensorOutput = Array.from({ length: expectedChannels }, () => 
      Array.from({ length: numDetections }, () => 0)
    );
    for (let c = 0; c < expectedChannels; c++) {
      for (let n = 0; n < numDetections; n++) {
        const index = n * expectedChannels + c;
        tensorOutput[c]![n] = index < outputArray.length ? (outputArray[index] ?? 0) : 0;
      }
    }
  }
  
  if (debug) {
    console.log(`Tensor converted to [${expectedChannels}, ${numDetections}] format`);
  }

  // Parse detections in YOLO11s format: [channels, detections]
  for (let i = 0; i < numDetections; i++) {
    // Extract detection data for detection i
    const detection: number[] = [];
    for (let c = 0; c < expectedChannels; c++) {
      detection[c] = tensorOutput[c]?.[i] ?? 0;
    }
    
    // Extract bbox coordinates (first 4 values)
    const xCenter = detection[0];
    const yCenter = detection[1];
    const width = detection[2];
    const height = detection[3];
    
    if (xCenter == null || yCenter == null || width == null || height == null) {
      continue;
    }
    
    // Extract class probabilities (last 6 values)
    const classProbs = detection.slice(4, 4 + numClasses);
    
    if (classProbs.length > 0) {
      const classId = classProbs.indexOf(Math.max(...classProbs));
      const classConfidence = classProbs[classId];
      
      if (classConfidence != null && classConfidence > config.conf_threshold) {
        // Convert normalized coordinates to absolute coordinates
        const x1 = Math.max(0, Math.min((xCenter - width/2) * inputWidth, inputWidth));
        const y1 = Math.max(0, Math.min((yCenter - height/2) * inputHeight, inputHeight));
        const x2 = Math.max(0, Math.min((xCenter + width/2) * inputWidth, inputWidth));
        const y2 = Math.max(0, Math.min((yCenter + height/2) * inputHeight, inputHeight));
        
        detections.push({
          bbox: [x1, y1, x2, y2],
          confidence: classConfidence,
          class_id: classId,
          class_name: config.classes[classId] || `class_${classId}`,
          class_probs: classProbs
        });
        
        if (debug) {
          console.log(`Detection ${detections.length}: ${config.classes[classId]} (${(classConfidence * 100).toFixed(2)}%)`);
        }
      }
    }
  }
  
  if (debug) {
    console.log(`Found ${detections.length} valid detections`);
  }

  return detections;
}

function formatFERResults(detections: Detection[], config: ModelConfig): FERResult {
  if (detections.length === 0) {
    const defaultExpressions: Record<string, string> = {};
    config.classes.forEach(className => {
      defaultExpressions[className] = className === 'neutral' ? '100.00%' : '0.00%';
    });

    return {
      detected_faces: 0,
      primary_expression: 'neutral',
      confidence: 0.0,
      all_expressions: defaultExpressions,
      raw_detections: []
    };
  }

  const primaryDetection = detections.reduce((prev, current) =>
    prev.confidence > current.confidence ? prev : current
  );

  const allExpressions: Record<string, string> = {};
  config.classes.forEach((className, i) => {
    if (i < primaryDetection.class_probs.length) {
      const prob = primaryDetection.class_probs[i];
      if (prob != null) {
        allExpressions[className] = `${(prob * 100).toFixed(2)}%`;
      } else {
        allExpressions[className] = '0.00%';
      }
    } else {
      allExpressions[className] = '0.00%';
    }
  });

  return {
    detected_faces: detections.length,
    primary_expression: primaryDetection.class_name,
    confidence: primaryDetection.confidence,
    all_expressions: allExpressions,
    raw_detections: detections
  };
}

function getImageFiles(folderPath: string): string[] {
  try {
    const files = readdirSync(folderPath);
    return files
      .filter(file => {
        const ext = file.toLowerCase().substring(file.lastIndexOf('.'));
        return SUPPORTED_EXTENSIONS.includes(ext);
      })
      .sort()
      .map(file => resolve(folderPath, file));
  } catch (error) {
    throw new Error(`Failed to read directory: ${error}`);
  }
}

async function loadOnnxModel(modelPath: string): Promise<ort.InferenceSession> {
  try {
    const session = await ort.InferenceSession.create(modelPath);
    return session;
  } catch (error) {
    throw new Error(`Failed to load ONNX model: ${error}`);
  }
}

async function preprocessImage(imagePath: string): Promise<Float32Array> {
  try {
    const image = await Jimp.read(imagePath);
    const resized = image.resize({ w: 320, h: 320 });

    const { width, height } = resized.bitmap;
    const inputData = new Float32Array(3 * width * height);

    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const pixel = resized.getPixelColor(x, y);
          const r = (pixel >> 24) & 255;
          const g = (pixel >> 16) & 255;
          const b = (pixel >> 8) & 255;

          let value = 0;
          if (c === 0) value = r;
          else if (c === 1) value = g;
          else value = b;

          inputData[idx++] = value / 255.0;
        }
      }
    }

    return inputData;
  } catch (error) {
    throw new Error(`Failed to preprocess image ${imagePath}: ${error}`);
  }
}

async function recognizeEmotion(session: ort.InferenceSession, imageData: Float32Array, config: ModelConfig, debug: boolean = false): Promise<FERResult> {
  try {
    const inputTensor = new ort.Tensor('float32', imageData, [1, 3, 320, 320]);

    if (!session.inputNames[0]) {
      throw new Error('Model has no input names defined');
    }

    const feeds: Record<string, ort.Tensor> = {};
    feeds[session.inputNames[0]] = inputTensor;

    const results = await session.run(feeds);

    const outputName = session.outputNames[0];
    if (!outputName) {
      throw new Error('Model has no output names defined');
    }

    const output = results[outputName];
    if (!output) {
      throw new Error('Model produced no output');
    }

    const outputData = output.data as Float32Array;

    let detections = parseYOLO11sOutput(outputData, config, 320, 320, debug);

    if (detections.length > 1) {
      detections = applyNMS(detections, config.iou_threshold);
    }

    return formatFERResults(detections, config);
  } catch (error) {
    throw new Error(`Failed to recognize emotion: ${error}`);
  }
}

function formatEmotionsAsCSV(result: FERResult): string {
  if (result.detected_faces === 0) {
    return 'no_face_detected';
  }

  return Object.entries(result.all_expressions)
    .sort(([, a], [, b]) => parseFloat(b.replace('%', '')) - parseFloat(a.replace('%', '')))
    .map(([emotion, percentage]) => `${emotion}/${percentage}`)
    .join(', ');
}

async function processImages(inputFolder: string, outputFile: string, modelPath: string, debug: boolean = false) {
  console.log(`Processing images from: ${inputFolder}`);
  console.log(`Results will be saved to: ${outputFile}`);
  console.log(`Using model: ${modelPath}`);

  const config = loadModelConfig();
  console.log(`Model config loaded: ${config.name} (${config.type})`);
  console.log(`Emotion classes: ${config.classes.join(', ')}`);
  console.log(`Confidence threshold: ${config.conf_threshold}, IoU threshold: ${config.iou_threshold}`);
  if (debug) {
    console.log(`🐛 Debug mode enabled`);
  }

  const imageFiles = getImageFiles(inputFolder);
  if (imageFiles.length === 0) {
    console.log('No supported image files found in the input folder.');
    return;
  }

  console.log(`Found ${imageFiles.length} image files to process.`);

  const session = await loadOnnxModel(modelPath);
  console.log('ONNX model loaded successfully.');

  const csvResults: string[] = [];

  for (let i = 0; i < imageFiles.length; i++) {
    const imagePath = imageFiles[i];
    if (!imagePath) {
      console.error(`Invalid image path at index ${i}`);
      continue;
    }

    const filename = imagePath.substring(imagePath.lastIndexOf('/') + 1);

    try {
      console.log(`Processing ${i + 1}/${imageFiles.length}: ${filename}`);

      const imageData = await preprocessImage(imagePath);
      const result = await recognizeEmotion(session, imageData, config, debug);
      const emotionString = formatEmotionsAsCSV(result);

      csvResults.push(`${filename}, ${emotionString}`);

      if (result.detected_faces > 0) {
        console.log(`  → ${result.primary_expression} (${(result.confidence * 100).toFixed(2)}%)`);
      } else {
        console.log(`  → No face detected`);
      }
    } catch (error) {
      console.error(`Error processing ${filename}: ${error}`);
      csvResults.push(`${filename}, Error: ${error}`);
    }
  }

  writeFileSync(outputFile, csvResults.join('\n'));
  console.log(`Results saved to: ${outputFile}`);
}

// Export functions for testing
export { loadModelConfig, parseYOLO11sOutput, formatFERResults, applyNMS };

// Only run CLI if this is the main module
if (import.meta.main) {
  program.parse();
}
