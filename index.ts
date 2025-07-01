#!/usr/bin/env bun

import { Command } from 'commander';
import { resolve } from 'path';
import { existsSync, readdirSync, writeFileSync } from 'fs';
import { Jimp } from 'jimp';
import * as ort from 'onnxruntime-web';

const program = new Command();

function getDefaultModelPath(): string {
  return process.env.FER_MODEL || resolve(process.cwd(), 'model.onnx');
}

program
  .name('fertool')
  .description('Facial Expression Recognition CLI Tool')
  .version('1.0.0')
  .option('-i, --input <folder>', 'input folder path where picture files are located (.jpg, .jpeg, .png, .bmp)', process.cwd())
  .option('-o, --output <file>', 'output filename for recognized results as CSV format', resolve(process.cwd(), 'results.csv'))
  .option('-m, --model <file>', 'model file path for facial expression recognition model in ONNX format')
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
      await processImages(options.input, options.output, modelPath);
    } catch (error) {
      console.error('Error processing images:', error);
      process.exit(1);
    }
  });

const SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'];
const EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'];

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

async function recognizeEmotion(session: ort.InferenceSession, imageData: Float32Array): Promise<Record<string, number>> {
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
    
    const probabilities = Array.from(output.data as Float32Array);
    
    const emotions: Record<string, number> = {};
    for (let i = 0; i < Math.min(EMOTION_LABELS.length, probabilities.length); i++) {
      const label = EMOTION_LABELS[i];
      const probability = probabilities[i];
      if (label && probability !== undefined) {
        emotions[label] = probability * 100;
      }
    }
    
    return emotions;
  } catch (error) {
    throw new Error(`Failed to recognize emotion: ${error}`);
  }
}

function formatEmotionsAsCSV(emotions: Record<string, number>): string {
  return Object.entries(emotions)
    .sort(([, a], [, b]) => b - a)
    .map(([emotion, confidence]) => `${emotion}/${confidence.toFixed(2)}%`)
    .join(', ');
}

async function processImages(inputFolder: string, outputFile: string, modelPath: string) {
  console.log(`Processing images from: ${inputFolder}`);
  console.log(`Results will be saved to: ${outputFile}`);
  console.log(`Using model: ${modelPath}`);
  
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
      const emotions = await recognizeEmotion(session, imageData);
      const emotionString = formatEmotionsAsCSV(emotions);
      
      csvResults.push(`${filename}, ${emotionString}`);
    } catch (error) {
      console.error(`Error processing ${filename}: ${error}`);
      csvResults.push(`${filename}, Error: ${error}`);
    }
  }
  
  writeFileSync(outputFile, csvResults.join('\n'));
  console.log(`Results saved to: ${outputFile}`);
}

program.parse();